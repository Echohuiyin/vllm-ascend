# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import os
import threading
from bisect import bisect_right
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from itertools import accumulate
from typing import Any, NoReturn

import torch
from vllm.logger import logger
from vllm.v1.worker.worker_base import WorkerFatalError

from vllm_ascend.device_allocator.famem_client import FamemBusyError, FamemClient
from vllm_ascend.device_allocator.famem_native import (
    FamemNativeAllocation,
    FamemNativeLibrary,
    FamemNativeStats,
    FamemPageType,
    FamemWorkerState,
)
from vllm_ascend.device_allocator.kv_cache_plan import validate_npu_allocator_config
from vllm_ascend.worker.copier import CopierProcess
from vllm_ascend.worker.famem_copier import (
    FamemCopierMemoryBackend,
    FamemMappingDesc,
    FamemSharedWeightDesc,
    FamemWeightSpan,
)

_SUPPORTED_TAGS = ("weights", "kv_cache")


class FaMemAllocator:
    """Whole-arena bump allocator backed by an external HBM server."""

    def __init__(
        self,
        device: int,
        capacity: int,
        socket_dir: str,
        *,
        native: FamemNativeLibrary | Any | None = None,
        client_factory: Callable[..., FamemClient] = FamemClient,
    ) -> None:
        if device < 0:
            raise ValueError("Famem device index must be non-negative.")
        if capacity <= 0:
            raise ValueError("Famem arena capacity must be positive.")
        allocator_config = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        validate_npu_allocator_config(allocator_config)
        self.device = device
        self.capacity = capacity
        self.socket_dir = socket_dir
        self.native = native or FamemNativeLibrary()
        self.client_factory = client_factory
        self.client: FamemClient | None = None
        self.base_address = 0
        self.weight_end = 0
        self._entered_tags: list[str] = []
        self._active_pool_tag: str | None = None
        self._pools: dict[str, Any] = {}
        self._pluggable_allocator: Any | None = None
        self._backup_bytes = 0
        self._copier: CopierProcess | None = None
        self._copier_tgid = 0
        self._pipeline_initialized = False
        self._descriptors_sent = False
        self._num_layers = 0
        self._ready = True
        self._cycle_state = "active"
        self._sleeping = False
        self._poisoned = False
        self._closed = False
        self._cleanup_complete = False
        self._lock = threading.RLock()

        granularity = self.native.allocation_granularity(device)
        if capacity % granularity:
            raise ValueError(f"Famem arena capacity {capacity} is not aligned to CANN granularity {granularity}.")
        native_state = self.native.worker_stats().state
        if native_state != FamemWorkerState.UNINITIALIZED:
            raise RuntimeError(
                f"Famem supports one allocator lifecycle per worker process; native state is {native_state.name}."
            )
        self.granularity = granularity

    @property
    def is_sleeping(self) -> bool:
        return self._sleeping

    @property
    def ready(self) -> bool:
        return self._ready

    @ready.setter
    def ready(self, value: bool) -> None:
        self._ready = value
        if value and self._cycle_state == "resuming":
            self._cycle_state = "active"

    @property
    def multiproc_pipe_enabled(self) -> bool:
        return True

    @property
    def available_bytes(self) -> int:
        stats = self.stats()
        if stats.heap_top > stats.capacity:
            raise RuntimeError(
                "Famem native allocator invariant violated: "
                f"heap_top={stats.heap_top} exceeds capacity={stats.capacity}. "
                "Verify that the worker loaded the current native library."
            )
        return stats.capacity - stats.heap_top

    def get_current_usage(self) -> int:
        if self.client is None:
            return 0
        return self.stats().heap_top

    def stats(self) -> FamemNativeStats:
        if self.client is None:
            return FamemNativeStats(0, 0, 0, 0, 0, FamemWorkerState.UNINITIALIZED)
        return self.native.worker_stats()

    def diagnostics(self) -> dict[str, int | str | bool]:
        stats = self.stats()
        page_types = self.client.extent_page_types if self.client else []
        extent_sizes = self.client.extent_sizes if self.client else []
        page_bytes = {page_type: 0 for page_type in FamemPageType}
        for page_type, size in zip(page_types, extent_sizes, strict=True):
            page_bytes[page_type] += size
        return {
            "enabled": True,
            "device": self.device,
            "capacity": stats.capacity,
            "heap_top": stats.heap_top,
            "live_bytes": stats.live_bytes,
            "freed_bytes": stats.freed_bytes,
            "allocation_count": stats.allocation_count,
            "base_address": stats.base_address,
            "state": stats.state.name,
            "weight_end": self.weight_end,
            "backup_bytes": self._backup_bytes,
            "extent_count": len(extent_sizes),
            "huge_1g_bytes": page_bytes[FamemPageType.HUGE_1G],
            "huge_2m_bytes": page_bytes[FamemPageType.HUGE_2M],
            "poisoned": self._poisoned or bool(self.client and self.client.poisoned),
            "pipeline_initialized": self._pipeline_initialized,
            "pipeline_ready": self._ready,
        }

    def start_pipeline(
        self,
        device: int,
        num_layers: int,
        layer_ready_events: dict[int, Any] | None = None,
        *,
        tp_size: int = 1,
        local_rank: int = 0,
    ) -> None:
        """Start the Copier before the Worker acquires and imports the arena."""
        if device != self.device:
            raise RuntimeError("Famem Copier device does not match the allocator device")
        if self.client is not None or self._pools:
            raise RuntimeError("Famem multiproc_pipe must start before acquiring the arena")
        if self._copier is not None:
            if self._num_layers != num_layers:
                raise RuntimeError("Cannot reconfigure an existing Famem Copier")
            return
        from vllm_ascend.device_allocator.camem import CaMemAllocator

        queues, layer_ready_events = CaMemAllocator._pipeline_resources(layer_ready_events)
        copier_kwargs = {"queues": queues} if queues is not None else {}
        copier = CopierProcess(
            device,
            num_layers,
            layer_ready_events,
            backend_factory=FamemCopierMemoryBackend,
            label="Famem Copier",
            tp_size=tp_size,
            local_rank=local_rank,
            **copier_kwargs,
        )
        try:
            copier_tgid = copier.start()
        except BaseException:
            try:
                copier.close()
            except Exception as cleanup_error:
                self._copier = copier
                self._num_layers = num_layers
                self._poisoned = True
                self._closed = True
                raise RuntimeError(
                    "Famem Copier startup failed and the process could not be stopped"
                ) from cleanup_error
            raise
        self._copier = copier
        self._copier_tgid = copier_tgid
        self._num_layers = num_layers
        CaMemAllocator.set_desc_queue(copier.desc_queue)
        CaMemAllocator.set_npu_recover_queue(copier.npu_recover_queue)
        CaMemAllocator.set_ctrl_queue(copier.ctrl_queue)
        CaMemAllocator.set_layer_ready_events(copier.layer_ready_events)

    def send_descs_for_scattered_weights(self) -> None:
        """Send post-processed Famem weight spans to the Copier."""
        if self._copier is None or self.client is None:
            raise RuntimeError("Famem multiproc_pipe was not started before model loading")
        if self._descriptors_sent or self._pipeline_initialized:
            raise RuntimeError("Famem multiproc_pipe was already initialized")
        try:
            torch.npu.synchronize(self.device)
            weights = self._build_weight_descriptors()
            self._copier.send_initial((self._mapping_descriptor(), weights))
        except BaseException:
            self._close_after_terminal_failure("Copier initialization")
            raise
        self._descriptors_sent = True
        logger.info(
            "multiproc_pipe sent %d Famem allocation spans across %d layers",
            sum(len(descriptor.spans) for descriptor in weights),
            self._num_layers,
        )

    def wait_for_copier_ready(self) -> None:
        if self._copier is None or not self._descriptors_sent or self._pipeline_initialized:
            raise RuntimeError("Famem multiproc_pipe is not waiting for Copier initialization")
        try:
            self._copier.wait_until_initialized()
        except BaseException:
            self._close_after_terminal_failure("Copier initialization")
            raise
        self._pipeline_initialized = True

    def _mapping_descriptor(self) -> FamemMappingDesc:
        if self.client is None or not self.client.active:
            raise RuntimeError("Famem cannot describe an unmapped arena")
        sizes = self.client.extent_sizes
        shareable_handles = self.client.shareable_handles
        if len(sizes) != len(shareable_handles) or len(sizes) != len(self.client.extent_page_types):
            raise RuntimeError("Famem control-plane extent metadata does not match")
        address = self.base_address
        handles = []
        for size, shareable_handle in zip(sizes, shareable_handles):
            handles.append((self.device, size, address, shareable_handle))
            address += size
        if address != self.base_address + self.capacity:
            raise RuntimeError("Famem control-plane extents do not cover the arena")
        return FamemMappingDesc(
            capacity=self.capacity,
            extent_page_types=[int(value) for value in self.client.extent_page_types],
            handles=handles,
        )

    def _build_weight_descriptors(self) -> list[FamemSharedWeightDesc]:
        from vllm.model_executor.model_loader.base_loader import layer_to_addr

        allocations = self.native.worker_allocations()
        if not allocations:
            raise RuntimeError("multiproc_pipe found no live Famem weight allocations")
        allocations = sorted(allocations, key=lambda allocation: allocation.address)
        bases = [allocation.address for allocation in allocations]
        for allocation in allocations:
            offset = allocation.address - self.base_address
            if offset < 0 or allocation.aligned_size <= 0 or offset + allocation.aligned_size > self.weight_end:
                raise RuntimeError("Famem weight allocation lies outside the sealed weight region")
        for left, right in zip(allocations, allocations[1:]):
            if left.address + left.aligned_size > right.address:
                raise RuntimeError("Famem weight allocation ranges overlap")

        expected_layers = [
            "unknown",
            "pub",
            *(f"layers.{index}" for index in range(self._num_layers)),
        ]
        if list(layer_to_addr) != expected_layers:
            raise RuntimeError("The model-loader layer map does not match Famem multiproc_pipe")
        groups: dict[str, list[FamemNativeAllocation]] = {layer_name: [] for layer_name in expected_layers}

        def allocation_for(address: int) -> FamemNativeAllocation | None:
            position = bisect_right(bases, address) - 1
            if position < 0:
                return None
            allocation = allocations[position]
            if address < allocation.address + allocation.aligned_size:
                return allocation
            return None

        seen: set[int] = set()
        for layer_name in expected_layers:
            for address in layer_to_addr[layer_name]:
                allocation = allocation_for(address)
                if allocation is None or allocation.address in seen:
                    continue
                seen.add(allocation.address)
                groups[layer_name].append(allocation)
        groups["unknown"].extend(allocation for allocation in allocations if allocation.address not in seen)

        assert self.client is not None
        extent_ends = list(accumulate(self.client.extent_sizes))

        def split_span(allocation: FamemNativeAllocation) -> list[FamemWeightSpan]:
            offset = allocation.address - self.base_address
            remaining = allocation.aligned_size
            spans: list[FamemWeightSpan] = []
            while remaining:
                extent_index = bisect_right(extent_ends, offset)
                if extent_index >= len(extent_ends):
                    raise RuntimeError("Famem weight allocation exceeds its physical extents")
                size = min(remaining, extent_ends[extent_index] - offset)
                spans.append(FamemWeightSpan(offset, size))
                offset += size
                remaining -= size
            return spans

        layer_to_spans = {
            layer_name: [span for allocation in values for span in split_span(allocation)]
            for layer_name, values in groups.items()
        }
        self._backup_bytes = sum(span.size for spans in layer_to_spans.values() for span in spans)
        return [FamemSharedWeightDesc(layer_name, layer_to_spans[layer_name]) for layer_name in expected_layers]

    def _ensure_active(self) -> None:
        if self._closed:
            raise RuntimeError("Famem allocator is closed.")
        if self._poisoned:
            raise RuntimeError("Famem allocator is poisoned; restart the worker before allocating NPU memory.")
        if self._sleeping:
            raise RuntimeError("Famem allocator is sleeping.")
        if self.client is not None:
            return
        client_args: dict[str, Any] = {"native": self.native}
        if self._copier_tgid:
            client_args["copier_bare_tgid"] = self._copier_tgid
        client = self.client_factory(self.device, self.socket_dir, **client_args)
        try:
            base_address = client.acquire(self.capacity)
        except Exception:
            try:
                client.close()
            except Exception as cleanup_error:
                self.client = client
                self._poisoned = True
                self._closed = True
                raise RuntimeError("Famem arena acquisition failed and client cleanup also failed") from cleanup_error
            raise
        self.client = client
        self.base_address = base_address
        logger.info(
            "Famem arena mapped: device=%d capacity=%.2f GiB extents=%d",
            self.device,
            self.capacity / (1 << 30),
            len(client.extent_sizes),
        )

    def _get_pool(self, tag: str) -> Any:
        existing = self._pools.get(tag)
        if existing is not None:
            return existing
        if self._pluggable_allocator is None:
            self._pluggable_allocator = torch.npu.memory.NPUPluggableAllocator(
                self.native.library_path,
                "famem_malloc",
                "famem_free",
            )
        pool = torch.npu.memory.MemPool(self._pluggable_allocator._allocator)
        self._pools[tag] = pool
        return pool

    @contextmanager
    def use_memory_pool(self, tag: str | None = None) -> Iterator[None]:
        if tag not in _SUPPORTED_TAGS:
            raise ValueError(f"Famem pool tag must be one of {_SUPPORTED_TAGS}, got {tag!r}.")
        with self._lock:
            if self._active_pool_tag is not None:
                raise RuntimeError(f"Famem pool {self._active_pool_tag!r} is already active in this worker.")
            expected = _SUPPORTED_TAGS[len(self._entered_tags)] if len(self._entered_tags) < 2 else None
            if tag != expected:
                raise RuntimeError(
                    f"Famem pools must be entered once in order {_SUPPORTED_TAGS}; expected {expected!r}, got {tag!r}."
                )
            try:
                self._ensure_active()
                pool = self._get_pool(tag)
                self._active_pool_tag = tag
            except FamemBusyError:
                raise
            except BaseException:
                self._close_after_terminal_failure("memory-pool setup")
                raise

        try:
            with torch.npu.memory.use_mem_pool(pool, device=self.device):
                yield
        except BaseException:
            with self._lock:
                self._active_pool_tag = None
                self._poisoned = True
            self._close_after_terminal_failure("memory-pool transaction")
            raise
        with self._lock:
            self._active_pool_tag = None
            if tag == "weights":
                try:
                    self.weight_end = self.stats().heap_top
                except BaseException:
                    self._close_after_terminal_failure("weight-boundary query")
                    raise
                logger.info(
                    "Famem weight region sealed at %.2f GiB; %.2f GiB remains for KV cache.",
                    self.weight_end / (1 << 30),
                    (self.capacity - self.weight_end) / (1 << 30),
                )
            self._entered_tags.append(tag)

    @contextmanager
    def use_memory_pool_share(self, tag: str | None = None) -> Iterator[None]:
        """Expose the canonical Camem pipeline pool interface.

        Famem shares the server arena rather than individual Camem allocation
        handles, so its physical backend uses the ordinary tagged pool here.
        """
        with self.use_memory_pool(tag):
            yield

    @staticmethod
    def _resume_copier(copier: CopierProcess, mapping: FamemMappingDesc) -> None:
        copier.begin_resume()
        try:
            copier.send_recovery(mapping)
            copier.finish_resume()
        except BaseException:
            copier.abort_resume()
            raise

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        with self._lock:
            if self._cycle_state in {"suspended", "resuming"}:
                raise RuntimeError("A suspend cycle must be completed with resume, not wake_up")
            if offload_tags not in (None, "weights", ("weights",)):
                raise ValueError("Famem supports only level 1 sleep with weight offload.")
            self._require_pipeline_initialized()
            self._require_ready_for_sleep()
            assert self._copier is not None
            with self._transition("Copier suspend"):
                self._copier.suspend()
            with self._transition("sleep"):
                torch.npu.synchronize(self.device)
                assert self.client is not None
                self.client.sleep()
            self._sleeping = True
            self._cycle_state = "sync_sleeping"

    def wake_up(self, tags: list[str] | None = None) -> None:
        if tags is not None:
            raise ValueError("Famem v1 only supports whole-arena wake_up; pass tags=None.")
        with self._lock:
            if self._closed or not self._sleeping or self.client is None:
                raise RuntimeError("Famem allocator is not in a wakeable sleeping state.")
            if self._cycle_state in {"suspended", "resuming"}:
                raise RuntimeError("A suspend cycle must be completed with resume, not wake_up")
            if self._cycle_state != "sync_sleeping":
                raise RuntimeError("Famem allocator is not in a synchronous sleep cycle")
            if self._poisoned:
                raise RuntimeError("Famem allocator is poisoned; restart the worker before waking it.")
            self._require_pipeline_initialized()
            assert self._copier is not None
            with self._transition("wake", retry_busy=True):
                self.client.wake()
            with self._transition("Copier resume"):
                self._resume_copier(self._copier, self._mapping_descriptor())
            with self._transition("synchronous restore"):
                self._copier.wait_for_layer(self._num_layers - 1)
            self._sleeping = False
            self._cycle_state = "active"

    def suspend(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        """Relinquish the resident pool after the Copier has dropped its mapping."""
        if offload_tags not in (None, "weights", ("weights",)):
            raise ValueError("Famem suspend requires the immutable weight backup.")
        with self._lock:
            self._require_pipeline_initialized()
            if self._cycle_state == "sync_sleeping":
                raise RuntimeError("A sleep cycle must be completed with wake_up, not resume")
            if self._cycle_state == "suspended":
                raise RuntimeError("Famem allocator is already suspended")
            if self._cycle_state == "resuming":
                self.wait_for_layer(self._num_layers - 1)
                self.ready = True
            self._require_ready_for_sleep()
            assert self._copier is not None and self.client is not None
            with self._transition("Copier suspend"):
                self._copier.suspend()
            with self._transition("suspend"):
                torch.npu.synchronize(self.device)
                self.client.sleep()
            self._sleeping = True
            self._ready = False
            self._cycle_state = "suspended"

    def resume(self, tags: list[str] | None = None) -> None:
        """Remap the arena and start asynchronous layer-wise H2D restoration."""
        if tags is not None:
            raise ValueError("Famem v1 only supports whole-arena resume; pass tags=None.")
        with self._lock:
            self._require_pipeline_initialized()
            if self._cycle_state == "sync_sleeping":
                raise RuntimeError("A sleep cycle must be completed with wake_up, not resume")
            if self._cycle_state != "suspended" or not self._sleeping or self.client is None:
                raise RuntimeError("Famem allocator is not suspended")
            assert self._copier is not None
            copier = self._copier
            with self._transition("resume wake", retry_busy=True):
                self.client.wake()
            with self._transition("Copier resume"):
                self._resume_copier(copier, self._mapping_descriptor())
            self._sleeping = False
            self._cycle_state = "resuming"
        with self._transition("layer-zero restore"):
            copier.wait_for_layer(0)

    def wait_for_layer(self, layer_index: int) -> None:
        if self._ready:
            return
        if self._cycle_state != "resuming" or self._copier is None:
            raise RuntimeError("Famem layer readiness was requested outside a resume cycle")
        with self._transition(f"layer-{layer_index} restore"):
            self._copier.wait_for_layer(layer_index)

    def close(self) -> None:
        with self._lock:
            if self._cleanup_complete:
                return
            self._closed = True
            copier = self._copier
            first_error: Exception | None = None
            copier_stopped = copier is None
            copier_clean = copier is None
            if copier is not None:
                try:
                    copier.close()
                    copier_clean = bool(getattr(copier, "_cleanup_complete", True))
                    copier_stopped = True
                except Exception as error:
                    first_error = error
                    with suppress(Exception):
                        copier_stopped = not copier.process.is_alive()
                    copier_clean = bool(getattr(copier, "_cleanup_complete", False))
                if copier_clean:
                    self._copier = None
            if self.client is None:
                self._cleanup_complete = copier_clean
                if first_error is not None:
                    raise first_error
                return
            if not copier_stopped:
                # A live Copier still has access to the resident server pool.
                self.client.poisoned = True
            try:
                torch.npu.synchronize(self.device)
            except Exception as error:
                logger.warning("NPU synchronization failed during Famem shutdown: %s", error)
                self.client.poisoned = True
            try:
                self.client.close()
            except Exception as error:
                first_error = first_error or error
            client_clean = bool(getattr(self.client, "_cleanup_complete", self.client.closed))
            self._cleanup_complete = copier_clean and client_clean
            if first_error is not None:
                raise first_error

    def _close_after_terminal_failure(self, operation: str) -> None:
        self._poisoned = True
        try:
            self.close()
        except Exception as cleanup_error:
            raise RuntimeError(f"Famem {operation} failed and allocator cleanup also failed") from cleanup_error

    @contextmanager
    def _transition(self, operation: str, *, retry_busy: bool = False) -> Iterator[None]:
        try:
            yield
        except BaseException as error:
            if retry_busy and isinstance(error, FamemBusyError):
                raise
            self._fail_stop_after_transition_error(operation, error)

    def _fail_stop_after_transition_error(
        self,
        operation: str,
        error: BaseException,
    ) -> NoReturn:
        """Clean up and terminate a Worker after an ambiguous transition."""
        cleanup_error: BaseException | None = None
        try:
            self._close_after_terminal_failure(operation)
        except BaseException as caught_cleanup_error:
            cleanup_error = caught_cleanup_error

        message = (
            f"Famem {operation} left worker memory state non-recoverable; "
            "terminating the worker so the server can reuse its resident HBM pool. "
            f"Cause: {error}"
        )
        logger.critical(
            message,
            exc_info=(type(error), error, error.__traceback__),
        )
        if cleanup_error is not None:
            raise WorkerFatalError(f"{message} Cleanup also failed: {cleanup_error}") from cleanup_error
        raise WorkerFatalError(message) from error

    def _require_ready_for_sleep(self) -> None:
        if self._closed:
            raise RuntimeError("Famem allocator is closed.")
        if self._poisoned:
            raise RuntimeError("Famem allocator is poisoned; restart the worker before sleeping it.")
        if self._sleeping:
            raise RuntimeError("Famem allocator is already sleeping.")
        if self._cycle_state != "active":
            raise RuntimeError(f"Famem allocator cannot sleep from state {self._cycle_state!r}.")
        if self.client is None or self._entered_tags != list(_SUPPORTED_TAGS):
            raise RuntimeError("Famem can sleep only after both weight and KV cache pools are initialized.")
        if self._active_pool_tag is not None:
            raise RuntimeError("Famem cannot sleep while a memory-pool transaction is active.")

    def _require_pipeline_initialized(self) -> None:
        if not self._pipeline_initialized or self._copier is None:
            raise RuntimeError("Famem multiproc_pipe is not initialized")
