#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# CANN-mem-based pytorch pluggable allocator to implement sleep mode.
#
import dataclasses
import os
from bisect import bisect_right
from collections.abc import Callable
from contextlib import contextmanager
from threading import RLock
from typing import Any, NoReturn, cast

import torch
from vllm.logger import logger
from vllm.v1.worker.worker_base import WorkerFatalError

from vllm_ascend.worker.copier import (
    CopierProcess,
    RecoverWeightDesc,
    SharedWeightDesc,
)


def find_loaded_library(lib_name) -> str | None:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """  # noqa
    found_line = None
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found_line = line
                break
    if found_line is None:
        # the library is not loaded in the current process
        return None
    # if lib_name is libcudart, we need to match a line with:
    # address /path/to/libcudart-hash.so.11.0
    start = found_line.index("/")
    path = found_line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(lib_name), f"Unexpected filename: {filename} for library {lib_name}"
    return path


camem_available = False
try:
    from vllm_ascend.vllm_ascend_C import (  # type: ignore # noqa: F401
        init_module,
        init_module_share,
        python_copy_free,
        python_copy_malloc_use_share,
        python_create_and_map,
        python_create_and_map_share,
        python_create_and_map_share_alloc,
        python_memcpy_device_to_host,
        python_memcpy_host_to_device,
        python_share_memHandle_free,
        python_share_memHandle_import,
        python_unmap_and_release,
        python_unmap_and_release_share_alloc,
    )

    lib_name = find_loaded_library("vllm_ascend_C")
    camem_available = True
except ImportError as e:
    logger.warning("Failed to import vllm_ascend_C:%s. Sleep mode will be disabled. ", e)
    init_module = None
    init_module_share = None
    python_copy_free = None
    python_copy_malloc_use_share = None
    python_create_and_map = None
    python_create_and_map_share = None
    python_create_and_map_share_alloc = None
    python_memcpy_device_to_host = None
    python_memcpy_host_to_device = None
    python_share_memHandle_import = None
    python_share_memHandle_free = None
    python_unmap_and_release = None
    python_unmap_and_release_share_alloc = None
    lib_name = None
    libcudart = None

# The default Camem ABI remains the original four-field tuple. multiproc_pipe
# enables the fifth share_handle field consumed by the Copier process.
LegacyHandleType = tuple[int, int, int, int]
ShareHandleType = tuple[int, int, int, int, int]
SharedHandleType = ShareHandleType
HandleType = LegacyHandleType | SharedHandleType


@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None
    visit: bool = False


def _validate_native_handle(allocation_handle: object, expected_length: int) -> HandleType:
    if expected_length not in (4, 5):
        raise RuntimeError("Camem allocation handles must contain four or five fields")
    if not isinstance(allocation_handle, tuple) or len(allocation_handle) != expected_length:
        raise RuntimeError("Camem native extension returned an invalid allocation handle")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in allocation_handle):
        raise RuntimeError("Camem native extension returned an invalid allocation handle")
    if allocation_handle[1] == 0 or allocation_handle[2] == 0 or allocation_handle[3] == 0:
        raise RuntimeError("Camem native extension returned an invalid allocation handle")
    return allocation_handle


def _validate_handle(allocation_handle: object) -> HandleType:
    if not isinstance(allocation_handle, tuple) or len(allocation_handle) not in (4, 5):
        raise RuntimeError("Camem allocation handles must contain four or five fields")
    return _validate_native_handle(allocation_handle, len(allocation_handle))


def create_and_map(allocation_handle: LegacyHandleType) -> None:
    allocation_handle = _validate_native_handle(allocation_handle, 4)
    python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: LegacyHandleType) -> None:
    allocation_handle = _validate_native_handle(allocation_handle, 4)
    python_unmap_and_release(*allocation_handle)


def create_and_map_share_alloc(allocation_handle: SharedHandleType) -> None:
    allocation_handle = _validate_native_handle(allocation_handle, 5)
    python_create_and_map_share_alloc(*allocation_handle)


def unmap_and_release_share_alloc(allocation_handle: SharedHandleType) -> None:
    allocation_handle = _validate_native_handle(allocation_handle, 5)
    python_unmap_and_release_share_alloc(*allocation_handle)


def create_and_map_share(
    allocation_handle: SharedHandleType,
) -> SharedHandleType:
    allocation_handle = cast(SharedHandleType, _validate_native_handle(allocation_handle, 5))
    result = python_create_and_map_share(*allocation_handle)
    if not isinstance(result, int) or isinstance(result, bool) or result <= 0:
        raise RuntimeError("Camem resume returned an invalid shareable handle")
    return (*allocation_handle[:4], result)


def copy_malloc_use_share(
    allocation_handle: tuple[int, int, int, int],
) -> LegacyHandleType:
    if not isinstance(allocation_handle, tuple) or len(allocation_handle) != 4:
        raise RuntimeError("Camem Copier import requires a four-field descriptor")
    device, size, _, share_handle = allocation_handle
    result = python_copy_malloc_use_share(device, size, 0, share_handle)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("Camem Copier import returned an invalid mapping")
    return _validate_native_handle((device, size, result[0], result[1]), 4)


def copy_free(allocation_handle: LegacyHandleType) -> None:
    allocation_handle = _validate_native_handle(allocation_handle, 4)
    python_copy_free(*allocation_handle)


# The private 0.18.0 branch used both ``copy_*`` and ``copier_*`` spellings in
# callers. Keep both names over the same implementation.
copier_malloc_use_share = copy_malloc_use_share
copier_free = copy_free


def share_memHandle_import(share_handle: int, device: int) -> int:
    result = python_share_memHandle_import(share_handle, device)
    if not isinstance(result, int) or isinstance(result, bool) or result <= 0:
        raise RuntimeError("Camem share handle import returned an invalid physical handle")
    return result


def share_memHandle_free(physical_handle: int, device: int) -> None:
    if not isinstance(physical_handle, int) or isinstance(physical_handle, bool) or physical_handle <= 0:
        raise ValueError("Camem imported physical handle must be a positive integer")
    python_share_memHandle_free(physical_handle, device)


def get_pluggable_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
    enable_share_handle: bool = False,
) -> torch.npu.memory.NPUPluggableAllocator:
    if enable_share_handle:
        init_module_share(python_malloc_fn, python_free_func)
        malloc_name, free_name = "my_malloc_share", "my_free_share"
    else:
        init_module(python_malloc_fn, python_free_func)
        malloc_name, free_name = "my_malloc", "my_free"
    new_alloc = torch.npu.memory.NPUPluggableAllocator(lib_name, malloc_name, free_name)
    return new_alloc


@contextmanager
def use_memory_pool_with_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
    enable_share_handle: bool = False,
):
    new_alloc = get_pluggable_allocator(
        python_malloc_fn,
        python_free_func,
        enable_share_handle,
    )
    mem_pool = torch.npu.memory.MemPool(new_alloc._allocator)
    with torch.npu.memory.use_mem_pool(mem_pool):
        yield mem_pool, new_alloc


class CaMemAllocator:
    """
    A singleton class that manages a memory pool for CANN tensors.
    The memory in this pool can be offloaded or discarded when the
    allocator sleeps.
    Inside the `use_memory_pool(tag)` context, all tensors created will
    be allocated in the memory pool, and has the same tag as the
    tag passed to the context.
    When we call `sleep`, all tensors with the specified tag will be
    offloaded to CPU memory, and the rest of the tensors will be discarded.
    When we call `wake_up`, all tensors that are previously offloaded
    will be loaded back to GPU memory, and the rest of the tensors will
    have empty memory.
    Why it needs to be a singleton?
    When allocated tensors are garbage collected, PyTorch will call
    the free callback, which will call the `python_free_callback` method.
    The C-extension uses a global variable to store the function of an
    instance of this class. If we create multiple instances of this class,
    the global variable will be overwritten and the free callback will
    not work as expected.
    """

    instance = None
    default_tag: str = "default"
    pipeline_switch = False
    desc_queue: Any | None = None
    npu_recover_queue: Any | None = None
    ctrl_queue: Any | None = None
    layer_ready_events: dict[int, Any] | None = None

    @staticmethod
    def set_pipeline_switch(enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("multiproc_pipe switch must be a boolean")
        instance = CaMemAllocator.instance
        if instance is not None and instance.enable_share_handle != enabled:
            if instance.pointer_to_data or instance.allocator_and_pools:
                raise RuntimeError("Cannot change multiproc_pipe after Camem allocation")
            instance.enable_share_handle = enabled
        CaMemAllocator.pipeline_switch = enabled

    @staticmethod
    def set_desc_queue(queue: Any) -> None:
        CaMemAllocator.desc_queue = queue

    @staticmethod
    def set_npu_recover_queue(queue: Any) -> None:
        CaMemAllocator.npu_recover_queue = queue

    @staticmethod
    def set_ctrl_queue(queue: Any) -> None:
        CaMemAllocator.ctrl_queue = queue

    @staticmethod
    def set_layer_ready_events(events: dict[int, Any]) -> None:
        from vllm_ascend.worker.copier import set_layer_ready_events

        set_layer_ready_events(events)
        CaMemAllocator.layer_ready_events = events

    @staticmethod
    def _pipeline_resources(
        layer_ready_events: dict[int, Any] | None,
    ) -> tuple[tuple[Any, Any, Any] | None, dict[int, Any] | None]:
        queues = (
            CaMemAllocator.desc_queue,
            CaMemAllocator.npu_recover_queue,
            CaMemAllocator.ctrl_queue,
        )
        present = tuple(queue is not None for queue in queues)
        if any(present) and not all(present):
            raise RuntimeError("multiproc_pipe requires all three Camem queues")
        configured_events = CaMemAllocator.layer_ready_events
        if (
            layer_ready_events is not None
            and configured_events is not None
            and layer_ready_events is not configured_events
        ):
            raise RuntimeError("multiproc_pipe Worker and allocator must share the same layer events")
        selected_events = layer_ready_events if layer_ready_events is not None else configured_events
        return (cast(tuple[Any, Any, Any], queues) if all(present) else None), selected_events

    @staticmethod
    def get_instance() -> "CaMemAllocator":
        """
        CaMemAllocator is a singleton class.
        We cannot call the constructor directly.
        Call this method to get the instance.
        """
        if CaMemAllocator.instance is None:
            CaMemAllocator.instance = CaMemAllocator()
        return CaMemAllocator.instance

    def __init__(self):
        conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        assert "expandable_segments:True" not in conf, (
            "Expandable segments are not compatible with memory pool. "
            "Please track https://github.com/pytorch/pytorch/issues/147851 "
            "for the latest updates."
        )

        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = CaMemAllocator.default_tag
        self.allocator_and_pools: dict[str, Any] = {}
        self.enable_share_handle = CaMemAllocator.pipeline_switch
        self._copier: CopierProcess | None = None
        self._copier_tgid = 0
        self._pipeline_initialized = False
        self._descriptors_sent = False
        self._num_layers = 0
        self._ready = True
        self._cycle_state = "active"
        self._poisoned = False
        self._sync_sleeping_tags: set[str] = set()
        self._state_lock = RLock()
        self.layer_to_addr: dict[str, list[int]] = {}
        self.addr_to_layer: dict[int, str] = {}

    @property
    def ready(self) -> bool:
        return self._ready

    @ready.setter
    def ready(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("Camem ready must be a boolean")
        self._ready = value
        if value and self._cycle_state == "resuming":
            self._cycle_state = "active"

    @property
    def multiproc_pipe_enabled(self) -> bool:
        return self.enable_share_handle

    def start_pipeline(
        self,
        device: int,
        num_layers: int,
        layer_ready_events: dict[int, Any] | None = None,
        *,
        tp_size: int = 1,
        local_rank: int = 0,
    ) -> None:
        """Start Copier before the first shareable Camem allocation."""
        if not self.enable_share_handle:
            raise RuntimeError("Cannot start multiproc_pipe in four-field Camem mode")
        if self.pointer_to_data or self.allocator_and_pools:
            raise RuntimeError("multiproc_pipe must start before the Camem memory pools")
        if self._copier is not None:
            if self._copier.device != device or self._num_layers != num_layers:
                raise RuntimeError("Cannot reconfigure an existing multiproc_pipe Copier")
            return
        queues, layer_ready_events = self._pipeline_resources(layer_ready_events)
        copier_kwargs = {"queues": queues} if queues is not None else {}
        copier = CopierProcess(
            device=device,
            num_layers=num_layers,
            layer_ready_events=layer_ready_events,
            tp_size=tp_size,
            local_rank=local_rank,
            **copier_kwargs,
        )
        try:
            copier_tgid = copier.start()
        except BaseException:
            copier.close()
            raise
        self._copier = copier
        self._copier_tgid = copier_tgid
        self._num_layers = num_layers
        self.set_desc_queue(copier.desc_queue)
        self.set_npu_recover_queue(copier.npu_recover_queue)
        self.set_ctrl_queue(copier.ctrl_queue)
        self.set_layer_ready_events(copier.layer_ready_events)

    def send_descs_for_scattered_weights(self) -> None:
        """Send the original layer-grouped five-tuples to the Copier."""
        if not self.enable_share_handle or self._copier is None:
            raise RuntimeError("multiproc_pipe Copier was not started")
        if self._descriptors_sent or self._pipeline_initialized:
            raise RuntimeError("multiproc_pipe descriptors were already sent")
        try:
            torch.npu.synchronize()
            descriptors = self._build_weight_descriptors()
            self._copier.send_descriptors(descriptors)
        except BaseException as error:
            self._fail_stop_after_transition_error("Copier initialization", error)
        self._descriptors_sent = True
        logger.info(
            "multiproc_pipe sent %d Camem weight allocations across %d layers",
            sum(len(descriptor.handles) for descriptor in descriptors),
            self._num_layers,
        )

    def wait_for_copier_ready(self) -> None:
        if self._copier is None or not self._descriptors_sent or self._pipeline_initialized:
            raise RuntimeError("multiproc_pipe is not waiting for Copier initialization")
        try:
            self._copier.wait_until_initialized()
        except BaseException as error:
            self._fail_stop_after_transition_error("Copier initialization", error)
        self._pipeline_initialized = True

    def get_aligned_start(self, weight_ptr: int) -> int | None:
        """Return the owning Camem allocation for an interior weight address."""
        if not isinstance(weight_ptr, int) or isinstance(weight_ptr, bool) or weight_ptr < 0:
            raise TypeError("Camem weight pointer must be a non-negative integer")
        bases = sorted(pointer for pointer, data in self.pointer_to_data.items() if data.tag == "weights")
        position = bisect_right(bases, weight_ptr) - 1
        if position < 0:
            return None
        base = bases[position]
        return base if weight_ptr < base + self.pointer_to_data[base].handle[1] else None

    def _build_weight_descriptors(self) -> list[SharedWeightDesc]:
        from vllm.model_executor.model_loader.base_loader import layer_to_addr

        weight_allocations: dict[int, AllocationData] = {
            pointer: data for pointer, data in self.pointer_to_data.items() if data.tag == "weights"
        }
        if not weight_allocations:
            raise RuntimeError("multiproc_pipe found no Camem weight allocations")
        bases = sorted(weight_allocations)
        for index, base in enumerate(bases[:-1]):
            if base + weight_allocations[base].handle[1] > bases[index + 1]:
                raise RuntimeError("Camem weight allocation ranges overlap")

        expected_layers = [
            "unknown",
            "pub",
            *(f"layers.{index}" for index in range(self._num_layers)),
        ]
        if list(layer_to_addr) != expected_layers:
            raise RuntimeError("The model-loader layer map does not match multiproc_pipe")
        groups = {layer_name: [] for layer_name in expected_layers}
        seen: set[int] = set()
        for layer_name in expected_layers:
            for address in layer_to_addr[layer_name]:
                base = self.get_aligned_start(address)
                if base is None or base in seen:
                    continue
                seen.add(base)
                groups[layer_name].append(base)
        groups["unknown"].extend(base for base in bases if base not in seen)

        self.layer_to_addr = groups
        self.addr_to_layer = {
            pointer: layer_name for layer_name, pointers in self.layer_to_addr.items() for pointer in pointers
        }
        descriptors: list[SharedWeightDesc] = []
        for layer_name in expected_layers:
            handles: list[SharedHandleType] = []
            for pointer in self.layer_to_addr[layer_name]:
                handle = _validate_native_handle(weight_allocations[pointer].handle, 5)
                if handle[4] == 0:
                    raise RuntimeError("Cannot back up a sleeping Camem allocation")
                handles.append(handle)
            descriptors.append(SharedWeightDesc(layer_name=layer_name, handles=handles))
        return descriptors

    def _record_allocation(self, allocation_handle: object, expected_length: int) -> None:
        allocation_handle = _validate_native_handle(allocation_handle, expected_length)
        if expected_length == 5 and allocation_handle[4] == 0:
            raise RuntimeError("Camem allocation is missing its share handle")
        py_d_mem = allocation_handle[2]
        if py_d_mem in self.pointer_to_data:
            raise RuntimeError(f"Camem allocation pointer {py_d_mem:#x} is already tracked")
        self.pointer_to_data[py_d_mem] = AllocationData(allocation_handle, self.current_tag)

    def python_malloc_callback(self, allocation_handle: LegacyHandleType) -> None:
        """
        Internal method to store the allocation data
        when memory is allocated in the memory pool."""
        self._record_allocation(allocation_handle, 4)

    def python_malloc_share_callback(self, allocation_handle: SharedHandleType) -> None:
        """Track the original Camem five-field share callback ABI."""
        self._record_allocation(allocation_handle, 5)

    def _release_allocation(self, ptr: int, expected_length: int) -> HandleType:
        data = self.pointer_to_data.pop(ptr)
        if len(data.handle) != expected_length:
            self.pointer_to_data[ptr] = data
            raise RuntimeError("Camem free callback does not match its allocation mode")
        if data.cpu_backup_tensor is not None:
            data.cpu_backup_tensor = None
        return data.handle

    def python_free_callback(self, ptr: int) -> LegacyHandleType:
        """
        Internal method to look up the allocation data
        when memory is freed in the memory pool."""
        return cast(LegacyHandleType, self._release_allocation(ptr, 4))

    def python_free_share_callback(self, ptr: int) -> SharedHandleType:
        """Release the original Camem five-field share callback ABI."""
        return cast(SharedHandleType, self._release_allocation(ptr, 5))

    @staticmethod
    def _unmap_allocation(handle: HandleType) -> None:
        if len(handle) == 5:
            unmap_and_release_share_alloc(handle)
        else:
            unmap_and_release(handle)

    @staticmethod
    def _map_without_copier(handle: HandleType) -> None:
        if len(handle) == 5:
            create_and_map_share_alloc(handle)
        else:
            create_and_map(handle)

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        """
        Put the allocator in sleep mode.
        All data in the memory allocation with the specified tag will be
        offloaded to CPU memory, and others will be discarded.
        :param offload_tags: The tags of the memory allocation that will be
            offloaded. The rest of the memory allocation will be discarded.
        """
        if offload_tags is None:
            # by default, allocated tensors are offloaded
            # when the allocator sleeps
            offload_tags = (CaMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)
        elif not isinstance(offload_tags, tuple):
            raise TypeError("Camem offload_tags must be a string, tuple, or None")

        with self._state_lock:
            self._require_healthy()
            if self._cycle_state in {"suspended", "resuming"}:
                raise RuntimeError("A suspend cycle must be completed with resume, not wake_up")
            if self._cycle_state == "sync_sleeping":
                raise RuntimeError("Camem allocator is already sleeping")
            if self.enable_share_handle:
                self._require_pipeline_initialized()
            try:
                if self.enable_share_handle:
                    assert self._copier is not None
                    self._copier.suspend()
                torch.npu.synchronize()
                for ptr, data in self.pointer_to_data.items():
                    handle = data.handle
                    if data.tag in offload_tags:
                        size_in_bytes = handle[1]
                        cpu_backup_tensor = torch.empty(
                            size_in_bytes,
                            dtype=torch.uint8,
                            device="cpu",
                            pin_memory=True,
                        )
                        cpu_ptr = cpu_backup_tensor.data_ptr()
                        python_memcpy_device_to_host(handle[0], cpu_ptr, ptr, size_in_bytes)
                        data.cpu_backup_tensor = cpu_backup_tensor
                    self._unmap_allocation(handle)
            except BaseException as error:
                self._fail_stop_after_transition_error("synchronous sleep", error)
            self._sync_sleeping_tags = {data.tag for data in self.pointer_to_data.values()}
            self._cycle_state = "sync_sleeping"

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest of the data will have empty memory."""
        with self._state_lock:
            self._require_healthy()
            if self._cycle_state in {"suspended", "resuming"}:
                raise RuntimeError("A suspend cycle must be completed with resume, not wake_up")
            if self._cycle_state != "sync_sleeping":
                raise RuntimeError("Camem allocator is not sleeping")
            try:
                for ptr, data in self.pointer_to_data.items():
                    if tags is None or data.tag in tags:
                        handle = data.handle
                        self._map_without_copier(handle)
                        if data.cpu_backup_tensor is not None:
                            cpu_backup_tensor = data.cpu_backup_tensor
                            size_in_bytes = cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                            cpu_ptr = cpu_backup_tensor.data_ptr()
                            python_memcpy_host_to_device(handle[0], ptr, cpu_ptr, size_in_bytes)
                            data.cpu_backup_tensor = None
            except BaseException as error:
                self._fail_stop_after_transition_error("synchronous wake", error)
            if tags is None:
                self._sync_sleeping_tags.clear()
            else:
                self._sync_sleeping_tags.difference_update(tags)
            if not self._sync_sleeping_tags:
                self._cycle_state = "active"

    def suspend(
        self,
        offload_tags: tuple[str, ...] | str | None = None,
    ) -> None:
        """Release physical memory without a worker-side D2H backup."""
        if offload_tags not in (None, "weights", ("weights",)):
            raise ValueError("Camem Copier suspend requires the immutable weight backup")
        with self._state_lock:
            self._require_healthy()
            self._require_pipeline_initialized()
            if self._cycle_state == "sync_sleeping":
                raise RuntimeError("A sleep cycle must be completed with wake_up, not resume")
            if self._cycle_state == "suspended":
                raise RuntimeError("Camem allocator is already suspended")
            if self._cycle_state == "resuming":
                # A resumed model may be handed off again before it receives a
                # request. Complete the outstanding restore without requiring
                # a synthetic forward to mark the allocator ready.
                self.wait_for_layer(self._num_layers - 1)
                self.ready = True
            if self._cycle_state != "active":
                raise RuntimeError(f"Camem allocator cannot suspend from state {self._cycle_state!r}")
            assert self._copier is not None
            try:
                self._copier.suspend()
                torch.npu.synchronize()
                for data in self.pointer_to_data.values():
                    self._unmap_allocation(data.handle)
            except BaseException as error:
                self._fail_stop_after_transition_error("suspend", error)
            self._ready = False
            self._cycle_state = "suspended"

    def resume(self, tags: list[str] | None = None) -> None:
        """Map fresh backing and let Copier restore weights layer by layer."""
        if tags is not None and "weights" not in tags:
            raise ValueError("Camem Copier resume must include the weights tag")
        with self._state_lock:
            self._require_healthy()
            self._require_pipeline_initialized()
            if self._cycle_state == "sync_sleeping":
                raise RuntimeError("A sleep cycle must be completed with wake_up, not resume")
            if self._cycle_state != "suspended":
                raise RuntimeError("Camem allocator is not suspended")
            assert self._copier is not None
            copier = self._copier
            mapped: set[int] = set()
            try:
                copier.begin_resume()
                for layer_name in [
                    "unknown",
                    "pub",
                    *(f"layers.{index}" for index in range(self._num_layers)),
                ]:
                    handles: list[SharedHandleType] = []
                    for pointer in self.layer_to_addr[layer_name]:
                        data = self.pointer_to_data[pointer]
                        handle = create_and_map_share(cast(SharedHandleType, _validate_native_handle(data.handle, 5)))
                        data.handle = handle
                        handles.append(handle)
                        mapped.add(pointer)
                    copier.send_recovery_descriptor(RecoverWeightDesc(layer_name=layer_name, handles=handles))

                # Let Copier begin H2D while the Worker recreates blocks that
                # do not carry model weights.
                copier.finish_resume()

                # KV cache and allocator blocks outside the model layout need
                # backing memory but are not restored by the Copier.
                for pointer, data in self.pointer_to_data.items():
                    if pointer not in mapped:
                        self._map_without_copier(data.handle)
            except BaseException as error:
                try:
                    copier.abort_resume()
                except BaseException as abort_error:
                    logger.warning("Camem Copier resume abort failed: %s", abort_error)
                self._fail_stop_after_transition_error("resume", error)
            self._cycle_state = "resuming"

        # Resume request admission only after public weights and layer zero
        # have completed H2D restoration. Use the controller wait so a Copier
        # failure or timeout is surfaced instead of blocking the Worker forever.
        self.wait_for_layer(0)

    def wait_for_layer(self, layer_index: int) -> None:
        if self._ready:
            return
        if self._cycle_state != "resuming" or self._copier is None:
            raise RuntimeError("Layer readiness was requested outside a resume cycle")
        try:
            self._copier.wait_for_layer(layer_index)
        except BaseException as error:
            self._fail_stop_after_transition_error(f"layer-{layer_index} restore", error)

    def finish_forward(self) -> None:
        self.ready = True

    def close(self) -> None:
        copier = self._copier
        if copier is None:
            return
        copier.close()
        if getattr(copier, "_cleanup_complete", True):
            self._copier = None

    def _fail_stop_after_transition_error(self, operation: str, error: BaseException) -> NoReturn:
        """Terminate a worker after a potentially partial mapping transition."""
        self._poisoned = True
        self._ready = False
        self._cycle_state = "poisoned"
        cleanup_error: BaseException | None = None
        copier = self._copier
        if copier is not None:
            try:
                copier.close()
            except BaseException as caught_cleanup_error:
                cleanup_error = caught_cleanup_error
            else:
                self._copier = None

        message = (
            f"Camem {operation} left worker memory state non-recoverable; "
            f"terminating the worker to release its physical allocations. Cause: {error}"
        )
        logger.critical(message, exc_info=(type(error), error, error.__traceback__))
        if cleanup_error is not None:
            raise WorkerFatalError(f"{message} Copier cleanup also failed: {cleanup_error}") from cleanup_error
        raise WorkerFatalError(message) from error

    def _require_healthy(self) -> None:
        if self._poisoned:
            raise RuntimeError("Camem allocator is poisoned; restart the worker")

    def _require_pipeline_initialized(self) -> None:
        if not self.enable_share_handle or not self._pipeline_initialized or self._copier is None:
            raise RuntimeError("multiproc_pipe is not initialized")

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        """Use the original four-tuple Camem pool."""
        with self._use_memory_pool(tag, share=False):
            yield

    @contextmanager
    def use_memory_pool_share(self, tag: str | None = None):
        """Use the five-tuple shareable Camem pool for model weights."""
        if not self.enable_share_handle or self._copier is None:
            raise RuntimeError("Camem share pool requires a started multiproc_pipe Copier")
        try:
            with self._use_memory_pool(tag, share=True):
                yield
        except BaseException as error:
            self._fail_stop_after_transition_error("memory-pool transaction", error)

    @contextmanager
    def _use_memory_pool(self, tag: str | None, *, share: bool):
        """
        A context manager to use the memory pool.
        All memory allocation created inside the context will be allocated
        in the memory pool, and has the specified tag.
        :param tag: The tag of the memory allocation. If None, the default tag
            will be used.
        """
        if tag is None:
            tag = CaMemAllocator.default_tag

        assert isinstance(tag, str)

        old_tag = self.current_tag
        self.current_tag = tag
        try:
            malloc_callback = self.python_malloc_share_callback if share else self.python_malloc_callback
            free_callback = self.python_free_share_callback if share else self.python_free_callback
            with use_memory_pool_with_allocator(
                malloc_callback,
                free_callback,
                share,
            ) as data:
                # start to hit another PyTorch bug in PyTorch 2.6,
                # possibly because of gc-related issue w.r.t. the allocator and
                # the memory pool.
                # to avoid the issue, we keep a reference of the data.
                # see https://github.com/pytorch/pytorch/issues/146431 .
                self.allocator_and_pools[f"{'share' if share else 'local'}:{tag}"] = data
                yield
                # PyTorch's bug, calling torch.cuda.empty_cache() will error
                # when using pluggable allocator, see
                # https://github.com/pytorch/pytorch/issues/145168 .
                # if we have some memory allocated and then freed,
                # the memory will not be released.
                # right now it is fine, because we only use this allocator
                # during weight loading and kv cache creation, where we only
                # allocate memory.
                # TODO: we need to find a way to release the memory,
                # i.e. calling torch.cuda.empty_cache()
        finally:
            self.current_tag = old_tag

    def get_current_usage(self) -> int:
        """
        Get the total number of bytes allocated in the memory pool.
        """
        sum_bytes: int = 0
        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            sum_bytes += handle[1]
        return sum_bytes
