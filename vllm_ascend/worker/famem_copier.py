# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing.synchronize import Event
from typing import Any

import torch

from vllm_ascend.device_allocator.famem_native import FamemNativeLibrary, FamemPageType
from vllm_ascend.worker.copier import (
    MSG_NPU_MEM_READY,
    MSG_SHUTDOWN,
    _allocate_host_backup,
    _require_layer_name,
    _validate_int_handle,
)


@dataclass(frozen=True)
class FamemWeightSpan:
    offset: int
    size: int


@dataclass
class FamemSharedWeightDesc:
    layer_name: str
    spans: list[FamemWeightSpan]
    cpu_tensors: list[torch.Tensor] = field(default_factory=list)


@dataclass(frozen=True)
class FamemMappingDesc:
    capacity: int
    extent_page_types: list[int]
    handles: list[tuple[int, int, int, int]]


def _validate_mapping(mapping: FamemMappingDesc) -> tuple[int, list[FamemPageType], list[int], list[int]]:
    if not isinstance(mapping, FamemMappingDesc) or mapping.capacity <= 0 or not mapping.handles:
        raise RuntimeError("Copier received an invalid Famem mapping")
    if len(mapping.handles) != len(mapping.extent_page_types) or len(mapping.handles) > 2:
        raise RuntimeError("Famem mapping extent metadata does not match")
    page_types = [FamemPageType(value) for value in mapping.extent_page_types]
    if page_types != sorted(set(page_types)):
        raise RuntimeError("Famem mapping page types are not in canonical order")

    device = -1
    sizes: list[int] = []
    shareable_handles: list[int] = []
    expected_address = 0
    for index, handle in enumerate(mapping.handles):
        extent_device, size, address, shareable_handle = _validate_int_handle(
            handle, 4, "Famem requires an active four-field extent handle"
        )
        if index == 0:
            device = extent_device
            expected_address = address
        if extent_device != device or address != expected_address:
            raise RuntimeError("Famem extent handles do not form one contiguous arena")
        if size % page_types[index].granularity_bytes:
            raise RuntimeError("Famem extent size is not aligned to its page type")
        sizes.append(size)
        shareable_handles.append(shareable_handle)
        expected_address += size
    if sum(sizes) != mapping.capacity:
        raise RuntimeError("Famem extent handles do not cover the configured arena")
    return device, page_types, sizes, shareable_handles


def _validate_weight_layout(descriptors, capacity: int, num_layers: int) -> list[FamemSharedWeightDesc]:
    expected_layers = {"unknown", "pub", *(f"layers.{index}" for index in range(num_layers))}
    received_layers = {descriptor.layer_name for descriptor in descriptors}
    if received_layers != expected_layers or len(descriptors) != len(expected_layers):
        raise RuntimeError("Famem weight descriptors do not cover every model layer exactly once")
    occupied: list[tuple[int, int]] = []
    for descriptor in descriptors:
        if descriptor.cpu_tensors:
            raise RuntimeError("Worker must not populate Copier-owned Famem backup tensors")
        for span in descriptor.spans:
            if not isinstance(span, FamemWeightSpan) or span.offset < 0 or span.size <= 0:
                raise RuntimeError("Famem received an invalid weight span")
            if span.offset > capacity or span.size > capacity - span.offset:
                raise RuntimeError("Famem weight span exceeds the arena")
            occupied.append((span.offset, span.offset + span.size))
    occupied.sort()
    if any(left[1] > right[0] for left, right in zip(occupied, occupied[1:])):
        raise RuntimeError("Famem weight spans overlap")
    return sorted(descriptors, key=lambda descriptor: _require_layer_name(descriptor.layer_name))


class FamemCopierOps:
    def __init__(self, device: int) -> None:
        self.device = device
        self.native = FamemNativeLibrary()
        self.native.device_uuid(device)
        self.base_address = 0
        self.prepared = self.mapped = False

    def prepare(self, mapping: FamemMappingDesc) -> None:
        device, page_types, sizes, handles = _validate_mapping(mapping)
        if device != self.device:
            raise RuntimeError("Famem mapping targets the wrong Copier device")
        self.base_address = self.native.worker_prepare(device, mapping.capacity, page_types, sizes, handles)
        self.prepared = self.mapped = True

    def remap(self, mapping: FamemMappingDesc) -> None:
        device, page_types, sizes, handles = _validate_mapping(mapping)
        if device != self.device:
            raise RuntimeError("Famem remap targets the wrong Copier device")
        if self.mapped:
            raise RuntimeError("Famem Copier is already mapped")
        self.native.worker_remap(device, page_types, sizes, handles)
        self.mapped = True

    def unmap(self) -> None:
        """Drop only this process's mapping/imports, never server-owned memory."""
        if self.mapped:
            self.native.worker_unmap(self.device)
            self.mapped = False

    def release(self) -> None:
        self.unmap()
        if self.prepared:
            self.native.worker_release(self.device)
            self.prepared = False

    def copy_device_to_host(self, tensor: torch.Tensor, span: FamemWeightSpan) -> None:
        self.native.copy_device_to_host(tensor.data_ptr(), self.base_address + span.offset, span.size)

    def copy_host_to_device(self, span: FamemWeightSpan, tensor: torch.Tensor) -> None:
        self.native.copy_host_to_device(self.base_address + span.offset, tensor.data_ptr(), span.size)


def _mapping_layout(mapping: FamemMappingDesc) -> tuple[int, tuple[int, ...], tuple[tuple[int, int, int, int], ...]]:
    _, page_types, _, _ = _validate_mapping(mapping)
    return mapping.capacity, tuple(int(value) for value in page_types), tuple(mapping.handles)


def _receive_recovery_message(recover_queue: Any) -> Any:
    message = recover_queue.get()
    if message == MSG_SHUTDOWN:
        raise InterruptedError("Copier shutdown requested during recovery")
    return message


class FamemCopierMemoryBackend:
    """Famem physical-memory hooks for the canonical Camem Copier loop."""

    def __init__(self, device: int, layer_events: dict[int, Event]) -> None:
        self.ops = FamemCopierOps(device)
        self.layer_events = layer_events
        self.initial_layout: tuple[int, tuple[int, ...], tuple[tuple[int, int, int, int], ...]] | None = None
        self.backups: list[FamemSharedWeightDesc] = []

    def bare_tgid(self) -> int:
        return self.ops.native.bare_tgid()

    def initialize(self, desc_queue: Any) -> None:
        initial = desc_queue.get()
        if initial == MSG_SHUTDOWN:
            raise InterruptedError("Copier shutdown requested during initialization")
        if not isinstance(initial, tuple) or len(initial) != 2:
            raise RuntimeError("Famem Copier received an invalid initialization descriptor")
        mapping, weights = initial
        if not isinstance(mapping, FamemMappingDesc) or not isinstance(weights, list):
            raise RuntimeError("Famem Copier received an invalid initialization descriptor")
        self.initial_layout = _mapping_layout(mapping)
        self.ops.prepare(mapping)
        self.backups = _validate_weight_layout(weights, mapping.capacity, len(self.layer_events))
        for descriptor in self.backups:
            for span in descriptor.spans:
                tensor = _allocate_host_backup(span.size)
                self.ops.copy_device_to_host(tensor, span)
                descriptor.cpu_tensors.append(tensor)
        self.ops.unmap()

    def suspend(self) -> None:
        self.ops.unmap()

    def resume(self, recover_queue: Any) -> None:
        mapping = _receive_recovery_message(recover_queue)
        if _receive_recovery_message(recover_queue) != MSG_NPU_MEM_READY:
            raise RuntimeError("Famem Copier recovery transaction is incomplete")
        if _mapping_layout(mapping) != self.initial_layout:
            raise RuntimeError("Famem recovery changed the server-owned physical handles")
        self.ops.remap(mapping)
        for descriptor in self.backups:
            for span, tensor in zip(descriptor.spans, descriptor.cpu_tensors, strict=True):
                self.ops.copy_host_to_device(span, tensor)
            layer_index = _require_layer_name(descriptor.layer_name)
            if layer_index >= 0:
                self.layer_events[layer_index].set()

    def close(self) -> None:
        self.ops.release()
