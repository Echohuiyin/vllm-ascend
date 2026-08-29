# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from __future__ import annotations

import ctypes
import dataclasses
import enum
from pathlib import Path
from typing import Any


class FamemWorkerState(enum.IntEnum):
    UNINITIALIZED = 0
    ACTIVE = 1
    SLEEPING = 2
    POISONED = 3
    CLOSED = 4


class FamemPageType(enum.IntEnum):
    """Physical page classes supported by the Famem HBM server."""

    HUGE_1G = 1
    HUGE_2M = 2

    @property
    def granularity_bytes(self) -> int:
        return 1 << (30 if self is FamemPageType.HUGE_1G else 21)


FAMEM_MAX_EXTENT_COUNT = 2
_PageTypes = list[FamemPageType]
_Ints = list[int]


@dataclasses.dataclass(frozen=True)
class FamemNativeStats:
    capacity: int
    heap_top: int
    live_bytes: int
    allocation_count: int
    base_address: int
    state: FamemWorkerState

    @property
    def freed_bytes(self) -> int:
        return self.heap_top - self.live_bytes


@dataclasses.dataclass(frozen=True)
class FamemNativeAllocation:
    address: int
    aligned_size: int


class FamemNativeLibrary:
    """Typed ctypes wrapper around the standalone Famem C ABI."""

    def __init__(self, library_path: str | None = None) -> None:
        self.library_path = library_path or str(Path(__file__).resolve().parents[1] / "libvllm_ascend_famem.so")
        try:
            self._library = ctypes.CDLL(self.library_path)
        except OSError as error:
            raise RuntimeError(
                f"Unable to load Famem native library {self.library_path!r}; reinstall vllm-ascend."
            ) from error
        try:
            self._configure_signatures()
        except AttributeError as error:
            raise RuntimeError(
                "The Famem native library is incompatible; rebuild and reinstall vllm-ascend."
            ) from error

    def _configure_signatures(self) -> None:
        library = self._library
        u64 = ctypes.c_uint64
        u64_pointer = ctypes.POINTER(u64)
        i32_pointer = ctypes.POINTER(ctypes.c_int32)
        extent_args = [ctypes.c_size_t, i32_pointer, u64_pointer, u64_pointer]
        signatures = {
            "famem_last_error": [],
            "famem_get_allocation_granularity": [ctypes.c_int, u64_pointer],
            "famem_get_device_uuid": [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t],
            "famem_get_bare_tgid": [i32_pointer],
            "famem_worker_prepare_v2": [ctypes.c_int, u64, *extent_args, u64_pointer],
            "famem_worker_unmap": [ctypes.c_int],
            "famem_worker_remap_v2": [ctypes.c_int, *extent_args],
            "famem_worker_release": [ctypes.c_int],
            "famem_worker_get_stats": [u64_pointer] * 5 + [ctypes.POINTER(ctypes.c_int)],
            "famem_worker_get_allocations": [ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), *[u64_pointer] * 2],
            "famem_memcpy_device_to_host": [u64, u64, u64],
            "famem_memcpy_host_to_device": [u64, u64, u64],
        }
        for name, argtypes in signatures.items():
            function = getattr(library, name)
            function.argtypes = argtypes
            function.restype = ctypes.c_char_p if name == "famem_last_error" else ctypes.c_int

    def _check(self, result: int, operation: str) -> None:
        if result:
            raw = self._library.famem_last_error()
            detail = raw.decode("utf-8", errors="replace") if raw else "unknown native error"
            raise RuntimeError(f"{operation} failed: {detail}")

    def allocation_granularity(self, device: int) -> int:
        output = ctypes.c_uint64()
        self._check(self._library.famem_get_allocation_granularity(device, ctypes.byref(output)), "query granularity")
        return output.value

    def device_uuid(self, device: int) -> str:
        output = ctypes.create_string_buffer(33)
        self._check(self._library.famem_get_device_uuid(device, output, len(output)), "query NPU UUID")
        return output.value.decode("ascii")

    def bare_tgid(self) -> int:
        output = ctypes.c_int32()
        self._check(self._library.famem_get_bare_tgid(ctypes.byref(output)), "querying the CANN bare TGID")
        return output.value

    @staticmethod
    def _extent_arrays(
        extent_page_types: _PageTypes, extent_sizes: _Ints, shareable_handles: _Ints
    ) -> tuple[Any, Any, Any]:
        if any(not isinstance(value, int) or isinstance(value, bool) for value in extent_page_types):
            raise ValueError("Famem physical page types must be integers.")
        try:
            page_types = [FamemPageType(value) for value in extent_page_types]
        except ValueError as error:
            raise ValueError("Famem received an unknown physical page type.") from error
        if (
            not 1 <= len(page_types) <= FAMEM_MAX_EXTENT_COUNT
            or len({len(page_types), len(extent_sizes), len(shareable_handles)}) != 1
            or page_types != sorted(set(page_types))
            or any(type(value) is not int or value <= 0 for value in [*extent_sizes, *shareable_handles])
            or any(size % page_type.granularity_bytes for page_type, size in zip(page_types, extent_sizes, strict=True))
        ):
            raise ValueError("Famem requires aligned, canonical positive extents and shareable handles.")
        page_type_array = ctypes.c_int32 * len(page_types)
        u64_array = ctypes.c_uint64 * len(page_types)
        return page_type_array(*page_types), u64_array(*extent_sizes), u64_array(*shareable_handles)

    def worker_prepare(
        self, device: int, capacity: int, extent_page_types: _PageTypes, extent_sizes: _Ints, shareable_handles: _Ints
    ) -> int:
        page_types, sizes, handles = self._extent_arrays(extent_page_types, extent_sizes, shareable_handles)
        base_address = ctypes.c_uint64()
        prepare = self._library.famem_worker_prepare_v2
        self._check(
            prepare(device, capacity, len(extent_sizes), page_types, sizes, handles, ctypes.byref(base_address)),
            "map worker arena",
        )
        return base_address.value

    def worker_unmap(self, device: int) -> None:
        self._check(self._library.famem_worker_unmap(device), "unmapping the Famem worker arena")

    def worker_remap(
        self, device: int, extent_page_types: _PageTypes, extent_sizes: _Ints, shareable_handles: _Ints
    ) -> None:
        page_types, sizes, handles = self._extent_arrays(extent_page_types, extent_sizes, shareable_handles)
        result = self._library.famem_worker_remap_v2(device, len(extent_sizes), page_types, sizes, handles)
        self._check(result, "remap worker arena")

    def worker_release(self, device: int) -> None:
        self._check(self._library.famem_worker_release(device), "releasing the Famem worker virtual address")

    def worker_stats(self) -> FamemNativeStats:
        outputs = [ctypes.c_uint64() for _ in range(5)]
        state = ctypes.c_int()
        result = self._library.famem_worker_get_stats(
            *(ctypes.byref(output) for output in outputs), ctypes.byref(state)
        )
        self._check(result, "read allocator statistics")
        try:
            worker_state = FamemWorkerState(state.value)
        except ValueError as error:
            raise RuntimeError(f"Famem native library returned unknown state {state.value}.") from error
        return FamemNativeStats(*(output.value for output in outputs), state=worker_state)

    def worker_allocations(self) -> list[FamemNativeAllocation]:
        stats = self.worker_stats()
        capacity = max(1, stats.allocation_count)
        array_type = ctypes.c_uint64 * capacity
        buffers = [array_type() for _ in range(2)]
        count = ctypes.c_size_t()
        result = self._library.famem_worker_get_allocations(capacity, ctypes.byref(count), *buffers)
        self._check(result, "read live allocations")
        if count.value > capacity:
            raise RuntimeError("Famem native library returned too many allocation records.")
        return [FamemNativeAllocation(*(buffer[index] for buffer in buffers)) for index in range(count.value)]

    def copy_device_to_host(self, host_address: int, device_address: int, size: int) -> None:
        result = self._library.famem_memcpy_device_to_host(host_address, device_address, size)
        self._check(result, "copy weights to host")

    def copy_host_to_device(self, device_address: int, host_address: int, size: int) -> None:
        result = self._library.famem_memcpy_host_to_device(device_address, host_address, size)
        self._check(result, "restore weights to device")
