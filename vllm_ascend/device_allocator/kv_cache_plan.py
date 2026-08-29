# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from vllm.v1.kv_cache_interface import KVCacheConfig

MIB, GIB = 1 << 20, 1 << 30
NPU_MIN_BLOCK_BYTES, NPU_COMPAT_PADDING_BYTES = 512, 32
NPU_SMALL_POOL_LIMIT_BYTES, NPU_SMALL_SEGMENT_BYTES = MIB, 2 * MIB
NPU_MEDIUM_REQUEST_LIMIT_BYTES, NPU_MEDIUM_SEGMENT_BYTES = 10 * MIB, 20 * MIB
NPU_LARGE_SEGMENT_ROUND_BYTES, NPU_HUGE_SEGMENT_ROUND_BYTES = 2 * MIB, GIB
KV_CACHE_ADDRESS_ALIGNMENT_BYTES = 2 * MIB


def align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("Alignment inputs must be non-negative and alignment must be positive.")
    return (value + alignment - 1) // alignment * alignment


def validate_npu_allocator_config(allocator_config: str) -> None:
    normalized_config = allocator_config.lower().replace(" ", "")
    if "expandable_segments:true" in normalized_config:
        raise ValueError("Famem memory pools are incompatible with expandable_segments:True.")
    if "roundup_power2_divisions:" in normalized_config:
        raise ValueError(
            "Famem does not support PYTORCH_NPU_ALLOC_CONF=roundup_power2_divisions because it changes "
            "the physical request size used for KV cache calibration."
        )


def npu_allocation_segment_upper_bound(requested_bytes: int, allocator_config: str = "") -> int:
    validate_npu_allocator_config(allocator_config)
    if requested_bytes < 0:
        raise ValueError("NPU allocation size must be non-negative.")
    if requested_bytes == 0:
        return 0

    rounded_bytes = align_up(requested_bytes + NPU_COMPAT_PADDING_BYTES, NPU_MIN_BLOCK_BYTES)
    if rounded_bytes <= NPU_SMALL_POOL_LIMIT_BYTES:
        return NPU_SMALL_SEGMENT_BYTES
    if rounded_bytes < NPU_MEDIUM_REQUEST_LIMIT_BYTES:
        return NPU_MEDIUM_SEGMENT_BYTES

    normalized_config = allocator_config.lower().replace(" ", "")
    rounds = NPU_LARGE_SEGMENT_ROUND_BYTES, NPU_HUGE_SEGMENT_ROUND_BYTES
    return align_up(rounded_bytes, rounds["page_size:1g" in normalized_config])


@dataclasses.dataclass(frozen=True)
class KVCacheAllocationRequest:
    payload_bytes: int
    alignment_bytes: int = 0

    def __post_init__(self) -> None:
        if self.payload_bytes < 0:
            raise ValueError("KV cache payload size must be non-negative.")
        if self.alignment_bytes < 0:
            raise ValueError("KV cache alignment padding must be non-negative.")

    @property
    def requested_bytes(self) -> int:
        return self.payload_bytes + self.alignment_bytes


@dataclasses.dataclass(frozen=True)
class KVCacheAllocationPlan:
    requests: tuple[KVCacheAllocationRequest, ...]

    def native_bytes_upper_bound(self, allocator_config: str = "") -> int:
        bound = npu_allocation_segment_upper_bound
        return sum(bound(request.requested_bytes, allocator_config) for request in self.requests)


@dataclasses.dataclass(frozen=True)
class KVCacheBlockFit:
    num_blocks: int
    native_bytes: int
    available_bytes: int

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


def resize_kv_cache_config(kv_cache_config: KVCacheConfig, num_blocks: int) -> None:
    old_num_blocks = kv_cache_config.num_blocks
    if old_num_blocks <= 0:
        raise ValueError("Cannot resize a KV cache config with no blocks.")
    if num_blocks <= 0 or num_blocks > old_num_blocks:
        raise ValueError(f"KV cache block target must be in [1, {old_num_blocks}], got {num_blocks}.")
    if num_blocks == old_num_blocks:
        return

    for tensor in kv_cache_config.kv_cache_tensors:
        if tensor.size % old_num_blocks:
            raise ValueError(
                f"KV cache tensor size {tensor.size} is not divisible by its {old_num_blocks} configured blocks."
            )
        tensor.size = tensor.size // old_num_blocks * num_blocks
    kv_cache_config.num_blocks = num_blocks


_PlanBuilder = Callable[[KVCacheConfig], KVCacheAllocationPlan]


def find_max_fitting_blocks(
    kv_cache_config: KVCacheConfig, available_bytes: int, plan_builder: _PlanBuilder, allocator_config: str = ""
) -> KVCacheBlockFit:
    if available_bytes < 0:
        raise ValueError("Available KV cache memory must be non-negative.")
    if kv_cache_config.num_blocks <= 0:
        raise ValueError("KV cache config must contain at least one block.")

    def build(candidate_blocks: int) -> tuple[KVCacheAllocationPlan, int]:
        candidate = deepcopy(kv_cache_config)
        resize_kv_cache_config(candidate, candidate_blocks)
        plan = plan_builder(candidate)
        return plan, plan.native_bytes_upper_bound(allocator_config)

    def result(blocks: int, plan: KVCacheAllocationPlan | None, native_bytes: int) -> KVCacheBlockFit:
        return KVCacheBlockFit(blocks, native_bytes, available_bytes)

    current_blocks = kv_cache_config.num_blocks
    current_plan, current_native_bytes = build(current_blocks)
    if current_native_bytes <= available_bytes:
        return result(current_blocks, current_plan, current_native_bytes)

    best: tuple[int, KVCacheAllocationPlan | None, int] = (0, None, 0)
    low, high = 1, current_blocks - 1
    while low <= high:
        candidate_blocks = (low + high) // 2
        candidate_plan, candidate_native_bytes = build(candidate_blocks)
        if candidate_native_bytes <= available_bytes:
            best = candidate_blocks, candidate_plan, candidate_native_bytes
            low = candidate_blocks + 1
        else:
            high = candidate_blocks - 1

    return result(*best)


def validate_fit_response(response: Any) -> KVCacheBlockFit:
    if not isinstance(response, dict):
        raise RuntimeError("Famem KV cache calibration returned a non-object response.")
    field_names = {field.name for field in dataclasses.fields(KVCacheBlockFit)}
    if set(response) != field_names:
        raise RuntimeError("Famem KV cache calibration returned unexpected fields.")
    if any(not isinstance(response[name], int) or isinstance(response[name], bool) for name in field_names):
        raise RuntimeError("Famem KV cache calibration returned non-integer values.")
    if min(response[name] for name in field_names) < 0:
        raise RuntimeError("Famem KV cache calibration returned negative values.")
    return KVCacheBlockFit(**response)
