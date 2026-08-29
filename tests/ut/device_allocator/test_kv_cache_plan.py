#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from copy import deepcopy

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.device_allocator.kv_cache_plan import (
    GIB,
    MIB,
    KVCacheAllocationPlan,
    KVCacheAllocationRequest,
    find_max_fitting_blocks,
    npu_allocation_segment_upper_bound,
    resize_kv_cache_config,
    validate_fit_response,
)


@pytest.mark.parametrize(
    ("requested_bytes", "expected_bytes"),
    [
        (1, 2 * MIB),
        (MIB - 32, 2 * MIB),
        (MIB - 31, 20 * MIB),
        (10 * MIB - 32, 10 * MIB),
        (10 * MIB - 31, 12 * MIB),
        (10 * MIB + 1, 12 * MIB),
    ],
)
def test_npu_allocation_segment_upper_bound(requested_bytes, expected_bytes):
    assert npu_allocation_segment_upper_bound(requested_bytes) == expected_bytes


def test_npu_allocation_segment_upper_bound_honors_1g_page_policy():
    assert npu_allocation_segment_upper_bound(800 * MIB, "page_size:1g") == GIB


def test_npu_allocation_segment_upper_bound_rejects_power2_rounding():
    with pytest.raises(ValueError, match="roundup_power2_divisions"):
        npu_allocation_segment_upper_bound(MIB, "roundup_power2_divisions:4")


def _production_kv_cache_config() -> KVCacheConfig:
    num_layers = 28
    num_blocks = 798
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=32,
        head_size=1024,
        head_size_v=1024,
        dtype=torch.float16,
    )
    layer_names = [f"model.layers.{index}.self_attn" for index in range(num_layers)]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=spec.page_size_bytes * num_blocks, shared_by=[layer_name]) for layer_name in layer_names
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)],
    )


def _split_kv_plan(config: KVCacheConfig) -> KVCacheAllocationPlan:
    requests = []
    for tensor in config.kv_cache_tensors:
        for _ in range(2):
            requests.append(
                KVCacheAllocationRequest(
                    payload_bytes=tensor.size // 2,
                    alignment_bytes=2 * MIB,
                )
            )
    return KVCacheAllocationPlan(tuple(requests))


def test_production_alignment_regression_reduces_blocks_before_allocation():
    config = _production_kv_cache_config()
    plan = _split_kv_plan(config)
    available_bytes = 44_708 * MIB

    assert sum(request.payload_bytes for request in plan.requests) == 44_688 * MIB
    assert sum(request.requested_bytes for request in plan.requests) == 44_800 * MIB
    assert plan.native_bytes_upper_bound() == 44_912 * MIB
    assert plan.native_bytes_upper_bound() - available_bytes == 204 * MIB

    fit = find_max_fitting_blocks(config, available_bytes, _split_kv_plan)

    assert fit.num_blocks == 795
    assert fit.native_bytes == 44_688 * MIB
    assert fit.native_bytes <= available_bytes


def test_resize_kv_cache_config_preserves_bytes_per_block():
    config = _production_kv_cache_config()
    original = deepcopy(config)

    resize_kv_cache_config(config, 795)

    assert config.num_blocks == 795
    for old_tensor, new_tensor in zip(original.kv_cache_tensors, config.kv_cache_tensors, strict=True):
        assert new_tensor.size == old_tensor.size // original.num_blocks * 795


def test_validate_fit_response_rejects_boolean_integer():
    response = {
        "num_blocks": True,
        "native_bytes": 2 * MIB,
        "available_bytes": 2 * MIB,
    }

    with pytest.raises(RuntimeError, match="non-integer"):
        validate_fit_response(response)
