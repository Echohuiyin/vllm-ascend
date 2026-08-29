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

import pytest

from vllm_ascend.device_allocator.famem_config import (
    FAMEM_CONFIG_KEY,
    FamemConfig,
    calculate_famem_memory_budget,
    merge_famem_cli_config,
)


def test_famem_config_defaults_disabled():
    config = FamemConfig.from_additional_config(None)
    assert config == FamemConfig()
    config.validate()


def test_famem_config_rejects_non_object_additional_config():
    with pytest.raises(ValueError, match="additional_config must be an object"):
        FamemConfig.from_additional_config([])


def test_famem_config_parses_enabled_size():
    config = FamemConfig.from_additional_config({FAMEM_CONFIG_KEY: {"enabled": True, "size_gib": 48}})
    config.validate()
    assert config.size_bytes == 48 * (1 << 30)


def test_famem_config_rejects_renamed_key():
    with pytest.raises(ValueError, match="renamed to additional_config.use_fast_map_allocator"):
        FamemConfig.from_additional_config({"famem_allocator": {"enabled": True, "size_gib": 48}})


@pytest.mark.parametrize(
    "raw,match",
    [
        ({FAMEM_CONFIG_KEY: []}, "must be an object"),
        ({FAMEM_CONFIG_KEY: {"enabled": 1}}, "must be a boolean"),
        ({FAMEM_CONFIG_KEY: {"enabled": True}}, "positive integer size"),
        ({FAMEM_CONFIG_KEY: {"enabled": False, "size_gib": 1}}, "not enabled"),
        ({FAMEM_CONFIG_KEY: {"enabled": True, "size_gib": 0}}, "positive integer size"),
        ({FAMEM_CONFIG_KEY: {"enabled": True, "size_gib": 4097}}, "cannot exceed"),
        ({FAMEM_CONFIG_KEY: {"unknown": True}}, "Unknown"),
    ],
)
def test_famem_config_rejects_invalid_values(raw, match):
    with pytest.raises(ValueError, match=match):
        FamemConfig.from_additional_config(raw).validate()


def test_cli_values_override_programmatic_config():
    merged = merge_famem_cli_config(
        {FAMEM_CONFIG_KEY: {"enabled": True, "size_gib": 32}, "other": 1},
        enabled=False,
        size_gib=40,
    )
    assert merged == {
        FAMEM_CONFIG_KEY: {"enabled": False, "size_gib": 40},
        "other": 1,
    }


def test_cli_disable_removes_programmatic_size():
    merged = merge_famem_cli_config(
        {FAMEM_CONFIG_KEY: {"enabled": True, "size_gib": 32}},
        enabled=False,
        size_gib=None,
    )

    assert merged[FAMEM_CONFIG_KEY] == {"enabled": False}


def test_famem_memory_budget_uses_physical_peak_and_aligns_kv():
    physical_peak, available_kv = calculate_famem_memory_budget(
        initial_free_bytes=10000,
        after_profile_free_bytes=4000,
        torch_peak_increase_bytes=1000,
        requested_memory_bytes=8000,
        arena_remaining_bytes=2500,
        granularity=1024,
        pool_capacity_bytes=500,
    )
    assert physical_peak == 7500
    assert available_kv == 2048


def test_famem_memory_budget_rejects_runtime_peak_over_budget():
    with pytest.raises(MemoryError, match="exceeds gpu_memory_utilization"):
        calculate_famem_memory_budget(
            initial_free_bytes=10000,
            after_profile_free_bytes=9000,
            torch_peak_increase_bytes=0,
            requested_memory_bytes=8000,
            arena_remaining_bytes=1024,
            granularity=1024,
            pool_capacity_bytes=7500,
        )


def test_famem_memory_budget_rejects_external_memory_release():
    with pytest.raises(RuntimeError, match="increased during Famem profiling"):
        calculate_famem_memory_budget(
            initial_free_bytes=10000,
            after_profile_free_bytes=11000,
            torch_peak_increase_bytes=0,
            requested_memory_bytes=8000,
            arena_remaining_bytes=1024,
            granularity=1024,
            pool_capacity_bytes=0,
        )
