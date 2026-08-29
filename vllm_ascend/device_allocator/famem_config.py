# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from vllm.utils.mem_constants import GiB_bytes

FAMEM_CONFIG_KEY = "use_fast_map_allocator"
_LEGACY_FAMEM_CONFIG_KEY = "famem_allocator"
MAX_FAMEM_SIZE_GIB = 4096


@dataclasses.dataclass(frozen=True)
class FamemConfig:
    enabled: bool = False
    size_gib: int | None = None

    @property
    def size_bytes(self) -> int:
        if not self.enabled or self.size_gib is None:
            raise ValueError("Famem size is unavailable while the allocator is disabled or unconfigured.")
        return self.size_gib * GiB_bytes

    @classmethod
    def from_additional_config(cls, additional_config: Mapping[str, Any] | None) -> FamemConfig:
        if additional_config is None:
            return cls()
        if not isinstance(additional_config, Mapping):
            raise ValueError("additional_config must be an object when configuring Famem.")
        if _LEGACY_FAMEM_CONFIG_KEY in additional_config:
            raise ValueError(f"additional_config.famem_allocator was renamed to additional_config.{FAMEM_CONFIG_KEY}.")
        raw = additional_config.get(FAMEM_CONFIG_KEY)
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError(f"additional_config.{FAMEM_CONFIG_KEY} must be an object.")

        unknown = set(raw) - {"enabled", "size_gib"}
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"Unknown additional_config.{FAMEM_CONFIG_KEY} fields: {names}.")
        enabled = raw.get("enabled", False)
        size_gib = raw.get("size_gib")
        if not isinstance(enabled, bool):
            raise ValueError(f"additional_config.{FAMEM_CONFIG_KEY}.enabled must be a boolean.")
        if size_gib is not None and (not isinstance(size_gib, int) or isinstance(size_gib, bool)):
            raise ValueError(f"additional_config.{FAMEM_CONFIG_KEY}.size_gib must be an integer.")
        return cls(enabled=enabled, size_gib=size_gib)

    def validate(self) -> None:
        if not self.enabled:
            if self.size_gib is not None:
                raise ValueError("Famem allocator size was configured, but the allocator is not enabled.")
            return
        if self.size_gib is None or self.size_gib <= 0:
            raise ValueError("Famem requires a positive integer size in GiB.")
        if self.size_gib > MAX_FAMEM_SIZE_GIB:
            raise ValueError(f"Famem size cannot exceed {MAX_FAMEM_SIZE_GIB} GiB per NPU.")


def calculate_famem_memory_budget(
    *,
    initial_free_bytes: int,
    after_profile_free_bytes: int,
    torch_peak_increase_bytes: int,
    requested_memory_bytes: int,
    arena_remaining_bytes: int,
    granularity: int,
    pool_capacity_bytes: int,
) -> tuple[int, int]:
    values = (initial_free_bytes, after_profile_free_bytes, torch_peak_increase_bytes)
    values += (requested_memory_bytes, arena_remaining_bytes, pool_capacity_bytes)
    if any(value < 0 for value in values) or granularity <= 0:
        raise ValueError("Famem memory budget inputs must be non-negative and granularity must be positive.")
    if after_profile_free_bytes > initial_free_bytes:
        raise RuntimeError(
            "Free NPU memory increased during Famem profiling; isolate the worker before sizing KV cache."
        )
    physical_peak = pool_capacity_bytes + initial_free_bytes - after_profile_free_bytes + torch_peak_increase_bytes
    if physical_peak > requested_memory_bytes:
        raise MemoryError(
            "Famem arena plus non-pool runtime peak exceeds gpu_memory_utilization: "
            f"peak={physical_peak} bytes, budget={requested_memory_bytes} bytes."
        )
    available_kv = arena_remaining_bytes - arena_remaining_bytes % granularity
    return physical_peak, available_kv


def merge_famem_cli_config(
    additional_config: Mapping[str, Any] | None, *, enabled: bool | None, size_gib: int | None
) -> dict[str, Any]:
    merged = dict(additional_config or {})
    if _LEGACY_FAMEM_CONFIG_KEY in merged:
        raise ValueError(f"additional_config.famem_allocator was renamed to additional_config.{FAMEM_CONFIG_KEY}.")
    if enabled is None and size_gib is None:
        return merged

    raw = merged.get(FAMEM_CONFIG_KEY, {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"additional_config.{FAMEM_CONFIG_KEY} must be an object.")
    famem = dict(raw)
    famem.update((name, value) for name, value in (("enabled", enabled), ("size_gib", size_gib)) if value is not None)
    if enabled is not None and not enabled and size_gib is None:
        famem.pop("size_gib", None)
    merged[FAMEM_CONFIG_KEY] = famem
    return merged


def get_famem_config(vllm_config: Any) -> FamemConfig:
    return FamemConfig.from_additional_config(getattr(vllm_config, "additional_config", None))
