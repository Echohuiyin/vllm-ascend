#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from __future__ import annotations

from typing import Any

from vllm_ascend import envs
from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.device_allocator.famem import FaMemAllocator
from vllm_ascend.device_allocator.famem_config import get_famem_config
from vllm_ascend.device_allocator.multiproc_pipe_config import is_multiproc_pipe_enabled

_active_sleep_mode_allocator: CaMemAllocator | FaMemAllocator | None = None


def get_active_sleep_mode_allocator() -> CaMemAllocator | FaMemAllocator:
    """Return the allocator used by the current worker's model forward."""
    if _active_sleep_mode_allocator is None:
        raise RuntimeError("The worker sleep-mode allocator is not initialized")
    return _active_sleep_mode_allocator


def _register_active_allocator(
    allocator: CaMemAllocator | FaMemAllocator,
) -> CaMemAllocator | FaMemAllocator:
    global _active_sleep_mode_allocator
    _active_sleep_mode_allocator = allocator
    return allocator


def create_sleep_mode_allocator(vllm_config: Any, device: Any) -> CaMemAllocator | FaMemAllocator:
    famem_config = get_famem_config(vllm_config)
    multiproc_pipe = is_multiproc_pipe_enabled(vllm_config)
    if not famem_config.enabled:
        CaMemAllocator.set_pipeline_switch(multiproc_pipe)
        return _register_active_allocator(CaMemAllocator.get_instance())
    if not multiproc_pipe:
        raise ValueError("Famem requires multiproc_pipe to be enabled.")
    device_index = getattr(device, "index", device)
    if device_index is None:
        raise ValueError("Famem requires an explicit NPU device index.")
    return _register_active_allocator(
        FaMemAllocator(
            device=int(device_index),
            capacity=famem_config.size_bytes,
            socket_dir=envs.VLLM_ASCEND_FAMEM_SOCKET_DIR,
        )
    )
