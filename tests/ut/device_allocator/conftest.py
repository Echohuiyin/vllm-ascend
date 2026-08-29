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

import importlib
import sys
import types
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock

import torch

try:
    importlib.import_module("torch_npu")
except ModuleNotFoundError as error:
    if error.name != "torch_npu":
        raise
    torch_npu_module = types.ModuleType("torch_npu")
    torch_npu_module.__spec__ = ModuleSpec("torch_npu", loader=None)
    torch_npu_module.npu = MagicMock(name="torch_npu.npu")
    torch_npu_module.npu_fusion_attention = MagicMock(name="torch_npu.npu_fusion_attention")
    sys.modules.setdefault("torch_npu", torch_npu_module)


try:
    importlib.import_module("acl.rt")
except ModuleNotFoundError as error:
    if error.name not in {"acl", "acl.rt"}:
        raise
    acl_module = types.ModuleType("acl")
    acl_rt_module = types.ModuleType("acl.rt")
    acl_rt_module.memcpy = MagicMock(name="acl.rt.memcpy")
    acl_module.rt = acl_rt_module
    sys.modules.setdefault("acl", acl_module)
    sys.modules.setdefault("acl.rt", acl_rt_module)

if not hasattr(torch, "npu"):
    torch.npu = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=MagicMock(return_value=False),
        synchronize=MagicMock(name="torch.npu.synchronize"),
        set_device=MagicMock(name="torch.npu.set_device"),
        memory=types.SimpleNamespace(
            NPUPluggableAllocator=MagicMock,
            MemPool=MagicMock,
            use_mem_pool=MagicMock(),
        ),
    )
