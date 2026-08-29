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

from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
from vllm_ascend.ops.linear_op import Flashcomm2OProjRowParallelOp
from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding


def test_alltoall_exposes_persistent_routing_tensor() -> None:
    routing_tensor = torch.ones(4, dtype=torch.int32)
    scratch_tensor = torch.zeros(4, dtype=torch.int32)
    dispatcher = MagicMock(
        expert_ids_per_ep_rank=routing_tensor,
        expert_token_nums=scratch_tensor,
    )
    comm_method = MagicMock(token_dispatcher=dispatcher)

    with patch(
        "vllm_ascend.ops.fused_moe.fused_moe.get_moe_comm_method",
        return_value=comm_method,
    ):
        tensors = AscendFusedMoE.get_multiproc_pipe_persistent_tensors(object.__new__(AscendFusedMoE))

    assert len(tensors) == 1
    assert tensors[0] is routing_tensor
    assert all(tensor is not scratch_tensor for tensor in tensors)


def test_flashcomm2_oproj_exposes_persistent_group_indices() -> None:
    group_indices = torch.tensor([1, 0], dtype=torch.int64)
    custom_op = object.__new__(Flashcomm2OProjRowParallelOp)
    custom_op.group_indices = group_indices

    tensors = custom_op.get_multiproc_pipe_persistent_tensors()

    assert len(tensors) == 1
    assert tensors[0] is group_indices


def test_default_rope_exposes_detached_global_caches() -> None:
    cos_cache = torch.ones(4, 8)
    sin_cache = torch.zeros(4, 8)

    with (
        patch("vllm_ascend.ops.rotary_embedding._cos_cache", cos_cache),
        patch("vllm_ascend.ops.rotary_embedding._sin_cache", sin_cache),
    ):
        tensors = AscendRotaryEmbedding.get_multiproc_pipe_persistent_tensors(object())

    assert len(tensors) == 2
    assert tensors[0] is cos_cache
    assert tensors[1] is sin_cache
