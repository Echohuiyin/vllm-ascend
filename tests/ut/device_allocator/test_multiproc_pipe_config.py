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

from types import SimpleNamespace

import pytest

from vllm_ascend.device_allocator.multiproc_pipe_config import (
    SUPPORTED_MULTIPROC_PIPE_TP_SIZES,
    is_multiproc_pipe_enabled,
    merge_multiproc_pipe_cli_config,
)


def test_multiproc_pipe_defaults_to_disabled():
    assert not is_multiproc_pipe_enabled(SimpleNamespace(additional_config=None))
    assert not is_multiproc_pipe_enabled(SimpleNamespace(additional_config={}))


def test_multiproc_pipe_supported_tensor_parallel_sizes():
    assert SUPPORTED_MULTIPROC_PIPE_TP_SIZES == (1, 2, 4, 8)


def test_multiproc_pipe_cli_overrides_additional_config():
    assert merge_multiproc_pipe_cli_config(
        {"multiproc_pipe": False, "other": True},
        enabled=True,
    ) == {"multiproc_pipe": True, "other": True}


def test_multiproc_pipe_unspecified_preserves_additional_config():
    assert merge_multiproc_pipe_cli_config(
        {"multiproc_pipe": True, "other": True},
        enabled=None,
    ) == {"multiproc_pipe": True, "other": True}


@pytest.mark.parametrize("value", [1, "true", {}, []])
def test_multiproc_pipe_cli_rejects_non_boolean_value(value):
    with pytest.raises(ValueError, match="CLI value must be a boolean"):
        merge_multiproc_pipe_cli_config({}, enabled=value)


@pytest.mark.parametrize("value", [1, "true", {}, []])
def test_multiproc_pipe_rejects_non_boolean_value(value):
    config = SimpleNamespace(additional_config={"multiproc_pipe": value})
    with pytest.raises(ValueError, match="must be a boolean"):
        is_multiproc_pipe_enabled(config)
