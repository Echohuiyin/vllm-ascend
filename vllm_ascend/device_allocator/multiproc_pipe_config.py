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

from collections.abc import Mapping
from typing import Any

MULTIPROC_PIPE_CONFIG_KEY = "multiproc_pipe"
SUPPORTED_MULTIPROC_PIPE_TP_SIZES = (1, 2, 4, 8)


def merge_multiproc_pipe_cli_config(
    additional_config: Mapping[str, Any] | None,
    *,
    enabled: bool | None,
) -> dict[str, Any]:
    """Merge the CLI compatibility switch into additional_config."""
    if additional_config is not None and not isinstance(additional_config, Mapping):
        raise ValueError("additional_config must be an object when configuring multiproc_pipe.")
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("multiproc_pipe CLI value must be a boolean.")
    merged = dict(additional_config or {})
    if enabled is not None:
        merged[MULTIPROC_PIPE_CONFIG_KEY] = enabled
    return merged


def is_multiproc_pipe_enabled(vllm_config: Any) -> bool:
    additional_config = getattr(vllm_config, "additional_config", None)
    if additional_config is None:
        return False
    if not isinstance(additional_config, Mapping):
        raise ValueError("additional_config must be an object when configuring multiproc_pipe.")
    enabled = additional_config.get(MULTIPROC_PIPE_CONFIG_KEY, False)
    if not isinstance(enabled, bool):
        raise ValueError("additional_config.multiproc_pipe must be a boolean.")
    return enabled
