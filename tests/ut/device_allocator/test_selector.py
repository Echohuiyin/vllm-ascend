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
from unittest.mock import patch

import pytest

from vllm_ascend.device_allocator.selector import (
    create_sleep_mode_allocator,
    get_active_sleep_mode_allocator,
)


@pytest.mark.parametrize(
    ("multiproc_pipe", "enable_share_handle"),
    [
        (False, False),
        (True, True),
    ],
    ids=("camem-default", "camem-multiproc-pipe"),
)
def test_selector_supports_camem_modes(multiproc_pipe, enable_share_handle):
    additional_config = {"multiproc_pipe": multiproc_pipe}
    config = SimpleNamespace(additional_config=additional_config)
    camem_instance = object()
    with (
        patch(
            "vllm_ascend.device_allocator.selector.CaMemAllocator.get_instance",
            return_value=camem_instance,
        ) as get_camem_instance,
        patch("vllm_ascend.device_allocator.selector.CaMemAllocator.set_pipeline_switch") as set_pipeline_switch,
    ):
        allocator = create_sleep_mode_allocator(config, 1)
        assert get_active_sleep_mode_allocator() is allocator

    assert allocator is camem_instance
    set_pipeline_switch.assert_called_once_with(enable_share_handle)
    get_camem_instance.assert_called_once_with()
