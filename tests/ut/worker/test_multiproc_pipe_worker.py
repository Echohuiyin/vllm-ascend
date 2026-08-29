# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.worker.worker import NPUWorker


@patch(
    "vllm_ascend.worker.worker.is_multiproc_pipe_enabled",
    return_value=False,
)
@patch("vllm_ascend.worker.worker.logger")
def test_suspend_resume_guard_and_argument_forwarding(
    mock_logger: MagicMock,
    mock_is_enabled: MagicMock,
) -> None:
    worker = object.__new__(NPUWorker)
    worker.vllm_config = MagicMock()
    worker._get_sleep_mode_allocator = MagicMock()
    worker._restore_moe_weight_layout = MagicMock()

    worker.suspend(level=1)
    worker.resume(tags=["weights"])

    assert mock_is_enabled.call_count == 2
    assert mock_logger.warning.call_count == 2
    worker._get_sleep_mode_allocator.assert_not_called()

    mock_is_enabled.return_value = True
    allocator = MagicMock()
    worker._get_sleep_mode_allocator.return_value = allocator
    with patch(
        "vllm_ascend.worker.worker.torch.npu.mem_get_info",
        side_effect=[(100, 1000), (200, 1000)],
    ):
        with pytest.raises(ValueError, match="only level 1"):
            worker.suspend(level=2)
        worker.suspend(level=1)
    worker.resume(tags=["weights"])

    allocator.suspend.assert_called_once_with(offload_tags=("weights",))
    allocator.resume.assert_called_once_with(["weights"])
    worker._restore_moe_weight_layout.assert_not_called()

    model_config = MagicMock(enable_sleep_mode=True)
    model_config.get_num_layers.return_value = 3
    worker.vllm_config.model_config = worker.model_config = model_config
    worker.parallel_config = SimpleNamespace(tensor_parallel_size=4)
    worker.local_rank = 2
    worker.device = SimpleNamespace(index=2)
    worker.layer_ready_events = {index: MagicMock() for index in range(3)}
    worker.model_runner = MagicMock()
    allocator.get_current_usage.return_value = 0
    allocator.multiproc_pipe_enabled = True
    allocator.use_memory_pool_share.return_value = nullcontext()
    with patch("vllm_ascend.worker.worker.set_current_vllm_config", return_value=nullcontext()):
        worker.load_model()

    allocator.start_pipeline.assert_called_once_with(
        2,
        3,
        worker.layer_ready_events,
        tp_size=4,
        local_rank=2,
    )
    allocator.send_descs_for_scattered_weights.assert_called_once_with()
    allocator.wait_for_copier_ready.assert_called_once_with()


@patch("vllm_ascend.worker.worker.ensure_kv_transfer_shutdown")
def test_shutdown_without_sleep_allocator_still_releases_services(
    mock_kv_transfer_shutdown: MagicMock,
) -> None:
    worker = object.__new__(NPUWorker)
    worker._sleep_mode_allocator = None
    worker.profiler = MagicMock()
    worker.weight_transfer_engine = MagicMock()

    worker.shutdown()

    mock_kv_transfer_shutdown.assert_called_once_with()
    worker.profiler.shutdown.assert_called_once_with()
    worker.weight_transfer_engine.shutdown.assert_called_once_with()
