# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import os
import subprocess
import sys
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.worker.copier import (
    MSG_COPIER_INIT_DONE,
    MSG_COPIER_STARTED,
    MSG_COPIER_SUSPEND_DONE,
    MSG_NPU_MEM_READY,
    MSG_RESUME_START,
    MSG_SHUTDOWN,
    MSG_SUSPEND_START,
    CopierProcess,
    _CopierControlChannel,
    _run_copier_with_parent_guard,
    copier_main,
)
from vllm_ascend.worker.famem_copier import (
    FamemCopierMemoryBackend,
    FamemMappingDesc,
    FamemSharedWeightDesc,
    FamemWeightSpan,
    _validate_mapping,
    _validate_weight_layout,
    _mapping_layout,
)

MIB = 1 << 20


class FakeTensor:
    def __init__(self, size: int):
        self.size = size

    def numel(self) -> int:
        return self.size

    def element_size(self) -> int:
        return 1

    def data_ptr(self) -> int:
        return 1234


class FakeQueue:
    def __init__(self, values):
        self.values = deque(values)

    def get(self):
        return self.values.popleft()


class FakeControl:
    def __init__(self, messages):
        self.messages = deque(messages)
        self.sent = []

    def get(self):
        return self.messages.popleft()

    def put(self, message):
        self.sent.append(message)


class FakeOps:
    def __init__(self, device):
        self.device = device
        self.native = self
        self.operations = []
        self.imported_handles = []
        self.mapped = False

    def bare_tgid(self):
        self.operations.append(("bare_tgid",))
        return 4321

    def prepare(self, mapping):
        self.operations.append(("prepare", mapping.handles[0][3]))
        self.imported_handles.append(tuple(mapping.handles))
        self.mapped = True

    def remap(self, mapping):
        self.operations.append(("remap", mapping.handles[0][3]))
        self.imported_handles.append(tuple(mapping.handles))
        self.mapped = True

    def unmap(self):
        if self.mapped:
            self.operations.append(("unmap",))
            self.mapped = False

    def release(self):
        self.unmap()
        self.operations.append(("release",))

    def copy_device_to_host(self, tensor, span):
        self.operations.append(("d2h", span.offset, span.size))

    def copy_host_to_device(self, span, tensor):
        self.operations.append(("h2d", span.offset, span.size))


def _mapping(shareable_handle: int) -> FamemMappingDesc:
    capacity = 2 * MIB
    return FamemMappingDesc(
        capacity=capacity,
        extent_page_types=[2],
        handles=[(0, capacity, 0x20000000, shareable_handle)],
    )


def _weights() -> list[FamemSharedWeightDesc]:
    return [
        FamemSharedWeightDesc("unknown", [FamemWeightSpan(0, 256)]),
        FamemSharedWeightDesc("pub", [FamemWeightSpan(256, 512)]),
        FamemSharedWeightDesc("layers.0", [FamemWeightSpan(768, 1024)]),
    ]


def test_mapping_validation_preserves_four_tuple_extent_layout():
    device, page_types, sizes, handles = _validate_mapping(_mapping(800))

    assert device == 0
    assert [int(page_type) for page_type in page_types] == [2]
    assert sizes == [2 * MIB]
    assert handles == [800]
    assert _mapping_layout(_mapping(800)) != _mapping_layout(_mapping(801))

    backend = object.__new__(FamemCopierMemoryBackend)
    with pytest.raises(InterruptedError, match="shutdown requested"):
        backend.resume(FakeQueue([MSG_SHUTDOWN]))
    with pytest.raises(InterruptedError, match="shutdown requested"):
        backend.resume(FakeQueue([_mapping(800), MSG_SHUTDOWN]))


def test_weight_layout_rejects_missing_layers_and_overlaps():
    with pytest.raises(RuntimeError, match="every model layer"):
        _validate_weight_layout([FamemSharedWeightDesc("pub", [])], 4096, 1)

    overlapping = [
        FamemSharedWeightDesc("unknown", []),
        FamemSharedWeightDesc("pub", [FamemWeightSpan(0, 1024)]),
        FamemSharedWeightDesc("layers.0", [FamemWeightSpan(512, 1024)]),
    ]
    with pytest.raises(RuntimeError, match="overlap"):
        _validate_weight_layout(overlapping, 4096, 1)


def test_famem_uses_the_shared_copier_process_and_backend():
    context = MagicMock()
    context.get_start_method.return_value = "spawn"

    with patch("vllm_ascend.worker.copier.get_mp_context", return_value=context):
        copier = CopierProcess(
            0,
            1,
            {0: MagicMock()},
            backend_factory=FamemCopierMemoryBackend,
            label="Famem Copier",
        )

    process_args = context.Process.call_args.kwargs
    assert process_args["target"] is _run_copier_with_parent_guard
    assert process_args["args"][0] == copier._owner_pid
    assert process_args["args"][1] is copier_main
    assert len(process_args["args"][2]) == 6
    channel = process_args["args"][2][2]
    assert isinstance(channel, _CopierControlChannel)
    assert channel.backend_factory is FamemCopierMemoryBackend


def test_shared_copier_main_unmaps_famem_imports_and_remaps_same_handles():
    mapping = _mapping(800)
    init_queue = FakeQueue([(mapping, _weights())])
    recover_queue = FakeQueue(
        [
            mapping,
            MSG_NPU_MEM_READY,
            mapping,
            MSG_NPU_MEM_READY,
        ]
    )
    control = FakeControl(
        [
            MSG_SUSPEND_START,
            MSG_RESUME_START,
            MSG_SUSPEND_START,
            MSG_RESUME_START,
            MSG_SHUTDOWN,
        ]
    )
    responses = FakeControl([])
    event = MagicMock()
    ops = FakeOps(0)
    fake_torch = SimpleNamespace(npu=SimpleNamespace(set_device=MagicMock()))

    with (
        patch("vllm_ascend.worker.famem_copier.FamemCopierOps", return_value=ops),
        patch("vllm_ascend.worker.copier.torch", fake_torch),
        patch(
            "vllm_ascend.worker.famem_copier._allocate_host_backup",
            side_effect=lambda size: FakeTensor(size),
        ),
    ):
        copier_main(
            init_queue,
            recover_queue,
            _CopierControlChannel(control, responses, 0, FamemCopierMemoryBackend),
            1,
            0,
            {0: event},
        )

    assert responses.sent == [
        (MSG_COPIER_STARTED, 4321),
        MSG_COPIER_INIT_DONE,
        MSG_COPIER_SUSPEND_DONE,
        MSG_COPIER_SUSPEND_DONE,
    ]
    assert ops.operations == [
        ("bare_tgid",),
        ("prepare", 800),
        ("d2h", 0, 256),
        ("d2h", 256, 512),
        ("d2h", 768, 1024),
        ("unmap",),
        ("remap", 800),
        ("h2d", 0, 256),
        ("h2d", 256, 512),
        ("h2d", 768, 1024),
        ("unmap",),
        ("remap", 800),
        ("h2d", 0, 256),
        ("h2d", 256, 512),
        ("h2d", 768, 1024),
        ("unmap",),
        ("release",),
    ]
    assert ops.imported_handles == [tuple(mapping.handles)] * 3
    assert event.clear.call_count == 2
    assert event.set.call_count == 2


def test_shared_controller_close_releases_queues_before_process_start():
    copier = CopierProcess.__new__(CopierProcess)
    copier._cleanup_complete = False
    copier._closed = False
    copier._started = False
    queues = [MagicMock() for _ in range(4)]
    copier.desc_queue, copier.npu_recover_queue, copier.ctrl_queue, copier.ctrl_response_queue = queues

    copier.close()

    assert copier._cleanup_complete is True
    for queue in queues:
        queue.cancel_join_thread.assert_called_once_with()
        queue.close.assert_called_once_with()


def test_shared_controller_close_does_not_flush_to_a_dead_reader():
    code = """
import multiprocessing as mp
from vllm_ascend.worker.copier import CopierProcess

context = mp.get_context("spawn")
queues = [context.Queue() for _ in range(4)]
queues[0].put(b"x" * (8 << 20))
copier = CopierProcess.__new__(CopierProcess)
copier._cleanup_complete = False
copier._closed = False
copier._started = False
copier.desc_queue, copier.npu_recover_queue, copier.ctrl_queue, copier.ctrl_response_queue = queues
copier.close()
assert copier._cleanup_complete
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "VLLM_PLUGINS": "", "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert completed.returncode == 0, completed.stderr


def test_shared_controller_close_can_retry_a_process_that_did_not_stop():
    copier = CopierProcess.__new__(CopierProcess)
    copier._cleanup_complete = False
    copier._closed = False
    copier._started = True
    copier._initialized = True
    copier.process = MagicMock()
    copier.process.is_alive.return_value = True
    queues = [MagicMock() for _ in range(4)]
    copier.desc_queue, copier.npu_recover_queue, copier.ctrl_queue, copier.ctrl_response_queue = queues

    with pytest.raises(RuntimeError, match="did not exit"):
        copier.close()
    assert copier._cleanup_complete is False

    copier.process.is_alive.return_value = False
    copier.close()
    assert copier._cleanup_complete is True
