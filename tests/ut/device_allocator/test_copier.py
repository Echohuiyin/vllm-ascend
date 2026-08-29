# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import os
import signal
import socket
import subprocess
import sys
import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.worker.copier import (
    MSG_COPIER_INIT_DONE,
    MSG_COPIER_STARTED,
    MSG_COPIER_SUSPEND_DONE,
    MSG_DESC_SEND_FINISH,
    MSG_NPU_MEM_READY,
    MSG_RESUME_START,
    MSG_SHUTDOWN,
    MSG_SUSPEND_START,
    CamemCopierMemoryBackend,
    CopierProcess,
    RecoverWeightDesc,
    SharedWeightDesc,
    _CopierControlChannel,
    _arm_parent_death_signal,
    _require_copier_start_owner,
    _require_spawn_context,
    _run_copier_with_parent_guard,
    backup_weight_descriptors,
    copier_main,
    import_recovery_descriptors,
    parse_layer_index,
    restore_weight_descriptors,
    should_preload,
)


def test_copier_requires_spawn_context():
    context = MagicMock()
    context.get_start_method.return_value = "fork"

    with pytest.raises(RuntimeError, match="VLLM_WORKER_MULTIPROC_METHOD=spawn"):
        _require_spawn_context(context)


class SpawnSmokeBackend:
    def __init__(self, device, layer_events):
        del device, layer_events

    def bare_tgid(self):
        return os.getpid()

    def initialize(self, desc_queue):
        assert desc_queue.get() == MSG_DESC_SEND_FINISH

    def suspend(self):
        pass

    def resume(self, recover_queue):
        del recover_queue

    def close(self):
        pass


def test_copier_process_uses_parent_guard_and_shared_main(monkeypatch):
    context = MagicMock()
    context.get_start_method.return_value = "spawn"
    queues = (MagicMock(), MagicMock(), MagicMock())

    with patch("vllm_ascend.worker.copier.get_mp_context", return_value=context):
        copier = CopierProcess(0, 1, {0: MagicMock()}, queues=queues, tp_size=4, local_rank=2)

    process_args = context.Process.call_args.kwargs
    assert process_args["target"] is _run_copier_with_parent_guard
    assert process_args["args"][0] == copier._owner_pid
    assert process_args["args"][1] is copier_main
    assert len(process_args["args"][2]) == 6
    control = process_args["args"][2][2]
    assert isinstance(control, _CopierControlChannel)
    assert control.commands is copier.ctrl_queue is queues[2]
    assert control.backend_factory is CamemCopierMemoryBackend
    assert process_args["args"][2][3:5] == (4, 2)

    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    real_copier = CopierProcess(0, 1, timeout=10, backend_factory=SpawnSmokeBackend)
    try:
        assert real_copier.start() > 0
        real_copier.send_initial(MSG_DESC_SEND_FINISH)
        real_copier.wait_until_initialized()
        real_copier.suspend()
    finally:
        real_copier.close()
    assert real_copier._cleanup_complete


def test_copier_must_start_from_worker_main_thread():
    worker_thread = object()
    main_thread = object()

    with (
        patch("vllm_ascend.worker.copier.threading.current_thread", return_value=worker_thread),
        patch("vllm_ascend.worker.copier.threading.main_thread", return_value=main_thread),
        pytest.raises(RuntimeError, match="Worker main thread"),
    ):
        _require_copier_start_owner(os.getpid())


def test_parent_death_guard_arms_signal_before_running_target():
    calls = []

    with patch(
        "vllm_ascend.worker.copier._arm_parent_death_signal",
        side_effect=lambda parent_pid: calls.append(("guard", parent_pid)),
    ):
        _run_copier_with_parent_guard(
            123,
            lambda value: calls.append(("target", value)),
            ("ready",),
        )

    assert calls == [("guard", 123), ("target", "ready")]


def test_parent_death_signal_checks_parent_after_prctl():
    prctl = MagicMock(return_value=0)
    libc = SimpleNamespace(prctl=prctl)

    with (
        patch("vllm.utils.system_utils.ctypes.CDLL", return_value=libc),
        patch("vllm.utils.system_utils.os.getppid", return_value=123),
    ):
        _arm_parent_death_signal(123)

    prctl.assert_called_once_with(1, int(signal.SIGKILL), 0, 0, 0)


def test_parent_death_signal_kills_child_if_parent_already_exited():
    libc = SimpleNamespace(prctl=MagicMock(return_value=0))

    with (
        patch("vllm.utils.system_utils.ctypes.CDLL", return_value=libc),
        patch("vllm.utils.system_utils.os.getppid", return_value=1),
        patch("vllm.utils.system_utils.os.getpid", return_value=456),
        patch("vllm.utils.system_utils.os.kill") as kill,
        pytest.raises(RuntimeError, match="parent exited"),
    ):
        _arm_parent_death_signal(123)

    kill.assert_called_once_with(456, signal.SIGKILL)


def test_parent_death_signal_fails_closed_when_prctl_fails():
    libc = SimpleNamespace(prctl=MagicMock(return_value=-1))

    with (
        patch("vllm.utils.system_utils.ctypes.CDLL", return_value=libc),
        patch("vllm.utils.system_utils.ctypes.get_errno", return_value=1),
        pytest.raises(OSError),
    ):
        _arm_parent_death_signal(123)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-specific")
def test_parent_death_signal_terminates_child_when_creator_thread_exits():
    parent_socket, child_socket = socket.socketpair()
    processes = []
    launch_errors = []
    child_code = """
import os
import signal
import sys
from vllm_ascend.worker.copier import _arm_parent_death_signal

_arm_parent_death_signal(int(sys.argv[1]))
os.write(int(sys.argv[2]), b"ready")
os.close(int(sys.argv[2]))
signal.pause()
"""

    def launch_from_worker_thread():
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    child_code,
                    str(os.getpid()),
                    str(child_socket.fileno()),
                ],
                env={**os.environ, "VLLM_PLUGINS": "", "PYTHONDONTWRITEBYTECODE": "1"},
                pass_fds=(child_socket.fileno(),),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append(process)
            child_socket.close()
            parent_socket.settimeout(20)
            if parent_socket.recv(5, socket.MSG_WAITALL) != b"ready":
                launch_errors.append("Copier child exited before arming parent-death protection")
        except BaseException as error:
            launch_errors.append(repr(error))
        finally:
            child_socket.close()
            parent_socket.close()

    # Linux tracks the thread that created the child. Keep that thread alive
    # until the child reports that PR_SET_PDEATHSIG has been armed, then let it
    # exit while this pytest process remains available to reap the child.
    launcher = threading.Thread(target=launch_from_worker_thread)
    launcher.start()
    launcher.join(timeout=25)

    assert not launcher.is_alive(), "Timed out while starting guarded Copier child"
    assert processes, launch_errors
    process = processes[0]
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
        pytest.fail(f"Copier survived creator-thread exit\nstdout:\n{stdout}\nstderr:\n{stderr}")

    assert not launch_errors, launch_errors
    assert process.returncode == -signal.SIGKILL, f"stdout:\n{stdout}\nstderr:\n{stderr}"


class FakeTensor:
    def __init__(self, size: int):
        self.size = size

    def numel(self) -> int:
        return self.size

    def element_size(self) -> int:
        return 1

    def data_ptr(self) -> int:
        return 1234


class FakeOps:
    def __init__(self):
        self.imported = []
        self.freed = []
        self.d2h = []
        self.h2d = []

    def import_handle(self, handle):
        local = (*handle[:2], handle[2] + 10_000, handle[3] + 10_000)
        self.imported.append((handle, local))
        return local

    def free_handle(self, handle):
        self.freed.append(handle)

    def copy_device_to_host(self, tensor, handle):
        self.d2h.append((tensor, handle))

    def copy_host_to_device(self, handle, tensor):
        self.h2d.append((handle, tensor))

    def _get_bare_tgid(self, device):
        assert device == 0
        return 4321

    def bare_tgid(self, device):
        return self._get_bare_tgid(device)


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


@pytest.mark.parametrize(
    ("name", "index", "preload"),
    [
        ("unknown", -2, True),
        ("pub", -1, True),
        ("layers.0", 0, False),
        ("layers.27", 27, False),
        ("invalid", -3, False),
    ],
)
def test_layer_sentinels_and_preload_contract(name, index, preload):
    assert parse_layer_index(name) == index
    assert should_preload(name) is preload


def test_backup_and_restore_descriptors_in_layer_order():
    handles = {
        "unknown": (0, 32, 50, 500, 5),
        "pub": (0, 64, 100, 1000, 10),
        "layers.0": (0, 128, 200, 2000, 20),
        "layers.1": (0, 256, 300, 3000, 30),
    }
    descriptors = [
        SharedWeightDesc("layers.1", [handles["layers.1"]]),
        SharedWeightDesc("unknown", [handles["unknown"]]),
        SharedWeightDesc("pub", [handles["pub"]]),
        SharedWeightDesc("layers.0", [handles["layers.0"]]),
    ]
    ops = FakeOps()

    with patch(
        "vllm_ascend.worker.copier._allocate_host_backup",
        side_effect=lambda size: FakeTensor(size),
    ):
        backups = backup_weight_descriptors(descriptors, ops)

    assert [descriptor.layer_name for descriptor in backups] == [
        "unknown",
        "pub",
        "layers.0",
        "layers.1",
    ]
    assert len(ops.imported) == len(ops.freed) == 4

    recovery_descriptors = [RecoverWeightDesc(name, [handle]) for name, handle in handles.items()]
    recoveries, imported = import_recovery_descriptors(recovery_descriptors, ops)
    events = {0: MagicMock(), 1: MagicMock()}
    restore_weight_descriptors(backups, recoveries, events, ops)

    assert len(imported) == 4
    assert len(ops.h2d) == 4
    events[0].set.assert_called_once_with()
    events[1].set.assert_called_once_with()


def test_backup_rejects_duplicate_allocation():
    handle = (0, 64, 100, 1000, 10)
    descriptors = [
        SharedWeightDesc("pub", [handle]),
        SharedWeightDesc("layers.0", [handle]),
    ]

    with (
        patch(
            "vllm_ascend.worker.copier._allocate_host_backup",
            side_effect=lambda size: FakeTensor(size),
        ),
        pytest.raises(RuntimeError, match="more than one layer"),
    ):
        backup_weight_descriptors(descriptors, FakeOps())


def test_backup_releases_import_when_copy_fails():
    handle = (0, 64, 100, 1000, 10)
    ops = FakeOps()
    ops.copy_device_to_host = MagicMock(side_effect=RuntimeError("D2H failed"))

    with (
        patch(
            "vllm_ascend.worker.copier._allocate_host_backup",
            side_effect=lambda size: FakeTensor(size),
        ),
        pytest.raises(RuntimeError, match="D2H failed"),
    ):
        backup_weight_descriptors([SharedWeightDesc("pub", [handle])], ops)

    assert len(ops.imported) == 1
    assert ops.freed == [ops.imported[0][1]]


def test_recovery_import_rolls_back_all_prior_handles():
    handles = [
        (0, 64, 100, 1000, 10),
        (0, 64, 200, 2000, 20),
        (0, 64, 300, 3000, 30),
    ]
    ops = FakeOps()
    original_import = ops.import_handle

    def fail_third_import(handle):
        if len(ops.imported) == 2:
            raise RuntimeError("import failed")
        return original_import(handle)

    ops.import_handle = fail_third_import
    descriptors = [RecoverWeightDesc("pub", handles)]

    with pytest.raises(RuntimeError, match="import failed"):
        import_recovery_descriptors(descriptors, ops)

    imported_locals = [local for _, local in ops.imported]
    assert ops.freed == list(reversed(imported_locals))


def test_restore_rejects_layout_mismatch_before_copying():
    backup = SharedWeightDesc(
        "pub",
        [(0, 64, 100, 1000, 10)],
        [FakeTensor(64)],
    )
    ops = FakeOps()

    with pytest.raises(RuntimeError, match="do not match"):
        restore_weight_descriptors(
            [backup],
            {"layers.0": [(0, 64, 200, 2000)]},
            {0: MagicMock()},
            ops,
        )

    assert ops.h2d == []


def test_shared_copier_main_runs_two_camem_suspend_resume_cycles():
    initial_handles = {
        "unknown": (0, 32, 50, 500, 5),
        "pub": (0, 64, 100, 1000, 10),
        "layers.0": (0, 128, 200, 2000, 20),
    }

    def recovery_handles(cycle):
        return {
            name: (*handle[:3], handle[3] + cycle * 10_000, handle[4] + cycle * 100)
            for name, handle in initial_handles.items()
        }

    first_recovery = recovery_handles(1)
    second_recovery = recovery_handles(2)
    desc_queue = FakeQueue(
        [
            *(SharedWeightDesc(name, [handle]) for name, handle in initial_handles.items()),
            MSG_DESC_SEND_FINISH,
        ]
    )
    recover_queue = FakeQueue(
        [
            *(RecoverWeightDesc(name, [handle]) for name, handle in first_recovery.items()),
            MSG_NPU_MEM_READY,
            *(RecoverWeightDesc(name, [handle]) for name, handle in second_recovery.items()),
            MSG_NPU_MEM_READY,
        ]
    )
    control = FakeControl(
        [
            MSG_SUSPEND_START,
            MSG_SUSPEND_START,
            MSG_RESUME_START,
            MSG_SUSPEND_START,
            MSG_RESUME_START,
            MSG_SHUTDOWN,
        ]
    )
    responses = FakeControl([])
    event = MagicMock()
    ops = FakeOps()
    fake_torch = SimpleNamespace(npu=SimpleNamespace(set_device=MagicMock()))

    with (
        patch("vllm_ascend.worker.copier.NativeCopierOps", return_value=ops),
        patch("vllm_ascend.worker.copier.torch", fake_torch),
        patch(
            "vllm_ascend.worker.copier._allocate_host_backup",
            side_effect=lambda size: FakeTensor(size),
        ),
    ):
        copier_main(
            desc_queue,
            recover_queue,
            _CopierControlChannel(control, responses, 0, CamemCopierMemoryBackend),
            1,
            0,
            {0: event},
        )

    assert responses.sent == [
        (MSG_COPIER_STARTED, 4321),
        MSG_COPIER_INIT_DONE,
        MSG_COPIER_SUSPEND_DONE,
        MSG_COPIER_SUSPEND_DONE,
        MSG_COPIER_SUSPEND_DONE,
    ]
    assert len(ops.d2h) == 3
    assert len(ops.h2d) == 6
    assert len(ops.freed) == 9
    assert event.clear.call_count == 3
    assert event.set.call_count == 2


def test_copier_controller_clears_stale_events_before_suspend_message():
    copier = CopierProcess.__new__(CopierProcess)
    copier._require_initialized = MagicMock()
    copier.layer_ready_events = {0: MagicMock(), 1: MagicMock()}
    copier.ctrl_queue = MagicMock()
    copier._receive_control = MagicMock()

    copier.suspend()

    copier.layer_ready_events[0].clear.assert_called_once_with()
    copier.layer_ready_events[1].clear.assert_called_once_with()
    copier.ctrl_queue.put.assert_called_once_with(MSG_SUSPEND_START)
    copier._receive_control.assert_called_once_with(MSG_COPIER_SUSPEND_DONE)
