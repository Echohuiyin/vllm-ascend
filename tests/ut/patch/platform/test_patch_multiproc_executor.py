# SPDX-License-Identifier: Apache-2.0

import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from vllm_ascend.patch.platform.patch_multiproc_executor import (
    AscendMultiprocExecutor,
    AscendWorkerProc,
)


class _StopInitialization(Exception):
    pass


def test_ascend_executor_initializes_upstream_failure_state_first():
    executor = object.__new__(AscendMultiprocExecutor)
    executor._init_failure_state = Mock()
    executor._get_parallel_sizes = Mock(side_effect=_StopInitialization)

    with (
        patch(
            "vllm_ascend.patch.platform.patch_multiproc_executor.weakref.finalize",
            return_value=Mock(),
        ),
        pytest.raises(_StopInitialization),
    ):
        executor._init_executor()

    executor._init_failure_state.assert_called_once_with()


@pytest.mark.parametrize(
    ("method", "requested_timeout", "pipe_enabled", "expected_timeout"),
    [
        ("suspend", None, True, 37.0),
        ("resume", 11.0, True, 11.0),
        ("sleep", None, False, None),
        ("execute_model", None, True, None),
        ("initialize_from_config", None, True, None),
    ],
)
def test_lifecycle_rpc_default_timeout_is_scoped_to_multiproc_pipe(
    method: str,
    requested_timeout: float | None,
    pipe_enabled: bool,
    expected_timeout: float | None,
):
    executor = object.__new__(AscendMultiprocExecutor)
    executor.vllm_config = Mock()
    base_executor = AscendMultiprocExecutor.__mro__[1]

    with (
        patch(
            "vllm_ascend.patch.platform.patch_multiproc_executor.is_multiproc_pipe_enabled",
            return_value=pipe_enabled,
        ),
        patch(
            "vllm_ascend.patch.platform.patch_multiproc_executor.ascend_envs.VLLM_ASCEND_MULTIPROC_PIPE_TIMEOUT",
            37.0,
        ),
        patch.object(base_executor, "collective_rpc", return_value=["ok"]) as collective_rpc,
    ):
        assert executor.collective_rpc(method, timeout=requested_timeout) == ["ok"]

    collective_rpc.assert_called_once_with(
        method,
        timeout=expected_timeout,
        args=(),
        kwargs=None,
        non_block=False,
        unique_reply_rank=None,
        kv_output_aggregator=None,
    )


@pytest.mark.parametrize("method", [lambda worker: worker, "apply_model", "reload_weights", "update_weights"])
def test_multiproc_pipe_rejects_runtime_model_mutation(method):
    executor = object.__new__(AscendMultiprocExecutor)
    executor.vllm_config = Mock()
    base_executor = AscendMultiprocExecutor.__mro__[1]

    with (
        patch(
            "vllm_ascend.patch.platform.patch_multiproc_executor.is_multiproc_pipe_enabled",
            return_value=True,
        ),
        patch.object(base_executor, "collective_rpc") as collective_rpc,
        pytest.raises(ValueError, match="rejects runtime model mutation"),
    ):
        executor.collective_rpc(method)

    collective_rpc.assert_not_called()


def _make_worker(context: Mock, enabled: bool, *, on_main_thread: bool = True):
    context.Pipe.side_effect = [(Mock(), Mock()), (Mock(), Mock())]
    main_thread = object()
    current_thread = main_thread if on_main_thread else object()
    with (
        patch("vllm_ascend.patch.platform.patch_multiproc_executor.get_mp_context", return_value=context),
        patch(
            "vllm_ascend.patch.platform.patch_multiproc_executor.is_multiproc_pipe_enabled",
            return_value=enabled,
        ),
        patch("vllm_ascend.patch.platform.patch_multiproc_executor.os.getpid", return_value=321),
        patch(
            "vllm_ascend.patch.platform.patch_multiproc_executor.threading.current_thread",
            return_value=current_thread,
        ),
        patch(
            "vllm_ascend.patch.platform.patch_multiproc_executor.threading.main_thread",
            return_value=main_thread,
        ),
    ):
        return AscendWorkerProc.make_worker_process(
            vllm_config=Mock(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:1",
            input_shm_handle=None,
            shared_worker_lock=Mock(),
        )


@pytest.mark.parametrize(("enabled", "on_main_thread"), [(True, True), (False, False)])
def test_worker_parent_guard_is_scoped_to_multiproc_pipe(enabled: bool, on_main_thread: bool):
    context = Mock()
    _make_worker(context, enabled, on_main_thread=on_main_thread)
    process = context.Process.return_value
    kwargs = context.Process.call_args.kwargs
    assert kwargs["target"] is AscendWorkerProc.worker_main
    assert kwargs["kwargs"].get("expected_parent_pid") == (321 if enabled else None)
    process.start.assert_called_once_with()


def test_guarded_worker_requires_engine_core_main_thread():
    with pytest.raises(RuntimeError, match="EngineCore main thread"):
        _make_worker(Mock(), True, on_main_thread=False)


def test_worker_arms_guard_before_platform_patch_and_initialization():
    calls = []
    base_worker = AscendWorkerProc.__mro__[1]
    fake_utils = ModuleType("vllm_ascend.utils")
    fake_utils.adapt_patch = lambda **kwargs: calls.append(("adapt_patch", kwargs))
    with (
        patch(
            "vllm_ascend.patch.platform.patch_multiproc_executor.arm_parent_death_signal",
            side_effect=lambda parent_pid, **kwargs: calls.append(("guard", parent_pid, kwargs)),
        ),
        patch.dict(sys.modules, {"vllm_ascend.utils": fake_utils}),
        patch.object(
            base_worker,
            "worker_main",
            side_effect=lambda *args, **kwargs: calls.append(("worker_main", args, kwargs)),
        ),
    ):
        AscendWorkerProc.worker_main("arg", expected_parent_pid=321, marker="value")

    assert calls == [
        ("guard", 321, {"process_name": "Ascend Worker"}),
        ("adapt_patch", {"is_global_patch": True}),
        ("worker_main", ("arg",), {"marker": "value"}),
    ]


_GUARDED_WORKER_SCRIPT = """
import os
import signal
import sys
import time

from vllm.utils.system_utils import arm_parent_death_signal

channel_fd = int(sys.argv[1])
mode = sys.argv[2]
expected_parent_pid = os.getpid()
if mode == "cascade":
    engine_pid = os.fork()
    if engine_pid:
        signal.pause()
    arm_parent_death_signal(expected_parent_pid, process_name="EngineCore")
    expected_parent_pid = os.getpid()

worker_pid = os.fork()
if worker_pid:
    if mode in ("stopped", "cascade"):
        signal.pause()
    os._exit(0)

os.write(channel_fd, f"{os.getpid()}\\n".encode())
if mode == "late_arm":
    while os.getppid() == expected_parent_pid:
        time.sleep(0.001)

arm_parent_death_signal(expected_parent_pid, process_name="Ascend Worker")
if mode in ("stopped", "cascade"):
    os.kill(os.getpid(), signal.SIGSTOP)
os._exit(97)
"""


def _recv_pid(channel: socket.socket) -> int:
    payload = b""
    while not payload.endswith(b"\n"):
        chunk = channel.recv(32)
        assert chunk, "Guarded Worker exited before reporting its PID"
        payload += chunk
    return int(payload)


def _wait_until_stopped(pid: int) -> None:
    deadline = time.monotonic() + 5
    status_path = Path(f"/proc/{pid}/status")
    while time.monotonic() < deadline:
        with suppress(FileNotFoundError):
            state = next(line for line in status_path.read_text().splitlines() if line.startswith("State:"))
            if "T (stopped)" in state:
                return
        time.sleep(0.01)
    pytest.fail(f"Guarded Worker {pid} did not enter SIGSTOP state")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-specific")
@pytest.mark.parametrize("mode", ["stopped", "late_arm", "cascade"])
def test_worker_parent_death_guard_is_kernel_enforced(mode: str):
    channel, child_channel = socket.socketpair()
    repo_root = Path(__file__).resolve().parents[4]
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(filter(None, (str(repo_root), os.environ.get("PYTHONPATH")))),
        "VLLM_PLUGINS": "",
    }
    parent = subprocess.Popen(
        [sys.executable, "-c", _GUARDED_WORKER_SCRIPT, str(child_channel.fileno()), mode],
        env=env,
        pass_fds=(child_channel.fileno(),),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child_channel.close()
    channel.settimeout(10)
    worker_pid = None
    worker_exited = False
    try:
        worker_pid = _recv_pid(channel)
        if mode in ("stopped", "cascade"):
            _wait_until_stopped(worker_pid)
            parent.kill()
            assert parent.wait(timeout=5) == -signal.SIGKILL
        else:
            # The child deliberately waits until it has been reparented before
            # arming prctl, exercising the post-prctl PPID race check.
            assert parent.wait(timeout=5) == 0

        # Both the intermediate parent and guarded Worker inherited this
        # socket. EOF therefore proves that the stopped/late-arming Worker was
        # killed too, instead of surviving as an orphan.
        assert channel.recv(1) == b""
        worker_exited = True
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if worker_pid is not None and not worker_exited:
            with suppress(ProcessLookupError):
                os.kill(worker_pid, signal.SIGKILL)
        child_channel.close()
        channel.close()
