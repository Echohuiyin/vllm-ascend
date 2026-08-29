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

import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

import pytest

from vllm_ascend.device_allocator import hbm_server_launcher
from vllm_ascend.device_allocator.famem_client import (
    FAMEM_PROTOCOL_VERSION,
    receive_message,
    send_message,
)

_DEVICE_UUID = "0123456789abcdef0123456789abcdef"
_SESSION_ID = "fedcba9876543210fedcba9876543210"
_SECOND_SESSION_ID = "abcdef0123456789abcdef0123456789"
_POOL_SIZE_GIB = 2
_ARENA_SIZE = _POOL_SIZE_GIB << 30
_STUB_SHAREABLE_OFFSET = 1000


def _server_command(binary: Path, socket_dir: Path, *extra: str) -> list[str]:
    return [
        str(binary),
        "--device",
        "0",
        "--size-gib",
        str(_POOL_SIZE_GIB),
        "--socket-dir",
        str(socket_dir),
        *extra,
    ]


def test_console_launcher_execs_native_server(monkeypatch):
    calls = []
    server = Path("/installed/vllm_ascend_hbm_server")
    monkeypatch.setattr(hbm_server_launcher, "native_server_path", lambda: server)
    monkeypatch.setattr(hbm_server_launcher.os, "execv", lambda path, args: calls.append((path, args)))

    assert hbm_server_launcher.main(["--device", "2", "--size-gib", "48"]) == 1
    assert calls == [(server, [str(server), "--device", "2", "--size-gib", "48"])]


@pytest.fixture()
def native_server_binary(tmp_path: Path) -> Path:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("A C compiler is required for the native HBM server test.")
    repository = Path(__file__).resolve().parents[3]
    library = tmp_path / "libvllm_ascend_famem.so"
    binary = tmp_path / "vllm_ascend_hbm_server"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fPIC",
            "-shared",
            "-I",
            str(repository / "csrc"),
            str(Path(__file__).with_name("famem_server_stub.c")),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-pthread",
            "-I",
            str(repository / "csrc"),
            str(repository / "csrc/famem_hbm_server.c"),
            "-L",
            str(tmp_path),
            "-lvllm_ascend_famem",
            "-Wl,-rpath,$ORIGIN",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.fixture()
def copier_process():
    process = subprocess.Popen(["sleep", "30"])
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def _request(connection: socket.socket, request_id: int, operation: str, **fields):
    send_message(
        connection,
        {
            "version": FAMEM_PROTOCOL_VERSION,
            "request_id": request_id,
            "op": operation,
            **fields,
        },
    )
    response = receive_message(connection)
    assert response is not None
    assert response["request_id"] == request_id
    assert response["op"] == operation
    assert response["ok"], response
    return response


def _trace_events(path: Path) -> list[tuple[str, int]]:
    if not path.exists():
        return []
    return [(event, int(handle)) for event, handle in (line.split() for line in path.read_text().splitlines())]


def test_server_preallocates_before_publish_and_keeps_original_handles_resident(
    native_server_binary: Path,
    copier_process,
):
    with tempfile.TemporaryDirectory(prefix="famem-sock-", dir="/tmp") as directory:
        socket_dir = Path(directory)
        socket_path = socket_dir / f"{_DEVICE_UUID}.sock"
        trace_path = socket_dir / "native.trace"
        process = subprocess.Popen(
            _server_command(native_server_binary, socket_dir),
            env={**os.environ, "FAMEM_STUB_TRACE": str(trace_path)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        second_connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            deadline = time.monotonic() + 5
            startup_events = _trace_events(trace_path)
            while not any(event == "LISTEN" for event, _ in startup_events) and time.monotonic() < deadline:
                if process.poll() is not None:
                    _, stderr = process.communicate()
                    pytest.fail(f"Native HBM server exited during startup:\n{stderr}")
                time.sleep(0.01)
                startup_events = _trace_events(trace_path)
            assert socket_path.exists()
            assert [event for event, _ in startup_events] == [
                "ALLOCATE_EXPORT",
                "ALLOCATE_EXPORT",
                "LISTEN",
            ]
            startup_handles = [handle for event, handle in startup_events if event == "ALLOCATE_EXPORT"]
            connection.connect(str(socket_path))

            hello = _request(
                connection,
                1,
                "HELLO",
                device_uuid=_DEVICE_UUID,
                session_id=_SESSION_ID,
                bare_tgid=os.getpid(),
                copier_bare_tgid=copier_process.pid,
            )
            assert hello["device_uuid"] == _DEVICE_UUID

            send_message(
                connection,
                {
                    "version": FAMEM_PROTOCOL_VERSION,
                    "request_id": 2,
                    "op": "ACQUIRE",
                    "session_id": _SESSION_ID,
                    "size": _ARENA_SIZE // 2,
                },
            )
            mismatch = receive_message(connection)
            assert mismatch is not None and not mismatch["ok"]
            assert mismatch["error_type"] == "FamemProtocolError"
            assert "configured pool" in mismatch["error"]
            assert all(event != "AUTHORIZE" for event, _ in _trace_events(trace_path))

            mapping = _request(
                connection,
                3,
                "ACQUIRE",
                session_id=_SESSION_ID,
                size=_ARENA_SIZE,
            )
            assert mapping["mapping"][:4] == (_ARENA_SIZE, 1, [1, 2], [1 << 30, 1 << 30])
            pool_handles = mapping["mapping"][4]
            assert pool_handles == startup_handles
            assert [event for event, _ in _trace_events(trace_path)].count("ALLOCATE_EXPORT") == 2

            second_connection.connect(str(socket_path))
            _request(
                second_connection,
                1,
                "HELLO",
                device_uuid=_DEVICE_UUID,
                session_id=_SECOND_SESSION_ID,
                bare_tgid=os.getpid(),
                copier_bare_tgid=copier_process.pid,
            )
            send_message(
                second_connection,
                {
                    "version": FAMEM_PROTOCOL_VERSION,
                    "request_id": 2,
                    "op": "ACQUIRE",
                    "session_id": _SECOND_SESSION_ID,
                    "size": _ARENA_SIZE,
                },
            )
            busy = receive_message(second_connection)
            assert busy is not None
            assert busy["ok"] is False
            assert busy["error_type"] == "FamemBusyError"

            _request(connection, 4, "SLEEP", session_id=_SESSION_ID, generation=1)
            sleeping_events = _trace_events(trace_path)
            assert all(event != "FREE" for event, _ in sleeping_events)

            # Keep model 1's sleeping lease connected while model 2 acquires
            # and releases the same NPU, then resume model 1.
            second_mapping = _request(
                second_connection,
                3,
                "ACQUIRE",
                session_id=_SECOND_SESSION_ID,
                size=_ARENA_SIZE,
            )
            assert second_mapping["mapping"][1] == 2
            assert second_mapping["mapping"][4] == pool_handles
            _request(second_connection, 4, "SLEEP", session_id=_SECOND_SESSION_ID, generation=2)

            remapping = _request(connection, 4, "WAKE", session_id=_SESSION_ID, generation=1)
            assert remapping["mapping"][1:3] == (3, [1, 2])
            assert remapping["mapping"][4] == pool_handles
            wake_events = _trace_events(trace_path)
            assert [event for event, _ in wake_events].count("ALLOCATE_EXPORT") == 2
            assert all(event != "FREE" for event, _ in wake_events)
            _request(connection, 5, "RELEASE", session_id=_SESSION_ID, generation=3)
            _request(second_connection, 5, "RELEASE", session_id=_SECOND_SESSION_ID, generation=2)
            running_events = _trace_events(trace_path)
            assert [event for event, _ in running_events].count("ALLOCATE_EXPORT") == 2
            assert [event for event, _ in running_events].count("AUTHORIZE") == 6
            assert all(event != "FREE" for event, _ in running_events)
        finally:
            connection.close()
            second_connection.close()
            if process.poll() is None:
                process.terminate()
            stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
        final_events = _trace_events(trace_path)
        expected_shutdown = [
            *(
                ("FREE", shareable_handle - _STUB_SHAREABLE_OFFSET)
                for shareable_handle in reversed(startup_handles)
            ),
            ("FINALIZE", 0),
        ]
        assert final_events == [*running_events, *expected_shutdown]


def test_native_server_preallocation_failure_publishes_no_socket(native_server_binary: Path):
    with tempfile.TemporaryDirectory(prefix="famem-sock-", dir="/tmp") as directory:
        socket_dir = Path(directory)
        socket_path = socket_dir / f"{_DEVICE_UUID}.sock"
        trace_path = socket_dir / "native.trace"
        environment = os.environ.copy()
        environment.update(
            FAMEM_STUB_FAIL_ALLOC_AT="2",
            FAMEM_STUB_FAIL_FREE_AT="1",
            FAMEM_STUB_TRACE=str(trace_path),
        )
        process = subprocess.Popen(
            _server_command(native_server_binary, socket_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode != 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
        assert not socket_path.exists()
        assert [event for event, _ in _trace_events(trace_path)] == ["ALLOCATE_EXPORT", "FREE", "FINALIZE"]


def test_native_server_rejects_truncated_hello_and_keeps_serving(native_server_binary: Path, copier_process):
    with tempfile.TemporaryDirectory(prefix="famem-sock-", dir="/tmp") as directory:
        socket_dir = Path(directory)
        socket_path = socket_dir / f"{_DEVICE_UUID}.sock"
        process = subprocess.Popen(
            _server_command(native_server_binary, socket_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        valid_connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            deadline = time.monotonic() + 5
            while not socket_path.exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    _, stderr = process.communicate()
                    pytest.fail(f"Native HBM server exited during startup:\n{stderr}")
                time.sleep(0.01)

            truncated_connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            truncated_connection.connect(str(socket_path))
            truncated_connection.sendall(b"FAME\x00\x04\x00\x01")
            truncated_connection.close()

            invalid_identities = (
                (0, copier_process.pid, "invalid Famem HELLO identity"),
                (os.getpid(), 0, "invalid Famem HELLO identity"),
                (os.getpid(), (1 << 31) - 1, "Copier is not a live child of the Worker"),
            )
            for bare_tgid, copier_bare_tgid, error in invalid_identities:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as invalid_connection:
                    invalid_connection.connect(str(socket_path))
                    send_message(
                        invalid_connection,
                        {
                            "version": FAMEM_PROTOCOL_VERSION,
                            "request_id": 1,
                            "op": "HELLO",
                            "device_uuid": _DEVICE_UUID,
                            "session_id": _SECOND_SESSION_ID,
                            "bare_tgid": bare_tgid,
                            "copier_bare_tgid": copier_bare_tgid,
                        },
                    )
                    rejected = receive_message(invalid_connection)
                    assert rejected is not None and not rejected["ok"]
                    assert rejected["error_type"] == "FamemProtocolError"
                    assert rejected["error"] == error

            valid_connection.connect(str(socket_path))
            _request(
                valid_connection,
                1,
                "HELLO",
                device_uuid=_DEVICE_UUID,
                session_id=_SESSION_ID,
                bare_tgid=os.getpid(),
                copier_bare_tgid=copier_process.pid,
            )
        finally:
            valid_connection.close()
            if process.poll() is None:
                process.terminate()
            stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"


def test_native_server_socket_cleanup_is_inode_scoped(native_server_binary: Path):
    with (
        tempfile.TemporaryDirectory(prefix="famem-sock-", dir="/tmp") as directory,
        tempfile.TemporaryFile(mode="w+") as server_log,
    ):
        socket_dir = Path(directory)
        socket_path = socket_dir / f"{_DEVICE_UUID}.sock"
        flock_ready = socket_dir / "flock.ready"
        listen_barrier = socket_dir / "listen.barrier"
        trace_path = socket_dir / "native.trace"
        first = subprocess.Popen(
            _server_command(native_server_binary, socket_dir),
            env={
                **os.environ,
                "FAMEM_STUB_LISTEN_BARRIER": str(listen_barrier),
                "FAMEM_STUB_TRACE": str(trace_path),
            },
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        second: subprocess.Popen[str] | None = None
        try:
            deadline = time.monotonic() + 5
            while not listen_barrier.exists() and time.monotonic() < deadline:
                assert first.poll() is None
                time.sleep(0.01)
            assert listen_barrier.exists()
            second = subprocess.Popen(
                _server_command(native_server_binary, socket_dir),
                env={
                    **os.environ,
                    "FAMEM_STUB_FLOCK_READY": str(flock_ready),
                    "FAMEM_STUB_TRACE": str(trace_path),
                },
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            deadline = time.monotonic() + 5
            while not flock_ready.exists() and time.monotonic() < deadline:
                assert second.poll() is None
                time.sleep(0.01)
            assert flock_ready.exists()
            listen_barrier.unlink()
            assert second.wait(timeout=5) != 0
            assert [event for event, _ in _trace_events(trace_path)].count("ALLOCATE_EXPORT") == 2
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.connect(str(socket_path))
            first.terminate()
            assert first.wait(timeout=5) == 0
            assert not socket_path.exists()
        finally:
            flock_ready.unlink(missing_ok=True)
            listen_barrier.unlink(missing_ok=True)
            for process in (second, first):
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            socket_path.unlink(missing_ok=True)

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as owner:
            owner.bind(str(socket_path))
            owner.listen(1)
            contender = subprocess.run(
                _server_command(native_server_binary, socket_dir),
                env={**os.environ, "FAMEM_STUB_CONNECT_ENOENT": "1"},
                capture_output=True,
                text=True,
                timeout=5,
            )
            assert contender.returncode != 0
            assert "cannot remove stale socket" not in contender.stderr
            assert "cannot create Famem listener" in contender.stderr
            assert socket_path.exists()
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.connect(str(socket_path))
        socket_path.unlink()

        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(socket_path))
        stale.close()

        process = subprocess.Popen(
            _server_command(native_server_binary, socket_dir),
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        replacement: socket.socket | None = None
        try:
            deadline = time.monotonic() + 5
            server_ready = False
            while process.poll() is None and time.monotonic() < deadline:
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    probe.connect(str(socket_path))
                except OSError:
                    time.sleep(0.01)
                    continue
                finally:
                    probe.close()
                server_ready = True
                break
            assert server_ready and process.poll() is None

            os.unlink(socket_path)
            replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            replacement.bind(str(socket_path))
            replacement.listen(1)
            process.terminate()
            process.wait(timeout=10)
            assert socket_path.exists()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            if replacement is not None:
                replacement.close()
            if socket_path.exists():
                socket_path.unlink()
        server_log.seek(0)
        assert process.returncode == 0, f"server log:\n{server_log.read()}"


def test_native_server_drains_surviving_copier_before_next_activation(
    native_server_binary: Path, tmp_path: Path, copier_process
):
    socket_dir = Path(tempfile.mkdtemp(prefix="famem-sock-", dir="/tmp"))
    socket_path = socket_dir / f"{_DEVICE_UUID}.sock"
    ready_path = tmp_path / "worker-ready"
    server = subprocess.Popen(
        _server_command(native_server_binary, socket_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    copier_pid = 0
    worker: subprocess.Popen[str] | None = None
    second_connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        deadline = time.monotonic() + 15
        while not socket_path.exists() and time.monotonic() < deadline:
            if server.poll() is not None:
                _, stderr = server.communicate()
                pytest.fail(f"Native HBM server exited during startup:\n{stderr}")
            time.sleep(0.01)

        worker_code = """
import os
import socket
import struct
import subprocess
import sys

connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
connection.connect(sys.argv[1])
copier = subprocess.Popen(["sleep", "30"])
header = struct.Struct("!IHHHHIII")
def exchange(operation, request_id, payload):
    connection.sendall(header.pack(0x46414D45, 5, 1, operation, 0, request_id, 0, len(payload)) + payload)
    response = connection.recv(header.size, socket.MSG_WAITALL)
    *_, status, size = header.unpack(response)
    assert not status and len(connection.recv(size, socket.MSG_WAITALL)) == size
exchange(1, 1, struct.pack("!32s32sII", sys.argv[2].encode(), sys.argv[3].encode(), os.getpid(), copier.pid))
exchange(2, 2, struct.pack("!32sQ", sys.argv[3].encode(), int(sys.argv[4])))
ready_path = sys.argv[5]
with open(f"{ready_path}.tmp", "w", encoding="ascii") as ready:
    ready.write(str(copier.pid))
os.replace(f"{ready_path}.tmp", ready_path)
os._exit(0)
"""
        worker = subprocess.Popen(
            [
                sys.executable,
                "-c",
                worker_code,
                str(socket_path),
                _DEVICE_UUID,
                _SESSION_ID,
                str(_ARENA_SIZE),
                str(ready_path),
            ],
            text=True,
        )
        deadline = time.monotonic() + 15
        while not ready_path.exists() and time.monotonic() < deadline:
            assert worker.poll() is None, f"Worker exited early with code {worker.returncode}"
            time.sleep(0.01)
        assert ready_path.exists()
        copier_pid = int(ready_path.read_text(encoding="ascii"))
        assert worker.wait(timeout=5) == 0

        second_connection.connect(str(socket_path))
        _request(
            second_connection,
            1,
            "HELLO",
            device_uuid=_DEVICE_UUID,
            session_id=_SECOND_SESSION_ID,
            bare_tgid=os.getpid(),
            copier_bare_tgid=copier_process.pid,
        )
        send_message(
            second_connection,
            {
                "version": FAMEM_PROTOCOL_VERSION,
                "request_id": 2,
                "op": "ACQUIRE",
                "session_id": _SECOND_SESSION_ID,
                "size": _ARENA_SIZE,
            },
        )
        busy = receive_message(second_connection)
        assert busy is not None
        assert busy["ok"] is False
        assert busy["error_type"] == "FamemBusyError"

        os.kill(copier_pid, signal.SIGTERM)
        copier_pid = 0
        mapping = None
        request_id = 3
        deadline = time.monotonic() + 5
        while mapping is None and time.monotonic() < deadline:
            send_message(
                second_connection,
                {
                    "version": FAMEM_PROTOCOL_VERSION,
                    "request_id": request_id,
                    "op": "ACQUIRE",
                    "session_id": _SECOND_SESSION_ID,
                    "size": _ARENA_SIZE,
                },
            )
            response = receive_message(second_connection)
            assert response is not None
            if response["ok"]:
                mapping = response
                break
            assert response["error_type"] == "FamemBusyError"
            request_id += 1
            time.sleep(0.02)
        assert mapping is not None
        _request(
            second_connection,
            request_id + 1,
            "RELEASE",
            session_id=_SECOND_SESSION_ID,
            generation=mapping["mapping"][1],
        )
    finally:
        second_connection.close()
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=5)
        if copier_pid:
            with suppress(ProcessLookupError):
                os.kill(copier_pid, signal.SIGKILL)
        if server.poll() is None:
            server.terminate()
        stdout, stderr = server.communicate(timeout=5)
        socket_dir.rmdir()
    assert server.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
