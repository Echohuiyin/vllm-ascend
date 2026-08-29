# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

import socket
import struct
import threading
from unittest.mock import MagicMock

import pytest
from vllm.v1.worker.worker_base import WorkerRetryableError

import vllm_ascend.device_allocator.famem_client as rpc

_HEADER = struct.Struct("!IHHHHIII")
_MAPPING_PREFIX = struct.Struct("!QQI")
_MAPPING_EXTENT = struct.Struct("!IQQ")
_SESSION_ID = "a" * 32


def _frame(
    operation: int,
    payload: bytes = b"",
    *,
    request_id: int = 1,
    status: int = rpc.FAMEM_STATUS_OK,
    kind: int = rpc.FAMEM_MESSAGE_RESPONSE,
) -> bytes:
    fields = (
        rpc.FAMEM_PROTOCOL_MAGIC,
        rpc.FAMEM_PROTOCOL_VERSION,
        kind,
        operation,
        0,
        request_id,
        status,
        len(payload),
    )
    return _HEADER.pack(*fields) + payload


def _decode(frame: bytes, **expected):
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(frame)
        return rpc.receive_message(receiver, **expected)
    finally:
        sender.close()
        receiver.close()


def test_decodes_mapping_response():
    payload = _MAPPING_PREFIX.pack((1 << 30) + (2 << 20), 1, 2)
    payload += _MAPPING_EXTENT.pack(1, 1 << 30, 101)
    payload += _MAPPING_EXTENT.pack(2, 2 << 20, 102)
    response = _decode(_frame(2, payload), request_id=1, operation="ACQUIRE")
    assert response["mapping"] == ((1 << 30) + (2 << 20), 1, [1, 2], [1 << 30, 2 << 20], [101, 102])


@pytest.mark.parametrize(
    "payload",
    [
        b"x",
        _MAPPING_PREFIX.pack(2 << 20, 1, 0),
        _MAPPING_PREFIX.pack(2 << 20, 1, 1) + _MAPPING_EXTENT.pack(3, 2 << 20, 101),
        _MAPPING_PREFIX.pack(2 << 20, 1, 1) + _MAPPING_EXTENT.pack(2, 2 << 20, 0),
        _MAPPING_PREFIX.pack(4 << 20, 1, 1) + _MAPPING_EXTENT.pack(2, 2 << 20, 101),
    ],
)
def test_rejects_invalid_mapping_response(payload):
    with pytest.raises(rpc.FamemProtocolError, match="mapping response"):
        _decode(_frame(2, payload))


def test_decodes_busy_response():
    response = _decode(_frame(4, b"lease is busy", status=rpc.FAMEM_STATUS_BUSY))
    assert (response["ok"], response["error_type"], response["error"]) == (False, "FamemBusyError", "lease is busy")
    assert issubclass(rpc.FamemBusyError, WorkerRetryableError)


@pytest.mark.parametrize(
    "frame,match",
    [
        (_frame(2, kind=rpc.FAMEM_MESSAGE_REQUEST), "invalid header"),
        (_frame(2, request_id=2), "does not match"),
        (_frame(2), "does not match"),
    ],
)
def test_rejects_mismatched_response(frame, match):
    with pytest.raises(rpc.FamemProtocolError, match=match):
        _decode(frame, request_id=1, operation="WAKE")


def test_rejects_truncated_response():
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(_frame(1, b"x" * 32)[:-1])
        sender.shutdown(socket.SHUT_WR)
        with pytest.raises(rpc.FamemProtocolError, match="closed during"):
            rpc.receive_message(receiver)
    finally:
        sender.close()
        receiver.close()


def _client() -> rpc.FamemClient:
    client = object.__new__(rpc.FamemClient)
    client.closed = client.poisoned = False
    client._request_id = 0
    client.session_id = _SESSION_ID
    client._connection = object()
    return client


@pytest.mark.parametrize(
    "reply,error_type,poisoned",
    [
        ({"ok": False, "error_type": "FamemBusyError", "error": "busy"}, rpc.FamemBusyError, False),
        (rpc.FamemProtocolError("bad response"), rpc.FamemTransportError, True),
    ],
)
def test_request_failure_poison_policy(monkeypatch, reply, error_type, poisoned):
    client = _client()
    monkeypatch.setattr(rpc, "send_message", lambda *_: None)

    def receive(*_args, **_kwargs):
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(rpc, "receive_message", receive)
    with pytest.raises(error_type):
        client._request("WAKE", generation=1)
    assert client.poisoned is poisoned


def test_invalid_mapping_releases_generation_and_poisons(monkeypatch):
    client = _client()
    operations = []
    replies = iter([rpc._MappingError("bad mapping", 7), {"ok": True}])

    def receive(*_args, **_kwargs):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(rpc, "send_message", lambda _connection, message: operations.append(message["op"]))
    monkeypatch.setattr(rpc, "receive_message", receive)

    with pytest.raises(rpc.FamemProtocolError, match="bad mapping"):
        client._request("WAKE", generation=1)
    assert client.poisoned
    assert operations == ["WAKE", "RELEASE"]


@pytest.mark.parametrize(
    "generation,handle,accepted",
    [(5, 700, True), (3, 700, False), (5, 701, False)],
)
def test_wake_validates_global_epoch_and_resident_pool(monkeypatch, generation, handle, accepted):
    client = _client()
    client._lock = threading.RLock()
    client.device = 0
    client.native = MagicMock()
    client.capacity, client.generation, client.base_address = 2 << 20, 3, 0x20000000
    client.extent_page_types = [rpc.FamemPageType.HUGE_2M]
    client.extent_sizes = [2 << 20]
    client.shareable_handles = [700]
    client.active = False
    release = MagicMock()
    monkeypatch.setattr(client, "_best_effort_request", release)
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: {
            "mapping": (2 << 20, generation, [rpc.FamemPageType.HUGE_2M], [2 << 20], [handle])
        },
    )

    if accepted:
        client.wake()
        assert client.active and client.generation == generation
        client.native.worker_remap.assert_called_once()
        release.assert_not_called()
        return

    with pytest.raises(rpc.FamemProtocolError, match="changed the arena"):
        client.wake()
    assert client.poisoned
    release.assert_called_once_with("RELEASE", generation=generation)
    client.native.worker_remap.assert_not_called()
