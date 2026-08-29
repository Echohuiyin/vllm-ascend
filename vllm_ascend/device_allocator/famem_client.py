# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from __future__ import annotations

import os
import socket
import stat
import struct
import threading
import time
import uuid
from contextlib import suppress
from typing import Any

from vllm.v1.worker.worker_base import WorkerRetryableError

from vllm_ascend.device_allocator.famem_native import FamemNativeLibrary, FamemPageType

FAMEM_PROTOCOL_MAGIC = 0x46414D45  # "FAME"
FAMEM_PROTOCOL_VERSION = 5
MAX_MESSAGE_BYTES = 4096
FAMEM_MESSAGE_REQUEST, FAMEM_MESSAGE_RESPONSE = 1, 2
FAMEM_STATUS_OK, FAMEM_STATUS_PROTOCOL, FAMEM_STATUS_BUSY, FAMEM_STATUS_PERMISSION, FAMEM_STATUS_INTERNAL = range(5)

_OP_CODES = {"HELLO": 1, "ACQUIRE": 2, "SLEEP": 3, "WAKE": 4, "RELEASE": 5}
_OP_NAMES = {code: name for name, code in _OP_CODES.items()}
_HEADER = struct.Struct("!IHHHHIII")
_HELLO_REQUEST = struct.Struct("!32s32sII")
_SESSION_U64_REQUEST = struct.Struct("!32sQ")
_MAPPING_PREFIX = struct.Struct("!QQI")
_MAPPING_EXTENT = struct.Struct("!IQQ")
_GENERATION_RESPONSE = struct.Struct("!Q")
_ERROR_TYPES = (None, "FamemProtocolError", "FamemBusyError", "PermissionError", "RuntimeError")
_REQUEST_HEADER = (FAMEM_PROTOCOL_MAGIC, FAMEM_PROTOCOL_VERSION, FAMEM_MESSAGE_REQUEST)
_RESPONSE_HEADER = (FAMEM_PROTOCOL_MAGIC, FAMEM_PROTOCOL_VERSION, FAMEM_MESSAGE_RESPONSE, 0)


class FamemError(RuntimeError): ...


class FamemProtocolError(FamemError): ...


class FamemTransportError(FamemError): ...


class FamemBusyError(WorkerRetryableError, FamemError): ...


class _MappingError(FamemProtocolError):
    def __init__(self, message: str, generation: int) -> None:
        super().__init__(message)
        self.generation = generation


_Mapping = tuple[int, int, list[FamemPageType], list[int], list[int]]


def _hex(value: Any, field: str) -> str:
    try:
        value = value.decode("ascii") if isinstance(value, bytes) else value
    except UnicodeDecodeError as error:
        raise FamemProtocolError(f"Famem {field} is not ASCII.") from error
    if not isinstance(value, str) or len(value) != 32 or set(value) - set("0123456789abcdef"):
        raise FamemProtocolError(f"Famem {field} must be 32 lowercase hexadecimal characters.")
    return value


def send_message(connection: socket.socket, message: dict[str, Any]) -> None:
    """Encode one client request for the native Famem server."""
    operation = message["op"]
    if operation == "HELLO":
        payload = _HELLO_REQUEST.pack(
            message["device_uuid"].encode(),
            message["session_id"].encode(),
            message["bare_tgid"],
            message["copier_bare_tgid"],
        )
    else:
        field = "size" if operation == "ACQUIRE" else "generation"
        payload = _SESSION_U64_REQUEST.pack(message["session_id"].encode(), message[field])
    header = _HEADER.pack(
        *_REQUEST_HEADER, _OP_CODES[operation], 0, message["request_id"], FAMEM_STATUS_OK, len(payload)
    )
    connection.sendall(header + payload)


def _require_size(operation: str, payload: bytes, expected: int) -> None:
    if len(payload) != expected:
        raise FamemProtocolError(f"Famem {operation} payload size is {len(payload)} bytes; expected {expected}.")


def _decode_mapping(payload: bytes) -> _Mapping:
    if len(payload) < _MAPPING_PREFIX.size:
        raise FamemProtocolError("Famem mapping response is truncated.")
    size, generation, count = _MAPPING_PREFIX.unpack_from(payload)
    if count not in (1, 2) or len(payload) != _MAPPING_PREFIX.size + count * _MAPPING_EXTENT.size:
        raise _MappingError("Famem mapping response contains an invalid extent count.", generation)
    extents = [
        _MAPPING_EXTENT.unpack_from(payload, _MAPPING_PREFIX.size + index * _MAPPING_EXTENT.size)
        for index in range(count)
    ]
    try:
        page_types = [FamemPageType(page_type) for page_type, _, _ in extents]
    except ValueError as error:
        raise _MappingError("Famem mapping response contains an unknown page type.", generation) from error
    extent_sizes = [extent_size for _, extent_size, _ in extents]
    handles = [handle for _, _, handle in extents]
    if (
        not size
        or not generation
        or page_types != sorted(set(page_types))
        or any(
            not extent_size or extent_size % page_type.granularity_bytes or not handle
            for page_type, (_, extent_size, handle) in zip(page_types, extents, strict=True)
        )
        or sum(extent_sizes) != size
    ):
        raise _MappingError("Famem mapping response contains an invalid layout.", generation)
    return size, generation, page_types, extent_sizes, handles


def _receive_exact(connection: socket.socket, size: int) -> bytes | None:
    if size == 0:
        return b""
    payload = connection.recv(size, socket.MSG_WAITALL)
    if not payload:
        return None
    if len(payload) != size:
        raise FamemProtocolError("Famem RPC peer closed during a message.")
    return payload


def receive_message(
    connection: socket.socket,
    *,
    request_id: int | None = None,
    operation: str | None = None,
) -> dict[str, Any] | None:
    """Decode and validate one native-server response."""
    raw_header = _receive_exact(connection, _HEADER.size)
    if raw_header is None:
        return None
    magic, version, kind, operation_code, flags, response_id, status, payload_size = _HEADER.unpack(raw_header)
    response_operation = _OP_NAMES.get(operation_code)
    if (
        (magic, version, kind, flags) != _RESPONSE_HEADER
        or not response_operation
        or response_id <= 0
        or payload_size > MAX_MESSAGE_BYTES
    ):
        raise FamemProtocolError("Famem RPC response has an invalid header.")
    if request_id not in (None, response_id) or operation not in (None, response_operation):
        raise FamemProtocolError("Famem response does not match its request.")
    payload = _receive_exact(connection, payload_size)
    if payload is None:
        raise FamemProtocolError("Famem RPC peer closed during a message.")

    response: dict[str, Any] = {
        "version": version,
        "request_id": response_id,
        "op": response_operation,
        "ok": not status,
    }
    if status != FAMEM_STATUS_OK:
        if status >= len(_ERROR_TYPES):
            raise FamemProtocolError(f"Famem response has an unknown status {status}.")
        error_type = _ERROR_TYPES[status]
        try:
            error = payload.decode("utf-8")
        except UnicodeDecodeError as decode_error:
            raise FamemProtocolError("Famem error response is not valid UTF-8.") from decode_error
        response.update(error_type=error_type, error=error)
        return response

    if response_operation == "HELLO":
        _require_size(response_operation, payload, 32)
        response["device_uuid"] = _hex(payload, "device_uuid")
    elif response_operation in ("ACQUIRE", "WAKE"):
        response["mapping"] = _decode_mapping(payload)
    elif response_operation == "SLEEP":
        _require_size(response_operation, payload, _GENERATION_RESPONSE.size)
        (generation,) = _GENERATION_RESPONSE.unpack(payload)
        if generation == 0:
            raise FamemProtocolError("Famem SLEEP returned an invalid generation.")
        response["generation"] = generation
    else:
        _require_size(response_operation, payload, 0)
    return response


def _socket_path(socket_dir: str, device_uuid: str) -> str:
    _hex(device_uuid, "device_uuid")
    path = os.path.join(socket_dir, f"{device_uuid}.sock")
    if len(os.fsencode(path)) > 103:
        raise ValueError(f"Famem Unix socket path is too long: {path!r}.")
    return path


class FamemClient:
    """Control-plane client plus transactional worker mapping lifecycle."""

    def __init__(
        self,
        device: int,
        socket_dir: str,
        *,
        native: FamemNativeLibrary | Any | None = None,
        timeout: float = 120.0,
        copier_bare_tgid: int = 0,
    ) -> None:
        if timeout <= 0 or type(copier_bare_tgid) is not int or copier_bare_tgid < 0:
            raise ValueError("Famem requires a positive timeout and non-negative integer Copier bare TGID.")
        self.device = device
        self.native = native or FamemNativeLibrary()
        self.device_uuid = self.native.device_uuid(device)
        self.socket_path = _socket_path(socket_dir, self.device_uuid)
        self.session_id = uuid.uuid4().hex
        self.timeout = timeout
        self.generation = self.capacity = self.base_address = 0
        self.extent_page_types: list[FamemPageType] = []
        self.extent_sizes: list[int] = []
        self.shareable_handles: list[int] = []
        self.active = self.poisoned = self.closed = False
        self._server_released = self._cleanup_complete = False
        self._request_id = 0
        self._lock = threading.RLock()
        self._connection = self._connect()
        try:
            response = self._request(
                "HELLO",
                include_session=False,
                device_uuid=self.device_uuid,
                session_id=self.session_id,
                bare_tgid=self.native.bare_tgid(),
                copier_bare_tgid=copier_bare_tgid,
            )
            if response.get("device_uuid") != self.device_uuid:
                raise FamemProtocolError("Famem server returned a different NPU UUID.")
        except Exception:
            self._connection.close()
            self.closed = True
            raise

    def _connect(self) -> socket.socket:
        try:
            socket_stat = os.stat(self.socket_path, follow_symlinks=False)
        except FileNotFoundError as error:
            raise FamemTransportError(
                f"Famem HBM server socket {self.socket_path!r} does not exist; start the server first."
            ) from error
        if not stat.S_ISSOCK(socket_stat.st_mode) or socket_stat.st_uid != os.getuid():
            raise FamemTransportError("Famem server path is not a same-UID Unix socket.")
        deadline = time.monotonic() + self.timeout
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(min(1.0, self.timeout))
            try:
                connection.connect(self.socket_path)
                connection.settimeout(self.timeout)
                return connection
            except OSError as error:
                last_error = error
                connection.close()
                time.sleep(0.02)
        raise FamemTransportError(f"Unable to connect to Famem server at {self.socket_path!r}.") from last_error

    def acquire(self, size: int) -> int:
        with self._lock:
            self._require_healthy()
            if self.capacity:
                raise RuntimeError("Famem client already acquired an arena.")
            response = self._request("ACQUIRE", size=size)
            fields: _Mapping = response["mapping"]
            capacity, generation, page_types, sizes, handles = fields
            if capacity != size:
                self._best_effort_request("RELEASE", generation=generation)
                self.poisoned = True
                raise FamemProtocolError("Famem ACQUIRE returned an unexpected size.")
            try:
                base_address = self.native.worker_prepare(self.device, capacity, page_types, sizes, handles)
            except Exception:
                self._release_local_mapping(fields)
                self._best_effort_request("RELEASE", generation=generation)
                raise
            self._remember_mapping(fields, base_address)
            self.active = True
            return base_address

    def _remember_mapping(self, fields: _Mapping, base_address: int = 0) -> None:
        self.capacity, self.generation, self.extent_page_types, self.extent_sizes, self.shareable_handles = fields
        self.base_address = base_address

    def _release_local_mapping(self, fields: _Mapping | None = None) -> None:
        try:
            self.native.worker_release(self.device)
        except Exception as error:
            if fields is not None:
                self._remember_mapping(fields)
            self.poisoned = True
            raise RuntimeError("Famem imported-handle cleanup failed; restart the worker.") from error

    def sleep(self) -> None:
        with self._lock:
            self._require_healthy()
            if not self.active:
                raise RuntimeError("Famem arena is not active.")
            try:
                self.native.worker_unmap(self.device)
                self.active = False
                self._request("SLEEP", generation=self.generation)
            except Exception:
                self.poisoned = True
                raise

    def wake(self) -> None:
        with self._lock:
            self._require_healthy()
            if not self.capacity or self.active:
                raise RuntimeError("Famem arena is not sleeping.")
            response = self._request("WAKE", generation=self.generation)
            fields: _Mapping = response["mapping"]
            capacity, generation, page_types, sizes, handles = fields
            if generation <= self.generation or (capacity, page_types, sizes, handles) != (
                self.capacity,
                self.extent_page_types,
                self.extent_sizes,
                self.shareable_handles,
            ):
                self._best_effort_request("RELEASE", generation=generation)
                self.poisoned = True
                raise FamemProtocolError("Famem WAKE changed the arena layout or generation.")
            try:
                self.native.worker_remap(self.device, page_types, sizes, handles)
            except Exception:
                self._release_local_mapping()
                self._best_effort_request("RELEASE", generation=generation)
                self.poisoned = True
                raise
            self._remember_mapping(fields, self.base_address)
            self.active = True

    def close(self) -> None:
        with self._lock:
            if self._cleanup_complete:
                return
            self.closed = False
            first_error: Exception | None = None
            native_released = not self.capacity
            server_released = not self.capacity or self._server_released or self.poisoned
            if self.capacity:
                if self.active:
                    try:
                        self.native.worker_unmap(self.device)
                        self.active = False
                    except Exception as error:
                        first_error = error
                        self.poisoned = True
                if not self.poisoned and not self._server_released:
                    try:
                        self._request("RELEASE", generation=self.generation)
                        self._server_released = True
                    except Exception as error:
                        first_error = first_error or error
                        if not (isinstance(error, FamemProtocolError) and str(error).startswith("RuntimeError:")):
                            self.poisoned = True
                try:
                    self.native.worker_release(self.device)
                    native_released = True
                except Exception as error:
                    first_error = first_error or error
                server_released = self._server_released or self.poisoned
            if native_released and server_released:
                self.capacity = self.base_address = 0
                self.active = False
                self.extent_page_types, self.extent_sizes, self.shareable_handles = [], [], []
            self.closed = True
            if server_released:
                with suppress(OSError):
                    self._connection.close()
            self._cleanup_complete = native_released and server_released
            if first_error is not None:
                raise first_error

    def _request(self, operation: str, *, include_session: bool = True, **fields: Any) -> dict[str, Any]:
        if self.closed:
            raise FamemTransportError("Famem client is closed.")
        self._request_id += 1
        request_id = self._request_id
        request = {"version": FAMEM_PROTOCOL_VERSION, "request_id": request_id, "op": operation, **fields}
        if include_session:
            request["session_id"] = self.session_id
        try:
            send_message(self._connection, request)
            response = receive_message(self._connection, request_id=request_id, operation=operation)
            if response is None:
                raise FamemTransportError("Famem server closed the control connection.")
        except _MappingError as error:
            self.poisoned = True
            if error.generation:
                self._best_effort_request("RELEASE", generation=error.generation)
            raise
        except Exception as error:
            self.poisoned = True
            raise FamemTransportError("Famem RPC failed with ambiguous memory state; restart the worker.") from error
        if not response["ok"]:
            error_type = response["error_type"]
            error_message = response["error"]
            if error_type == "FamemBusyError":
                raise FamemBusyError(error_message)
            if operation == "WAKE":
                self.poisoned = True
            raise FamemProtocolError(f"{error_type}: {error_message}")
        return response

    def _best_effort_request(self, operation: str, **fields: Any) -> None:
        with suppress(Exception):
            self._request(operation, **fields)

    def _require_healthy(self) -> None:
        if self.closed or self.poisoned:
            state = "closed" if self.closed else "poisoned"
            raise RuntimeError(f"Famem client is {state}; restart the worker before using NPU memory.")
