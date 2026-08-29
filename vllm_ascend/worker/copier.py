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

from __future__ import annotations

import os
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field
from multiprocessing.synchronize import Event
from queue import Empty
from typing import Any, Protocol

import regex as re
import torch
from vllm.utils.system_utils import arm_parent_death_signal, get_mp_context

from vllm_ascend import envs

MSG_DESC_SEND_FINISH = "cpu_mem_done"
MSG_NPU_MEM_READY = "NPU_mem_ready"
MSG_SUSPEND_START = "suspend"
MSG_RESUME_START = "resume"
MSG_SHUTDOWN = "shutdown"

MSG_COPIER_STARTED = "copier_started"
MSG_COPIER_INIT_DONE = "copier_init_done"
MSG_COPIER_SUSPEND_DONE = "copier_suspend_done"
MSG_COPIER_ERROR = "copier_error"

ShareHandleType = tuple[int, int, int, int, int]
LocalHandleType = tuple[int, int, int, int]

_LAYER_NAME = re.compile(r"layers\.(\d+)$")
_layer_ready_events: dict[int, Event] = {}


def _require_spawn_context(context: Any) -> None:
    if context.get_start_method() != "spawn":
        raise RuntimeError(
            "multiproc_pipe Copier processes require "
            "VLLM_WORKER_MULTIPROC_METHOD=spawn; set it before constructing LLM."
        )


def _require_copier_start_owner(expected_owner_pid: int) -> None:
    if os.getpid() != expected_owner_pid:
        raise RuntimeError("Copier process must be started by the Worker that created it")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Copier process must be started from the Worker main thread")


def _arm_parent_death_signal(expected_parent_pid: int) -> None:
    arm_parent_death_signal(expected_parent_pid, process_name="Copier")


def _run_copier_with_parent_guard(expected_parent_pid: int, target, target_args) -> None:
    _arm_parent_death_signal(expected_parent_pid)
    target(*target_args)


def set_layer_ready_events(events: dict[int, Event]) -> None:
    if sorted(events) != list(range(len(events))):
        raise ValueError("multiproc_pipe layer event indices must be contiguous")
    global _layer_ready_events
    _layer_ready_events = events


def get_layer_ready_event(layer_index: int) -> Event:
    try:
        return _layer_ready_events[layer_index]
    except KeyError as error:
        raise RuntimeError(f"multiproc_pipe event for layer {layer_index} is not initialized") from error


@dataclass
class SharedWeightDesc:
    layer_name: str
    handles: list[ShareHandleType]
    cpu_tensors: list[torch.Tensor] = field(default_factory=list)


@dataclass
class RecoverWeightDesc:
    layer_name: str
    handles: list[ShareHandleType]


def parse_layer_index(layer_name: str) -> int:
    """Return the original Camem layer sentinel used by the Copier.

    Unknown allocator blocks and public weights are restored before numbered
    transformer layers.  ``-3`` is deliberately a non-raising invalid marker;
    callers that consume descriptors must reject it explicitly.
    """
    if layer_name == "unknown":
        return -2
    if layer_name == "pub":
        return -1
    match = _LAYER_NAME.fullmatch(layer_name)
    if match is None:
        return -3
    return int(match.group(1))


def should_preload(layer_name: str) -> bool:
    """Whether a descriptor must be restored before layer zero is released."""
    return parse_layer_index(layer_name) in (-2, -1)


def _require_layer_name(layer_name: str) -> int:
    layer_index = parse_layer_index(layer_name)
    if layer_index == -3:
        raise ValueError(f"Invalid multiproc_pipe layer name: {layer_name!r}")
    return layer_index


def _expected_layer_names(num_layers: int) -> list[str]:
    return ["unknown", "pub", *(f"layers.{index}" for index in range(num_layers))]


def _validate_layer_coverage(descriptors: Sequence[Any], num_layers: int, label: str) -> None:
    received = [descriptor.layer_name for descriptor in descriptors]
    expected = _expected_layer_names(num_layers)
    if received != expected:
        raise RuntimeError(
            f"{label} descriptors do not cover the Camem layer contract: expected={expected}, received={received}"
        )


def _validate_int_handle(handle, length: int, error: str):
    if not isinstance(handle, tuple) or len(handle) != length:
        raise RuntimeError(error)
    malformed = any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in handle)
    if malformed or not all(handle[1:]):
        raise RuntimeError(error)
    return handle


def _validate_share_handle(handle: object) -> ShareHandleType:
    return _validate_int_handle(handle, 5, "multiproc_pipe requires an active five-field Camem handle")


def _validate_local_handle(handle: object) -> LocalHandleType:
    return _validate_int_handle(handle, 4, "Copier returned a malformed or inactive local Camem handle")


class NativeCopierOps:
    def __init__(self) -> None:
        from vllm_ascend.device_allocator.camem import (
            copy_free,
            copy_malloc_use_share,
        )
        from vllm_ascend.vllm_ascend_C import (
            python_get_bare_tgid,
            python_memcpy_device_to_host,
            python_memcpy_host_to_device,
        )

        self._copy_free = copy_free
        self._copy_malloc = copy_malloc_use_share
        self._get_bare_tgid = python_get_bare_tgid
        self._device_to_host = python_memcpy_device_to_host
        self._host_to_device = python_memcpy_host_to_device

    def bare_tgid(self, device: int) -> int:
        value = self._get_bare_tgid(device)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError("Camem returned an invalid Copier bare TGID")
        return value

    def import_handle(self, handle: ShareHandleType) -> LocalHandleType:
        device, size, _, _, share_handle = _validate_share_handle(handle)
        return _validate_local_handle(self._copy_malloc((device, size, 0, share_handle)))

    def free_handle(self, handle: LocalHandleType) -> None:
        self._copy_free(_validate_local_handle(handle))

    def copy_device_to_host(self, tensor: torch.Tensor, handle: LocalHandleType) -> None:
        device, size, device_address, _ = _validate_local_handle(handle)
        if tensor.numel() * tensor.element_size() != size:
            raise RuntimeError("Copier host backup size does not match its Camem allocation")
        self._device_to_host(device, tensor.data_ptr(), device_address, size)

    def copy_host_to_device(self, handle: LocalHandleType, tensor: torch.Tensor) -> None:
        device, size, device_address, _ = _validate_local_handle(handle)
        if tensor.numel() * tensor.element_size() != size:
            raise RuntimeError("Copier restore size does not match its Camem allocation")
        self._host_to_device(device, device_address, tensor.data_ptr(), size)


def _allocate_host_backup(size: int) -> torch.Tensor:
    try:
        return torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
    except RuntimeError:
        return torch.empty(size, dtype=torch.uint8, device="cpu")


def backup_weight_descriptors(descriptors: list[SharedWeightDesc], ops) -> list[SharedWeightDesc]:
    seen_layers: set[str] = set()
    seen_allocations: set[tuple[int, int]] = set()
    for descriptor in descriptors:
        _require_layer_name(descriptor.layer_name)
        if descriptor.layer_name in seen_layers:
            raise RuntimeError(f"Duplicate multiproc_pipe descriptor for {descriptor.layer_name}")
        seen_layers.add(descriptor.layer_name)
        if descriptor.cpu_tensors:
            raise RuntimeError("Worker must not populate copier-owned CPU backup tensors")
        for raw_handle in descriptor.handles:
            handle = _validate_share_handle(raw_handle)
            allocation_key = (handle[0], handle[2])
            if allocation_key in seen_allocations:
                raise RuntimeError("A Camem allocation was assigned to more than one layer")
            seen_allocations.add(allocation_key)
            local_handle = ops.import_handle(handle)
            try:
                tensor = _allocate_host_backup(handle[1])
                ops.copy_device_to_host(tensor, local_handle)
                descriptor.cpu_tensors.append(tensor)
            finally:
                ops.free_handle(local_handle)
    return sorted(
        descriptors,
        key=lambda descriptor: (
            not should_preload(descriptor.layer_name),
            _require_layer_name(descriptor.layer_name),
        ),
    )


def import_recovery_descriptors(descriptors: list[RecoverWeightDesc], ops):
    imported_by_layer: dict[str, list[LocalHandleType]] = {}
    imported: list[LocalHandleType] = []
    try:
        for descriptor in descriptors:
            _require_layer_name(descriptor.layer_name)
            if descriptor.layer_name in imported_by_layer:
                raise RuntimeError(f"Duplicate recovery descriptor for {descriptor.layer_name}")
            local_handles: list[LocalHandleType] = []
            for handle in descriptor.handles:
                local_handle = ops.import_handle(_validate_share_handle(handle))
                local_handles.append(local_handle)
                imported.append(local_handle)
            imported_by_layer[descriptor.layer_name] = local_handles
    except BaseException:
        for handle in reversed(imported):
            with suppress(Exception):
                ops.free_handle(handle)
        raise
    return imported_by_layer, imported


def restore_weight_descriptors(backups, recoveries, layer_ready_events, ops, device_id: int | None = None) -> None:
    expected_layers = {descriptor.layer_name for descriptor in backups}
    if set(recoveries) != expected_layers:
        raise RuntimeError(
            "Recovery descriptors do not match the initial weight layout: "
            f"expected={sorted(expected_layers)}, received={sorted(recoveries)}"
        )
    for descriptor in backups:
        local_handles = recoveries[descriptor.layer_name]
        if len(local_handles) != len(descriptor.cpu_tensors):
            raise RuntimeError(f"Recovery handle count changed for {descriptor.layer_name}")
        if device_id is not None:
            layer_index = _preload_weight(descriptor, local_handles, device_id)
        else:
            for local_handle, tensor in zip(local_handles, descriptor.cpu_tensors, strict=True):
                ops.copy_host_to_device(local_handle, tensor)
            layer_index = _require_layer_name(descriptor.layer_name)
        if layer_index >= 0:
            if layer_index not in layer_ready_events:
                raise RuntimeError(f"Layer readiness index {layer_index} is out of range")
            layer_ready_events[layer_index].set()


def _preload_weight(desc: SharedWeightDesc, npu_handle: Any, device_id: int) -> int:
    """Restore one Camem descriptor and return its layer sentinel.

    This is the compatibility helper used by the original Camem Copier.  The
    current controller may pass either one imported handle or the layer's list
    of imported handles.
    """
    if hasattr(torch, "npu"):
        torch.npu.set_device(device_id)
    handles = [npu_handle] if isinstance(npu_handle, tuple) else list(npu_handle)
    if len(handles) != len(desc.cpu_tensors):
        raise RuntimeError(f"Recovery handle count changed for {desc.layer_name}")
    ops = NativeCopierOps()
    for handle, tensor in zip(handles, desc.cpu_tensors, strict=True):
        ops.copy_host_to_device(_validate_local_handle(handle), tensor)
    return _require_layer_name(desc.layer_name)


def _receive_until(queue: Any, sentinel: str, expected_type: type[Any]) -> list[Any]:
    values: list[Any] = []
    while True:
        value = queue.get()
        if value == sentinel:
            return values
        if value == MSG_SHUTDOWN:
            raise InterruptedError("Copier shutdown requested during initialization")
        if not isinstance(value, expected_type):
            raise RuntimeError(f"Copier received an unexpected {type(value).__name__} message")
        values.append(value)


def _free_imported(handles: list[LocalHandleType], ops: NativeCopierOps | Any) -> None:
    try:
        with ExitStack() as cleanup:
            for handle in handles:
                cleanup.callback(ops.free_handle, handle)
    finally:
        handles.clear()


class CopierMemoryBackend(Protocol):
    """Physical-memory adapter for the single Camem Copier state machine."""

    def bare_tgid(self) -> int: ...

    def initialize(self, desc_queue: Any) -> None: ...

    def suspend(self) -> None: ...

    def resume(self, recover_queue: Any) -> None: ...

    def close(self) -> None: ...


class CamemCopierMemoryBackend:
    """Camem 5-tuple implementation of the common Copier memory contract."""

    def __init__(self, device: int, layer_events: dict[int, Event]) -> None:
        self.device = device
        self.layer_events = layer_events
        self.ops = NativeCopierOps()
        self.backups: list[SharedWeightDesc] = []
        self.imported_handles: list[LocalHandleType] = []

    def bare_tgid(self) -> int:
        return self.ops.bare_tgid(self.device)

    def initialize(self, desc_queue: Any) -> None:
        descriptors = _receive_until(desc_queue, MSG_DESC_SEND_FINISH, SharedWeightDesc)
        self.backups = backup_weight_descriptors(descriptors, self.ops)
        _validate_layer_coverage(self.backups, len(self.layer_events), "Initial")

    def suspend(self) -> None:
        _free_imported(self.imported_handles, self.ops)

    def resume(self, recover_queue: Any) -> None:
        descriptors = _receive_until(
            recover_queue,
            MSG_NPU_MEM_READY,
            RecoverWeightDesc,
        )
        descriptors.sort(key=lambda descriptor: _require_layer_name(descriptor.layer_name))
        _validate_layer_coverage(descriptors, len(self.layer_events), "Recovery")
        recoveries, self.imported_handles = import_recovery_descriptors(descriptors, self.ops)
        restore_weight_descriptors(
            self.backups,
            recoveries,
            self.layer_events,
            self.ops,
            self.device,
        )

    def close(self) -> None:
        _free_imported(self.imported_handles, self.ops)


def _run_copier_loop(
    device: int,
    desc_queue: Any,
    recover_queue: Any,
    ctrl_queue: Any,
    response_queue: Any,
    layer_events: dict[int, Event],
    backend_factory: Callable[[int, dict[int, Event]], CopierMemoryBackend],
) -> None:
    """Run the only Copier initialization/suspend/resume control loop."""
    backend: CopierMemoryBackend | None = None
    state = "initializing"
    try:
        if hasattr(torch, "npu"):
            torch.npu.set_device(device)
        backend = backend_factory(device, layer_events)
        if response_queue is not None:
            response_queue.put((MSG_COPIER_STARTED, backend.bare_tgid()))
        backend.initialize(desc_queue)
        state = "active"
        if response_queue is not None:
            response_queue.put(MSG_COPIER_INIT_DONE)

        while True:
            message = ctrl_queue.get()
            if message == MSG_SUSPEND_START:
                if state not in {"active", "suspended"}:
                    raise RuntimeError(f"Copier cannot suspend from state {state!r}")
                backend.suspend()
                for event in layer_events.values():
                    event.clear()
                if response_queue is not None:
                    response_queue.put(MSG_COPIER_SUSPEND_DONE)
                state = "suspended"
            elif message == MSG_RESUME_START:
                if state != "suspended":
                    raise RuntimeError(f"Copier cannot resume from state {state!r}")
                backend.resume(recover_queue)
                state = "active"
            elif message == MSG_SHUTDOWN:
                backend.close()
                return
            else:
                raise RuntimeError(f"Copier received unknown control message {message!r}")
    except InterruptedError:
        if backend is not None:
            with suppress(Exception):
                backend.close()
        return
    except BaseException:
        if backend is not None:
            with suppress(Exception):
                backend.close()
        if response_queue is not None:
            with suppress(Exception):
                response_queue.put((MSG_COPIER_ERROR, traceback.format_exc()))
        raise


@dataclass(frozen=True)
class _CopierControlChannel:
    """Health-aware extension carried through the original ctrl_queue slot."""

    commands: Any
    responses: Any
    device: int
    backend_factory: Callable[[int, dict[int, Event]], CopierMemoryBackend]


def _rank_queue(queue: Any, tp_size: int, local_rank: int) -> Any:
    if isinstance(queue, (list, tuple)):
        if len(queue) != tp_size:
            raise ValueError("Camem Copier queue count does not match tensor parallel size")
        return queue[local_rank]
    return queue


def copier_main(
    desc_queue: Any,
    npu_recover_queue: Any,
    ctrl_queue: Any,
    tp_size: int,
    local_rank: int,
    layer_ready_events: dict[int, Event],
) -> None:
    """Run the shared Copier through the original six-argument Camem ABI."""
    if tp_size <= 0 or local_rank < 0 or local_rank >= tp_size:
        raise ValueError("Camem Copier received an invalid rank configuration")
    control = _rank_queue(ctrl_queue, tp_size, local_rank)
    if isinstance(control, _CopierControlChannel):
        device = control.device
        command_queue = control.commands
        response_queue = control.responses
        backend_factory = control.backend_factory
    else:
        device = local_rank
        command_queue = control
        response_queue = None
        backend_factory = CamemCopierMemoryBackend
    _run_copier_loop(
        device,
        _rank_queue(desc_queue, tp_size, local_rank),
        _rank_queue(npu_recover_queue, tp_size, local_rank),
        command_queue,
        response_queue,
        layer_ready_events,
        backend_factory,
    )


class CopierController:
    _label = "Copier"
    _queue_names = ("init_queue", "recover_queue", "ctrl_queue", "ctrl_response_queue")

    def __init__(
        self,
        device: int,
        num_layers: int,
        layer_ready_events=None,
        timeout=None,
        *,
        backend_factory: Callable[[int, dict[int, Event]], CopierMemoryBackend] = CamemCopierMemoryBackend,
        label: str | None = None,
        queues: tuple[Any, Any, Any] | None = None,
        tp_size: int = 1,
        local_rank: int = 0,
    ) -> None:
        if label is not None:
            self._label = label
        if device < 0 or num_layers <= 0:
            raise ValueError(f"{self._label} requires a valid device and at least one layer")
        if tp_size <= 0 or local_rank < 0 or local_rank >= tp_size:
            raise ValueError(f"{self._label} received an invalid rank configuration")
        self.device, self.num_layers = device, num_layers
        self.tp_size, self.local_rank = tp_size, local_rank
        self.timeout = float(timeout or envs.VLLM_ASCEND_MULTIPROC_PIPE_TIMEOUT)
        if self.timeout <= 0:
            raise ValueError(f"{self._label} timeout must be positive")
        context = get_mp_context()
        _require_spawn_context(context)
        if queues is None:
            queues = tuple(context.Queue() for _ in range(3))
        elif len(queues) != 3 or any(queue is None for queue in queues):
            raise ValueError(f"{self._label} requires desc, recovery, and control queues")
        desc_queue, recover_queue, command_queue = queues
        response_queue = context.Queue()
        control = _CopierControlChannel(
            command_queue,
            response_queue,
            device,
            backend_factory,
        )
        self._control_channel = control
        all_queues = (desc_queue, recover_queue, command_queue, response_queue)
        for name, queue in zip(self._queue_names, all_queues, strict=True):
            setattr(self, name, queue)
        self.layer_ready_events = (
            layer_ready_events
            if layer_ready_events is not None
            else {index: context.Event() for index in range(num_layers)}
        )
        if sorted(self.layer_ready_events) != list(range(num_layers)):
            raise ValueError(f"{self._label} layer events do not match the model")
        self._owner_pid = os.getpid()
        self.process = context.Process(
            target=_run_copier_with_parent_guard,
            args=(
                self._owner_pid,
                copier_main,
                (
                    self.desc_queue,
                    self.npu_recover_queue,
                    self._control_channel,
                    self.tp_size,
                    self.local_rank,
                    self.layer_ready_events,
                ),
            ),
            name=f"VllmAscend{self._label.replace(' ', '')}-{device}",
            daemon=True,
        )
        self._started = self._initial_sent = self._initialized = self._closed = False
        self._cleanup_complete = False
        self._resume_started = False
        self.bare_tgid = 0

    def start(self) -> int:
        if self._started:
            return self.bare_tgid
        _require_copier_start_owner(self._owner_pid)
        self.process.start()
        self._started = True
        message = self._receive_control(MSG_COPIER_STARTED)
        value = message[1] if isinstance(message, tuple) and len(message) == 2 else None
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(f"{self._label} startup returned an invalid response")
        self.bare_tgid = value
        return value

    def _send_initial(self, *messages: Any) -> None:
        if not self._started or self._initial_sent:
            raise RuntimeError(f"{self._label} initialization must be sent exactly once after startup")
        for message in messages:
            getattr(self, self._queue_names[0]).put(message)
        self._initial_sent = True

    def wait_until_initialized(self) -> None:
        if not self._initial_sent or self._initialized:
            raise RuntimeError(f"{self._label} is not waiting for initialization")
        self._receive_control(MSG_COPIER_INIT_DONE)
        self._initialized = True

    def suspend(self) -> None:
        self._require_initialized()
        # Clear before publishing suspend so forward cannot observe stale readiness.
        for event in self.layer_ready_events.values():
            event.clear()
        self.ctrl_queue.put(MSG_SUSPEND_START)
        self._receive_control(MSG_COPIER_SUSPEND_DONE)

    def wait_for_layer(self, layer_index: int) -> None:
        self._require_initialized()
        if layer_index < 0 or layer_index >= self.num_layers:
            raise ValueError(f"Layer index {layer_index} is out of range")
        deadline = time.monotonic() + self.timeout
        event = self.layer_ready_events[layer_index]
        while not event.wait(timeout=min(0.1, max(0.0, deadline - time.monotonic()))):
            self._raise_if_failed()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {self._label} layer {layer_index}")

    def close(self) -> None:
        if getattr(self, "_cleanup_complete", False):
            return
        self._closed = True
        error: Exception | None = None
        alive = False
        if getattr(self, "_started", False):
            try:
                alive = self.process.is_alive()
                if alive:
                    if getattr(self, "_resume_started", False):
                        shutdown_queues = (self.npu_recover_queue,)
                        self._resume_started = False
                    elif self._initialized:
                        shutdown_queues = (self.ctrl_queue,)
                    else:
                        # Initialization may be reading descriptors, copying
                        # weights, or already waiting for control before its
                        # acknowledgement reaches the Worker.
                        shutdown_queues = (getattr(self, self._queue_names[0]), self.ctrl_queue)
                    for queue in shutdown_queues:
                        queue.put(MSG_SHUTDOWN)
                self.process.join(timeout=min(getattr(self, "timeout", 5.0), 10.0))
                alive = self.process.is_alive()
            except Exception as current:
                error, alive = current, True
            for method in ("terminate", "kill"):
                if not alive or (method == "kill" and not hasattr(self.process, method)):
                    continue
                try:
                    getattr(self.process, method)()
                    self.process.join(timeout=5.0)
                    alive = self.process.is_alive()
                except Exception as current:
                    error = error or current
            if alive:
                error = error or RuntimeError(f"{self._label} process did not exit during shutdown")
        try:
            with ExitStack() as cleanup:
                for name in self._queue_names:
                    queue = getattr(self, name)
                    cleanup.callback(queue.close)
                    cleanup.callback(queue.cancel_join_thread)
            queues_ok = True
        except Exception as current:
            queues_ok = False
            error = error or current
        process_ok = not alive
        process = getattr(self, "process", None)
        if process_ok and queues_ok and process is not None and hasattr(process, "close"):
            try:
                process.close()
            except Exception as current:
                process_ok = False
                error = error or current
        self._cleanup_complete = process_ok and queues_ok
        if error is not None:
            raise error

    def _require_initialized(self) -> None:
        if self._closed or not self._initialized:
            raise RuntimeError(f"{self._label} process is not initialized")
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._started and not self.process.is_alive():
            raise RuntimeError(f"{self._label} process exited with code {self.process.exitcode}")

    def _receive_control(self, expected: str) -> Any:
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                message = self.ctrl_response_queue.get(timeout=min(0.1, max(0.0, remaining)))
            except Empty:
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for {self._label} message {expected!r}")
                self._raise_if_failed()
                continue
            if isinstance(message, tuple) and message and message[0] == MSG_COPIER_ERROR:
                raise RuntimeError(f"{self._label} process failed:\n{message[1]}")
            if (message[0] if isinstance(message, tuple) and message else message) != expected:
                raise RuntimeError(f"Expected {self._label} message {expected!r}, received {message!r}")
            return message


class CopierProcess(CopierController):
    _queue_names = ("desc_queue", "npu_recover_queue", "ctrl_queue", "ctrl_response_queue")

    def initialize(self, descriptors: list[SharedWeightDesc]) -> None:
        self.send_descriptors(descriptors)
        self.wait_until_initialized()

    def send_initial(self, *messages: Any) -> None:
        self._send_initial(*messages)

    def send_descriptors(self, descriptors: list[SharedWeightDesc]) -> None:
        self.send_initial(*descriptors, MSG_DESC_SEND_FINISH)

    def begin_resume(self) -> None:
        self._require_initialized()
        if self._resume_started:
            raise RuntimeError("A Copier resume transaction is already active")
        self.ctrl_queue.put(MSG_RESUME_START)
        self._resume_started = True

    def send_recovery_descriptor(self, descriptor: RecoverWeightDesc) -> None:
        if not self._resume_started:
            raise RuntimeError("Copier resume has not started")
        if not isinstance(descriptor, RecoverWeightDesc):
            raise TypeError("Copier recovery message must be RecoverWeightDesc")
        self.send_recovery(descriptor)

    def resume(self, descriptors: list[RecoverWeightDesc]) -> None:
        self.begin_resume()
        for descriptor in descriptors:
            self.send_recovery_descriptor(descriptor)
        self.finish_resume()

    def send_recovery(self, message: Any) -> None:
        if not self._resume_started:
            raise RuntimeError("Copier resume has not started")
        self.npu_recover_queue.put(message)

    def finish_resume(self) -> None:
        if not self._resume_started:
            raise RuntimeError("Copier resume has not started")
        self.npu_recover_queue.put(MSG_NPU_MEM_READY)
        self._resume_started = False

    def abort_resume(self) -> None:
        if self._resume_started:
            self.npu_recover_queue.put(MSG_SHUTDOWN)
            self._resume_started = False
