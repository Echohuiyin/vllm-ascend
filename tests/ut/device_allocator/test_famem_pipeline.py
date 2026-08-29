# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm.v1.worker.worker_base import WorkerFatalError

from vllm_ascend.device_allocator.famem import FaMemAllocator
from vllm_ascend.device_allocator.famem_client import FamemBusyError
from vllm_ascend.device_allocator.famem_native import (
    FamemNativeAllocation,
    FamemNativeStats,
    FamemPageType,
    FamemWorkerState,
)
from vllm_ascend.worker.famem_copier import FamemCopierMemoryBackend

MIB = 1 << 20


class PipelineNative:
    library_path = "/unused/libvllm_ascend_famem.so"

    def __init__(self):
        self.capacity = 4 * MIB
        self.base = 0x20000000
        self.heap_top = 0
        self.state = FamemWorkerState.UNINITIALIZED

    def allocation_granularity(self, device):
        return 512

    def worker_stats(self):
        return FamemNativeStats(
            self.capacity,
            self.heap_top,
            self.heap_top,
            3,
            self.base,
            self.state,
        )

    def worker_allocations(self):
        return [
            FamemNativeAllocation(self.base, 512),
            FamemNativeAllocation(self.base + 512, 1024),
            FamemNativeAllocation(self.base + 1536, 512),
        ]


class PipelineClient:
    instances = []

    def __init__(self, device, socket_dir, native, copier_bare_tgid=0):
        del device, socket_dir
        self.native = native
        self.copier_bare_tgid = copier_bare_tgid
        self.extent_page_types = [FamemPageType.HUGE_2M]
        self.extent_sizes = [native.capacity]
        self.shareable_handles = [800]
        self.active = False
        self.poisoned = False
        self.closed = False
        PipelineClient.instances.append(self)

    def acquire(self, capacity):
        assert capacity == self.native.capacity
        self.active = True
        self.native.state = FamemWorkerState.ACTIVE
        return self.native.base

    def sleep(self):
        self.active = False
        self.native.state = FamemWorkerState.SLEEPING

    def wake(self):
        self.active = True
        self.native.state = FamemWorkerState.ACTIVE

    def close(self):
        self.closed = True


class PipelineCopier:
    instances = []

    def __init__(
        self,
        device,
        num_layers,
        layer_ready_events=None,
        *,
        backend_factory=None,
        label=None,
        queues=None,
        tp_size=1,
        local_rank=0,
    ):
        self.device = device
        self.num_layers = num_layers
        self.backend_factory = backend_factory
        self.label = label
        self.tp_size = tp_size
        self.local_rank = local_rank
        self.initial = None
        self.recoveries = []
        self.resume_started = False
        self.suspend_calls = 0
        self.waits = []
        self.closed = False
        self.layer_ready_events = layer_ready_events or {index: MagicMock() for index in range(num_layers)}
        self.desc_queue, self.npu_recover_queue, self.ctrl_queue = queues or (MagicMock(), MagicMock(), MagicMock())
        PipelineCopier.instances.append(self)

    def start(self):
        return 222

    def send_initial(self, initial):
        self.initial = initial

    def wait_until_initialized(self):
        return None

    def suspend(self):
        self.suspend_calls += 1

    def begin_resume(self):
        assert not self.resume_started
        self.resume_started = True

    def send_recovery(self, mapping):
        assert self.resume_started
        self.recoveries.append(mapping)

    def finish_resume(self):
        assert self.resume_started
        self.resume_started = False

    def abort_resume(self):
        self.resume_started = False

    def wait_for_layer(self, layer_index):
        self.waits.append(layer_index)

    def close(self):
        self.closed = True


class FailingLayerZeroCopier(PipelineCopier):
    def wait_for_layer(self, layer_index):
        assert layer_index == 0
        raise RuntimeError("injected layer-zero restore failure")


class FailingInitializationCopier(PipelineCopier):
    def wait_until_initialized(self):
        raise RuntimeError("injected Copier initialization failure")


class FailingLaterLayerCopier(PipelineCopier):
    def wait_for_layer(self, layer_index):
        raise RuntimeError(f"injected layer {layer_index} restore failure")


class RetriableCloseCopier(PipelineCopier):
    def __init__(self, device, num_layers):
        super().__init__(device, num_layers)
        self.close_attempts = 0
        self.alive = True
        self.process = SimpleNamespace(is_alive=lambda: self.alive)

    def close(self):
        self.close_attempts += 1
        if self.close_attempts == 1:
            raise RuntimeError("injected live Copier close failure")
        self.alive = False
        self.closed = True


class RetriableStoppedCopier(PipelineCopier):
    def __init__(self, device, num_layers):
        super().__init__(device, num_layers)
        self.close_attempts = 0
        self._cleanup_complete = False
        self.process = SimpleNamespace(is_alive=lambda: False)

    def close(self):
        self.close_attempts += 1
        if self.close_attempts == 1:
            raise RuntimeError("injected stopped Copier queue cleanup failure")
        self._cleanup_complete = True
        self.closed = True


class FailingStartupCleanupCopier(PipelineCopier):
    def __init__(self, device, num_layers, layer_ready_events=None, **kwargs):
        super().__init__(device, num_layers, layer_ready_events, **kwargs)
        self.close_attempts = 0
        self.alive = True
        self._cleanup_complete = False
        self.process = SimpleNamespace(is_alive=lambda: self.alive)

    def start(self):
        raise RuntimeError("injected Copier startup failure")

    def close(self):
        self.close_attempts += 1
        if self.close_attempts == 1:
            raise RuntimeError("injected Copier stop failure")
        self.alive = False
        self._cleanup_complete = True
        self.closed = True


def make_pipeline_allocator():
    native = PipelineNative()
    return native, FaMemAllocator(0, native.capacity, "/unused", native=native, client_factory=PipelineClient)


@pytest.fixture(autouse=True)
def reset_camem_pipeline_resources():
    previous = (
        CaMemAllocator.desc_queue,
        CaMemAllocator.npu_recover_queue,
        CaMemAllocator.ctrl_queue,
        CaMemAllocator.layer_ready_events,
    )
    CaMemAllocator.desc_queue = None
    CaMemAllocator.npu_recover_queue = None
    CaMemAllocator.ctrl_queue = None
    CaMemAllocator.layer_ready_events = None
    yield
    (
        CaMemAllocator.desc_queue,
        CaMemAllocator.npu_recover_queue,
        CaMemAllocator.ctrl_queue,
        CaMemAllocator.layer_ready_events,
    ) = previous


def attach_pipeline(copier_type=PipelineCopier, **state):
    native, allocator = make_pipeline_allocator()
    client, copier = PipelineClient(0, "/unused", native), copier_type(0, 2)
    allocator.client, allocator._copier = client, copier
    allocator.base_address = native.base
    for name, value in state.items():
        setattr(allocator, name, value)
    return allocator, client, copier


def track_close_order(copier, client):
    order = []
    for name, resource in (("copier", copier), ("client", client)):
        close = resource.close

        def tracked_close(name=name, close=close):
            order.append(name)
            close()

        resource.close = tracked_close
    return order


def test_famem_pipeline_groups_weights_and_runs_suspend_resume(monkeypatch):
    from vllm.model_executor.model_loader.base_loader import layer_to_addr

    @contextmanager
    def use_mem_pool(pool, device=None):
        del pool, device
        yield

    memory = SimpleNamespace(
        NPUPluggableAllocator=lambda *args: SimpleNamespace(_allocator=object()),
        MemPool=lambda allocator: object(),
        use_mem_pool=use_mem_pool,
    )
    monkeypatch.setattr(torch, "npu", SimpleNamespace(memory=memory, synchronize=lambda device: None), raising=False)
    monkeypatch.setattr("vllm_ascend.device_allocator.famem.CopierProcess", PipelineCopier)

    PipelineClient.instances.clear()
    PipelineCopier.instances.clear()
    native, allocator = make_pipeline_allocator()
    allocator.start_pipeline(0, 2)
    with allocator.use_memory_pool("weights"):
        native.heap_top = 2048

    layer_to_addr.clear()
    layer_to_addr.update(
        {
            "unknown": [],
            "pub": [native.base + 10],
            "layers.0": [native.base + 600],
            "layers.1": [native.base + 1600],
        }
    )
    allocator.send_descs_for_scattered_weights()
    allocator.wait_for_copier_ready()

    copier = PipelineCopier.instances[-1]
    client = PipelineClient.instances[-1]
    assert client.copier_bare_tgid == 222
    assert copier.backend_factory is FamemCopierMemoryBackend
    assert copier.label == "Famem Copier"
    mapping, weights = copier.initial
    assert mapping.handles == [(0, native.capacity, native.base, 800)]
    assert [descriptor.layer_name for descriptor in weights] == [
        "unknown",
        "pub",
        "layers.0",
        "layers.1",
    ]
    assert [[(span.offset, span.size) for span in descriptor.spans] for descriptor in weights] == [
        [],
        [(0, 512)],
        [(512, 1024)],
        [(1536, 512)],
    ]
    assert allocator.diagnostics()["backup_bytes"] == 2048
    with allocator.use_memory_pool("kv_cache"):
        native.heap_top = 3 * MIB

    allocator.sleep()
    assert allocator.is_sleeping
    allocator.wake_up()
    assert not allocator.is_sleeping
    assert copier.suspend_calls == 1
    assert copier.waits == [1]

    allocator.suspend()
    allocator.resume()

    assert not allocator.ready
    assert copier.suspend_calls == 2
    assert copier.waits == [1, 0]
    assert copier.recoveries[-1].handles == [(0, native.capacity, native.base, 800)]

    # A scheduler may hand the NPU away again before this model receives a
    # forward. Suspend must finish the in-progress restore instead of requiring
    # a synthetic request to mark the allocator ready.
    allocator.suspend()
    assert allocator._cycle_state == "suspended"
    assert copier.waits == [1, 0, 1]
    allocator.resume()
    assert copier.waits == [1, 0, 1, 0]

    allocator.wait_for_layer(1)
    assert copier.waits == [1, 0, 1, 0, 1]

    allocator.close()
    assert copier.closed


def test_famem_pipeline_retains_failed_startup_cleanup_for_retry(monkeypatch):
    _, allocator = make_pipeline_allocator()
    monkeypatch.setattr(
        "vllm_ascend.device_allocator.famem.CopierProcess",
        FailingStartupCleanupCopier,
    )

    with pytest.raises(RuntimeError, match="process could not be stopped"):
        allocator.start_pipeline(0, 2)

    copier = allocator._copier
    assert isinstance(copier, FailingStartupCleanupCopier)
    assert allocator._closed is True
    assert allocator._poisoned is True
    assert allocator._cleanup_complete is False

    allocator.close()

    assert allocator._cleanup_complete is True
    assert allocator._copier is None
    assert copier.close_attempts == 2


def test_famem_pipeline_rejects_crossed_cycle_pairs(monkeypatch):
    _, allocator = make_pipeline_allocator()
    allocator._copier = MagicMock()
    allocator._pipeline_initialized = True
    allocator._cycle_state = "suspended"
    allocator._sleeping = True
    allocator.client = MagicMock()

    try:
        allocator.wake_up()
    except RuntimeError as error:
        assert "completed with resume" in str(error)
    else:
        raise AssertionError("Famem accepted suspend followed by wake_up")

    allocator._cycle_state = "sync_sleeping"
    try:
        allocator.resume()
    except RuntimeError as error:
        assert "completed with wake_up" in str(error)
    else:
        raise AssertionError("Famem accepted sleep followed by resume")


def test_famem_resume_failure_closes_copier_before_client(monkeypatch):
    allocator, client, copier = attach_pipeline(
        FailingLayerZeroCopier,
        _pipeline_initialized=True,
        _cycle_state="suspended",
        _sleeping=True,
    )
    order = track_close_order(copier, client)
    monkeypatch.setattr(torch.npu, "synchronize", lambda device: None)

    with pytest.raises(WorkerFatalError, match="injected layer-zero restore failure"):
        allocator.resume()

    assert order == ["copier", "client"]
    assert allocator._closed is True
    assert copier.closed is True
    assert client.closed is True


def test_famem_resume_busy_remains_suspended_and_retryable():
    allocator, client, copier = attach_pipeline(
        _pipeline_initialized=True,
        _cycle_state="suspended",
        _sleeping=True,
    )
    original_wake = client.wake
    client.wake = lambda: (_ for _ in ()).throw(FamemBusyError("another model is active"))

    with pytest.raises(FamemBusyError, match="another model is active"):
        allocator.resume()

    assert allocator._cycle_state == "suspended"
    assert allocator.is_sleeping
    assert allocator._poisoned is False
    assert client.closed is False
    assert copier.recoveries == []

    client.wake = original_wake
    allocator.resume()
    assert allocator._cycle_state == "resuming"


def test_famem_initialization_failure_closes_copier_before_client():
    allocator, client, copier = attach_pipeline(FailingInitializationCopier, _descriptors_sent=True)
    order = track_close_order(copier, client)

    with pytest.raises(RuntimeError, match="injected Copier initialization failure"):
        allocator.wait_for_copier_ready()

    assert order == ["copier", "client"]
    assert allocator._closed is True


def test_famem_later_layer_failure_closes_copier_before_client(monkeypatch):
    allocator, client, copier = attach_pipeline(
        FailingLaterLayerCopier,
        _pipeline_initialized=True,
        _cycle_state="resuming",
        _ready=False,
    )
    order = track_close_order(copier, client)
    monkeypatch.setattr(torch.npu, "synchronize", lambda device: None)

    with pytest.raises(WorkerFatalError, match="injected layer 1 restore failure"):
        allocator.wait_for_layer(1)

    assert order == ["copier", "client"]
    assert allocator._closed is True


@pytest.mark.parametrize(
    ("copier_type", "message", "poisoned"),
    [
        (RetriableCloseCopier, "injected live Copier close failure", True),
        (RetriableStoppedCopier, "injected stopped Copier queue cleanup failure", False),
    ],
)
def test_famem_allocator_retries_copier_cleanup(monkeypatch, copier_type, message, poisoned):
    allocator, client, copier = attach_pipeline(copier_type)
    monkeypatch.setattr(torch.npu, "synchronize", lambda device: None)

    with pytest.raises(RuntimeError, match=message):
        allocator.close()

    assert allocator._cleanup_complete is False
    assert allocator._copier is copier
    assert client.poisoned is poisoned
    if not poisoned:
        assert client.closed is True

    allocator.close()

    assert allocator._cleanup_complete is True
    assert allocator._copier is None
    assert copier.close_attempts == 2


def test_famem_weight_descriptor_splits_allocation_at_extent_boundary():
    from vllm.model_executor.model_loader.base_loader import layer_to_addr

    native, allocator = make_pipeline_allocator()
    allocator.base_address = native.base
    allocator.weight_end = 2048
    allocator._num_layers = 2
    allocator.client = SimpleNamespace(extent_sizes=[1024, native.capacity - 1024])
    layer_to_addr.clear()
    layer_to_addr.update(
        {
            "unknown": [],
            "pub": [],
            "layers.0": [native.base + 600],
            "layers.1": [native.base + 1600],
        }
    )

    descriptors = allocator._build_weight_descriptors()

    layer_zero = next(descriptor for descriptor in descriptors if descriptor.layer_name == "layers.0")
    assert [(span.offset, span.size) for span in layer_zero.spans] == [
        (512, 512),
        (1024, 512),
    ]
