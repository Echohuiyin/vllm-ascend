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

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.worker.worker_base import WorkerFatalError

from vllm_ascend.device_allocator.famem import FaMemAllocator
from vllm_ascend.device_allocator.famem_client import FamemBusyError
from vllm_ascend.device_allocator.famem_native import FamemNativeStats, FamemPageType, FamemWorkerState


class FakeNative:
    library_path = "/unused/libvllm_ascend_famem.so"

    def __init__(self):
        self.capacity = 16384
        self.heap_top = 0
        self.state = FamemWorkerState.UNINITIALIZED

    def allocation_granularity(self, device):
        return 1024

    def worker_stats(self):
        return FamemNativeStats(
            self.capacity,
            self.heap_top,
            self.heap_top,
            2,
            0x20000000,
            self.state,
        )


class FakeClient:
    instances = []

    def __init__(self, device, socket_dir, native):
        self.native = native
        self.extent_page_types = [FamemPageType.HUGE_2M]
        self.extent_sizes = [native.capacity]
        self.shareable_handles = [800]
        self.active = False
        self.poisoned = False
        self.closed = False
        FakeClient.instances.append(self)

    def acquire(self, capacity):
        assert capacity == self.native.capacity
        self.active = True
        self.native.state = FamemWorkerState.ACTIVE
        return 0x20000000

    def sleep(self):
        self.active = False
        self.native.state = FamemWorkerState.SLEEPING

    def wake(self):
        self.active = True
        self.native.state = FamemWorkerState.ACTIVE

    def close(self):
        self.closed = True
        self.active = False
        self.native.state = FamemWorkerState.CLOSED


class FakeCopier:
    _cleanup_complete = True

    def __init__(self):
        self.suspend_calls = 0
        self.mappings = []
        self.waits = []
        self.closed = False
        self.resume_started = False

    def suspend(self):
        self.suspend_calls += 1

    def begin_resume(self):
        self.resume_started = True

    def send_recovery(self, mapping):
        assert self.resume_started
        self.mappings.append(mapping)

    def finish_resume(self):
        assert self.resume_started
        self.resume_started = False

    def abort_resume(self):
        self.resume_started = False

    def wait_for_layer(self, layer_index):
        self.waits.append(layer_index)

    def close(self):
        self.closed = True


class BusyOnceClient(FakeClient):
    acquire_attempts = 0

    def acquire(self, capacity):
        type(self).acquire_attempts += 1
        if type(self).acquire_attempts == 1:
            raise FamemBusyError("another model is active")
        return super().acquire(capacity)


@pytest.fixture
def fake_npu(monkeypatch):
    @contextmanager
    def use_mem_pool(pool, device=None):
        yield

    class PluggableAllocator:
        def __init__(self, path, malloc_name, free_name):
            self._allocator = object()

    memory = SimpleNamespace(
        NPUPluggableAllocator=PluggableAllocator,
        MemPool=lambda allocator: object(),
        use_mem_pool=use_mem_pool,
    )
    npu = SimpleNamespace(memory=memory, synchronize=lambda device: None)
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    return npu


def make_allocator(native):
    allocator = FaMemAllocator(
        device=0,
        capacity=native.capacity,
        socket_dir="/unused",
        native=native,
        client_factory=FakeClient,
    )
    allocator._copier = FakeCopier()
    allocator._pipeline_initialized = True
    allocator._num_layers = 2
    return allocator


def initialize_pools(allocator, native):
    with allocator.use_memory_pool("weights"):
        native.heap_top = 2500
    with allocator.use_memory_pool("kv_cache"):
        native.heap_top = 8192


def test_pool_order_and_weight_boundary(fake_npu):
    native = FakeNative()
    allocator = make_allocator(native)
    with pytest.raises(RuntimeError, match="expected 'weights'"), allocator.use_memory_pool("kv_cache"):
        pass

    initialize_pools(allocator, native)
    assert allocator.weight_end == 2500
    assert allocator.available_bytes == native.capacity - native.heap_top
    with pytest.raises(RuntimeError, match="expected None"), allocator.use_memory_pool("kv_cache"):
        pass


def test_available_bytes_rejects_native_heap_overflow(fake_npu):
    native = FakeNative()
    allocator = make_allocator(native)
    with allocator.use_memory_pool("weights"):
        native.heap_top = native.capacity + 512

    with pytest.raises(RuntimeError, match="heap_top=.*exceeds capacity"):
        _ = allocator.available_bytes


def test_level_one_sleep_uses_copier_backup(fake_npu):
    native = FakeNative()
    allocator = make_allocator(native)
    initialize_pools(allocator, native)

    allocator.sleep(offload_tags=("weights",))
    assert allocator.is_sleeping
    assert allocator._copier.suspend_calls == 1
    diagnostics = allocator.diagnostics()
    assert diagnostics["state"] == "SLEEPING"
    assert diagnostics["weight_end"] == 2500
    assert diagnostics["freed_bytes"] == 0
    assert "free_count" not in diagnostics
    assert diagnostics["base_address"] == 0x20000000
    assert diagnostics["extent_count"] == 1
    assert diagnostics["huge_1g_bytes"] == 0
    assert diagnostics["huge_2m_bytes"] == native.capacity
    assert allocator.client.shareable_handles == [800]

    allocator.wake_up()
    assert not allocator.is_sleeping
    assert len(allocator._copier.mappings) == 1
    assert allocator._copier.mappings[0].handles[0][3] == 800
    assert allocator._copier.waits == [1]


@pytest.mark.parametrize("offload_tags", [(), ("kv_cache",)])
def test_non_level_one_sleep_is_rejected_without_state_change(fake_npu, offload_tags):
    native = FakeNative()
    allocator = make_allocator(native)
    initialize_pools(allocator, native)
    client = allocator.client

    with pytest.raises(ValueError, match="only level 1"):
        allocator.sleep(offload_tags=offload_tags)

    assert not allocator.is_sleeping
    assert native.state == FamemWorkerState.ACTIVE
    assert client.closed is False
    assert allocator._copier.suspend_calls == 0


def test_worker_rejects_level_two_and_preserves_backup_layout(fake_npu, monkeypatch):
    worker_module = sys.modules.get("vllm_ascend.worker.worker")
    if worker_module is None:
        with monkeypatch.context() as patch:
            patch.setattr(torch, "npu", MagicMock())
            for name in ("torch_npu.op_plugin", "torch_npu.op_plugin.atb", "torch_npu.op_plugin.atb._atb_ops"):
                patch.setitem(sys.modules, name, MagicMock())
            patch.setitem(sys.modules, "torch_npu.profiler", MagicMock())
            patch.setitem(
                sys.modules, "vllm_ascend.worker.model_runner_v1", SimpleNamespace(NPUModelRunner=MagicMock())
            )
            from vllm_ascend.worker.worker import NPUWorker

        sys.modules.pop("vllm_ascend.worker.worker", None)
    else:
        NPUWorker = worker_module.NPUWorker

    allocator = make_allocator(FakeNative())
    worker = object.__new__(NPUWorker)
    worker._get_sleep_mode_allocator = lambda: allocator
    buffer = MagicMock()
    worker.model_runner = SimpleNamespace(model=MagicMock())
    worker.model_runner.model.named_buffers.return_value = [("buffer", buffer)]
    fake_npu.mem_get_info = MagicMock()
    allocator.sleep = MagicMock()

    with pytest.raises(ValueError, match="only level 1"):
        worker.sleep(level=2)

    fake_npu.mem_get_info.assert_not_called()
    worker.model_runner.model.named_buffers.assert_not_called()
    buffer.cpu.return_value.clone.assert_not_called()
    allocator.sleep.assert_not_called()

    camem_copier = SimpleNamespace(multiproc_pipe_enabled=True, sleep=MagicMock())
    worker._get_sleep_mode_allocator = lambda: camem_copier
    with pytest.raises(ValueError, match="only level 1"):
        worker.sleep(level=2)
    camem_copier.sleep.assert_not_called()

    worker._get_sleep_mode_allocator = lambda: allocator
    worker.vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(hidden_size=64)))
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_NZ", "0")
    worker._sleep_saved_buffers = {}
    allocator.wake_up = MagicMock()
    worker.wake_up()
    allocator.wake_up.assert_called_once_with(tags=None)
    worker.model_runner.model.named_parameters.assert_not_called()


@pytest.mark.parametrize("tags", [[], ["weights"]])
def test_partial_wake_is_rejected(fake_npu, tags):
    native = FakeNative()
    allocator = make_allocator(native)
    with pytest.raises(ValueError, match="whole-arena"):
        allocator.wake_up(tags=tags)


def test_close_is_idempotent(fake_npu):
    native = FakeNative()
    allocator = make_allocator(native)
    initialize_pools(allocator, native)
    client = allocator.client
    allocator.close()
    allocator.close()
    assert client.closed


def test_failed_pool_transaction_closes_bump_allocator(fake_npu):
    native = FakeNative()
    allocator = make_allocator(native)

    with pytest.raises(RuntimeError, match="load failed"), allocator.use_memory_pool("weights"):
        native.heap_top = 1024
        raise RuntimeError("load failed")

    assert allocator.diagnostics()["poisoned"] is True
    assert allocator._closed is True
    assert allocator.client.closed is True
    with pytest.raises(RuntimeError, match="closed"), allocator.use_memory_pool("weights"):
        pass


def test_busy_pool_setup_remains_retryable(fake_npu):
    native = FakeNative()
    BusyOnceClient.acquire_attempts = 0
    allocator = make_allocator(native)
    allocator.client_factory = BusyOnceClient

    with pytest.raises(FamemBusyError, match="another model is active"), allocator.use_memory_pool("weights"):
        pass

    assert allocator._closed is False
    assert allocator._poisoned is False
    assert allocator.client is None
    assert BusyOnceClient.instances[-1].closed is True

    with allocator.use_memory_pool("weights"):
        native.heap_top = 1024

    assert allocator.weight_end == 1024


def test_shutdown_sync_failure_uses_fail_closed_release(fake_npu):
    native = FakeNative()
    allocator = make_allocator(native)
    initialize_pools(allocator, native)
    client = allocator.client

    def fail_synchronize(device):
        del device
        raise RuntimeError("injected shutdown sync failure")

    fake_npu.synchronize = fail_synchronize

    allocator.close()

    assert client.poisoned is True
    assert client.closed is True


def test_synchronize_failure_poisons_allocator(fake_npu):
    native = FakeNative()
    allocator = make_allocator(native)
    initialize_pools(allocator, native)

    def fail_synchronize(device):
        del device
        raise RuntimeError("sync failed")

    fake_npu.synchronize = fail_synchronize

    with pytest.raises(WorkerFatalError, match="sync failed"):
        allocator.sleep(offload_tags=("weights",))

    assert allocator.diagnostics()["poisoned"] is True
    assert allocator._closed is True
    assert allocator.client.closed is True


def test_restore_failure_poisons_allocator(fake_npu):
    native = FakeNative()
    allocator = make_allocator(native)
    initialize_pools(allocator, native)
    allocator.sleep(offload_tags=("weights",))

    def fail_restore(layer_index):
        del layer_index
        raise RuntimeError("restore failed")

    allocator._copier.wait_for_layer = fail_restore

    with pytest.raises(WorkerFatalError, match="restore failed"):
        allocator.wake_up()

    assert allocator.diagnostics()["poisoned"] is True
    assert allocator._closed is True
    assert allocator.client.closed is True


@pytest.mark.parametrize("operation", ["sleep", "wake"])
def test_lost_cycle_response_fail_stops_worker(fake_npu, operation):
    native = FakeNative()
    allocator = make_allocator(native)
    initialize_pools(allocator, native)
    if operation == "wake":
        allocator.sleep(offload_tags=("weights",))
    client = allocator.client

    def lose_response():
        client.poisoned = True
        raise RuntimeError(f"lost {operation.upper()} response")

    setattr(client, operation, lose_response)
    transition = allocator.wake_up if operation == "wake" else lambda: allocator.sleep(offload_tags=("weights",))

    with pytest.raises(WorkerFatalError, match=f"lost {operation.upper()} response"):
        transition()

    assert allocator._poisoned is True
    assert allocator._closed is True
    assert client.closed is True


def test_busy_wake_remains_retryable(fake_npu):
    native = FakeNative()
    allocator = make_allocator(native)
    initialize_pools(allocator, native)
    allocator.sleep(offload_tags=("weights",))
    client = allocator.client
    original_wake = client.wake
    client.wake = lambda: (_ for _ in ()).throw(FamemBusyError("another model is active"))

    with pytest.raises(FamemBusyError, match="another model is active"):
        allocator.wake_up()

    assert allocator.is_sleeping
    assert allocator._cycle_state == "sync_sleeping"
    assert allocator._poisoned is False
    assert client.closed is False

    client.wake = original_wake
    allocator.wake_up()
    assert not allocator.is_sleeping


def test_expandable_segments_are_rejected(fake_npu, monkeypatch):
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

    with pytest.raises(ValueError, match="incompatible with expandable_segments"):
        make_allocator(FakeNative())


def test_power2_rounding_is_rejected(fake_npu, monkeypatch):
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "roundup_power2_divisions:4")

    with pytest.raises(ValueError, match="roundup_power2_divisions"):
        make_allocator(FakeNative())


def test_second_native_allocator_lifecycle_is_rejected(fake_npu):
    native = FakeNative()
    native.state = FamemWorkerState.ACTIVE

    with pytest.raises(RuntimeError, match="one allocator lifecycle"):
        make_allocator(native)


def test_weight_boundary_stats_failure_closes_allocator(fake_npu):
    native = FakeNative()
    allocator = make_allocator(native)

    def fail_stats():
        raise RuntimeError("stats failed")

    native.worker_stats = fail_stats

    with pytest.raises(RuntimeError, match="stats failed"), allocator.use_memory_pool("weights"):
        native.heap_top = 1024

    # The mapping is still unambiguous, so close may commit a clean RELEASE.
    assert allocator.client.poisoned is False
    assert allocator.client.closed is True
    assert allocator._poisoned is True
