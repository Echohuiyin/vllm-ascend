#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from unittest.mock import MagicMock, call, patch

import pytest
import torch
from vllm.v1.worker.worker_base import WorkerFatalError

from vllm_ascend.device_allocator.camem import (
    AllocationData,
    CaMemAllocator,
    copier_free,
    copier_malloc_use_share,
    copy_free,
    copy_malloc_use_share,
    create_and_map,
    create_and_map_share,
    create_and_map_share_alloc,
    find_loaded_library,
    get_pluggable_allocator,
    unmap_and_release,
    unmap_and_release_share_alloc,
)


def dummy_malloc(args):
    pass


def dummy_free(ptr):
    return (0, 0, 0, 0)


class TestCaMem:
    @pytest.fixture(autouse=True)
    def reset_allocator_singleton(self):
        previous_instance = CaMemAllocator.instance
        previous_switch = CaMemAllocator.pipeline_switch
        previous_resources = (
            CaMemAllocator.desc_queue,
            CaMemAllocator.npu_recover_queue,
            CaMemAllocator.ctrl_queue,
            CaMemAllocator.layer_ready_events,
        )
        CaMemAllocator.instance, CaMemAllocator.pipeline_switch = None, False
        CaMemAllocator.desc_queue = None
        CaMemAllocator.npu_recover_queue = None
        CaMemAllocator.ctrl_queue = None
        CaMemAllocator.layer_ready_events = None
        with patch("vllm_ascend.device_allocator.camem.torch.npu.synchronize"):
            yield
        CaMemAllocator.instance = previous_instance
        CaMemAllocator.pipeline_switch = previous_switch
        (
            CaMemAllocator.desc_queue,
            CaMemAllocator.npu_recover_queue,
            CaMemAllocator.ctrl_queue,
            CaMemAllocator.layer_ready_events,
        ) = previous_resources

    @staticmethod
    def shared_allocator():
        CaMemAllocator.set_pipeline_switch(True)
        return CaMemAllocator.get_instance()

    def test_copier_compatibility_aliases(self):
        assert copier_malloc_use_share is copy_malloc_use_share
        assert copier_free is copy_free

    def test_find_loaded_library_success_and_not_found(self):
        path = find_loaded_library("libc")
        assert path is not None, "Expected to find libc library"
        assert path.endswith(".so.6") or ".so" in path
        assert "libc" in path

        path = find_loaded_library("non_existent_library")
        assert path is None, "Expected to not find non-existent library"

    def test_create_and_map_share_returns_refreshed_share_handle(self):
        sleeping_handle = (1, 2, 3, 4, 55)
        active_handle = (1, 2, 3, 4, 99)
        with patch(
            "vllm_ascend.device_allocator.camem.python_create_and_map_share",
            return_value=99,
        ) as mock_create:
            assert create_and_map_share(sleeping_handle) == active_handle
            mock_create.assert_called_once_with(*sleeping_handle)

    def test_legacy_create_and_release_preserve_four_field_handle(self):
        handle = (1, 2, 3, 4)
        with (
            patch("vllm_ascend.device_allocator.camem.python_create_and_map", return_value=None) as mock_create,
            patch("vllm_ascend.device_allocator.camem.python_unmap_and_release", return_value=None) as mock_release,
        ):
            assert create_and_map(handle) is None
            assert unmap_and_release(handle) is None
            mock_create.assert_called_once_with(*handle)
            mock_release.assert_called_once_with(*handle)

    @pytest.mark.parametrize("handle", [None, 1, "handle", (), (1, 2, 3), (1, 2, 3, 4, 5)])
    def test_create_and_map_rejects_invalid_input_handle(self, handle):
        with pytest.raises(RuntimeError, match="invalid allocation handle"):
            create_and_map(handle)

    def test_share_alloc_helpers_use_five_field_handle(self):
        active_handle = (1, 2, 3, 4, 99)
        with (
            patch("vllm_ascend.device_allocator.camem.python_create_and_map_share_alloc") as mock_create,
            patch("vllm_ascend.device_allocator.camem.python_unmap_and_release_share_alloc") as mock_release,
        ):
            create_and_map_share_alloc(active_handle)
            unmap_and_release_share_alloc(active_handle)
        mock_create.assert_called_once_with(*active_handle)
        mock_release.assert_called_once_with(*active_handle)

    @patch("vllm_ascend.device_allocator.camem.init_module")
    @patch("vllm_ascend.device_allocator.camem.torch.npu.memory.NPUPluggableAllocator")
    def test_get_pluggable_allocator(self, mock_allocator_class, mock_init_module):
        mock_allocator_instance = MagicMock()
        mock_allocator_class.return_value = mock_allocator_instance

        def side_effect_malloc_and_free(malloc_fn, free_fn):
            malloc_fn((1, 2, 3, 4))
            free_fn(123)

        mock_init_module.side_effect = side_effect_malloc_and_free

        allocator = get_pluggable_allocator(dummy_malloc, dummy_free)
        mock_init_module.assert_called_once_with(dummy_malloc, dummy_free)
        assert allocator == mock_allocator_instance

    @patch("vllm_ascend.device_allocator.camem.init_module_share")
    @patch("vllm_ascend.device_allocator.camem.torch.npu.memory.NPUPluggableAllocator")
    def test_get_pluggable_allocator_enables_five_field_mode(self, mock_allocator_class, mock_init_module_share):
        get_pluggable_allocator(
            dummy_malloc,
            dummy_free,
            enable_share_handle=True,
        )
        mock_init_module_share.assert_called_once_with(dummy_malloc, dummy_free)

    def test_singleton_behavior(self):
        instance1 = CaMemAllocator.get_instance()
        instance2 = CaMemAllocator.get_instance()
        assert instance1 is instance2
        CaMemAllocator.set_pipeline_switch(True)
        assert CaMemAllocator.get_instance().multiproc_pipe_enabled

        queues = (MagicMock(), MagicMock(), MagicMock())
        events = {0: MagicMock()}
        CaMemAllocator.set_desc_queue(queues[0])
        CaMemAllocator.set_npu_recover_queue(queues[1])
        CaMemAllocator.set_ctrl_queue(queues[2])
        CaMemAllocator.set_layer_ready_events(events)
        copier = MagicMock(
            device=0,
            desc_queue=queues[0],
            npu_recover_queue=queues[1],
            ctrl_queue=queues[2],
            layer_ready_events=events,
        )
        copier.start.return_value = 123
        with patch("vllm_ascend.device_allocator.camem.CopierProcess", return_value=copier) as process:
            instance1.start_pipeline(0, 1, events)
        process.assert_called_once_with(
            device=0,
            num_layers=1,
            layer_ready_events=events,
            tp_size=1,
            local_rank=0,
            queues=queues,
        )

    def test_pipeline_switch_rejects_non_boolean(self):
        with pytest.raises(TypeError, match="must be a boolean"):
            CaMemAllocator.set_pipeline_switch(1)

    def test_python_malloc_and_free_callback(self):
        allocator = CaMemAllocator.get_instance()
        handle = (1, 100, 1234, 4321)
        allocator.current_tag = "test_tag"

        allocator.python_malloc_callback(handle)
        ptr = handle[2]
        assert ptr in allocator.pointer_to_data
        data = allocator.pointer_to_data[ptr]
        assert data.handle == handle
        assert data.tag == "test_tag"

        data.cpu_backup_tensor = torch.zeros(1)
        result_handle = allocator.python_free_callback(ptr)
        assert result_handle == handle
        assert ptr not in allocator.pointer_to_data
        assert data.cpu_backup_tensor is None

    def test_python_malloc_callbacks_reject_wrong_abi_and_duplicates(self):
        allocator = self.shared_allocator()
        with pytest.raises(RuntimeError, match="invalid allocation handle"):
            allocator.python_malloc_callback((1, 100, 1234, 4321, 5678))
        with pytest.raises(RuntimeError, match="missing its share handle"):
            allocator.python_malloc_share_callback((1, 100, 1234, 4321, 0))

        handle = (1, 100, 1234, 4321, 5678)
        allocator.python_malloc_share_callback(handle)
        with pytest.raises(RuntimeError, match="already tracked"):
            allocator.python_malloc_share_callback(handle)

    def test_allocator_tracks_and_frees_four_and_five_field_abis_separately(self):
        allocator = CaMemAllocator.get_instance()
        shared_handle = (1, 100, 1234, 4321, 5678)
        legacy_handle = (1, 100, 2234, 5321)
        allocator.python_malloc_share_callback(shared_handle)
        allocator.python_malloc_callback(legacy_handle)
        assert len(allocator.pointer_to_data[1234].handle) == 5
        assert len(allocator.pointer_to_data[2234].handle) == 4
        with pytest.raises(RuntimeError, match="does not match"):
            allocator.python_free_callback(1234)
        with pytest.raises(RuntimeError, match="does not match"):
            allocator.python_free_share_callback(2234)
        assert allocator.python_free_share_callback(1234) == shared_handle
        assert allocator.python_free_callback(2234) == legacy_handle

    @patch("vllm_ascend.device_allocator.camem.unmap_and_release_share_alloc")
    @patch("vllm_ascend.device_allocator.camem.python_memcpy_device_to_host")
    def test_sleep_offload_and_discard(self, mock_memcpy, mock_unmap):
        allocator = self.shared_allocator()
        handle1 = (1, 10, 1000, 1100, 1200)
        data1 = AllocationData(handle1, "tag1")
        handle2 = (2, 20, 2000, 2100, 2200)
        data2 = AllocationData(handle2, "tag2")
        allocator.pointer_to_data = {1000: data1, 2000: data2}
        allocator._pipeline_initialized = True
        allocator._copier = MagicMock()
        original_torch_empty = torch.empty

        def mock_torch_empty(*args, **kwargs):
            if kwargs.get("pin_memory") is True:
                kwargs["pin_memory"] = False
            return original_torch_empty(*args, **kwargs)

        with patch("vllm_ascend.device_allocator.camem.torch.empty", side_effect=mock_torch_empty):
            allocator.sleep(offload_tags="tag1")

        assert data1.cpu_backup_tensor is not None
        assert data2.cpu_backup_tensor is None
        assert data1.handle == handle1
        assert data2.handle == handle2
        mock_unmap.assert_any_call(handle1)
        mock_unmap.assert_any_call(handle2)
        assert mock_unmap.call_count == 2
        assert mock_memcpy.called
        allocator._copier.suspend.assert_called_once_with()

    @patch("vllm_ascend.device_allocator.camem.create_and_map_share_alloc")
    @patch("vllm_ascend.device_allocator.camem.python_memcpy_host_to_device")
    def test_wake_up_loads_and_clears_cpu_backup(self, mock_memcpy, mock_create_and_map):
        allocator = self.shared_allocator()
        sleeping_handle = (1, 10, 1000, 1100, 0)
        tensor = torch.zeros(5, dtype=torch.uint8)
        data = AllocationData(sleeping_handle, "tag1", cpu_backup_tensor=tensor)
        allocator.pointer_to_data = {1000: data}
        allocator._cycle_state = "sync_sleeping"
        allocator._sync_sleeping_tags = {"tag1"}
        allocator.wake_up(tags=["tag1"])

        mock_create_and_map.assert_called_once_with(sleeping_handle)
        assert data.handle == sleeping_handle
        assert data.cpu_backup_tensor is None
        assert mock_memcpy.called

    def test_use_memory_pool_context_manager(self):
        allocator = CaMemAllocator.get_instance()
        old_tag = allocator.current_tag
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = "data"
        mock_ctx.__exit__.return_value = None

        with patch(
            "vllm_ascend.device_allocator.camem.use_memory_pool_with_allocator",
            return_value=mock_ctx,
        ) as use_pool:
            with allocator.use_memory_pool(tag="my_tag"):
                assert allocator.current_tag == "my_tag"
            assert allocator.current_tag == old_tag
            use_pool.assert_called_once_with(
                allocator.python_malloc_callback,
                allocator.python_free_callback,
                False,
            )

            allocator.enable_share_handle = True
            allocator._copier = MagicMock()
            with allocator.use_memory_pool_share(tag="weights"):
                assert allocator.current_tag == "weights"
            assert allocator.current_tag == old_tag
            assert use_pool.call_args_list[-1] == call(
                allocator.python_malloc_share_callback,
                allocator.python_free_share_callback,
                True,
            )

    def test_use_memory_pool_restores_tag_after_failure(self):
        allocator = CaMemAllocator.get_instance()
        old_tag = allocator.current_tag
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = "data"
        mock_ctx.__exit__.return_value = None

        with (
            patch(
                "vllm_ascend.device_allocator.camem.use_memory_pool_with_allocator",
                return_value=mock_ctx,
            ),
            pytest.raises(RuntimeError, match="allocation failed"),
            allocator.use_memory_pool(tag="weights"),
        ):
            raise RuntimeError("allocation failed")

        assert allocator.current_tag == old_tag

        allocator.enable_share_handle = True
        copier = allocator._copier = MagicMock()
        with (
            patch(
                "vllm_ascend.device_allocator.camem.use_memory_pool_with_allocator",
                return_value=mock_ctx,
            ),
            pytest.raises(WorkerFatalError, match="memory-pool transaction"),
            allocator.use_memory_pool_share(tag="weights"),
        ):
            raise RuntimeError("allocation failed")

        assert allocator.current_tag == old_tag
        assert allocator._poisoned
        assert allocator._copier is None
        copier.close.assert_called_once_with()

    def test_get_current_usage(self):
        allocator = CaMemAllocator.get_instance()
        allocator.pointer_to_data = {
            1: AllocationData((0, 100, 1, 10), "tag"),
            2: AllocationData((0, 200, 2, 20), "tag"),
        }

        assert allocator.get_current_usage() == 300

    def test_build_weight_descriptors_groups_known_layers_once(self):
        from vllm.model_executor.model_loader.base_loader import layer_to_addr

        allocator = self.shared_allocator()
        allocator._num_layers = 2
        allocator.pointer_to_data = {
            1000: AllocationData((0, 100, 1000, 10, 101), "weights"),
            2000: AllocationData((0, 100, 2000, 20, 102), "weights"),
            3000: AllocationData((0, 100, 3000, 30, 103), "weights"),
            4000: AllocationData((0, 100, 4000, 40, 104), "weights"),
        }

        layer_to_addr.clear()
        layer_to_addr.update(
            {
                "unknown": [],
                "pub": [1010],
                "layers.0": [2010],
                "layers.1": [3010],
            }
        )

        descriptors = allocator._build_weight_descriptors()

        assert [descriptor.layer_name for descriptor in descriptors] == [
            "unknown",
            "pub",
            "layers.0",
            "layers.1",
        ]
        assert allocator.layer_to_addr == {
            "unknown": [4000],
            "pub": [1000],
            "layers.0": [2000],
            "layers.1": [3000],
        }

    @patch("vllm_ascend.device_allocator.camem.create_and_map_share")
    @patch("vllm_ascend.device_allocator.camem.create_and_map")
    @patch("vllm_ascend.device_allocator.camem.unmap_and_release_share_alloc")
    @patch("vllm_ascend.device_allocator.camem.unmap_and_release")
    def test_suspend_resume_streams_layer_descriptors(
        self,
        mock_unmap,
        mock_unmap_share,
        mock_create,
        mock_create_share,
    ):
        allocator = self.shared_allocator()
        allocator._pipeline_initialized = True
        allocator._num_layers = 2
        allocator._copier = MagicMock()
        allocator._copier_tgid = 789
        allocator.layer_to_addr = {
            "unknown": [],
            "pub": [1000],
            "layers.0": [2000],
            "layers.1": [3000],
        }
        allocator.pointer_to_data = {
            1000: AllocationData((0, 100, 1000, 10, 101), "weights"),
            2000: AllocationData((0, 100, 2000, 20, 102), "weights"),
            3000: AllocationData((0, 100, 3000, 30, 103), "weights"),
            4000: AllocationData((0, 100, 4000, 40), "kv_cache"),
        }
        mock_create_share.side_effect = lambda handle: (*handle[:4], handle[2] + 10_000)

        allocator.suspend()
        allocator.resume()

        assert not allocator.ready
        assert allocator._cycle_state == "resuming"
        allocator._copier.suspend.assert_called_once_with()
        allocator._copier.begin_resume.assert_called_once_with()
        allocator._copier.finish_resume.assert_called_once_with()
        assert allocator._copier.send_recovery_descriptor.call_count == 4
        allocator._copier.wait_for_layer.assert_called_once_with(0)
        assert mock_create_share.call_count == 3
        mock_create.assert_called_once_with((0, 100, 4000, 40))
        assert mock_unmap_share.call_count == 3
        mock_unmap.assert_called_once_with((0, 100, 4000, 40))

        # A model can be handed off again without receiving a request after
        # resume. Suspend must finish the background restore in that case.
        allocator.suspend()
        assert allocator._cycle_state == "suspended"
        assert allocator._copier.wait_for_layer.call_args_list == [call(0), call(1)]
        assert allocator._copier.suspend.call_count == 2
        allocator.resume()
        assert allocator._copier.wait_for_layer.call_args_list == [
            call(0),
            call(1),
            call(0),
        ]

        allocator.wait_for_layer(1)
        allocator.finish_forward()
        assert allocator.ready
        assert allocator._cycle_state == "active"

    def test_resume_layer_zero_failure_fail_stops_worker(self):
        allocator = self.shared_allocator()
        allocator._pipeline_initialized = True
        allocator._num_layers = 1
        allocator._cycle_state = "suspended"
        allocator._ready = False
        allocator._copier_tgid = 789
        copier = MagicMock()
        allocator._copier = copier
        copier.wait_for_layer.side_effect = TimeoutError("layer zero timed out")
        allocator.layer_to_addr = {"unknown": [], "pub": [], "layers.0": []}

        with pytest.raises(WorkerFatalError, match="layer zero timed out"):
            allocator.resume()

        copier.wait_for_layer.assert_called_once_with(0)
        copier.close.assert_called_once_with()
        assert allocator._copier is None
        assert allocator._cycle_state == "poisoned"
        assert allocator._poisoned
        assert not allocator.ready

    def test_suspend_partial_unmap_failure_fail_stops_worker(self):
        allocator = self.shared_allocator()
        allocator._pipeline_initialized = True
        allocator._num_layers = 1
        copier = MagicMock()
        allocator._copier = copier
        allocator.pointer_to_data = {
            1000: AllocationData((0, 100, 1000, 10, 101), "weights"),
            2000: AllocationData((0, 100, 2000, 20, 102), "weights"),
        }

        with (
            patch.object(
                allocator,
                "_unmap_allocation",
                side_effect=[None, RuntimeError("second unmap failed")],
            ),
            pytest.raises(WorkerFatalError, match="second unmap failed"),
        ):
            allocator.suspend()

        copier.suspend.assert_called_once_with()
        copier.close.assert_called_once_with()
        assert allocator._cycle_state == "poisoned"
        assert allocator._poisoned
        assert not allocator.ready

    def test_resume_partial_map_failure_fail_stops_worker(self):
        allocator = self.shared_allocator()
        allocator._pipeline_initialized = True
        allocator._num_layers = 1
        allocator._cycle_state = "suspended"
        allocator._ready = False
        allocator._copier_tgid = 789
        copier = MagicMock()
        allocator._copier = copier
        allocator.layer_to_addr = {"unknown": [], "pub": [1000], "layers.0": [2000]}
        allocator.pointer_to_data = {
            1000: AllocationData((0, 100, 1000, 10, 101), "weights"),
            2000: AllocationData((0, 100, 2000, 20, 102), "weights"),
        }

        with (
            patch(
                "vllm_ascend.device_allocator.camem.create_and_map_share",
                side_effect=[
                    (0, 100, 1000, 10, 201),
                    RuntimeError("second map failed"),
                ],
            ),
            pytest.raises(WorkerFatalError, match="second map failed"),
        ):
            allocator.resume()

        copier.abort_resume.assert_called_once_with()
        copier.close.assert_called_once_with()
        assert allocator.pointer_to_data[1000].handle == (0, 100, 1000, 10, 201)
        assert allocator._cycle_state == "poisoned"
        assert allocator._poisoned

    def test_synchronous_sleep_partial_unmap_failure_fail_stops_worker(self):
        allocator = CaMemAllocator.get_instance()
        allocator.pointer_to_data = {
            1000: AllocationData((0, 100, 1000, 10), "weights"),
            2000: AllocationData((0, 100, 2000, 20), "kv_cache"),
        }

        with (
            patch.object(
                allocator,
                "_unmap_allocation",
                side_effect=[None, RuntimeError("second synchronous unmap failed")],
            ),
            pytest.raises(WorkerFatalError, match="second synchronous unmap failed"),
        ):
            allocator.sleep(offload_tags=())

        assert allocator._cycle_state == "poisoned"
        assert allocator._poisoned

    def test_synchronous_wake_partial_map_failure_fail_stops_worker(self):
        allocator = CaMemAllocator.get_instance()
        allocator._cycle_state = "sync_sleeping"
        allocator._sync_sleeping_tags = {"weights", "kv_cache"}
        allocator.pointer_to_data = {
            1000: AllocationData((0, 100, 1000, 10), "weights"),
            2000: AllocationData((0, 100, 2000, 20), "kv_cache"),
        }

        with (
            patch.object(
                allocator,
                "_map_without_copier",
                side_effect=[None, RuntimeError("second synchronous map failed")],
            ),
            pytest.raises(WorkerFatalError, match="second synchronous map failed"),
        ):
            allocator.wake_up()

        assert allocator._cycle_state == "poisoned"
        assert allocator._poisoned

    def test_allocator_rejects_crossed_cycle_pairs(self):
        allocator = self.shared_allocator()
        allocator._pipeline_initialized = True
        allocator._copier = MagicMock()
        allocator._cycle_state = "suspended"

        with pytest.raises(RuntimeError, match="completed with resume"):
            allocator.wake_up()

        allocator._cycle_state = "sync_sleeping"
        with pytest.raises(RuntimeError, match="completed with wake_up"):
            allocator.resume()
