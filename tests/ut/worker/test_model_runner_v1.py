import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from vllm.model_executor.model_loader.base_loader import layer_to_addr
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

MIB = 1 << 20


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))

    def test_kv_cache_allocation_plan_includes_split_alignment_requests(self):
        runner = self._build_runner()
        runner.vllm_config.kv_transfer_config = object()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        tensor_size = kv_cache_spec.page_size_bytes * 2
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=tensor_size, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        plan = runner.get_kv_cache_allocation_plan(kv_cache_config)

        self.assertEqual(len(plan.requests), 2)
        self.assertEqual(sum(request.payload_bytes for request in plan.requests), tensor_size)
        self.assertEqual(sum(request.requested_bytes for request in plan.requests), tensor_size + 4 * MIB)
        self.assertEqual([request.alignment_bytes for request in plan.requests], [2 * MIB, 2 * MIB])

    def test_load_model_builds_multiproc_pipe_map_after_loader_returns(self):
        layer_to_addr.clear()
        self.addCleanup(layer_to_addr.clear)
        model = nn.Module()
        model.public_weight = nn.Parameter(torch.zeros(1))
        model.layers = nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)])

        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.model_config = SimpleNamespace(model="test", get_num_layers=lambda parallel_config: 2)
        runner.vllm_config = SimpleNamespace(
            additional_config={"multiproc_pipe": True},
            parallel_config=SimpleNamespace(enable_eplb=False),
        )
        runner.eplb_enable = False
        runner.dynamic_eplb = False
        runner.drafter = None
        runner.lora_config = None
        runner.compilation_config = SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False))

        profiler = MagicMock()
        profiler.consumed_memory = 0
        profiler.__enter__.return_value = profiler
        with (
            patch("vllm_ascend.worker.model_runner_v1.DeviceMemoryProfiler", return_value=profiler),
            patch("vllm_ascend.worker.model_runner_v1.get_model", return_value=model),
        ):
            runner.load_model()

        self.assertIn(model.public_weight.data_ptr(), layer_to_addr["pub"])
        for index, layer in enumerate(model.layers):
            self.assertTrue(
                {parameter.data_ptr() for parameter in layer.parameters()} <= set(layer_to_addr[f"layers.{index}"])
            )


if __name__ == "__main__":
    unittest.main()
