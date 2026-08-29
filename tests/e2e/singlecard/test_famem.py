#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from tests.e2e.conftest import VllmRunner
from tests.e2e.utils import fork_new_process_for_each_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = os.environ.get("FAMEM_E2E_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_2 = os.environ.get("FAMEM_E2E_MODEL_2", "Qwen/Qwen2.5-0.5B")
ARENA_SIZE_GIB = int(os.environ.get("FAMEM_E2E_SIZE_GIB", "4"))
PROMPTS = ["Explain virtual memory in one sentence."]
MAX_TOKENS = 8


@contextmanager
def _production_hbm_server() -> Iterator[int]:
    previous_socket_dir = os.environ.get("VLLM_ASCEND_FAMEM_SOCKET_DIR")
    with tempfile.TemporaryDirectory(prefix="famem-e2e-", dir="/tmp") as directory:
        root = Path(directory)
        socket_dir = root / "socket"
        socket_dir.mkdir(mode=0o750)
        log_path = root / "server.log"
        environment = os.environ.copy()
        environment["VLLM_ASCEND_FAMEM_SOCKET_DIR"] = str(socket_dir)
        with log_path.open("w+", encoding="utf-8") as log:
            body_completed = False
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vllm_ascend.device_allocator.hbm_server_launcher",
                    "--device",
                    "0",
                    "--size-gib",
                    str(ARENA_SIZE_GIB),
                    "--socket-dir",
                    str(socket_dir),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        log.flush()
                        server_log = log_path.read_text(encoding="utf-8")
                        raise RuntimeError(f"Famem production HBM server exited during startup:\n{server_log}")
                    if any(socket_dir.glob("*.sock")):
                        break
                    time.sleep(0.1)
                else:
                    raise TimeoutError("Timed out waiting for the Famem production HBM server socket")

                os.environ["VLLM_ASCEND_FAMEM_SOCKET_DIR"] = str(socket_dir)
                yield torch.npu.mem_get_info()[0]
                body_completed = True
            finally:
                exited_before_cleanup = body_completed and process.poll() is not None
                if previous_socket_dir is None:
                    os.environ.pop("VLLM_ASCEND_FAMEM_SOCKET_DIR", None)
                else:
                    os.environ["VLLM_ASCEND_FAMEM_SOCKET_DIR"] = previous_socket_dir
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                if exited_before_cleanup:
                    log.flush()
                    raise RuntimeError(
                        f"Famem production HBM server exited unexpectedly with {process.returncode}:\n"
                        f"{log_path.read_text(encoding='utf-8')}"
                    )


def _runner(model: str = MODEL) -> VllmRunner:
    return VllmRunner(
        model,
        enable_sleep_mode=True,
        enforce_eager=True,
        max_model_len=512,
        distributed_executor_backend="mp",
        additional_config={
            "multiproc_pipe": True,
            "use_fast_map_allocator": {"enabled": True, "size_gib": ARENA_SIZE_GIB},
        },
    )


def _assert_sleeping(model) -> None:
    stats = model.collective_rpc("get_famem_stats")
    assert stats and all(entry["state"] == "SLEEPING" for entry in stats)


@fork_new_process_for_each_test
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "0"})
def test_famem_lifecycle_and_multi_instance_handoff_e2e() -> None:
    assert MODEL_2 != MODEL, "Famem E2E requires checkpoints with different weight images"
    free_before_server = torch.npu.mem_get_info()[0]
    pool_bytes = ARENA_SIZE_GIB << 30
    tolerance = 256 << 20
    with _production_hbm_server() as free_with_resident_pool, _runner() as runner1:
        assert free_before_server - free_with_resident_pool >= pool_bytes - tolerance
        baseline1 = runner1.generate_greedy(PROMPTS, max_tokens=MAX_TOKENS)

        # Keep the synchronous compatibility path in the same hardware-heavy
        # scenario instead of starting another model and HBM server just for
        # sleep/wake coverage.
        runner1.model.sleep(level=1)
        _assert_sleeping(runner1.model)
        runner1.model.wake_up()
        assert runner1.generate_greedy(PROMPTS, max_tokens=MAX_TOKENS) == baseline1

        runner1.model.suspend()
        _assert_sleeping(runner1.model)
        assert torch.npu.mem_get_info()[0] <= free_before_server - pool_bytes + tolerance

        with _runner(MODEL_2) as runner2:
            baseline2 = runner2.generate_greedy(PROMPTS, max_tokens=MAX_TOKENS)
            runner2.model.suspend()
            _assert_sleeping(runner2.model)

            runner1.model.resume()
            restored1 = runner1.generate_greedy(PROMPTS, max_tokens=MAX_TOKENS)
            runner1.model.suspend()
            _assert_sleeping(runner1.model)

    assert torch.npu.mem_get_info()[0] - free_with_resident_pool >= pool_bytes - tolerance
    assert restored1 == baseline1
    assert baseline2 != baseline1
