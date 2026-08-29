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

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest

_HUGE_2M = 2 << 20
_PAGE_TYPE_HUGE_2M = 2
_FAKE_SHAREABLE_OFFSET = 10_000


@pytest.fixture()
def native_allocator(tmp_path: Path) -> tuple[ctypes.CDLL, ctypes.CDLL]:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the native Famem allocator test.")
    repository = Path(__file__).resolve().parents[3]
    fake_acl = Path(__file__).with_name("fake_acl")
    library = tmp_path / "libvllm_ascend_famem_test.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fPIC",
            "-shared",
            "-Wl,-Bsymbolic",
            "-I",
            str(fake_acl),
            str(repository / "csrc/famem_allocator.cpp"),
            str(fake_acl / "fake_acl.cpp"),
            "-ldl",
            "-pthread",
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    native = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    _declare_signatures(native)
    observations = native
    _declare_observation_signatures(observations)
    return native, observations


def _declare_signatures(native: ctypes.CDLL) -> None:
    uint64_pointer = ctypes.POINTER(ctypes.c_uint64)
    int32_pointer = ctypes.POINTER(ctypes.c_int32)

    native.famem_last_error.argtypes = []
    native.famem_last_error.restype = ctypes.c_char_p
    native.famem_server_initialize.argtypes = [ctypes.c_int]
    native.famem_server_initialize.restype = ctypes.c_int
    native.famem_server_allocate_export.argtypes = [
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_int,
        uint64_pointer,
        uint64_pointer,
    ]
    native.famem_server_allocate_export.restype = ctypes.c_int
    native.famem_server_authorize.argtypes = [ctypes.c_uint64, int32_pointer, ctypes.c_size_t]
    native.famem_server_authorize.restype = ctypes.c_int
    native.famem_server_free.argtypes = [ctypes.c_uint64]
    native.famem_server_free.restype = ctypes.c_int
    native.famem_server_finalize.argtypes = []
    native.famem_server_finalize.restype = ctypes.c_int
    native.famem_worker_prepare_v2.argtypes = [
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_size_t,
        int32_pointer,
        uint64_pointer,
        uint64_pointer,
        uint64_pointer,
    ]
    native.famem_worker_prepare_v2.restype = ctypes.c_int
    native.famem_worker_unmap.argtypes = [ctypes.c_int]
    native.famem_worker_unmap.restype = ctypes.c_int
    native.famem_worker_remap_v2.argtypes = [
        ctypes.c_int,
        ctypes.c_size_t,
        int32_pointer,
        uint64_pointer,
        uint64_pointer,
    ]
    native.famem_worker_remap_v2.restype = ctypes.c_int
    native.famem_worker_release.argtypes = [ctypes.c_int]
    native.famem_worker_release.restype = ctypes.c_int
    native.famem_worker_get_stats.argtypes = [
        uint64_pointer,
        uint64_pointer,
        uint64_pointer,
        uint64_pointer,
        uint64_pointer,
        ctypes.POINTER(ctypes.c_int),
    ]
    native.famem_worker_get_stats.restype = ctypes.c_int
    native.famem_malloc.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    native.famem_malloc.restype = ctypes.c_void_p
    native.famem_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    native.famem_free.restype = None


def _declare_observation_signatures(observations: ctypes.CDLL) -> None:
    observations.fake_acl_reset_observations.argtypes = []
    observations.fake_acl_reset_observations.restype = None
    observations.fake_acl_fail_set_pid.argtypes = [ctypes.c_int]
    observations.fake_acl_fail_set_pid.restype = None
    observations.fake_acl_fail_export.argtypes = [ctypes.c_int]
    observations.fake_acl_fail_export.restype = None
    observations.fake_acl_fail_map.argtypes = [ctypes.c_int]
    observations.fake_acl_fail_map.restype = None
    observations.fake_acl_fail_free.argtypes = [ctypes.c_int]
    observations.fake_acl_fail_free.restype = None
    observations.fake_acl_export_call_count.argtypes = []
    observations.fake_acl_export_call_count.restype = ctypes.c_size_t
    observations.fake_acl_set_pid_call_count.argtypes = []
    observations.fake_acl_set_pid_call_count.restype = ctypes.c_size_t
    observations.fake_acl_malloc_physical_call_count.argtypes = []
    observations.fake_acl_malloc_physical_call_count.restype = ctypes.c_size_t
    observations.fake_acl_free_physical_call_count.argtypes = []
    observations.fake_acl_free_physical_call_count.restype = ctypes.c_size_t
    observations.fake_acl_last_freed_handle.argtypes = []
    observations.fake_acl_last_freed_handle.restype = ctypes.c_uint64
    observations.fake_acl_live_handle_count.argtypes = []
    observations.fake_acl_live_handle_count.restype = ctypes.c_size_t
    observations.fake_acl_last_set_pid_handle.argtypes = []
    observations.fake_acl_last_set_pid_handle.restype = ctypes.c_uint64
    observations.fake_acl_last_target_count.argtypes = []
    observations.fake_acl_last_target_count.restype = ctypes.c_size_t
    observations.fake_acl_last_target.argtypes = [ctypes.c_size_t]
    observations.fake_acl_last_target.restype = ctypes.c_int32


def _last_error(native: ctypes.CDLL) -> str:
    return native.famem_last_error().decode("utf-8")


def _worker_stats(native: ctypes.CDLL) -> tuple[int, ...]:
    values = [ctypes.c_uint64() for _ in range(5)]
    state = ctypes.c_int()
    assert native.famem_worker_get_stats(
        *(ctypes.byref(value) for value in values), ctypes.byref(state)
    ) == 0
    return *(value.value for value in values), state.value


def test_native_allocation_rollback_failure_is_tracked_until_finalize(
    native_allocator: tuple[ctypes.CDLL, ctypes.CDLL],
):
    native, observations = native_allocator
    observations.fake_acl_reset_observations()
    assert native.famem_server_initialize(0) == 0, _last_error(native)
    physical_handle = ctypes.c_uint64()
    shareable_handle = ctypes.c_uint64()

    observations.fake_acl_fail_export(1)
    observations.fake_acl_fail_free(1)
    assert (
        native.famem_server_allocate_export(
            0,
            _HUGE_2M,
            _PAGE_TYPE_HUGE_2M,
            ctypes.byref(physical_handle),
            ctypes.byref(shareable_handle),
        )
        == -1
    )
    assert observations.fake_acl_live_handle_count() == 1

    assert (
        native.famem_server_allocate_export(
            0,
            _HUGE_2M,
            _PAGE_TYPE_HUGE_2M,
            ctypes.byref(physical_handle),
            ctypes.byref(shareable_handle),
        )
        == -1
    )
    assert "poisoned" in _last_error(native)
    assert native.famem_server_finalize() == -1
    assert observations.fake_acl_live_handle_count() == 1

    observations.fake_acl_fail_free(0)
    assert native.famem_server_finalize() == 0, _last_error(native)
    assert observations.fake_acl_live_handle_count() == 0


def test_native_finalize_retains_failed_handles_for_retry(
    native_allocator: tuple[ctypes.CDLL, ctypes.CDLL],
):
    native, observations = native_allocator
    observations.fake_acl_reset_observations()
    assert native.famem_server_initialize(0) == 0, _last_error(native)
    physical_handle = ctypes.c_uint64()
    shareable_handle = ctypes.c_uint64()
    assert (
        native.famem_server_allocate_export(
            0,
            _HUGE_2M,
            _PAGE_TYPE_HUGE_2M,
            ctypes.byref(physical_handle),
            ctypes.byref(shareable_handle),
        )
        == 0
    ), _last_error(native)
    assert observations.fake_acl_live_handle_count() == 1

    observations.fake_acl_fail_free(1)
    assert native.famem_server_finalize() == -1
    assert "outstanding cleanup" in _last_error(native)
    assert observations.fake_acl_live_handle_count() == 1

    observations.fake_acl_fail_free(0)
    assert native.famem_server_finalize() == 0, _last_error(native)
    assert observations.fake_acl_live_handle_count() == 0


def test_native_resident_handle_survives_worker_unmap_remap_and_release(
    native_allocator: tuple[ctypes.CDLL, ctypes.CDLL],
):
    native, observations = native_allocator
    observations.fake_acl_reset_observations()
    assert native.famem_server_initialize(0) == 0, _last_error(native)
    targets = (ctypes.c_int32 * 2)(111, 222)
    physical_handle = ctypes.c_uint64()
    shareable_handle = ctypes.c_uint64()

    assert (
        native.famem_server_allocate_export(
            0,
            _HUGE_2M,
            _PAGE_TYPE_HUGE_2M,
            ctypes.byref(physical_handle),
            ctypes.byref(shareable_handle),
        )
        == 0
    ), _last_error(native)
    assert observations.fake_acl_malloc_physical_call_count() == 1
    assert observations.fake_acl_export_call_count() == 1
    assert observations.fake_acl_live_handle_count() == 1

    observations.fake_acl_fail_set_pid(1)
    assert native.famem_server_authorize(shareable_handle.value, targets, len(targets)) == -1
    assert observations.fake_acl_live_handle_count() == 1
    assert observations.fake_acl_free_physical_call_count() == 0

    # Each activation authorizes the same startup-exported token. It must not
    # allocate, export, or free the resident original handle.
    observations.fake_acl_fail_set_pid(0)
    assert native.famem_server_authorize(shareable_handle.value, targets, len(targets)) == 0, _last_error(native)
    assert native.famem_server_authorize(shareable_handle.value, targets, len(targets)) == 0, _last_error(native)
    assert observations.fake_acl_malloc_physical_call_count() == 1
    assert observations.fake_acl_export_call_count() == 1
    assert observations.fake_acl_set_pid_call_count() == 3
    assert observations.fake_acl_free_physical_call_count() == 0
    assert observations.fake_acl_last_set_pid_handle() == shareable_handle.value
    assert observations.fake_acl_last_target_count() == 2
    assert [observations.fake_acl_last_target(index) for index in range(2)] == [111, 222]

    page_types = (ctypes.c_int32 * 1)(_PAGE_TYPE_HUGE_2M)
    extent_sizes = (ctypes.c_uint64 * 1)(_HUGE_2M)
    extent_handles = (ctypes.c_uint64 * 1)(5001)
    base_address = ctypes.c_uint64()
    assert (
        native.famem_worker_prepare_v2(
            0,
            _HUGE_2M,
            1,
            page_types,
            extent_sizes,
            extent_handles,
            ctypes.byref(base_address),
        )
        == 0
    ), _last_error(native)
    assert base_address.value % _HUGE_2M == 0
    assert observations.fake_acl_live_handle_count() == 2  # Server original plus Worker import.

    initial_stats = _worker_stats(native)
    assert initial_stats == (_HUGE_2M, 0, 0, 0, base_address.value, 1)
    assert native.famem_malloc(ctypes.c_size_t(-1).value, 0, None) is None
    assert "arena exhausted" in _last_error(native)
    assert _worker_stats(native) == initial_stats

    first = native.famem_malloc(1, 0, None)
    second = native.famem_malloc(_HUGE_2M - 512, 0, None)
    assert first == base_address.value
    assert second == base_address.value + 512
    full_stats = _worker_stats(native)
    assert full_stats == (_HUGE_2M, _HUGE_2M, _HUGE_2M, 2, base_address.value, 1)

    assert native.famem_malloc(1, 0, None) is None
    assert "arena exhausted" in _last_error(native)
    assert _worker_stats(native) == full_stats

    native.famem_free(first, 1, 0, None)
    freed_stats = _worker_stats(native)
    assert freed_stats == (_HUGE_2M, _HUGE_2M, _HUGE_2M - 512, 2, base_address.value, 1)

    assert native.famem_worker_unmap(0) == 0, _last_error(native)
    assert observations.fake_acl_live_handle_count() == 1  # Imported handle freed; original remains resident.
    assert observations.fake_acl_free_physical_call_count() == 1
    assert observations.fake_acl_last_freed_handle() == 5001
    assert _worker_stats(native)[-1] == 2
    assert native.famem_malloc(1, 0, None) is None
    sleeping_stats = _worker_stats(native)
    assert sleeping_stats[:-1] == freed_stats[:-1]

    # A remap transaction that fails after import must release the imported
    # handle and return to the retryable sleeping state.
    observations.fake_acl_fail_map(1)
    failed_remap_handles = (ctypes.c_uint64 * 1)(5002)
    assert native.famem_worker_remap_v2(0, 1, page_types, extent_sizes, failed_remap_handles) == -1
    assert "aclrtMapMem" in _last_error(native)
    assert _worker_stats(native)[-1] == 2
    assert observations.fake_acl_live_handle_count() == 1
    assert observations.fake_acl_free_physical_call_count() == 2
    assert observations.fake_acl_last_freed_handle() == 5002
    observations.fake_acl_fail_map(0)

    remap_handles = (ctypes.c_uint64 * 1)(5002)
    assert (
        native.famem_worker_remap_v2(0, 1, page_types, extent_sizes, remap_handles) == 0
    ), _last_error(native)
    remapped_stats = _worker_stats(native)
    assert remapped_stats[4] == base_address.value
    assert remapped_stats[-1] == 1
    assert observations.fake_acl_live_handle_count() == 2
    assert observations.fake_acl_malloc_physical_call_count() == 1
    assert observations.fake_acl_export_call_count() == 1
    assert observations.fake_acl_free_physical_call_count() == 2
    assert native.famem_worker_release(0) == 0, _last_error(native)
    assert observations.fake_acl_live_handle_count() == 1
    assert observations.fake_acl_free_physical_call_count() == 3
    assert observations.fake_acl_last_freed_handle() == 5002

    # Only server finalization releases the original allocation.
    assert native.famem_server_finalize() == 0, _last_error(native)
    assert observations.fake_acl_live_handle_count() == 0
    assert observations.fake_acl_free_physical_call_count() == 4
    assert observations.fake_acl_last_freed_handle() == shareable_handle.value - _FAKE_SHAREABLE_OFFSET
