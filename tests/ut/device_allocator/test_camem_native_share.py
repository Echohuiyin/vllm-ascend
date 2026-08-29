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
import importlib.util
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest

_HUGE_2M = 2 << 20


@pytest.fixture()
def native_camem(tmp_path: Path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the native Camem share test.")
    repository = Path(__file__).resolve().parents[3]
    fake_acl = Path(__file__).with_name("fake_acl")
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    library = tmp_path / f"vllm_ascend_C{extension_suffix}"
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
            "-I",
            sysconfig.get_paths()["include"],
            str(repository / "csrc/camem_allocator.cpp"),
            str(fake_acl / "fake_acl.cpp"),
            "-ldl",
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    spec = importlib.util.spec_from_file_location("vllm_ascend_C", library)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    native.my_malloc.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    native.my_malloc.restype = ctypes.c_void_p
    native.my_free.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    native.my_free.restype = None
    native.my_malloc_share.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    native.my_malloc_share.restype = ctypes.c_void_p
    native.my_free_share.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    native.my_free_share.restype = None
    observations = native
    observations.fake_acl_reset_observations.argtypes = []
    observations.fake_acl_reset_observations.restype = None
    observations.fake_acl_fail_set_pid.argtypes = [ctypes.c_int]
    observations.fake_acl_fail_set_pid.restype = None
    observations.fake_acl_export_call_count.argtypes = []
    observations.fake_acl_export_call_count.restype = ctypes.c_size_t
    observations.fake_acl_set_pid_call_count.argtypes = []
    observations.fake_acl_set_pid_call_count.restype = ctypes.c_size_t
    observations.fake_acl_live_handle_count.argtypes = []
    observations.fake_acl_live_handle_count.restype = ctypes.c_size_t
    observations.fake_acl_last_set_pid_handle.argtypes = []
    observations.fake_acl_last_set_pid_handle.restype = ctypes.c_uint64
    observations.fake_acl_last_target_count.argtypes = []
    observations.fake_acl_last_target_count.restype = ctypes.c_size_t
    observations.fake_acl_last_target.argtypes = [ctypes.c_size_t]
    observations.fake_acl_last_target.restype = ctypes.c_int32
    return module, native, observations


def test_native_camem_exports_five_tuple_and_recreates_share_handle(native_camem):
    module, native, observations = native_camem
    allocations = {}

    def record_allocation(handle):
        allocations[handle[2]] = handle

    def release_allocation(pointer):
        return allocations.pop(pointer)

    module.init_module_share(record_allocation, release_allocation)
    observations.fake_acl_reset_observations()

    pointer = native.my_malloc_share(1, 0, None)
    assert pointer in allocations
    handle = allocations[pointer]
    assert len(handle) == 5
    assert handle[:3] == (0, _HUGE_2M, pointer)
    assert handle[3] != 0
    assert handle[4] != 0
    assert observations.fake_acl_export_call_count() == 1
    assert observations.fake_acl_set_pid_call_count() == 0

    assert module.python_unmap_and_release_share_alloc(*handle) is None
    assert observations.fake_acl_live_handle_count() == 0

    observations.fake_acl_reset_observations()
    observations.fake_acl_fail_set_pid(1)
    legacy_shareable_handle = module.python_create_and_map_share(*handle)
    legacy_handle = (*handle[:4], legacy_shareable_handle)
    assert legacy_shareable_handle != 0
    assert legacy_shareable_handle != handle[4]
    assert observations.fake_acl_export_call_count() == 1
    assert observations.fake_acl_set_pid_call_count() == 0
    assert module.python_unmap_and_release_share_alloc(*legacy_handle) is None
    assert observations.fake_acl_live_handle_count() == 0

    allocations[pointer] = legacy_handle
    native.my_free_share(pointer, 1, 0, None)
    assert allocations == {}
    assert observations.fake_acl_live_handle_count() == 0


def test_native_camem_keeps_legacy_and_share_callbacks_independent(native_camem):
    module, native, observations = native_camem
    local_allocations = {}
    shared_allocations = {}

    module.init_module(
        lambda handle: local_allocations.__setitem__(handle[2], handle),
        lambda pointer: local_allocations.pop(pointer),
    )
    module.init_module_share(
        lambda handle: shared_allocations.__setitem__(handle[2], handle),
        lambda pointer: shared_allocations.pop(pointer),
    )
    observations.fake_acl_reset_observations()
    observations.fake_acl_fail_set_pid(1)

    local_pointer = native.my_malloc(1, 0, None)
    shared_pointer = native.my_malloc_share(1, 0, None)
    assert len(local_allocations[local_pointer]) == 4
    assert len(shared_allocations[shared_pointer]) == 5
    assert observations.fake_acl_export_call_count() == 1
    assert observations.fake_acl_set_pid_call_count() == 0

    native.my_free(local_pointer, 1, 0, None)
    native.my_free_share(shared_pointer, 1, 0, None)
    assert local_allocations == {}
    assert shared_allocations == {}


def test_native_camem_exports_historical_copier_aliases(native_camem):
    module, _, _ = native_camem

    assert module.python_copier_malloc_use_share is not None
    assert module.python_copier_free is not None


def test_native_camem_copier_imports_and_releases_share_handle(native_camem):
    module, native, observations = native_camem
    allocations = {}

    module.init_module_share(
        lambda handle: allocations.__setitem__(handle[2], handle),
        lambda pointer: allocations.pop(pointer),
    )
    observations.fake_acl_reset_observations()

    pointer = native.my_malloc_share(1, 0, None)
    handle = allocations[pointer]
    imported_physical_handle = module.python_share_memHandle_import(handle[4], handle[0])
    assert imported_physical_handle != 0
    assert observations.fake_acl_live_handle_count() == 2
    assert module.python_share_memHandle_free(imported_physical_handle, handle[0]) is None
    assert observations.fake_acl_live_handle_count() == 1

    copier_pointer, copier_physical_handle = module.python_copy_malloc_use_share(handle[0], handle[1], 0, handle[4])
    assert copier_pointer != 0
    assert copier_physical_handle != 0
    assert observations.fake_acl_live_handle_count() == 2

    assert module.python_copy_free(handle[0], handle[1], copier_pointer, copier_physical_handle) is None
    assert observations.fake_acl_live_handle_count() == 1

    native.my_free_share(pointer, 1, 0, None)
    assert observations.fake_acl_live_handle_count() == 0
