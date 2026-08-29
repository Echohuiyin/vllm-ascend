/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdlib>
#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

extern "C" {

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <sys/types.h>
#include "acl/acl.h"

// Global references to Python callables. The extension owns these references
// because bound method objects passed by the allocator can otherwise expire
// while the native allocator still holds their addresses.
static PyObject* g_python_malloc_callback = nullptr;
static PyObject* g_python_free_callback = nullptr;
static PyObject* g_python_malloc_share_callback = nullptr;
static PyObject* g_python_free_share_callback = nullptr;

struct AllocationHandle {
  unsigned long long device;
  unsigned long long size;
  unsigned long long d_mem;
  unsigned long long p_mem_handle;
  unsigned long long share_handle;
};

static std::runtime_error acl_error(const char* operation, aclError error_code) {
  std::string message = std::string(operation) + " failed with acl error code: " + std::to_string(error_code);
  if (error_code == ACL_ERROR_RT_MEMORY_ALLOCATION) {
    message += " (OOM: Out of Memory, allocation failed)";
  }
  return std::runtime_error(message + " " + __FILE__);
}

static void ensure_context(unsigned long long device) {
  aclrtContext context = nullptr;
  aclError error_code = aclrtGetCurrentContext(&context);
  if (error_code != 0) {
    throw acl_error("aclrtGetCurrentContext", error_code);
  }
  if (context == nullptr) {
    error_code = aclrtCreateContext(&context, device);
    if (error_code != 0) {
      throw acl_error("aclrtCreateContext", error_code);
    }
    error_code = aclrtSetCurrentContext(context);
    if (error_code != 0) {
      throw acl_error("aclrtSetCurrentContext", error_code);
    }
  }
}

static aclrtPhysicalMemProp physical_memory_properties(unsigned long long device) {
  aclrtPhysicalMemProp properties = {};
  properties.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  properties.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  properties.memAttr = ACL_HBM_MEM_HUGE;
  properties.location.id = device;
  properties.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  properties.reserve = 0;
  return properties;
}

static void free_physical_after_failure(aclrtDrvMemHandle* p_mem_handle) noexcept {
  if (p_mem_handle == nullptr || *p_mem_handle == aclrtDrvMemHandle{}) {
    return;
  }
  aclError error_code = aclrtFreePhysical(*p_mem_handle);
  if (error_code != 0) {
    std::cerr << "Failed to roll back physical memory, acl error code: " << error_code << std::endl;
    return;
  }
  *p_mem_handle = aclrtDrvMemHandle{};
}

static void* load_acl_symbol(const char* name) {
  dlerror();
  void* symbol = dlsym(RTLD_DEFAULT, name);
  if (symbol == nullptr) {
    void* library = dlopen("libascendcl.so", RTLD_NOW | RTLD_GLOBAL);
    if (library != nullptr) {
      symbol = dlsym(library, name);
    }
  }
  if (symbol == nullptr) {
    throw std::runtime_error(std::string("CANN does not provide required symbol ") + name);
  }
  return symbol;
}

static unsigned long long export_share_handle(aclrtDrvMemHandle p_mem_handle, uint64_t flags = 1) {
  using ExportFunction = aclError (*)(aclrtDrvMemHandle, aclrtMemHandleType, uint64_t, uint64_t*);
  auto export_memory = reinterpret_cast<ExportFunction>(load_acl_symbol("aclrtMemExportToShareableHandle"));

  uint64_t share_handle = 0;
  aclError error_code = export_memory(p_mem_handle, ACL_MEM_HANDLE_TYPE_NONE, flags, &share_handle);
  if (error_code != 0) {
    throw acl_error("aclrtMemExportToShareableHandle", error_code);
  }
  if (share_handle == 0) {
    throw std::runtime_error("aclrtMemExportToShareableHandle returned an invalid zero handle");
  }
  return share_handle;
}

static unsigned long long create_and_map(unsigned long long device, size_t size, void* d_mem,
                                         aclrtDrvMemHandle* p_mem_handle, bool export_handle = false,
                                         uint64_t export_flags = 1) {
  ensure_context(device);
  aclrtPhysicalMemProp properties = physical_memory_properties(device);

  *p_mem_handle = aclrtDrvMemHandle{};
  aclError error_code = aclrtMallocPhysical(p_mem_handle, size, &properties, 0);
  if (error_code != 0) {
    throw acl_error("aclrtMallocPhysical", error_code);
  }

  uint64_t share_handle = 0;
  if (export_handle) {
    try {
      share_handle = export_share_handle(*p_mem_handle, export_flags);
    } catch (...) {
      free_physical_after_failure(p_mem_handle);
      throw;
    }
  }

  error_code = aclrtMapMem(d_mem, size, 0, *p_mem_handle, 0);
  if (error_code != 0) {
    free_physical_after_failure(p_mem_handle);
    throw acl_error("aclrtMapMem", error_code);
  }
  return share_handle;
}

static void unmap_and_release(unsigned long long device, void* d_mem, aclrtDrvMemHandle* p_mem_handle) {
  ensure_context(device);
  aclError error_code = aclrtUnmapMem(d_mem);
  if (error_code != 0) {
    throw acl_error("aclrtUnmapMem", error_code);
  }
  error_code = aclrtFreePhysical(*p_mem_handle);
  if (error_code != 0) {
    throw acl_error("aclrtFreePhysical", error_code);
  }
  *p_mem_handle = aclrtDrvMemHandle{};
}

static PyObject* create_allocation_tuple(const AllocationHandle& handle, bool shared) {
  if (!shared) {
    return Py_BuildValue("(KKKK)", handle.device, handle.size, handle.d_mem, handle.p_mem_handle);
  }
  return Py_BuildValue("(KKKKK)", handle.device, handle.size, handle.d_mem, handle.p_mem_handle, handle.share_handle);
}

static bool parse_allocation_tuple(PyObject* object, AllocationHandle* handle, bool shared) {
  if (object == nullptr) {
    return false;
  }
  Py_ssize_t expected_size = shared ? 5 : 4;
  if (!PyTuple_Check(object) || PyTuple_Size(object) != expected_size) {
    PyErr_Format(PyExc_TypeError, "Expected an allocation handle tuple of size %zd", expected_size);
    return false;
  }
  if (!shared) {
    handle->share_handle = 0;
    return PyArg_ParseTuple(object, "KKKK", &handle->device, &handle->size, &handle->d_mem, &handle->p_mem_handle) != 0;
  }
  return PyArg_ParseTuple(object, "KKKKK", &handle->device, &handle->size, &handle->d_mem, &handle->p_mem_handle,
                          &handle->share_handle) != 0;
}

static bool notify_python_allocation(const AllocationHandle& handle, PyObject* callback, bool shared) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* argument = create_allocation_tuple(handle, shared);
  if (argument == nullptr) {
    PyErr_Print();
    PyGILState_Release(gstate);
    return false;
  }

  PyObject* result = PyObject_CallFunctionObjArgs(callback, argument, nullptr);
  Py_DECREF(argument);
  if (result == nullptr) {
    PyErr_Print();
    PyGILState_Release(gstate);
    return false;
  }
  Py_DECREF(result);
  PyGILState_Release(gstate);
  return true;
}

static bool fetch_python_allocation(void* ptr, AllocationHandle* handle, PyObject* callback, bool shared) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* py_ptr = PyLong_FromUnsignedLongLong(reinterpret_cast<unsigned long long>(ptr));
  if (py_ptr == nullptr) {
    PyErr_Print();
    PyGILState_Release(gstate);
    return false;
  }

  PyObject* result = PyObject_CallFunctionObjArgs(callback, py_ptr, nullptr);
  Py_DECREF(py_ptr);
  bool parsed = parse_allocation_tuple(result, handle, shared);
  if (!parsed) {
    PyErr_Print();
  }
  Py_XDECREF(result);
  PyGILState_Release(gstate);
  return parsed;
}

static void release_virtual_address(void* d_mem) {
  aclError error_code = aclrtReleaseMemAddress(d_mem);
  if (error_code != 0) {
    throw acl_error("aclrtReleaseMemAddress", error_code);
  }
}

static void rollback_allocation(unsigned long long device, void* d_mem, aclrtDrvMemHandle* p_mem_handle,
                                bool mapped) noexcept {
  if (mapped) {
    try {
      unmap_and_release(device, d_mem, p_mem_handle);
    } catch (const std::exception& error) {
      std::cerr << "Failed to roll back mapped memory: " << error.what() << std::endl;
    }
  } else {
    free_physical_after_failure(p_mem_handle);
  }
  aclError error_code = aclrtReleaseMemAddress(d_mem);
  if (error_code != 0) {
    std::cerr << "Failed to roll back virtual address, acl error code: " << error_code << std::endl;
  }
  std::free(p_mem_handle);
}

// ---------------------------------------------------------------------------
// Exported allocator callbacks.

static void* allocator_malloc(ssize_t size, int device, aclrtStream stream, PyObject* callback, bool shared) {
  (void)stream;
  if (size <= 0) {
    return nullptr;
  }
  if (callback == nullptr) {
    throw std::runtime_error("Camem malloc failed: Python callback is not set");
  }

  ensure_context(device);
  aclrtPhysicalMemProp properties = physical_memory_properties(device);
  size_t granularity = 0;
  aclError error_code =
      aclrtMemGetAllocationGranularity(&properties, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity);
  if (error_code != 0) {
    throw acl_error("aclrtMemGetAllocationGranularity", error_code);
  }
  if (granularity == 0) {
    throw std::runtime_error("aclrtMemGetAllocationGranularity returned zero");
  }

  size_t requested_size = static_cast<size_t>(size);
  if (requested_size > std::numeric_limits<size_t>::max() - (granularity - 1)) {
    throw std::overflow_error("Camem allocation size overflow");
  }
  size_t aligned_size = ((requested_size + granularity - 1) / granularity) * granularity;

  void* d_mem = nullptr;
  error_code = aclrtReserveMemAddress(&d_mem, aligned_size, 0, nullptr, 0);
  if (error_code != 0) {
    throw acl_error("aclrtReserveMemAddress", error_code);
  }

  auto* p_mem_handle = static_cast<aclrtDrvMemHandle*>(std::malloc(sizeof(aclrtDrvMemHandle)));
  if (p_mem_handle == nullptr) {
    release_virtual_address(d_mem);
    throw std::bad_alloc();
  }
  *p_mem_handle = aclrtDrvMemHandle{};

  unsigned long long share_handle = 0;
  try {
    share_handle = create_and_map(device, aligned_size, d_mem, p_mem_handle, shared);
  } catch (...) {
    rollback_allocation(device, d_mem, p_mem_handle, false);
    throw;
  }

  AllocationHandle handle = {static_cast<unsigned long long>(device), static_cast<unsigned long long>(aligned_size),
                             reinterpret_cast<unsigned long long>(d_mem),
                             reinterpret_cast<unsigned long long>(p_mem_handle), share_handle};
  if (!notify_python_allocation(handle, callback, shared)) {
    rollback_allocation(device, d_mem, p_mem_handle, true);
    return nullptr;
  }
  return d_mem;
}

static void allocator_free(void* ptr, ssize_t size, int device, aclrtStream stream, PyObject* callback, bool shared) {
  (void)size;
  (void)stream;
  if (ptr == nullptr) {
    return;
  }
  if (callback == nullptr) {
    throw std::runtime_error("Camem free failed: Python callback is not set");
  }

  AllocationHandle handle = {};
  if (!fetch_python_allocation(ptr, &handle, callback, shared)) {
    throw std::runtime_error("Camem free failed: Python callback returned an invalid allocation handle");
  }
  if (handle.device != static_cast<unsigned long long>(device) ||
      handle.d_mem != reinterpret_cast<unsigned long long>(ptr) || handle.p_mem_handle == 0) {
    throw std::runtime_error("Camem free failed: allocation handle does not match the pointer or device");
  }

  auto* d_mem = reinterpret_cast<void*>(handle.d_mem);
  auto* p_mem_handle = reinterpret_cast<aclrtDrvMemHandle*>(handle.p_mem_handle);
  if (*p_mem_handle != aclrtDrvMemHandle{}) {
    unmap_and_release(handle.device, d_mem, p_mem_handle);
  }
  release_virtual_address(d_mem);
  std::free(p_mem_handle);
}

__attribute__((visibility("default"))) void* my_malloc(ssize_t size, int device, aclrtStream stream) {
  return allocator_malloc(size, device, stream, g_python_malloc_callback, false);
}

__attribute__((visibility("default"))) void my_free(void* ptr, ssize_t size, int device, aclrtStream stream) {
  allocator_free(ptr, size, device, stream, g_python_free_callback, false);
}

__attribute__((visibility("default"))) void* my_malloc_share(ssize_t size, int device, aclrtStream stream) {
  return allocator_malloc(size, device, stream, g_python_malloc_share_callback, true);
}

__attribute__((visibility("default"))) void my_free_share(void* ptr, ssize_t size, int device, aclrtStream stream) {
  allocator_free(ptr, size, device, stream, g_python_free_share_callback, true);
}

// ---------------------------------------------------------------------------
// Python extension boilerplate.

static bool install_callbacks(PyObject* malloc_callback, PyObject* free_callback, PyObject** malloc_slot,
                              PyObject** free_slot) {
  if (!PyCallable_Check(malloc_callback) || !PyCallable_Check(free_callback)) {
    PyErr_SetString(PyExc_TypeError, "Both arguments must be callables");
    return false;
  }
  Py_INCREF(malloc_callback);
  Py_INCREF(free_callback);
  Py_XDECREF(*malloc_slot);
  Py_XDECREF(*free_slot);
  *malloc_slot = malloc_callback;
  *free_slot = free_callback;
  return true;
}

static PyObject* py_init_module(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* malloc_callback = nullptr;
  PyObject* free_callback = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &malloc_callback, &free_callback)) {
    return nullptr;
  }
  if (!install_callbacks(malloc_callback, free_callback, &g_python_malloc_callback, &g_python_free_callback)) {
    return nullptr;
  }
  Py_RETURN_NONE;
}

static PyObject* py_init_module_share(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* malloc_callback = nullptr;
  PyObject* free_callback = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &malloc_callback, &free_callback)) {
    return nullptr;
  }
  if (!install_callbacks(malloc_callback, free_callback, &g_python_malloc_share_callback,
                         &g_python_free_share_callback)) {
    return nullptr;
  }
  Py_RETURN_NONE;
}

static void free_module(void* module) {
  (void)module;
  Py_CLEAR(g_python_malloc_callback);
  Py_CLEAR(g_python_free_callback);
  Py_CLEAR(g_python_malloc_share_callback);
  Py_CLEAR(g_python_free_share_callback);
}

static PyObject* python_unmap_and_release(PyObject* self, PyObject* args) {
  (void)self;
  AllocationHandle handle = {};
  if (!parse_allocation_tuple(args, &handle, false)) {
    return nullptr;
  }

  try {
    auto* d_mem = reinterpret_cast<void*>(handle.d_mem);
    auto* p_mem_handle = reinterpret_cast<aclrtDrvMemHandle*>(handle.p_mem_handle);
    unmap_and_release(handle.device, d_mem, p_mem_handle);
    Py_RETURN_NONE;
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

static PyObject* python_create_and_map(PyObject* self, PyObject* args) {
  (void)self;
  AllocationHandle handle = {};
  if (!parse_allocation_tuple(args, &handle, false)) {
    return nullptr;
  }

  try {
    auto* d_mem = reinterpret_cast<void*>(handle.d_mem);
    auto* p_mem_handle = reinterpret_cast<aclrtDrvMemHandle*>(handle.p_mem_handle);
    create_and_map(handle.device, handle.size, d_mem, p_mem_handle);
    Py_RETURN_NONE;
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

static PyObject* python_unmap_and_release_share_alloc(PyObject* self, PyObject* args) {
  (void)self;
  AllocationHandle handle = {};
  if (!parse_allocation_tuple(args, &handle, true)) {
    return nullptr;
  }
  try {
    auto* d_mem = reinterpret_cast<void*>(handle.d_mem);
    auto* p_mem_handle = reinterpret_cast<aclrtDrvMemHandle*>(handle.p_mem_handle);
    unmap_and_release(handle.device, d_mem, p_mem_handle);
    Py_RETURN_NONE;
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

static PyObject* python_create_and_map_share_alloc(PyObject* self, PyObject* args) {
  (void)self;
  AllocationHandle handle = {};
  if (!parse_allocation_tuple(args, &handle, true)) {
    return nullptr;
  }
  try {
    auto* d_mem = reinterpret_cast<void*>(handle.d_mem);
    auto* p_mem_handle = reinterpret_cast<aclrtDrvMemHandle*>(handle.p_mem_handle);
    create_and_map(handle.device, handle.size, d_mem, p_mem_handle);
    Py_RETURN_NONE;
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

static PyObject* python_create_and_map_share(PyObject* self, PyObject* args) {
  (void)self;
  AllocationHandle handle = {};
  if (!parse_allocation_tuple(args, &handle, true)) {
    return nullptr;
  }
  try {
    auto* d_mem = reinterpret_cast<void*>(handle.d_mem);
    auto* p_mem_handle = reinterpret_cast<aclrtDrvMemHandle*>(handle.p_mem_handle);
    // The fifth field is the previous export token. It is intentionally not
    // reused: resume creates new physical memory and exports a fresh token.
    handle.share_handle = create_and_map(handle.device, handle.size, d_mem, p_mem_handle, true);
    return PyLong_FromUnsignedLongLong(handle.share_handle);
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

static PyObject* python_copy_malloc_use_share(PyObject* self, PyObject* args) {
  (void)self;
  unsigned long long device = 0;
  unsigned long long raw_size = 0;
  unsigned long long ignored_address = 0;
  unsigned long long share_handle = 0;
  if (!PyArg_ParseTuple(args, "KKKK", &device, &raw_size, &ignored_address, &share_handle)) {
    return nullptr;
  }
  if (device > static_cast<unsigned long long>(std::numeric_limits<int32_t>::max()) || raw_size == 0 ||
      raw_size > std::numeric_limits<size_t>::max() || share_handle == 0) {
    PyErr_SetString(PyExc_ValueError, "Invalid shared Camem import arguments");
    return nullptr;
  }

  void* d_mem = nullptr;
  auto* p_mem_handle = static_cast<aclrtDrvMemHandle*>(std::malloc(sizeof(aclrtDrvMemHandle)));
  if (p_mem_handle == nullptr) {
    return PyErr_NoMemory();
  }
  *p_mem_handle = aclrtDrvMemHandle{};
  bool reserved = false;
  bool mapped = false;
  try {
    ensure_context(device);
    const size_t size = static_cast<size_t>(raw_size);
    aclError error_code = aclrtReserveMemAddress(&d_mem, size, 0, nullptr, 0);
    if (error_code != 0) {
      throw acl_error("aclrtReserveMemAddress(copier)", error_code);
    }
    reserved = true;

    using ImportFunction = aclError (*)(uint64_t, int32_t, aclrtDrvMemHandle*);
    auto import_memory = reinterpret_cast<ImportFunction>(load_acl_symbol("aclrtMemImportFromShareableHandle"));
    error_code = import_memory(share_handle, static_cast<int32_t>(device), p_mem_handle);
    if (error_code != 0) {
      throw acl_error("aclrtMemImportFromShareableHandle", error_code);
    }
    error_code = aclrtMapMem(d_mem, size, 0, *p_mem_handle, 0);
    if (error_code != 0) {
      throw acl_error("aclrtMapMem(copier)", error_code);
    }
    mapped = true;
    PyObject* result = Py_BuildValue("(KK)", reinterpret_cast<unsigned long long>(d_mem),
                                     reinterpret_cast<unsigned long long>(p_mem_handle));
    if (result == nullptr) {
      rollback_allocation(device, d_mem, p_mem_handle, true);
    }
    return result;
  } catch (const std::exception& error) {
    if (mapped) {
      aclrtUnmapMem(d_mem);
    }
    if (*p_mem_handle != aclrtDrvMemHandle{}) {
      aclrtFreePhysical(*p_mem_handle);
    }
    if (reserved) {
      aclrtReleaseMemAddress(d_mem);
    }
    std::free(p_mem_handle);
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

static PyObject* python_share_memHandle_import(PyObject* self, PyObject* args) {
  (void)self;
  unsigned long long share_handle = 0;
  int device = 0;
  if (!PyArg_ParseTuple(args, "Ki", &share_handle, &device)) {
    return nullptr;
  }
  if (share_handle == 0 || device < 0) {
    PyErr_SetString(PyExc_ValueError, "Invalid Camem share handle import arguments");
    return nullptr;
  }
  auto* p_mem_handle = static_cast<aclrtDrvMemHandle*>(std::malloc(sizeof(aclrtDrvMemHandle)));
  if (p_mem_handle == nullptr) {
    return PyErr_NoMemory();
  }
  *p_mem_handle = aclrtDrvMemHandle{};
  try {
    ensure_context(device);
    using ImportFunction = aclError (*)(uint64_t, int32_t, aclrtDrvMemHandle*);
    auto import_memory = reinterpret_cast<ImportFunction>(load_acl_symbol("aclrtMemImportFromShareableHandle"));
    aclError error_code = import_memory(share_handle, device, p_mem_handle);
    if (error_code != 0) {
      throw acl_error("aclrtMemImportFromShareableHandle", error_code);
    }
    return PyLong_FromUnsignedLongLong(reinterpret_cast<unsigned long long>(p_mem_handle));
  } catch (const std::exception& error) {
    std::free(p_mem_handle);
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

static PyObject* python_share_memHandle_free(PyObject* self, PyObject* args) {
  (void)self;
  unsigned long long raw_p_mem_handle = 0;
  int device = 0;
  if (!PyArg_ParseTuple(args, "Ki", &raw_p_mem_handle, &device)) {
    return nullptr;
  }
  if (raw_p_mem_handle == 0 || device < 0) {
    PyErr_SetString(PyExc_ValueError, "Invalid imported Camem physical handle");
    return nullptr;
  }
  auto* p_mem_handle = reinterpret_cast<aclrtDrvMemHandle*>(raw_p_mem_handle);
  try {
    ensure_context(device);
    aclError error_code = aclrtFreePhysical(*p_mem_handle);
    if (error_code != 0) {
      throw acl_error("aclrtFreePhysical(imported handle)", error_code);
    }
    std::free(p_mem_handle);
    Py_RETURN_NONE;
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

static PyObject* python_copy_free(PyObject* self, PyObject* args) {
  (void)self;
  unsigned long long device = 0;
  unsigned long long raw_size = 0;
  unsigned long long raw_d_mem = 0;
  unsigned long long raw_p_mem_handle = 0;
  if (!PyArg_ParseTuple(args, "KKKK", &device, &raw_size, &raw_d_mem, &raw_p_mem_handle)) {
    return nullptr;
  }
  if (raw_size == 0 || raw_d_mem == 0 || raw_p_mem_handle == 0) {
    PyErr_SetString(PyExc_ValueError, "Invalid copier Camem allocation handle");
    return nullptr;
  }
  try {
    auto* d_mem = reinterpret_cast<void*>(raw_d_mem);
    auto* p_mem_handle = reinterpret_cast<aclrtDrvMemHandle*>(raw_p_mem_handle);
    unmap_and_release(device, d_mem, p_mem_handle);
    release_virtual_address(d_mem);
    std::free(p_mem_handle);
    Py_RETURN_NONE;
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

static PyObject* python_memcpy_device_to_host(PyObject* self, PyObject* args) {
  (void)self;
  unsigned long long device = 0;
  unsigned long long host_address = 0;
  unsigned long long device_address = 0;
  unsigned long long raw_size = 0;
  if (!PyArg_ParseTuple(args, "KKKK", &device, &host_address, &device_address, &raw_size)) {
    return nullptr;
  }
  if (host_address == 0 || device_address == 0 || raw_size == 0 || raw_size > std::numeric_limits<size_t>::max()) {
    PyErr_SetString(PyExc_ValueError, "Invalid Camem device-to-host copy arguments");
    return nullptr;
  }
  try {
    ensure_context(device);
    const size_t size = static_cast<size_t>(raw_size);
    aclError error_code = aclrtMemcpy(reinterpret_cast<void*>(host_address), size,
                                      reinterpret_cast<void*>(device_address), size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (error_code != 0) {
      throw acl_error("aclrtMemcpy(device-to-host)", error_code);
    }
    Py_RETURN_NONE;
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

static PyObject* python_memcpy_host_to_device(PyObject* self, PyObject* args) {
  (void)self;
  unsigned long long device = 0;
  unsigned long long device_address = 0;
  unsigned long long host_address = 0;
  unsigned long long raw_size = 0;
  if (!PyArg_ParseTuple(args, "KKKK", &device, &device_address, &host_address, &raw_size)) {
    return nullptr;
  }
  if (host_address == 0 || device_address == 0 || raw_size == 0 || raw_size > std::numeric_limits<size_t>::max()) {
    PyErr_SetString(PyExc_ValueError, "Invalid Camem host-to-device copy arguments");
    return nullptr;
  }
  try {
    ensure_context(device);
    const size_t size = static_cast<size_t>(raw_size);
    aclError error_code = aclrtMemcpy(reinterpret_cast<void*>(device_address), size,
                                      reinterpret_cast<void*>(host_address), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (error_code != 0) {
      throw acl_error("aclrtMemcpy(host-to-device)", error_code);
    }
    Py_RETURN_NONE;
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

static PyObject* python_get_bare_tgid(PyObject* self, PyObject* args) {
  (void)self;
  unsigned long long device = 0;
  if (!PyArg_ParseTuple(args, "K", &device)) {
    return nullptr;
  }
  if (device > static_cast<unsigned long long>(std::numeric_limits<int32_t>::max())) {
    PyErr_SetString(PyExc_ValueError, "Camem device index is out of range");
    return nullptr;
  }
  try {
    ensure_context(device);
    using GetBareTgidFunction = aclError (*)(int32_t*);
    auto get_bare_tgid = reinterpret_cast<GetBareTgidFunction>(load_acl_symbol("aclrtDeviceGetBareTgid"));
    int32_t bare_tgid = 0;
    aclError error_code = get_bare_tgid(&bare_tgid);
    if (error_code != 0) {
      throw acl_error("aclrtDeviceGetBareTgid", error_code);
    }
    if (bare_tgid <= 0) {
      throw std::runtime_error("aclrtDeviceGetBareTgid returned a non-positive TGID");
    }
    return PyLong_FromLong(bare_tgid);
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

static PyMethodDef module_methods[] = {
    {"init_module", (PyCFunction)py_init_module, METH_VARARGS,
     "Initialize module with Python malloc and free callables."},
    {"init_module_share", (PyCFunction)py_init_module_share, METH_VARARGS,
     "Initialize the original five-field share-mode callbacks."},
    {"python_create_and_map", (PyCFunction)python_create_and_map, METH_VARARGS,
     "Create and map physical memory for a four-field allocation."},
    {"python_unmap_and_release", (PyCFunction)python_unmap_and_release, METH_VARARGS,
     "Unmap and release a four-field Camem allocation."},
    {"python_create_and_map_share_alloc", (PyCFunction)python_create_and_map_share_alloc, METH_VARARGS,
     "Map a five-field allocation without exporting a new handle."},
    {"python_unmap_and_release_share_alloc", (PyCFunction)python_unmap_and_release_share_alloc, METH_VARARGS,
     "Unmap and release a five-field allocation using its first four fields."},
    {"python_create_and_map_share", (PyCFunction)python_create_and_map_share, METH_VARARGS,
     "Recreate a five-field allocation and return its new shareable handle."},
    {"python_copy_malloc_use_share", (PyCFunction)python_copy_malloc_use_share, METH_VARARGS,
     "Import and map a Camem share handle in a copier process."},
    {"python_copier_malloc_use_share", (PyCFunction)python_copy_malloc_use_share, METH_VARARGS,
     "Compatibility alias for python_copy_malloc_use_share."},
    {"python_copy_free", (PyCFunction)python_copy_free, METH_VARARGS,
     "Release a copier process's imported Camem mapping."},
    {"python_copier_free", (PyCFunction)python_copy_free, METH_VARARGS, "Compatibility alias for python_copy_free."},
    {"python_share_memHandle_import", (PyCFunction)python_share_memHandle_import, METH_VARARGS,
     "Import a shareable handle without mapping it."},
    {"python_share_memHandle_free", (PyCFunction)python_share_memHandle_free, METH_VARARGS,
     "Release a physical handle returned by python_share_memHandle_import."},
    {"python_memcpy_device_to_host", (PyCFunction)python_memcpy_device_to_host, METH_VARARGS,
     "Copy Camem bytes from NPU to host."},
    {"python_memcpy_host_to_device", (PyCFunction)python_memcpy_host_to_device, METH_VARARGS,
     "Copy Camem bytes from host to NPU."},
    {"python_get_bare_tgid", (PyCFunction)python_get_bare_tgid, METH_VARARGS,
     "Return the current process's CANN bare TGID."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef camem_allocator_module = {
    PyModuleDef_HEAD_INIT,
    "camem_allocator",
    "CANN-mem-based allocator for NPUPluggableAllocator",
    -1,
    module_methods,
    nullptr,
    nullptr,
    nullptr,
    free_module,
};

PyMODINIT_FUNC PyInit_vllm_ascend_C(void) { return PyModule_Create(&camem_allocator_module); }
}  // extern "C"
