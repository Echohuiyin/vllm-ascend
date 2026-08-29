// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "acl/acl.h"

namespace {

constexpr size_t kAllocationAlignment = 512;
constexpr size_t kMaxExtentCount = 2;
constexpr size_t kHuge1GGranularity = size_t{1} << 30;
constexpr size_t kHuge2MGranularity = size_t{2} << 20;
constexpr uint64_t kHugePageReserveFlag = 1;
static_assert(sizeof(void*) <= sizeof(uint64_t), "Famem requires pointers representable by uint64_t");
static_assert(sizeof(size_t) == sizeof(uint64_t), "Famem requires a 64-bit address space");

enum class WorkerState : int {
  kUninitialized = 0,
  kActive = 1,
  kSleeping = 2,
  kPoisoned = 3,
  kClosed = 4,
};

enum class PageType : int {
  kHuge1G = 1,
  kHuge2M = 2,
};

struct Extent {
  size_t size = 0;
  aclrtDrvMemHandle imported_handle = nullptr;
  void* address = nullptr;
  bool mapped = false;
};

struct Allocation {
  size_t requested_size = 0;
  size_t aligned_size = 0;
};

struct WorkerArena {
  std::mutex mutex;
  WorkerState state = WorkerState::kUninitialized;
  int device = -1;
  void* reservation_base = nullptr;
  void* base = nullptr;
  size_t capacity = 0;
  size_t heap_top = 0;
  size_t live_bytes = 0;
  uint64_t allocation_count = 0;
  std::vector<Extent> extents;
  std::unordered_map<void*, Allocation> allocations;
};

WorkerArena g_worker;
std::mutex g_error_mutex;
std::string g_last_error;
bool g_server_initialized = false;
int g_server_device = -1;
bool g_server_poisoned = false;
std::mutex g_server_mutex;
std::vector<aclrtDrvMemHandle> g_server_allocated_handles;
std::vector<aclrtDrvMemHandle> g_server_orphaned_handles;

void SetError(const char* message) noexcept {
  try {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_last_error = message == nullptr ? "unknown native error" : message;
  } catch (...) {
  }
}

void SetError(std::string message) noexcept {
  try {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_last_error = std::move(message);
  } catch (...) {
  }
}

template <typename Result, typename Message>
Result Fail(Result result, Message&& message) {
  SetError(std::forward<Message>(message));
  return result;
}

void ClearError() noexcept { SetError(""); }

std::string ErrorSnapshot() {
  std::lock_guard<std::mutex> lock(g_error_mutex);
  return g_last_error;
}

std::string AclError(const char* operation, aclError error) {
  return std::string(operation) + " failed with ACL error " + std::to_string(error);
}

bool CheckAcl(const char* operation, aclError error) {
  if (error == ACL_SUCCESS) {
    return true;
  }
  return Fail(false, AclError(operation, error));
}

class ServerPhysicalHandleGuard {
 public:
  explicit ServerPhysicalHandleGuard(aclrtDrvMemHandle handle) : handle_(handle) {}
  ~ServerPhysicalHandleGuard() noexcept {
    if (handle_ == nullptr) {
      return;
    }
    const aclError error = aclrtFreePhysical(handle_);
    if (error == ACL_SUCCESS) {
      return;
    }
    g_server_poisoned = true;
    try {
      g_server_orphaned_handles.push_back(handle_);
      SetError(ErrorSnapshot() + "; " + AclError("aclrtFreePhysical(allocation rollback)", error));
    } catch (...) {
      SetError("Famem failed to track an unreleased physical handle; restart the HBM server");
    }
  }

  void release() noexcept { handle_ = nullptr; }

 private:
  aclrtDrvMemHandle handle_;
};

void PoisonWorker(const char* message) noexcept {
  try {
    std::lock_guard<std::mutex> lock(g_worker.mutex);
    g_worker.state = WorkerState::kPoisoned;
  } catch (...) {
  }
  SetError(message);
}

template <typename Result, typename Function>
Result GuardNative(Result failure, const char* unexpected_error, bool poison_worker, Function function) noexcept {
  try {
    return function();
  } catch (const std::exception& error) {
    if (poison_worker) {
      PoisonWorker(error.what());
    } else {
      SetError(error.what());
    }
  } catch (...) {
    if (poison_worker) {
      PoisonWorker(unexpected_error);
    } else {
      SetError(unexpected_error);
    }
  }
  return failure;
}

template <typename Function>
Function LoadAclSymbol(const char* name) {
  dlerror();
  auto* symbol = dlsym(RTLD_DEFAULT, name);
  if (symbol == nullptr) {
    void* library = dlopen("libascendcl.so", RTLD_NOW | RTLD_GLOBAL);
    if (library != nullptr) {
      symbol = dlsym(library, name);
    }
  }
  if (symbol == nullptr) {
    return Fail(nullptr, std::string("CANN does not provide required symbol ") + name);
  }
  return reinterpret_cast<Function>(symbol);
}

bool SetDevice(int device) { return CheckAcl("aclrtSetDevice", aclrtSetDevice(device)); }

bool ParsePageType(int value, PageType* page_type) {
  switch (value) {
    case static_cast<int>(PageType::kHuge1G):
      *page_type = PageType::kHuge1G;
      return true;
    case static_cast<int>(PageType::kHuge2M):
      *page_type = PageType::kHuge2M;
      return true;
    default:
      return Fail(false, "Famem received an unknown physical page type");
  }
}

aclrtMemAttr MemoryAttribute(PageType page_type) {
  return page_type == PageType::kHuge1G ? ACL_HBM_MEM_HUGE1G : ACL_HBM_MEM_HUGE;
}

size_t PageGranularity(PageType page_type) {
  return page_type == PageType::kHuge1G ? kHuge1GGranularity : kHuge2MGranularity;
}

aclrtPhysicalMemProp MemoryProperties(int device, PageType page_type) {
  aclrtPhysicalMemProp properties{};
  properties.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  properties.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  properties.memAttr = MemoryAttribute(page_type);
  properties.location.id = device;
  properties.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  properties.reserve = 0;
  return properties;
}

bool GetGranularity(int device, PageType page_type, size_t* granularity) {
  if (!SetDevice(device)) {
    return false;
  }
  auto properties = MemoryProperties(device, page_type);
  using GetGranularityFunction = aclError (*)(const aclrtPhysicalMemProp*, aclrtMemGranularityOptions, size_t*);
  auto get_granularity = LoadAclSymbol<GetGranularityFunction>("aclrtMemGetAllocationGranularity");
  if (get_granularity == nullptr) {
    return false;
  }
  if (!CheckAcl("aclrtMemGetAllocationGranularity",
                get_granularity(&properties, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, granularity))) {
    return false;
  }
  if (*granularity == 0) {
    return Fail(false, "aclrtMemGetAllocationGranularity returned zero");
  }
  return true;
}

bool GetFreeMemory(int device, PageType page_type, size_t* free_memory) {
  if (!SetDevice(device)) {
    return false;
  }
  using GetMemInfoFunction = aclError (*)(aclrtMemAttr, size_t*, size_t*);
  auto get_mem_info = LoadAclSymbol<GetMemInfoFunction>("aclrtGetMemInfo");
  if (get_mem_info == nullptr) {
    return false;
  }
  size_t total_memory = 0;
  return CheckAcl("aclrtGetMemInfo", get_mem_info(MemoryAttribute(page_type), free_memory, &total_memory));
}

int QueryPageMetric(int device, int page_type_value, uint64_t* output, bool free_memory) {
  return GuardNative(-1, "unexpected error while querying Famem page memory", false, [&] {
    ClearError();
    PageType page_type;
    size_t value = 0;
    if (output == nullptr || !ParsePageType(page_type_value, &page_type)) {
      if (output == nullptr) {
        SetError("page metric output pointer is null");
      }
      return -1;
    }
    const bool success =
        free_memory ? GetFreeMemory(device, page_type, &value) : GetGranularity(device, page_type, &value);
    if (!success) {
      return -1;
    }
    *output = value;
    return 0;
  });
}

bool AddWouldOverflow(size_t left, size_t right) { return right > std::numeric_limits<size_t>::max() - left; }

bool AlignUp(size_t value, size_t alignment, size_t* result) {
  if (result == nullptr || alignment == 0 || AddWouldOverflow(value, alignment - 1)) {
    return false;
  }
  *result = ((value + alignment - 1) / alignment) * alignment;
  return true;
}

bool ValidateExtentLayout(size_t capacity, size_t extent_count, const int32_t* extent_page_types,
                          const uint64_t* extent_sizes, size_t* address_alignment) {
  if (capacity == 0 || extent_count == 0 || extent_count > kMaxExtentCount || extent_page_types == nullptr ||
      extent_sizes == nullptr || address_alignment == nullptr) {
    return Fail(false, "invalid Famem extent layout arguments");
  }

  size_t total_size = 0;
  size_t required_alignment = kHuge2MGranularity;
  int32_t previous_page_type = 0;
  for (size_t index = 0; index < extent_count; ++index) {
    const uint64_t raw_size = extent_sizes[index];
    if (raw_size == 0) {
      return Fail(false, "Famem extent " + std::to_string(index) + " has an invalid size");
    }

    PageType page_type;
    const int32_t page_type_value = extent_page_types[index];
    if (!ParsePageType(page_type_value, &page_type)) {
      return Fail(false, "Famem extent " + std::to_string(index) + " has an unknown physical page type");
    }
    if (page_type_value <= previous_page_type) {
      return Fail(false, "Famem extent page types must be unique and in canonical order");
    }
    previous_page_type = page_type_value;
    const size_t granularity = PageGranularity(page_type);
    required_alignment = std::max(required_alignment, granularity);

    const size_t extent_size = static_cast<size_t>(raw_size);
    if (extent_size % granularity != 0) {
      return Fail(false, "Famem extent size is not aligned to its physical page granularity");
    }
    if (AddWouldOverflow(total_size, extent_size) || total_size + extent_size > capacity) {
      return Fail(false, "extent layout exceeds the reserved Famem arena");
    }
    total_size += extent_size;
  }
  if (total_size != capacity) {
    return Fail(false, "extent sizes do not exactly cover the Famem arena");
  }
  *address_alignment = required_alignment;
  return true;
}

bool ReserveAlignedArena(size_t capacity, size_t alignment, void** reservation_base, void** arena_base) {
  if (reservation_base == nullptr || arena_base == nullptr || alignment == 0 || AddWouldOverflow(capacity, alignment)) {
    return Fail(false, "Famem virtual address reservation size overflow");
  }

  const size_t requested_size = capacity + alignment;
  void* raw_base = nullptr;
  const aclError error = aclrtReserveMemAddress(&raw_base, requested_size, 0, nullptr, kHugePageReserveFlag);
  if (error != ACL_SUCCESS) {
    return Fail(false, AclError("aclrtReserveMemAddress", error));
  }

  const uintptr_t raw_address = reinterpret_cast<uintptr_t>(raw_base);
  size_t aligned_address = 0;
  const bool valid_range = raw_address <= std::numeric_limits<uintptr_t>::max() - requested_size &&
                           AlignUp(static_cast<size_t>(raw_address), alignment, &aligned_address) &&
                           aligned_address >= raw_address && aligned_address <= raw_address + requested_size - capacity;
  if (!valid_range) {
    const aclError release_error = aclrtReleaseMemAddress(raw_base);
    if (release_error == ACL_SUCCESS) {
      SetError("CANN returned a Famem virtual reservation that cannot be aligned safely");
    } else {
      SetError("CANN returned a Famem virtual reservation that cannot be aligned safely; " +
               AclError("aclrtReleaseMemAddress(reservation rollback)", release_error));
    }
    return false;
  }

  *reservation_base = raw_base;
  *arena_base = reinterpret_cast<void*>(aligned_address);
  return true;
}

bool ReleaseExtents(std::vector<Extent>* extents) {
  bool clean = true;
  for (auto iterator = extents->rbegin(); iterator != extents->rend(); ++iterator) {
    if (iterator->mapped) {
      const aclError error = aclrtUnmapMem(iterator->address);
      if (error != ACL_SUCCESS) {
        clean = false;
        SetError(AclError("aclrtUnmapMem(rollback)", error));
        continue;
      }
      iterator->mapped = false;
    }
    if (iterator->imported_handle != nullptr) {
      // Free only this process's imported reference; the server owns the original handle.
      const aclError error = aclrtFreePhysical(iterator->imported_handle);
      if (error != ACL_SUCCESS) {
        clean = false;
        SetError(AclError("aclrtFreePhysical(rollback)", error));
        continue;
      }
      iterator->imported_handle = nullptr;
    }
  }
  extents->erase(
      std::remove_if(extents->begin(), extents->end(),
                     [](const Extent& extent) { return !extent.mapped && extent.imported_handle == nullptr; }),
      extents->end());
  return clean;
}

using ImportFunction = aclError (*)(uint64_t, int32_t, aclrtDrvMemHandle*);

bool ImportAndMapExtents(int device, void* base, size_t extent_count, const uint64_t* extent_sizes,
                         const uint64_t* shareable_handles, std::vector<Extent>* output) {
  auto import_memory = LoadAclSymbol<ImportFunction>("aclrtMemImportFromShareableHandle");
  if (import_memory == nullptr) {
    return false;
  }

  output->clear();
  size_t offset = 0;
  for (size_t index = 0; index < extent_count; ++index) {
    output->emplace_back();
    Extent& extent = output->back();
    extent.size = static_cast<size_t>(extent_sizes[index]);
    extent.address = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + offset);
    aclError error = import_memory(shareable_handles[index], device, &extent.imported_handle);
    if (error != ACL_SUCCESS) {
      return Fail(false, AclError("aclrtMemImportFromShareableHandle", error));
    }
    error = aclrtMapMem(extent.address, extent.size, 0, extent.imported_handle, 0);
    if (error != ACL_SUCCESS) {
      return Fail(false, AclError("aclrtMapMem", error));
    }
    extent.mapped = true;
    offset += extent.size;
  }
  return true;
}

bool AddressRangeIsMapped(uint64_t address, uint64_t size) {
  if (g_worker.state != WorkerState::kActive || g_worker.base == nullptr) {
    return Fail(false, "Famem arena is not mapped");
  }
  const uint64_t base = reinterpret_cast<uint64_t>(g_worker.base);
  if (address < base || size > g_worker.capacity || address - base > g_worker.capacity - size) {
    return Fail(false, "memory copy range is outside the Famem arena");
  }
  return true;
}

aclError FreeTrackedHandles(std::vector<aclrtDrvMemHandle>* handles) {
  aclError first_error = ACL_SUCCESS;
  handles->erase(std::remove_if(handles->begin(), handles->end(),
                                [&first_error](aclrtDrvMemHandle handle) {
                                  const aclError error = aclrtFreePhysical(handle);
                                  if (error != ACL_SUCCESS && first_error == ACL_SUCCESS) {
                                    first_error = error;
                                  }
                                  return error == ACL_SUCCESS;
                                }),
                 handles->end());
  return first_error;
}

int CopyMemory(uint64_t host_address, uint64_t device_address, uint64_t size, bool device_to_host,
               const char* invalid_arguments, const char* operation, const char* unexpected_error) {
  return GuardNative(-1, unexpected_error, false, [&] {
    ClearError();
    std::lock_guard<std::mutex> lock(g_worker.mutex);
    if (host_address == 0 || size == 0 || !AddressRangeIsMapped(device_address, size)) {
      if (host_address == 0 || size == 0) {
        SetError(invalid_arguments);
      }
      return -1;
    }
    if (!SetDevice(g_worker.device)) {
      return -1;
    }
    void* destination = reinterpret_cast<void*>(device_to_host ? host_address : device_address);
    const void* source = reinterpret_cast<void*>(device_to_host ? device_address : host_address);
    const aclrtMemcpyKind kind = device_to_host ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_HOST_TO_DEVICE;
    if (!CheckAcl(operation,
                  aclrtMemcpy(destination, static_cast<size_t>(size), source, static_cast<size_t>(size), kind))) {
      return -1;
    }
    return 0;
  });
}

int WorkerPrepareImpl(int device, uint64_t raw_capacity, size_t extent_count, const int32_t* extent_page_types,
                      const uint64_t* extent_sizes, const uint64_t* shareable_handles, uint64_t* base_address) {
  ClearError();
  std::lock_guard<std::mutex> lock(g_worker.mutex);
  if (g_worker.state != WorkerState::kUninitialized || base_address == nullptr) {
    return Fail(-1, "Famem worker prepare called in an invalid state");
  }

  const size_t capacity = static_cast<size_t>(raw_capacity);
  size_t address_alignment = 0;
  if (!ValidateExtentLayout(capacity, extent_count, extent_page_types, extent_sizes, &address_alignment) ||
      shareable_handles == nullptr) {
    if (shareable_handles == nullptr) {
      SetError("Famem shareable handle array is null");
    }
    return -1;
  }
  if (!SetDevice(device)) {
    return -1;
  }

  g_worker.extents.clear();
  g_worker.extents.reserve(extent_count);
  void* reservation_base = nullptr;
  void* arena_base = nullptr;
  if (!ReserveAlignedArena(capacity, address_alignment, &reservation_base, &arena_base)) {
    return -1;
  }

  // Register ownership before imports so release can retry after exceptions.
  g_worker.state = WorkerState::kPoisoned;
  g_worker.device = device;
  g_worker.reservation_base = reservation_base;
  g_worker.base = arena_base;
  g_worker.capacity = capacity;
  if (!ImportAndMapExtents(device, arena_base, extent_count, extent_sizes, shareable_handles, &g_worker.extents)) {
    const std::string mapping_error = ErrorSnapshot();
    const bool extents_clean = ReleaseExtents(&g_worker.extents);
    const std::string extent_cleanup_error = extents_clean ? "" : ErrorSnapshot();
    aclError reservation_release_error = ACL_SUCCESS;
    if (extents_clean) {
      reservation_release_error = aclrtReleaseMemAddress(reservation_base);
    }
    if (!extents_clean || reservation_release_error != ACL_SUCCESS) {
      std::string error = mapping_error;
      if (!extents_clean) {
        error += "; extent rollback incomplete: " + extent_cleanup_error;
      }
      if (reservation_release_error != ACL_SUCCESS) {
        error += "; " + AclError("aclrtReleaseMemAddress(rollback)", reservation_release_error);
      }
      SetError(std::move(error));
    } else {
      g_worker.state = WorkerState::kUninitialized;
      g_worker.device = -1;
      g_worker.reservation_base = nullptr;
      g_worker.base = nullptr;
      g_worker.capacity = 0;
      SetError(mapping_error);
    }
    return -1;
  }

  g_worker.state = WorkerState::kActive;
  *base_address = reinterpret_cast<uint64_t>(arena_base);
  return 0;
}

int WorkerRemapImpl(int device, size_t extent_count, const int32_t* extent_page_types, const uint64_t* extent_sizes,
                    const uint64_t* shareable_handles) {
  ClearError();
  std::lock_guard<std::mutex> lock(g_worker.mutex);
  if (g_worker.state != WorkerState::kSleeping || device != g_worker.device || shareable_handles == nullptr ||
      !g_worker.extents.empty()) {
    return Fail(-1, "Famem worker remap called in an invalid state");
  }

  size_t address_alignment = 0;
  if (!ValidateExtentLayout(g_worker.capacity, extent_count, extent_page_types, extent_sizes, &address_alignment)) {
    return -1;
  }
  if (reinterpret_cast<uintptr_t>(g_worker.base) % address_alignment != 0) {
    return Fail(-1, "Famem worker virtual address is incompatible with the remap extent layout");
  }
  if (!SetDevice(device)) {
    return -1;
  }

  g_worker.extents.reserve(extent_count);
  g_worker.state = WorkerState::kPoisoned;
  if (!ImportAndMapExtents(device, g_worker.base, extent_count, extent_sizes, shareable_handles, &g_worker.extents)) {
    const std::string mapping_error = ErrorSnapshot();
    const bool clean = ReleaseExtents(&g_worker.extents);
    if (!clean) {
      const std::string cleanup_error = ErrorSnapshot();
      SetError(mapping_error + "; extent rollback incomplete: " + cleanup_error);
    } else {
      g_worker.state = WorkerState::kSleeping;
      SetError(mapping_error);
    }
    return -1;
  }
  g_worker.state = WorkerState::kActive;
  return 0;
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) const char* famem_last_error() noexcept {
  thread_local std::string error_snapshot;
  try {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    error_snapshot = g_last_error;
    return error_snapshot.c_str();
  } catch (...) {
    return "unable to read the Famem native error";
  }
}

__attribute__((visibility("default"))) int famem_get_allocation_granularity(int device, uint64_t* output) {
  return QueryPageMetric(device, static_cast<int>(PageType::kHuge2M), output, false);
}

__attribute__((visibility("default"))) int famem_get_page_granularity(int device, int page_type_value,
                                                                      uint64_t* output) {
  return QueryPageMetric(device, page_type_value, output, false);
}

__attribute__((visibility("default"))) int famem_get_page_free_memory(int device, int page_type_value,
                                                                      uint64_t* output) {
  return QueryPageMetric(device, page_type_value, output, true);
}

__attribute__((visibility("default"))) int famem_get_device_uuid(int device, char* output, size_t output_size) {
  return GuardNative(-1, "unexpected error while querying the NPU UUID", false, [&] {
    ClearError();
    if (output == nullptr || output_size < 33) {
      return Fail(-1, "device UUID output buffer must contain at least 33 bytes");
    }
    using GetUuidFunction = aclError (*)(int32_t, aclrtUuid*);
    auto get_uuid = LoadAclSymbol<GetUuidFunction>("aclrtDeviceGetUuid");
    if (get_uuid == nullptr || !SetDevice(device)) {
      return -1;
    }
    aclrtUuid uuid{};
    if (!CheckAcl("aclrtDeviceGetUuid", get_uuid(device, &uuid))) {
      return -1;
    }
    constexpr char hexadecimal[] = "0123456789abcdef";
    for (size_t index = 0; index < sizeof(uuid.bytes); ++index) {
      const auto byte = static_cast<unsigned char>(uuid.bytes[index]);
      output[index * 2] = hexadecimal[byte >> 4];
      output[index * 2 + 1] = hexadecimal[byte & 0x0f];
    }
    output[32] = '\0';
    return 0;
  });
}

__attribute__((visibility("default"))) int famem_get_bare_tgid(int32_t* output) {
  return GuardNative(-1, "unexpected error while querying the bare TGID", false, [&] {
    ClearError();
    if (output == nullptr) {
      return Fail(-1, "bare TGID output pointer is null");
    }
    using GetBareTgidFunction = aclError (*)(int32_t*);
    auto get_bare_tgid = LoadAclSymbol<GetBareTgidFunction>("aclrtDeviceGetBareTgid");
    if (get_bare_tgid == nullptr) {
      return -1;
    }
    if (!CheckAcl("aclrtDeviceGetBareTgid", get_bare_tgid(output))) {
      return -1;
    }
    if (*output <= 0) {
      return Fail(-1, "aclrtDeviceGetBareTgid returned a non-positive TGID");
    }
    return 0;
  });
}

__attribute__((visibility("default"))) int famem_server_initialize(int device) {
  return GuardNative(-1, "unexpected error while initializing the Famem server backend", false, [&] {
    ClearError();
    std::lock_guard<std::mutex> lock(g_server_mutex);
    if (g_server_initialized) {
      if (g_server_device == device) {
        return 0;
      }
      return Fail(-1, "Famem server backend is already initialized for another device");
    }
    g_server_allocated_handles.clear();
    g_server_allocated_handles.reserve(kMaxExtentCount);
    g_server_orphaned_handles.clear();
    g_server_orphaned_handles.reserve(kMaxExtentCount);
    g_server_poisoned = false;
    const aclError init_error = aclInit(nullptr);
    if (init_error != ACL_SUCCESS && init_error != ACL_ERROR_REPEAT_INITIALIZE) {
      return Fail(-1, AclError("aclInit", init_error));
    }
    if (!SetDevice(device)) {
      if (init_error == ACL_SUCCESS) {
        aclFinalize();
      }
      return -1;
    }
    g_server_initialized = true;
    g_server_device = device;
    return 0;
  });
}

__attribute__((visibility("default"))) int famem_server_allocate_export(int device, uint64_t size,
                                                                        int page_type_value,
                                                                        uint64_t* physical_handle,
                                                                        uint64_t* shareable_handle) {
  return GuardNative(-1, "unexpected error while allocating Famem physical memory", false, [&] {
    ClearError();
    std::lock_guard<std::mutex> lock(g_server_mutex);
    if (!g_server_initialized || device != g_server_device || size == 0 || physical_handle == nullptr ||
        shareable_handle == nullptr) {
      return Fail(-1, "invalid Famem server allocation arguments or state");
    }
    *physical_handle = 0;
    *shareable_handle = 0;
    if (g_server_poisoned) {
      return Fail(-1, "Famem server backend is poisoned by unreleased physical memory");
    }
    if (g_server_allocated_handles.size() >= kMaxExtentCount) {
      return Fail(-1, "Famem server cannot own more than two physical extents");
    }
    PageType page_type;
    if (!ParsePageType(page_type_value, &page_type)) {
      return -1;
    }
    size_t granularity = 0;
    if (!GetGranularity(device, page_type, &granularity)) {
      return -1;
    }
    if (size % granularity != 0) {
      return Fail(-1, "Famem extent size is not aligned to CANN granularity");
    }
    aclrtDrvMemHandle handle = nullptr;
    auto properties = MemoryProperties(device, page_type);
    aclError error = aclrtMallocPhysical(&handle, static_cast<size_t>(size), &properties, 0);
    if (!CheckAcl("aclrtMallocPhysical", error)) {
      return -1;
    }
    ServerPhysicalHandleGuard handle_guard(handle);
    using ExportFunction = aclError (*)(aclrtDrvMemHandle, aclrtMemHandleType, uint64_t, uint64_t*);
    auto export_memory = LoadAclSymbol<ExportFunction>("aclrtMemExportToShareableHandle");
    uint64_t exported = 0;
    if (export_memory == nullptr) {
      return -1;
    }
    error = export_memory(handle, ACL_MEM_HANDLE_TYPE_NONE, 0, &exported);
    if (error == ACL_SUCCESS && exported == 0) {
      return Fail(-1, "aclrtMemExportToShareableHandle returned a zero handle");
    }
    if (error != ACL_SUCCESS) {
      return Fail(-1, AclError("aclrtMemExportToShareableHandle", error));
    }
    g_server_allocated_handles.push_back(handle);
    handle_guard.release();
    *physical_handle = reinterpret_cast<uint64_t>(handle);
    *shareable_handle = exported;
    return 0;
  });
}

__attribute__((visibility("default"))) int famem_server_authorize(uint64_t shareable_handle,
                                                                  const int32_t* bare_tgids,
                                                                  size_t bare_tgid_count) {
  return GuardNative(-1, "unexpected error while authorizing Famem share targets", false, [&] {
    ClearError();
    std::lock_guard<std::mutex> lock(g_server_mutex);
    if (!g_server_initialized || g_server_poisoned || shareable_handle == 0 || bare_tgids == nullptr ||
        bare_tgid_count == 0 || bare_tgid_count > 2 || bare_tgids[0] <= 0 ||
        (bare_tgid_count == 2 && (bare_tgids[1] <= 0 || bare_tgids[0] == bare_tgids[1]))) {
      return Fail(-1, "invalid Famem share authorization arguments or state");
    }
    using SetPidFunction = aclError (*)(uint64_t, int32_t*, size_t);
    auto set_pid = LoadAclSymbol<SetPidFunction>("aclrtMemSetPidToShareableHandle");
    if (set_pid == nullptr) {
      return -1;
    }
    return CheckAcl("aclrtMemSetPidToShareableHandle",
                    set_pid(shareable_handle, const_cast<int32_t*>(bare_tgids), bare_tgid_count))
               ? 0
               : -1;
  });
}

__attribute__((visibility("default"))) int famem_server_free(uint64_t physical_handle) {
  return GuardNative(-1, "unexpected error while freeing Famem physical memory", false, [&] {
    ClearError();
    std::lock_guard<std::mutex> lock(g_server_mutex);
    if (!g_server_initialized || physical_handle == 0) {
      return Fail(-1, "Famem server is not initialized or physical handle is null");
    }
    if (!SetDevice(g_server_device)) {
      return -1;
    }
    const auto handle = reinterpret_cast<aclrtDrvMemHandle>(physical_handle);
    const auto tracked = std::find(g_server_allocated_handles.begin(), g_server_allocated_handles.end(), handle);
    if (tracked == g_server_allocated_handles.end()) {
      return Fail(-1, "Famem server received an unknown or duplicate physical handle free");
    }
    if (!CheckAcl("aclrtFreePhysical", aclrtFreePhysical(handle))) {
      return -1;
    }
    g_server_allocated_handles.erase(tracked);
    return 0;
  });
}

__attribute__((visibility("default"))) int famem_server_finalize() {
  return GuardNative(-1, "unexpected error while finalizing the Famem server backend", false, [&] {
    ClearError();
    std::lock_guard<std::mutex> lock(g_server_mutex);
    if (!g_server_initialized) {
      return 0;
    }
    const aclError set_device_error = aclrtSetDevice(g_server_device);
    if (set_device_error != ACL_SUCCESS) {
      return Fail(-1, AclError("aclrtSetDevice(finalize)", set_device_error));
    }
    const aclError allocation_error = FreeTrackedHandles(&g_server_allocated_handles);
    const aclError orphan_error = FreeTrackedHandles(&g_server_orphaned_handles);
    g_server_poisoned = !g_server_allocated_handles.empty() || !g_server_orphaned_handles.empty();
    if (allocation_error != ACL_SUCCESS) {
      return Fail(-1, AclError("aclrtFreePhysical(outstanding cleanup)", allocation_error));
    }
    if (orphan_error != ACL_SUCCESS) {
      return Fail(-1, AclError("aclrtFreePhysical(orphan cleanup)", orphan_error));
    }
    const aclError reset_error = aclrtResetDevice(g_server_device);
    if (reset_error != ACL_SUCCESS) {
      return Fail(-1, AclError("aclrtResetDevice", reset_error));
    }
    const aclError finalize_error = aclFinalize();
    if (finalize_error != ACL_SUCCESS && finalize_error != ACL_ERROR_REPEAT_FINALIZE) {
      return Fail(-1, AclError("aclFinalize", finalize_error));
    }
    g_server_initialized = false;
    g_server_device = -1;
    g_server_poisoned = false;
    return 0;
  });
}

__attribute__((visibility("default"))) int famem_worker_prepare_v2(int device, uint64_t capacity, size_t extent_count,
                                                                   const int32_t* extent_page_types,
                                                                   const uint64_t* extent_sizes,
                                                                   const uint64_t* shareable_handles,
                                                                   uint64_t* base_address) {
  return GuardNative(-1, "unexpected error while preparing the Famem worker arena", false, [&] {
    return WorkerPrepareImpl(device, capacity, extent_count, extent_page_types, extent_sizes, shareable_handles,
                             base_address);
  });
}

__attribute__((visibility("default"))) int famem_worker_unmap(int device) {
  return GuardNative(-1, "unexpected error while unmapping the Famem worker arena", true, [&] {
    ClearError();
    std::lock_guard<std::mutex> lock(g_worker.mutex);
    if (g_worker.state != WorkerState::kActive || device != g_worker.device) {
      return Fail(-1, "Famem worker unmap called in an invalid state");
    }
    if (!SetDevice(device)) {
      return -1;
    }
    if (!CheckAcl("aclrtSynchronizeDevice", aclrtSynchronizeDevice())) {
      return -1;
    }
    const bool clean = ReleaseExtents(&g_worker.extents);
    g_worker.state = clean ? WorkerState::kSleeping : WorkerState::kPoisoned;
    return clean ? 0 : -1;
  });
}

__attribute__((visibility("default"))) int famem_worker_remap_v2(int device, size_t extent_count,
                                                                 const int32_t* extent_page_types,
                                                                 const uint64_t* extent_sizes,
                                                                 const uint64_t* shareable_handles) {
  return GuardNative(-1, "unexpected error while remapping the Famem worker arena", true, [&] {
    return WorkerRemapImpl(device, extent_count, extent_page_types, extent_sizes, shareable_handles);
  });
}

__attribute__((visibility("default"))) int famem_worker_release(int device) {
  return GuardNative(-1, "unexpected error while releasing the Famem worker arena", true, [&] {
    ClearError();
    std::lock_guard<std::mutex> lock(g_worker.mutex);
    if (g_worker.state == WorkerState::kClosed || g_worker.state == WorkerState::kUninitialized) {
      g_worker.state = WorkerState::kClosed;
      return 0;
    }
    if (device != g_worker.device) {
      return Fail(-1, "Famem worker release targets the wrong device");
    }
    if (!SetDevice(device)) {
      g_worker.state = WorkerState::kPoisoned;
      return -1;
    }
    const bool has_mapped_extents = std::any_of(g_worker.extents.begin(), g_worker.extents.end(),
                                                [](const Extent& extent) { return extent.mapped; });
    if (has_mapped_extents) {
      if (!CheckAcl("aclrtSynchronizeDevice(release)", aclrtSynchronizeDevice())) {
        g_worker.state = WorkerState::kPoisoned;
        return -1;
      }
    }
    if (!g_worker.extents.empty() && !ReleaseExtents(&g_worker.extents)) {
      g_worker.state = WorkerState::kPoisoned;
      return -1;
    }
    if (g_worker.reservation_base != nullptr) {
      const aclError error = aclrtReleaseMemAddress(g_worker.reservation_base);
      if (error != ACL_SUCCESS) {
        g_worker.state = WorkerState::kPoisoned;
        return Fail(-1, AclError("aclrtReleaseMemAddress", error));
      }
    }
    g_worker.reservation_base = nullptr;
    g_worker.base = nullptr;
    g_worker.capacity = 0;
    g_worker.heap_top = 0;
    g_worker.live_bytes = 0;
    g_worker.allocation_count = 0;
    g_worker.allocations.clear();
    g_worker.state = WorkerState::kClosed;
    return 0;
  });
}

__attribute__((visibility("default"))) int famem_worker_get_stats(uint64_t* capacity, uint64_t* heap_top,
                                                                  uint64_t* live_bytes, uint64_t* allocation_count,
                                                                  uint64_t* base_address, int* state) {
  return GuardNative(-1, "unexpected error while reading Famem worker stats", false, [&] {
    ClearError();
    if (capacity == nullptr || heap_top == nullptr || live_bytes == nullptr || allocation_count == nullptr ||
        base_address == nullptr || state == nullptr) {
      return Fail(-1, "Famem stats output pointer is null");
    }
    std::lock_guard<std::mutex> lock(g_worker.mutex);
    *capacity = g_worker.capacity;
    *heap_top = g_worker.heap_top;
    *live_bytes = g_worker.live_bytes;
    *allocation_count = g_worker.allocation_count;
    *base_address = reinterpret_cast<uint64_t>(g_worker.base);
    *state = static_cast<int>(g_worker.state);
    return 0;
  });
}

__attribute__((visibility("default"))) int famem_worker_get_allocations(size_t capacity, size_t* allocation_count,
                                                                        uint64_t* addresses, uint64_t* aligned_sizes) {
  return GuardNative(-1, "unexpected error while reading Famem allocations", false, [&] {
    ClearError();
    if (allocation_count == nullptr || addresses == nullptr || aligned_sizes == nullptr) {
      return Fail(-1, "Famem allocation snapshot output pointer is null");
    }
    std::lock_guard<std::mutex> lock(g_worker.mutex);
    if (g_worker.state != WorkerState::kActive || capacity < g_worker.allocations.size()) {
      return Fail(-1, "Famem allocation snapshot called in an invalid state or with insufficient capacity");
    }
    std::vector<std::pair<void*, Allocation>> allocations(g_worker.allocations.begin(), g_worker.allocations.end());
    std::sort(allocations.begin(), allocations.end(), [](const auto& left, const auto& right) {
      return reinterpret_cast<uintptr_t>(left.first) < reinterpret_cast<uintptr_t>(right.first);
    });
    *allocation_count = allocations.size();
    for (size_t index = 0; index < allocations.size(); ++index) {
      addresses[index] = reinterpret_cast<uint64_t>(allocations[index].first);
      aligned_sizes[index] = allocations[index].second.aligned_size;
    }
    return 0;
  });
}

__attribute__((visibility("default"))) int famem_memcpy_device_to_host(uint64_t host_address, uint64_t device_address,
                                                                       uint64_t size) {
  return CopyMemory(host_address, device_address, size, true, "invalid Famem device-to-host copy arguments",
                    "aclrtMemcpy(device-to-host)", "unexpected error during Famem device-to-host copy");
}

__attribute__((visibility("default"))) int famem_memcpy_host_to_device(uint64_t device_address, uint64_t host_address,
                                                                       uint64_t size) {
  return CopyMemory(host_address, device_address, size, false, "invalid Famem host-to-device copy arguments",
                    "aclrtMemcpy(host-to-device)", "unexpected error during Famem host-to-device copy");
}

__attribute__((visibility("default"))) void* famem_malloc(size_t size, int device, aclrtStream stream) {
  (void)stream;
  return GuardNative<void*>(nullptr, "unexpected error during Famem allocation", false, [&]() -> void* {
    ClearError();
    if (size == 0) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_worker.mutex);
    if (g_worker.state != WorkerState::kActive || device != g_worker.device) {
      return Fail(static_cast<void*>(nullptr), "Famem allocation requested while the arena is not active");
    }
    size_t aligned_size = 0;
    if (!AlignUp(size, kAllocationAlignment, &aligned_size) || g_worker.heap_top > g_worker.capacity ||
        aligned_size > g_worker.capacity - g_worker.heap_top) {
      return Fail(static_cast<void*>(nullptr), "Famem arena exhausted: freed blocks cannot be reused");
    }
    void* pointer = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(g_worker.base) + g_worker.heap_top);
    const auto insertion = g_worker.allocations.emplace(pointer, Allocation{size, aligned_size});
    if (!insertion.second) {
      return Fail(static_cast<void*>(nullptr), "Famem generated a duplicate allocation address");
    }
    g_worker.heap_top += aligned_size;
    g_worker.live_bytes += aligned_size;
    ++g_worker.allocation_count;
    return pointer;
  });
}

__attribute__((visibility("default"))) void famem_free(void* pointer, size_t size, int device, aclrtStream stream) {
  (void)stream;
  try {
    ClearError();
    if (pointer == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(g_worker.mutex);
    if (device != g_worker.device) {
      SetError("Famem free targets the wrong device");
      return;
    }
    auto allocation = g_worker.allocations.find(pointer);
    if (allocation == g_worker.allocations.end()) {
      SetError("Famem received an unknown or duplicate pointer free");
      return;
    }
    if (size != allocation->second.requested_size) {
      SetError("Famem free size does not match the original allocation");
    }
    g_worker.live_bytes -= allocation->second.aligned_size;
    g_worker.allocations.erase(allocation);
  } catch (const std::exception& error) {
    SetError(error.what());
  } catch (...) {
    SetError("unexpected error during Famem free");
  }
}

}  // extern "C"
