#include "acl/acl.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

namespace {

constexpr size_t kHuge2M = size_t{2} << 20;
constexpr size_t kHuge1G = size_t{1} << 30;
constexpr aclError kFakeError = 1;

struct FakePhysicalHandle {
  uint64_t value;
};

uint64_t g_next_handle = 1;
uint64_t g_last_exported = 0;
size_t g_export_calls = 0;
size_t g_set_pid_calls = 0;
size_t g_malloc_physical_calls = 0;
size_t g_free_physical_calls = 0;
uint64_t g_last_freed_handle = 0;
uint64_t g_last_set_pid_handle = 0;
size_t g_last_target_count = 0;
int32_t g_last_targets[2] = {0, 0};
size_t g_live_handles = 0;
bool g_fail_set_pid = false;
bool g_fail_export = false;
bool g_fail_map = false;
bool g_fail_free = false;
int g_context_storage = 0;
aclrtContext g_current_context = nullptr;

FakePhysicalHandle* NewHandle(uint64_t value) {
  auto* handle = new (std::nothrow) FakePhysicalHandle{value};
  if (handle != nullptr) {
    ++g_live_handles;
  }
  return handle;
}

}  // namespace

extern "C" {

aclError aclInit(const char* config_path) {
  (void)config_path;
  return ACL_SUCCESS;
}

aclError aclFinalize(void) { return ACL_SUCCESS; }

aclError aclrtSetDevice(int32_t device) { return device < 0 ? kFakeError : ACL_SUCCESS; }

aclError aclrtResetDevice(int32_t device) { return device < 0 ? kFakeError : ACL_SUCCESS; }

aclError aclrtGetCurrentContext(aclrtContext* context) {
  if (context == nullptr) {
    return kFakeError;
  }
  *context = g_current_context;
  return ACL_SUCCESS;
}

aclError aclrtCreateContext(aclrtContext* context, int32_t device) {
  if (context == nullptr || device < 0) {
    return kFakeError;
  }
  *context = &g_context_storage;
  return ACL_SUCCESS;
}

aclError aclrtSetCurrentContext(aclrtContext context) {
  if (context == nullptr) {
    return kFakeError;
  }
  g_current_context = context;
  return ACL_SUCCESS;
}

aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t* free_memory, size_t* total_memory) {
  if (free_memory == nullptr || total_memory == nullptr || (attr != ACL_HBM_MEM_HUGE1G && attr != ACL_HBM_MEM_HUGE)) {
    return kFakeError;
  }
  *free_memory = kHuge1G * 64;
  *total_memory = kHuge1G * 64;
  return ACL_SUCCESS;
}

aclError aclrtMemGetAllocationGranularity(const aclrtPhysicalMemProp* properties, aclrtMemGranularityOptions option,
                                          size_t* granularity) {
  (void)option;
  if (properties == nullptr || granularity == nullptr) {
    return kFakeError;
  }
  *granularity = properties->memAttr == ACL_HBM_MEM_HUGE1G ? kHuge1G : kHuge2M;
  return ACL_SUCCESS;
}

aclError aclrtDeviceGetUuid(int32_t device, aclrtUuid* uuid) {
  if (device < 0 || uuid == nullptr) {
    return kFakeError;
  }
  for (size_t index = 0; index < sizeof(uuid->bytes); ++index) {
    uuid->bytes[index] = static_cast<uint8_t>(index);
  }
  return ACL_SUCCESS;
}

aclError aclrtDeviceGetBareTgid(int32_t* bare_tgid) {
  if (bare_tgid == nullptr) {
    return kFakeError;
  }
  *bare_tgid = 4321;
  return ACL_SUCCESS;
}

aclError aclrtMallocPhysical(aclrtDrvMemHandle* handle, size_t size, const aclrtPhysicalMemProp* properties,
                             uint64_t flags) {
  (void)flags;
  ++g_malloc_physical_calls;
  if (handle == nullptr || size == 0 || properties == nullptr) {
    return kFakeError;
  }
  auto* allocated = NewHandle(g_next_handle++);
  if (allocated == nullptr) {
    return kFakeError;
  }
  *handle = allocated;
  return ACL_SUCCESS;
}

aclError aclrtFreePhysical(aclrtDrvMemHandle handle) {
  ++g_free_physical_calls;
  if (handle == nullptr || g_fail_free) {
    return kFakeError;
  }
  auto* physical_handle = static_cast<FakePhysicalHandle*>(handle);
  g_last_freed_handle = physical_handle->value;
  delete physical_handle;
  --g_live_handles;
  return ACL_SUCCESS;
}

aclError aclrtMemExportToShareableHandle(aclrtDrvMemHandle handle, aclrtMemHandleType type, uint64_t flags,
                                         uint64_t* shareable_handle) {
  (void)type;
  (void)flags;
  if (handle == nullptr || shareable_handle == nullptr || g_fail_export) {
    return kFakeError;
  }
  g_last_exported = static_cast<FakePhysicalHandle*>(handle)->value + 10'000;
  *shareable_handle = g_last_exported;
  ++g_export_calls;
  return ACL_SUCCESS;
}

aclError aclrtMemSetPidToShareableHandle(uint64_t shareable_handle, int32_t* bare_tgids, size_t bare_tgid_count) {
  if (shareable_handle == 0 || shareable_handle != g_last_exported || bare_tgids == nullptr || bare_tgid_count == 0 ||
      bare_tgid_count > 2) {
    return kFakeError;
  }
  ++g_set_pid_calls;
  g_last_set_pid_handle = shareable_handle;
  g_last_target_count = bare_tgid_count;
  g_last_targets[0] = bare_tgids[0];
  g_last_targets[1] = bare_tgid_count == 2 ? bare_tgids[1] : 0;
  return g_fail_set_pid ? kFakeError : ACL_SUCCESS;
}

aclError aclrtMemImportFromShareableHandle(uint64_t shareable_handle, int32_t device, aclrtDrvMemHandle* handle) {
  if (shareable_handle == 0 || device < 0 || handle == nullptr) {
    return kFakeError;
  }
  auto* imported = NewHandle(shareable_handle);
  if (imported == nullptr) {
    return kFakeError;
  }
  *handle = imported;
  return ACL_SUCCESS;
}

aclError aclrtReserveMemAddress(void** address, size_t size, size_t alignment, void* requested_address,
                                uint64_t flags) {
  (void)alignment;
  (void)requested_address;
  (void)flags;
  if (address == nullptr || size == 0) {
    return kFakeError;
  }
  void* allocation = nullptr;
  if (posix_memalign(&allocation, kHuge2M, size) != 0) {
    return kFakeError;
  }
  *address = allocation;
  return ACL_SUCCESS;
}

aclError aclrtReleaseMemAddress(void* address) {
  if (address == nullptr) {
    return kFakeError;
  }
  std::free(address);
  return ACL_SUCCESS;
}

aclError aclrtMapMem(void* address, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags) {
  (void)offset;
  (void)flags;
  return g_fail_map || address == nullptr || size == 0 || handle == nullptr ? kFakeError : ACL_SUCCESS;
}

aclError aclrtUnmapMem(void* address) { return address == nullptr ? kFakeError : ACL_SUCCESS; }

aclError aclrtSynchronizeDevice(void) { return ACL_SUCCESS; }

aclError aclrtMemcpy(void* destination, size_t destination_size, const void* source, size_t size,
                     aclrtMemcpyKind kind) {
  (void)kind;
  if (destination == nullptr || source == nullptr || size == 0 || destination_size < size) {
    return kFakeError;
  }
  std::memcpy(destination, source, size);
  return ACL_SUCCESS;
}

void fake_acl_reset_observations(void) {
  g_last_exported = 0;
  g_export_calls = 0;
  g_set_pid_calls = 0;
  g_malloc_physical_calls = 0;
  g_free_physical_calls = 0;
  g_last_freed_handle = 0;
  g_last_set_pid_handle = 0;
  g_last_target_count = 0;
  g_last_targets[0] = 0;
  g_last_targets[1] = 0;
  g_fail_set_pid = false;
  g_fail_export = false;
  g_fail_map = false;
  g_fail_free = false;
}

void fake_acl_fail_set_pid(int enabled) { g_fail_set_pid = enabled != 0; }

void fake_acl_fail_export(int enabled) { g_fail_export = enabled != 0; }

void fake_acl_fail_map(int enabled) { g_fail_map = enabled != 0; }

void fake_acl_fail_free(int enabled) { g_fail_free = enabled != 0; }

size_t fake_acl_export_call_count(void) { return g_export_calls; }

size_t fake_acl_set_pid_call_count(void) { return g_set_pid_calls; }

size_t fake_acl_malloc_physical_call_count(void) { return g_malloc_physical_calls; }

size_t fake_acl_free_physical_call_count(void) { return g_free_physical_calls; }

uint64_t fake_acl_last_freed_handle(void) { return g_last_freed_handle; }

size_t fake_acl_live_handle_count(void) { return g_live_handles; }

uint64_t fake_acl_last_set_pid_handle(void) { return g_last_set_pid_handle; }

size_t fake_acl_last_target_count(void) { return g_last_target_count; }

int32_t fake_acl_last_target(size_t index) {
  return index < 2 ? g_last_targets[index] : std::numeric_limits<int32_t>::min();
}

}  // extern "C"
