#ifndef VLLM_ASCEND_TESTS_FAKE_ACL_H
#define VLLM_ASCEND_TESTS_FAKE_ACL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int aclError;
typedef void* aclrtDrvMemHandle;
typedef void* aclrtStream;
typedef void* aclrtContext;
typedef int aclrtMemAttr;
typedef int aclrtMemGranularityOptions;
typedef int aclrtMemHandleType;
typedef int aclrtMemcpyKind;

typedef struct aclrtMemLocation {
  int32_t id;
  uint32_t type;
} aclrtMemLocation;

typedef struct aclrtPhysicalMemProp {
  aclrtMemHandleType handleType;
  int allocationType;
  aclrtMemAttr memAttr;
  aclrtMemLocation location;
  uint64_t reserve;
} aclrtPhysicalMemProp;

typedef struct aclrtUuid {
  uint8_t bytes[16];
} aclrtUuid;

#define ACL_SUCCESS 0
#define ACL_ERROR_REPEAT_INITIALIZE 1001
#define ACL_ERROR_REPEAT_FINALIZE 1002
#define ACL_ERROR_RT_MEMORY_ALLOCATION 1003
#define ACL_HBM_MEM_HUGE1G 1
#define ACL_HBM_MEM_HUGE 2
#define ACL_MEM_HANDLE_TYPE_NONE 0
#define ACL_MEM_ALLOCATION_TYPE_PINNED 1
#define ACL_MEM_LOCATION_TYPE_DEVICE 1
#define ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM 0
#define ACL_MEMCPY_HOST_TO_DEVICE 1
#define ACL_MEMCPY_DEVICE_TO_HOST 2

aclError aclInit(const char* config_path);
aclError aclFinalize(void);
aclError aclrtSetDevice(int32_t device);
aclError aclrtResetDevice(int32_t device);
aclError aclrtGetCurrentContext(aclrtContext* context);
aclError aclrtCreateContext(aclrtContext* context, int32_t device);
aclError aclrtSetCurrentContext(aclrtContext context);
aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t* free_memory, size_t* total_memory);
aclError aclrtMemGetAllocationGranularity(const aclrtPhysicalMemProp* properties, aclrtMemGranularityOptions option,
                                          size_t* granularity);
aclError aclrtDeviceGetUuid(int32_t device, aclrtUuid* uuid);
aclError aclrtDeviceGetBareTgid(int32_t* bare_tgid);
aclError aclrtMallocPhysical(aclrtDrvMemHandle* handle, size_t size, const aclrtPhysicalMemProp* properties,
                             uint64_t flags);
aclError aclrtFreePhysical(aclrtDrvMemHandle handle);
aclError aclrtMemExportToShareableHandle(aclrtDrvMemHandle handle, aclrtMemHandleType type, uint64_t flags,
                                         uint64_t* shareable_handle);
aclError aclrtMemSetPidToShareableHandle(uint64_t shareable_handle, int32_t* bare_tgids, size_t bare_tgid_count);
aclError aclrtMemImportFromShareableHandle(uint64_t shareable_handle, int32_t device, aclrtDrvMemHandle* handle);
aclError aclrtReserveMemAddress(void** address, size_t size, size_t alignment, void* requested_address, uint64_t flags);
aclError aclrtReleaseMemAddress(void* address);
aclError aclrtMapMem(void* address, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags);
aclError aclrtUnmapMem(void* address);
aclError aclrtSynchronizeDevice(void);
aclError aclrtMemcpy(void* destination, size_t destination_size, const void* source, size_t size, aclrtMemcpyKind kind);

void fake_acl_reset_observations(void);
void fake_acl_fail_set_pid(int enabled);
void fake_acl_fail_export(int enabled);
void fake_acl_fail_map(int enabled);
void fake_acl_fail_free(int enabled);
size_t fake_acl_export_call_count(void);
size_t fake_acl_set_pid_call_count(void);
size_t fake_acl_malloc_physical_call_count(void);
size_t fake_acl_free_physical_call_count(void);
uint64_t fake_acl_last_freed_handle(void);
size_t fake_acl_live_handle_count(void);
uint64_t fake_acl_last_set_pid_handle(void);
size_t fake_acl_last_target_count(void);
int32_t fake_acl_last_target(size_t index);

#ifdef __cplusplus
}
#endif

#endif
