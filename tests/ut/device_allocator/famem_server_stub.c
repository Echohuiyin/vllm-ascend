/* Test-only Famem C ABI backend for the native HBM server integration test. */

#define _DEFAULT_SOURCE

#include "famem_allocator_api.h"

#include <errno.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>

#define TEST_UUID "0123456789abcdef0123456789abcdef"

static const char* g_error = "";
static uint64_t g_next_handle = 100U;
static int g_initialized = 0;
static unsigned int g_allocation_count = 0U;
static unsigned int g_authorize_count = 0U;
static unsigned int g_free_count = 0U;
static size_t g_shareable_count = 0U;
static uint64_t g_shareable_handles[2] = {0U, 0U};

static void trace_event(const char* event, uint64_t handle) {
  const char* trace_path = getenv("FAMEM_STUB_TRACE");
  FILE* trace;
  if (trace_path == NULL) {
    return;
  }
  trace = fopen(trace_path, "a");
  if (trace != NULL) {
    (void)fprintf(trace, "%s %llu\n", event, (unsigned long long)handle);
    (void)fclose(trace);
  }
}

int connect(int socket_fd, const struct sockaddr* address, socklen_t address_length) {
  if (getenv("FAMEM_STUB_CONNECT_ENOENT") != NULL) {
    errno = ENOENT;
    return -1;
  }
  return (int)syscall(SYS_connect, socket_fd, address, address_length);
}

int flock(int file_descriptor, int operation) {
  const char* ready_path = getenv("FAMEM_STUB_FLOCK_READY");
  if (ready_path != NULL) {
    if (syscall(SYS_flock, file_descriptor, operation | LOCK_NB) == 0) {
      return 0;
    }
    if (errno != EWOULDBLOCK) {
      return -1;
    }
    FILE* ready = fopen(ready_path, "w");
    if (ready == NULL) {
      return -1;
    }
    (void)fclose(ready);
  }
  return (int)syscall(SYS_flock, file_descriptor, operation);
}

int listen(int socket_fd, int backlog) {
  const char* barrier = getenv("FAMEM_STUB_LISTEN_BARRIER");
  trace_event("LISTEN", 0U);
  if (barrier != NULL) {
    FILE* ready = fopen(barrier, "w");
    if (ready == NULL) {
      return -1;
    }
    (void)fclose(ready);
    while (access(barrier, F_OK) == 0) {
      (void)usleep(1000U);
    }
  }
  return (int)syscall(SYS_listen, socket_fd, backlog);
}

static int fail_call(const char* environment_name, unsigned int call_count) {
  const char* value = getenv(environment_name);
  char* end = NULL;
  unsigned long configured_call;
  if (value == NULL || value[0] == '\0') {
    return 0;
  }
  errno = 0;
  configured_call = strtoul(value, &end, 10);
  return errno == 0 && end != value && *end == '\0' && configured_call == call_count;
}

const char* famem_last_error(void) { return g_error; }

int famem_get_page_granularity(int device, int page_type, uint64_t* output) {
  (void)device;
  if (output == NULL || (page_type != 1 && page_type != 2)) {
    g_error = "invalid granularity request";
    return -1;
  }
  *output = page_type == 1 ? UINT64_C(1) << 30 : UINT64_C(2) << 20;
  return 0;
}

int famem_get_page_free_memory(int device, int page_type, uint64_t* output) {
  (void)device;
  if (output == NULL || page_type != 1) {
    g_error = "invalid free-memory request";
    return -1;
  }
  *output = UINT64_C(1) << 30;
  return 0;
}

int famem_get_device_uuid(int device, char* output, size_t output_size) {
  (void)device;
  if (output == NULL || output_size < sizeof(TEST_UUID)) {
    g_error = "UUID buffer is too small";
    return -1;
  }
  memcpy(output, TEST_UUID, sizeof(TEST_UUID));
  return 0;
}

int famem_server_initialize(int device) {
  if (device < 0) {
    g_error = "invalid device";
    return -1;
  }
  g_initialized = 1;
  return 0;
}

int famem_server_allocate_export(int device, uint64_t size, int page_type, uint64_t* physical_handle,
                                 uint64_t* shareable_handle) {
  (void)device;
  if (!g_initialized || size == 0U || (page_type != 1 && page_type != 2) || physical_handle == NULL ||
      shareable_handle == NULL || g_shareable_count >= 2U) {
    g_error = "invalid allocation";
    return -1;
  }
  ++g_allocation_count;
  *physical_handle = 0U;
  *shareable_handle = 0U;
  if (fail_call("FAMEM_STUB_FAIL_ALLOC_AT", g_allocation_count)) {
    g_error = "injected allocation failure";
    return -1;
  }
  *physical_handle = g_next_handle++;
  *shareable_handle = *physical_handle + 1000U;
  g_shareable_handles[g_shareable_count++] = *shareable_handle;
  trace_event("ALLOCATE_EXPORT", *shareable_handle);
  return 0;
}

int famem_server_authorize(uint64_t shareable_handle, const int32_t* bare_tgids, size_t bare_tgid_count) {
  if (!g_initialized || shareable_handle == 0U || bare_tgids == NULL || bare_tgid_count != 2U || bare_tgids[0] <= 0 ||
      bare_tgids[1] <= 0 || bare_tgids[0] == bare_tgids[1] ||
      (shareable_handle != g_shareable_handles[0] && shareable_handle != g_shareable_handles[1])) {
    g_error = "invalid authorization";
    return -1;
  }
  ++g_authorize_count;
  if (fail_call("FAMEM_STUB_FAIL_AUTHORIZE_AT", g_authorize_count)) {
    g_error = "injected authorization failure";
    return -1;
  }
  trace_event("AUTHORIZE", shareable_handle);
  return 0;
}

int famem_server_free(uint64_t physical_handle) {
  if (!g_initialized || physical_handle == 0U) {
    g_error = "invalid free";
    return -1;
  }
  ++g_free_count;
  if (fail_call("FAMEM_STUB_FAIL_FREE_AT", g_free_count)) {
    g_error = "injected free failure";
    return -1;
  }
  trace_event("FREE", physical_handle);
  return 0;
}

int famem_server_finalize(void) {
  trace_event("FINALIZE", 0U);
  g_initialized = 0;
  g_allocation_count = 0U;
  g_authorize_count = 0U;
  g_free_count = 0U;
  g_shareable_count = 0U;
  g_shareable_handles[0] = 0U;
  g_shareable_handles[1] = 0U;
  return 0;
}
