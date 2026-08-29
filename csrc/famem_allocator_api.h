// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#ifndef VLLM_ASCEND_FAMEM_ALLOCATOR_API_H
#define VLLM_ASCEND_FAMEM_ALLOCATOR_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* famem_last_error(void);
int famem_get_page_granularity(int device, int page_type, uint64_t* output);
int famem_get_page_free_memory(int device, int page_type, uint64_t* output);
int famem_get_device_uuid(int device, char* output, size_t output_size);
int famem_server_initialize(int device);
int famem_server_allocate_export(int device, uint64_t size, int page_type, uint64_t* physical_handle,
                                 uint64_t* shareable_handle);
int famem_server_authorize(uint64_t shareable_handle, const int32_t* bare_tgids, size_t bare_tgid_count);
int famem_server_free(uint64_t physical_handle);
int famem_server_finalize(void);
int famem_worker_get_allocations(size_t capacity, size_t* allocation_count, uint64_t* addresses,
                                 uint64_t* aligned_sizes);

#ifdef __cplusplus
}
#endif

#endif
