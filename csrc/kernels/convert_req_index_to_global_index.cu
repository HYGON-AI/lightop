// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"

#include <stdint.h>
#include <limits>
#include <stdio.h>
#include <cmath>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <torch/extension.h>

#include <c10/macros/Macros.h>
#include <c10/cuda/CUDAMathCompat.h>
namespace at {  
namespace native {
template <bool HAS_PREFILL>
__device__ __forceinline__ int process_single_token(
    int tok, int req, bool is_prefill, int prefill_req_id,
    const int32_t* __restrict__ block_table_ptr,
    const int32_t* __restrict__ workspace_starts_ptr,
    int max_num_blocks_per_req, int BLOCK_SIZE,
    int bt_stride0, int bt_stride1)
{
    if (tok < 0) return -1;
    
    int block_id = tok / BLOCK_SIZE;
    int inblock_off = tok % BLOCK_SIZE;

    if (block_id < 0 || block_id >= max_num_blocks_per_req) return -1;

    if constexpr (HAS_PREFILL) {
        if (is_prefill) {
            return workspace_starts_ptr[prefill_req_id] + tok;
        }
    }
    
    int base = block_table_ptr[req * bt_stride0 + block_id * bt_stride1];
    return base * BLOCK_SIZE + inblock_off;
}

template <bool HAS_PREFILL>
__global__ void convert_req_index_to_global_index_kernel(
    const int32_t* __restrict__ req_id_ptr,
    const int32_t* __restrict__ block_table_ptr,
    const int32_t* __restrict__ token_indices_ptr,
    int32_t* __restrict__ out_ptr,
    const int32_t* __restrict__ prefill_request_id_ptr,
    const int32_t* __restrict__ workspace_starts_ptr,
    const int max_num_blocks_per_req,
    const int BLOCK_SIZE,
    const int num_tokens,
    const int num_topk_tokens,
    const int bt_stride0, const int bt_stride1,
    const int ti_stride0, const int ti_stride1,
    const int out_stride0, const int out_stride1) 
{
    int token_id = blockIdx.y; 
    int vec_id = blockIdx.x * blockDim.x + threadIdx.x;

    int indice_id = vec_id * 4; 
    if (token_id >= num_tokens || indice_id >= num_topk_tokens) {
        return;
    }

    int req = req_id_ptr[token_id];

    bool is_prefill = false;
    int prefill_req_id = -1;

    if constexpr (HAS_PREFILL) {
        prefill_req_id = prefill_request_id_ptr[token_id];
        is_prefill = (prefill_req_id >= 0);
    }

    const int32_t* in_ptr = &token_indices_ptr[token_id * ti_stride0 + indice_id * ti_stride1];
    int32_t* out_ptr_curr = &out_ptr[token_id * out_stride0 + indice_id * out_stride1];

    bool can_vec4 = (indice_id + 3 < num_topk_tokens) && 
                    (ti_stride1 == 1) && (out_stride1 == 1) &&
                    (reinterpret_cast<uintptr_t>(in_ptr) % 16 == 0) && 
                    (reinterpret_cast<uintptr_t>(out_ptr_curr) % 16 == 0);

    if (can_vec4) {
        const int4* toks_vec = reinterpret_cast<const int4*>(in_ptr);
        int4 toks = *toks_vec;
        
        int4 outs;
        outs.x = process_single_token<HAS_PREFILL>(toks.x, req, is_prefill, prefill_req_id, block_table_ptr, workspace_starts_ptr, max_num_blocks_per_req, BLOCK_SIZE, bt_stride0, bt_stride1);
        outs.y = process_single_token<HAS_PREFILL>(toks.y, req, is_prefill, prefill_req_id, block_table_ptr, workspace_starts_ptr, max_num_blocks_per_req, BLOCK_SIZE, bt_stride0, bt_stride1);
        outs.z = process_single_token<HAS_PREFILL>(toks.z, req, is_prefill, prefill_req_id, block_table_ptr, workspace_starts_ptr, max_num_blocks_per_req, BLOCK_SIZE, bt_stride0, bt_stride1);
        outs.w = process_single_token<HAS_PREFILL>(toks.w, req, is_prefill, prefill_req_id, block_table_ptr, workspace_starts_ptr, max_num_blocks_per_req, BLOCK_SIZE, bt_stride0, bt_stride1);
        
        *reinterpret_cast<int4*>(out_ptr_curr) = outs;
    } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (indice_id + i < num_topk_tokens) {
                int tok = token_indices_ptr[token_id * ti_stride0 + (indice_id + i) * ti_stride1];
                out_ptr[token_id * out_stride0 + (indice_id + i) * out_stride1] = 
                    process_single_token<HAS_PREFILL>(tok, req, is_prefill, prefill_req_id, block_table_ptr, workspace_starts_ptr, max_num_blocks_per_req, BLOCK_SIZE, bt_stride0, bt_stride1);
            }
        }
    }
}

torch::Tensor convert_req_index_to_global_index(
    torch::Tensor req_id,
    torch::Tensor block_table,
    torch::Tensor token_indices,
    int BLOCK_SIZE,
    int NUM_TOPK_TOKENS,
    bool HAS_PREFILL_WORKSPACE,
    std::optional<torch::Tensor> prefill_workspace_request_ids,
    std::optional<torch::Tensor> prefill_workspace_starts)
{
    TORCH_CHECK(req_id.is_cuda() && req_id.dtype() == torch::kInt32);
    TORCH_CHECK(block_table.is_cuda() && block_table.dtype() == torch::kInt32);
    TORCH_CHECK(token_indices.is_cuda() && token_indices.dtype() == torch::kInt32);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int num_tokens = req_id.size(0);
    int max_num_blocks_per_req = block_table.size(1);
    auto req_id_c = req_id.contiguous();
    auto block_table_c = block_table.contiguous();
    auto token_indices_c = token_indices.contiguous();
    auto out = torch::empty_like(token_indices_c);

    int bt_stride0 = block_table_c.stride(0);
    int bt_stride1 = block_table_c.stride(1);
    int ti_stride0 = token_indices_c.stride(0);
    int ti_stride1 = token_indices_c.stride(1);
    int out_stride0 = out.stride(0);
    int out_stride1 = out.stride(1);

    const int32_t* prefill_req_id_ptr = nullptr;
    const int32_t* workspace_starts_ptr = nullptr;

    if (HAS_PREFILL_WORKSPACE) {
        TORCH_CHECK(prefill_workspace_request_ids.has_value());
        TORCH_CHECK(prefill_workspace_starts.has_value());
        prefill_req_id_ptr = prefill_workspace_request_ids.value().contiguous().data_ptr<int32_t>();
        workspace_starts_ptr = prefill_workspace_starts.value().contiguous().data_ptr<int32_t>();
    }
    const int threads_per_block = 128;

    int num_chunks = (NUM_TOPK_TOKENS + 3) / 4;
    int blocks_x = (num_chunks + threads_per_block - 1) / threads_per_block;
    int blocks_y = num_tokens;
    
    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block);

    if (HAS_PREFILL_WORKSPACE) {
        convert_req_index_to_global_index_kernel<true><<<grid, block,0,stream>>>(
            req_id_c.data_ptr<int32_t>(),
            block_table_c.data_ptr<int32_t>(),
            token_indices_c.data_ptr<int32_t>(),
            out.data_ptr<int32_t>(),
            prefill_req_id_ptr,
            workspace_starts_ptr,
            max_num_blocks_per_req,
            BLOCK_SIZE,
            num_tokens,
            NUM_TOPK_TOKENS,
            bt_stride0, bt_stride1,
            ti_stride0, ti_stride1,
            out_stride0, out_stride1
        );
    } else {
        convert_req_index_to_global_index_kernel<false><<<grid, block,0,stream>>>(
            req_id_c.data_ptr<int32_t>(),
            block_table_c.data_ptr<int32_t>(),
            token_indices_c.data_ptr<int32_t>(),
            out.data_ptr<int32_t>(),
            nullptr, nullptr,
            max_num_blocks_per_req,
            BLOCK_SIZE,
            num_tokens,
            NUM_TOPK_TOKENS,
            bt_stride0, bt_stride1,
            ti_stride0, ti_stride1,
            out_stride0, out_stride1
        );
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));
    return out;
}

}
}