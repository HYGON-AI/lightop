
/* coding=utf-8
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <limits>
#include <stdint.h>
#include <ATen/Dispatch.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>
#include <hiprand_kernel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <thrust/pair.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/autograd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


#define Q(sq_id, b_id, head_id, hn_id) (Q + sq_id * b * np * hn + b_id * np * hn + head_id * hn + hn_id)
#define cos(sq_id, b_id, hn_id) (cos + sq_id * b * hn + b_id * hn + hn_id)
#define sin(sq_id, b_id, hn_id) (sin + sq_id * b * hn + b_id * hn + hn_id)
#define q_emb(sq_id, b_id, head_id, hn_id) (q_emb + sq_id * b * np * hn + b_id * np * hn + head_id * hn + hn_id)

#define res_grad(sq_id, b_id, head_id, hn_id) \
        (res_grad + sq_id * b * np * hn + b_id * np * hn + head_id * hn + hn_id)
#define grad_out(sq_id, b_id, head_id, hn_id) \
        (grad_out + sq_id * b * np * hn + b_id * np * hn + head_id * hn + hn_id)

template <typename T, int N>
struct alignas(sizeof(T) * N) Vector {
    T val[N];
};
template <typename Element, size_t len>
struct vec {
    using type = __attribute__((__vector_size__(len * sizeof(Element)))) Element;
};
namespace at{
namespace native{


// 相对传统的 RoPE 实现
// 输入： const scalar_t* Q, const scalar_t* cos, const scalar_t* sin
// 输出： scalar_t* q_emb
template <typename scalar_t, int BLOCK_NP, int BLOCK_HN, int VEC>
__global__ void RoPE_gpu_kernel(int sq, int b, int np, int hn,
                                const scalar_t* Q, const scalar_t* cos, const scalar_t* sin,
                                scalar_t* q_emb) {
    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC>;
    int sq_id = blockIdx.x, b_id = blockIdx.y;
    int np_id = threadIdx.y, hn_cid = threadIdx.x * VEC;  // 向量化访存
    scalar_t v_cos[VEC];
    scalar_t v_sin[VEC];
    scalar_t v_q[VEC];
    scalar_t v_q_rotate[VEC];
    scalar_t v_q_emb[VEC];
    #pragma unroll
    for (int head_id = np_id; head_id < np; head_id += BLOCK_NP) {
        #pragma unroll
        for(int hn_id = hn_cid; hn_id < hn; hn_id += BLOCK_HN * VEC) {
            *(LoadT*)v_cos = *(LoadT*)cos(sq_id, b_id, hn_id);
            *(LoadT*)v_sin = *(LoadT*)sin(sq_id, b_id, hn_id);
            *(LoadT*)v_q = *(LoadT*)Q(sq_id, b_id, head_id, hn_id);
            *(LoadT*)v_q_rotate = (hn_id + hn / 2 < hn) ? *(LoadT*)Q(sq_id, b_id, head_id, hn_id + hn / 2) : *(LoadT*)Q(sq_id, b_id, head_id, hn_id + hn / 2 - hn);
            #pragma unroll
            for (int vec_id = 0; vec_id < VEC; vec_id++) {
                v_q_emb[vec_id] = v_q[vec_id] * v_cos[vec_id] + ((hn_id + hn / 2 < hn) ? -v_q_rotate[vec_id] : v_q_rotate[vec_id]) * v_sin[vec_id];
            }
            *(LoadT*)q_emb(sq_id, b_id, head_id, hn_id) = *(LoadT*)v_q_emb;
        }
    }
}



template <typename scalar_t>
void host_apply_RoPE_gpu(int sq, int b, int np, int hn,
                         const scalar_t* Q, const scalar_t* cos,
                         const scalar_t* sin, scalar_t* q_emb) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 blocks(sq, b);

    if(np <= 16) {
        dim3 threads(16, 16);
        RoPE_gpu_kernel<scalar_t, 16, 16, 4><<<blocks, threads, 0, stream>>>(sq, b, np, hn, Q, cos, sin, q_emb);
    } else {
        dim3 threads(32, 32);
        RoPE_gpu_kernel<scalar_t, 32, 32, 4><<<blocks, threads, 0, stream>>>(sq, b, np, hn, Q, cos, sin, q_emb);
    }


}
void RoPE_gpu_launch(int sq, int b, int np, int hn, at::Tensor* Q,
                     at::Tensor* cos, at::Tensor* sin, at::Tensor* q_emb) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        cos->scalar_type(), "RoPE_forward_gpu",
        [&] {
        host_apply_RoPE_gpu(
            sq, b, np, hn, Q->data_ptr<scalar_t>(),
            cos->data_ptr<scalar_t>(), sin->data_ptr<scalar_t>(), q_emb->data_ptr<scalar_t>());
        });
}

template <typename scalar_t, int BLOCK_NP, int BLOCK_HN, int VEC>
__global__ void RoPE_backward_gpu_kernel(int sq, int b, int np,
                                         int hn, const scalar_t* grad_out,
                                         const scalar_t* cos, const scalar_t* sin,
                                         scalar_t* res_grad) {
    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC>;
    int sq_id = blockIdx.x, b_id = blockIdx.y;
    int np_id = threadIdx.y, hn_cid = threadIdx.x * VEC;
    scalar_t v_cos[VEC];
    scalar_t v_sin[VEC];
    scalar_t v_grad_out[VEC];
    scalar_t v_grad_out_rotate[VEC];
    scalar_t v_res_grad[VEC];
    #pragma unroll
    for (int head_id = np_id; head_id < np; head_id += BLOCK_NP) {
        #pragma unroll
        for(int hn_id = hn_cid; hn_id < hn; hn_id += BLOCK_HN * VEC) {
            *(LoadT*)v_cos = *(LoadT*)cos(sq_id, b_id, hn_id);
            *(LoadT*)v_sin = (hn_id + hn / 2 < hn) ? *(LoadT*)sin(sq_id, b_id, hn_id + hn / 2) : *(LoadT*)sin(sq_id, b_id, hn_id + hn / 2 - hn);
            *(LoadT*)v_grad_out = *(LoadT*)grad_out(sq_id, b_id, head_id, hn_id);
            *(LoadT*)v_grad_out_rotate = (hn_id + hn / 2 < hn) ? *(LoadT*)grad_out(sq_id, b_id, head_id, hn_id + hn / 2) : *(LoadT*)grad_out(sq_id, b_id, head_id, hn_id + hn / 2 - hn);
            #pragma unroll
            for (int vec_id = 0; vec_id < VEC; vec_id++) {
                v_res_grad[vec_id] = v_grad_out[vec_id] * v_cos[vec_id] + v_grad_out_rotate[vec_id] * ((hn_id + hn / 2 < hn) ? v_sin[vec_id] : -v_sin[vec_id]);
            }
            *(LoadT*)res_grad(sq_id, b_id, head_id, hn_id) = *(LoadT*)v_res_grad;
        }
    }
}

// 一次 kernel 同时处理 Q 和 K
template <typename scalar_t, bool IS_NEOX, int num_tokens_per_block,
          int total_thread_per_tokens, int num_group_per_token_q,
          int num_group_per_token_k, int thread_per_head_q,
          int thread_per_head_k, int VEC_SIZE_Q, int VEC_SIZE_K, int Rot_dim>
__global__ void rotary_embedding_kernel_opt(
    const int64_t* __restrict__ positions,      // [batch_size, seq_len] or [num_tokens]
    scalar_t* __restrict__ query,              // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,                // nullptr or [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim / 2]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int num_kv_heads,
    const int head_size, const int num_tokens) {
    const int token_idx =
        blockIdx.x * num_tokens_per_block + threadIdx.x / total_thread_per_tokens;
    if (token_idx >= num_tokens) {
        return;
    }
    int64_t pos = positions[token_idx];
    const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

    const int embed_dim = rot_dim / 2;

    using vector_q = Vector<scalar_t, VEC_SIZE_Q>;
    using vector_k = Vector<scalar_t, VEC_SIZE_K>;

    const int group_id_q = (threadIdx.x / thread_per_head_q) % num_group_per_token_q;
    const int local_id_q = threadIdx.x % thread_per_head_q;
    const int group_id_k = (threadIdx.x / thread_per_head_k) % num_group_per_token_k;
    const int local_id_k = threadIdx.x % thread_per_head_k;
    const int rot_offset_q = local_id_q * VEC_SIZE_Q;
    int rot_offset_k = local_id_k * VEC_SIZE_K;

    // for seme
    const int local_token_idx = threadIdx.x / total_thread_per_tokens;
    const int laneid = threadIdx.x & (total_thread_per_tokens - 1);
    const int thread_per_load_num = Rot_dim / total_thread_per_tokens;
    __shared__ scalar_t cos_sin_seme[num_tokens_per_block][Rot_dim];

#pragma unroll
    for (int i = 0; i < thread_per_load_num; i++) {
        cos_sin_seme[local_token_idx][laneid * thread_per_load_num + i] = cache_ptr[laneid * thread_per_load_num + i];
    }
    __syncthreads();

    if constexpr(IS_NEOX) {  // gpt-neox style
        const vector_q cos_vec = *reinterpret_cast<const vector_q*>(&cos_sin_seme[local_token_idx][rot_offset_q]);
        const vector_q sin_vec = *reinterpret_cast<const vector_q*>(&cos_sin_seme[local_token_idx][rot_offset_q + embed_dim]);

        for (int head_idx = group_id_q; head_idx < num_heads; head_idx += num_group_per_token_q) {
            const int64_t token_head_base = token_idx * query_stride + head_idx * head_stride;

            auto x_ptr = reinterpret_cast<vector_q*>(&query[token_head_base + rot_offset_q]);
            auto y_ptr = reinterpret_cast<vector_q*>(&query[token_head_base + rot_offset_q + embed_dim]);

            const vector_q x = *x_ptr;
            const vector_q y = *y_ptr;

            vector_q x_out;
            vector_q y_out;

#pragma unroll
            for (int i = 0; i < VEC_SIZE_Q; i++) {
                y_out.val[i] = fma(y.val[i], cos_vec.val[i], x.val[i] * sin_vec.val[i]);
                x_out.val[i] = fma(x.val[i], cos_vec.val[i], -y.val[i] * sin_vec.val[i]);
            }

            *x_ptr = x_out;
            *y_ptr = y_out;
        }

        if (key != nullptr) {
            const vector_k cos_vec_k = *reinterpret_cast<const vector_k*>(&cos_sin_seme[local_token_idx][rot_offset_k]);
            const vector_k sin_vec_k = *reinterpret_cast<const vector_k*>(&cos_sin_seme[local_token_idx][rot_offset_k + embed_dim]);
            for (int head_idx = group_id_k; head_idx < num_kv_heads; head_idx += num_group_per_token_k) {
                const int64_t token_head_base = token_idx * key_stride + head_idx * head_stride;

                auto x_ptr = reinterpret_cast<vector_k*>(&key[token_head_base + rot_offset_k]);
                auto y_ptr = reinterpret_cast<vector_k*>(&key[token_head_base + rot_offset_k + embed_dim]);

                const vector_k x = *x_ptr;
                const vector_k y = *y_ptr;

                vector_k x_out;
                vector_k y_out;

#pragma unroll
                for (int i = 0; i < VEC_SIZE_K; i++) {
                    y_out.val[i] = fma(y.val[i], cos_vec_k.val[i], x.val[i] * sin_vec_k.val[i]);
                    x_out.val[i] = fma(x.val[i], cos_vec_k.val[i], -y.val[i] * sin_vec_k.val[i]);
                }
                *x_ptr = x_out;
                *y_ptr = y_out;
            }
        }
    } else {  // gpt-j style
        for (int head_idx = group_id_q; head_idx < num_heads; head_idx += num_group_per_token_q) {

            const int current_rot_offset = local_id_q * VEC_SIZE_Q * 2;

            if (current_rot_offset >= rot_dim) {
                continue;
            }

            const int64_t token_head_base = token_idx * query_stride + head_idx * head_stride;

            auto* x_ptr = reinterpret_cast<vector_q*>(&query[token_head_base + current_rot_offset]);
            auto* y_ptr = reinterpret_cast<vector_q*>(&query[token_head_base + current_rot_offset + VEC_SIZE_Q]);

            const vector_q x_in = *x_ptr;
            const vector_q y_in = *y_ptr;
            vector_q x_out;
            vector_q y_out;


            #pragma unroll
            for (int i = 0; i < VEC_SIZE_Q; i += 2) {
                const int cache_idx = (current_rot_offset + i) / 2;
                const scalar_t cos_val = cos_sin_seme[local_token_idx][cache_idx];
                const scalar_t sin_val = cos_sin_seme[local_token_idx][cache_idx + embed_dim];
                const scalar_t x0 = x_in.val[i];
                const scalar_t x1 = x_in.val[i + 1];
                x_out.val[i]     = fma(x0, cos_val, -x1 * sin_val);
                x_out.val[i + 1] = fma(x1, cos_val, x0 * sin_val);
            }

            #pragma unroll
            for (int i = 0; i < VEC_SIZE_Q; i += 2) {
                const int cache_idx = (current_rot_offset + VEC_SIZE_Q + i) / 2;
                const scalar_t cos_val = cos_sin_seme[local_token_idx][cache_idx];
                const scalar_t sin_val = cos_sin_seme[local_token_idx][cache_idx + embed_dim];
                const scalar_t y0 = y_in.val[i];
                const scalar_t y1 = y_in.val[i + 1];
                y_out.val[i]     = fma(y0, cos_val, -y1 * sin_val);
                y_out.val[i + 1] = fma(y1, cos_val, y0 * sin_val);
            }

            *x_ptr = x_out;
            *y_ptr = y_out;
        }
        if (key != nullptr) {
            for (int head_idx = group_id_k; head_idx < num_kv_heads; head_idx += num_group_per_token_k) {
                const int current_rot_offset = local_id_k * VEC_SIZE_K * 2;
                if (current_rot_offset >= rot_dim) {
                    continue;
                }
                if constexpr(VEC_SIZE_K == 1){
                    const int64_t token_head_base = token_idx * key_stride + head_idx * head_stride;
                    using vector_2 = Vector<scalar_t, 2>;
                    auto* x_ptr = reinterpret_cast<vector_2*>(&key[token_head_base + current_rot_offset]);
                    const vector_2 x_in = *x_ptr;
                    vector_2 x_out;
                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE_K; i += 2) {
                        const int cache_idx = (current_rot_offset + i) / 2;
                        const scalar_t cos_val = cos_sin_seme[local_token_idx][cache_idx];
                        const scalar_t sin_val = cos_sin_seme[local_token_idx][cache_idx + embed_dim];
                        const scalar_t x0 = x_in.val[i];
                        const scalar_t x1 = x_in.val[i + 1];
                        x_out.val[i]     = fma(x0, cos_val, -x1 * sin_val);
                        x_out.val[i + 1] = fma(x1, cos_val, x0 * sin_val);
                    }
                    *x_ptr = x_out;
                }else{
                    const int64_t token_head_base = token_idx * key_stride + head_idx * head_stride;

                    auto* x_ptr = reinterpret_cast<vector_k*>(&key[token_head_base + current_rot_offset]);
                    auto* y_ptr = reinterpret_cast<vector_k*>(&key[token_head_base + current_rot_offset + VEC_SIZE_K]);

                    const vector_k x_in = *x_ptr;
                    const vector_k y_in = *y_ptr;
                    vector_k x_out;
                    vector_k y_out;

                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE_K; i += 2) {
                        const int cache_idx = (current_rot_offset + i) / 2;
                        const scalar_t cos_val = cos_sin_seme[local_token_idx][cache_idx];
                        const scalar_t sin_val = cos_sin_seme[local_token_idx][cache_idx + embed_dim];
                        const scalar_t x0 = x_in.val[i];
                        const scalar_t x1 = x_in.val[i + 1];
                        x_out.val[i]     = fma(x0, cos_val, -x1 * sin_val);
                        x_out.val[i + 1] = fma(x1, cos_val, x0 * sin_val);
                    }

                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE_K; i += 2) {
                        const int cache_idx = (current_rot_offset + VEC_SIZE_K + i) / 2;
                        const scalar_t cos_val = cos_sin_seme[local_token_idx][cache_idx];
                        const scalar_t sin_val = cos_sin_seme[local_token_idx][cache_idx + embed_dim];
                        const scalar_t y0 = y_in.val[i];
                        const scalar_t y1 = y_in.val[i + 1];
                        y_out.val[i]     = fma(y0, cos_val, -y1 * sin_val);
                        y_out.val[i + 1] = fma(y1, cos_val, y0 * sin_val);
                    }

                    *x_ptr = x_out;
                    *y_ptr = y_out;
                }
            }
        }
    }
}

//deepseek 融合q、k
template <typename scalar_t, bool IS_NEOX, int num_tokens_per_block,
          int total_thread_per_tokens, int num_group_per_token_q,
          int num_group_per_token_k, int thread_per_head_q,
          int thread_per_head_k, int VEC_SIZE_Q, int VEC_SIZE_K, int Rot_dim>
__global__ void rotary_embedding_deepseek_kernel_opt(
    const int64_t* __restrict__ positions,      // [batch_size, seq_len] or [num_tokens]
    scalar_t* __restrict__ query,              // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,                // nullptr or [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim / 2]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int num_kv_heads,
    const int head_size, const int num_tokens,
    int64_t* __restrict__ offsets) {
    const int token_idx =
        blockIdx.x * num_tokens_per_block + threadIdx.x / total_thread_per_tokens;
    if (token_idx >= num_tokens) {
        return;
    }
    int64_t pos = positions[token_idx];
    //lj
    if (offsets != nullptr) {
        pos += offsets[token_idx];
    }
    //lj
    const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

    const int embed_dim = rot_dim / 2;

    using vector_q = Vector<scalar_t, VEC_SIZE_Q>;
    using vector_k = Vector<scalar_t, VEC_SIZE_K>;

    const int group_id_q = (threadIdx.x / thread_per_head_q) % num_group_per_token_q;
    const int local_id_q = threadIdx.x % thread_per_head_q;
    const int group_id_k = (threadIdx.x / thread_per_head_k) % num_group_per_token_k;
    const int local_id_k = threadIdx.x % thread_per_head_k;
    const int rot_offset_q = local_id_q * VEC_SIZE_Q;
    int rot_offset_k = local_id_k * VEC_SIZE_K;

    // for seme
    const int local_token_idx = threadIdx.x / total_thread_per_tokens;
    const int laneid = threadIdx.x & (total_thread_per_tokens - 1);
    const int thread_per_load_num = Rot_dim / total_thread_per_tokens;
    __shared__ scalar_t cos_sin_seme[num_tokens_per_block][Rot_dim];

#pragma unroll
    for (int i = 0; i < thread_per_load_num; i++) {
        cos_sin_seme[local_token_idx][laneid * thread_per_load_num + i] = cache_ptr[laneid * thread_per_load_num + i];
    }
    __syncthreads();

    if constexpr(IS_NEOX) {  // gpt-neox style
        const vector_q cos_vec = *reinterpret_cast<const vector_q*>(&cos_sin_seme[local_token_idx][rot_offset_q]);
        const vector_q sin_vec = *reinterpret_cast<const vector_q*>(&cos_sin_seme[local_token_idx][rot_offset_q + embed_dim]);

        for (int head_idx = group_id_q; head_idx < num_heads; head_idx += num_group_per_token_q) {
            const int64_t token_head_base = token_idx * query_stride + head_idx * head_stride;

            auto x_ptr = reinterpret_cast<vector_q*>(&query[token_head_base + rot_offset_q]);
            auto y_ptr = reinterpret_cast<vector_q*>(&query[token_head_base + rot_offset_q + embed_dim]);

            const vector_q x = *x_ptr;
            const vector_q y = *y_ptr;

            vector_q x_out;
            vector_q y_out;

#pragma unroll
            for (int i = 0; i < VEC_SIZE_Q; i++) {
                x_out.val[i] = x.val[i] * cos_vec.val[i] - y.val[i] * sin_vec.val[i];
                y_out.val[i] = y.val[i] * cos_vec.val[i] + x.val[i] * sin_vec.val[i];
            }

            *x_ptr = x_out;
            *y_ptr = y_out;
        }

        if (key != nullptr) {
            const vector_k cos_vec_k = *reinterpret_cast<const vector_k*>(&cos_sin_seme[local_token_idx][rot_offset_k]);
            const vector_k sin_vec_k = *reinterpret_cast<const vector_k*>(&cos_sin_seme[local_token_idx][rot_offset_k + embed_dim]);
            for (int head_idx = group_id_k; head_idx < num_kv_heads; head_idx += num_group_per_token_k) {
                const int64_t token_head_base = token_idx * key_stride + head_idx * head_stride;

                auto x_ptr = reinterpret_cast<vector_k*>(&key[token_head_base + rot_offset_k]);
                auto y_ptr = reinterpret_cast<vector_k*>(&key[token_head_base + rot_offset_k + embed_dim]);

                const vector_k x = *x_ptr;
                const vector_k y = *y_ptr;

                vector_k x_out;
                vector_k y_out;

#pragma unroll
                for (int i = 0; i < VEC_SIZE_K; i++) {
                    y_out.val[i] = y.val[i] * cos_vec_k.val[i] + x.val[i] * sin_vec_k.val[i];
                    x_out.val[i] = x.val[i] * cos_vec_k.val[i] - y.val[i] * sin_vec_k.val[i];
                }
                *x_ptr = x_out;
                *y_ptr = y_out;
            }
        }
    } else {  // gpt-j style
        for (int head_idx = group_id_q; head_idx < num_heads; head_idx += num_group_per_token_q) {

            const int current_rot_offset = local_id_q * VEC_SIZE_Q * 2;

            if (current_rot_offset >= rot_dim) {
                continue;
            }

            const int64_t token_head_base = token_idx * query_stride + head_idx * head_stride;

            auto* x_ptr = reinterpret_cast<vector_q*>(&query[token_head_base + current_rot_offset]);
            auto* y_ptr = reinterpret_cast<vector_q*>(&query[token_head_base + current_rot_offset + VEC_SIZE_Q]);

            const vector_q x_in = *x_ptr;
            const vector_q y_in = *y_ptr;
            vector_q x_out;
            vector_q y_out;


            #pragma unroll
            for (int i = 0; i < VEC_SIZE_Q; i += 2) {
                const int cache_idx = (current_rot_offset + i) / 2;
                const scalar_t cos_val = cos_sin_seme[local_token_idx][cache_idx];
                const scalar_t sin_val = cos_sin_seme[local_token_idx][cache_idx + embed_dim];
                const scalar_t x0 = x_in.val[i];
                const scalar_t x1 = x_in.val[i + 1];
                x_out.val[i]     = x0 * cos_val - x1 * sin_val;
                x_out.val[i + 1] = x1 * cos_val + x0 * sin_val;
            }

            #pragma unroll
            for (int i = 0; i < VEC_SIZE_Q; i += 2) {
                const int cache_idx = (current_rot_offset + VEC_SIZE_Q + i) / 2;
                const scalar_t cos_val = cos_sin_seme[local_token_idx][cache_idx];
                const scalar_t sin_val = cos_sin_seme[local_token_idx][cache_idx + embed_dim];
                const scalar_t y0 = y_in.val[i];
                const scalar_t y1 = y_in.val[i + 1];
                y_out.val[i]     = y0 * cos_val - y1 * sin_val;
                y_out.val[i + 1] = y1 * cos_val + y0 * sin_val;
            }

            *x_ptr = x_out;
            *y_ptr = y_out;
        }
        if (key != nullptr) {
            for (int head_idx = group_id_k; head_idx < num_kv_heads; head_idx += num_group_per_token_k) {
                const int current_rot_offset = local_id_k * VEC_SIZE_K * 2;
                if (current_rot_offset >= rot_dim) {
                    continue;
                }
                if constexpr(VEC_SIZE_K == 1){
                    const int64_t token_head_base = token_idx * key_stride + head_idx * head_stride;
                    using vector_2 = Vector<scalar_t, 2>;
                    auto* x_ptr = reinterpret_cast<vector_2*>(&key[token_head_base + current_rot_offset]);
                    const vector_2 x_in = *x_ptr;
                    vector_2 x_out;
                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE_K; i += 2) {
                        const int cache_idx = (current_rot_offset + i) / 2;
                        const scalar_t cos_val = cos_sin_seme[local_token_idx][cache_idx];
                        const scalar_t sin_val = cos_sin_seme[local_token_idx][cache_idx + embed_dim];
                        const scalar_t x0 = x_in.val[i];
                        const scalar_t x1 = x_in.val[i + 1];
                        x_out.val[i]     = x0 * cos_val - x1 * sin_val;
                        x_out.val[i + 1] = x1 * cos_val + x0 * sin_val;
                    }
                    *x_ptr = x_out;
                }else{
                    const int64_t token_head_base = token_idx * key_stride + head_idx * head_stride;

                    auto* x_ptr = reinterpret_cast<vector_k*>(&key[token_head_base + current_rot_offset]);
                    auto* y_ptr = reinterpret_cast<vector_k*>(&key[token_head_base + current_rot_offset + VEC_SIZE_K]);

                    const vector_k x_in = *x_ptr;
                    const vector_k y_in = *y_ptr;
                    vector_k x_out;
                    vector_k y_out;

                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE_K; i += 2) {
                        const int cache_idx = (current_rot_offset + i) / 2;
                        const scalar_t cos_val = cos_sin_seme[local_token_idx][cache_idx];
                        const scalar_t sin_val = cos_sin_seme[local_token_idx][cache_idx + embed_dim];
                        const scalar_t x0 = x_in.val[i];
                        const scalar_t x1 = x_in.val[i + 1];
                        x_out.val[i]     = x0 * cos_val - x1 * sin_val;
                        x_out.val[i + 1] = x1 * cos_val + x0 * sin_val;
                    }

                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE_K; i += 2) {
                        const int cache_idx = (current_rot_offset + VEC_SIZE_K + i) / 2;
                        const scalar_t cos_val = cos_sin_seme[local_token_idx][cache_idx];
                        const scalar_t sin_val = cos_sin_seme[local_token_idx][cache_idx + embed_dim];
                        const scalar_t y0 = y_in.val[i];
                        const scalar_t y1 = y_in.val[i + 1];
                        y_out.val[i]     = y0 * cos_val - y1 * sin_val;
                        y_out.val[i + 1] = y1 * cos_val + y0 * sin_val;
                    }

                    *x_ptr = x_out;
                    *y_ptr = y_out;
                }
            }
        }
    }
}


template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding_head72(
    scalar_t* __restrict__ tensor,
    const scalar_t* __restrict__ cos_sin,
    const int token_idx, const int64_t token_stride,
    const int64_t head_stride, const int num_heads,
    const int group_id, const int local_id, const int num_groups) {
  constexpr int kHeadSize = 72;
  constexpr int kEmbedDim = kHeadSize / 2;
  constexpr int kVecSize = 4;
  using vector_t = Vector<scalar_t, kVecSize>;

  const int rot_offset = local_id * kVecSize;
  if constexpr (IS_NEOX) {
    const vector_t cos_vec =
        *reinterpret_cast<const vector_t*>(cos_sin + rot_offset);
    const vector_t sin_vec =
        *reinterpret_cast<const vector_t*>(cos_sin + kEmbedDim + rot_offset);
    for (int head_idx = group_id; head_idx < num_heads;
         head_idx += num_groups) {
      const int64_t base =
          token_idx * token_stride + head_idx * head_stride;
      auto* x_ptr = reinterpret_cast<vector_t*>(tensor + base + rot_offset);
      auto* y_ptr = reinterpret_cast<vector_t*>(
          tensor + base + kEmbedDim + rot_offset);
      const vector_t x = *x_ptr;
      const vector_t y = *y_ptr;
      vector_t x_out;
      vector_t y_out;
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
        x_out.val[i] = x.val[i] * cos_vec.val[i] -
                       y.val[i] * sin_vec.val[i];
        y_out.val[i] = y.val[i] * cos_vec.val[i] +
                       x.val[i] * sin_vec.val[i];
      }
      *x_ptr = x_out;
      *y_ptr = y_out;
    }
  } else {
    const int pair_offset = local_id * kVecSize * 2;
    for (int head_idx = group_id; head_idx < num_heads;
         head_idx += num_groups) {
      const int64_t base =
          token_idx * token_stride + head_idx * head_stride;
      auto* x_ptr = reinterpret_cast<vector_t*>(tensor + base + pair_offset);
      auto* y_ptr = reinterpret_cast<vector_t*>(
          tensor + base + pair_offset + kVecSize);
      const vector_t x = *x_ptr;
      const vector_t y = *y_ptr;
      vector_t x_out;
      vector_t y_out;
#pragma unroll
      for (int i = 0; i < kVecSize; i += 2) {
        const int cache_idx = (pair_offset + i) / 2;
        const scalar_t cos_val = cos_sin[cache_idx];
        const scalar_t sin_val = cos_sin[kEmbedDim + cache_idx];
        x_out.val[i] = x.val[i] * cos_val - x.val[i + 1] * sin_val;
        x_out.val[i + 1] =
            x.val[i + 1] * cos_val + x.val[i] * sin_val;
      }
#pragma unroll
      for (int i = 0; i < kVecSize; i += 2) {
        const int cache_idx = (pair_offset + kVecSize + i) / 2;
        const scalar_t cos_val = cos_sin[cache_idx];
        const scalar_t sin_val = cos_sin[kEmbedDim + cache_idx];
        y_out.val[i] = y.val[i] * cos_val - y.val[i + 1] * sin_val;
        y_out.val[i + 1] =
            y.val[i + 1] * cos_val + y.val[i] * sin_val;
      }
      *x_ptr = x_out;
      *y_ptr = y_out;
    }
  }
}

template <typename scalar_t, bool IS_NEOX, int NUM_GROUPS>
__global__ void rotary_embedding_head72_kernel(
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* __restrict__ cos_sin_cache,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t query_head_stride,
    const int64_t key_head_stride,
    const int num_heads,
    const int num_kv_heads,
    const int num_tokens,
    const int64_t* __restrict__ offsets) {

  constexpr int kRotDim = 72;
  constexpr int kVecSize = 4;
  constexpr int kTokensPerBlock = 2;

  constexpr int kThreadsPerHead =
      (kRotDim / 2) / kVecSize;

  static_assert(
      kThreadsPerHead == 9,
      "head-72 vec4 tiling must be exact");

  constexpr int kThreadsPerToken =
      NUM_GROUPS * kThreadsPerHead;

  constexpr int kNumCacheVec =
      kRotDim / kVecSize;  // 18


  const int local_token_idx =
      threadIdx.x / kThreadsPerToken;

  const int local_tid =
      threadIdx.x % kThreadsPerToken;

  const int token_idx =
      blockIdx.x * kTokensPerBlock +
      local_token_idx;

  const bool valid_token =
      token_idx < num_tokens;


  __shared__ Vector<scalar_t, kVecSize>
      cos_sin_shared_vec
          [kTokensPerBlock]
          [kNumCacheVec];

  scalar_t* cos_sin_shared =
      reinterpret_cast<scalar_t*>(
          &cos_sin_shared_vec
              [local_token_idx][0]);

  int64_t pos = 0;

  if (valid_token) {
    pos = positions[token_idx];

    if (offsets != nullptr) {
      pos += offsets[token_idx];
    }

    const auto* cache_vec =
        reinterpret_cast<
            const Vector<scalar_t, kVecSize>*>(
                cos_sin_cache +
                pos * kRotDim);

    if (local_tid < kNumCacheVec) {
      cos_sin_shared_vec
          [local_token_idx]
          [local_tid] =
              cache_vec[local_tid];
    }
  }

  __syncthreads();

  if (!valid_token) {
    return;
  }

  const int group_id =
      local_tid / kThreadsPerHead;

  const int local_id =
      local_tid % kThreadsPerHead;

  apply_rotary_embedding_head72<
      scalar_t,
      IS_NEOX>(
          query,
          cos_sin_shared,
          token_idx,
          query_stride,
          query_head_stride,
          num_heads,
          group_id,
          local_id,
          NUM_GROUPS);

  if (key != nullptr) {
    apply_rotary_embedding_head72<
        scalar_t,
        IS_NEOX>(
            key,
            cos_sin_shared,
            token_idx,
            key_stride,
            key_head_stride,
            num_kv_heads,
            group_id,
            local_id,
            NUM_GROUPS);
  }
}

template <typename scalar_t, bool IS_NEOX>
void launch_rotary_embedding_head72(
    const int64_t* positions,
    scalar_t* query,
    scalar_t* key,
    const scalar_t* cos_sin_cache,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t query_head_stride,
    const int64_t key_head_stride,
    const int num_heads,
    const int num_kv_heads,
    const int num_tokens,
    const int64_t* offsets,
    cudaStream_t stream) {

  constexpr int kTokensPerBlock = 2;
  constexpr int kThreadsPerHead = 9;

  dim3 grid(
      (num_tokens + kTokensPerBlock - 1) /
      kTokensPerBlock);

  if (num_heads <= 8) {

    constexpr int kGroups = 8;

    dim3 block(
        kGroups *
        kThreadsPerHead *
        kTokensPerBlock);

    rotary_embedding_head72_kernel<
        scalar_t,
        IS_NEOX,
        kGroups>
        <<<grid, block, 0, stream>>>(
            positions,
            query,
            key,
            cos_sin_cache,
            query_stride,
            key_stride,
            query_head_stride,
            key_head_stride,
            num_heads,
            num_kv_heads,
            num_tokens,
            offsets);

  } else if (num_heads <= 16) {

    constexpr int kGroups = 16;
    dim3 block(
        kGroups *
        kThreadsPerHead *
        kTokensPerBlock);

    rotary_embedding_head72_kernel<
        scalar_t,
        IS_NEOX,
        kGroups>
        <<<grid, block, 0, stream>>>(
            positions,
            query,
            key,
            cos_sin_cache,
            query_stride,
            key_stride,
            query_head_stride,
            key_head_stride,
            num_heads,
            num_kv_heads,
            num_tokens,
            offsets);

  } else {

    constexpr int kGroups = 32;

    dim3 block(
        kGroups *
        kThreadsPerHead *
        kTokensPerBlock);

    rotary_embedding_head72_kernel<
        scalar_t,
        IS_NEOX,
        kGroups>
        <<<grid, block, 0, stream>>>(
            positions,
            query,
            key,
            cos_sin_cache,
            query_stride,
            key_stride,
            query_head_stride,
            key_head_stride,
            num_heads,
            num_kv_heads,
            num_tokens,
            offsets);
  }
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr, const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr, int rot_offset, int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = cos_ptr[x_index];
    sin = sin_ptr[x_index];
  } else {
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = cos_ptr[x_index / 2]; //VLLM_LDG(cos_ptr + x_index / 2);
    sin = sin_ptr[x_index / 2]; //VLLM_LDG(sin_ptr + x_index / 2);
  }
  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* cache_ptr,
    const int head_size,
    const int num_heads,
    const int num_kv_heads,
    const int rot_dim,
    const int token_idx,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t head_stride) {

  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;
  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head =
        token_idx * query_stride + head_idx * head_stride;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);

  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int64_t token_head =
          token_idx * key_stride + head_idx * head_stride;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, IS_NEOX>(
          key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
    }
  }
}


// fallback
template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                           // [num_tokens]
    scalar_t* __restrict__ query,          // [batch_size, seq_len, num_heads,
                                           // head_size] or [num_tokens, num_heads,
                                           // head_size]
    scalar_t* __restrict__ key,  // nullptr or
                                 // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int num_kv_heads,
    const int head_size) {
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];

  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
  apply_rotary_embedding<scalar_t, IS_NEOX>(
      query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
      token_idx, query_stride, key_stride, head_stride);
}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel_deepseek(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                           // [num_tokens]
    scalar_t* __restrict__ query,          // [batch_size, seq_len, num_heads,
                                           // head_size] or [num_tokens, num_heads,
                                           // head_size]
    scalar_t* __restrict__ key,  // nullptr or
                                 // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int num_kv_heads,
    const int head_size,
    int64_t* __restrict__ offsets) {
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
      if (offsets != nullptr) {
        pos += offsets[token_idx];
    }
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
  apply_rotary_embedding<scalar_t, IS_NEOX>(
      query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
      token_idx, query_stride, key_stride, head_stride);
}

template <typename scalar_t>
void host_apply_RoPE_backward_gpu(int sq, int b, int np, int hn,
                                  const scalar_t* grad_out, const scalar_t* cos,
                                  const scalar_t* sin, scalar_t* res_grad) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 blocks(sq, b);

    if(np <= 16) {
        dim3 threads(16, 16);
        RoPE_backward_gpu_kernel<scalar_t, 16, 16, 4><<<blocks, threads, 0, stream>>>(sq, b, np, hn, grad_out, cos, sin, res_grad);
    } else {
        dim3 threads(32, 32);
        RoPE_backward_gpu_kernel<scalar_t, 32, 32, 4><<<blocks, threads, 0, stream>>>(sq, b, np, hn, grad_out, cos, sin, res_grad);
    }
}

void RoPE_backward_gpu_launch(int sq, int b, int np, int hn,
                              at::Tensor* grad_out, at::Tensor* cos,
                              at::Tensor* sin, at::Tensor* res_grad) {

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        cos->scalar_type(), "RoPE_backward_gpu",
        [&] {
            host_apply_RoPE_backward_gpu(
            sq, b, np, hn, grad_out->data_ptr<scalar_t>(),
            cos->data_ptr<scalar_t>(), sin->data_ptr<scalar_t>(),
            res_grad->data_ptr<scalar_t>());
        });
}

at::Tensor RoPE_forward_gpu(int sq, int b, int np, int hn, at::Tensor tensor_q,
                            at::Tensor tensor_cos, at::Tensor tensor_sin) {
    at::Tensor tensor_q_emb =
        at::empty({sq, b, np, hn},
                  at::device(torch::kCUDA).dtype(tensor_cos.scalar_type()));

    RoPE_gpu_launch(sq, b, np, hn, &tensor_q, &tensor_cos, &tensor_sin, &tensor_q_emb);
    return tensor_q_emb;
}

at::Tensor RoPE_backward_gpu(int sq, int b, int np, int hn,
                             at::Tensor tensor_grad_out, at::Tensor tensor_cos,
                             at::Tensor tensor_sin) {
    at::Tensor tensor_res =
        at::empty({sq, b, np, hn},
                  at::device(torch::kCUDA).dtype(tensor_cos.scalar_type()));
    RoPE_backward_gpu_launch(sq, b, np, hn, &tensor_grad_out, &tensor_cos,
                             &tensor_sin, &tensor_res);
    return tensor_res;
}

//融合qk
void rotary_embedding_fuse(
    torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& query,  // [batch_size, seq_len, num_heads * head_size] or
                           // [num_tokens, num_heads * head_size] or
                           // [batch_size, seq_len, num_heads, head_size] or
                           // [num_tokens, num_heads, head_size]
    std::optional<torch::Tensor> key,
    // null or
    // [batch_size, seq_len, num_kv_heads * head_size] or
    // [num_tokens, num_kv_heads * head_size] or
    // [batch_size, seq_len, num_heads, head_size] or
    // [num_tokens, num_heads, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox) {

    int64_t num_tokens = positions.numel();
    int positions_ndim = positions.dim();
  TORCH_CHECK(
      positions_ndim == 1 || positions_ndim == 2,
      "positions must have shape [num_tokens] or [batch_size, seq_len]");
  if (positions_ndim == 1) {
    TORCH_CHECK(query.size(0) == positions.size(0) &&
                    (!key.has_value() || key->size(0) == positions.size(0)),
                "query, key and positions must have the same number of tokens");
  }
  if (positions_ndim == 2) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            (!key.has_value() || key->size(0) == positions.size(0)) &&
            query.size(1) == positions.size(1) &&
            (!key.has_value() || key->size(1) == positions.size(1)),
        "query, key and positions must have the same batch_size and seq_len");
  }

  int query_hidden_size = query.numel() / num_tokens;
  int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;

  TORCH_CHECK(query_hidden_size % head_size == 0);
  TORCH_CHECK(key_hidden_size % head_size == 0);

  int num_heads = query_hidden_size / head_size;
  int num_kv_heads = key.has_value() ? key_hidden_size / head_size : num_heads;
  TORCH_CHECK(num_heads % num_kv_heads == 0);


  int rot_dim = cos_sin_cache.size(1);
  if (head_size == 72) {
    TORCH_CHECK(rot_dim > 0 && rot_dim <= head_size && rot_dim % 2 == 0,
                "rot_dim must be a positive even value no greater than "
                "head_size, but got rot_dim=", rot_dim,
                " and head_size=", head_size);
  }
  int seq_dim_idx = positions_ndim - 1;

  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;

  int query_ndim = query.dim();
  int64_t head_stride =
      (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;
  int64_t key_head_stride = key.has_value()
      ? ((key->dim() == positions_ndim + 2) ? key->stride(-2) : head_size)
      : head_stride;

  const bool has_dense_head72_tokens =
      positions_ndim == 1 ||
      (query.stride(0) == positions.size(1) * query_stride &&
       (!key.has_value() ||
        key->stride(0) == positions.size(1) * key_stride));
  const bool has_head72_supported_layout =
      head_size == 72 && rot_dim == 72 &&
      (query.dim() == positions_ndim + 1 ||
       query.dim() == positions_ndim + 2) &&
      (!key.has_value() || key->dim() == query.dim()) &&
      positions.is_contiguous() && query.stride(-1) == 1 &&
      (!key.has_value() || key->stride(-1) == 1) &&
      cos_sin_cache.stride(1) == 1 &&
      cos_sin_cache.stride(0) == rot_dim && has_dense_head72_tokens &&
      positions.scalar_type() == at::ScalarType::Long &&
      query.scalar_type() == cos_sin_cache.scalar_type() &&
      (!key.has_value() || query.scalar_type() == key->scalar_type()) &&
      positions.device() == query.device() &&
      cos_sin_cache.device() == query.device() &&
      (!key.has_value() || key->device() == query.device());

  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();


   AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(), "rotary_embedding", [&] {

  int rot_dim = cos_sin_cache.size(1);

  const int q_k_size_multiple = num_heads / num_kv_heads;
  bool is_allign_qk = (num_heads % num_kv_heads == 0) && (num_heads % 2 == 0)
                        && (num_heads >= 8) && ((q_k_size_multiple <= 8) || (q_k_size_multiple == 10));

  constexpr uintptr_t head72_vec4_alignment = alignof(Vector<scalar_t, 4>);
  const bool can_use_head72_vec4 =
      has_head72_supported_layout &&
      reinterpret_cast<uintptr_t>(query.data_ptr<scalar_t>()) %
              head72_vec4_alignment == 0 &&
      reinterpret_cast<uintptr_t>(cos_sin_cache.data_ptr<scalar_t>()) %
              head72_vec4_alignment == 0 &&
      query_stride % 4 == 0 && head_stride % 4 == 0 &&
      (!key.has_value() ||
       (reinterpret_cast<uintptr_t>(key->data_ptr<scalar_t>()) %
                head72_vec4_alignment == 0 &&
        key_stride % 4 == 0 && key_head_stride % 4 == 0));

    if (head_size == 72 && can_use_head72_vec4) {
        if (is_neox) {
          launch_rotary_embedding_head72<scalar_t, true>(
              positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
              key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
              cos_sin_cache.data_ptr<scalar_t>(), query_stride, key_stride,
              head_stride, key_head_stride, num_heads, num_kv_heads,
              num_tokens, nullptr, stream);
        } else {
          launch_rotary_embedding_head72<scalar_t, false>(
              positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
              key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
              cos_sin_cache.data_ptr<scalar_t>(), query_stride, key_stride,
              head_stride, key_head_stride, num_heads, num_kv_heads,
              num_tokens, nullptr, stream);
        }
    } else if (head_size == 64 && is_allign_qk) {
        constexpr int num_tokens_per_block = 2;
        constexpr int total_thread_per_tokens = 64;

        constexpr int num_group_per_token_q = 8;
        constexpr int thread_per_head_q = 8;
        constexpr int vec_size_q = 4;

        dim3 grid((num_tokens + num_tokens_per_block -1) / num_tokens_per_block);
        dim3 block(total_thread_per_tokens * num_tokens_per_block);


      #define LAUNCH_ROTARY_KERNEL_64(IS_NEOX,NUM_GROUP_K, THREAD_PER_HEAD_K, VEC_SIZE_K, ROT_DIM_VAL) \
        rotary_embedding_kernel_opt<scalar_t, IS_NEOX, num_tokens_per_block, total_thread_per_tokens, \
                                          num_group_per_token_q, NUM_GROUP_K, \
                                          thread_per_head_q, THREAD_PER_HEAD_K, \
                                          vec_size_q, VEC_SIZE_K, ROT_DIM_VAL><<<grid, block, 0, stream>>>( \
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(), \
            key.has_value() ? key->data_ptr<scalar_t>() : nullptr, \
            cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride, \
            head_stride, num_heads, num_kv_heads, head_size,num_tokens)

        if (q_k_size_multiple == 8) {
            switch (num_kv_heads) {
                case 1:
                case 2:
                {
                    if(is_neox){
                        LAUNCH_ROTARY_KERNEL_64(true,2, 32, 1, 64);
                    }else{
                        LAUNCH_ROTARY_KERNEL_64(false,2, 32, 1, 64);
                    }
                }  break;
                case 4: {
                    if(is_neox){
                        LAUNCH_ROTARY_KERNEL_64(true,4, 16, 2, 64);
                    }else{
                        LAUNCH_ROTARY_KERNEL_64(false,4, 16, 2, 64);
                    }
                } break;
                default: {
                     if(is_neox){
                         LAUNCH_ROTARY_KERNEL_64(true,8, 8, 4, 64);
                    }else{
                         LAUNCH_ROTARY_KERNEL_64(false,8, 8, 4, 64);
                    }
                } break;
            }
        } else if (q_k_size_multiple == 4) {
            switch (num_kv_heads) {
                  case 2:
                {
                    if(is_neox){
                        LAUNCH_ROTARY_KERNEL_64(true,2, 32, 1, 64);
                    }else{
                        LAUNCH_ROTARY_KERNEL_64(false,2, 32, 1, 64);
                    }
                }  break;
                case 4: {
                    if(is_neox){
                        LAUNCH_ROTARY_KERNEL_64(true,4, 16, 2, 64);
                    }else{
                        LAUNCH_ROTARY_KERNEL_64(false,4, 16, 2, 64);
                    }
                } break;
                default: {
                     if(is_neox){
                         LAUNCH_ROTARY_KERNEL_64(true,8, 8, 4, 64);
                    }else{
                         LAUNCH_ROTARY_KERNEL_64(false,8, 8, 4, 64);
                    }
                } break;
            }
        } else if (q_k_size_multiple == 2) {
            switch (num_kv_heads) {
                   case 4: {
                    if(is_neox){
                        LAUNCH_ROTARY_KERNEL_64(true,4, 16, 2, 64);
                    }else{
                        LAUNCH_ROTARY_KERNEL_64(false,4, 16, 2, 64);
                    }
                } break;
                default: {
                     if(is_neox){
                         LAUNCH_ROTARY_KERNEL_64(true,8, 8, 4, 64);
                    }else{
                         LAUNCH_ROTARY_KERNEL_64(false,8, 8, 4, 64);
                    }
                } break;
            }
        } else if (q_k_size_multiple == 1) {
              if(is_neox){
                         LAUNCH_ROTARY_KERNEL_64(true,8, 8, 4, 64);
                    }else{
                         LAUNCH_ROTARY_KERNEL_64(false,8, 8, 4, 64);
                    }
        } else if (q_k_size_multiple == 10) {
               if(is_neox){
                          LAUNCH_ROTARY_KERNEL_64(true,8, 8, 4, 64);
                    }else{
                          LAUNCH_ROTARY_KERNEL_64(false,8, 8, 4, 64);
                }
        }

    } else if (head_size == 128 && is_allign_qk) {


        constexpr int num_tokens_per_block = 2;
        constexpr int total_thread_per_tokens = 64;

        constexpr int num_group_per_token_q = 8;
        constexpr int thread_per_head_q = 8;
        constexpr int vec_size_q = 8;

        dim3 grid((num_tokens + num_tokens_per_block -1) / num_tokens_per_block);
        dim3 block(total_thread_per_tokens * num_tokens_per_block);


      #define LAUNCH_ROTARY_KERNEL_128(IS_NEOX,NUM_GROUP_K, THREAD_PER_HEAD_K, VEC_SIZE_K, ROT_DIM_VAL) \
        rotary_embedding_kernel_opt<scalar_t, IS_NEOX, num_tokens_per_block, total_thread_per_tokens, \
                                          num_group_per_token_q, NUM_GROUP_K, \
                                          thread_per_head_q, THREAD_PER_HEAD_K, \
                                          vec_size_q, VEC_SIZE_K, ROT_DIM_VAL><<<grid, block, 0, stream>>>( \
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(), \
            key.has_value() ? key->data_ptr<scalar_t>() : nullptr, \
            cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride, \
            head_stride, num_heads, num_kv_heads, head_size,num_tokens)

        if (q_k_size_multiple == 8) {
            switch (num_kv_heads) {
                case 1: {
                    if(is_neox){
                         LAUNCH_ROTARY_KERNEL_128(true,1, 64, 1, 128);
                    }else{
                         LAUNCH_ROTARY_KERNEL_128(false,1, 64, 1, 128);
                    }
                } break;
                case 2:{
                    if(is_neox){
                          LAUNCH_ROTARY_KERNEL_128(true,2, 32, 2, 128);
                    }else{
                          LAUNCH_ROTARY_KERNEL_128(false,2, 32, 2, 128);
                    }
                } break;
                case 4: {
                    if(is_neox){
                          LAUNCH_ROTARY_KERNEL_128(true,4, 16, 4, 128);
                    }else{
                          LAUNCH_ROTARY_KERNEL_128(false,4, 16, 4, 128);
                    }
                } break;
                default:{
                     if(is_neox){
                          LAUNCH_ROTARY_KERNEL_128(true,8, 8, 8, 128);
                    }else{
                          LAUNCH_ROTARY_KERNEL_128(false,8, 8, 8, 128);
                    }
                }  break; // Fallback
            }
        } else if (q_k_size_multiple == 4) {
            switch (num_kv_heads) {
               case 2:{
                    if(is_neox){
                          LAUNCH_ROTARY_KERNEL_128(true,2, 32, 2, 128);
                    }else{
                          LAUNCH_ROTARY_KERNEL_128(false,2, 32, 2, 128);
                    }
                } break;
                case 4: {
                    if(is_neox){
                          LAUNCH_ROTARY_KERNEL_128(true,4, 16, 4, 128);
                    }else{
                          LAUNCH_ROTARY_KERNEL_128(false,4, 16, 4, 128);
                    }
                } break;
                default:{
                     if(is_neox){
                          LAUNCH_ROTARY_KERNEL_128(true,1, 64, 1, 128);
                    }else{
                          LAUNCH_ROTARY_KERNEL_128(false,1, 64, 1, 128);
                    }
                }  break; // Fallback
            }
        } else if (q_k_size_multiple == 2) {
            switch (num_kv_heads) {
                 case 4: {
                    if(is_neox){
                          LAUNCH_ROTARY_KERNEL_128(true,4, 16, 4, 128);
                    }else{
                          LAUNCH_ROTARY_KERNEL_128(false,4, 16, 4, 128);
                    }
                } break;
                default:{
                     if(is_neox){
                          LAUNCH_ROTARY_KERNEL_128(true,8, 8, 8, 128);
                    }else{
                          LAUNCH_ROTARY_KERNEL_128(false,8, 8, 8, 128);
                    }
                }  break; // Fallback
            }
        } else if (q_k_size_multiple == 1) {
               if(is_neox){
                          LAUNCH_ROTARY_KERNEL_128(true,8, 8, 8, 128);
                    }else{
                          LAUNCH_ROTARY_KERNEL_128(false,8, 8, 8, 128);
                }
        } else if (q_k_size_multiple == 10) {
               if(is_neox){
                          LAUNCH_ROTARY_KERNEL_128(true,4, 16, 4, 128);
                    }else{
                          LAUNCH_ROTARY_KERNEL_128(false,4, 16, 4, 128);
                }
        }

    } else if (head_size == 256 && is_allign_qk) {
        constexpr int num_tokens_per_block = 2;
        constexpr int total_thread_per_tokens = 64;

        constexpr int num_group_per_token_q = 4;
        constexpr int thread_per_head_q = 16;
        constexpr int vec_size_q = 8;

        dim3 grid((num_tokens + num_tokens_per_block -1) / num_tokens_per_block);
        dim3 block(total_thread_per_tokens * num_tokens_per_block);

      #define LAUNCH_ROTARY_KERNEL_256(IS_NEOX,NUM_GROUP_K, THREAD_PER_HEAD_K, VEC_SIZE_K, ROT_DIM_VAL) \
        rotary_embedding_kernel_opt<scalar_t, IS_NEOX, num_tokens_per_block, total_thread_per_tokens, \
                                          num_group_per_token_q, NUM_GROUP_K, \
                                          thread_per_head_q, THREAD_PER_HEAD_K, \
                                          vec_size_q, VEC_SIZE_K, ROT_DIM_VAL><<<grid, block, 0, stream>>>( \
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(), \
            key.has_value() ? key->data_ptr<scalar_t>() : nullptr, \
            cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride, \
            head_stride, num_heads, num_kv_heads, head_size,num_tokens)

        if (q_k_size_multiple == 8) {
             switch (num_kv_heads) {
                case 1:{
                    if(is_neox){
                          LAUNCH_ROTARY_KERNEL_256(true,1, 64, 2, 256);
                    }else{
                          LAUNCH_ROTARY_KERNEL_256(false,1, 64, 2, 256);
                    }
                }  break;
                case 2:{
                     if(is_neox){
                          LAUNCH_ROTARY_KERNEL_256(true,2, 32, 4, 256);
                    }else{
                          LAUNCH_ROTARY_KERNEL_256(false,2, 32, 4, 256);
                    }
                } break;
                default: {
                  if(is_neox){
                          LAUNCH_ROTARY_KERNEL_256(true,4, 16, 8, 256);
                    }else{
                          LAUNCH_ROTARY_KERNEL_256(false,4, 16, 8, 256);
                    }
                }break; // Fallback
            }
        } else if (q_k_size_multiple == 4) {
            switch (num_kv_heads) {
               case 2:{
                     if(is_neox){
                          LAUNCH_ROTARY_KERNEL_256(true,2, 32, 4, 256);
                    }else{
                          LAUNCH_ROTARY_KERNEL_256(false,2, 32, 4, 256);
                    }
                } break;
                default: {
                  if(is_neox){
                          LAUNCH_ROTARY_KERNEL_256(true,4, 16, 8, 256);
                    }else{
                          LAUNCH_ROTARY_KERNEL_256(false,4, 16, 8, 256);
                    }
                }break; // Fallback
            }
        } else if (q_k_size_multiple == 2) {
             if(is_neox){
                          LAUNCH_ROTARY_KERNEL_256(true,2, 32, 4, 256);
                    }else{
                          LAUNCH_ROTARY_KERNEL_256(false,2, 32, 4, 256);
                    }
        } else if (q_k_size_multiple == 1) {
               if(is_neox){
                          LAUNCH_ROTARY_KERNEL_256(true,4, 16, 8, 256);
                    }else{
                          LAUNCH_ROTARY_KERNEL_256(false,4, 16, 8, 256);
                    }
        } else if (q_k_size_multiple == 10) {
               if(is_neox){
                          LAUNCH_ROTARY_KERNEL_128(true,8, 8, 16, 256);
                    }else{
                          LAUNCH_ROTARY_KERNEL_128(false,8, 8, 16, 256);
                }
        }
    } else {

        dim3 grid(num_tokens);
        dim3 block(std::min<int64_t>(num_heads * (rot_dim / 2), 512));
        if (is_neox) {
            rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
                positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
                key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
                cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride,
                head_stride, num_heads, num_kv_heads, head_size);
        } else {
            rotary_embedding_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
                positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
                key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
                cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride,
                head_stride, num_heads, num_kv_heads, head_size);
        }
    }
  }
);
}

//融合opt
void rotary_embedding_deepseek_fuse(
    torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& query,  // [batch_size, seq_len, num_heads * head_size] or
                           // [num_tokens, num_heads * head_size] or
                           // [batch_size, seq_len, num_heads, head_size] or
                           // [num_tokens, num_heads, head_size]
    std::optional<torch::Tensor> key,
    // null or
    // [batch_size, seq_len, num_kv_heads * head_size] or
    // [num_tokens, num_kv_heads * head_size] or
    // [batch_size, seq_len, num_heads, head_size] or
    // [num_tokens, num_heads, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox,
    std::optional<torch::Tensor> offsets) {

    int64_t num_tokens = positions.numel();
    int positions_ndim = positions.dim();
    TORCH_CHECK(
        positions_ndim == 1 || positions_ndim == 2,
        "positions must have shape [num_tokens] or [batch_size, seq_len]");
    if (positions_ndim == 1) {
        TORCH_CHECK(query.size(0) == positions.size(0) &&
                        (!key.has_value() || key->size(0) == positions.size(0)),
                    "query, key and positions must have the same number of tokens");
    }
    if (positions_ndim == 2) {
        TORCH_CHECK(
            query.size(0) == positions.size(0) &&
                (!key.has_value() || key->size(0) == positions.size(0)) &&
                query.size(1) == positions.size(1) &&
                (!key.has_value() || key->size(1) == positions.size(1)),
            "query, key and positions must have the same batch_size and seq_len");
    }

    int batch_size = query.size(0);
    int query_hidden_size = query.numel() / num_tokens;
    int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;

    TORCH_CHECK(query_hidden_size % head_size == 0);
    TORCH_CHECK(key_hidden_size % head_size == 0);

    int num_heads = query_hidden_size / head_size;
    int num_kv_heads = key.has_value() ? key_hidden_size / head_size : num_heads;
    TORCH_CHECK(num_heads % num_kv_heads == 0);


    int rot_dim = cos_sin_cache.size(1);
    if (head_size == 72) {
      TORCH_CHECK(rot_dim > 0 && rot_dim <= head_size && rot_dim % 2 == 0,
                  "rot_dim must be a positive even value no greater than "
                  "head_size, but got rot_dim=", rot_dim,
                  " and head_size=", head_size);
    }
    int seq_dim_idx = positions_ndim - 1;

    int64_t query_stride = query.stride(seq_dim_idx);
    int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;

    int query_ndim = query.dim();
    int64_t head_stride =
        (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;
    int64_t key_head_stride = key.has_value()
        ? ((key->dim() == positions_ndim + 2) ? key->stride(-2) : head_size)
        : head_stride;

    const bool has_dense_head72_tokens =
        positions_ndim == 1 ||
        (query.stride(0) == positions.size(1) * query_stride &&
         (!key.has_value() ||
          key->stride(0) == positions.size(1) * key_stride));
    const bool valid_head72_offsets =
        !offsets.has_value() ||
        (offsets->scalar_type() == at::ScalarType::Long &&
         offsets->sizes() == positions.sizes() && offsets->is_contiguous() &&
         offsets->device() == query.device());
    const bool has_head72_supported_layout =
        head_size == 72 && rot_dim == 72 &&
        (query.dim() == positions_ndim + 1 ||
         query.dim() == positions_ndim + 2) &&
        (!key.has_value() || key->dim() == query.dim()) &&
        positions.is_contiguous() && query.stride(-1) == 1 &&
        (!key.has_value() || key->stride(-1) == 1) &&
        cos_sin_cache.stride(1) == 1 &&
        cos_sin_cache.stride(0) == rot_dim && has_dense_head72_tokens &&
        valid_head72_offsets &&
        positions.scalar_type() == at::ScalarType::Long &&
        query.scalar_type() == cos_sin_cache.scalar_type() &&
        (!key.has_value() || query.scalar_type() == key->scalar_type()) &&
        positions.device() == query.device() &&
        cos_sin_cache.device() == query.device() &&
        (!key.has_value() || key->device() == query.device());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();


   AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(), "rotary_embedding", [&] {

    int rot_dim = cos_sin_cache.size(1);
    const int q_k_size_multiple = num_heads / num_kv_heads;

    bool is_allign_qk = (num_heads % num_kv_heads == 0) && (num_heads % 2 == 0) && (num_heads >= 8) && (q_k_size_multiple <= 128);

    constexpr uintptr_t head72_vec4_alignment = alignof(Vector<scalar_t, 4>);
    const bool can_use_head72_vec4 =
        has_head72_supported_layout &&
        reinterpret_cast<uintptr_t>(query.data_ptr<scalar_t>()) %
                head72_vec4_alignment == 0 &&
        reinterpret_cast<uintptr_t>(cos_sin_cache.data_ptr<scalar_t>()) %
                head72_vec4_alignment == 0 &&
        query_stride % 4 == 0 && head_stride % 4 == 0 &&
        (!key.has_value() ||
         (reinterpret_cast<uintptr_t>(key->data_ptr<scalar_t>()) %
                  head72_vec4_alignment == 0 &&
          key_stride % 4 == 0 && key_head_stride % 4 == 0));

    if (head_size == 72 && can_use_head72_vec4) {
        const int64_t* offsets_ptr =
            offsets.has_value() ? offsets->data_ptr<int64_t>() : nullptr;
        if (is_neox) {
          launch_rotary_embedding_head72<scalar_t, true>(
              positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
              key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
              cos_sin_cache.data_ptr<scalar_t>(), query_stride, key_stride,
              head_stride, key_head_stride, num_heads, num_kv_heads,
              num_tokens, offsets_ptr, stream);
        } else {
          launch_rotary_embedding_head72<scalar_t, false>(
              positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
              key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
              cos_sin_cache.data_ptr<scalar_t>(), query_stride, key_stride,
              head_stride, key_head_stride, num_heads, num_kv_heads,
              num_tokens, offsets_ptr, stream);
        }
    } else if (head_size == 64 && is_allign_qk && batch_size > 16) {
        constexpr int num_tokens_per_block = 2;
        constexpr int total_thread_per_tokens = 64;

        constexpr int num_group_per_token_q = 8;
        constexpr int thread_per_head_q = 8;
        constexpr int vec_size_q = 4;

        dim3 grid((num_tokens + num_tokens_per_block -1) / num_tokens_per_block);
        dim3 block(total_thread_per_tokens * num_tokens_per_block);

      #define LAUNCH_ROTARY_DEEPSEEK_KERNEL_64(IS_NEOX,NUM_GROUP_K, THREAD_PER_HEAD_K, VEC_SIZE_K, ROT_DIM_VAL) \
        rotary_embedding_deepseek_kernel_opt<scalar_t, IS_NEOX, num_tokens_per_block, total_thread_per_tokens, \
                                          num_group_per_token_q, NUM_GROUP_K, \
                                          thread_per_head_q, THREAD_PER_HEAD_K, \
                                          vec_size_q, VEC_SIZE_K, ROT_DIM_VAL><<<grid, block, 0, stream>>>( \
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(), \
            key.has_value() ? key->data_ptr<scalar_t>() : nullptr, \
            cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride, \
            head_stride, num_heads, num_kv_heads, head_size,num_tokens,\
            offsets.has_value() ? offsets->data_ptr<int64_t>() : nullptr)


        if(is_neox){
            LAUNCH_ROTARY_DEEPSEEK_KERNEL_64(true,2, 32, 2, 64);
        }else{
            LAUNCH_ROTARY_DEEPSEEK_KERNEL_64(false,2, 32, 2, 64);
        }

    } else {
        dim3 grid(num_tokens);
        dim3 block(std::min<int64_t>(num_heads * (rot_dim / 2), 512));
        if (is_neox) {
            rotary_embedding_kernel_deepseek<scalar_t, true><<<grid, block, 0, stream>>>(
                positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
                key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
                cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride,
                head_stride, num_heads, num_kv_heads, head_size,
                offsets.has_value() ? offsets->data_ptr<int64_t>() : nullptr);
        } else {
            rotary_embedding_kernel_deepseek<scalar_t, false><<<grid, block, 0, stream>>>(
                positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
                key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
                cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride,
                head_stride, num_heads, num_kv_heads, head_size,
                offsets.has_value() ? offsets->data_ptr<int64_t>() : nullptr);
        }
    }
  }
);
}

}
}
