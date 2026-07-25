#include <limits>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPMathCompat.h>
#include <c10/core/DeviceGuard.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <type_traits>

#define DISPATCH_BOOL(VAL, NAME, ...) \
  if (VAL) {                          \
    constexpr bool NAME = true;       \
    __VA_ARGS__();                    \
  } else {                            \
    constexpr bool NAME = false;      \
    __VA_ARGS__();                    \
  }

#define DISPATCH_CUSTOM_TYPE(TYPE, NAME, ...) \
    switch (TYPE) { \
        case at::ScalarType::Float: { \
            using NAME = float; \
            __VA_ARGS__(); \
            break; \
        } \
        case at::ScalarType::Half: { \
            using NAME = at::Half; \
            __VA_ARGS__(); \
            break; \
        } \
        case at::ScalarType::BFloat16: { \
            using NAME = at::BFloat16; \
            __VA_ARGS__(); \
            break; \
        } \
        default: \
            TORCH_CHECK(false, "不支持的融合参数类型"); \
    }

namespace at {
namespace native {

template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
    T val[N];
};

template<typename T, int WIDTH>
__device__ __forceinline__ T WarpReduceSum_Local(T val) {
    #pragma unroll
    for (int mask = WIDTH / 2; mask > 0; mask >>= 1) {
        val += __shfl_xor(val, mask);
    }
    return val;
}

template <typename T_ACC, typename scalar_t, typename cos_sin_t, typename norm_t, bool HAS_RESIDUAL, bool IS_NEOX, int VEC_SIZE, int THEAD_PER_HEAD>
__global__ void opt_layernorm_rope_split_kernel(
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,      
    scalar_t* __restrict__ key,        
    const cos_sin_t* __restrict__ cos_sin_cache,
    const int rot_dim,
    const int head_dim,                
    const int64_t query_stride,        
    const int64_t key_stride,          
    const int64_t head_stride_q,       
    const int64_t head_stride_k,      
    const norm_t* __restrict__ gamma_q, 
    const norm_t* __restrict__ beta_q,  
    const norm_t* __restrict__ gamma_k,
    const norm_t* __restrict__ beta_k,
    scalar_t* __restrict__ residual_q,
    scalar_t* __restrict__ residual_k,
    const scalar_t eps,
    const int num_tokens,
    const int num_heads,
    const int num_kv_heads,
    const int total_q_blocks,     
    const int blocks_q_per_token, 
    const int blocks_k_per_token, 
    const int HEADS_PER_BLOCK     
) {
    extern __shared__ char smem_buffer[];
    scalar_t* my_s_cos_sin = reinterpret_cast<scalar_t*>(smem_buffer);
    scalar_t* smem_heads = reinterpret_cast<scalar_t*>(smem_buffer + rot_dim * sizeof(scalar_t));

    const int block_id = blockIdx.x;
    
    const bool is_query = block_id < total_q_blocks;

    int token_idx;
    int head_start;

    if (is_query) {
        token_idx = block_id / blocks_q_per_token;
        int b_y = block_id % blocks_q_per_token;
        head_start = b_y * HEADS_PER_BLOCK;
    } else {
        int k_block_id = block_id - total_q_blocks;
        token_idx = k_block_id / blocks_k_per_token;
        int b_y = k_block_id % blocks_k_per_token;
        head_start = b_y * HEADS_PER_BLOCK;
    }

    const int local_head_idx = threadIdx.y; 
    const int lane_in_head = threadIdx.x;  
    const int head_idx = head_start + local_head_idx;
    
    const int num_heads_this_type = is_query ? num_heads : num_kv_heads;
    const bool is_valid_head = (token_idx < num_tokens) && (head_idx < num_heads_this_type);

    const int64_t pos = positions[token_idx];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.y * blockDim.x;

    for (int i = tid; i < rot_dim; i += total_threads) {
        my_s_cos_sin[i] = static_cast<scalar_t>(cos_sin_cache[pos * rot_dim + i]);
    }
    __syncthreads(); 

    if (!is_valid_head) return;

    scalar_t* head_ptr = is_query ? 
        (query + token_idx * query_stride + head_idx * head_stride_q) :
        (key + token_idx * key_stride + head_idx * head_stride_k);
        
    scalar_t* res_head_ptr = HAS_RESIDUAL ? (is_query ? 
        (residual_q + token_idx * query_stride + head_idx * head_stride_q) :
        (residual_k + token_idx * key_stride + head_idx * head_stride_k)) : nullptr;
        
    const norm_t* gamma_ptr = is_query ? gamma_q : gamma_k;
    const norm_t* beta_ptr  = is_query ? beta_q : beta_k;
    const bool do_layernorm = is_query ? (gamma_q != nullptr) : true;

    using LoadT = AlignedVector<scalar_t, VEC_SIZE>;
    scalar_t* my_smem_head = smem_heads + local_head_idx * head_dim;

    T_ACC local_sum = 0;
  
    for (int i = lane_in_head * VEC_SIZE; i < head_dim; i += THEAD_PER_HEAD * VEC_SIZE) {
        if (i < head_dim) {
            LoadT vals_vec = *(LoadT*)(head_ptr + i);

            if constexpr (HAS_RESIDUAL) {
                LoadT res_vec = *(LoadT*)(res_head_ptr + i);
                #pragma unroll
                for(int j=0; j<VEC_SIZE; ++j) {
                    vals_vec.val[j] += res_vec.val[j];
                    res_vec.val[j] = vals_vec.val[j];
                }
                *(LoadT*)(res_head_ptr + i) = res_vec;
            }

            #pragma unroll
            for(int j=0; j<VEC_SIZE; ++j) {
                my_smem_head[i + j] = vals_vec.val[j];
                local_sum += static_cast<T_ACC>(vals_vec.val[j]);
            }
        }
    }

    T_ACC head_sum = WarpReduceSum_Local<T_ACC, THEAD_PER_HEAD>(local_sum);
    T_ACC mean = head_sum / head_dim;

    T_ACC local_var = 0;
    for (int i = lane_in_head * VEC_SIZE; i < head_dim; i += THEAD_PER_HEAD * VEC_SIZE) {
        if (i < head_dim) {
            #pragma unroll
            for(int j=0; j<VEC_SIZE; ++j) {
                T_ACC diff = static_cast<T_ACC>(my_smem_head[i + j]) - mean;
                local_var += diff * diff;
            }
        }
    }
    
    T_ACC head_var = WarpReduceSum_Local<T_ACC, THEAD_PER_HEAD>(local_var);
    T_ACC inv_std = c10::hip::compat::rsqrt(head_var / head_dim + static_cast<T_ACC>(eps));

    for (int i = lane_in_head * VEC_SIZE; i < head_dim; i += THEAD_PER_HEAD * VEC_SIZE) {
        if (i < head_dim) {
            #pragma unroll
            for(int j=0; j<VEC_SIZE; ++j) {
                if (do_layernorm) {
                    T_ACC n = (static_cast<T_ACC>(my_smem_head[i + j]) - mean) * inv_std;
                    T_ACC g = gamma_ptr != nullptr ? static_cast<T_ACC>(gamma_ptr[i + j]) : static_cast<T_ACC>(1.0f);
                    T_ACC b = beta_ptr != nullptr ? static_cast<T_ACC>(beta_ptr[i + j]) : static_cast<T_ACC>(0.0f);
                    my_smem_head[i + j] = n * g + b;
                }
            }
        }
    }
    
    __syncthreads(); 

    const int HALF_ROT = rot_dim / 2;
    
    if constexpr (IS_NEOX) {
        for (int i = lane_in_head * VEC_SIZE; i < HALF_ROT; i += THEAD_PER_HEAD * VEC_SIZE) {
            if (i < HALF_ROT) {
                LoadT c_vec = *(LoadT*)(my_s_cos_sin + i);
                LoadT s_vec = *(LoadT*)(my_s_cos_sin + HALF_ROT + i);

                LoadT q_l_vec = *(LoadT*)(my_smem_head + i);
                LoadT q_h_vec = *(LoadT*)(my_smem_head + HALF_ROT + i);

                #pragma unroll
                for(int j = 0; j < VEC_SIZE; ++j) {
                    scalar_t cl = c_vec.val[j], sl = s_vec.val[j];
                    scalar_t ql = q_l_vec.val[j], qh = q_h_vec.val[j];
                    my_smem_head[i + j]            = ql * cl - qh * sl;
                    my_smem_head[HALF_ROT + i + j] = ql * sl + qh * cl;
                }
            }
        }
    } else {
        for (int i = lane_in_head * VEC_SIZE; i < rot_dim; i += THEAD_PER_HEAD * VEC_SIZE) {
            if (i < rot_dim) {
                using LoadCacheT = AlignedVector<scalar_t, VEC_SIZE / 2>;
                LoadCacheT c_vec = *(LoadCacheT*)(my_s_cos_sin + i / 2);
                LoadCacheT s_vec = *(LoadCacheT*)(my_s_cos_sin + HALF_ROT + i / 2);

                LoadT q_vec = *(LoadT*)(my_smem_head + i);

                #pragma unroll
                for(int j = 0; j < VEC_SIZE; j += 2) {
                    int c_idx = j / 2;
                    scalar_t q0 = q_vec.val[j], q1 = q_vec.val[j+1];
                    scalar_t cc = c_vec.val[c_idx], ss = s_vec.val[c_idx];
                    my_smem_head[i + j]   = q0 * cc - q1 * ss;
                    my_smem_head[i + j+1] = q1 * cc + q0 * ss;
                }
            }
        }
    }
    
    __syncthreads(); 
    
    for (int i = lane_in_head * VEC_SIZE; i < head_dim; i += THEAD_PER_HEAD * VEC_SIZE) {
        if (i < head_dim) {
            LoadT out_vec;
            #pragma unroll
            for(int j=0; j<VEC_SIZE; ++j) {
                out_vec.val[j] = my_smem_head[i + j];
            }
            *(LoadT*)(head_ptr + i) = out_vec;
        }
    }
}

template <typename T_ACC, typename scalar_t, typename cos_sin_t, typename norm_t>
void launch_opt_layernorm_rope_split(
    const int64_t* positions,
    scalar_t* query,
    scalar_t* key,
    const cos_sin_t* cos_sin_cache,
    int rot_dim,
    int head_dim,
    int64_t query_stride,
    int64_t key_stride,
    int64_t head_stride_q,
    int64_t head_stride_k,
    norm_t* gamma_q, 
    norm_t* beta_q,  
    norm_t* gamma_k,
    norm_t* beta_k,
    scalar_t* residual_q_ptr, 
    scalar_t* residual_k_ptr,
    scalar_t eps,
    int num_tokens,
    bool is_neox,
    const int num_heads,
    const int num_kv_heads,
    hipStream_t stream
) {
    bool has_residual = (residual_q_ptr != nullptr && residual_k_ptr != nullptr);

 
    constexpr int VEC = 8;
    constexpr int THREAD_PER_HEAD = 16;  
    int HEADS_PER_BLOCK = 4;   
    if(num_tokens >= 1024){
        HEADS_PER_BLOCK = 8;   
    }

    size_t smem_size = (rot_dim + HEADS_PER_BLOCK * head_dim) * sizeof(scalar_t);

    int blocks_q_per_token = (num_heads + HEADS_PER_BLOCK - 1) / HEADS_PER_BLOCK;
    int blocks_k_per_token = (num_kv_heads + HEADS_PER_BLOCK - 1) / HEADS_PER_BLOCK;
    
    int total_q_blocks = num_tokens * blocks_q_per_token;
    int total_k_blocks = num_tokens * blocks_k_per_token;
    int total_blocks = total_q_blocks + total_k_blocks;

    dim3 grid(total_blocks);
    dim3 block(THREAD_PER_HEAD, HEADS_PER_BLOCK); 
    
    DISPATCH_BOOL(has_residual, HAS_RESIDUAL_CONST, [&] {
        DISPATCH_BOOL(is_neox, IS_NEOX_CONST, [&] {
            opt_layernorm_rope_split_kernel<T_ACC, scalar_t, cos_sin_t, norm_t, HAS_RESIDUAL_CONST, IS_NEOX_CONST, VEC, THREAD_PER_HEAD>
                <<<grid, block, smem_size, stream>>>(
                positions, query, key, cos_sin_cache, 
                rot_dim, head_dim, 
                query_stride, key_stride, head_stride_q, head_stride_k, 
                gamma_q, beta_q, gamma_k, beta_k,
                residual_q_ptr, residual_k_ptr, 
                eps, num_tokens, num_heads, num_kv_heads,
                total_q_blocks, blocks_q_per_token, blocks_k_per_token, HEADS_PER_BLOCK
            );
        });
    });
}

void fuse_layernorm_rotary_embedding(
    at::Tensor& positions, 
    at::Tensor& query, 
    at::Tensor& key,
    int64_t head_size, 
    at::Tensor& cos_sin_cache, 
    bool is_neox, 
    std::optional<at::Tensor> weight_q, 
    std::optional<at::Tensor> bias_q,   
    std::optional<at::Tensor> weight_k, 
    std::optional<at::Tensor> bias_k,   
    std::optional<at::Tensor> residual_q,
    std::optional<at::Tensor> residual_k,
    double epsilon) {

    int64_t num_tokens = positions.numel();
    int positions_ndim = positions.dim();
    
    int query_hidden_size = query.numel() / num_tokens;
    int key_hidden_size = key.numel() / num_tokens;

    int num_heads = query_hidden_size / head_size;
    int num_kv_heads = key_hidden_size / head_size;
    
    int rot_dim = cos_sin_cache.size(1);
    int head_dim = head_size; 
  
    int seq_dim_idx = positions_ndim - 1;
    int64_t query_stride = query.stride(seq_dim_idx);
    int64_t key_stride = key.stride(seq_dim_idx);
    int query_ndim = query.dim();
    int key_ndim = key.dim();
    int64_t head_stride_q = (query_ndim >= 3) ? query.stride(-2) : head_size;
    int64_t head_stride_k = (key_ndim >= 3) ? key.stride(-2) : head_size;
      
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    at::ScalarType q_type = query.scalar_type();
    at::ScalarType cos_sin_type = cos_sin_cache.scalar_type();
    
    at::ScalarType norm_type = q_type;
    if (weight_k.has_value() && weight_k->defined()) {
        norm_type = weight_k->scalar_type();
    } else if (weight_q.has_value() && weight_q->defined()) {
        norm_type = weight_q->scalar_type();
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16, 
        q_type, "fuse_layernorm_rotary_embedding", [&] {
        
        using T_ACC = at::acc_type<scalar_t, true>;
        
        DISPATCH_CUSTOM_TYPE(cos_sin_type, cos_sin_t, [&] {
            DISPATCH_CUSTOM_TYPE(norm_type, norm_t, [&] {
                
                norm_t* w_q_ptr = weight_q.has_value() && weight_q->defined() ? weight_q->data_ptr<norm_t>() : nullptr;
                norm_t* b_q_ptr = bias_q.has_value() && bias_q->defined() ? bias_q->data_ptr<norm_t>() : nullptr;
                norm_t* w_k_ptr = weight_k.has_value() && weight_k->defined() ? weight_k->data_ptr<norm_t>() : nullptr;
                norm_t* b_k_ptr = bias_k.has_value() && bias_k->defined() ? bias_k->data_ptr<norm_t>() : nullptr;

                scalar_t* res_q_ptr = residual_q.has_value() && residual_q->defined() ? residual_q->data_ptr<scalar_t>() : nullptr;
                scalar_t* res_k_ptr = residual_k.has_value() && residual_k->defined() ? residual_k->data_ptr<scalar_t>() : nullptr;
                
                launch_opt_layernorm_rope_split<T_ACC, scalar_t, cos_sin_t, norm_t>(
                    positions.data_ptr<int64_t>(),
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    cos_sin_cache.data_ptr<cos_sin_t>(),
                    rot_dim, head_dim,
                    query_stride, key_stride, head_stride_q, head_stride_k,
                    w_q_ptr, b_q_ptr, 
                    w_k_ptr, b_k_ptr,
                    res_q_ptr, res_k_ptr,
                    static_cast<scalar_t>(epsilon),
                    (int)num_tokens,
                    is_neox,
                    num_heads, num_kv_heads,
                    stream
                );
            });
        });
    });
}

} // namespace native
} // namespace at
