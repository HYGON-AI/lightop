#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#ifndef USE_ROCM
#define USE_ROCM 0
#endif

#if USE_ROCM
#include <hip/hip_fp8.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
typedef __hip_bfloat16 __nv_bfloat16;
#define SHFL_XOR(val, mask) __shfl_xor(val, mask)
#else
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define SHFL_XOR(val, mask) __shfl_xor_sync(0xffffffff, val, mask)
#endif

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#ifndef HIP_FP8_TYPE_OCP
#define HIP_FP8_TYPE_OCP 0
#endif

#if USE_ROCM
    #if HIP_FP8_TYPE_OCP
    using fp8_e4m3_type = __hip_fp8_e4m3;
    using fp8_e5m2_type = __hip_fp8_e5m2;
    #else
    using fp8_e4m3_type = __hip_fp8_e4m3_fnuz;
    using fp8_e5m2_type = __hip_fp8_e5m2_fnuz;
    #endif
#endif

namespace at {
namespace native {
inline __device__ uint8_t float_to_fp8e5m2(float f) {
    constexpr uint32_t fp32_inf = UINT32_C(255) << 23;
    constexpr uint32_t fp8_max = UINT32_C(143) << 23;
    constexpr uint32_t denorm_mask = UINT32_C(134) << 23;
    uint32_t f_bits = c10::detail::fp32_to_bits(f);
    uint8_t result = 0u;
    const uint32_t sign = f_bits & UINT32_C(0x80000000);
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
        result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
    } else {
        if (f_bits < (UINT32_C(113) << 23)) {
            f_bits = c10::detail::fp32_to_bits(c10::detail::fp32_from_bits(f_bits)
                     + c10::detail::fp32_from_bits(denorm_mask));
            result = static_cast<uint8_t>(f_bits - denorm_mask);
        } else {
            uint32_t mant_odd = (f_bits >> 21) & 1;
            f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;
            f_bits += mant_odd;
            result = static_cast<uint8_t>(f_bits >> 21);
        }
    }
    result |= static_cast<uint8_t>(sign >> 24);
    return result;
}

inline __device__ uint8_t float_to_fp8e4m3(float f) {
    constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
    constexpr uint32_t denorm_mask = UINT32_C(141) << 23;
    uint32_t f_bits = c10::detail::fp32_to_bits(f);
    uint8_t result = 0u;
    const uint32_t sign = f_bits & UINT32_C(0x80000000);
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
        result = 0x7f;  
    } else {
        if (f_bits < (UINT32_C(121) << 23)) {
            f_bits = c10::detail::fp32_to_bits(
                c10::detail::fp32_from_bits(f_bits) + c10::detail::fp32_from_bits(denorm_mask)
            );
            result = static_cast<uint8_t>(f_bits - denorm_mask);
        } else {
            uint8_t mant_odd = (f_bits >> 20) & 1;
            f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
            f_bits += mant_odd;
            result = static_cast<uint8_t>(f_bits >> 20);
        }
    }
    result |= static_cast<uint8_t>(sign >> 24);
    return result;
}

inline __device__ uint32_t float_to_fp8e4m3_x4(float a, float b, float c, float d) {
#if defined(__gfx938__)
    int32_t res;
    res = __builtin_hcu_cvt_pk_fp8_f32(a, b, res, false);
    res = __builtin_hcu_cvt_pk_fp8_f32(c, d, res, true);
    return res;
#else
    int32_t out;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e4m3(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e4m3(b);
    reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e4m3(c);
    reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e4m3(d);
    return out;
#endif
}

inline __device__ uint32_t float_to_fp8e5m2_x4(float a, float b, float c, float d) {
#if defined(__gfx938__)
    int32_t res;
    res = __builtin_hcu_cvt_pk_bf8_f32(a, b, res, false);
    res = __builtin_hcu_cvt_pk_bf8_f32(c, d, res, true);
    return res;
#else
    int32_t out;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e5m2(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e5m2(b);
    reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e5m2(c);
    reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e5m2(d);
    return out;
#endif
}

template <typename scalar_t, bool is_e4m3>
__global__ void fused_quantize_and_store_mla_kv_kernel(
    const scalar_t* __restrict__ k_nope,    
    const scalar_t* __restrict__ k_rope,     
    uint8_t* __restrict__ kv_buffer,         
    const int64_t* __restrict__ loc,         
    const int64_t buffer_stride_bytes,      
    const float eps,
    const int num_tokens,
    const int nope_blocks                   
) {
    constexpr int NOPE_THREADS_PER_TOKEN = 32; 
    constexpr int ROPE_THREADS_PER_TOKEN = 8;  
    constexpr int VEC_NOPE = 16;
    constexpr int VEC_ROPE = 8;
    using RopeVecType = at::native::memory::aligned_vector<scalar_t, VEC_ROPE>;
    using NopeVecType = at::native::memory::aligned_vector<scalar_t, VEC_NOPE>;
    using Fp8VecType = at::native::memory::aligned_vector<uint8_t, VEC_NOPE>;

    if (blockIdx.x < nope_blocks) {
        int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
        int token_idx = global_tid / NOPE_THREADS_PER_TOKEN;
        int lane_id = global_tid % NOPE_THREADS_PER_TOKEN;

        if (token_idx >= num_tokens) return;

        int group_id = lane_id / 8;        
        int lane_in_group = lane_id % 8;  
        int nope_idx = group_id * 128 + lane_in_group * VEC_NOPE;

        int64_t token_loc = loc[token_idx];
        uint8_t* dst_ptr = kv_buffer + token_loc * buffer_stride_bytes;
        scalar_t input_vec[VEC_NOPE];
        *(NopeVecType*)input_vec = *(NopeVecType*)(k_nope + token_idx * 512 + nope_idx);

        float input_float_vec[VEC_NOPE];
        float max_val = 0.f;
        #pragma unroll
        for (int i = 0; i < VEC_NOPE; i++) {
            input_float_vec[i] = static_cast<float>(input_vec[i]);
            max_val = fmaxf(max_val, fabsf(input_float_vec[i]));
        }

        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            max_val = fmaxf(max_val, SHFL_XOR(max_val, mask));
        }
        constexpr float FP8_MAX_VAL = is_e4m3 ? 448.0f : 57344.0f;
        max_val = fmaxf(max_val, eps);
        float scale = max_val / FP8_MAX_VAL;

        if (lane_in_group == 0) {
            float* scale_dst = reinterpret_cast<float*>(dst_ptr + 512);
            scale_dst[group_id] = scale;
        }
        float inv_s = (scale == 0.f) ? 0.f : 1.0f / scale;
        uint8_t out_vec[VEC_NOPE];
        
        #pragma unroll
        for(int i = 0; i < VEC_NOPE; i += 4) {
            uint32_t packed;
            if constexpr (is_e4m3) {
                packed = float_to_fp8e4m3_x4(
                    input_float_vec[i] * inv_s, input_float_vec[i+1] * inv_s,
                    input_float_vec[i+2] * inv_s, input_float_vec[i+3] * inv_s);
            } else {
                packed = float_to_fp8e5m2_x4(
                    input_float_vec[i] * inv_s, input_float_vec[i+1] * inv_s,
                    input_float_vec[i+2] * inv_s, input_float_vec[i+3] * inv_s);
            }
            *(uint32_t*)(&out_vec[i]) = packed;
        }
        *(Fp8VecType*)(dst_ptr + nope_idx) = *(Fp8VecType*)out_vec;

    } else {
        int k_block_idx = blockIdx.x - nope_blocks;
        int global_tid = k_block_idx * blockDim.x + threadIdx.x;
        
        int token_idx = global_tid / ROPE_THREADS_PER_TOKEN;
        int lane_id = global_tid % ROPE_THREADS_PER_TOKEN;

        if (token_idx >= num_tokens) return;

        constexpr int ROPE_BYTE_OFFSET = 512 + 16; 
        int rope_idx = lane_id * VEC_ROPE;
        
        int64_t token_loc = loc[token_idx];
        uint8_t* dst_ptr = kv_buffer + token_loc * buffer_stride_bytes;
        RopeVecType rope_val = *(RopeVecType*)(k_rope + token_idx * 64 + rope_idx);
        *(RopeVecType*)(dst_ptr + ROPE_BYTE_OFFSET + rope_idx * sizeof(scalar_t)) = rope_val;
    }
}

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

void fused_quantize_and_store_mla_kv_cache(
    torch::Tensor k_nope,     // [num_tokens, 512]
    torch::Tensor k_rope,     // [num_tokens, 64]
    torch::Tensor kv_buffer,  // [..., buffer_stride_bytes]
    torch::Tensor loc,        // [num_tokens]
    const std::string& fp8_dtype = "fp8_e4m3",
    float eps = 1e-5f
) {
    if (k_nope.dim() == 3) k_nope = k_nope.squeeze(1);
    if (k_rope.dim() == 3) k_rope = k_rope.squeeze(1);

    k_nope = k_nope.contiguous();
    k_rope = k_rope.contiguous();
    loc = loc.contiguous();
    TORCH_CHECK(k_nope.size(-1) == 512, "DeepSeek MLA dim_nope must be 512");
    TORCH_CHECK(k_rope.size(-1) == 64, "DeepSeek MLA dim_rope must be 64");
    TORCH_CHECK(loc.scalar_type() == torch::kInt64, "loc tensor must be Int64");
    TORCH_CHECK(kv_buffer.scalar_type() == torch::kUInt8, "kv_buffer must be UInt8");
    TORCH_CHECK(k_nope.scalar_type() == k_rope.scalar_type(), "Precision must match");
    int num_tokens = loc.size(0);
    int64_t buffer_stride_bytes = kv_buffer.stride(0); 
    bool is_e4m3 = (fp8_dtype == "fp8_e4m3" || fp8_dtype == "fp8_e4m3fn");
    int threads_per_block = 256; 
    int nope_threads = num_tokens * 32;
    int nope_blocks = CEIL_DIV(nope_threads, threads_per_block);
    int rope_threads = num_tokens * 8;
    int rope_blocks = CEIL_DIV(rope_threads, threads_per_block);

    int total_blocks = nope_blocks + rope_blocks;

    if (total_blocks == 0) return;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    VLLM_DISPATCH_FLOATING_TYPES(
        k_nope.scalar_type(), "fused_quantize_and_store_mla_kv_kernel", [&] {
            if (is_e4m3) {
                fused_quantize_and_store_mla_kv_kernel<scalar_t, true>
                    <<<total_blocks, threads_per_block, 0, stream>>>(
                        k_nope.data_ptr<scalar_t>(),
                        k_rope.data_ptr<scalar_t>(),
                        kv_buffer.data_ptr<uint8_t>(),
                        loc.data_ptr<int64_t>(),
                        buffer_stride_bytes,
                        eps,
                        num_tokens,
                        nope_blocks
                    );
            } else {
                fused_quantize_and_store_mla_kv_kernel<scalar_t, false>
                    <<<total_blocks, threads_per_block, 0, stream>>>(
                        k_nope.data_ptr<scalar_t>(),
                        k_rope.data_ptr<scalar_t>(),
                        kv_buffer.data_ptr<uint8_t>(),
                        loc.data_ptr<int64_t>(),
                        buffer_stride_bytes,
                        eps,
                        num_tokens,
                        nope_blocks
                    );
            }
        }
    );
}

template <typename scalar_t>
__global__ void quantize_nope_fp8_rope_bf16_pack_kernel(
    const scalar_t* __restrict__ k_bf16,
    uint8_t* __restrict__ k_nope_fp8,
    scalar_t* __restrict__ k_rope_bf16,
    uint8_t* __restrict__ scale_k_nope,
    const int64_t k_bf16_stride_0,
    const int64_t k_nope_fp8_stride_0,
    const int64_t k_rope_bf16_stride_0,
    const int64_t scale_stride_0,
    const float eps,
    const int num_tokens,
    const int nope_blocks
) {
    constexpr int VEC = 16;
    constexpr int GROUP_SIZE = 64;
    constexpr int NUM_GROUPS = 7;
    constexpr int GROUP_THREADS = GROUP_SIZE / VEC;
    constexpr int NOPE_THREADS = NUM_GROUPS * GROUP_THREADS;
    constexpr int ROPE_THREADS = 8;
    constexpr int ROPE_VEC = 8;
    constexpr int DIM_NOPE = 448;
    constexpr float FP8_MAX_VAL = 448.0f;

    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using Fp8VecType = at::native::memory::aligned_vector<uint8_t, VEC>;
    using RopeVecType = at::native::memory::aligned_vector<scalar_t, ROPE_VEC>;

    if (blockIdx.x < nope_blocks) {
        int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
        int token_idx = global_tid / NOPE_THREADS;
        int lane_id = global_tid % NOPE_THREADS;

        if (token_idx >= num_tokens) return;

        int group_id = lane_id / GROUP_THREADS;
        int lane_in_group = lane_id % GROUP_THREADS;
        int elem_offset = group_id * GROUP_SIZE + lane_in_group * VEC;

        scalar_t input_vec[VEC];
        *(VecType*)input_vec = *(VecType*)(k_bf16 + token_idx * k_bf16_stride_0 + elem_offset);

        float input_float_vec[VEC];
        float max_val = 0.f;
        #pragma unroll
        for (int i = 0; i < VEC; i++) {
            input_float_vec[i] = static_cast<float>(input_vec[i]);
            max_val = fmaxf(max_val, fabsf(input_float_vec[i]));
        }

        #pragma unroll
        for (int mask = 2; mask > 0; mask >>= 1) {
            max_val = fmaxf(max_val, SHFL_XOR(max_val, mask));
        }

        max_val = fmaxf(max_val, eps);
        float scale = max_val / FP8_MAX_VAL;

        int exp;
        float mant = frexpf(scale, &exp);
        int exp_ceil = (mant > 0.5f) ? exp : exp - 1;
        float inv_scale = scalbnf(1.0f, -exp_ceil);

        if (lane_in_group == 0) {
            scale_k_nope[token_idx * scale_stride_0 + group_id] = (uint8_t)(exp_ceil + 127);
        }

        uint8_t out_vec[VEC];
        #pragma unroll
        for (int i = 0; i < VEC; i += 4) {
            uint32_t packed = float_to_fp8e4m3_x4(
                input_float_vec[i] * inv_scale, input_float_vec[i+1] * inv_scale,
                input_float_vec[i+2] * inv_scale, input_float_vec[i+3] * inv_scale);
            *(uint32_t*)(&out_vec[i]) = packed;
        }

        int64_t out_offset = token_idx * k_nope_fp8_stride_0 + elem_offset;
        *(Fp8VecType*)(k_nope_fp8 + out_offset) = *(Fp8VecType*)out_vec;

    } else {
        int rope_block_idx = blockIdx.x - nope_blocks;
        int global_tid = rope_block_idx * blockDim.x + threadIdx.x;
        int token_idx = global_tid / ROPE_THREADS;
        int lane_id = global_tid % ROPE_THREADS;

        if (token_idx >= num_tokens) return;

        int rope_offset = DIM_NOPE + lane_id * ROPE_VEC;

        RopeVecType rope_val = *(RopeVecType*)(k_bf16 + token_idx * k_bf16_stride_0 + rope_offset);
        *(RopeVecType*)(k_rope_bf16 + token_idx * k_rope_bf16_stride_0 + lane_id * ROPE_VEC) = rope_val;
    }
}

template <typename scalar_t, typename index_t>
__global__ void quantize_nope_fp8_rope_bf16_pack_store_kernel(
    const scalar_t* __restrict__ k_bf16,
    uint8_t* __restrict__ buf,
    const index_t* __restrict__ loc,
    const int64_t k_bf16_stride_0,
    const int64_t buf_stride_0,
    const int page_size,
    const float eps,
    const int num_tokens,
    const int nope_blocks
) {
    constexpr int VEC = 16;
    constexpr int GROUP_SIZE = 64;
    constexpr int NUM_GROUPS = 7;
    constexpr int GROUP_THREADS = GROUP_SIZE / VEC;
    constexpr int NOPE_THREADS = NUM_GROUPS * GROUP_THREADS;
    constexpr int ROPE_THREADS = 8;
    constexpr int ROPE_VEC = 8;
    constexpr int DIM_NOPE = 448;
    constexpr int DIM_ROPE = 64;
    constexpr int NUM_NOPE_ROPE_BYTES_PER_TOKEN = DIM_NOPE + DIM_ROPE * 2;
    constexpr int PADDED_SCALE_ELEMS_PER_TOKEN = NUM_GROUPS + 1;
    constexpr float FP8_MAX_VAL = 448.0f;

    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using Fp8VecType = at::native::memory::aligned_vector<uint8_t, VEC>;
    using RopeVecType = at::native::memory::aligned_vector<scalar_t, ROPE_VEC>;

    if (blockIdx.x < nope_blocks) {
        int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
        int token_idx = global_tid / NOPE_THREADS;
        int lane_id = global_tid % NOPE_THREADS;

        if (token_idx >= num_tokens) return;

        int group_id = lane_id / GROUP_THREADS;
        int lane_in_group = lane_id % GROUP_THREADS;
        int elem_offset = group_id * GROUP_SIZE + lane_in_group * VEC;

        int64_t cache_loc = static_cast<int64_t>(loc[token_idx]);
        int64_t page_idx = cache_loc / page_size;
        int64_t token_offset = cache_loc % page_size;
        uint8_t* page_base = buf + page_idx * buf_stride_0;
        uint8_t* token_base =
            page_base + token_offset * NUM_NOPE_ROPE_BYTES_PER_TOKEN;

        scalar_t input_vec[VEC];
        *(VecType*)input_vec = *(VecType*)(k_bf16 + token_idx * k_bf16_stride_0 + elem_offset);

        float input_float_vec[VEC];
        float max_val = 0.f;
        #pragma unroll
        for (int i = 0; i < VEC; i++) {
            input_float_vec[i] = static_cast<float>(input_vec[i]);
            max_val = fmaxf(max_val, fabsf(input_float_vec[i]));
        }

        #pragma unroll
        for (int mask = 2; mask > 0; mask >>= 1) {
            max_val = fmaxf(max_val, SHFL_XOR(max_val, mask));
        }

        max_val = fmaxf(max_val, eps);
        float scale = max_val / FP8_MAX_VAL;

        int exp;
        float mant = frexpf(scale, &exp);
        int exp_ceil = (mant > 0.5f) ? exp : exp - 1;
        float inv_scale = scalbnf(1.0f, -exp_ceil);

        uint8_t* scale_base =
            page_base + page_size * NUM_NOPE_ROPE_BYTES_PER_TOKEN +
            token_offset * PADDED_SCALE_ELEMS_PER_TOKEN;
        if (lane_in_group == 0) {
            scale_base[group_id] = (uint8_t)(exp_ceil + 127);
            if (group_id == 0) {
                scale_base[NUM_GROUPS] = 0;
            }
        }

        uint8_t out_vec[VEC];
        #pragma unroll
        for (int i = 0; i < VEC; i += 4) {
            uint32_t packed = float_to_fp8e4m3_x4(
                input_float_vec[i] * inv_scale, input_float_vec[i+1] * inv_scale,
                input_float_vec[i+2] * inv_scale, input_float_vec[i+3] * inv_scale);
            *(uint32_t*)(&out_vec[i]) = packed;
        }
        *(Fp8VecType*)(token_base + elem_offset) = *(Fp8VecType*)out_vec;
    } else {
        int rope_block_idx = blockIdx.x - nope_blocks;
        int global_tid = rope_block_idx * blockDim.x + threadIdx.x;
        int token_idx = global_tid / ROPE_THREADS;
        int lane_id = global_tid % ROPE_THREADS;

        if (token_idx >= num_tokens) return;

        int64_t cache_loc = static_cast<int64_t>(loc[token_idx]);
        int64_t page_idx = cache_loc / page_size;
        int64_t token_offset = cache_loc % page_size;
        uint8_t* token_base =
            buf + page_idx * buf_stride_0 +
            token_offset * NUM_NOPE_ROPE_BYTES_PER_TOKEN;

        int rope_offset = DIM_NOPE + lane_id * ROPE_VEC;
        RopeVecType rope_val = *(RopeVecType*)(k_bf16 + token_idx * k_bf16_stride_0 + rope_offset);
        *(RopeVecType*)(token_base + DIM_NOPE + lane_id * ROPE_VEC * sizeof(scalar_t)) = rope_val;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize_nope_fp8_rope_bf16_pack(
    torch::Tensor k_bf16,
    float eps
) {
    constexpr int DIM_NOPE = 448;
    constexpr int DIM_ROPE = 64;
    constexpr int TOTAL_DIM = 512;
    constexpr int GROUP_SIZE = 64;
    constexpr int NUM_GROUPS = DIM_NOPE / GROUP_SIZE;
    constexpr int VEC = 16;
    constexpr int NOPE_THREADS_PER_TOKEN = NUM_GROUPS * (GROUP_SIZE / VEC);
    constexpr int ROPE_THREADS_PER_TOKEN = 8;
    constexpr int THREADS_PER_BLOCK = 128;

    TORCH_CHECK(k_bf16.dim() == 2, "k_bf16 must be 2D, got ", k_bf16.dim(), "D");
    TORCH_CHECK(k_bf16.size(1) >= TOTAL_DIM,
        "k_bf16 last dim must be >= 512, got ", k_bf16.size(1));
    TORCH_CHECK(k_bf16.scalar_type() == torch::kBFloat16,
        "k_bf16 must be bfloat16, got ", k_bf16.scalar_type());

    k_bf16 = k_bf16.contiguous();
    int num_tokens = (int)k_bf16.size(0);

    auto k_nope_fp8 = torch::empty({num_tokens, DIM_NOPE},
        torch::TensorOptions().dtype(torch::kUInt8).device(k_bf16.device()));
    auto k_rope_bf16 = torch::empty({num_tokens, DIM_ROPE},
        torch::TensorOptions().dtype(torch::kBFloat16).device(k_bf16.device()));
    auto scale_k_nope = torch::empty({num_tokens, NUM_GROUPS},
        torch::TensorOptions().dtype(torch::kUInt8).device(k_bf16.device()));

    int nope_threads = num_tokens * NOPE_THREADS_PER_TOKEN;
    int nope_blocks = CEIL_DIV(nope_threads, THREADS_PER_BLOCK);
    int rope_threads = num_tokens * ROPE_THREADS_PER_TOKEN;
    int rope_blocks = CEIL_DIV(rope_threads, THREADS_PER_BLOCK);
    int total_blocks = nope_blocks + rope_blocks;

    if (total_blocks > 0) {
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        VLLM_DISPATCH_FLOATING_TYPES(
            k_bf16.scalar_type(), "quantize_nope_fp8_rope_bf16_pack_kernel", [&] {
                quantize_nope_fp8_rope_bf16_pack_kernel<scalar_t>
                    <<<total_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                        k_bf16.data_ptr<scalar_t>(),
                        k_nope_fp8.data_ptr<uint8_t>(),
                        k_rope_bf16.data_ptr<scalar_t>(),
                        scale_k_nope.data_ptr<uint8_t>(),
                        k_bf16.stride(0),
                        k_nope_fp8.stride(0),
                        k_rope_bf16.stride(0),
                        scale_k_nope.stride(0),
                        eps,
                        num_tokens,
                        nope_blocks
                    );
            }
        );
    }

    return std::make_tuple(k_nope_fp8, k_rope_bf16, scale_k_nope);
}

template <typename index_t>
void launch_quantize_nope_fp8_rope_bf16_pack_store(
    torch::Tensor k_bf16,
    torch::Tensor buf,
    torch::Tensor loc,
    int page_size,
    float eps
) {
    constexpr int DIM_NOPE = 448;
    constexpr int DIM_ROPE = 64;
    constexpr int TOTAL_DIM = 512;
    constexpr int GROUP_SIZE = 64;
    constexpr int NUM_GROUPS = DIM_NOPE / GROUP_SIZE;
    constexpr int VEC = 16;
    constexpr int NOPE_THREADS_PER_TOKEN = NUM_GROUPS * (GROUP_SIZE / VEC);
    constexpr int ROPE_THREADS_PER_TOKEN = 8;
    constexpr int THREADS_PER_BLOCK = 128;

    int num_tokens = (int)loc.size(0);
    int nope_threads = num_tokens * NOPE_THREADS_PER_TOKEN;
    int nope_blocks = CEIL_DIV(nope_threads, THREADS_PER_BLOCK);
    int rope_threads = num_tokens * ROPE_THREADS_PER_TOKEN;
    int rope_blocks = CEIL_DIV(rope_threads, THREADS_PER_BLOCK);
    int total_blocks = nope_blocks + rope_blocks;
    if (total_blocks == 0) return;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    VLLM_DISPATCH_FLOATING_TYPES(
        k_bf16.scalar_type(), "quantize_nope_fp8_rope_bf16_pack_store_kernel", [&] {
            quantize_nope_fp8_rope_bf16_pack_store_kernel<scalar_t, index_t>
                <<<total_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    k_bf16.data_ptr<scalar_t>(),
                    buf.data_ptr<uint8_t>(),
                    loc.data_ptr<index_t>(),
                    k_bf16.stride(0),
                    buf.stride(0),
                    page_size,
                    eps,
                    num_tokens,
                    nope_blocks
                );
        }
    );
}

// MODE: 0=plain cast (bf16/fp16/fp32 -> bf16/fp16/fp32)
//       1=fp8_e4m3 saturating cast (dst is uint8)
//       2=fp8_e5m2 saturating cast (dst is uint8)
template <typename src_t, typename dst_t, int MODE>
__global__ void fused_concat_and_store_mla_kv_kernel(
    const src_t* __restrict__ k_nope,
    const src_t* __restrict__ k_rope,
    dst_t* __restrict__ kv_buffer,
    const int64_t* __restrict__ loc,
    const int64_t buffer_stride,
    const int64_t nope_stride,
    const int64_t rope_stride,
    const int nope_dim,
    const int rope_dim,
    const int num_tokens,
    const int nope_blocks,
    const int nope_threads_per_token,
    const int rope_threads_per_token
) {
    constexpr int VEC = 8;
    using SrcVec = at::native::memory::aligned_vector<src_t, VEC>;
    using DstVec = at::native::memory::aligned_vector<dst_t, VEC>;

    const src_t* src_base;
    int lane_id;
    int64_t token_loc;
    int64_t dst_col_offset;
    int64_t src_row_stride;

    if (blockIdx.x < nope_blocks) {
        int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
        int token_idx = global_tid / nope_threads_per_token;
        lane_id = global_tid % nope_threads_per_token;
        if (token_idx >= num_tokens) return;

        token_loc = loc[token_idx];
        src_base = k_nope + token_idx * nope_stride;
        src_row_stride = nope_stride;
        dst_col_offset = 0;
    } else {
        int rope_block_idx = blockIdx.x - nope_blocks;
        int global_tid = rope_block_idx * blockDim.x + threadIdx.x;
        int token_idx = global_tid / rope_threads_per_token;
        lane_id = global_tid % rope_threads_per_token;
        if (token_idx >= num_tokens) return;

        token_loc = loc[token_idx];
        src_base = k_rope + token_idx * rope_stride;
        src_row_stride = rope_stride;
        dst_col_offset = nope_dim;
    }
    (void)src_row_stride;

    int elem_offset = lane_id * VEC;

    src_t in_vec[VEC];
    *(SrcVec*)in_vec = *(SrcVec*)(src_base + elem_offset);

    dst_t out_vec[VEC];
    if constexpr (MODE == 0) {
        #pragma unroll
        for (int i = 0; i < VEC; i++) {
            out_vec[i] = static_cast<dst_t>(static_cast<float>(in_vec[i]));
        }
    } else {
        float in_f[VEC];
        #pragma unroll
        for (int i = 0; i < VEC; i++) in_f[i] = static_cast<float>(in_vec[i]);

        #pragma unroll
        for (int i = 0; i < VEC; i += 4) {
            uint32_t packed;
            if constexpr (MODE == 1) {
                packed = float_to_fp8e4m3_x4(
                    in_f[i], in_f[i + 1], in_f[i + 2], in_f[i + 3]);
            } else {
                packed = float_to_fp8e5m2_x4(
                    in_f[i], in_f[i + 1], in_f[i + 2], in_f[i + 3]);
            }
            *(uint32_t*)(reinterpret_cast<uint8_t*>(out_vec) + i) = packed;
        }
    }

    *(DstVec*)(kv_buffer + token_loc * buffer_stride + dst_col_offset + elem_offset) =
        *(DstVec*)out_vec;
}

template <typename src_t, typename dst_t, int MODE>
static void launch_fused_concat_and_store_mla_kv_cache(
    const torch::Tensor& k_nope,
    const torch::Tensor& k_rope,
    torch::Tensor& kv_buffer,
    const torch::Tensor& loc
) {
    constexpr int VEC = 8;
    constexpr int THREADS_PER_BLOCK = 128;

    int num_tokens = (int)loc.size(0);
    int nope_dim = (int)k_nope.size(-1);
    int rope_dim = (int)k_rope.size(-1);

    TORCH_CHECK(nope_dim % VEC == 0,
        "nope_dim must be divisible by ", VEC, ", got ", nope_dim);
    TORCH_CHECK(rope_dim % VEC == 0,
        "rope_dim must be divisible by ", VEC, ", got ", rope_dim);

    int nope_threads_per_token = nope_dim / VEC;
    int rope_threads_per_token = rope_dim / VEC;
    int nope_threads = num_tokens * nope_threads_per_token;
    int rope_threads = num_tokens * rope_threads_per_token;
    int nope_blocks = CEIL_DIV(nope_threads, THREADS_PER_BLOCK);
    int rope_blocks = CEIL_DIV(rope_threads, THREADS_PER_BLOCK);
    int total_blocks = nope_blocks + rope_blocks;
    if (total_blocks == 0) return;

    int64_t buffer_stride = kv_buffer.stride(0);
    int64_t nope_stride = k_nope.stride(0);
    int64_t rope_stride = k_rope.stride(0);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_concat_and_store_mla_kv_kernel<src_t, dst_t, MODE>
        <<<total_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            k_nope.data_ptr<src_t>(),
            k_rope.data_ptr<src_t>(),
            reinterpret_cast<dst_t*>(kv_buffer.data_ptr()),
            loc.data_ptr<int64_t>(),
            buffer_stride,
            nope_stride,
            rope_stride,
            nope_dim,
            rope_dim,
            num_tokens,
            nope_blocks,
            nope_threads_per_token,
            rope_threads_per_token
        );
}

template <typename src_t>
static void dispatch_fused_concat_and_store_mla_kv_cache_dst(
    const torch::Tensor& k_nope,
    const torch::Tensor& k_rope,
    torch::Tensor& kv_buffer,
    const torch::Tensor& loc,
    const std::string& fp8_dtype
) {
    auto dst_dtype = kv_buffer.scalar_type();
    if (dst_dtype == torch::kBFloat16) {
        launch_fused_concat_and_store_mla_kv_cache<src_t, at::BFloat16, 0>(
            k_nope, k_rope, kv_buffer, loc);
    } else if (dst_dtype == torch::kHalf) {
        launch_fused_concat_and_store_mla_kv_cache<src_t, at::Half, 0>(
            k_nope, k_rope, kv_buffer, loc);
    } else if (dst_dtype == torch::kFloat) {
        launch_fused_concat_and_store_mla_kv_cache<src_t, float, 0>(
            k_nope, k_rope, kv_buffer, loc);
    } else if (dst_dtype == torch::kUInt8) {
        bool is_e4m3 = (fp8_dtype == "fp8_e4m3" || fp8_dtype == "fp8_e4m3fn" ||
                        fp8_dtype == "fp8_e4m3fnuz");
        if (is_e4m3) {
            launch_fused_concat_and_store_mla_kv_cache<src_t, uint8_t, 1>(
                k_nope, k_rope, kv_buffer, loc);
        } else {
            launch_fused_concat_and_store_mla_kv_cache<src_t, uint8_t, 2>(
                k_nope, k_rope, kv_buffer, loc);
        }
    } else {
        TORCH_CHECK(false,
            "fused_concat_and_store_mla_kv_cache: unsupported kv_buffer dtype ",
            dst_dtype);
    }
}

void fused_concat_and_store_mla_kv_cache(
    torch::Tensor k_nope,     // [..., nope_dim]
    torch::Tensor k_rope,     // [..., rope_dim]
    torch::Tensor kv_buffer,  // [total_slots, 1, nope_dim + rope_dim]
    torch::Tensor loc,        // [num_tokens], int64
    const std::string& fp8_dtype = "fp8_e4m3"
) {
    TORCH_CHECK(k_nope.dim() >= 2,
        "k_nope must have at least 2 dimensions, got ", k_nope.dim());
    TORCH_CHECK(k_rope.dim() >= 2,
        "k_rope must have at least 2 dimensions, got ", k_rope.dim());

    const int64_t nope_dim = k_nope.size(-1);
    const int64_t rope_dim = k_rope.size(-1);
    TORCH_CHECK(nope_dim > 0, "k_nope last dimension must be positive");
    TORCH_CHECK(rope_dim > 0, "k_rope last dimension must be positive");

    // Treat the last dimension as the feature dimension and collapse all
    // leading dimensions into the token dimension. This accepts layouts such
    // as [N, D], [N, 1, D], and [1, N, 1, D].
    k_nope = k_nope.reshape({k_nope.numel() / nope_dim, nope_dim});
    k_rope = k_rope.reshape({k_rope.numel() / rope_dim, rope_dim});

    torch::Tensor kv_buffer_view = kv_buffer;
    if (kv_buffer_view.dim() == 3) kv_buffer_view = kv_buffer_view.squeeze(1);

    k_nope = k_nope.contiguous();
    k_rope = k_rope.contiguous();
    loc = loc.contiguous();

    TORCH_CHECK(loc.scalar_type() == torch::kInt64,
        "loc tensor must be Int64, got ", loc.scalar_type());
    TORCH_CHECK(k_nope.scalar_type() == k_rope.scalar_type(),
        "k_nope and k_rope must have the same dtype");
    TORCH_CHECK(k_nope.size(0) == loc.size(0),
        "num_tokens mismatch between k_nope and loc");
    TORCH_CHECK(k_rope.size(0) == loc.size(0),
        "num_tokens mismatch between k_rope and loc");
    TORCH_CHECK(kv_buffer_view.size(-1) == k_nope.size(-1) + k_rope.size(-1),
        "kv_buffer last dim (", kv_buffer_view.size(-1),
        ") must equal nope_dim + rope_dim (",
        k_nope.size(-1) + k_rope.size(-1), ")");

    auto src_dtype = k_nope.scalar_type();
    if (src_dtype == torch::kBFloat16) {
        dispatch_fused_concat_and_store_mla_kv_cache_dst<at::BFloat16>(
            k_nope, k_rope, kv_buffer_view, loc, fp8_dtype);
    } else if (src_dtype == torch::kHalf) {
        dispatch_fused_concat_and_store_mla_kv_cache_dst<at::Half>(
            k_nope, k_rope, kv_buffer_view, loc, fp8_dtype);
    } else if (src_dtype == torch::kFloat) {
        dispatch_fused_concat_and_store_mla_kv_cache_dst<float>(
            k_nope, k_rope, kv_buffer_view, loc, fp8_dtype);
    } else {
        TORCH_CHECK(false,
            "fused_concat_and_store_mla_kv_cache: unsupported input dtype ",
            src_dtype);
    }
}

void quantize_nope_fp8_rope_bf16_pack_store(
    torch::Tensor k_bf16,
    torch::Tensor buf,
    torch::Tensor loc,
    int page_size,
    float eps
) {
    constexpr int TOTAL_DIM = 512;
    constexpr int NUM_NOPE_ROPE_BYTES_PER_TOKEN = 448 + 64 * 2;
    constexpr int PADDED_SCALE_ELEMS_PER_TOKEN = 8;

    TORCH_CHECK(k_bf16.dim() == 2, "k_bf16 must be 2D, got ", k_bf16.dim(), "D");
    TORCH_CHECK(k_bf16.size(1) >= TOTAL_DIM,
        "k_bf16 last dim must be >= 512, got ", k_bf16.size(1));
    TORCH_CHECK(k_bf16.scalar_type() == torch::kBFloat16,
        "k_bf16 must be bfloat16, got ", k_bf16.scalar_type());
    TORCH_CHECK(buf.dim() == 2, "buf must be 2D, got ", buf.dim(), "D");
    TORCH_CHECK(buf.scalar_type() == torch::kUInt8,
        "buf must be uint8, got ", buf.scalar_type());
    TORCH_CHECK(buf.is_contiguous(),
        "buf must be contiguous because it is written in-place");
    TORCH_CHECK(loc.dim() == 1, "loc must be 1D, got ", loc.dim(), "D");
    TORCH_CHECK(loc.scalar_type() == torch::kInt64 || loc.scalar_type() == torch::kInt32,
        "loc must be int64 or int32, got ", loc.scalar_type());
    TORCH_CHECK(page_size > 0, "page_size must be positive");
    TORCH_CHECK(buf.size(1) >= page_size * (NUM_NOPE_ROPE_BYTES_PER_TOKEN + PADDED_SCALE_ELEMS_PER_TOKEN),
        "buf page is too small for DeepSeek V4 packed K cache");
    TORCH_CHECK(k_bf16.size(0) == loc.size(0),
        "k_bf16 token count must match loc length");

    k_bf16 = k_bf16.contiguous();
    loc = loc.contiguous();

    if (loc.scalar_type() == torch::kInt64) {
        launch_quantize_nope_fp8_rope_bf16_pack_store<int64_t>(
            k_bf16, buf, loc, page_size, eps);
    } else {
        launch_quantize_nope_fp8_rope_bf16_pack_store<int32_t>(
            k_bf16, buf, loc, page_size, eps);
    }
}

} // namespace native
} // namespace at
