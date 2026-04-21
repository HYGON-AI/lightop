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

} // namespace native
} // namespace at
