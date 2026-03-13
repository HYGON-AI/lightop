#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/AccumulateType.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

namespace at {  
namespace native {

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))    

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
        c10::detail::fp32_from_bits(f_bits) + 
        c10::detail::fp32_from_bits(denorm_mask)
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

inline __device__ uint32_t float_to_fp8e4m3_x4(float a, float b, float c, float d)
{
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

inline __device__ uint32_t float_to_fp8e5m2_x4(float a, float b, float c, float d)
{
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

template <typename T,int reducesize=C10_WARP_SIZE>
__inline__ __device__ T WarpReduceMax_ROW(T val) {
#pragma unroll
  for (int offset = reducesize/2; offset > 0; offset >>= 1) {
    val =  fmaxf(val,  __shfl_down(val, offset));
  }
  return val;
}

template <typename T,int block_size=512>
__inline__ __device__ T BlockReduceMax_ROW(T val, T* shared) {
  constexpr int share_size=block_size/C10_WARP_SIZE;
  val = WarpReduceMax_ROW<T>(val);
  if constexpr(block_size==C10_WARP_SIZE)
  {
    return val;
  }
  else{
    const int lid = threadIdx.x % C10_WARP_SIZE;
    const int wid = threadIdx.x / C10_WARP_SIZE;
    if (lid == 0&&wid<share_size) {
      shared[wid] = val;
    }
    __syncthreads();//lj
    if (wid == 0&&lid<share_size) {
      val = WarpReduceMax_ROW<T,share_size>(shared[lid]);
    }
    return val;
  }
}

/*
scale = Xmax / fp8max
quant_out = x / scale = x * (fp8max / Xmax)
 */
template <typename scalar_t, typename scale_t, 
          int VEC, int block_size, bool is_e4m3>
__global__ void per_token_quant_fp8_kernel(
    const scalar_t* __restrict__ input, uint8_t* __restrict__ output,
    float* scale_out, const int hidden_size) {

    int bid = blockIdx.x;  
    
    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecFp8Type = at::native::memory::aligned_vector<uint8_t, VEC>;

    int tid = threadIdx.x;
    int tcol = hidden_size / VEC;
    
    scalar_t input_vec[VEC];
    int64_t idx = bid * tcol + tid;
    idx *= VEC;

    float input_float_vec[VEC];
    float row_max = 0.f;
    if (tid < tcol) {
        *(VecType*)input_vec = *(VecType*)(input+idx);
        #pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            input_float_vec[ii] = static_cast<float>(input_vec[ii]);
            row_max = fmaxf(row_max,fabs(input_float_vec[ii]));
        }
    }
    row_max = BlockReduceMax_ROW<float, block_size>(row_max, val_shared);
    __shared__ float s_token_scale;
    
    // e4m3 max 448.0, e5m2 max 57344.0
    constexpr float FP8_MAX_VAL = is_e4m3 ? 448.0f : 57344.0f;

    if(tid == 0){
        s_token_scale = row_max;  
        scale_out[bid] = s_token_scale / FP8_MAX_VAL;
    }
    __syncthreads();

    float inv_s = (s_token_scale == 0.f) ? 0.f : FP8_MAX_VAL / s_token_scale;

    uint8_t out_vec[VEC];
    if (tid < tcol) {
        if constexpr (VEC % 4 == 0) {
            #pragma unroll
            for(int ii = 0; ii < VEC; ii+=4){
                uint32_t packed;
                if constexpr (is_e4m3) {
                    packed = float_to_fp8e4m3_x4(
                        input_float_vec[ii] * inv_s,
                        input_float_vec[ii+1] * inv_s,
                        input_float_vec[ii+2] * inv_s,
                        input_float_vec[ii+3] * inv_s);
                } else {
                    packed = float_to_fp8e5m2_x4(
                        input_float_vec[ii] * inv_s,
                        input_float_vec[ii+1] * inv_s,
                        input_float_vec[ii+2] * inv_s,
                        input_float_vec[ii+3] * inv_s);
                }
                *(uint32_t*)(&out_vec[ii]) = packed;
            }
        } else {
            #pragma unroll
            for(int ii = 0; ii < VEC; ii++){
                if constexpr (is_e4m3) {
                    out_vec[ii] = float_to_fp8e4m3(input_float_vec[ii] * inv_s);
                } else {
                    out_vec[ii] = float_to_fp8e5m2(input_float_vec[ii] * inv_s);
                }
            }
        }
        *(VecFp8Type*)(output+idx) = *(VecFp8Type*)out_vec;
    }
}

void per_token_quant_fp8(
    torch::Tensor& out,         
    torch::Tensor const& input, 
    torch::Tensor& scales)
{
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(scales.is_contiguous());
    TORCH_CHECK(out.scalar_type() == torch::kFloat8_e4m3fn || out.scalar_type() == torch::kFloat8_e5m2, 
                "Output tensor must be of type float8_e4m3fn or float8_e5m2");
 
    bool is_e4m3 = (out.scalar_type() == torch::kFloat8_e4m3fn);

    int const hidden_size = input.size(-1);
    int64_t num_tokens = input.numel() / hidden_size;
    TORCH_CHECK(hidden_size <= 16384, "hidden_size (", hidden_size, ") exceeds maximum supported size of 16384");
    TORCH_CHECK(hidden_size % 16 == 0, "hidden_size (", hidden_size, ") must be a multiple of 16 for vectorized memory access");

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
    bool ptrs_are_aligned = inp_ptr % 16 == 0;
    
    auto out_ptr = reinterpret_cast<uint8_t*>(out.data_ptr());

    #define LAUNCH_FP8_KERNEL(VEC_SIZE, BLOCK_SIZE, IS_E4M3_BOOL)                     \
        per_token_quant_fp8_kernel<scalar_t, float, VEC_SIZE, BLOCK_SIZE, IS_E4M3_BOOL> \
            <<<num_tokens, BLOCK_SIZE, 0, stream>>>(                                    \
                input.data_ptr<scalar_t>(), out_ptr, scales.data_ptr<float>(),          \
                hidden_size)

    #define DISPATCH_FP8_KERNEL_SIZE(IS_E4M3_BOOL)               \
        if (hidden_size <= 64) {                                 \
            LAUNCH_FP8_KERNEL(1, 64, IS_E4M3_BOOL);              \
        } else if (hidden_size <= 128) {                         \
            LAUNCH_FP8_KERNEL(2, 64, IS_E4M3_BOOL);              \
        } else if (hidden_size <= 256) {                         \
            LAUNCH_FP8_KERNEL(4, 64, IS_E4M3_BOOL);              \
        } else if (hidden_size <= 512) {                         \
            LAUNCH_FP8_KERNEL(4, 128, IS_E4M3_BOOL);             \
        } else if (hidden_size <= 1024) {                        \
            LAUNCH_FP8_KERNEL(8, 128, IS_E4M3_BOOL);             \
        } else if (hidden_size <= 2048) {                        \
            LAUNCH_FP8_KERNEL(16, 128, IS_E4M3_BOOL);            \
        } else if (hidden_size <= 4096) {                        \
            LAUNCH_FP8_KERNEL(16, 256, IS_E4M3_BOOL);            \
        } else if (hidden_size <= 8192) {                        \
            LAUNCH_FP8_KERNEL(16, 512, IS_E4M3_BOOL);            \
        }else if (hidden_size <= 16384) {                        \
            LAUNCH_FP8_KERNEL(16, 1024, IS_E4M3_BOOL);           \
        }

      VLLM_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "per_token_quant_fp8_kernel", [&] {
                if (is_e4m3) {
                    DISPATCH_FP8_KERNEL_SIZE(true);
                } else {
                    DISPATCH_FP8_KERNEL_SIZE(false);
                }
            }
      );
}
} // namespace native
} // namespace at