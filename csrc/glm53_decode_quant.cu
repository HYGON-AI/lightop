// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/dtk_macros.h>
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>
#include <c10/hip/HIPGuard.h>
#include <ATen/native/hip/MemoryAccess.cuh>
#include <ATen/AccumulateType.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>


#if defined(TORCH_VERSION_OVER_2120)
#define C10_WARP_SIZE 64
#elif defined(TORCH_VERSION_OVER_290)
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif

namespace glm53_quant_broadcast_e512a408c6134981b05834a6a5ed4654 {
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
template <typename scalar_t, typename scale_t,
          int VEC, int block_size, bool is_e4m3>
__global__ void per_token_quant_fp8_kernel_broadcast_inverse(
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
    __shared__ float s_inverse_scale;

    // e4m3 max 448.0, e5m2 max 57344.0
    constexpr float FP8_MAX_VAL = is_e4m3 ? 448.0f : 57344.0f;

    if(tid == 0){
        s_token_scale = row_max;
        scale_out[bid] = s_token_scale / FP8_MAX_VAL;
        s_inverse_scale = (s_token_scale == 0.f) ? 0.f : FP8_MAX_VAL / s_token_scale;
    }
    __syncthreads();

    float inv_s = s_inverse_scale;

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

}  // namespace glm53_quant_broadcast_e512a408c6134981b05834a6a5ed4654

#include <cstdint>
#include <limits>

namespace glm53_quant_broadcast_e512a408c6134981b05834a6a5ed4654 {
static_assert(C10_WARP_SIZE == 64, "Audited gfx938 reduction requires warp64");
struct StorageRange {
  uintptr_t begin;
  uintptr_t end;
  const void* identity;
};

StorageRange checked_storage(const torch::Tensor& tensor) {
  const auto storage = tensor.storage();
  const auto begin = reinterpret_cast<uintptr_t>(storage.data_ptr().get());
  const auto bytes = storage.nbytes();
  TORCH_CHECK(begin != 0 && bytes > 0, "Expected allocated storage");
  TORCH_CHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin,
              "Storage address overflow");
  const auto offset = tensor.storage_offset();
  const auto count = tensor.numel();
  const auto element_bytes = tensor.element_size();
  TORCH_CHECK(offset >= 0 && count > 0, "Invalid storage extent");
  TORCH_CHECK(static_cast<uint64_t>(offset) <= bytes / element_bytes,
              "Storage offset exceeds backing allocation");
  const auto offset_bytes = static_cast<uint64_t>(offset) * element_bytes;
  TORCH_CHECK(static_cast<uint64_t>(count) <=
                  (bytes - offset_bytes) / element_bytes,
              "Tensor extent exceeds backing allocation");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) ==
                  begin + offset_bytes,
              "Tensor pointer does not match storage offset");
  return {begin, begin + bytes, storage.unsafeGetStorageImpl()};
}

void check_disjoint(const StorageRange& a, const StorageRange& b) {
  TORCH_CHECK(a.identity != b.identity &&
                  (a.end <= b.begin || b.end <= a.begin),
              "Input, output and scales must have disjoint backing storage");
}

void checked_launch(torch::Tensor& out_q, const torch::Tensor& x,
                    torch::Tensor& out_scale, bool candidate) {
  TORCH_CHECK(x.is_cuda() && out_q.is_cuda() && out_scale.is_cuda(),
              "Expected HIP device tensors");
  TORCH_CHECK(x.device() == out_q.device() && x.device() == out_scale.device(),
              "Tensor devices differ");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 &&
                  out_q.scalar_type() == torch::kFloat8_e4m3fn &&
                  out_scale.scalar_type() == torch::kFloat32,
              "Only BF16 -> E4M3FN with FP32 scales is validated for this scope");
  TORCH_CHECK(x.dim() == 2 && out_q.dim() == 2 && out_scale.dim() == 2,
              "Expected two-dimensional tensors");
  const auto rows = x.size(0);
  TORCH_CHECK(rows >= 1 && rows <= 30 && x.size(1) == 16384,
              "Only sampled rows1..30 and K16384 are admitted");
  TORCH_CHECK(out_q.sizes() == x.sizes() && out_scale.size(0) == rows &&
                  out_scale.size(1) == 1,
              "Output or scale shape differs");
  TORCH_CHECK(x.is_contiguous() && out_q.is_contiguous() &&
                  out_scale.is_contiguous(), "Expected contiguous tensors");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 64 == 0 &&
                  reinterpret_cast<uintptr_t>(out_q.data_ptr()) % 64 == 0 &&
                  reinterpret_cast<uintptr_t>(out_scale.data_ptr()) % 4 == 0,
              "Input/output require64B alignment and scales require4B");
  const auto xs = checked_storage(x);
  const auto qs = checked_storage(out_q);
  const auto ss = checked_storage(out_scale);
  check_disjoint(xs, qs);
  check_disjoint(xs, ss);
  check_disjoint(qs, ss);
  const c10::cuda::CUDAGuard guard(x.device());
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  auto* q = reinterpret_cast<uint8_t*>(out_q.data_ptr());
  if (candidate) {
    hipLaunchKernelGGL((per_token_quant_fp8_kernel_broadcast_inverse<
                           c10::BFloat16, float, 16, 1024, true>),
                       dim3(rows), dim3(1024), 0, stream,
                       x.data_ptr<c10::BFloat16>(), q,
                       out_scale.data_ptr<float>(), 16384);
  } else {
    hipLaunchKernelGGL((per_token_quant_fp8_kernel<c10::BFloat16, float, 16,
                                                  1024, true>),
                       dim3(rows), dim3(1024), 0, stream,
                       x.data_ptr<c10::BFloat16>(), q,
                       out_scale.data_ptr<float>(), 16384);
  }
}

void control(torch::Tensor& out_q, const torch::Tensor& x,
             torch::Tensor& out_scale) {
  checked_launch(out_q, x, out_scale, false);
}
void candidate(torch::Tensor& out_q, const torch::Tensor& x,
               torch::Tensor& out_scale) {
  checked_launch(out_q, x, out_scale, true);
}
}  // namespace glm53_quant_broadcast_e512a408c6134981b05834a6a5ed4654
