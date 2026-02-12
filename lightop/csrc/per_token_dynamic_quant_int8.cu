#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/AccumulateType.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
// #include "reduce.h"

namespace at {
namespace native {
#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))    

static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
// lj
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
// lj

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
    float dst = std::nearbyint(x);//lj
    // float dst = rintf(x);

  // saturate

  // See https://github.com/pytorch/pytorch/issues/127666
  // See https://github.com/llvm/llvm-project/issues/95183
  // hip-clang std::clamp __glibcxx_assert_fail host function when building on
  // Arch/gcc14. The following replaces std::clamp usage with similar logic

//   dst = std::clamp(dst, i8_min, i8_max);

//   dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;//lj
    dst = fminf(fmaxf(dst, i8_min), i8_max);
    return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
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

template <typename scalar_t, typename scale_t, 
          int VEC, int block_size>
__global__ void per_token_dynamic_quant_int8_kernel(
    const scalar_t* __restrict__ input, 
    const float* __restrict__ smoothscale, 
    int8_t* __restrict__ output, 
    float* scale_out, 
    const int hidden_size) {

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    constexpr int share_size = block_size / C10_WARP_SIZE;
    __shared__ float val_shared[share_size];
    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
    using VecInt8Type = at::native::memory::aligned_vector<int8_t, VEC>;
    using VecFloatType = at::native::memory::aligned_vector<float, VEC>;

    int tcol = hidden_size / VEC;
    scalar_t input_vec[VEC];
    int64_t idx = bid * tcol + tid;
    idx *= VEC;

    float smoothscale_vec;
    float input_float_vec[VEC];
    float row_max = 0.f;
    if (tid < tcol) {
        *(VecType*)input_vec = *(VecType*)(input+idx);
        // *(VecFloatType)smoothscale_vec = *(VecFloatType*)(smoothscale+tid*VEC);
        // *(VecFloatType)smoothscale_vec = *reinterpret_cast<VecFloatType*>(const_cast<float*>(&smoothscale[tid * VEC]));
        #pragma unroll
        for (int ii = 0; ii < VEC; ii++) {
            smoothscale_vec = smoothscale[tid * VEC + ii];
            input_float_vec[ii] = static_cast<float>(input_vec[ii]) * smoothscale_vec;
            row_max = fmaxf(row_max,fabs(input_float_vec[ii]));
        }
    }
    row_max = BlockReduceMax_ROW<float, block_size>(row_max, val_shared);
    __shared__ float s_token_scale;     
    if(tid == 0){
        s_token_scale = row_max;  
        scale_out[bid] = s_token_scale / 127.f;
    }
    __syncthreads();//lj

    // float scale_x = s_token_scale / 127.f;
    float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;
    int8_t out_vec[VEC];
    if (tid < tcol) {
        #pragma unroll
        for(int ii = 0;ii < VEC;ii++){
          out_vec[ii] = float_to_int8_rn(input_float_vec[ii] * inv_s);
        }
        *(VecInt8Type*)(output+idx) = *(VecInt8Type*)out_vec;
    }
}

void per_token_dynamic_quant_int8(
    torch::Tensor& out,          // [..., hidden_size]
    torch::Tensor const& input,  // [..., hidden_size]
    torch::Tensor& scales,
    torch::Tensor const& smoothscale) {
    
    TORCH_CHECK(smoothscale.is_contiguous());
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(scales.is_contiguous());

    int const hidden_size = input.size(-1);
    int64_t num_tokens = input.numel() / hidden_size;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
    bool ptrs_are_aligned = inp_ptr % 16 == 0;
    if(hidden_size%16 == 0 && hidden_size <= 8192 && ptrs_are_aligned){
        VLLM_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "per_token_dynamic_quant_int8_kernel", [&] {
                if (hidden_size <= 64){
                    per_token_dynamic_quant_int8_kernel<scalar_t, float, 1, 64>
                        <<<num_tokens, 64, 0, stream>>>(
                            input.data_ptr<scalar_t>(), 
                            smoothscale.data_ptr<float>(),
                            out.data_ptr<int8_t>(),
                            scales.data_ptr<float>(), hidden_size); 
                }
                else if (hidden_size <= 128){
                    per_token_dynamic_quant_int8_kernel<scalar_t, float, 2, 64>
                        <<<num_tokens, 64, 0, stream>>>(
                            input.data_ptr<scalar_t>(), 
                            smoothscale.data_ptr<float>(),
                            out.data_ptr<int8_t>(),
                            scales.data_ptr<float>(), hidden_size); 
                }
                else if (hidden_size <= 256){
                    per_token_dynamic_quant_int8_kernel<scalar_t, float, 4, 64>
                        <<<num_tokens, 128, 0, stream>>>(
                            input.data_ptr<scalar_t>(),
                            smoothscale.data_ptr<float>(),
                            out.data_ptr<int8_t>(),
                            scales.data_ptr<float>(), hidden_size); 
                }
                else if (hidden_size <= 512){
                    per_token_dynamic_quant_int8_kernel<scalar_t, float, 4, 128>
                        <<<num_tokens, 128, 0, stream>>>(
                            input.data_ptr<scalar_t>(),
                            smoothscale.data_ptr<float>(),
                            out.data_ptr<int8_t>(),
                            scales.data_ptr<float>(), hidden_size); 
                }
                else if (hidden_size <= 1024){
                    per_token_dynamic_quant_int8_kernel<scalar_t, float, 8, 128>
                        <<<num_tokens, 128, 0, stream>>>(
                            input.data_ptr<scalar_t>(),
                            smoothscale.data_ptr<float>(),
                            out.data_ptr<int8_t>(),
                            scales.data_ptr<float>(), hidden_size); 
                }
                else if (hidden_size <= 2048){
                    per_token_dynamic_quant_int8_kernel<scalar_t, float, 16, 128>
                        <<<num_tokens, 128, 0, stream>>>(
                            input.data_ptr<scalar_t>(),
                            smoothscale.data_ptr<float>(),
                            out.data_ptr<int8_t>(),
                            scales.data_ptr<float>(), hidden_size); 
                }
                else if (hidden_size <= 4096){
                    per_token_dynamic_quant_int8_kernel<scalar_t, float, 16, 256>
                        <<<num_tokens, 256, 0, stream>>>(
                            input.data_ptr<scalar_t>(),
                            smoothscale.data_ptr<float>(),
                            out.data_ptr<int8_t>(),
                            scales.data_ptr<float>(), hidden_size); 
                }
                else if (hidden_size <= 8192){
                    per_token_dynamic_quant_int8_kernel<scalar_t, float, 16, 512>
                        <<<num_tokens, 512, 0, stream>>>(
                            input.data_ptr<scalar_t>(),
                            smoothscale.data_ptr<float>(),
                            out.data_ptr<int8_t>(),
                            scales.data_ptr<float>(), hidden_size); 
                }
            });
    }
}

}
}