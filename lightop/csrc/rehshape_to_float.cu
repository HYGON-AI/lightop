#include <iostream>
#include <vector>
#include "hip/hip_runtime.h"
#include <torch/extension.h>
#include <c10/util/BFloat16.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/cuda/CUDAGuard.h>

template<typename scalar_t> 
static __device__ inline float to_float(scalar_t in){
  if constexpr(!std::is_same<scalar_t, uint16_t>::value){
    return in;
  }
  else{
    union{
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(in) << 16};
    return u.fp32;
  }
}

namespace at {
namespace native {


template <typename scalar_t>
union SafeFloat2Reader {
    float2 f2;
    scalar_t vals[4];
};

template <typename scalar_t>
__global__ void reshape_to_float_kernel(
    const scalar_t* __restrict__ input,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t padded_vocab_size,
    int64_t vocab_size) {

    int64_t v_chunks = (vocab_size + 3) / 4; 
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_chunks = batch_size * v_chunks;

    if (idx < total_chunks) {
        int64_t b = idx / v_chunks;          
        int64_t v_chunk_idx = idx % v_chunks; 
        int64_t v = v_chunk_idx * 4;         
        
        int64_t in_idx = b * padded_vocab_size + v;
        int64_t out_idx = b * vocab_size + v;

        bool is_aligned = (((uintptr_t)(&input[in_idx]) % 8) == 0) &&
                          (((uintptr_t)(&output[out_idx]) % 16) == 0);

        if (v + 3 < vocab_size && is_aligned) {
            SafeFloat2Reader<scalar_t> reader;
            reader.f2 = *reinterpret_cast<const float2*>(&input[in_idx]);

            float4 vec_out = make_float4(
                to_float(reader.vals[0]), to_float(reader.vals[1]),
                to_float(reader.vals[2]), to_float(reader.vals[3])
            );

            *reinterpret_cast<float4*>(&output[out_idx]) = vec_out;
        } else {
            int64_t remaining = (v + 4 > vocab_size) ? (vocab_size - v) : 4;
            #pragma unroll
            for (int i = 0; i < remaining; ++i) {
                output[out_idx + i] = to_float(input[in_idx + i]);
            }
        }
    }
}

torch::Tensor reshape_to_float(torch::Tensor input, int64_t vocab_size) {
    
    input = input.is_contiguous() ? input : input.contiguous();

    auto in_shape = input.sizes();
    TORCH_CHECK(in_shape.size() == 3, "Input tensor must be exactly 3D (e.g., [batch, world_size, chunk])!");

    int64_t bs = in_shape[0];
    int64_t d1 = in_shape[1];
    int64_t d2 = in_shape[2];

    int64_t padded_vocab_size = d1 * d2;
    TORCH_CHECK(vocab_size <= padded_vocab_size, "vocab_size 不能大于 d1 * d2 的总和");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto output = torch::empty({bs, vocab_size}, options);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int64_t v_chunks = (vocab_size + 3) / 4;
    int64_t total_chunks = bs * v_chunks;
    int threads = 512;
    int blocks = (total_chunks + threads - 1) / threads;

    if (input.scalar_type() == torch::ScalarType::Half) {
        reshape_to_float_kernel<torch::Half><<<blocks, threads, 0, stream>>>(
            input.data_ptr<torch::Half>(), output.data_ptr<float>(),
            bs, padded_vocab_size, vocab_size
        );
    } else if (input.scalar_type() == torch::ScalarType::BFloat16) {
        reshape_to_float_kernel<torch::BFloat16><<<blocks, threads, 0, stream>>>(
            input.data_ptr<torch::BFloat16>(), output.data_ptr<float>(),
            bs, padded_vocab_size, vocab_size
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype, expect Half or BFloat16");
    }

    return output;
}

} // namespace native
} // namespace at