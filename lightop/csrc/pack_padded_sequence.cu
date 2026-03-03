#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/AccumulateType.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <tuple>

namespace at {
namespace native {
#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))    

// 向量化
template <typename scalar_t, int VEC>
__global__ void pack_padded_sequence_vec_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t* __restrict__ lengths,
    const int64_t* __restrict__ offsets,
    int T, int B, int D,
    int stride_t, int stride_b) {

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    int64_t tcol = D / VEC;
    
    int64_t total_vec = static_cast<int64_t>(T) * B * tcol;

    using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;

    for (int64_t i = idx; i < total_vec; i += stride) {
        int64_t d_vec = i % tcol;
        int64_t row_idx = i / tcol;
        int64_t b = row_idx % B;
        int64_t t = row_idx / B;

        if (t < lengths[b]) {
            int64_t in_idx = t * stride_t + b * stride_b + d_vec * VEC;
            int64_t out_idx = offsets[t] * D + b * D + d_vec * VEC;
            *(reinterpret_cast<VecType*>(output + out_idx)) = *(reinterpret_cast<const VecType*>(input + in_idx));
        }
    }
}


// 标量化
template <typename scalar_t>
__global__ void pack_padded_sequence_scalar_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t* __restrict__ lengths,
    const int64_t* __restrict__ offsets,
    int T, int B, int D,
    int stride_t, int stride_b, int stride_d) {

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t total = T * B * D;

    // if (idx < total) {
    for (int64_t i = idx; i < total; i += stride) {
        int d = i % D;
        int b = (i / D) % B;
        int t = i / (B * D);

        if (t < lengths[b]) {
            int64_t in_idx = t * stride_t + b * stride_b + d * stride_d;
            int64_t out_idx = offsets[t] * D + b * D + d;
            output[out_idx] = input[in_idx];
        }
    }
}

// API
std::tuple<torch::Tensor, torch::Tensor> pack_padded_sequence_custom(
    const torch::Tensor& input,
    const torch::Tensor& lengths,
    bool batch_first) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(lengths.is_cpu(), "lengths must be a CPU tensor for fast host processing");
    TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");

    auto logical_input = batch_first ? input.transpose(0, 1) : input;
    
    int T = logical_input.size(0);
    int B = logical_input.size(1);
    int D = logical_input.size(2);

    int stride_t = logical_input.stride(0);
    int stride_b = logical_input.stride(1);
    int stride_d = logical_input.stride(2);

    const int64_t* lengths_data = lengths.data_ptr<int64_t>();
    int64_t max_len = lengths_data[0]; 
    
    std::vector<int64_t> batch_sizes_vec(max_len);
    std::vector<int64_t> offsets_vec(max_len);
    int64_t total_elements = 0;

    for (int64_t t = 0; t < max_len; ++t) {
        int64_t count = 0;
        for (int b = 0; b < B; ++b) {
            if (lengths_data[b] > t) count++;
            else break;
        }
        batch_sizes_vec[t] = count;
        offsets_vec[t] = total_elements;
        total_elements += count;
    }

    auto output = torch::empty({total_elements, D}, logical_input.options());
    auto batch_sizes_tensor = torch::tensor(batch_sizes_vec, torch::dtype(torch::kInt64));
    
    auto lengths_gpu = lengths.to(input.device());
    auto offsets_gpu = torch::tensor(offsets_vec, torch::dtype(torch::kInt64).device(input.device()));

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto inp_ptr = reinterpret_cast<std::uintptr_t>(logical_input.data_ptr());
    auto out_ptr = reinterpret_cast<std::uintptr_t>(output.data_ptr());
    
    bool ptrs_aligned = (inp_ptr % 16 == 0) && (out_ptr % 16 == 0);
    bool is_dense_d = (stride_d == 1);

    VLLM_DISPATCH_FLOATING_TYPES(
        logical_input.scalar_type(),
        "pack_padded_sequence_kernel",
        [&] {
        constexpr int VEC = 16 / sizeof(scalar_t);

        if (is_dense_d && ptrs_aligned && (D % VEC == 0)) {
            int64_t tcol = D / VEC;
            int64_t total_vec = static_cast<int64_t>(T) * B * tcol;
            
            int threads = 256;
            int blocks = std::min(static_cast<int64_t>(65535), (total_vec + threads - 1) / threads);

            pack_padded_sequence_vec_kernel<scalar_t, VEC><<<blocks, threads, 0, stream>>>(
                logical_input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                lengths_gpu.data_ptr<int64_t>(),
                offsets_gpu.data_ptr<int64_t>(),
                T, B, D,
                stride_t, stride_b
            );
        } else {
            int64_t total = static_cast<int64_t>(T) * B * D;
            int threads = 256;
            // int blocks = (total + threads - 1) / threads;
            int blocks = std::min(static_cast<int64_t>(65535), (total + threads - 1) / threads);
            pack_padded_sequence_scalar_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                logical_input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                lengths_gpu.data_ptr<int64_t>(),
                offsets_gpu.data_ptr<int64_t>(),
                T, B, D,
                stride_t, stride_b, stride_d
            );
        }
    });

    return std::make_tuple(output, batch_sizes_tensor);
}

} // namespace native
} // namespace at
