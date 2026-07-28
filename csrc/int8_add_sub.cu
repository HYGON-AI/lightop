#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

namespace at {
namespace native {

// 1. 向量化
template <int VEC, bool IS_ADD, bool SATURATE>
__global__ void int8_math_vec_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int8_t* __restrict__ out,
    int64_t total_vec) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    using VecType = at::native::memory::aligned_vector<int8_t, VEC>;
    const VecType* a_vec = reinterpret_cast<const VecType*>(a);
    const VecType* b_vec = reinterpret_cast<const VecType*>(b);
    VecType* out_vec = reinterpret_cast<VecType*>(out);

    for (int64_t i = idx; i < total_vec; i += stride) {
        VecType va = a_vec[i];
        VecType vb = b_vec[i];
        VecType vout;

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            if constexpr (SATURATE) {
                // 饱和运算
                int val_a = static_cast<int>(va.val[j]);
                int val_b = static_cast<int>(vb.val[j]);
                int res = IS_ADD ? (val_a + val_b) : (val_a - val_b);
                res = res > 127 ? 127 : (res < -128 ? -128 : res);
                vout.val[j] = static_cast<int8_t>(res);
            } else {
                // 截断运算
                uint8_t val_a = static_cast<uint8_t>(va.val[j]);
                uint8_t val_b = static_cast<uint8_t>(vb.val[j]);
                uint8_t res = IS_ADD ? (val_a + val_b) : (val_a - val_b);
                vout.val[j] = static_cast<int8_t>(res);
            }
        }
        out_vec[i] = vout;
    }
}

// 2. 标量化
template <bool IS_ADD, bool SATURATE>
__global__ void int8_math_scalar_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int8_t* __restrict__ out,
    int64_t total) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        if constexpr (SATURATE) {
            int val_a = static_cast<int>(a[i]);
            int val_b = static_cast<int>(b[i]);
            int res = IS_ADD ? (val_a + val_b) : (val_a - val_b);
            res = res > 127 ? 127 : (res < -128 ? -128 : res);
            out[i] = static_cast<int8_t>(res);
        } else {
            uint8_t val_a = static_cast<uint8_t>(a[i]);
            uint8_t val_b = static_cast<uint8_t>(b[i]);
            uint8_t res = IS_ADD ? (val_a + val_b) : (val_a - val_b);
            out[i] = static_cast<int8_t>(res);
        }
    }
}

template <bool IS_ADD>
torch::Tensor launch_int8_math_ultimate(const torch::Tensor& a, const torch::Tensor& b, bool saturate) {
    auto a_contig = a.contiguous();
    auto b_contig = b.contiguous();
    auto out = torch::empty_like(a_contig);

    int64_t total = a_contig.numel();
    if (total == 0) return out;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // 获取物理首地址
    auto ptr_a = reinterpret_cast<std::uintptr_t>(a_contig.data_ptr<int8_t>());
    auto ptr_b = reinterpret_cast<std::uintptr_t>(b_contig.data_ptr<int8_t>());
    auto ptr_out = reinterpret_cast<std::uintptr_t>(out.data_ptr<int8_t>());

    // 计算16字节对齐的偏移量
    uintptr_t offset_a = ptr_a % 16;
    uintptr_t offset_b = ptr_b % 16;
    uintptr_t offset_out = ptr_out % 16;

    constexpr int VEC = 16;
    int threads = 256;

    if (offset_a == offset_b && offset_b == offset_out) {
        
        // 计算单独处理，以便后续对齐
        int64_t head_elements = (offset_a == 0) ? 0 : (16 - offset_a);
        head_elements = std::min(head_elements, total);

        int64_t remaining = total - head_elements;
        int64_t vec_elements = (remaining / VEC) * VEC;
        int64_t tail_elements = remaining % VEC;

        // 指针基址获取
        const int8_t* a_ptr = a_contig.data_ptr<int8_t>();
        const int8_t* b_ptr = b_contig.data_ptr<int8_t>();
        int8_t* out_ptr = out.data_ptr<int8_t>();

        // 1. 处理不对齐的前几个元素 (Scalar)
        if (head_elements > 0) {
            if (saturate) {
                int8_math_scalar_kernel<IS_ADD, true><<<1, 32, 0, stream>>>(a_ptr, b_ptr, out_ptr, head_elements);
            } else {
                int8_math_scalar_kernel<IS_ADD, false><<<1, 32, 0, stream>>>(a_ptr, b_ptr, out_ptr, head_elements);
            }
        }

        // 2. 地址已经对齐 (Vec 模式)
        if (vec_elements > 0) {
            int64_t total_vec = vec_elements / VEC;
            int blocks = std::min(static_cast<int64_t>(65535), (total_vec + threads - 1) / threads);
            
            // 指针偏移 head_elements
            if (saturate) {
                int8_math_vec_kernel<VEC, IS_ADD, true><<<blocks, threads, 0, stream>>>(
                    a_ptr + head_elements, b_ptr + head_elements, out_ptr + head_elements, total_vec);
            } else {
                int8_math_vec_kernel<VEC, IS_ADD, false><<<blocks, threads, 0, stream>>>(
                    a_ptr + head_elements, b_ptr + head_elements, out_ptr + head_elements, total_vec);
            }
        }

        // 3. 处理剩下的 (Scalar 模式)
        if (tail_elements > 0) {
            int64_t offset = head_elements + vec_elements;
            if (saturate) {
                int8_math_scalar_kernel<IS_ADD, true><<<1, 32, 0, stream>>>(
                    a_ptr + offset, b_ptr + offset, out_ptr + offset, tail_elements);
            } else {
                int8_math_scalar_kernel<IS_ADD, false><<<1, 32, 0, stream>>>(
                    a_ptr + offset, b_ptr + offset, out_ptr + offset, tail_elements);
            }
        }
    } 
    else {
        // 纯标量模式
        int blocks = std::min(static_cast<int64_t>(65535), (total + threads - 1) / threads);
        if (saturate) {
            int8_math_scalar_kernel<IS_ADD, true><<<blocks, threads, 0, stream>>>(
                a_contig.data_ptr<int8_t>(), b_contig.data_ptr<int8_t>(), out.data_ptr<int8_t>(), total);
        } else {
            int8_math_scalar_kernel<IS_ADD, false><<<blocks, threads, 0, stream>>>(
                a_contig.data_ptr<int8_t>(), b_contig.data_ptr<int8_t>(), out.data_ptr<int8_t>(), total);
        }
    }
    return out;
}

torch::Tensor add_int8(const torch::Tensor& a, const torch::Tensor& b, bool saturate=true) {
    return launch_int8_math_ultimate<true>(a, b, saturate);
}

torch::Tensor sub_int8(const torch::Tensor& a, const torch::Tensor& b, bool saturate=true) {
    return launch_int8_math_ultimate<false>(a, b, saturate);
}

} // namespace native
} // namespace at
