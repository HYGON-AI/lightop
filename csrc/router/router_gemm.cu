#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <hip/hip_runtime.h>
#include <cstring>

namespace {
constexpr int Rows = 64, Experts = 128, Hidden = 6144, Parts = 16;
using Short4 = short __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));
struct alignas(16) Pack { Short4 low, high; };

__global__ void router_partial(const short* x, const short* weight, float* scratch) {
#if defined(__gfx938__)
    const int lane = threadIdx.x, group = lane / 16;
    const int row = blockIdx.y * 16 + lane % 16;
    const int expert = blockIdx.x * 16 + lane % 16;
    const int start = blockIdx.z * (Hidden / Parts);
    Float4 acc = {};
    for (int k = start + 8 * group; k < start + Hidden / Parts; k += 32) {
        const Pack a = *reinterpret_cast<const Pack*>(x + row * Hidden + k);
        const Pack b = *reinterpret_cast<const Pack*>(weight + expert * Hidden + k);
        acc = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a.low, b.low, acc, false, false);
        acc = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a.high, b.high, acc, false, false);
    }
    for (int j = 0; j < 4; ++j)
        scratch[(blockIdx.z * Rows + row) * Experts + blockIdx.x * 16 + group + j * 4] = acc[j];
#endif
}

__global__ void router_sum(const float* scratch, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int part = 0; part < Parts; ++part) sum += scratch[part * Rows * Experts + i];
    out[i] = sum;
}

at::Tensor forward(const at::Tensor& x, const at::Tensor& weight) {
    TORCH_CHECK(x.is_cuda() && weight.device() == x.device(), "expected tensors on the same GPU");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
                x.is_contiguous() && weight.is_contiguous(), "expected contiguous BF16");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == Rows && x.size(1) == Hidden &&
                weight.dim() == 2 && weight.size(0) == Experts && weight.size(1) == Hidden,
                "expected [64,6144] and [128,6144]");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0) &&
                (reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0), "expected 16-byte alignment");
    c10::cuda::CUDAGuard guard(x.device());
    auto arch = at::cuda::getCurrentDeviceProperties()->gcnArchName;
    TORCH_CHECK(std::strncmp(arch, "gfx938", 6) == 0 && (arch[6] == 0 || arch[6] == ':'), "expected gfx938");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto options = x.options().dtype(at::kFloat);
    auto scratch = at::empty({Parts, Rows, Experts}, options);
    auto out = at::empty({Rows, Experts}, options);
    router_partial<<<dim3(Experts / 16, Rows / 16, Parts), 64, 0, stream>>>(
        static_cast<const short*>(x.data_ptr()), static_cast<const short*>(weight.data_ptr()),
        scratch.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    router_sum<<<Rows * Experts / 256, 256, 0, stream>>>(scratch.data_ptr<float>(), out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }
