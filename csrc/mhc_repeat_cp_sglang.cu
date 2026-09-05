// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Opt-in SGLang mHC repeat+CP-select specialization. Generic repeat and
// collective implementations remain unchanged.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <cstring>
#include <unordered_map>

#if defined(USE_ROCM) && USE_ROCM
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#endif

namespace at {
namespace native {
namespace mhc_repeat_cp_sglang_detail {

constexpr int64_t kRows = 32768;
constexpr int64_t kHidden = 4096;
constexpr int64_t kHC = 4;
constexpr int64_t kCP = 8;
constexpr int64_t kLocalRows = kRows / kCP;
constexpr uint32_t kVector = 8;
constexpr uint32_t kVectorsPerRow = kHidden / kVector;
constexpr uint32_t kVectorCount = kLocalRows * kVectorsPerRow;
constexpr uint32_t kBlock = 256;

const char* metadata_error(
    const torch::Tensor& hidden_states, int64_t hc_mult,
    int64_t cp_size, int64_t cp_rank) {
  if (!hidden_states.defined() || hidden_states.layout() != c10::Layout::Strided ||
      !hidden_states.is_cuda()) return "defined strided GPU input required";
  if (hidden_states.requires_grad()) return "inference-only input required";
  if (hidden_states.is_conj() || hidden_states.is_neg())
    return "unresolved conjugate/negative view unsupported";
  if (hidden_states.scalar_type() != torch::kBFloat16)
    return "BF16 input required";
  if (hidden_states.dim() != 2 || hidden_states.size(0) != kRows ||
      hidden_states.size(1) != kHidden ||
      hidden_states.stride(0) != kHidden || hidden_states.stride(1) != 1)
    return "unsupported input; target is dense BF16[32768,4096]";
  if (hc_mult != kHC || cp_size != kCP || cp_rank < 0 || cp_rank >= kCP)
    return "unsupported HC/CP metadata; target is HC4,CP8,rank[0,7]";
  if ((reinterpret_cast<uintptr_t>(hidden_states.data_ptr()) & 15) != 0)
    return "16-byte aligned input pointer required";
  return nullptr;
}

bool supported_device(int device_index) {
#if defined(USE_ROCM) && USE_ROCM
  static thread_local std::unordered_map<int, bool> supported;
  const auto found = supported.find(device_index);
  if (found != supported.end()) return found->second;
  hipDeviceProp_t_v2 properties{};
  if (hipGetDeviceProperties_v2(&properties, device_index) != hipSuccess) return false;
  const bool value = std::strncmp(properties.gcnArchName, "gfx936", 6) == 0 &&
      (properties.gcnArchName[6] == '\0' || properties.gcnArchName[6] == ':');
  supported.emplace(device_index, value);
  return value;
#else
  return false;
#endif
}

#if defined(USE_ROCM) && USE_ROCM
struct alignas(16) Bf16x8 { __hip_bfloat16 values[kVector]; };
static_assert(sizeof(Bf16x8) == 16);

__global__ void mhc_repeat_cp_sglang_vec8_kernel(
    const __hip_bfloat16* __restrict__ hidden_states,
    __hip_bfloat16* __restrict__ out, int32_t cp_rank) {
  const uint32_t vector_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (vector_index >= kVectorCount) return;
  const uint32_t local_row = vector_index / kVectorsPerRow;
  const uint32_t column_vector = vector_index - local_row * kVectorsPerRow;
  const int64_t source_row = static_cast<int64_t>(local_row) * kCP + cp_rank;
  const auto value = reinterpret_cast<const Bf16x8*>(
      hidden_states + source_row * kHidden)[column_vector];
  auto* output = reinterpret_cast<Bf16x8*>(
      out + static_cast<int64_t>(local_row) * kHC * kHidden);
#pragma unroll
  for (int64_t plane = 0; plane < kHC; ++plane) {
    output[plane * kVectorsPerRow + column_vector] = value;
  }
}
#endif

}  // namespace mhc_repeat_cp_sglang_detail

bool supports_mhc_repeat_cp_sglang(
    const torch::Tensor& hidden_states, int64_t hc_mult,
    int64_t cp_size, int64_t cp_rank) {
  using namespace mhc_repeat_cp_sglang_detail;
  return metadata_error(hidden_states, hc_mult, cp_size, cp_rank) == nullptr &&
      supported_device(hidden_states.get_device());
}

torch::Tensor mhc_repeat_cp_sglang(
    const torch::Tensor& hidden_states, int64_t hc_mult,
    int64_t cp_size, int64_t cp_rank) {
  using namespace mhc_repeat_cp_sglang_detail;
  const auto* error = metadata_error(hidden_states, hc_mult, cp_size, cp_rank);
  TORCH_CHECK(error == nullptr, "SGLang mHC repeat+CP: ", error);
  TORCH_CHECK(supported_device(hidden_states.get_device()),
              "SGLang mHC repeat+CP requires gfx936/ROCm");
  auto out = torch::empty({kLocalRows, kHC, kHidden}, hidden_states.options());
#if defined(USE_ROCM) && USE_ROCM
  const at::cuda::OptionalCUDAGuard device_guard(device_of(hidden_states));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* input = reinterpret_cast<const __hip_bfloat16*>(
      hidden_states.data_ptr<at::BFloat16>());
  auto* output = reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>());
  constexpr uint32_t blocks = (kVectorCount + kBlock - 1) / kBlock;
  mhc_repeat_cp_sglang_vec8_kernel<<<blocks, kBlock, 0, stream>>>(
      input, output, static_cast<int32_t>(cp_rank));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "SGLang mHC repeat+CP requires a ROCm build");
#endif
  return out;
}

}  // namespace native
}  // namespace at
