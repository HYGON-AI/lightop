// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Copy x[rank::cp] directly to [local_rows, hc, hidden]. No global HC-expanded
// tensor, dtype conversion, synchronization, or host read of tensor data.
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>

#if defined(USE_ROCM) && USE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace at::native {
namespace mhc_repeat_cp_detail {

constexpr int kThreads = 256;
constexpr int64_t kVectorBytes = 16;
constexpr int64_t kMaxBlocks = 65535;

int64_t selected_rows(int64_t rows, int64_t cp_size, int64_t rank) {
  return rows <= rank ? 0 : 1 + (rows - 1 - rank) / cp_size;
}

const char* metadata_error(const torch::Tensor& input, int64_t hc_mult,
                           int64_t cp_size, int64_t rank) {
  if (!input.defined() || input.layout() != c10::Layout::Strided ||
      !input.is_cuda()) {
    return "defined strided GPU input required";
  }
  if (input.requires_grad()) return "inference-only input required";
  if (input.is_conj() || input.is_neg() || input.is_quantized()) {
    return "unresolved conjugate/negative or quantized view unsupported";
  }
  if (input.dim() != 2)
    return "two-dimensional [tokens, hidden] input required";
  if (hc_mult <= 0 || cp_size <= 0 || rank < 0 || rank >= cp_size) {
    return "positive HC/CP and rank in [0, CP) required";
  }
  if (input.stride(0) < 0 || input.stride(1) < 0) {
    return "negative input strides unsupported";
  }
  const auto bytes = input.element_size();
  if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8 && bytes != 16) {
    return "unsupported element storage width";
  }
  return nullptr;
}

#if defined(USE_ROCM) && USE_ROCM
struct alignas(16) Vector {
  uint32_t words[4];
};
static_assert(sizeof(Vector) == kVectorBytes);

// A block visits a contiguous tile of one selected row. Source-row addressing
// is independent of HC, so each vector is read once and written HC times.
template <int FixedHC>
__global__ void repeat_vector_kernel(const uint8_t* __restrict__ input,
                                     uint8_t* __restrict__ output,
                                     int64_t row_vectors,
                                     int64_t input_row_bytes,
                                     int64_t local_rows, int64_t hc_mult,
                                     int64_t cp_size, int64_t rank) {
  const int64_t tiles_per_row = (row_vectors + kThreads - 1) / kThreads;
  const int64_t total_tiles = local_rows * tiles_per_row;
  const int64_t copies = FixedHC == 0 ? hc_mult : FixedHC;
  for (int64_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    const int64_t row = tile / tiles_per_row;
    const int64_t column = (tile % tiles_per_row) * kThreads + threadIdx.x;
    if (column >= row_vectors) continue;
    const auto* source = reinterpret_cast<const Vector*>(
        input + (row * cp_size + rank) * input_row_bytes);
    const Vector value = source[column];
    auto* destination =
        reinterpret_cast<Vector*>(output) + row * copies * row_vectors;
#pragma unroll
    for (int64_t copy = 0; copy < copies; ++copy) {
      destination[copy * row_vectors + column] = value;
    }
  }
}

// The fallback accepts arbitrary hidden sizes, storage offsets, and positive
// or zero strides. Byte copies preserve NaN payloads and signed zero exactly.
template <int ElementBytes>
__global__ void repeat_element_kernel(const uint8_t* __restrict__ input,
                                      uint8_t* __restrict__ output,
                                      int64_t hidden, int64_t row_stride,
                                      int64_t column_stride, int64_t local_rows,
                                      int64_t hc_mult, int64_t cp_size,
                                      int64_t rank) {
  const int64_t count = local_rows * hidden;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = index / hidden;
    const int64_t column = index % hidden;
    const auto* source =
        input + ((row * cp_size + rank) * row_stride + column * column_stride) *
                    ElementBytes;
    uint8_t value[ElementBytes];
#pragma unroll
    for (int byte = 0; byte < ElementBytes; ++byte) value[byte] = source[byte];
    for (int64_t copy = 0; copy < hc_mult; ++copy) {
      auto* destination =
          output + ((row * hc_mult + copy) * hidden + column) * ElementBytes;
#pragma unroll
      for (int byte = 0; byte < ElementBytes; ++byte)
        destination[byte] = value[byte];
    }
  }
}

template <int ElementBytes>
void launch_element(const torch::Tensor& input, torch::Tensor& out,
                    int64_t local_rows, int64_t hc_mult, int64_t cp_size,
                    int64_t rank, cudaStream_t stream) {
  const int64_t count = local_rows * input.size(1);
  const int blocks = std::min((count + kThreads - 1) / kThreads, kMaxBlocks);
  repeat_element_kernel<ElementBytes><<<blocks, kThreads, 0, stream>>>(
      static_cast<const uint8_t*>(input.data_ptr()),
      static_cast<uint8_t*>(out.data_ptr()), input.size(1), input.stride(0),
      input.stride(1), local_rows, hc_mult, cp_size, rank);
}
#endif

}  // namespace mhc_repeat_cp_detail

bool supports_mhc_repeat_cp_sglang(const torch::Tensor& input, int64_t hc_mult,
                                   int64_t cp_size, int64_t rank) {
#if defined(USE_ROCM) && USE_ROCM
  return mhc_repeat_cp_detail::metadata_error(input, hc_mult, cp_size, rank) ==
         nullptr;
#else
  return false;
#endif
}

torch::Tensor& mhc_repeat_cp_sglang_out(const torch::Tensor& input,
                                        int64_t hc_mult, int64_t cp_size,
                                        int64_t rank, torch::Tensor& out) {
  using namespace mhc_repeat_cp_detail;
  const auto* error = metadata_error(input, hc_mult, cp_size, rank);
  TORCH_CHECK(error == nullptr, "SGLang mHC repeat+CP: ", error);
  const int64_t rows = selected_rows(input.size(0), cp_size, rank);
  TORCH_CHECK(out.defined() && out.layout() == c10::Layout::Strided &&
                  out.device() == input.device() &&
                  out.scalar_type() == input.scalar_type() && out.dim() == 3 &&
                  out.size(0) == rows && out.size(1) == hc_mult &&
                  out.size(2) == input.size(1) && out.is_contiguous() &&
                  !out.requires_grad() && !out.is_conj() && !out.is_neg(),
              "SGLang mHC repeat+CP: invalid output metadata");
  at::assert_no_overlap(input, out);
  if (out.numel() == 0) return out;
#if defined(USE_ROCM) && USE_ROCM
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const auto current_stream = at::cuda::getCurrentCUDAStream();
  const cudaStream_t stream = current_stream;
  // Keep buffers live if they were allocated on a different stream and the
  // caller releases its last tensor reference before this asynchronous copy.
  c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(),
                                                current_stream);
  c10::cuda::CUDACachingAllocator::recordStream(out.storage().data_ptr(),
                                                current_stream);
  const int64_t row_bytes = input.size(1) * input.element_size();
  const bool vectorized =
      input.stride(1) == 1 && row_bytes % kVectorBytes == 0 &&
      (input.stride(0) * input.element_size()) % kVectorBytes == 0 &&
      reinterpret_cast<uintptr_t>(input.data_ptr()) % kVectorBytes == 0 &&
      reinterpret_cast<uintptr_t>(out.data_ptr()) % kVectorBytes == 0;
  if (vectorized) {
    const int64_t vectors = row_bytes / kVectorBytes;
    const int64_t tiles = rows * ((vectors + kThreads - 1) / kThreads);
    const int blocks = std::min(tiles, kMaxBlocks);
    const auto* source = static_cast<const uint8_t*>(input.data_ptr());
    auto* destination = static_cast<uint8_t*>(out.data_ptr());
    const int64_t stride_bytes = input.stride(0) * input.element_size();
    if (hc_mult == 4) {
      repeat_vector_kernel<4><<<blocks, kThreads, 0, stream>>>(
          source, destination, vectors, stride_bytes, rows, hc_mult, cp_size,
          rank);
    } else {
      repeat_vector_kernel<0><<<blocks, kThreads, 0, stream>>>(
          source, destination, vectors, stride_bytes, rows, hc_mult, cp_size,
          rank);
    }
  } else {
    switch (input.element_size()) {
      case 1:
        launch_element<1>(input, out, rows, hc_mult, cp_size, rank, stream);
        break;
      case 2:
        launch_element<2>(input, out, rows, hc_mult, cp_size, rank, stream);
        break;
      case 4:
        launch_element<4>(input, out, rows, hc_mult, cp_size, rank, stream);
        break;
      case 8:
        launch_element<8>(input, out, rows, hc_mult, cp_size, rank, stream);
        break;
      case 16:
        launch_element<16>(input, out, rows, hc_mult, cp_size, rank, stream);
        break;
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "SGLang mHC repeat+CP requires a ROCm build");
#endif
  return out;
}

torch::Tensor mhc_repeat_cp_sglang(const torch::Tensor& input, int64_t hc_mult,
                                   int64_t cp_size, int64_t rank) {
  using namespace mhc_repeat_cp_detail;
  const auto* error = metadata_error(input, hc_mult, cp_size, rank);
  TORCH_CHECK(error == nullptr, "SGLang mHC repeat+CP: ", error);
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  auto out = torch::empty(
      {selected_rows(input.size(0), cp_size, rank), hc_mult, input.size(1)},
      input.options());
  return mhc_repeat_cp_sglang_out(input, hc_mult, cp_size, rank, out);
}

}  // namespace at::native
