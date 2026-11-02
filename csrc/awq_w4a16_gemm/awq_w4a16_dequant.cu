// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "awq_w4a16.h"
#include "awq_w4a16_common.h"
#include <c10/cuda/CUDAGuard.h>

namespace {
// Scalar output indexing also covers tails and padded column-major storage.
template <bool Marlin>
__global__ void dequant_awq_kernel(half_t* output, const uint32_t* weight,
                                 const half2_t* scale_zero, int64_t n,
                                 int64_t k, int64_t stride_k, int64_t group,
                                 int64_t tile_k) {
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < n * k; index += int64_t(blockDim.x) * gridDim.x) {
    const int64_t row = index / k;
    const int64_t col = index % k;
    const int64_t packed_index = Marlin
        ? (col / tile_k) * n * (tile_k / 8) + row * (tile_k / 8) + (col % tile_k) / 8
        : row * (stride_k / 8) + col / 8;
    const int lane = col % 8;
    const int nibble = lane / 2 + (lane % 2) * 4;
    const float q = float((weight[packed_index] >> (4 * nibble)) & 15);
    const half2_t sz = scale_zero[row * (stride_k / group) + col / group];
    output[index] = half_t((q + 64.0f - float(sz[0])) * float(sz[1]));
  }
}
}  // namespace

namespace lightop::awq {
void dequant_colmajor(at::Tensor& output, const at::Tensor& weight,
                      const at::Tensor& scale_zero, int64_t n, int64_t k,
                      int64_t stride_k, int64_t group_size) {
  TORCH_CHECK(n >= 0 && k >= 0 && stride_k >= k && group_size > 0,
              "invalid AWQ dimensions, stride, or group_size");
  TORCH_CHECK(group_size % 8 == 0 && k % group_size == 0 &&
              stride_k % group_size == 0, "AWQ K and stride must contain complete groups of 8 or more");
  TORCH_CHECK(n <= INT32_MAX && stride_k <= INT32_MAX, "AWQ dimensions exceed int32 range");
  const auto device = weight.device();
  check_tensor(weight, at::kInt, device, n * (stride_k / 8), "weight");
  check_tensor(scale_zero, at::kInt, device, n * (stride_k / group_size), "scale_zero");
  check_tensor(output, at::kHalf, device, n * k, "output");
  at::assert_no_overlap(output, weight);
  at::assert_no_overlap(output, scale_zero);
  const at::cuda::OptionalCUDAGuard guard(at::device_of(weight));
  if (n == 0 || k == 0) return;
  const int blocks = int(std::min<int64_t>((n * k + 255) / 256, 65535));
  dequant_awq_kernel<false><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half_t*>(output.data_ptr()),
      reinterpret_cast<const uint32_t*>(weight.data_ptr()),
      reinterpret_cast<const half2_t*>(scale_zero.data_ptr()), n, k, stride_k, group_size, 0);
  check_launch();
}
}  // namespace lightop::awq

namespace at::native {
void gemm_awq_w4a16_dequant(Tensor& output, const Tensor& weight,
                            const Tensor& scale_zero, bool use_marlin) {
  TORCH_CHECK(output.dim() == 2 && scale_zero.dim() == 2, "AWQ output and scale_zero must be matrices");
  const int64_t n = output.size(0), k = output.size(1);
  TORCH_CHECK(scale_zero.size(0) == n, "AWQ scale_zero rows must match output");
  const int64_t groups = scale_zero.size(1);
  TORCH_CHECK((k == 0 && groups == 0) || (groups > 0 && k % groups == 0), "invalid AWQ group count");
  const int64_t group = k == 0 ? 8 : k / groups;
  if (!use_marlin) {
    lightop::awq::dequant_colmajor(output, weight, scale_zero, n, k, k, group);
    return;
  }
  TORCH_CHECK(k % 128 == 0 && group % 8 == 0, "Marlin K must be divisible by 128 and group by 8");
  const int64_t tile = k % 256 == 0 ? 256 : 128;
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == k / tile &&
              weight.size(1) == n * (tile / 8), "invalid Marlin weight shape");
  lightop::awq::check_tensor(weight, at::kInt, output.device(), n * (k / 8), "weight");
  lightop::awq::check_tensor(scale_zero, at::kInt, output.device(), n * groups, "scale_zero");
  lightop::awq::check_tensor(output, at::kHalf, output.device(), n * k, "output");
  at::assert_no_overlap(output, weight);
  at::assert_no_overlap(output, scale_zero);
  const at::cuda::OptionalCUDAGuard guard(at::device_of(output));
  if (n == 0 || k == 0) return;
  const int blocks = int(std::min<int64_t>((n * k + 255) / 256, 65535));
  dequant_awq_kernel<true><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half_t*>(output.data_ptr()),
      reinterpret_cast<const uint32_t*>(weight.data_ptr()),
      reinterpret_cast<const half2_t*>(scale_zero.data_ptr()), n, k, k, group, tile);
  lightop::awq::check_launch();
}
}  // namespace at::native
