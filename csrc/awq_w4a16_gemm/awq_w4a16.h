// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <hip/hip_runtime.h>

namespace lightop::awq {
inline void check_tensor(const at::Tensor& value, at::ScalarType type,
                         at::Device device, int64_t size, const char* name) {
  TORCH_CHECK(value.is_cuda() && value.device() == device,
              name, " must be on the input GPU");
  TORCH_CHECK(value.scalar_type() == type && value.is_contiguous(),
              name, " has an invalid dtype or must be contiguous");
  TORCH_CHECK(size >= 0 && value.numel() >= size, name, " has insufficient storage");
}

inline void check_launch() {
  const hipError_t error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, "AWQ HIP launch failed: ", hipGetErrorString(error));
}

// Packed weights are N rows with stride_k logical U4 values per row.
// Each int32 holds logical positions [0,2,4,6,1,3,5,7] in low-to-high nibbles.
// Each scale/zero int32 contains FP16 {64 + zero, scale}.
void dequant_colmajor(at::Tensor& output, const at::Tensor& weight,
                      const at::Tensor& scale_zero, int64_t n, int64_t k,
                      int64_t stride_k, int64_t group_size);
}  // namespace lightop::awq

namespace at::native {
void gemm_awq_w4a16(Tensor& output, const Tensor& input, const Tensor& weight,
                    const Tensor& scale_zero, int num_n_warps, int num_k_warps,
                    int n_per_warp, int k_per_warp, int splitk, bool use_marlin);
void gemm_awq_w4a16_dequant(Tensor& output, const Tensor& weight,
                            const Tensor& scale_zero, bool use_marlin);
}  // namespace at::native
