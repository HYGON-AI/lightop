// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#if defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#else
#include <cuda_fp16.h>
#include <cuda.h>
#endif

namespace at {
namespace native {
Tensor fuse_softmax_forward(const Tensor &input, const int64_t &dim, const bool &half_to_float);

Tensor fuse_softmax_backward(const Tensor& grad, const Tensor& output, 
                             const int64_t &dim, const Tensor& input);

Tensor fuse_log_softmax_forward(const at::Tensor &input, const int64_t &dim, const bool &half_to_float);

Tensor fuse_log_softmax_backward(const at::Tensor &grad_output, const at::Tensor &output, 
                                 const int64_t &dim, const at::Tensor &input);

} // namespace native
} // namespace at
