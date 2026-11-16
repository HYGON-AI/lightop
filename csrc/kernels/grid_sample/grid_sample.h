// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <array>
#include <cstdint>

namespace at {
class TensorBase;
}

namespace at::native {

void launch_grid_sampler_2d_forward_kernel(
    const TensorBase &output, const TensorBase &input, const TensorBase &grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners);

// void launch_grid_sampler_3d_forward_kernel(
//     const TensorBase &output, const TensorBase &input, const TensorBase &grid,
//     int64_t interpolation_mode, int64_t padding_mode, bool align_corners);

void launch_grid_sampler_2d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask);

// void launch_grid_sampler_3d_backward_kernel(
//     const TensorBase &grad_input, const TensorBase &grad_grid,
//     const TensorBase &grad_output, const TensorBase &input,
//     const TensorBase &grid, int64_t interpolation_mode, int64_t padding_mode,
//     bool align_corners, std::array<bool, 2> output_mask);

}  // namespace at::native