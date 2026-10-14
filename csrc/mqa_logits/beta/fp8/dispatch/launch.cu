// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "../mqa_logits_fp8_dispatch.h"
#include "auto.hpp"
#include "selectors.hpp"

namespace lightop::mqa_fp8 {
KernelId resolve_kernel(const MqaFp8Params &p) { return select_kernel(p); }

hipError_t launch(const MqaFp8Params &p, KernelId id, hipStream_t stream, uint32_t max_grid_x,
                  uint32_t max_grid_y) {
    LaunchGrid grid;
    if (!make_grid(p, id, max_grid_x, max_grid_y, grid)) {
        return hipErrorInvalidValue;
    }
    if (!grid.x || !grid.y)
        return hipSuccess;
    Kernel kernel = nullptr;
    switch (traits(id).family) {
    case Family::M64:
        kernel = mqa_select_m64(p);
        break;
    case Family::M16:
        kernel = mqa_select_m16(p, id);
        break;
    case Family::M16G:
        kernel = mqa_select_m16_grouped(p, id);
        break;
    case Family::QH:
        kernel = mqa_select_qh(p, traits(id).block_n);
        break;
    default:
        return hipErrorInvalidValue;
    }
    if (!kernel)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(kernel, dim3(grid.x, grid.y), dim3(256), 0, stream, p);
    return hipGetLastError();
}
} // namespace lightop::mqa_fp8
