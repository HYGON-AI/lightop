// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "../mqa_logits_fp8_dispatch.h"
#include "auto.hpp"
#include "selectors.hpp"

namespace lightop::mqa_fp8 {
KernelConfig resolve(const MqaFp8Params &p) {
    KernelConfig config;
    mqa_auto_config(p, config);
    return config;
}

hipError_t launch(const MqaFp8Params &p, const KernelConfig &config, hipStream_t stream,
                  uint32_t max_grid_x, uint32_t max_grid_y) {
    LaunchGrid grid;
    if (!make_grid(p, config, max_grid_x, max_grid_y, grid)) {
        return hipErrorInvalidValue;
    }
    if (!grid.x || !grid.y)
        return hipSuccess;
    Kernel kernel = nullptr;
    switch (config.kind) {
    case KernelKind::M64:
        kernel = mqa_select_m64(p);
        break;
    case KernelKind::M16:
        kernel = mqa_select_m16(p, config.block_n);
        break;
    case KernelKind::QH:
        kernel = mqa_select_qh(p, config.block_n);
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
