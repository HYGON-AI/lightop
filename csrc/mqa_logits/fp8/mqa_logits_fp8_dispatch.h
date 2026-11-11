// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "dispatch/config.hpp"
#include <hip/hip_runtime.h>

namespace lightop::mqa_fp8 {
// Internal HIP core. The Torch adapter validates tensor memory and architecture.
// No public Python/C ABI is added by these declarations.
KernelConfig resolve(const MqaFp8Params &params);
hipError_t launch(const MqaFp8Params &params, const KernelConfig &config, hipStream_t stream,
                  uint32_t max_grid_x, uint32_t max_grid_y);
} // namespace lightop::mqa_fp8
