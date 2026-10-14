// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "dispatch/config.hpp"
#include <hip/hip_runtime.h>

namespace lightop::mqa_fp8 {
// Internal HIP core. The Torch adapter validates tensor memory and architecture.
// No public Python/C ABI is added by these declarations.
// resolve_kernel picks a KernelId from the rule table; launch runs exactly that variant.
// Internal ABI changed from resolve/KernelConfig: ctypes/dlsym clients must rebuild
// their bindings for resolve_kernel and a uint8_t KernelId. See DISPATCH_REFACTOR.md.
// launch rejects unsupported runtime requirements; verified tuning groups apply only to resolve.
KernelId resolve_kernel(const MqaFp8Params &params);
hipError_t launch(const MqaFp8Params &params, KernelId id, hipStream_t stream, uint32_t max_grid_x,
                  uint32_t max_grid_y);
} // namespace lightop::mqa_fp8
