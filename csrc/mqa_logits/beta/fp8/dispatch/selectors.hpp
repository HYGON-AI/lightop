// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "config.hpp"

namespace lightop::mqa_fp8 {
using Kernel = void (*)(MqaFp8Params);
// Selectors only specialize the chosen KernelId on runtime flags; they return
// nullptr when the params violate the kernel's correctness requirements. KernelId
// selectors also reject invalid ids and ids from another family. verified_groups
// restricts automatic tuning only; forced calls use KernelTraits::required.
Kernel mqa_select_m64(const MqaFp8Params &params);
Kernel mqa_select_m16(const MqaFp8Params &params, KernelId id);
Kernel mqa_select_m16_grouped(const MqaFp8Params &params, KernelId id);
Kernel mqa_select_qh(const MqaFp8Params &params, int block_n);
} // namespace lightop::mqa_fp8
