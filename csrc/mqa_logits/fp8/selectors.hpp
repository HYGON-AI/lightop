// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include "../../params.h"

namespace lightop::mqa_fp8 {
using Kernel = void (*)(MqaFp8Params);
Kernel mqa_select_m64(const MqaFp8Params &params);
Kernel mqa_select_m16(const MqaFp8Params &params, int block_n);
Kernel mqa_select_m16_grouped(const MqaFp8Params &params);
Kernel mqa_select_qh(const MqaFp8Params &params, int block_n);
} // namespace lightop::mqa_fp8
