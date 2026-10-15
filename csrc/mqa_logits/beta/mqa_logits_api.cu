// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "mqa_logits_api.h"
#include "fp8/mqa_logits_fp8_interface.h"
#include <cstdlib>
#include <cstring>

namespace lightop::mqa_beta {
torch::optional<torch::Tensor>
try_dispatch(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &weights,
             const torch::Tensor &ks, const torch::Tensor &ke, int m, int n, int h, int d,
             const torch::optional<torch::Tensor> &kv_scale, bool clean_logits,
             const torch::optional<torch::Tensor> &out) {
    // Read on each host call; a captured graph retains its chosen kernels.
    const char *setting = std::getenv("LIGHTOP_MQA_LOGITS_BETA");
    if (setting == nullptr || std::strcmp(setting, "1") != 0)
        return torch::nullopt;
    if (q.scalar_type() == torch::kInt8)
        return mqa_int8::mqa_logits_int8(q, k, weights, ks, ke, m, n, h, d, kv_scale, clean_logits, out);
    if (q.scalar_type() == torch::kFloat8_e4m3fn)
        return at::native::mqa_logits_fp8(q, k, weights, ks, ke, m, n, h, d, kv_scale,
                                          clean_logits, out);
    return torch::nullopt;
}
} // namespace lightop::mqa_beta
