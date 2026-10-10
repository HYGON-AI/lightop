// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "../mqa_logits_api.h"
#include "dispatch/auto.hpp"
#include "../memory.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <limits>
#include <string>

namespace lightop::mqa_int8 {
namespace {
using lightop::mqa_beta::ByteRows;

void check_tensor(const torch::Tensor &t, const torch::Tensor &q, at::ScalarType dtype,
                  const char *name) {
    TORCH_CHECK(t.defined() && t.is_cuda() && t.device() == q.device(), "mqa_logits INT8: ", name,
                " must be on Q's CUDA/HIP device");
    TORCH_CHECK(t.scalar_type() == dtype, "mqa_logits INT8: ", name, " must have dtype ", dtype,
                ", got ", t.scalar_type());
}

ByteRows tensor_rows(const torch::Tensor &t, int64_t rows, int64_t stride, int64_t width,
                     const char *name) {
    const uint64_t bytes = t.element_size();
    const uint64_t limit = uint64_t(std::numeric_limits<int64_t>::max()) / bytes;
    TORCH_CHECK(rows > 0 && width > 0 && stride >= width && uint64_t(width) <= limit &&
                    uint64_t(rows - 1) <= (limit - width) / uint64_t(stride),
                "mqa_logits INT8: ", name, " address span exceeds signed 64-bit range");
    const uint64_t address = reinterpret_cast<uintptr_t>(t.data_ptr());
    TORCH_CHECK((address & 3) == 0, "mqa_logits INT8: ", name,
                " requires 4-byte pointer alignment");
    const ByteRows result{address, uint64_t(rows), uint64_t(rows == 1 ? width : stride) * bytes,
                          uint64_t(width) * bytes};
    TORCH_CHECK(lightop::mqa_beta::valid_rows(result), "mqa_logits INT8: ", name,
                " address span overflows");
    return result;
}

void check_disjoint(const ByteRows &out, const ByteRows &input, const char *name) {
    TORCH_CHECK(!lightop::mqa_beta::rows_overlap(out, input), "mqa_logits INT8: D_out overlaps ",
                name);
}
} // namespace

torch::Tensor run(const torch::Tensor &Q, const torch::Tensor &K, const torch::Tensor &weights,
                  const torch::Tensor &ks, const torch::Tensor &ke, int m, int n, int h, int d,
                  const torch::optional<torch::Tensor> &kv_scale, bool clean_logits,
                  const torch::optional<torch::Tensor> &out) {
    TORCH_CHECK(Q.defined() && Q.is_cuda(), "mqa_logits INT8 requires CUDA/HIP Q");
    TORCH_CHECK(Q.scalar_type() == at::kChar, "mqa_logits INT8 requires INT8 Q/K dtype");
    check_tensor(K, Q, at::kChar, "K");
    check_tensor(weights, Q, at::kFloat, "weights");
    check_tensor(ks, Q, at::kInt, "ks");
    check_tensor(ke, Q, at::kInt, "ke");
    TORCH_CHECK(m >= 0 && n >= 0 && h >= 1 && h <= 128 && !(h & (h - 1)) && d == 128,
                "mqa_logits INT8 requires M,N>=0, D=128 and power-of-two H in [1,128]");
    TORCH_CHECK(Q.dim() == 3 && Q.size(0) == m && Q.size(1) == h && Q.size(2) == d &&
                    K.dim() == 2 && K.size(0) == n && K.size(1) == d,
                "mqa_logits INT8: explicit M/N/H/D must match Q and K shapes");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == m && weights.size(1) == h,
                "mqa_logits INT8: weights must have shape [M,H]");
    TORCH_CHECK(ks.dim() == 1 && ke.dim() == 1 && ks.numel() >= m && ke.numel() >= m &&
                    ks.is_contiguous() && ke.is_contiguous(),
                "mqa_logits INT8: ks/ke must be contiguous int32 vectors with at least M elements");
    const bool scaled = kv_scale.has_value() && kv_scale->defined();
    TORCH_CHECK(scaled, "mqa_logits INT8 requires kv_scale");
    if (scaled) {
        check_tensor(*kv_scale, Q, at::kFloat, "kv_scale");
        TORCH_CHECK(kv_scale->is_contiguous() && kv_scale->numel() >= n,
                    "mqa_logits INT8: kv_scale must be contiguous with at least N elements");
    }
    const c10::cuda::CUDAGuard guard(Q.device());
    hipDeviceProp_t prop{};
    auto status = hipGetDeviceProperties(&prop, Q.get_device());
    TORCH_CHECK(status == hipSuccess, "mqa_logits: ", hipGetErrorString(status));
    TORCH_CHECK(std::string(prop.gcnArchName).substr(0, 6) == "gfx936" && prop.warpSize == 64,
                "mqa_logits beta INT8 requires gfx936/wave64; unset LIGHTOP_MQA_LOGITS_BETA "
                "or set it to 0 to use the original INT8 implementation");
    torch::Tensor result;
    if (out.has_value() && out->defined()) {
        check_tensor(*out, Q, at::kFloat, "D_out");
        TORCH_CHECK(out->dim() == 2 && out->size(0) >= m && out->size(1) >= n,
                    "mqa_logits INT8: D_out must cover [M,N]");
        result =
            (out->size(0) == m && out->size(1) == n) ? *out : out->slice(0, 0, m).slice(1, 0, n);
    } else {
        result = torch::empty({m, n}, Q.options().dtype(at::kFloat));
    }
    if (!m || !n)
        return result;
    TORCH_CHECK(Q.stride(2) == 1 && Q.stride(1) == 128 && Q.stride(0) == int64_t(h) * 128 &&
                    K.stride(1) == 1 && K.stride(0) == 128,
                "mqa_logits INT8: Q/K require contiguous D=128 layout; got strides ", Q.strides(),
                " and ", K.strides());
    TORCH_CHECK(weights.stride(1) == 1 && weights.stride(0) >= h && result.stride(1) == 1 &&
                    result.stride(0) >= n,
                "mqa_logits INT8: weights/D_out need contiguous columns and nonoverlapping rows; "
                "got strides ",
                weights.strides(), " and ", result.strides());
    const auto output_rows = tensor_rows(result, m, result.stride(0), n, "D_out");
    check_disjoint(output_rows, tensor_rows(Q, m, int64_t(h) * 128, int64_t(h) * 128, "Q"), "Q");
    check_disjoint(output_rows, tensor_rows(K, n, 128, 128, "K"), "K");
    check_disjoint(output_rows, tensor_rows(weights, m, weights.stride(0), h, "weights"),
                   "weights");
    check_disjoint(output_rows, tensor_rows(ks, 1, m, m, "ks"), "ks");
    check_disjoint(output_rows, tensor_rows(ke, 1, m, m, "ke"), "ke");
    if (scaled) {
        check_disjoint(output_rows, tensor_rows(*kv_scale, 1, n, n, "kv_scale"), "kv_scale");
    }
    // Match FP8's checked maximum weight-row span.
    const int64_t weight_rows = std::min(m, 64);
    TORCH_CHECK(weight_rows <= 1 ||
                    weights.stride(0) <= (int64_t(UINT32_MAX) / 4 - h) / (weight_rows - 1),
                "mqa_logits INT8: weights CTA span must fit the 32-bit buffer offset");
    MqaInt8Params p{};
    p.q = Q.data_ptr<int8_t>();
    p.kv = K.data_ptr<int8_t>();
    p.weights = weights.data_ptr<float>();
    p.kv_scale = scaled ? kv_scale->data_ptr<float>() : nullptr;
    p.out = result.data_ptr<float>();
    p.m = m;
    p.n = n;
    p.h = h;
    p.d = d;
    p.weights_stride_m = weights.stride(0);
    p.out_stride_m = result.stride(0);
    p.ks = ks.data_ptr<int32_t>();
    p.ke = ke.data_ptr<int32_t>();
    p.clean_logits = clean_logits;
    const auto stream = at::cuda::getCurrentCUDAStream(Q.get_device());
    status = launch(p, resolve(p), stream, prop.maxGridSize[0], prop.maxGridSize[1]);
    TORCH_CHECK(status == hipSuccess,
                "mqa_logits INT8 launch/configuration failed: ", hipGetErrorString(status));
    return result;
}
} // namespace lightop::mqa_int8
