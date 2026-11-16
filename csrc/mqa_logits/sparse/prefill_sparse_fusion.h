// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <vector>

std::vector<at::Tensor> prefill_sparse_native_prepare_query(
    const at::Tensor& q, const at::Tensor& raw, const at::Tensor& frequencies,
    const at::Tensor& positions, int64_t rope, double weight_scale,
    int64_t ratio, int64_t width, bool fast_pow2,
    c10::optional<at::Tensor> source, c10::optional<at::Tensor> request_map,
    c10::optional<at::Tensor> request_id, int64_t context, int64_t page,
    c10::optional<at::Tensor> flags, int64_t lanes);

std::vector<at::Tensor> prefill_sparse_native_prepare_candidates(
    const at::Tensor& source, int64_t width, const at::Tensor& request_map,
    const at::Tensor& request_id, int64_t context, int64_t ratio, int64_t page,
    c10::optional<at::Tensor> flags);

void prefill_sparse_native_finalize(
    const at::Tensor& compact, const at::Tensor& blocks,
    const at::Tensor& lengths, const at::Tensor& valid, at::Tensor logical,
    c10::optional<at::Tensor> physical, c10::optional<at::Tensor> request_map,
    c10::optional<at::Tensor> request_id, int64_t ratio,
    c10::optional<at::Tensor> flags);
