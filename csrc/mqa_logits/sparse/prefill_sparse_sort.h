// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <ATen/ATen.h>

void prefill_sparse_sort_candidates(
    const at::Tensor& source, at::Tensor output, at::Tensor flags,
    int64_t capacity, int64_t algorithm, int64_t threads,
    c10::optional<at::Tensor> pending);

void prefill_sparse_finalize_indices(
    const at::Tensor& compact, const at::Tensor& blocks,
    const at::Tensor& lengths, const at::Tensor& valid,
    at::Tensor logical, c10::optional<at::Tensor> physical,
    c10::optional<at::Tensor> request_map, c10::optional<at::Tensor> request_id,
    int64_t ratio, int64_t capacity, int64_t algorithm, int64_t threads,
    c10::optional<at::Tensor> candidate_flags);
