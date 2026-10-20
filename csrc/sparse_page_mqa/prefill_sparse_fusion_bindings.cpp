// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include "prefill_sparse_fusion.h"
#include "prefill_sparse_sort.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.attr("PREFILL_SPARSE_FUSION_ABI")=1;
    m.attr("PREFILL_SPARSE_FUSION_BUILD_ID")=PREFILL_FUSION_BUILD_ID;
    m.def("prefill_sparse_native_prepare_query",&prefill_sparse_native_prepare_query);
    m.def("prefill_sparse_native_prepare_candidates",&prefill_sparse_native_prepare_candidates);
    m.def("prefill_sparse_native_finalize",&prefill_sparse_native_finalize);
}
