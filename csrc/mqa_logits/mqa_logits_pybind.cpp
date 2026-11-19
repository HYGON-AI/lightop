// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

namespace at::native {
torch::Tensor mqa_logits(
    torch::Tensor& Q,
    torch::Tensor& K,
    torch::Tensor& Weights,
    torch::Tensor& KS,
    torch::Tensor& KE,
    int q_seq_len,
    int kv_seq_len,
    int num_heads,
    int head_dim,
    const torch::optional<torch::Tensor>& KV_scale = torch::nullopt,
    bool clean_logits = true,
    const torch::optional<torch::Tensor>& D_out = torch::nullopt);
}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mqa_logits", &at::native::mqa_logits, "mqa_logits",
        py::arg("Q"),
        py::arg("K"),
        py::arg("Weights"),
        py::arg("KS"),
        py::arg("KE"),
        py::arg("seq_len"),
        py::arg("kv_seq_len"),
        py::arg("num_heads"),
        py::arg("head_dim"),
        py::arg("KV_scale") = torch::nullopt,
        py::arg("clean_logits") = true,
        py::arg("D_out") = torch::nullopt);
}
