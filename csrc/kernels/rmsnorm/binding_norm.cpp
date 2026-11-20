// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Phase-0 pilot: split the "norm" family bindings into a dedicated extension
// module (lightop._ext.module_norm), mirroring AITER's "one operator backend = one
// module_xxx.so" layout.
//
// Ported from the 1113 tree to the 1118/lightop-public tree, where the kernel
// sources moved from csrc/rmsnorm/*.cu to csrc/kernels/rmsnorm/*.cu. The kernel
// definitions already live there under `namespace at::native`. This file only
// *declares* the pilot subset and exposes it on the new extension module, so
// that the full extension and module_norm never define the same Python module init symbol.
// It follows the style of csrc/pybind/export.cpp (declaration block +
// PYBIND11_MODULE + m.def).
//
// Build selection is controlled by the source list in module_registry.json.

#include <torch/extension.h>

namespace at::native {

// Declarations mirrored verbatim from csrc/pybind/export.cpp and
// csrc/kernels/rmsnorm/*.cu. Only ops defined in
// csrc/kernels/rmsnorm/fuse_rms_add_opt.cu and
// csrc/kernels/rmsnorm/fuse_rmsnorm_contiguous.cu are pulled in here so this
// extension stays self-contained (no cross-family symbols to resolve at link
// time).
void rms_norm_opt(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
                  double epsilon, double limit);
void fused_add_rms_norm_opt(torch::Tensor& input, torch::Tensor& residual,
                            torch::Tensor& weight, double epsilon, double limit,
                            bool use_aligned_loads = false);
void gemma_rmsnorm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
                   double epsilon);
void gemma_fused_add_rmsnorm(torch::Tensor& input, torch::Tensor& residual,
                             torch::Tensor& weight, double epsilon);
void rms_norm_no_weight(c10::optional<torch::Tensor> out, torch::Tensor& input,
                        c10::optional<torch::Tensor> weight, double epsilon);
void fused_rms_norm_contiguous(torch::Tensor& out, torch::Tensor& input,
                               torch::Tensor& weight, double epsilon);

}  // namespace at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "LightOp norm operators (Phase-0 module split pilot)";
  // Marker proving the aggregator (lightop/core.py) routed to this module.
  m.attr("LIGHTOP_NORM_MODULE") = 1;

  m.def("rms_norm_opt", &at::native::rms_norm_opt, "rms_norm_opt",
        py::arg("out"), py::arg("input"), py::arg("weight"),
        py::arg("epsilon"), py::arg("limit") = -1.0);
  m.def("fused_add_rms_norm_opt", &at::native::fused_add_rms_norm_opt,
        "fused_add_rms_norm_opt",
        py::arg("input"), py::arg("residual"), py::arg("weight"),
        py::arg("epsilon"), py::arg("limit") = -1.0,
        py::arg("use_aligned_loads") = false);
  m.def("gemma_rmsnorm", &at::native::gemma_rmsnorm,
        "gemma_rmsnorm(out, input, weight, epsilon)");
  m.def("gemma_fused_add_rmsnorm", &at::native::gemma_fused_add_rmsnorm,
        "gemma_fused_add_rmsnorm(input, residual, weight, epsilon)");
  m.def("rms_norm_no_weight", &at::native::rms_norm_no_weight,
        "rms_norm_no_weight");
  m.def("fused_rms_norm_contiguous", &at::native::fused_rms_norm_contiguous,
        "fused_rms_norm_contiguous");
}
