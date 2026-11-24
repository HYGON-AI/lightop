#include <torch/extension.h>
#include <optional>

namespace at::native {
void swiglu_oai_build_gate_table(at::Tensor& table, double alpha, double limit);
void swiglu_oai_int8_quant_vllm_contiguous(
    const at::Tensor& input, at::Tensor& output, at::Tensor& scales,
    double alpha, double beta, double limit,
    const std::optional<at::Tensor>& gate_table, int64_t threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("build_gate_table", &at::native::swiglu_oai_build_gate_table);
    module.def("quantize", &at::native::swiglu_oai_int8_quant_vllm_contiguous);
}
