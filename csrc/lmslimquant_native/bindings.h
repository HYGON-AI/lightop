#pragma once

#include <torch/extension.h>

namespace lightop::lmslimquant_native {

torch::Tensor rocblas_gemm_int8(
    torch::Tensor input,
    torch::Tensor weight,
    bool layout_tn);

}  // namespace lightop::lmslimquant_native
