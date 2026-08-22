#pragma once

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>
#include <torch/extension.h>

namespace lightop {
namespace deep_gemm {

torch::Tensor dequantize_index_k_int8_paged(
    const torch::Tensor& packed_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    const torch::Tensor& workspace,
    const torch::Tensor& page_claims,
    const int waves_per_page);

}  // namespace deep_gemm
}  // namespace lightop
