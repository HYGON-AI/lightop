#include "lmslimquant_native/bindings.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/cuda/CUDAGuard.h>

#include "lmslimquant_native/smoothquant/smquant_gemm.cuh"

// Keep the vendor implementation's heavy HIP headers out of this C++ binding
// translation unit.  These declarations match lmslimquant_native/quant_kernels/torch_bridge/call_gemm.h.
torch::Tensor run_gemm_w4a16_v2(
    torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int, int, int,
    int, int, torch::Tensor, int, std::string);
void run_awq_input_padding(torch::Tensor, int, int, int, torch::Tensor, int);
void run_convert_s4_k_m8(torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, torch::Tensor, torch::Tensor,
                         int, int, int);
void run_permute_trans(torch::Tensor, torch::Tensor, int64_t, int64_t);
void run_awq_dequant_colmajor(torch::Tensor, torch::Tensor, torch::Tensor,
                              int64_t, int64_t, int64_t, int64_t, int64_t);
void run_dequant_w4_gemm_colmajor(torch::Tensor, torch::Tensor, torch::Tensor,
                                  int64_t, int64_t, int64_t);
torch::Tensor gptq_gemm(torch::Tensor, torch::Tensor, torch::Tensor,
                        torch::Tensor, torch::Tensor, bool, int64_t);
void gptq_shuffle(torch::Tensor, torch::Tensor, int64_t);
torch::Tensor rocblas_gemm_fp16(torch::Tensor, torch::Tensor, bool);

namespace lightop::lmslimquant_native {

// Source-preserving extraction of LMSlim src/lmslim/call_gemm.cpp's
// rocblas_gemm_int8 wrapper.  AWQ/GPTQ symbols from that translation unit are
// intentionally not linked because LightOp's public migrated APIs do not use
// them.
torch::Tensor rocblas_gemm_int8(
    torch::Tensor input,
    torch::Tensor weight,
    bool layout_tn) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .device(input.device());
  const int m = input.size(0);
  const int k = input.size(1);
  const int n = weight.size(1);
  at::Tensor output = torch::zeros({m, n}, options);
  const hipStream_t stream =
      at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
  gemm_rocblas_int8_gpu(
      stream,
      reinterpret_cast<rocblas_handle>(at::cuda::getCurrentCUDABlasHandle()),
      reinterpret_cast<int32_t*>(output.data_ptr()),
      reinterpret_cast<const int8_t*>(input.data_ptr()),
      reinterpret_cast<const int8_t*>(weight.data_ptr()),
      m,
      n,
      k,
      layout_tn);
  return output;
}

torch::Tensor rocblas_gemm_fp16(
    torch::Tensor input, torch::Tensor weight, bool layout_tn) {
  return ::rocblas_gemm_fp16(input, weight, layout_tn);
}

torch::Tensor gemm_w4a16_v2(
    torch::Tensor input, torch::Tensor weight, torch::Tensor zeros_and_scales,
    int m, int n, int k, int stride_a, int stride_b, int stride_b_padded,
    int stride_c, int group_size, torch::Tensor workspace,
    int workspace_size, std::string config) {
  return ::run_gemm_w4a16_v2(
      input, weight, zeros_and_scales, m, n, k, stride_a, stride_b,
      stride_b_padded, stride_c, group_size, workspace, workspace_size, config);
}

void awq_input_padding(torch::Tensor input, int m, int k, int stride_b_padded,
                       torch::Tensor workspace, int workspace_size) {
  ::run_awq_input_padding(input, m, k, stride_b_padded, workspace,
                          workspace_size);
}

void convert_s4_k_m8(torch::Tensor out_qweight, torch::Tensor out_scale_zero,
                     torch::Tensor workspace, torch::Tensor qweight,
                     torch::Tensor scales, torch::Tensor qzeros,
                     int m, int k, int group_size) {
  ::run_convert_s4_k_m8(out_qweight, out_scale_zero, workspace, qweight, scales,
                        qzeros, m, k, group_size);
}

void permute_trans(torch::Tensor output, torch::Tensor input,
                   int64_t dim0, int64_t n) {
  ::run_permute_trans(output, input, dim0, n);
}

void awq_dequant_colmajor(torch::Tensor output, torch::Tensor qweight,
                          torch::Tensor zeros_and_scales, int64_t k,
                          int64_t k_padded, int64_t n, int64_t group_size,
                          int64_t workspace_size) {
  ::run_awq_dequant_colmajor(output, qweight, zeros_and_scales, k, k_padded, n,
                             group_size, workspace_size);
}

void dequant_w4_gemm_colmajor(torch::Tensor output, torch::Tensor qweight,
                              torch::Tensor zeros_and_scales, int64_t k,
                              int64_t n, int64_t group_size) {
  ::run_dequant_w4_gemm_colmajor(output, qweight, zeros_and_scales, k, n,
                                 group_size);
}

torch::Tensor gptq_gemm_native(
    torch::Tensor input, torch::Tensor qweight, torch::Tensor qzeros,
    torch::Tensor scales, torch::Tensor g_idx, bool use_exllama, int64_t bit) {
  return ::gptq_gemm(input, qweight, qzeros, scales, g_idx, use_exllama, bit);
}

void gptq_shuffle_native(torch::Tensor qweight, torch::Tensor q_perm,
                         int64_t bit) {
  ::gptq_shuffle(qweight, q_perm, bit);
}

}  // namespace lightop::lmslimquant_native
