#include "lmslimquant_native/bindings.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/cuda/CUDAGuard.h>

#include "lmslimquant_native/smoothquant/smquant_gemm.cuh"

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

}  // namespace lightop::lmslimquant_native
