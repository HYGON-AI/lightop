#if defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#else
#include <cuda_fp16.h>
#include <cuda.h>
#endif

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/AccumulateType.h>

#include <THC/THCGeneral.h>
#include <torch/extension.h>

namespace at {
namespace native {

at::Tensor merge_tensor(const at::Tensor & input1, const at::Tensor & input2) {
  auto input1_size = input1.element_size()*input1.numel();
  auto input2_size = input2.element_size()*input2.numel();

  Tensor result = at::empty({input1_size+input2_size}, input1.options().dtype(torch::kUInt8));

  hipMemcpyDtoD(result.data_ptr(), input1.data_ptr(), input1_size);
  hipMemcpyDtoD(result.data_ptr()+input1_size, input2.data_ptr(), input2_size);
  return result;
}

void split_tensor(const at::Tensor & buff, std::vector<long> meta, at::Tensor & input1, at::Tensor & input2) {
  auto input1_size = meta[0];
  auto input2_size = meta[1];
  hipMemcpyDtoD(input1.data_ptr(), buff.data_ptr(), input1_size);
  hipMemcpyDtoD(input2.data_ptr(), buff.data_ptr()+input1_size, input2_size);
}
}
}