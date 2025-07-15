#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/cuda/CUDAMathCompat.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <ATen/native/cuda/Loops.cuh>
#include <torch/extension.h>

#include "reduce.h"


namespace at{
namespace native{

Tensor fused_conv_add_forward(const Tensor &x, 
                           const Tensor &add,
                           const Tensor &weight, 
                           const std::vector<int64_t>& pads,
                           const std::vector<int64_t>& strides,
                           const std::vector<int64_t>& dilations,
                           const std::string& layout, 
                           const int groups = 1, 
                           const float alpha = 1,
                           const float beta = 0, 
                           const double scale = 1, 
                           const double zero_point = 0)
{
    return ;
}

}
}