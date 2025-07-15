#include "miopen/miopen-wrapper.h"
#include "miopen/op/op.h"
#include "miopen/graph/tensor.hpp"
#include "miopen/graph/execution_plan.hpp"
#include "miopen/graph/varpack.hpp"
#include "miopen/Handle.h"
#include <miopen/Exceptions.h>
#include <vector>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <tuple>


namespace at{
namespace native{

int fused_conv_add_forward(const Tensor &x, 
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
    return 0;
}

}
}