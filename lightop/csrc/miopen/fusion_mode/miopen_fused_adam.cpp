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

namespace at {
namespace native {

std::vector<Tensor> miopen_fused_adam(Tensor& params,
                                      Tensor& grads,
                                      Tensor& exp_avgs,
                                      Tensor& exp_avg_sqs,
                                      Tensor& max_exp_avg_sqs,
                                      Tensor& state_steps,
                                      const float& lr,
                                      const float& beta1,
                                      const float& beta2,
                                      const float& weight_decay,
                                      const float& eps,
                                      const bool& amsgrad,
                                      const bool& maximize,
                                      const bool& adamw,
                                      const Tensor& grad_scale,
                                      const Tensor& found_inf)
{
    TensorDescriptor params_tensor(params);
    TensorDescriptor grads_tensor(grads);
    TensorDescriptor exp_avgs_tensor(exp_avgs);
    TensorDescriptor exp_avg_sqs_tensor(exp_avg_sqs);
    TensorDescriptor max_exp_avg_sqs_tensor(max_exp_avg_sqs);
    TensorDescriptor state_steps_tensor(state_steps);
    TensorDescriptor grad_scale_tensor(grad_scale);
    TensorDescriptor found_inf_tensor(found_inf);

    MIOPEN_CHECK(miopenFusedAdam(getMiopenHandle_new(),
                                 params_tensor,
                                 params.data_ptr(),
                                 grads_tensor,
                                 grads.data_ptr(),
                                 exp_avgs_tensor,
                                 exp_avgs.data_ptr(),
                                 exp_avg_sqs_tensor,
                                 exp_avg_sqs.data_ptr(),
                                 max_exp_avg_sqs_tensor,
                                 max_exp_avg_sqs.data_ptr(),
                                 state_steps_tensor,
                                 state_steps.data_ptr(),
                                 10,
                                 lr,
                                 beta1,
                                 beta2,
                                 weight_decay,
                                 eps,
                                 amsgrad,
                                 maximize,
                                 adamw,
                                 grad_scale_tensor,
                                 grad_scale.data_ptr(),
                                 found_inf_tensor,
                                 found_inf.data_ptr()));

    return {params, params, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps};
}

} // namespace native
} // namespace at