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

void miopen_fused_adam(Tensor& params,
                       const Tensor& grads,
                       Tensor& exp_avgs,
                       Tensor& exp_avg_sqs,
                       Tensor& max_exp_avg_sqs,
                       Tensor& state_steps,
                       const float lr,
                       const float beta1,
                       const float beta2,
                       const float weight_decay,
                       const float eps,
                       const bool amsgrad,
                       const bool maximize,
                       const c10::optional<Tensor>& grad_scale,
                       const c10::optional<Tensor>& found_inf)
{
    if(params.dim() != 4)
    {
        MIOPEN_THROW("Fused_Adam: Only Support 4D.");
    }
    TensorDescriptor params_tensor(params);
    TensorDescriptor grads_tensor(grads);
    TensorDescriptor exp_avgs_tensor(exp_avgs);
    TensorDescriptor exp_avg_sqs_tensor(exp_avg_sqs);

    TensorDescriptor state_steps_tensor(state_steps);

    TensorDescriptor max_exp_avg_sqs_tensor = amsgrad ? TensorDescriptor(max_exp_avg_sqs) : TensorDescriptor();
    TensorDescriptor grad_scale_tensor =
        grad_scale.has_value() ? TensorDescriptor(grad_scale.value()) : TensorDescriptor();
    TensorDescriptor found_inf_tensor =
        found_inf.has_value() ? TensorDescriptor(found_inf.value()) : TensorDescriptor();

    const unsigned int state_step = 10;
    const bool adamw              = true;

    MIOPEN_CHECK(miopenFusedAdam(getMiopenHandle_new(),
                                 params_tensor.desc(),
                                 params.data_ptr(),
                                 grads_tensor.desc(),
                                 grads.data_ptr(),
                                 exp_avgs_tensor.desc(),
                                 exp_avgs.data_ptr(),
                                 exp_avg_sqs_tensor.desc(),
                                 exp_avg_sqs.data_ptr(),
                                 amsgrad ? max_exp_avg_sqs_tensor.desc() : nullptr,
                                 amsgrad ? max_exp_avg_sqs.data_ptr() : nullptr,
                                 nullptr,
                                 nullptr,
                                 state_step,
                                 lr,
                                 beta1,
                                 beta2,
                                 weight_decay,
                                 eps,
                                 amsgrad,
                                 maximize,
                                 adamw,
                                 grad_scale.has_value() ? grad_scale_tensor.desc() : nullptr,
                                 grad_scale.has_value() ? grad_scale.value().data_ptr() : nullptr,
                                 found_inf.has_value() ? found_inf_tensor.desc() : nullptr,
                                 found_inf.has_value() ? found_inf.value().data_ptr() : nullptr));
}

} // namespace native
} // namespace at