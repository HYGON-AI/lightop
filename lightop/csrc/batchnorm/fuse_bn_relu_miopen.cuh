#include <cstdlib>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/AccumulateType.h>
#include <ATen/Parallel.h>
#include <ATen/Config.h>

#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/batch_norm.h>
#include <ATen/native/Normalization.h>

#include <vector>
#include <miopen/miopen.h>

#include "miopen/Descriptors.h"
#include "miopen/Types.h"
#include "miopen/Utils.h"
#include "miopen/manage_ptr.hpp"
static const int MIOPEN_DIM_MAX = 5;
namespace at { namespace native {

namespace {

Tensor expandScale(const Tensor& t, int64_t dim) {
  std::vector<int64_t> size{ 1, t.numel() };
  while (static_cast<int64_t>(size.size()) < dim) {
    size.emplace_back(1);
  }
  return t.view(size);
}

void check_dims_match_num_input_features(const char* arg_name, int64_t expected, int64_t actual){
    TORCH_CHECK(actual == expected,
             arg_name, " should contain ", expected, " elements not ", actual);
}
static inline Tensor repeat_if_defined(const Tensor& t, int64_t repeat) {
    if (t.defined()) {
      return t.repeat(repeat);
    }
    return t;
}
}// namespace

// forward code
using ptr_ActivationDesc = MIOPEN_MANAGE_PTR(miopenActivationDescriptor_t,
                                             miopenDestroyActivationDescriptor);
using ptr_FusionPlanDesc = MIOPEN_MANAGE_PTR(miopenFusionPlanDescriptor_t, miopenDestroyFusionPlan);
using ptr_FusionPlanArgs = MIOPEN_MANAGE_PTR(miopenOperatorArgs_t, miopenDestroyOperatorArgs);

ptr_FusionPlanDesc GetManagedFusionPlanDesc(miopenTensorDescriptor_t inputDesc)
{
    miopenFusionPlanDescriptor_t fusePlanDesc;
    miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputDesc);
    return ptr_FusionPlanDesc{fusePlanDesc};
}

ptr_FusionPlanArgs GetManageFusionPlanArgs()
{
    miopenOperatorArgs_t fusionArgs;
    miopenCreateOperatorArgs(&fusionArgs);
    return ptr_FusionPlanArgs{fusionArgs};
}
ptr_ActivationDesc GetManagedActivDesc()
{
    miopenActivationDescriptor_t activdesc;
    miopenCreateActivationDescriptor(&activdesc);
    return ptr_ActivationDesc{activdesc};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> miopen_bnrelu_forward(
    const Tensor& input_t, const Tensor& weight_t, const c10::optional<Tensor>& bias_t_opt, const c10::optional<Tensor>& running_mean_t_opt, const c10::optional<Tensor>& running_var_t_opt,
    bool training, double exponential_average_factor, double epsilon, int act_mode, double act_alpha, double act_beta, double act_gamma)
{
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;
  const Tensor& running_mean_t = c10::value_or_else(running_mean_t_opt, [] {return Tensor();});
  const Tensor& running_var_t = c10::value_or_else(running_var_t_opt, [] {return Tensor();});

  TensorArg input{ input_t, "input", 1 },
            weight{ weight_t, "weight", 2 },
            bias{ bias_t, "bias", 3 },
            running_mean{ running_mean_t, "running_mean", 4 },
            running_var{ running_var_t, "running_var", 5 };
  CheckedFrom c = "miopen_batch_norm";

  checkAllDefined(c, {input, weight, bias});
  if (!training) {
    checkAllDefined(c, {running_mean, running_var});
  }
  checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});
  if (input->scalar_type() != ScalarType::Half) {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {weight, bias, running_mean, running_var});
  checkAllContiguous(c, {input, weight, bias, running_mean, running_var});
  checkDimRange(c, input, 2, 6 /* exclusive */);
  auto num_features = input->size(1);
  for (auto t : {weight, bias, running_mean, running_var}) {
    if (t->defined()) {
      checkNumel(c, t, num_features);
    }
  }
  miopenBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = miopenBNPerActivation;
  } else {
    mode = miopenBNSpatial;
  }

  auto output_t = at::empty(input->sizes(), input->options());
  TensorArg output{ output_t, "output", 0 };

  auto handle = getMiopenHandle_new();
  //auto handle = mio::handle();
  auto dataType = getMiopenDataType(*input);
  TensorDescriptor idesc{ *input, 4 };  // input descriptor
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, running_mean, etc.

  Tensor save_mean, save_var;
  miopenActivationMode_t activ_mode = miopenActivationMode_t(act_mode);
  miopenFusionOpDescriptor_t bNormFwdOp = nullptr;
  miopenFusionOpDescriptor_t activFwdOp = nullptr;
  auto ptr_fwdfusionplan = GetManagedFusionPlanDesc(idesc.desc());
  ptr_ActivationDesc ptr_activdesc = nullptr;
  ptr_activdesc = GetManagedActivDesc();
  miopenSetActivationDescriptor(ptr_activdesc.get(), activ_mode, act_alpha, act_beta, act_gamma);

  if (training) {
    int64_t num_features = input_t.size(1);
    save_mean = at::empty({ num_features }, weight_t.options());
    save_var = at::empty({ num_features }, weight_t.options());

    miopenCreateOpBatchNormForward(ptr_fwdfusionplan.get(), &bNormFwdOp, mode, true);
    miopenCreateOpActivationForward(ptr_fwdfusionplan.get(), &activFwdOp, activ_mode);
    auto status = miopenCompileFusionPlan(handle, ptr_fwdfusionplan.get());
    if (status != miopenStatusSuccess) {
      printf("Error: miopenCompileFusionPlan error");
      //return None, None, None;
    }

    float alpha = static_cast<float>(1), beta = static_cast<float>(0);
    //miopenOperatorArgs_t ptr_fusionargs;
    //miopenCreateOperatorArgs(&ptr_fusionargs);
    auto ptr_fusionargs = GetManageFusionPlanArgs();

    miopenSetOpArgsBatchNormForward(ptr_fusionargs.get(), 
                                    bNormFwdOp, 
                                    &alpha, 
                                    &beta, 
                                    weight->data_ptr(), 
                                    bias->data_ptr(), 
                                    save_mean.data_ptr(),
                                    save_var.data_ptr(), 
                                    at::maybe_data_ptr(running_mean), 
                                    at::maybe_data_ptr(running_var), 
                                    exponential_average_factor, 
                                    epsilon);
    miopenSetOpArgsActivForward(ptr_fusionargs.get(), 
                                activFwdOp, 
                                &alpha, 
                                &beta, 
                                act_alpha, 
                                act_beta, 
                                act_gamma);

    miopenExecuteFusionPlan(handle, 
                            ptr_fwdfusionplan.get(), 
                            idesc.desc(), input->data_ptr(),
                            idesc.desc(), output->data_ptr(), 
                            ptr_fusionargs.get());
  } else {
    
    save_mean = at::empty({ num_features }, weight_t.options());
    save_var = at::empty({ num_features }, weight_t.options());
    //auto ptr_fusionplan = GetManagedFusionPlanDesc(&idesc.desc());
    miopenCreateOpBatchNormInference(ptr_fwdfusionplan.get(), &bNormFwdOp, mode, wdesc.desc());
    miopenCreateOpActivationForward(ptr_fwdfusionplan.get(), &activFwdOp, activ_mode);
    miopenStatus_t miopenError = miopenCompileFusionPlan(handle, ptr_fwdfusionplan.get());
    if (miopenError != miopenStatusSuccess) {
      printf("Error: miopenCompileFusionPlan error");
      //return None, None, None;
    }

    double alpha_bn = 1., beta_bn = 0.;
    auto ptr_fusionargs = GetManageFusionPlanArgs();
    miopenSetOpArgsBatchNormInference(ptr_fusionargs.get(),
                                      bNormFwdOp,
                                      &alpha_bn,
                                      &beta_bn,
                                      weight->data_ptr(), 
                                      bias->data_ptr(), 
                                      at::maybe_data_ptr(running_mean), 
                                      at::maybe_data_ptr(running_var), 
                                      epsilon);
    miopenSetOpArgsActivForward(
            ptr_fusionargs.get(), activFwdOp, &alpha_bn, &beta_bn, act_alpha, act_beta, act_gamma);
    miopenExecuteFusionPlan(handle,
                            ptr_fwdfusionplan.get(),
                            idesc.desc(),input->data_ptr(),
                            idesc.desc(),output->data_ptr(),
                            ptr_fusionargs.get());
  }

  return std::tuple<Tensor, Tensor, Tensor, Tensor>{output_t, output_t, save_mean, save_var};
}

// backward code
std::tuple<Tensor, Tensor, Tensor> miopen_bnrelu_backward(
    const Tensor& input_t,
    const Tensor& output_relu_t,
    const Tensor& grad_output_t,
    const Tensor& weight_t,
    const Tensor& bias_t,
    const optional<Tensor>& running_mean_opt,
    const optional<Tensor>& running_var_opt,
    const optional<Tensor>& save_mean_t_opt,
    const optional<Tensor>& save_var_t_opt,
    double epsilon,
    int act_mode, 
    double act_alpha, 
    double act_beta, 
    double act_gamma) {

  const Tensor& running_mean =
      c10::value_or_else(running_mean_opt, [] { return Tensor(); });
  const Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return Tensor(); });
  const Tensor& save_mean_t =
      c10::value_or_else(save_mean_t_opt, [] { return Tensor(); });
  const Tensor& save_var_t =
      c10::value_or_else(save_var_t_opt, [] { return Tensor(); });
  
  TensorArg input{ input_t, "input", 1 },
            output_relu{output_relu_t, "output_relu", 2},
            grad_output{ grad_output_t, "grad_output", 3 },
            weight{ weight_t, "weight", 4 },
            bias{ bias_t, "bias", 5 },
            save_mean{ save_mean_t, "save_mean", 6 },
            save_var{ save_var_t, "save_var", 7 };
  CheckedFrom c = "miopen_batch_norm_backward";

  checkAllDefined(c, {input, grad_output, weight, save_mean, save_var});
  checkAllSameGPU(c, {input, grad_output, weight, save_mean, save_var});
  if (input->scalar_type() == ScalarType::Half) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {input, grad_output});
  checkAllSameType(c, {weight, save_mean, save_var});
  checkAllContiguous(c, {input, grad_output, save_mean, save_var});
  checkDimRange(c, input, 2, 6 /* exclusive */);
  checkSameSize(c, input, grad_output);
  auto num_features = input->size(1);
  for (auto t : {weight, save_mean, save_var}) {
    checkNumel(c, t, num_features);
  }
  
  miopenBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = miopenBNPerActivation;
  } else {
    mode = miopenBNSpatial;
  }

  auto grad_input_t  = at::empty(input->sizes(), input->options());
  auto grad_weight_t = at::empty(weight->sizes(), weight->options());
  auto grad_bias_t   = at::empty(weight->sizes(), weight->options());
  
  auto handle = getMiopenHandle_new();
  auto dataType = getMiopenDataType(*input);

  TensorDescriptor idesc{ *input, 4 };  // input, output, grad_output descriptor
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, save_mean, etc.

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  miopenActivationMode_t activ_mode = miopenActivationMode_t(act_mode);
  miopenFusionOpDescriptor_t bNormBwdOp = nullptr;
  miopenFusionOpDescriptor_t activBwdOp = nullptr;
  auto ptr_bwdfusionplan = GetManagedFusionPlanDesc(idesc.desc());
  //miopenCreateFusionPlan(&ptr_bwdfusionplan, miopenVerticalFusion, idesc.desc());

  miopenCreateOpBatchNormBackward(ptr_bwdfusionplan.get(), &bNormBwdOp, mode);
  miopenCreateOpActivationBackward(ptr_bwdfusionplan.get(), &activBwdOp, activ_mode);
  miopenStatus_t miopenBwdError =
                miopenCompileFusionPlan(handle, ptr_bwdfusionplan.get());
  if(miopenBwdError != miopenStatusSuccess) {
    std::cerr << "Error: BatchNorm+Activation Spatial Backward Training plan not supported."
              << std::endl;
  }
  
  // double activ_alpha, activ_beta, activ_gamma;
  // ptr_ActivationDesc ptr_activdesc = nullptr;
  // ptr_activdesc = GetManagedActivDesc();
  // miopenActivationMode_t activ_mode;
  // miopenGetActivationDescriptor(
  //           ptr_activdesc.get(), &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
  // printf("===alpha:%lf beta:%lf gamma:%lf\n", activ_alpha, activ_beta, activ_gamma);
  auto ptr_fusionargs = GetManageFusionPlanArgs();      
  //miopenOperatorArgs_t ptr_fusionargs; 
  //miopenCreateOperatorArgs(&ptr_fusionargs);

  miopenSetOpArgsBatchNormBackward(ptr_fusionargs.get(),
                                   bNormBwdOp,
                                   &one,
                                   &zero,
                                   input->data_ptr(),
                                   weight->data_ptr(), 
                                   bias->data_ptr(), 
                                   grad_weight_t.data_ptr(),
                                   grad_bias_t.data_ptr(),
                                   save_mean->data_ptr(),
                                   save_var->data_ptr());

  miopenSetOpArgsActivBackward(ptr_fusionargs.get(),
                               activBwdOp,
                               &one,
                               &zero,
                               output_relu_t.data_ptr(),
                               nullptr,
                               act_alpha,
                               act_beta,
                               act_gamma);
  miopenExecuteFusionPlan(handle,
                          ptr_bwdfusionplan.get(),
                          idesc.desc(), grad_output->data_ptr(),
                          idesc.desc(), grad_input_t.data_ptr(), 
                          ptr_fusionargs.get());
  return std::tuple<Tensor,Tensor,Tensor>{grad_input_t, grad_weight_t, grad_bias_t};
}

}//native
}//at
