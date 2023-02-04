#pragma once
#include "softmax.h"
#include <torch/autograd.h>
#include <torch/types.h>

namespace at {
namespace native {
//test
class fuseSoftmaxFunction : public torch::autograd::Function<fuseSoftmaxFunction> {
 public:
  static torch::autograd::variable_list forward(
  // static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input,
      int64_t dim,
      const bool half_to_float) {
    ctx->saved_data["dim"] = dim;
    // at::AutoDispatchBelowADInplaceOrView g;
    at::AutoNonVariableTypeMode g;
    
    if (input.device().type() == DeviceType::CUDA) {
      auto result = fuse_softmax_forward(input, dim, half_to_float);
      ctx->save_for_backward({input, result});
      return {result};
    } 
    else {
        TORCH_CHECK(0, "Unsupported device ", input.device().type());}
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    auto save_input = saved[0];
    auto save_result = saved[1];

    if (grad_output[0].device().type() == DeviceType::CUDA) {
      auto grad_in = fuse_softmax_backward(
          grad_output[0],
          save_result,
          ctx->saved_data["dim"].toInt(),
          save_input);
      return {
          grad_in,
          torch::autograd::Variable(),
          torch::autograd::Variable()};
    }
    else  {
        TORCH_CHECK(0, "Unsupported device ", grad_output[0].device().type());}
  }
};



class fuseLogSoftmaxFunction : public torch::autograd::Function<fuseLogSoftmaxFunction> {
 public:
  static torch::autograd::variable_list forward(
  // static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input,
      int64_t dim,
      const bool half_to_float) {
    ctx->saved_data["dim"] = dim;
    // at::AutoDispatchBelowADInplaceOrView g;
    at::AutoNonVariableTypeMode g;
    
    if (input.device().type() == DeviceType::CUDA) {
      auto result = fuse_log_softmax_forward(input, dim, half_to_float);
      ctx->save_for_backward({input, result});
      return {result};
    } 
    else {
        TORCH_CHECK(0, "Unsupported device ", input.device().type());}
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    auto save_input = saved[0];
    auto save_result = saved[1];
    if (grad_output[0].device().type() == DeviceType::CUDA) {
      auto grad_in = fuse_log_softmax_backward(
          grad_output[0],
          save_result,
          ctx->saved_data["dim"].toInt(),
          save_input);
      return {
          grad_in,
          torch::autograd::Variable(),
          torch::autograd::Variable()};
    }
    else  {
        TORCH_CHECK(0, "Unsupported device ", grad_output[0].device().type());}
  }
};

// TODO: There should be an easier way to do this
class fuseSoftmaxBackwardFunction
    : public torch::autograd::Function<fuseSoftmaxBackwardFunction> {
 public:
  static torch::autograd::variable_list forward(
  // static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& grad,
      const torch::autograd::Variable& output,
      int64_t dim,
      const torch::autograd::Variable& input) {
    // at::AutoDispatchBelowADInplaceOrView g;
    at::AutoNonVariableTypeMode g;

    if (input.device().type() == DeviceType::CUDA) {
      auto grad_in = fuse_softmax_backward(
          grad,
          output,
          dim,
          input);
      return {grad_in};   
    }
    else {
        TORCH_CHECK(0, "Unsupported device ", input.device().type());}
  }

  static torch::autograd::variable_list backward(
  //static tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    TORCH_CHECK(0, "double backwards on fuse_softmax not supported");
  }
};



Tensor fuse_softmax_forward_autograd(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float) {
    auto result = fuseSoftmaxFunction::apply(input, dim, half_to_float);

   //auto result = fuse_softmax_forward(input, dim, half_to_float);
   return result[0];
  }
  
Tensor fuse_logsoftmax_forward_autograd(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float) {
    auto result = fuseLogSoftmaxFunction::apply(input, dim, half_to_float);

   //auto result = fuse_softmax_forward(input, dim, half_to_float);
   return result[0];
  }  
  
Tensor fuse_softmax_backward_autograd(
    const at::Tensor& grad,
    const at::Tensor& output,
    int64_t dim,
    const at::Tensor& input) {
  return fuseSoftmaxBackwardFunction::apply(
      grad,
      output,
      dim,
      input)[0];
}

//
} // namespace native
} // namespace at
