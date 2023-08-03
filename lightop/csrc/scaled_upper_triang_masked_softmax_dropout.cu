#include <cuda_fp16.h>
#include <vector>
#include "scaled_upper_triang_masked_softmax_dropout.cuh"

using namespace std;

namespace at {
namespace native {

#define DISPATCH_HALF_AND_BFLOAT(TYPE, NAME, ...)			\
  switch(TYPE)								\
    {									\
    case at::ScalarType::Half:						\
      {									\
	using scalar_t = at::Half;					\
	__VA_ARGS__;							\
	break;								\
      }									\
    case at::ScalarType::BFloat16:					\
      {									\
	using scalar_t = at::BFloat16;					\
	__VA_ARGS__;							\
	break;								\
      }									\
    default:								\
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");	\
      }

Tensor fuse_scaled_upper_triang_masked_softmax_dropout_forward(
    Tensor const& input, float scale_factor, float p)
{
  // input is a 3d tensor with dimensions [attn_batches, seq_len, seq_len]
  const int attn_batches = input.size(0);
  const int seq_len = input.size(1);
  TORCH_INTERNAL_ASSERT(seq_len <= 2048);

  // Output 
  auto act_options = input.options().requires_grad(false);
  Tensor softmax_results = at::empty({attn_batches, seq_len, seq_len}, act_options);

  // Softmax Intermediate Result Ptr
  void* input_ptr = static_cast<void*>(input.data_ptr());
  void* softmax_results_ptr = static_cast<void*>(softmax_results.data_ptr());
  DISPATCH_HALF_AND_BFLOAT(
      input.scalar_type(),
      "dispatch_scaled_upper_triang_masked_softmax_dropout_forward",
      dispatch_scaled_upper_triang_masked_softmax_dropout_forward<scalar_t, scalar_t, float>(
	  reinterpret_cast<scalar_t*>(softmax_results_ptr),
	  reinterpret_cast<const scalar_t*>(input_ptr),
	  scale_factor,
	  seq_len,
	  seq_len,
	  attn_batches,
    p);
      );
  return softmax_results;
}

Tensor fuse_scaled_upper_triang_masked_softmax_dropout_backward(
    const Tensor& output_grads_, 
    const Tensor& softmax_results_, 
    float scale_factor)  {
	
  auto output_grads = output_grads_.contiguous();
  auto softmax_results = softmax_results_.contiguous();

  //output grads is a 3d tensor with dimensions [attn_batches, seq_len, seq_len]
  const int attn_batches = output_grads.size(0);
  const int seq_len = output_grads.size(1);
  TORCH_INTERNAL_ASSERT(output_grads.size(1) == output_grads.size(2));

  void* output_grads_ptr = static_cast<void*>(output_grads.data_ptr());

  //Softmax Grad
  DISPATCH_HALF_AND_BFLOAT(
      output_grads_.scalar_type(),
      "dispatch_scaled_upper_triang_masked_softmax_dropout_backward",
      dispatch_scaled_upper_triang_masked_softmax_dropout_backward<scalar_t, scalar_t, float>(
          reinterpret_cast<scalar_t*>(output_grads_ptr), 
	  reinterpret_cast<scalar_t*>(output_grads_ptr), 
	  reinterpret_cast<scalar_t const*>(softmax_results.data_ptr()),
	  scale_factor,
	  seq_len,
	  seq_len,
	  attn_batches);
      );
  
  //backward pass is completely in-place
  return output_grads;
}


class fuseUppertriangMaskedSoftmax : public torch::autograd::Function<fuseUppertriangMaskedSoftmax> {
 public:
  static torch::autograd::variable_list forward(
  // static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input, 
      float scale_factor,
      float p) {
    
    AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
	            (input.scalar_type() == at::ScalarType::BFloat16), 
                "Only fp16 and bf16 are supported");
    auto result = fuse_scaled_upper_triang_masked_softmax_dropout_forward(input, scale_factor, p);
    ctx->saved_data["scale"] = scale_factor;
    ctx->save_for_backward({result});
    return {result};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    AT_ASSERTM(grad_output[0].dim() == 3, "expected 3D tensor");
    AT_ASSERTM((grad_output[0].scalar_type() == at::ScalarType::Half) ||
	            (grad_output[0].scalar_type() == at::ScalarType::BFloat16), 
                "Only fp16 and bf16 are supported");

    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    float save_scale = (float)ctx->saved_data["scale"].toDouble();
    auto save_result = saved[0];
    AT_ASSERTM(save_result.dim() == 3, "expected 3D tensor");
    AT_ASSERTM((save_result.scalar_type() == at::ScalarType::Half) ||
	            (save_result.scalar_type() == at::ScalarType::BFloat16), 
                "Only fp16 and bf16 are supported");

    auto result = fuse_scaled_upper_triang_masked_softmax_dropout_backward(grad_output[0], save_result, save_scale);
    return {result, torch::autograd::Variable()};
  }
};
Tensor fuse_scaled_upper_triang_masked_softmax_dropout_forward_autograd(
    const Tensor& input, float scale_factor, float p) {
    auto result = fuseUppertriangMaskedSoftmax::apply(input, scale_factor, p);
    return result[0];
}
}}