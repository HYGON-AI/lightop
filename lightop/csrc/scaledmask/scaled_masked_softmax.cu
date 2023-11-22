#include <cuda_fp16.h>
#include <vector>
#include "scaled_masked_softmax.cuh"

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

Tensor fuse_scaled_masked_softmax_forward(
    Tensor const& input,  Tensor const& mask, float scale_factor)
{
  // input is a 4d tensor with dimensions [batches, attn_heads, seq_len, seq_len]
  const int batches = input.size(0);
  const int pad_batches = mask.size(0);
  const int attn_heads = input.size(1);
  const int query_seq_len = input.size(2);
  const int key_seq_len = input.size(3);
  TORCH_INTERNAL_ASSERT(key_seq_len <= 4096);
  TORCH_INTERNAL_ASSERT(query_seq_len > 1);
  TORCH_INTERNAL_ASSERT(pad_batches == 1 || pad_batches == batches);
  TORCH_INTERNAL_ASSERT(mask.size(1) == 1);
  TORCH_INTERNAL_ASSERT(mask.size(2) == query_seq_len);
  TORCH_INTERNAL_ASSERT(mask.size(3) == key_seq_len);

  // Output 
  auto act_options = input.options().requires_grad(false);
  Tensor softmax_results = at::empty({batches, attn_heads, query_seq_len, key_seq_len}, act_options);

  // Softmax Intermediate Result Ptr
  void* input_ptr = static_cast<void*>(input.data_ptr());
  void* mask_ptr = static_cast<void*>(mask.data_ptr());
  void* softmax_results_ptr = static_cast<void*>(softmax_results.data_ptr());

  DISPATCH_HALF_AND_BFLOAT(
      input.scalar_type(),
      "dispatch_scaled_masked_softmax_forward",
      dispatch_scaled_masked_softmax_forward<scalar_t, scalar_t, float>(
      reinterpret_cast<scalar_t*>(softmax_results_ptr),
	  reinterpret_cast<const scalar_t*>(input_ptr),
	  reinterpret_cast<const uint8_t*>(mask_ptr),
	  scale_factor,
	  query_seq_len,
	  key_seq_len,
	  batches,
	  attn_heads,
	  pad_batches);
      );
  return softmax_results;
}

Tensor fuse_scaled_masked_softmax_backward(
    const Tensor& output_grads_, 
    const Tensor& softmax_results_, 
    float scale_factor)  {
	
  auto output_grads = output_grads_.contiguous();
  auto softmax_results = softmax_results_.contiguous();

  //output grads is a 4d tensor with dimensions [batches, attn_heads, seq_len, seq_len]
  const int batches = output_grads.size(0);
  const int attn_heads = output_grads.size(1);
  const int query_seq_len = output_grads.size(2);
  const int key_seq_len = output_grads.size(3);

  auto act_options = output_grads.options().requires_grad(false);
  Tensor input_grads = at::empty({batches, attn_heads, query_seq_len, key_seq_len}, act_options);

  void* output_grads_ptr = static_cast<void*>(output_grads.data_ptr());
  void* input_grads_ptr = static_cast<void*>(input_grads.data_ptr());

  //Softmax Grad
  DISPATCH_HALF_AND_BFLOAT(
      output_grads_.scalar_type(),
      "dispatch_scaled_masked_softmax_backward",
      dispatch_scaled_masked_softmax_backward<scalar_t, scalar_t, float>(
      reinterpret_cast<scalar_t*>(input_grads_ptr), 
	  reinterpret_cast<scalar_t*>(output_grads_ptr), 
	  reinterpret_cast<scalar_t const*>(softmax_results.data_ptr()),
	  scale_factor,
	  query_seq_len,
	  key_seq_len,
	  batches,
	  attn_heads);
      );
  
  //backward pass is completely in-place
  return input_grads;
}


class fuseMaskedSoftmax : public torch::autograd::Function<fuseMaskedSoftmax> {
 public:
  static torch::autograd::variable_list forward(
  // static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input,
      const torch::autograd::Variable& mask, 
      float scale_factor) {
    
    AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
	            (input.scalar_type() == at::ScalarType::BFloat16), 
                "Only fp16 and bf16 are supported");
    auto result = fuse_scaled_masked_softmax_forward(input, mask, scale_factor);
    ctx->saved_data["scale"] = scale_factor;
    ctx->save_for_backward({result});
    return {result};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    AT_ASSERTM(grad_output[0].dim() == 4, "expected 4D tensor");
    AT_ASSERTM((grad_output[0].scalar_type() == at::ScalarType::Half) ||
	            (grad_output[0].scalar_type() == at::ScalarType::BFloat16), 
                "Only fp16 and bf16 are supported");

    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    float save_scale = (float)ctx->saved_data["scale"].toDouble();
    auto save_result = saved[0];
    AT_ASSERTM(save_result.dim() == 4, "expected 4D tensor");
    AT_ASSERTM((save_result.scalar_type() == at::ScalarType::Half) ||
	            (save_result.scalar_type() == at::ScalarType::BFloat16), 
                "Only fp16 and bf16 are supported");

    auto result = fuse_scaled_masked_softmax_backward(grad_output[0], save_result, save_scale);
    return {result, torch::autograd::Variable(), torch::autograd::Variable()};
  }
};
Tensor fuse_scaled_masked_softmax_forward_autograd(
    const Tensor& input, const Tensor& mask, float scale_factor) {
    auto result = fuseMaskedSoftmax::apply(input, mask, scale_factor);
    return result[0];
}
}}