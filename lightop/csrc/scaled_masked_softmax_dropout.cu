#include <cuda_fp16.h>
#include <vector>
#include "scaled_masked_softmax_dropout.cuh"

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

std::tuple<Tensor,Tensor,Tensor> fuse_scaled_masked_softmax_dropout_forward(
    Tensor const& input,  Tensor const& mask, float scale_factor, double p)
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

  if (p == 1) {
    Tensor output_zero = at::zeros_like(input);
    Tensor output_softtmax_zero = at::zeros_like(input);
    Tensor output_dropout_zero = at::zeros_like(input, input.options().dtype(kBool));
    return std::tuple<Tensor,Tensor,Tensor>(output_zero, output_softtmax_zero, output_dropout_zero);
  }

  // Output 
  auto act_options = input.options().requires_grad(false);
  Tensor softmax_results = at::empty({batches, attn_heads, query_seq_len, key_seq_len}, act_options);
  Tensor output_results = at::empty({batches, attn_heads, query_seq_len, key_seq_len}, act_options);
  Tensor dropout_results = at::empty_like(input, mask.options());

  // Softmax Intermediate Result Ptr
  void* input_ptr = static_cast<void*>(input.data_ptr());
  void* mask_ptr = static_cast<void*>(mask.data_ptr());
  void* softmax_results_ptr = static_cast<void*>(softmax_results.data_ptr());
  void* output_results_ptr = static_cast<void*>(output_results.data_ptr());
  void* dropout_results_ptr = static_cast<void*>(dropout_results.data_ptr());

  double pa = 1 - p;

  DISPATCH_HALF_AND_BFLOAT(
      input.scalar_type(),
      "dispatch_scaled_masked_softmax_dropout_forward",
      dispatch_scaled_masked_softmax_dropout_forward<scalar_t, scalar_t, float>(
    reinterpret_cast<scalar_t*>(output_results_ptr),
    reinterpret_cast<scalar_t*>(softmax_results_ptr),
    reinterpret_cast<uint8_t*>(dropout_results_ptr),
	  reinterpret_cast<const scalar_t*>(input_ptr),
	  reinterpret_cast<const uint8_t*>(mask_ptr),
	  scale_factor,
	  query_seq_len,
	  key_seq_len,
	  batches,
	  attn_heads,
	  pad_batches,
    pa);
      );
  return std::tuple<Tensor,Tensor,Tensor>(output_results, softmax_results, dropout_results);
}

Tensor fuse_scaled_masked_softmax_dropout_backward(
    const Tensor& output_grads_, 
    const Tensor& softmax_results_, 
    float scale_factor,
    const Tensor& dropout_mask_,
    double p)  {
	
  auto output_grads = output_grads_.contiguous();
  auto softmax_results = softmax_results_.contiguous();
  auto dropout_mask = dropout_mask_.contiguous();

  //output grads is a 4d tensor with dimensions [batches, attn_heads, seq_len, seq_len]
  const int batches = output_grads.size(0);
  const int attn_heads = output_grads.size(1);
  const int query_seq_len = output_grads.size(2);
  const int key_seq_len = output_grads.size(3);

  auto act_options = output_grads.options().requires_grad(false);
  Tensor input_grads = at::empty({batches, attn_heads, query_seq_len, key_seq_len}, act_options);

  void* output_grads_ptr = static_cast<void*>(output_grads.data_ptr());
  void* input_grads_ptr = static_cast<void*>(input_grads.data_ptr());
  double pa = 1 - p;
  //Softmax Grad
  DISPATCH_HALF_AND_BFLOAT(
      output_grads_.scalar_type(),
      "dispatch_scaled_masked_softmax_dropout_backward",
      dispatch_scaled_masked_softmax_dropout_backward<scalar_t, scalar_t, float>(
      reinterpret_cast<scalar_t*>(input_grads_ptr), 
	  reinterpret_cast<scalar_t*>(output_grads_ptr), 
	  reinterpret_cast<scalar_t const*>(softmax_results.data_ptr()),
    reinterpret_cast<uint8_t const*>(dropout_mask.data_ptr()),
	  scale_factor,
	  query_seq_len,
	  key_seq_len,
	  batches,
	  attn_heads,
    pa);
      );
  
  //backward pass is completely in-place
  return input_grads;
}


class fuseMaskedSoftmaxDropout : public torch::autograd::Function<fuseMaskedSoftmaxDropout> {
 public:
  static torch::autograd::variable_list forward(
  // static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input,
      const torch::autograd::Variable& mask, 
      float scale_factor,
      double p) {
    
    AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
	            (input.scalar_type() == at::ScalarType::BFloat16), 
                "Only fp16 and bf16 are supported");
    auto result = fuse_scaled_masked_softmax_dropout_forward(input, mask, scale_factor, p);
    ctx->saved_data["scale"] = scale_factor;
    ctx->saved_data["p"] = p;
    if (p == 0) {
      ctx->save_for_backward({std::get<0>(result), std::get<2>(result)});
    } else {
      ctx->save_for_backward({std::get<1>(result), std::get<2>(result)});
    }
    return {std::get<0>(result)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    AT_ASSERTM(grad_output[0].dim() == 4, "grad_output expected 4D tensor");
    AT_ASSERTM((grad_output[0].scalar_type() == at::ScalarType::Half) ||
	            (grad_output[0].scalar_type() == at::ScalarType::BFloat16), 
                "Only fp16 and bf16 are supported");
    double pa = ctx->saved_data["p"].toDouble();
    if (pa == 1) {
      auto output_zero = at::zeros_like(grad_output[0]);
      return {output_zero, torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable()};
    }
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    float save_scale = (float)ctx->saved_data["scale"].toDouble();
    auto softmax_result = saved[0];
    auto dropout_result = saved[1];

    AT_ASSERTM(dropout_result.dim() == 4, "dropout_result expected 4D tensor");
    AT_ASSERTM(softmax_result.dim() == 4, "softmax_result expected 4D tensor");
    AT_ASSERTM((softmax_result.scalar_type() == at::ScalarType::Half) ||
	            (softmax_result.scalar_type() == at::ScalarType::BFloat16), 
                "softmax_result Only fp16 and bf16 are supported");

    auto result = fuse_scaled_masked_softmax_dropout_backward(grad_output[0], /*out_result, */softmax_result, save_scale, dropout_result, pa);
    return {result, torch::autograd::Variable(), torch::autograd::Variable(),  torch::autograd::Variable()};
  }
};
Tensor fuse_scaled_masked_softmax_dropout_forward_autograd(
    const Tensor& input, const Tensor& mask, float scale_factor, double p) {
    auto result = fuseMaskedSoftmaxDropout::apply(input, mask, scale_factor, p);
    return result[0];
}
}}