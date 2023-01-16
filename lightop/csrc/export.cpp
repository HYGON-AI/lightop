#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>

namespace at {  namespace native {
std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t> bnrelu_forward(const Tensor& self, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, bool train, double momentum, double epsilon, int fuse_act_mode, double act_alpha, double act_beta, double act_gamma);
std::tuple<Tensor, Tensor, Tensor> bnrelu_backward(int64_t impl_index, const Tensor& grad_out, const Tensor& input, const Tensor& output_relu, const Tensor& output_relu_grad, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, const c10::optional<Tensor>& save_mean_opt, 
                                                   const c10::optional<Tensor>& save_invstd_opt, bool train, double epsilon, std::array<bool,3> grad_input_mask, int fuse_act_mode, double act_alpha, double act_beta, double act_gamma);
std::tuple<Tensor, Tensor, Tensor, Tensor> bnaddrelu_cuda(const Tensor& self, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, bool train, double momentum, double epsilon, const Tensor& add);
std::tuple<Tensor, Tensor, Tensor> bnaddrelu_backward_cuda(const Tensor& grad_out, const Tensor& input, const Tensor& output, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, const c10::optional<Tensor>& save_mean_opt, const c10::optional<Tensor>& save_invstd_opt, bool train, double epsilon, std::array<bool,3> grad_input_mask);

std::tuple<Tensor,Tensor,Tensor>layernorm_forward(const Tensor& self,const Tensor&weight,const Tensor& bias,double ln_epsilon);
std::tuple<Tensor,Tensor,Tensor>layernorm_backward(const Tensor& grad, const Tensor& X,const Tensor& mean,const Tensor& rstd,const Tensor& gamma);
std::tuple<Tensor,Tensor> fused_add_dropout_forward(const Tensor& self,const Tensor& other, double p, bool training);
std::tuple<Tensor,Tensor> fused_add_dropout_backward(Tensor grad, Tensor mask, double p);
std::tuple<Tensor,Tensor,Tensor,Tensor>fused_ln_add_dropout_forward(const Tensor& self,const Tensor& other,const Tensor&weight,const Tensor& bias, double ln_epsilon, double p, bool training);
std::tuple<Tensor,Tensor,Tensor,Tensor>fused_ln_add_dropout_backward(const Tensor& grad,const Tensor& X, const Tensor& mean,const Tensor& rstd, const Tensor& gamma);
Tensor fused_linear_bias_forward(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2);
Tensor fused_linear_bias_backward(at::Tensor& self, at::Tensor& result, const at::Tensor& mat1, const at::Tensor& mat2);
std::tuple <Tensor, Tensor> fused_relu_dropout_forward(const Tensor &self, const double &p, const bool &use_relu, const bool &training);
Tensor fused_relu_dropout_backward(Tensor &grad, Tensor &mask);
Tensor fuse_softmax_forward(const Tensor &input, const int64_t dim, const bool half_to_float);
Tensor fuse_softmax_backward(const Tensor& grad, const Tensor& output, int64_t dim, const Tensor& input);
Tensor fuse_log_softmax_forward(const at::Tensor & self, int64_t dim, bool half_to_float);
Tensor fuse_log_softmax_backward(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self);
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bnrelu_forward", &at::native::bnrelu_forward, "fusebnrelu forward");
  m.def("bnrelu_backward", &at::native::bnrelu_backward, "fusebnrelu backward");
  m.def("bnaddrelu_forward", &at::native::bnaddrelu_cuda, "fusebnaddrelu forward");
  m.def("bnaddrelu_backward", &at::native::bnaddrelu_backward_cuda, "fusebnaddrelu backward");
  m.def("layernorm_forward", &at::native::layernorm_forward, "layernorm forward");
  m.def("layernorm_backward", &at::native::layernorm_backward, "layernorm backward");
  m.def("add_dropout_forward", &at::native::fused_add_dropout_forward, "fused_dropout forward");
  m.def("add_dropout_backward", &at::native::fused_add_dropout_backward, "fused_dropout backward");
  m.def("ln_add_dropout_forward", &at::native::fused_ln_add_dropout_forward, "fused_ln_add_dropout forward");
  m.def("ln_add_dropout_backward", &at::native::fused_ln_add_dropout_backward, "fused_ln_add_dropout backward");
  m.def("linearbias_forward", &at::native::fused_linear_bias_forward, "fused_linearbias forward");
  m.def("linearbias_backward", &at::native::fused_linear_bias_backward, "fused_linearbias backward");
  m.def("relu_dropout_forward", &at::native::fused_relu_dropout_forward, "fused_relu_dropout forward");
  m.def("relu_dropout_backward", &at::native::fused_relu_dropout_backward, "fused_relu_dropout backward");
  m.def("softmax_forward", &at::native::fuse_softmax_forward, "softmax of forward");
  m.def("softmax_backward", &at::native::fuse_softmax_backward, "softmax of backward");
  m.def("log_softmax_forward", &at::native::fuse_log_softmax_forward, "logsoftmax of forward");
  m.def("log_softmax_backward", &at::native::fuse_log_softmax_backward, "logsoftmax of backward");
}
