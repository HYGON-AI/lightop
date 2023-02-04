#include <ATen/ATen.h>

namespace at {
namespace native {
Tensor fuse_softmax_forward(const Tensor &input, const int64_t dim, const bool half_to_float);

Tensor fuse_softmax_backward(const Tensor& grad, const Tensor& output, int64_t dim, const Tensor& input);

Tensor fuse_log_softmax_forward(const at::Tensor & self, int64_t dim, bool half_to_float);

Tensor fuse_log_softmax_backward(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self);

} // namespace native
} // namespace at
