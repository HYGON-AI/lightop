#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/AccumulateType.h>

#include <THC/THCGeneral.h>
#include <torch/extension.h>

#include "gemm_bias.h"
#include "cuda_meta.h"

#include <iostream>


namespace at { namespace native {

namespace{
// aten/src/ATen/native/cuda/Blas.cpp
// TODO: https://github.com/pytorch/pytorch/pull/59380#pullrequestreview-725310492
c10::MaybeOwned<Tensor> inline resolve_conj_if_indicated(const Tensor& tensor, bool resolve_conj) {
  if (resolve_conj && tensor.is_conj()) {
    return c10::MaybeOwned<Tensor>::owned(tensor.resolve_conj());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
}

c10::MaybeOwned<Tensor> inline prepare_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor, bool transpose_result) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
      transpose_tensor = tensor.is_contiguous();
      return resolve_conj_if_indicated(tensor, transpose_result ? transpose_tensor : !transpose_tensor);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, !transpose_result);
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, transpose_result);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
  }
}

c10::MaybeOwned<Tensor> inline prepare_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
      transpose_tensor = tensor.is_contiguous();
      return resolve_conj_if_indicated(tensor, true);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, true);
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, true);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
  }
}

} // namespace

void create_gemmbias_handle() {
  forwardHandler.mu.lock();
  initGemmbias();
  forwardHandler.num++;
  if (forwardHandler.handler) {
    forwardHandler.mu.unlock();
    return;
  }
  int ret = gemmbias_forward_create_handle(&forwardHandler.handler);
  forwardHandler.mu.unlock();
  if (ret) {
    printf("ERROR, gemmbias_forward_create_handle ret %d\n",ret);
  }
}
void destroy_gemmbias_handle() {
  forwardHandler.mu.lock();
  if (--forwardHandler.num == 0 && forwardHandler.handler) {
    int ret = gemmbias_forward_destroy_handle(forwardHandler.handler);
    if (ret) {
      printf("ERROR, gemmbias_forward_destroy_handle ret %d\n",ret);
    } else {
      forwardHandler.handler = nullptr;
    }
  }
  forwardHandler.mu.unlock();
}

void create_gemmdbias_handle() {
  backwardHandler.mu.lock();
  initGemmbias();
  backwardHandler.num++;
  if (backwardHandler.handler) {
    backwardHandler.mu.unlock();
    return;
  }
  int ret = gemmdbias_backward_create_handle(&backwardHandler.handler);
  backwardHandler.mu.unlock();
  if (ret) {
    printf("ERROR, gemmdbias_backward_create_handle ret %d\n",ret);
  }
}
void destroy_gemmdbias_handle() {
  if (!gemmdbias_backward_destroy_handle) {
    printf("ERROR, get gemmdbias_backward_destroy_handle err\n");
    return;
  }
  backwardHandler.mu.lock();
  if (--backwardHandler.num == 0 && backwardHandler.handler) {
    int ret = gemmdbias_backward_destroy_handle(backwardHandler.handler);
    if (ret) {
      printf("ERROR, gemmdbias_backward_destroy_handle ret %d\n",ret);
    } else {
      backwardHandler.handler = nullptr;
      backwardHandler.num = 0;
    }
  }
  backwardHandler.mu.unlock();
}

// HC Debug
void fused_linear_bias_forward_impl(const Tensor& self, const Tensor& mat1, const Tensor& mat2, Tensor& result, bool gemm_only = false) {
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");

  TensorArg args[]{{result, "out", 0}, {self, "self", 1}, {mat1, "mat1", 2}, {mat2, "mat2", 3}};
  checkAllSameGPU(__func__, args);

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();

  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return;
  }

  bool transpose_result;
  c10::MaybeOwned<Tensor> result_ = prepare_matrix_for_cublas(result, transpose_result);
  bool transpose_mat1;
  bool transpose_mat2;
  auto mat1_ = prepare_matrix_for_cublas(transpose_result ? mat2 : mat1, transpose_mat1, transpose_result);
  auto mat2_ = prepare_matrix_for_cublas(transpose_result ? mat1 : mat2, transpose_mat2, transpose_result);

  if (transpose_result) {
    transpose_mat1 = !transpose_mat1;
    transpose_mat2 = !transpose_mat2;
    mat1_sizes = mat1_->sizes();
    mat2_sizes = mat2_->sizes();
  }

  int64_t m = mat1_sizes[transpose_result ? 1 : 0];
  int64_t k = mat1_sizes[transpose_result ? 0 : 1];
  int64_t n = mat2_sizes[transpose_result ? 0 : 1];
  int64_t mat1_ld = mat1_->stride((transpose_mat1 == transpose_result) ? 1 : 0);
  int64_t mat2_ld = mat2_->stride((transpose_mat2 == transpose_result) ? 1 : 0);
  int64_t result_ld = result_->stride(transpose_result ? 0 : 1);
  at::ScalarType scalar_type = self.scalar_type();

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result_->is_conj());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(scalar_type, "fused_linear_bias_forward", [&] {
    scalar_t alpha_val = (scalar_t)1.0;
    scalar_t beta_val = gemm_only ? (scalar_t)0.0 : (scalar_t)1.0;
    scalar_t* mat1_ptr = mat1_->data_ptr<scalar_t>();
    scalar_t* mat2_ptr = mat2_->data_ptr<scalar_t>();
    scalar_t* result_ptr = result_->data_ptr<scalar_t>();
    scalar_t* self_ptr = self.data_ptr<scalar_t>();
    gemmbias_forward_half(
      forwardHandler.handler,
      transpose_mat1 ? int(1) : int(0),
      transpose_mat2 ? int(1) : int(0),
      (int)m, (int)n, (int)k,
      &alpha_val,
      mat1_ptr, mat2_ptr, 
      &beta_val,
      self_ptr, result_ptr
    );
    // gemm_bias(
    //   transpose_mat1 ? int(1) : int(0),
    //   transpose_mat2 ? int(1) : int(0),
    //   mat1_ptr, mat2_ptr, self_ptr, result_ptr,
    //   (int)m, (int)k, (int)n,
    //   &alpha_val,
    //   &beta_val
    // );
  });
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
}

// HC Debug
void fused_linear_bias_backward_impl(const Tensor& self, Tensor & output, const Tensor& mat1, const Tensor& mat2, bool gemm_only = false) {
  Tensor result = at::empty({mat1.sizes()[0], mat2.sizes()[1]},  mat1.options());
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");

  TensorArg args[]{{result, "out", 0}, {self, "self", 1}, {mat1, "mat1", 2}, {mat2, "mat2", 3}};
  checkAllSameGPU(__func__, args);

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();

  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return;
  }

  bool transpose_result;
  c10::MaybeOwned<Tensor> result_ = prepare_matrix_for_cublas(result, transpose_result);
  bool transpose_mat1;
  bool transpose_mat2;
  auto mat1_ = prepare_matrix_for_cublas(transpose_result ? mat2 : mat1, transpose_mat1, transpose_result);
  auto mat2_ = prepare_matrix_for_cublas(transpose_result ? mat1 : mat2, transpose_mat2, transpose_result);

  if (transpose_result) {
    transpose_mat1 = !transpose_mat1;
    transpose_mat2 = !transpose_mat2;
    mat1_sizes = mat1_->sizes();
    mat2_sizes = mat2_->sizes();
  }

  int64_t m = mat1_sizes[transpose_result ? 1 : 0];
  int64_t k = mat1_sizes[transpose_result ? 0 : 1];
  int64_t n = mat2_sizes[transpose_result ? 0 : 1];
  int64_t mat1_ld = mat1_->stride((transpose_mat1 == transpose_result) ? 1 : 0);
  int64_t mat2_ld = mat2_->stride((transpose_mat2 == transpose_result) ? 1 : 0);
  int64_t result_ld = result_->stride(transpose_result ? 0 : 1);
  at::ScalarType scalar_type = self.scalar_type();


  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result_->is_conj());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(scalar_type, "fused_linear_bias_backward", [&] {
    scalar_t alpha_val = (scalar_t)1.0;
    scalar_t beta_val = gemm_only ? (scalar_t)0.0 : (scalar_t)1.0;
    scalar_t* mat1_ptr = mat1_->data_ptr<scalar_t>();
    scalar_t* mat2_ptr = mat2_->data_ptr<scalar_t>();
    scalar_t* result_ptr = result_->data_ptr<scalar_t>();
    scalar_t* self_ptr = self.data_ptr<scalar_t>();
    gemmdbias_backward_half(
      backwardHandler.handler,
      transpose_mat1 ? int(1) : int(0),
      transpose_mat2 ? int(1) : int(0),
      (int)m, (int)n, (int)k,
      (const void*)&alpha_val,
      (const void*)mat1_ptr, (const void*)mat2_ptr, 
      (const void*)&beta_val,
      (void*)self_ptr, (void*)result_ptr
    );
    // gemm_dbias(
    //   transpose_mat1 ? int(1) : int(0),
    //   transpose_mat2 ? int(1) : int(0),
    //   mat1_ptr, mat2_ptr, self_ptr, result_ptr,
    //   (int)m, (int)k, (int)n,
    //   &alpha_val
    // );
  });

  if (!output.is_same(*result_)) {
    output.copy_(*result_);
  }
}

struct TORCH_API structured__fused_linear : public at::impl::MetaBase {  
  void meta(const Tensor& self, const Tensor& mat1, const Tensor& mat2) {
    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
    TORCH_CHECK(
        mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
        mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

    auto names = at::namedinference::propagate_names_for_addmm(mat1, mat2, self);
    set_output(0, {mat1.sizes()[0], mat2.sizes()[1]}, {}, self.options(), names);
    const auto& result = maybe_get_output(0);
    //this check can fire for inplace op only, for all other versions result is guaranteed to be correct size
    TORCH_CHECK(((result.dim() == 2) && (result.sizes()[0] == mat1.sizes()[0]) && (result.sizes()[1] == mat2.sizes()[1])),
    "The input tensor must be a matrix with size ", mat1.sizes()[0], "x", mat2.sizes()[1], ", but got a ", result.dim(),
    "-D tensor with size ", result.sizes()[0], "x", result.sizes()[1]);
  }
  void impl(const Tensor& self, const Tensor& mat1, const Tensor& mat2, Tensor& result, bool gemm_only){
    fused_linear_bias_forward_impl(self, mat1, mat2, result, gemm_only);
  }
    
};
struct structured__fused_linear_functional final : public at::native::structured__fused_linear {

    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options, DimnameList names) override {

        auto current_device = guard_.current_device();
        if (C10_UNLIKELY(current_device.has_value())) {
          TORCH_INTERNAL_ASSERT(*current_device == options.device(),
            "structured kernels don't support multi-device outputs");
        } else {
          guard_.reset_device(options.device());
        }


        outputs_[output_idx] = create_out(sizes, strides, options);
        if (!names.empty()) {
          namedinference::propagate_names(*outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output

    }

    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
    c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard_;
};
at::Tensor fused_linear_bias_forward(c10::optional<Tensor> & self_opt, const at::Tensor & mat1, const at::Tensor & mat2) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning

  auto has_self = self_opt.has_value();
  const at::Tensor & self = has_self ? *c10::MaybeOwned<Tensor>::borrowed(*self_opt) : torch::empty({0}, mat1.options());

  c10::impl::check_and_update_common_device(common_device, self, "wrapper_addmm", "self");
  c10::impl::check_and_update_common_device(common_device, mat1, "wrapper_addmm", "mat1");
  c10::impl::check_and_update_common_device(common_device, mat2, "wrapper_addmm", "mat2");

  auto dim_mat1 = mat1.dim();
  auto dim_mat2 = mat2.dim();
  if (dim_mat1 >= 3 && (dim_mat2 == 1 || dim_mat2 == 2)) {
    const at::Tensor t2 = dim_mat2 == 1 ? mat2.unsqueeze(-1) : mat2;
    auto size1 = mat1.sizes();
    auto size2 = t2.sizes();
    std::vector<int64_t> output_size;
    output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
    if (dim_mat2 > 1) {
      output_size.push_back(size2[dim_mat2 - 1]);
    }
    const auto dim1_size = c10::multiply_integers(size1.begin(), size1.end() - 1);
    auto t1 = mat1.expect_contiguous()->view({dim1_size, size1[size1.size() - 1]});
    structured__fused_linear_functional op;
    op.meta(self, t1, t2);
    op.impl(self, t1, t2, *op.outputs_[0], !has_self);
    return std::move(at::_unsafe_view(*op.outputs_[0], output_size));
  } else {
    structured__fused_linear_functional op;
    op.meta(self, mat1, mat2);
    op.impl(self, mat1, mat2, *op.outputs_[0], !has_self);
    return std::move(op.outputs_[0]).take();
  }
}

Tensor fused_linear_bias_backward(c10::optional<Tensor> & self_opt, Tensor & result, const at::Tensor & mat1, const at::Tensor & mat2) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning

  auto has_self = self_opt.has_value();
  const at::Tensor & self = has_self ? *c10::MaybeOwned<Tensor>::borrowed(*self_opt) : torch::empty({0}, mat1.options());

  c10::impl::check_and_update_common_device(common_device, self, "wrapper_addmm", "self");
  c10::impl::check_and_update_common_device(common_device, mat1, "wrapper_addmm", "mat1");
  c10::impl::check_and_update_common_device(common_device, mat2, "wrapper_addmm", "mat2");

  auto dim_mat1 = mat1.dim();
  auto dim_mat2 = mat2.dim();
  if (dim_mat1 >= 3 && (dim_mat2 == 1 || dim_mat2 == 2)) {
    const at::Tensor t2 = dim_mat2 == 1 ? mat2.unsqueeze(-1) : mat2;
    auto size1 = mat1.sizes();
    auto size2 = t2.sizes();
    std::vector<int64_t> output_size;
    output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
    if (dim_mat2 > 1) {
      output_size.push_back(size2[dim_mat2 - 1]);
    }
    const auto dim1_size = c10::multiply_integers(size1.begin(), size1.end() - 1);
    auto t1 = mat1.expect_contiguous()->view({dim1_size, size1[size1.size() - 1]});
    fused_linear_bias_backward_impl(self, result, t1, t2, !has_self);
    return std::move(result);
  } else {
    fused_linear_bias_backward_impl(self, result, mat1, mat2, !has_self);
    return std::move(result);
  }
}

} // namespace native
} //namespace at


