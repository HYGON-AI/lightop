#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/AccumulateType.h>

#include <THC/THCGeneral.h>
#include <torch/extension.h>

#include "gemm_bias.h"

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

// HC Debug
Tensor fused_linear_bias_forward(const Tensor& self, const Tensor& mat1, const Tensor& mat2) {
  Tensor result = at::empty({mat1.sizes()[0], mat2.sizes()[1]},  self.options());
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");

  TensorArg args[]{{result, "out", 0}, {self, "self", 1}, {mat1, "mat1", 2}, {mat2, "mat2", 3}};
  checkAllSameGPU(__func__, args);

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();

  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
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

  // Just For Debug
  //std::cout << "m " << m << " n " << n << " k "<< k << " mat1_ld " << mat1_ld << " mat2_ld " << mat2_ld  << " result_ld " << result_ld << "\n";
  //std::cout << "transpose_mat1 " << transpose_mat1 << " transpose_mat2 "  << transpose_mat2 << " transpose_result " << transpose_result << "\n";


  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result_->is_conj());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(scalar_type, "fused_linear_bias_forward", [&] {
    scalar_t alpha_val = (scalar_t)1.0;
    scalar_t beta_val = (scalar_t)1.0;
    scalar_t* mat1_ptr = mat1_->data_ptr<scalar_t>();
    scalar_t* mat2_ptr = mat2_->data_ptr<scalar_t>();
    scalar_t* result_ptr = result_->data_ptr<scalar_t>();
    scalar_t* self_ptr = self.data_ptr<scalar_t>();
    gemm_bias(
      transpose_mat1 ? int(1) : int(0),
      transpose_mat2 ? int(1) : int(0),
      mat1_ptr, mat2_ptr, self_ptr, result_ptr,
      (int)m, (int)k, (int)n,
      &alpha_val,
      &beta_val
    );
  });
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
 
//  at::ScalarType scalar_type = self.scalar_type();
//  AT_DISPATCH_FLOATING_TYPES_AND_HALF(scalar_type, "fused_linear_bias_forward", [&] {
//    scalar_t alpha_val = (scalar_t)1.0;
//    scalar_t beta_val = (scalar_t)1.0;
//    scalar_t* mat1_ptr = mat1.data_ptr<scalar_t>();
//    scalar_t* mat2_ptr = mat2.data_ptr<scalar_t>();
//    scalar_t* result_ptr = result.data_ptr<scalar_t>();
//    scalar_t* self_ptr = self.data_ptr<scalar_t>();
//    gemm_bias(
//      int(0),
//      int(0),
//      mat2_ptr, mat1_ptr, self_ptr, result_ptr,
//      (int)mat2.sizes()[1], (int)mat2.sizes()[0], (int)mat1.sizes()[0],
//      &alpha_val,
//      &beta_val
//    );
//  });
  
  return result;
}

// HC Debug
Tensor fused_linear_bias_backward(Tensor& self, Tensor & result, const Tensor& mat1, const Tensor& mat2) {
  //Tensor result = at::empty({mat1.sizes()[0], mat2.sizes()[1]},  mat1.options());
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");

  TensorArg args[]{{result, "out", 0}, {self, "self", 1}, {mat1, "mat1", 2}, {mat2, "mat2", 3}};
  checkAllSameGPU(__func__, args);

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();

  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
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


  // Just For Debug
  //std::cout << "m " << m << " n " << n << " k "<< k << " mat1_ld " << mat1_ld << " mat2_ld " << mat2_ld  << " result_ld " << result_ld << "\n";
  //std::cout << "transpose_mat1 " << transpose_mat1 << " transpose_mat2 "  << transpose_mat2 << " transpose_result " << transpose_result << "\n";


  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result_->is_conj());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(scalar_type, "fused_linear_bias_backward", [&] {
    scalar_t alpha_val = (scalar_t)1.0;
    scalar_t* mat1_ptr = mat1_->data_ptr<scalar_t>();
    scalar_t* mat2_ptr = mat2_->data_ptr<scalar_t>();
    scalar_t* result_ptr = result_->data_ptr<scalar_t>();
    scalar_t* self_ptr = self.data_ptr<scalar_t>();
    gemm_dbias(
      transpose_mat1 ? int(1) : int(0),
      transpose_mat2 ? int(1) : int(0),
      mat1_ptr, mat2_ptr, self_ptr, result_ptr,
      (int)m, (int)k, (int)n,
      &alpha_val
    );
  });

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }

//  Tensor result = at::empty({mat1.sizes()[1], mat2.sizes()[1]},  mat1.options());
//  std::cout << " m = " << (int)mat2.sizes()[1] << " k = " << (int)mat2.sizes()[0] << " n = " << (int)mat1.sizes()[1];
//  at::ScalarType scalar_type = self.scalar_type();
//  AT_DISPATCH_FLOATING_TYPES_AND_HALF(scalar_type, "fused_linear_bias_backward", [&] {
//    scalar_t alpha_val = (scalar_t)1.0;
//    scalar_t* mat1_ptr = mat1.data_ptr<scalar_t>();
//    scalar_t* mat2_ptr = mat2.data_ptr<scalar_t>();
//    scalar_t* result_ptr = result.data_ptr<scalar_t>();
//    scalar_t* self_ptr = self.data_ptr<scalar_t>();
//    gemm_dbias(
//      int(0),
//      int(1),
//      mat2_ptr, mat1_ptr, self_ptr, result_ptr,
//      (int)mat2.sizes()[1], (int)mat2.sizes()[0], (int)mat1.sizes()[1],
//      &alpha_val
//    );
//  });

  return result;
}

} // namespace native
} //namespace at


