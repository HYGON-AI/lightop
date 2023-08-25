#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/AccumulateType.h>

#include <torch/extension.h>
#include <torch/autograd.h>

#include "gemm_bias.h"
#include "cuda_meta.h"

#include <iostream>

#define SIZE_THRESHOLD 1024

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
  fmu.lock();
  initGemmbias();
  forwardHandler.num++;
  if (forwardHandler.handler) {
    fmu.unlock();
    return;
  }
  rocblas_create_handle(&handle);
  int ret = gemmbias_forward_create_handle(&forwardHandler.handler);
  fmu.unlock();
  if (ret) {
    printf("ERROR, gemmbias_forward_create_handle ret %d\n",ret);
  }
}
void destroy_gemmbias_handle() {
  fmu.lock();
  if (--forwardHandler.num == 0 && forwardHandler.handler) {
    rocblas_destroy_handle(handle);
    int ret = gemmbias_forward_destroy_handle(forwardHandler.handler);
    if (ret) {
      printf("ERROR, gemmbias_forward_destroy_handle ret %d\n",ret);
    } else {
      forwardHandler.handler = nullptr;
    }
  }
  fmu.unlock();
}

void create_gemmdbias_handle() {
  bmu.lock();
  initGemmbias();
  backwardHandler.num++;
  if (backwardHandler.handler) {
    bmu.unlock();
    return;
  }
  int ret = gemmdbias_backward_create_handle(&backwardHandler.handler);
  bmu.unlock();
  if (ret) {
    printf("ERROR, gemmdbias_backward_create_handle ret %d\n",ret);
  }
}
void destroy_gemmdbias_handle() {
  if (!gemmdbias_backward_destroy_handle) {
    printf("ERROR, get gemmdbias_backward_destroy_handle err\n");
    return;
  }
  bmu.lock();
  if (--backwardHandler.num == 0 && backwardHandler.handler) {
    int ret = gemmdbias_backward_destroy_handle(backwardHandler.handler);
    if (ret) {
      printf("ERROR, gemmdbias_backward_destroy_handle ret %d\n",ret);
    } else {
      backwardHandler.handler = nullptr;
      backwardHandler.num = 0;
    }
  }
  bmu.unlock();
}

void fused_linear_bias_forward_impl(const Tensor& self, const Tensor& mat1, const Tensor& mat2, Tensor& result) {
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
  
  // std::cout << "mat1:" << *mat1_ << std::endl;
  
  // std::cout << "mat2:" << *mat2_ << std::endl;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(scalar_type, "fused_linear_bias_forward", [&] {
    scalar_t alpha_val = (scalar_t)1.0;
    scalar_t beta_val = (scalar_t)1.0;
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
    // std::cout << __FILE__ << ":" << __LINE__ << " transpose_mat1:" << transpose_mat1 << " transpose_mat2:" << transpose_mat2 << " transpose_result:" << transpose_result << \
    " m:" << m << " n:" << n << " k:" << k << std::endl;
    // Tensor self_expand = self.expand_as(result);
    // Tensor self_expand = self.repeat({n, 1});
    // scalar_t* self_expand_ptr = self_expand.data_ptr<scalar_t>();
    // // std::cout << "self_expand:" << self_expand << " self_expand:" << self_expand.is_contiguous() <<  std::endl;
    // // std::cout << "self:" << self << std::endl;
    // rocblas_status rocblas_res;
    // rocblas_res = rocblas_gemm_ex(
    //   handle,
    //   transpose_mat1 ? rocblas_operation_transpose : rocblas_operation_none,
    //   transpose_mat2 ? rocblas_operation_transpose : rocblas_operation_none,
    //   m, n, k,
    //   (const void*)(&alpha_val),
    //   (const void*)mat1_ptr,
    //   rocblas_datatype_f16_r,
    //   transpose_mat1 ? k : m,
    //   (const void*)mat2_ptr,
    //   rocblas_datatype_f16_r,
    //   transpose_mat2 ? n : k,
    //   (const void*)(&beta_val),
    //   (const void*)self_expand_ptr,
    //   rocblas_datatype_f16_r,
    //   m,
    //   (void*)result_ptr,
    //   rocblas_datatype_f16_r,
    //   m,
    //   rocblas_datatype_f16_r,
    //   rocblas_gemm_algo_standard,
    //   0, 0
    // );
    
    // std::cout << "rocblas_res:" << (rocblas_res == rocblas_status_success) << ", " << rocblas_res << std::endl;
    // gemm_bias(
    //   transpose_mat1 ? int(1) : int(0),
    //   transpose_mat2 ? int(1) : int(0),
    //   mat1_ptr, mat2_ptr, self_ptr, result_ptr,
    //   (int)m, (int)k, (int)n,
    //   &alpha_val,
    //   &beta_val
    // );
  });
  // std::cout << "result_:" << *result_ << std::endl;
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
}

// HC Debug
Tensor fused_linear_bias_backward_impl(const Tensor& self, const Tensor & result, const Tensor& mat1, const Tensor& mat2) {
  // Tensor result = at::empty({mat1.sizes()[0], mat2.sizes()[1]},  mat1.options());
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

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result_->is_conj());

  // std::cout << "transpose_mat1:" << transpose_mat1 << " transpose_mat2:" << transpose_mat2 << " transpose_result:" << transpose_result << std::endl;

  // std::cout << "mat1_ is_continous:" << mat1_->is_contiguous() << std::endl;
  // __half * mat1_p = (__half *)mat1_->data_ptr();
  // for (int i = 0; i < 10; ++i) {
  //   std::cout << __half2float(mat1_p[i]) << " ";
  // }
  // std::cout << std::endl;
  // std::cout << *mat1_ << std::endl;
  // std::cout << "mat2_ is_continous:" << mat2_->is_contiguous() <<std::endl;
  // __half * mat2_p = (__half *)mat2_->data_ptr();
  // for (int i = 0; i < 10; ++i) {
  //   std::cout << __half2float(mat2_p[i]) << " ";
  // }
  // std::cout << std::endl;
  // std::cout << *mat2_ << std::endl;

  // rocblas_status rocblas_res;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(scalar_type, "fused_linear_bias_backward", [&] {
    scalar_t alpha_val = (scalar_t)1.0;
    scalar_t beta_val = (scalar_t)0.0;
    scalar_t* mat1_ptr = mat1_->data_ptr<scalar_t>();
    scalar_t* mat2_ptr = mat2_->data_ptr<scalar_t>();
    scalar_t* result_ptr = result_->data_ptr<scalar_t>();
    // std::cout << __FILE__ << ":" << __LINE__ << " k:" << k << std::endl;
    if (k <= SIZE_THRESHOLD) {
      scalar_t* self_ptr = self.data_ptr<scalar_t>();
      scalar_t beta_val = (scalar_t)1.0;
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
    }
    else {
      scalar_t beta_val = (scalar_t)0.0;
      TORCH_CUDABLAS_CHECK(rocblas_gemm_ex(
        handle,
        transpose_mat1 ? rocblas_operation_transpose : rocblas_operation_none,
        transpose_mat2 ? rocblas_operation_transpose : rocblas_operation_none,
        m, n, k,
        (const void*)(&alpha_val),
        (const void*)mat1_ptr,
        rocblas_datatype_f16_r,
        transpose_mat1 ? k : m,
        (const void*)mat2_ptr,
        rocblas_datatype_f16_r,
        transpose_mat2 ? n : k,
        (const void*)(&beta_val),
        (const void*)result_ptr,
        rocblas_datatype_f16_r,
        m,
        (void*)result_ptr,
        rocblas_datatype_f16_r,
        m,
        rocblas_datatype_f16_r,
        rocblas_gemm_algo_standard,
        0, 0
      ));
    }
    // rocblas_res= rocblas_gemm_strided_batched_ex(
    //   handle,
    //   transpose_mat1 ? rocblas_operation_transpose : rocblas_operation_none,
    //   transpose_mat2 ? rocblas_operation_transpose : rocblas_operation_none,
    //   m, n, k,
    //   (const void*)(&alpha_val),
    //   (const void*)mat1_ptr,
    //   rocblas_datatype_f16_r,
    //   transpose_mat1 ? k : m,
    //   mat1_->stride(1),
    //   (const void*)mat2_ptr,
    //   rocblas_datatype_f16_r,
    //   transpose_mat2 ? n : k,
    //   mat2_->stride(0),
    //   (const void*)(&beta_val),
    //   (const void*)result_ptr,
    //   rocblas_datatype_f16_r,
    //   m,
    //   result_->stride(1),
    //   (void*)result_ptr,
    //   rocblas_datatype_f16_r,
    //   m,
    //   result_->stride(1),
    //   1,
    //   rocblas_datatype_f16_r,
    //   rocblas_gemm_algo_standard,
    //   0, 0
    // );

    // gemm_dbias(
    //   transpose_mat1 ? int(1) : int(0),
    //   transpose_mat2 ? int(1) : int(0),
    //   mat1_ptr, mat2_ptr, self_ptr, result_ptr,
    //   (int)m, (int)k, (int)n,
    //   &alpha_val
    // );
  });
  // std::cout << "rocblas_res:" << (rocblas_res == rocblas_status_success) << ", " << rocblas_res << std::endl;
  // std::cout << "result_ is_continous:" << result_->is_contiguous() << std::endl;
  // __half * result_p = (__half *)result_->data_ptr();
  // for (int i = 0; i < 12; ++i) {
  //   std::cout << __half2float(result_p[i]) << " ";
  // }
  // std::cout << std::endl;
  // std::cout << *result_ << std::endl;

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}

Tensor fused_linear_bias_backward_3d_impl(const Tensor& self, const Tensor & result, const Tensor& mat1, const Tensor& mat2) {
  TensorArg args[]{{result, "out", 0}, {self, "self", 1}, {mat1, "mat1", 2}, {mat2, "mat2", 3}};
  checkAllSameGPU(__func__, args);

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();

  int64_t m = mat2_sizes[1];
  int64_t k = mat2_sizes[0];
  int64_t n = mat1_sizes[1];
  // int64_t mat1_ld = mat1_->stride((transpose_mat1 == transpose_result) ? 1 : 0);
  // int64_t mat2_ld = mat2_->stride((transpose_mat2 == transpose_result) ? 1 : 0);
  // int64_t result_ld = result_->stride(transpose_result ? 0 : 1);
  at::ScalarType scalar_type = self.scalar_type();

  // TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result_->is_conj());

  // std::cout << "transpose_mat1:" << transpose_mat1 << " transpose_mat2:" << transpose_mat2 << " transpose_result:" << transpose_result << std::endl;

  // std::cout << "mat1_ is_continous:" << mat1_->is_contiguous() << std::endl;
  // __half * mat1_p = (__half *)mat1_->data_ptr();
  // for (int i = 0; i < 10; ++i) {
  //   std::cout << __half2float(mat1_p[i]) << " ";
  // }
  // std::cout << std::endl;
  // std::cout << *mat1_ << std::endl;
  // std::cout << "mat2_ is_continous:" << mat2_->is_contiguous() <<std::endl;
  // __half * mat2_p = (__half *)mat2_->data_ptr();
  // for (int i = 0; i < 10; ++i) {
  //   std::cout << __half2float(mat2_p[i]) << " ";
  // }
  // std::cout << std::endl;
  // std::cout << *mat2_ << std::endl;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(scalar_type, "fused_linear_bias_backward", [&] {
    scalar_t alpha_val = (scalar_t)1.0;
    scalar_t beta_val = (scalar_t)0.0;
    scalar_t* mat1_ptr = mat2.data_ptr<scalar_t>();
    scalar_t* mat2_ptr = mat1.data_ptr<scalar_t>();
    scalar_t* result_ptr = result.data_ptr<scalar_t>();
    scalar_t* self_ptr = self.data_ptr<scalar_t>();
    // gemmdbias_backward_half(
    //   backwardHandler.handler,
    //   transpose_mat1 ? int(1) : int(0),
    //   transpose_mat2 ? int(1) : int(0),
    //   (int)m, (int)n, (int)k,
    //   (const void*)&alpha_val,
    //   (const void*)mat1_ptr, (const void*)mat2_ptr, 
    //   (const void*)&beta_val,
    //   (void*)self_ptr, (void*)result_ptr
    // );
    TORCH_CUDABLAS_CHECK(rocblas_gemm_ex(
      handle,
      rocblas_operation_none,
      rocblas_operation_transpose,
      m, n, k,
      (const void*)(&alpha_val),
      (const void*)mat1_ptr,
      rocblas_datatype_f16_r,
      m,
      (const void*)mat2_ptr,
      rocblas_datatype_f16_r,
      n,
      (const void*)(&beta_val),
      (const void*)result_ptr,
      rocblas_datatype_f16_r,
      m,
      (void*)result_ptr,
      rocblas_datatype_f16_r,
      m,
      rocblas_datatype_f16_r,
      rocblas_gemm_algo_standard,
      0, 0
    ));
    // rocblas_res= rocblas_gemm_strided_batched_ex(
    //   handle,
    //   transpose_mat1 ? rocblas_operation_transpose : rocblas_operation_none,
    //   transpose_mat2 ? rocblas_operation_transpose : rocblas_operation_none,
    //   m, n, k,
    //   (const void*)(&alpha_val),
    //   (const void*)mat1_ptr,
    //   rocblas_datatype_f16_r,
    //   transpose_mat1 ? k : m,
    //   mat1_->stride(1),
    //   (const void*)mat2_ptr,
    //   rocblas_datatype_f16_r,
    //   transpose_mat2 ? n : k,
    //   mat2_->stride(0),
    //   (const void*)(&beta_val),
    //   (const void*)result_ptr,
    //   rocblas_datatype_f16_r,
    //   m,
    //   result_->stride(1),
    //   (void*)result_ptr,
    //   rocblas_datatype_f16_r,
    //   m,
    //   result_->stride(1),
    //   1,
    //   rocblas_datatype_f16_r,
    //   rocblas_gemm_algo_standard,
    //   0, 0
    // );

    // gemm_dbias(
    //   transpose_mat1 ? int(1) : int(0),
    //   transpose_mat2 ? int(1) : int(0),
    //   mat1_ptr, mat2_ptr, self_ptr, result_ptr,
    //   (int)m, (int)k, (int)n,
    //   &alpha_val
    // );
  });
  // std::cout << "rocblas_res:" << (rocblas_res == rocblas_status_success) << ", " << rocblas_res << std::endl;
  // std::cout << "result_ is_continous:" << result_->is_contiguous() << std::endl;
  // __half * result_p = (__half *)result_->data_ptr();
  // for (int i = 0; i < 12; ++i) {
  //   std::cout << __half2float(result_p[i]) << " ";
  // }
  // std::cout << std::endl;
  // std::cout << *result_ << std::endl;

  // if (!result.is_same(*result_)) {
  //   result.copy_(*result_);
  // }
  return result;
}

struct TORCH_API structured__fused_linear : public at::impl::MetaBase {  
  void meta(const Tensor& self, const Tensor& mat1, const Tensor& mat2) {
    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
    TORCH_CHECK(
        mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
        mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

    auto names = at::namedinference::propagate_names_for_addmm(mat1, mat2, self);
    #ifdef TORCH_VERSION_OVER_110
        set_output_raw_strided(0, {mat1.sizes()[0], mat2.sizes()[1]}, {}, mat1.options(), names);
    #else
        set_output(0, {mat1.sizes()[0], mat2.sizes()[1]}, {}, self.options(), names);
    #endif
    const auto& result = maybe_get_output(0);
    //this check can fire for inplace op only, for all other versions result is guaranteed to be correct size
    TORCH_CHECK(((result.dim() == 2) && (result.sizes()[0] == mat1.sizes()[0]) && (result.sizes()[1] == mat2.sizes()[1])),
    "The input tensor must be a matrix with size ", mat1.sizes()[0], "x", mat2.sizes()[1], ", but got a ", result.dim(),
    "-D tensor with size ", result.sizes()[0], "x", result.sizes()[1]);
  }
  void impl(const Tensor& self, const Tensor& mat1, const Tensor& mat2, Tensor& result){
    fused_linear_bias_forward_impl(self, mat1, mat2, result);
  }
    
};
TORCH_META_FUNC_FINAL(_fused_linear)

at::Tensor fused_linear_bias_forward(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning

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
    // structured__fused_linear_functional op;
    // op.meta(self, t1, t2);
    // op.impl(self, t1, t2, *op.outputs_[0]);
    Tensor result = at::empty({t1.sizes()[0], t2.sizes()[1]},  self.options());
    fused_linear_bias_forward_impl(self, t1, t2, result);
    return std::move(at::_unsafe_view(result, output_size));
  } else {
    // structured__fused_linear_functional op;
    // op.meta(self, mat1, mat2);
    // op.impl(self, mat1, mat2, *op.outputs_[0]);
    Tensor result = at::empty({mat1.sizes()[0], mat2.sizes()[1]},  self.options());
    fused_linear_bias_forward_impl(self, mat1, mat2, result);
    return std::move(result);
  }
}

Tensor fused_linear_bias_backward(const Tensor& self, const Tensor & result, const Tensor& mat1, const Tensor& mat2) {
  auto dim_mat1 = mat1.dim();
  auto dim_mat2 = mat2.dim();

  if (dim_mat1 >= 3 && dim_mat1 == dim_mat2) {
    auto size1 = mat1.sizes();
    auto size2 = mat2.sizes();
    std::vector<int64_t> output_size;
    output_size.push_back(size2[dim_mat2 - 1]);
    output_size.push_back(size1[dim_mat1 - 1]);
    const auto dim1_size = c10::multiply_integers(size1.begin(), size1.end() - 1);
    auto t1 = mat1.expect_contiguous()->view({dim1_size, size1[size1.size() - 1]});
    auto t2 = mat2.expect_contiguous()->view({dim1_size, size2[size2.size() - 1]}); // 前面维度mat1和mat2相同
    if (dim1_size >= SIZE_THRESHOLD) self.set_(t1.sum(0).detach());
    fused_linear_bias_backward_impl(self, result, t1.t(), t2);
    // fused_linear_bias_backward_3d_impl(self, result, t1, t2);
    return std::move(at::_unsafe_view(result, output_size));
  } else {
    fused_linear_bias_backward_impl(self, result, mat1.t(), mat2);
    return std::move(result);
  }
}

class fusedLinearBiasFunction : public torch::autograd::Function<fusedLinearBiasFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input,
      const torch::autograd::Variable& weight,
      const torch::autograd::Variable& bias,
      const torch::autograd::Variable& grad_weight,
      const torch::autograd::Variable& grad_bias) {
    ctx->save_for_backward({input, weight, bias, grad_weight, grad_bias});
    
    if (input.device().type() == DeviceType::CUDA) {
      auto result = fused_linear_bias_forward(bias, input, weight.t());
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
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];
    auto grad_weight = saved[3];
    auto grad_bias = saved[4];
    // auto grad_weight = weight.grad().defined() ? weight.grad() : torch::empty(weight.sizes(), weight.options());
    // auto grad_bias = bias.grad().defined() ? bias.grad() : torch::empty(bias.sizes(), bias.options());

    if (grad_output[0].device().type() == DeviceType::CUDA) {
      // auto input_size = input.sizes();
      // const auto input_dim_size = c10::multiply_integers(input_size.begin(), input_size.end());
      // Tensor grad_input = torch::empty({input_dim_size}, input.options());
      // if (ctx->needs_input_grad(0)) {
      //   auto grad_input_ = at::matmul(grad_output[0], weight.t());
      //   grad_input.set_(grad_input_);
      // }

      // if (ctx->needs_input_grad(1) && ctx->needs_input_grad(2)) {
      //   fused_linear_bias_backward(grad_bias, grad_weight, grad_output[0].flatten(0, -2).t(), input.flatten(0, -2));
      // }
      // if (grad_output[0].dim() == 2) {
      //   fused_linear_bias_backward(grad_bias, grad_weight, grad_output[0].t(), input);
      // } else {
      //   fused_linear_bias_backward(grad_bias, grad_weight, grad_output[0].flatten(0, -2).t(), input.flatten(0, -2));
      // }
      // grad_bias = grad_output[0].sum(0);
      // std::cout << __FILE__ << ":" << __LINE__ << " grad_bias shape:" << grad_bias.sizes() << std::endl;
      fused_linear_bias_backward(grad_bias, grad_weight, grad_output[0], input);
      auto grad_input = at::matmul(grad_output[0], weight);
      return {
        grad_input,
        grad_weight,
        grad_bias,
        torch::autograd::Variable(),
        torch::autograd::Variable()};
    }
    else  {
      TORCH_CHECK(0, "Unsupported device ", grad_output[0].device().type());
    }
  }
};

Tensor fused_linear_bias_forward_autograd(
  const Tensor& input,
  const Tensor& weight,
  const Tensor& bias) {
  
  // auto has_grad_weight = grad_weight_opt.has_value();
  // const at::Tensor & grad_weight = has_grad_weight ? *c10::MaybeOwned<Tensor>::borrowed(*grad_weight_opt) : torch::empty(weight.sizes(), weight.options());
  // auto has_grad_bias = grad_bias_opt.has_value();
  // const at::Tensor & grad_bias = has_grad_bias ? *c10::MaybeOwned<Tensor>::borrowed(*grad_bias_opt) : torch::empty(bias.sizes(), bias.options());
  const at::Tensor & grad_weight = torch::empty(weight.sizes(), weight.options());
  const at::Tensor & grad_bias = torch::empty(bias.sizes(), bias.options());

  auto result = fusedLinearBiasFunction::apply(input, weight, bias, grad_weight, grad_bias);

  return result[0];
}

} // namespace native
} //namespace at


