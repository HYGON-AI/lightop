#include "lmslimquant_native/quant_kernels/base/base.cuh"


using namespace std;


inline rocblas_status rocblasGemmEx(rocblas_handle handle,
                                    rocblas_operation transa,
                                    rocblas_operation transb,
                                    const int m,const int n,const int k,
                                    const int32_t alpha,
                                    const int8_t* A, int lda,
                                    const int8_t* B, int ldb,
                                    const int32_t beta,
                                    int32_t* C, int ldc)
{
  // accumulating in int8
  return rocblas_gemm_ex(handle,
                         transa,
                         transb,
                         m, n, k,
                         &alpha,
                         A, rocblas_datatype_i8_r, lda,
                         B, rocblas_datatype_i8_r, ldb,
                         &beta,
                         C, rocblas_datatype_i32_r, ldc,
                         C, rocblas_datatype_i32_r, ldc,
                         rocblas_datatype_i32_r,
                         rocblas_gemm_algo_standard, 0,0);
}

void gemm_rocblas_int8_gpu(hipStream_t stream,
                            rocblas_handle handle,
                                    int32_t* output,
                                    const int8_t* input,
                                    const int8_t* weight,
                                    const int M,
                                    const int N,
                                    const int K,
                                    bool layout_tn);
