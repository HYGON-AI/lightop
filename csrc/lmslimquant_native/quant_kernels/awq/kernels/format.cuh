#include "lmslimquant_native/quant_kernels/base/base.cuh"


using namespace std;

//以下是kernel函数中的头文件

void dequant_w4_gemm(hipStream_t stream,
                        half* output,
                        const uint32_t* weight,
                        const half2* zeros_and_scales,
                        int k,
                        int n,
                        int group_size);
void addFusedSiluActivation(hipStream_t stream,
                              half* output,
                              const half* src,
                              int m,
                              int n,
                              int type);
void dequant_w4_gemm_colmajor(hipStream_t stream,
                                half* output,
                                const uint32_t* weight,
                                const half2* zeros_and_scales,int64_t k, int64_t n, int64_t group_size);

template <typename T>
void input_padding(hipStream_t stream, T* output,const T* input,int m,int k,int group_size,int pad_groupcount);




void launch_dequant_gemm_rocblas_gpu(hipStream_t stream,
                                void* output,
                                const void* input,
                                const void* zeros_and_scales,
                                const void* weight,
                                const int M,
                                const int N,
                                const int K,
                                void* dequant_weight,
                                int group_size);


inline rocblas_status rocblasGemmEx(rocblas_handle handle,
                                    rocblas_operation transa,
                                    rocblas_operation transb,
                                    int m, int n, int k,
                                    const float alpha,
                                    const half* A, int lda,
                                    const half* B, int ldb,
                                    const float beta,
                                    half* C, int ldc)
{
  // accumulating in FP32
  return rocblas_gemm_ex(handle,
                         transa,
                         transb,
                         m, n, k,
                         &alpha,
                         A, rocblas_datatype_f16_r, lda,
                         B, rocblas_datatype_f16_r, ldb,
                         &beta,
                         C, rocblas_datatype_f16_r, ldc,
                         C, rocblas_datatype_f16_r, ldc,
                         rocblas_datatype_f32_r,
                         rocblas_gemm_algo_standard, 0,0);
}



template <typename T>
void PrintData(hipStream_t stream, const T* input,int size);

void dequant_gemm_rocblas_gpu(hipStream_t stream,
                                    half* output,
                                    const half* input,
                                    const half2* zeros_and_scales,
                                    const uint32_t* weight,
                                    const int M,
                                    const int N,
                                    const int K,
                                    half* dequant_weight,
                                    int group_size);


void convert_s4_k_m8_(uint32_t*       A_dst,
                     half2*          Q_dst,
                     half*           workspace,
                     const uint32_t* A_src,
                     const half*     scales,
                     const uint32_t* qzeros,
                     int             m,
                     int             k,
                     int             group_size,
                     hipStream_t      stream
                     );


template <typename T>
void permute_trans(T* _sz, T* sz,int64_t dim0,int64_t n,hipStream_t stream);

void sgemm_rocblas_fp16_gpu(hipStream_t stream,
                            rocblas_handle handle,
                                    half* output,
                                    const half* input,
                                    const half* weight,
                                    const int M,
                                    const int N,
                                    const int K,
                                    bool layout_tn);
