#ifndef GEMMBIAS_H
#define GEMMBIAS_H

#ifdef __cplusplus 
extern "C" {
#endif
int gemm_bias(int transA,int transB,void *A,void *B,void *C,void *D,int m,int k,int n,void *alpha,void *beta);
int gemmbias_nn_m1024_k1024(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
int gemmbias_nn_m2_k1024(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
int gemmbias_tn_m1024_k4096(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
int gemmbias_tn_m30528_k1024(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
int gemmbias_tn_m1024_k1024(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
int gemmbias_tn_m4096_k1024(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);

int gemm_dbias(int transA,int transB,void *A,void *B,void *C,void *D,int m,int k,int n,void *alpha);
int gemmdbias_nt_m1024_n1024(void *A,void *B,void *C,void *D,void* alpha,int m,int k,int n);
#ifdef __cplusplus 
}
#endif
	
#endif
