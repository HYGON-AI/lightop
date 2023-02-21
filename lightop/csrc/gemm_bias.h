#ifndef GEMMBIAS_H
#define GEMMBIAS_H

#include <string>
#include <dlfcn.h>
#include <mutex>
// #include "rocblas.h"
using namespace std;
string GetModuleCurPath()
{
    Dl_info dl_info;
    dladdr((void*)GetModuleCurPath, &dl_info);
    string path(dl_info.dli_fname);
    return path.substr(0, path.find_last_of('/') );
}
void * get_dll_handle()
{
  string dll=GetModuleCurPath()+"/lib/libgemmbias.so";
  void* handle = dlopen(dll.c_str(), RTLD_LAZY);
  if(!handle){
     printf("ERROR, Message(%s).", dlerror());
  }
  return handle;
}

void * get_gemmbias_forward_dll_handle() {
  string dll=GetModuleCurPath()+"/lib/libhytsf.so";
  void* handle = dlopen(dll.c_str(), RTLD_LAZY);
  if(!handle){
     printf("ERROR, Message(%s).", dlerror());
  }
  return handle;
}
#ifdef __cplusplus 
extern "C" {
#endif

void * g_handle=get_dll_handle();

typedef int (*_gemm_bias)(int transA,int transB,void *A,void *B,void *C,void *D,int m,int k,int n,void *alpha,void *beta);
_gemm_bias gemm_bias=(_gemm_bias)dlsym(g_handle, "gemm_bias");
	
typedef int (*_gemmbias_nn_m1024_k1024)(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
_gemmbias_nn_m1024_k1024 gemmbias_nn_m1024_k1024=(_gemmbias_nn_m1024_k1024)dlsym(g_handle, "gemmbias_nn_m1024_k1024");
	
typedef int (*_gemmbias_nn_m2_k1024)(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
_gemmbias_nn_m2_k1024 gemmbias_nn_m2_k1024=(_gemmbias_nn_m2_k1024)dlsym(g_handle, "gemmbias_nn_m2_k1024");
	
typedef int (*_gemmbias_tn_m1024_k4096)(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
_gemmbias_tn_m1024_k4096 gemmbias_tn_m1024_k4096=(_gemmbias_tn_m1024_k4096)dlsym(g_handle, "gemmbias_tn_m1024_k4096");
	
typedef int (*_gemmbias_tn_m30528_k1024)(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
_gemmbias_tn_m30528_k1024 gemmbias_tn_m30528_k1024=(_gemmbias_tn_m30528_k1024)dlsym(g_handle, "gemmbias_tn_m30528_k1024");
	
typedef int (*_gemmbias_tn_m1024_k1024)(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
_gemmbias_tn_m1024_k1024 gemmbias_tn_m1024_k1024=(_gemmbias_tn_m1024_k1024)dlsym(g_handle, "gemmbias_tn_m1024_k1024");
	
typedef int (*_gemmbias_tn_m4096_k1024)(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
_gemmbias_tn_m4096_k1024 gemmbias_tn_m4096_k1024=(_gemmbias_tn_m4096_k1024)dlsym(g_handle, "gemmbias_tn_m4096_k1024");

typedef int (*_gemm_dbias)(int transA,int transB,void *A,void *B,void *C,void *D,int m,int k,int n,void *alpha);
_gemm_dbias gemm_dbias=(_gemm_dbias)dlsym(g_handle, "gemm_dbias");
	
typedef int (*_gemmdbias_nt_m1024_n1024)(void *A,void *B,void *C,void *D,void* alpha,int m,int k,int n);
_gemmdbias_nt_m1024_n1024 gemmdbias_nt_m1024_n1024=(_gemmdbias_nt_m1024_n1024)dlsym(g_handle, "gemmdbias_nt_m1024_n1024");

void * g_dll_handle=get_gemmbias_forward_dll_handle();

void * rhandle = nullptr;

int rhandle_num = 0;

mutex mu;

typedef int (*_gemmbias_forward_create_handle)(void **handle);
_gemmbias_forward_create_handle gemmbias_forward_create_handle = (_gemmbias_forward_create_handle)dlsym(g_dll_handle, "gemmbias_forward_create_handle");

typedef int (*_gemmbias_forward_destroy_handle)(void *handle);
_gemmbias_forward_destroy_handle gemmbias_forward_destroy_handle = (_gemmbias_forward_destroy_handle)dlsym(g_dll_handle, "gemmbias_forward_destroy_handle");

typedef int (*_gemmbias_forward_half)(void *handle,int trans_a,
                                                  int trans_b,
                                                  int       m,
                                                  int       n,
                                                  int       k,
                                                  const void*       alpha,
                                                  const void*       a,
                                                  const void*       b,
                                                  const void*       beta,
                                                  const void*       c,
                                                  void*             d);
_gemmbias_forward_half gemmbias_forward_half = (_gemmbias_forward_half)dlsym(g_dll_handle, "gemmbias_forward_half");


// rocblas_handle handle;

#ifdef __cplusplus 
}
#endif
	
#endif


