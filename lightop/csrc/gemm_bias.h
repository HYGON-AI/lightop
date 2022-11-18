#ifndef GEMMBIAS_H
#define GEMMBIAS_H

#include <string>
#include <dlfcn.h>
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
#ifdef __cplusplus 
}
#endif
	
#endif


