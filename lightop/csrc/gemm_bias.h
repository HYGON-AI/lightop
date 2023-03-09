#ifndef GEMMBIAS_H
#define GEMMBIAS_H

#include <string>
#include <dlfcn.h>
#include <mutex>
#include "assert.h"
#include <libgen.h>
// #include "rocblas.h"
using namespace std;
// string GetModuleCurPath()
// {
//     Dl_info dl_info;
//     dladdr((void*)GetModuleCurPath, &dl_info);
//     string path(dl_info.dli_fname);
//     return path.substr(0, path.find_last_of('/') );
// }
// void get_dll_handle()
// {
//   string dll=GetModuleCurPath()+"/lib/libgemmbias.so";
//   *handle = dlopen(dll.c_str(), RTLD_LAZY);
//   if(!handle){
//     printf("ERROR, Message(%s).", dlerror());
//     *handle = nullptr;
//   }
// }

// void* get_gemmbias_forward_dll_handle() {
//   string dll=GetModuleCurPath()+"/lib/libhytsf.so";
//   void * handle = dlopen(dll.c_str(), RTLD_LAZY);
//   if(!handle){
//     printf("ERROR, Message(%s).", dlerror());
//     *handle = nullptr;
//   }
//   assert( handle );
//   return handle;
// }

// void* get_gemmbias_backward_dll_handle() {
//   string dll=GetModuleCurPath()+"/lib/libhytsb.so";
//   void * handle = dlopen(dll.c_str(), RTLD_LAZY);
//   if(!handle){
//     printf("ERROR, Message(%s).", dlerror());
//     handle = nullptr;
//   }
//   assert( handle );
//   return handle;
// }

std::mutex fmu;
std::mutex bmu;
#ifdef __cplusplus 
extern "C" {
#endif

// void * g_handle=get_dll_handle();

// typedef int (*_gemm_bias)(int transA,int transB,void *A,void *B,void *C,void *D,int m,int k,int n,void *alpha,void *beta);
// _gemm_bias gemm_bias=(_gemm_bias)dlsym(g_handle, "gemm_bias");
	
// typedef int (*_gemmbias_nn_m1024_k1024)(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
// _gemmbias_nn_m1024_k1024 gemmbias_nn_m1024_k1024=(_gemmbias_nn_m1024_k1024)dlsym(g_handle, "gemmbias_nn_m1024_k1024");
	
// typedef int (*_gemmbias_nn_m2_k1024)(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
// _gemmbias_nn_m2_k1024 gemmbias_nn_m2_k1024=(_gemmbias_nn_m2_k1024)dlsym(g_handle, "gemmbias_nn_m2_k1024");
	
// typedef int (*_gemmbias_tn_m1024_k4096)(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
// _gemmbias_tn_m1024_k4096 gemmbias_tn_m1024_k4096=(_gemmbias_tn_m1024_k4096)dlsym(g_handle, "gemmbias_tn_m1024_k4096");
	
// typedef int (*_gemmbias_tn_m30528_k1024)(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
// _gemmbias_tn_m30528_k1024 gemmbias_tn_m30528_k1024=(_gemmbias_tn_m30528_k1024)dlsym(g_handle, "gemmbias_tn_m30528_k1024");
	
// typedef int (*_gemmbias_tn_m1024_k1024)(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
// _gemmbias_tn_m1024_k1024 gemmbias_tn_m1024_k1024=(_gemmbias_tn_m1024_k1024)dlsym(g_handle, "gemmbias_tn_m1024_k1024");
	
// typedef int (*_gemmbias_tn_m4096_k1024)(void *A,void *B,void *C,void *D,void* alpha,void* beta,int m,int k,int n);
// _gemmbias_tn_m4096_k1024 gemmbias_tn_m4096_k1024=(_gemmbias_tn_m4096_k1024)dlsym(g_handle, "gemmbias_tn_m4096_k1024");

// typedef int (*_gemm_dbias)(int transA,int transB,void *A,void *B,void *C,void *D,int m,int k,int n,void *alpha);
// _gemm_dbias gemm_dbias=(_gemm_dbias)dlsym(g_handle, "gemm_dbias");
	
// typedef int (*_gemmdbias_nt_m1024_n1024)(void *A,void *B,void *C,void *D,void* alpha,int m,int k,int n);
// _gemmdbias_nt_m1024_n1024 gemmdbias_nt_m1024_n1024=(_gemmdbias_nt_m1024_n1024)dlsym(g_handle, "gemmdbias_nt_m1024_n1024");

typedef struct  {
  void * handler;
  int num;
} gemm_hanndle;

gemm_hanndle forwardHandler = {nullptr, 0};

gemm_hanndle backwardHandler = {nullptr, 0};

int (*gemmbias_forward_create_handle)(void **handle);
int (*gemmbias_forward_destroy_handle)(void *handle);
int (*gemmbias_forward_half)(void *handle,int trans_a,
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


int (*gemmdbias_backward_create_handle)(void **handle);
int (*gemmdbias_backward_destroy_handle)(void *handle);
int (*gemmdbias_backward_half)(void *handle,int trans_a,
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
int initState = 0;
int initGemmbias(void) {
  if (initState) return 0;
  
  Dl_info dl_info;
  dladdr((void*)initGemmbias, &dl_info);
  char * modulePath = dirname((char*)dl_info.dli_fname);

  void* tmp;
  void** cast;

#define LOAD_SYM(handle, symbol, funcptr) do {           \
    cast = (void**)&funcptr;                             \
    tmp = dlsym(handle, symbol);                        \
    if (tmp == NULL) {                                   \
      printf("dlsym failed on %s - %s", symbol, dlerror());  \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

  char dllpath[1024];
  memset(dllpath, 0x0, sizeof dllpath);
  sprintf(dllpath, "%s/lib/libhytsf.so", modulePath);
  void * fHandle = dlopen(dllpath, RTLD_LAZY);
  if(!fHandle){
    printf("ERROR, Message(%s).", dlerror());
    fHandle = nullptr;
  }
  LOAD_SYM(fHandle, "gemmbias_forward_create_handle", gemmbias_forward_create_handle);
  LOAD_SYM(fHandle, "gemmbias_forward_destroy_handle", gemmbias_forward_destroy_handle);
  LOAD_SYM(fHandle, "gemmbias_forward_half", gemmbias_forward_half);

  assert(gemmbias_forward_create_handle);
  assert(gemmbias_forward_destroy_handle);
  assert(gemmbias_forward_half);

  
  memset(dllpath, 0x0, sizeof dllpath);
  sprintf(dllpath, "%s/lib/libhytsb.so", modulePath);
  void * bHandle = dlopen(dllpath, RTLD_LAZY);
  if(!bHandle){
    printf("ERROR, Message(%s).", dlerror());
    bHandle = nullptr;
  }
  LOAD_SYM(bHandle, "gemmdbias_backward_create_handle", gemmdbias_backward_create_handle);
  LOAD_SYM(bHandle, "gemmdbias_backward_destroy_handle", gemmdbias_backward_destroy_handle);
  LOAD_SYM(bHandle, "gemmdbias_backward_half", gemmdbias_backward_half);

  assert(gemmdbias_backward_create_handle);
  assert(gemmdbias_backward_destroy_handle);
  assert(gemmdbias_backward_half);

  initState = 1;
  return 0;
}

// rocblas_handle handle;

#ifdef __cplusplus 
}
#endif
	
#endif


