#ifndef HYTSF_H
#define HYTSF_H

#ifdef __cplusplus
extern "C" {
#endif

int gemmbias_forward_create_handle(void **handle);

int gemmbias_forward_destroy_handle(void* handle);

int gemmbias_forward_half(void *handle,int trans_a,
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

#ifdef __cplusplus
}
#endif

#endif
