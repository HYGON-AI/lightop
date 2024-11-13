#include "reduce.h"

namespace at {
namespace native {

#define FP16_MIN -65504
#define INVALID_INDEX 65535
#define RadixBit 2
#define RadixSize 4
#define Radix_Mask 3

#ifndef uint32_t
 #define uint32_t unsigned int
#endif

#ifndef uint16_t
 #define uint16_t unsigned short
#endif

#ifndef uint8_t
 #define uint8_t unsigned char
#endif

template <typename scalar_t>
struct TopKTypeConfig {};

template <>
struct TopKTypeConfig<float> {
  typedef uint32_t RadixType;
  static inline __device__ RadixType convert(float v) {
    RadixType x = __float_as_int(v);
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

    return (v == v) ? (x ^ mask) : 0xffffffff;
  }

  static inline __device__ float deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
    return __int_as_float(v ^ mask);
  }
};

template <>
struct TopKTypeConfig<at::Half> {
  typedef uint16_t RadixType;

  static inline __device__ RadixType convert(at::Half v) {
    RadixType x = __half_as_ushort(v);
    RadixType mask = (x & 0x00008000) ? 0x0000ffff : 0x00008000;
    return (v == v) ? (x ^ mask) : 0xffff;
  }

  static inline __device__  at::Half deconvert(RadixType v) {
    RadixType mask = (v & 0x00008000) ? 0x00008000 : 0x0000ffff;
    return __ushort_as_half(v ^ mask);
  }
};


template <typename scalar_t >
struct DataIndice{
  using Bitwise=typename TopKTypeConfig<scalar_t>::RadixType; 
  Bitwise data;
  uint16_t indice;
};

template <typename Bitwise>
__device__ inline Bitwise getBitwise(Bitwise val,int pos) {
  constexpr Bitwise RadixMask = Radix_Mask;
  return (val >> pos) & RadixMask; 
}

template <typename Bitwise>
__device__ inline Bitwise setBitwise(Bitwise val,Bitwise insertval,int pos) {
  constexpr Bitwise RadixMask = Radix_Mask;
  return (val & ~(RadixMask << pos)) | (insertval << pos);
}

__inline__ __device__ int WarpReduceSumInt(int val) {
  #pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename Bitwise,int loop=4>
__device__ void countingMaskedval(Bitwise *count,Bitwise * data,Bitwise desired,Bitwise mask,int pos,int *smem) {
  for(int i = 0;i < RadixSize;i++) {
    count[i] = 0;
  }
  if(threadIdx.y==0&&threadIdx.x < RadixSize) {
    smem[threadIdx.x] = 0;
  }
  __syncthreads();
  for(int m = 0;m < loop;m++) {
    Bitwise valbit = getBitwise(data[m], pos);
    bool hasval = ((data[m] & mask) == desired);
    for(int j = 0;j < RadixSize;j++) {
      bool vote = hasval && (valbit == j);
      count[j] += __popcll(__ballot(vote));
    }
  }

  if(threadIdx.x == 0) {
    for(int i = 0;i < RadixSize;i++) {
      atomicAdd(smem + i, count[i]);
    }
  }
  __syncthreads();
  for(int i = 0;i < RadixSize;i++) {
    count[i] = smem[i];
  }
  __syncthreads();

}

template <typename Bitwise,int loop=4>
bool __device__ RadixSelect(Bitwise* data,int *smem, int k,Bitwise& kthvalue) {
  //every thread has mask,desired,count to deal N/blockDim.x datas
  constexpr Bitwise RadixMask = Radix_Mask;
  Bitwise count[RadixSize];
  Bitwise desired = 0;
  Bitwise mask = 0;
  int ktofind = k;
  for(int pos = sizeof(Bitwise)*8 - RadixBit ;pos >=0;pos -= RadixBit) {
    countingMaskedval<Bitwise,loop>(count, data, desired, mask, pos,smem);
    auto find_unique = [&](Bitwise i, int counts) {
      if(counts == 1 && ktofind == 1) {
        desired = setBitwise(desired, i, pos);
        kthvalue = desired;
        return true;
      }
      return false;
    };
    auto find_non_unique = [&](Bitwise i, int counts) {
      if(counts >= ktofind) {
        desired = setBitwise(desired, i, pos);
        mask = setBitwise(mask, RadixMask, pos);
        return true;
      }
      ktofind -= counts;
      return false;
    };
    for(int i = RadixSize - 1;i >= 0;i--) {
      int c = count[i];
      if(find_unique(i, c)) {
        return true;
      }
      if(find_non_unique(i, c)) {
        break;
      }
    }
  }
  kthvalue=desired;
  return false;
}

template <typename T>
__device__ void radix_sort(T * ps_vector, int vec_length,T * sort_tmp_0, T * sort_tmp_1,int * counter) {
    constexpr int leap=64;
    int tid=threadIdx.x;
    using type = decltype(ps_vector->data);
    for (type bit=0; bit<sizeof(type)*8; bit++) {
        type bit_mask = (1<<bit);
        counter[0]=0;
        counter[1]=0;
        for (int i=tid; i<vec_length; i+=leap) { 
            T elem=ps_vector[i];
            if (elem.data & bit_mask) {
                int index=atomicAdd(counter+1,1);
                sort_tmp_1[index] = elem;
            }
            else{
                int index=atomicAdd(counter,1);
                sort_tmp_0[index] = elem;
            }
        }
        int base_excur_1=counter[1];
        if(base_excur_1==0||base_excur_1==vec_length)continue;
        int base_excur_0=counter[0];
        for (int i=tid; i<base_excur_1; i+=leap) {
            ps_vector[i] = sort_tmp_1[i];
        }
        for (int i=tid; i<base_excur_0; i+=leap) {
            ps_vector[base_excur_1+i] = sort_tmp_0[i];
        }
    }
}

template <typename scalar_t,int VEC,int NUMBER_WARPS>
__global__ void topk_kernel(scalar_t* x, scalar_t* ret,int* indice,int k, int cols,int stride){
  x+=blockIdx.x*cols*stride+blockIdx.y;
  ret+=blockIdx.x*k*stride+blockIdx.y;
  indice+=blockIdx.x*k*stride+blockIdx.y;
  int tidy=threadIdx.y;
  int tidx=threadIdx.x;
  using Bitwise=typename TopKTypeConfig<scalar_t>::RadixType;
  using DI = DataIndice<scalar_t>;
  extern __shared__ char smem[];
  DI* s_topk_out=(DI*)(smem);
  Bitwise* s_input=(Bitwise*)(smem);
  DI *tmp1=s_topk_out+k;
  DI *tmp2=tmp1+k;
  __shared__ int s_radix[RadixSize];//需要原子加，不能用uint16
  __shared__ int counter[2];
  Bitwise kthvalue;
  scalar_t invec[VEC];
  constexpr int SMALL_VEC=2;
  Bitwise in_SMALL_VEC[SMALL_VEC];
  if(tidx==0&&tidy==0)counter[0]=0;
  Bitwise *p_invec=(Bitwise *)invec;
  //全部数据加载到vgpr
  int tid_in_block=threadIdx.x+threadIdx.y*64;
  for(int i = 0;i < VEC;i++) {
    int offset=NUMBER_WARPS*64*i+tid_in_block;
    if(offset>=cols)p_invec[i]=0;
    else {
      invec[i]=x[offset*stride];
      p_invec[i]=TopKTypeConfig<scalar_t>::convert(invec[i]);
    }
  }
  bool find_unique;
  if constexpr(NUMBER_WARPS>=8){
    int k2=k/2;
    if(k2<10)k2=k;
    if(cols<=10000)RadixSelect<Bitwise,2>(p_invec,s_radix,k2,kthvalue);
    else RadixSelect<Bitwise,4>(p_invec,s_radix,k2,kthvalue);
    for(int i=0;i<VEC;i++){
      if(p_invec[i]>kthvalue){
        int index = atomicAdd(counter,1);
        s_input[index]=p_invec[i];
      }
    }
    __syncthreads();
    int count1=counter[0];
    for(int i = 0;i < SMALL_VEC;i++) {
      int offset=NUMBER_WARPS*64*i+tid_in_block;
      if(offset>=count1)in_SMALL_VEC[i]=0;
      else in_SMALL_VEC[i]=s_input[offset];
    }
    find_unique=RadixSelect<Bitwise,SMALL_VEC>(in_SMALL_VEC,s_radix,k,kthvalue);
  }else{
    find_unique=RadixSelect<Bitwise,VEC>(p_invec,s_radix,k,kthvalue);
  }
  if(tidx==0&&tidy==0)counter[0]=0;
  __syncthreads();
  if(find_unique==true){
   for(int i = 0;i < VEC;i++) {
      Bitwise t_topk=p_invec[i];
      if(t_topk>=kthvalue){
        int index = atomicAdd(counter,1);
        if(index>=k)break;
        s_topk_out[index].data=t_topk;
        s_topk_out[index].indice=NUMBER_WARPS*64*i+tid_in_block;
      }
    }
  }
  else{
    for(int i = 0;i < VEC;i++) {
      Bitwise t_topk=p_invec[i];
      if(t_topk>kthvalue){
        int index = atomicAdd(counter,1);
        s_topk_out[index].data=t_topk;
        s_topk_out[index].indice=NUMBER_WARPS*64*i+tid_in_block;
      }
    }
    __syncthreads();
    for(int i = 0;i < VEC;i++) {
      Bitwise t_topk=p_invec[i];
      if(t_topk==kthvalue){
        int index = atomicAdd(counter,1);
        if(index>=k)break;
        s_topk_out[index].data=t_topk;
        s_topk_out[index].indice=NUMBER_WARPS*64*i+tid_in_block;
      }
    }
  }
  __syncthreads();
  if(tidy==0){
    if(k>1)radix_sort(s_topk_out,k,tmp1,tmp2,counter);
    for(int i=tidx;i<k;i++){
      ret[i*stride]=TopKTypeConfig<scalar_t>::deconvert(s_topk_out[i].data);
      indice[i*stride]=s_topk_out[i].indice;
    }
  }
}

#define VEC_SWITCH(n, ...)              \
  [&] {                                         \
    if(n<=64){\
      constexpr int NUMBER_WARPS=1;\
      constexpr int VEC = 1;\
      return __VA_ARGS__();\
    }\
    else if(n<=128){\
      constexpr int NUMBER_WARPS=2;\
      constexpr int VEC = 1;\
      return __VA_ARGS__();\
    }\
    else if(n<=256){\
      constexpr int NUMBER_WARPS=2;\
      constexpr int VEC = 2;\
      return __VA_ARGS__();\
    }\
    else if(n<=512){\
      constexpr int NUMBER_WARPS=2;\
      constexpr int VEC = 4;\
      return __VA_ARGS__();\
    }\
    else if(n<=1024){\
      constexpr int NUMBER_WARPS=2;\
      constexpr int VEC = 8;\
      return __VA_ARGS__();\
    }\
    else if(n<=2048){\
      constexpr int NUMBER_WARPS=4;\
      constexpr int VEC = 8;\
      return __VA_ARGS__();\
    }\
    else if (n <=4096 ) {                          \
      constexpr int NUMBER_WARPS=8;\
      constexpr int VEC = 8;             \
      return __VA_ARGS__();                     \
    }else if (n <=10000 ) {                          \
      constexpr int NUMBER_WARPS=8;\
      constexpr int VEC = 20;             \
      return __VA_ARGS__();                     \
    }else if (n <=15000 ) {                          \
      constexpr int NUMBER_WARPS=8;\
      constexpr int VEC = 30;             \
      return __VA_ARGS__();                     \
    } else if(n <= 16384){                                    \
      constexpr int NUMBER_WARPS=8;\
      constexpr int VEC = 32;             \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr int NUMBER_WARPS=16;\
      constexpr int VEC = 32;             \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define TYPE_SWITCH(...)[&]{ \
  if constexpr (std::is_same<scalar_t, at::BFloat16>::value||std::is_same<scalar_t, at::Half>::value){\
    constexpr int lds_size=4096;\
    using T = at::Half;\
    return __VA_ARGS__();                     \
  }\
  else{\
    constexpr int lds_size=8192;\
    using T = float;\
    return __VA_ARGS__();                     \
  }\
}()

static inline int get_env_(const char *env_var,int default_value=0) {
  if (char *value = std::getenv(env_var)) {
    return atoi(value);
  }
  return default_value;
}


std::tuple<Tensor,Tensor> topk_opt(const Tensor& x,int k,int dim){
  std::vector<long> size(x.dim());
  int cols=x.sizes()[dim];
  int stride=1;
  int batch=1;
  for(int i=0;i<x.dim();i++){
    if(i==dim)size[i]=k;
    else{
      size[i] = x.sizes()[i];
      if(i<dim)batch*=size[i];
      else stride*=size[i];
    }
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  Tensor ret=at::empty(size,x.options());
  c10::TensorOptions options(DeviceType::CUDA);
  Tensor indice=at::empty(size,options.dtype(ScalarType::Int));
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16, 
    x.scalar_type(),
    "topk_opt",
    [&] {
    scalar_t* x_data = x.data_ptr<scalar_t>();
    scalar_t* ret_data=ret.data_ptr<scalar_t>();
    int *indice_data=indice.data_ptr<int>();
    VEC_SWITCH(cols,[&] {
      TYPE_SWITCH([&]{
        topk_kernel<T,VEC,NUMBER_WARPS><<<dim3(batch,stride),dim3(64,NUMBER_WARPS),lds_size,stream>>>((T*)x_data,(T*)ret_data,indice_data,k,cols,stride);
      });
    });
  });
  return std::tuple<Tensor,Tensor>(ret,indice);
}

}
}