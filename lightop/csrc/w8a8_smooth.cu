#include "reduce.h"
#include "hip/hip_bf16.h"
#include <sys/syscall.h>
#include <unistd.h>
namespace at {
namespace native {


#ifndef uint32_t
 #define uint32_t unsigned int
#endif
#define fp8 uint8_t
using bhalf_t = __hip_bfloat16;
using half2_t = __attribute__( (__vector_size__(2 * sizeof(_Float16)) )) _Float16;
using half4_t = __attribute__( (__vector_size__(4 * sizeof(_Float16)) )) _Float16;
using bf16x4_t = __attribute__( (__vector_size__(4 * sizeof(uint16_t)) )) uint16_t;
using half8_t = __attribute__( (__vector_size__(8 * sizeof(_Float16)) )) _Float16;
using half16_t = __attribute__( (__vector_size__(16 * sizeof(_Float16)) )) _Float16;
using v4bh = __attribute__( (__vector_size__(4 * sizeof(short)) )) short;
using floatx4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
using uintx4 = __attribute__( (__vector_size__(4 * sizeof(uint32_t)) )) uint32_t;
using fp8x16 = __attribute__( (__vector_size__(16 * sizeof(fp8)) )) fp8;
using intx4 = __attribute__( (__vector_size__(4 * sizeof(int32_t)) )) int32_t;
using int8x8 = __attribute__( (__vector_size__(8 * sizeof(int8_t)) )) int8_t;
using int8x16 = __attribute__( (__vector_size__(16 * sizeof(int8_t)) )) int8_t;

inline __device__ float fp8e4m3_to_fp32(const fp8& input) {

  const uint32_t w = (uint32_t)input << 24;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  uint32_t renorm_shift = __clz(nonsign);
  renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;
  uint32_t result = sign | ((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23));
  return c10::detail::fp32_from_bits(result);
}

template<typename scalar_t> 
static __device__ inline scalar_t f32_to_f16(float f){
  if constexpr(std::is_same<scalar_t, _Float16>::value){
    return f;
  }
  else{
    uint32_t u = *(uint32_t*)(&f);
    // if(~u&0x7f800000){
        u += 0x7fff + ((u >> 16) & 1); 
    // }
    return u>>16;
  }
}


template<typename scalar_t> 
static __device__ inline scalar_t f32_to_f16_add(float f, scalar_t bias){
  if constexpr(std::is_same<scalar_t, _Float16>::value){
    return (_Float16)f + bias;
  }
  else{
    uint32_t u = *(uint32_t*)(&f);
    // if(~u&0x7f800000){
        u += 0x7fff + ((u >> 16) & 1); 
    // }
    uint16_t f16_result = u>>16;
    // 将 uint16_t 转换为 bfloat16 进行加法运算，然后转换回 uint16_t
    bhalf_t bf16_result = *(bhalf_t*)(&f16_result);
    bhalf_t bf16_bias = *(bhalf_t*)(&bias);
    bhalf_t bf16_sum = bf16_result + bf16_bias;
    return *(uint16_t*)(&bf16_sum);
  }
}


struct half4x2{
  half4_t data[2];
};

struct half4x4{
  half4_t data[4];
};

template<int tc_number>
struct int8ABVector{
  union{
    int8x8 data[tc_number];
    int8x16 data_16[tc_number/2];
  };
};

inline __device__ void builtin_amdgcn_mmac(const int8x8& reg_a, const int8x8& reg_b, intx4& reg_c)
{
    #if defined(__gfx936__) || defined(__gfx928__)
    auto a = reinterpret_cast<const long*>(&reg_a);
    auto b = reinterpret_cast<const long*>(&reg_b);
    reg_c=__builtin_amdgcn_mmac_i32_16x16x32i8(*a, *b, reg_c);
    #endif
}

#define FastCompile 0
#if FastCompile
#define NUM_WARPS_SWITCH(warps, ...)          \
  [&] {                                       \
    constexpr static int NumWarps = 2;        \
    return __VA_ARGS__();                     \
  }()


#define NPerBlock_SWITCH(n_per_block, ...)    \
  [&] {                                       \
    constexpr static int NPerBlock = 32;      \
    return __VA_ARGS__();                     \
  }()
#define KPerWarp_SWITCH(k_per_warp, ...)    \
  [&] {                                       \
    constexpr static int KPerWarp = 256;      \
    return __VA_ARGS__();                     \
  }()

#define Output_Type_SWITCH(DST_DTYPE, ...)     \
  [&] {                                       \
    using scalar_t = _Float16;                \
    return __VA_ARGS__();                     \
  }()
#define Bias_Type_SWITCH(use_bias, ...)          \
  [&] {                                       \
    constexpr static bool UseBias = false;      \
    return __VA_ARGS__();                   \
  }()

#else
#define NUM_WARPS_SWITCH(warps, ...)          \
  [&] {                                       \
    if (warps == 1) {                         \
      constexpr static int NumWarps = 1;      \
      return __VA_ARGS__();                   \
    } else if (warps == 2) {                  \
      constexpr static int NumWarps = 2;      \
      return __VA_ARGS__();                   \
    } else if (warps == 4) {                  \
      constexpr static int NumWarps = 4;      \
      return __VA_ARGS__();                   \
    } else if (warps == 8) {                  \
      constexpr static int NumWarps = 8;      \
      return __VA_ARGS__();                   \
    }else {                                   \
      constexpr static int NumWarps = 16;     \
      return __VA_ARGS__();                   \
    }                                         \
  }()


#define NPerBlock_SWITCH(n_per_block, ...)    \
  [&] {                                       \
    if (n_per_block == 16) {                  \
      constexpr static int NPerBlock = 16;    \
      return __VA_ARGS__();                   \
    }else if(n_per_block == 32){              \
      constexpr static int NPerBlock = 32;    \
      return __VA_ARGS__();                   \
    }else {                                   \
      constexpr static int NPerBlock = 64;    \
      return __VA_ARGS__();                   \
    }                                         \
  }()
#define KPerWarp_SWITCH(k_per_warp, ...)    \
  [&] {                                       \
    if (k_per_warp == 32) {                  \
      constexpr static int KPerWarp = 32;    \
      return __VA_ARGS__();                   \
    }else if(k_per_warp == 64){              \
      constexpr static int KPerWarp = 64;    \
      return __VA_ARGS__();                   \
    }else if(k_per_warp == 128){              \
      constexpr static int KPerWarp = 128;    \
      return __VA_ARGS__();                   \
    }else {                                  \
      constexpr static int KPerWarp = 256;    \
      return __VA_ARGS__();                   \
    }                                       \
  }()





#define Output_Type_SWITCH(DST_DTYPE, ...)     \
  [&] {                                       \
    if (DST_DTYPE == at::ScalarType::Half) {  \
      using scalar_t=_Float16;                \
      return __VA_ARGS__();                   \
    }else {                                   \
      using scalar_t = uint16_t;          \
      return __VA_ARGS__();                   \
    }                                        \
  }()
#define Bias_Type_SWITCH(use_bias, ...)          \
  [&] {                                       \
    if (use_bias) {                         \
      constexpr static bool UseBias = true;      \
      return __VA_ARGS__();                   \
    } else {                  \
      constexpr static bool UseBias = false;      \
      return __VA_ARGS__();                   \
    }                                        \
  }()
#endif


template <typename scalar_t, int NUM_WARPS=4,int NPerBlock=16,int KPerWarp=64,bool UseBias=false>
__global__ void gemm_w8_a8_smooth_kernel_colmajor(int8_t *a, int8_t *b, scalar_t* d,float* scale_a, float* scale_b, scalar_t* bias, int M,int N,int K)
{
    constexpr int WarpK=KPerWarp;
    constexpr int tc_number = WarpK/32;
    constexpr int k_per_thread = tc_number * 8;
    constexpr int NSizePerWarp=16;
    constexpr int NLoop=NPerBlock/NSizePerWarp;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;
    const int row_id = lane%NSizePerWarp;  //不用改，warp内各个线程的layout与fp16的tensorcore一致，只是thread持有的元素数量变多了
    const int col_id = lane/NSizePerWarp;    
    const bool row_id_valid = row_id < M;
    intx4 d_vec[NLoop];
    for(int i=0;i<NLoop;i++){
        d_vec[i]={0,0,0,0};
    }
    d+=      NPerBlock*bid;
    b+=      NPerBlock*bid*K;
    scale_b += NPerBlock*bid;
    bias += NPerBlock*bid;

    const int N_lenth = N - NPerBlock*bid;
    
    using VecType = int8ABVector<tc_number>;
    int n_idx_valid[NLoop];
    for(int nn = 0; nn < NLoop; nn++) {
      int n_idx = NSizePerWarp * nn + row_id;
      if(n_idx < N_lenth){
        n_idx_valid[nn] = n_idx * K;
      } else {
        n_idx_valid[nn] = -1;
      }
    }



    for(int i=warp_idx*WarpK;i<K;i+=WarpK*NUM_WARPS){
      const int k_idx = i + col_id * k_per_thread;
      VecType a_vec;
      if(row_id_valid) {
        a_vec = *(VecType*)(a + row_id * K + k_idx);
      }
      

      for(int nn = 0; nn < NLoop; nn++) {

        VecType b_vec;
        if(n_idx_valid[nn] != -1){
          b_vec = *(VecType*)(b + n_idx_valid[nn] + k_idx);
        }
        #pragma unroll
        for(int ii = 0; ii < tc_number; ++ii)
        {
          builtin_amdgcn_mmac(a_vec.data[ii], b_vec.data[ii], d_vec[nn]);
        }
      }
    }

    extern __shared__ intx4 out_smem[];
    int* out_smem_ptr = reinterpret_cast<int*>(out_smem);


    if constexpr(NUM_WARPS>1){
        int lds_size_n_offset=NUM_WARPS/2*16*M/4;
          #pragma unroll
          for (int i = NUM_WARPS; i > 1; i /= 2) {
              int mid = i / 2;
              if(row_id_valid)
              {
                if (warp_idx >= mid && warp_idx < i) {
                    for(int nn=0;nn<NLoop;nn++){
                        out_smem[(warp_idx - mid) * 4 * M+row_id*4+col_id+lds_size_n_offset*nn]=d_vec[nn];
                    }
                }
              }
              __syncthreads();
              if (warp_idx < mid) {
                  for(int nn=0;nn<NLoop;nn++){
                    if(row_id_valid)
                    {
                      intx4 tmp=out_smem[warp_idx* 4 * M+row_id*4+col_id+lds_size_n_offset*nn];
                      #pragma unroll
                      for (int i = 0; i < 4; i++) {
                          d_vec[nn][i] += tmp[i];
                      }
                    }
                  }
              }
              __syncthreads();
          }
    }


    using out_vec_dtype = typename std::conditional<std::is_same<scalar_t, _Float16>::value, half4_t, bf16x4_t>::type;
    if(warp_idx == 0) {

      if (row_id_valid) { 
        #pragma unroll
        for(int nn = 0; nn < NLoop; nn++) {
            int out_smem_offset = nn*NSizePerWarp*M;
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                out_smem_ptr[out_smem_offset + row_id * NSizePerWarp + i*4 + col_id] = d_vec[nn][i];
            }
        }
      }
      __syncthreads();
      
      if(row_id_valid){
        const float scale_a_value = scale_a[row_id];
        for(int nn = 0; nn < NLoop; nn++) {
          int out_smem_offset = nn*NSizePerWarp*M;
      
          bool n_idx_valid = nn*NSizePerWarp+col_id * 4 < N_lenth;
          if(n_idx_valid)
          {

            out_vec_dtype out_vec_16bit;
            intx4 out_vec_int = *reinterpret_cast<intx4*>(out_smem_ptr + out_smem_offset + row_id * NSizePerWarp + col_id * 4);
            floatx4 scale_b_vec = *reinterpret_cast<floatx4*>(scale_b + nn * NSizePerWarp + col_id * 4);
            #pragma unroll
            for(int i = 0; i < 4; i++) {
              scale_b_vec[i] = scale_a_value * scale_b_vec[i];
            }
            if constexpr(UseBias)
            {
              out_vec_16bit = *reinterpret_cast<out_vec_dtype*>(bias + nn * NSizePerWarp + col_id * 4);
              #pragma unroll
              for(int i = 0; i < 4; i++) {


                out_vec_16bit[i] = f32_to_f16_add<scalar_t>(out_vec_int[i] * scale_b_vec[i], out_vec_16bit[i]);
              }
            }
            else{
              #pragma unroll
              for(int i = 0; i < 4; i++) {
                out_vec_16bit[i] = f32_to_f16<scalar_t>(out_vec_int[i] * scale_b_vec[i]);
              }
            }
            *reinterpret_cast<out_vec_dtype*>(d + row_id * N + nn * NSizePerWarp + col_id * 4) = out_vec_16bit;
          }
        }
      }

    }

}

template <typename scalar_t, int NUM_WARPS=4,int NPerBlock=16,int KPerWarp=64,bool UseBias=false>
__global__ void gemm_w8_a8_smooth_kernel_colmajor_NWarps(int8_t *a, int8_t *b, scalar_t* d,float* scale_a, float* scale_b, scalar_t* bias, int M,int N,int K)
{
    constexpr int WarpK=KPerWarp;
    constexpr int tc_number = WarpK/32;
    constexpr int k_per_thread = tc_number * 8;
    constexpr int NSizePerWarp=16;
    constexpr int NLoop=NPerBlock/NSizePerWarp;
    const int tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int bid = blockIdx.x*NUM_WARPS + warp_idx;
    const int lane = tid % C10_WARP_SIZE;
    const int row_id = lane%NSizePerWarp;  //不用改，warp内各个线程的layout与fp16的tensorcore一致，只是thread持有的元素数量变多了
    const int col_id = lane/NSizePerWarp;    
    const bool row_id_valid = row_id < M;
    intx4 d_vec[NLoop];
    for(int i=0;i<NLoop;i++){
        d_vec[i]={0,0,0,0};
    }
    d+=      NPerBlock*bid;
    b+=      NPerBlock*bid*K;
    scale_b += NPerBlock*bid;
    bias += NPerBlock*bid;

    const int N_lenth = N - NPerBlock*bid;
    
    using VecType = int8ABVector<tc_number>;
    int n_idx_valid[NLoop];
    for(int nn = 0; nn < NLoop; nn++) {
      int n_idx = NSizePerWarp * nn + row_id;
      if(n_idx < N_lenth){
        n_idx_valid[nn] = n_idx * K;
      } else {
        n_idx_valid[nn] = -1;
      }
    }



    for(int i=0;i<K;i+=WarpK){
      const int k_idx = i + col_id * k_per_thread;
      VecType a_vec;
      if(row_id_valid) {
        a_vec = *(VecType*)(a + row_id * K + k_idx);
      }
      

      for(int nn = 0; nn < NLoop; nn++) {

        VecType b_vec;
        if(n_idx_valid[nn] != -1){
          b_vec = *(VecType*)(b + n_idx_valid[nn] + k_idx);
        }
        #pragma unroll
        for(int ii = 0; ii < tc_number; ++ii)
        {
          builtin_amdgcn_mmac(a_vec.data[ii], b_vec.data[ii], d_vec[nn]);
        }
      }
    }



    extern __shared__ int out_smem_int[];
    int* out_smem_ptr = reinterpret_cast<int*>(out_smem_int+warp_idx*NPerBlock*M);

    using out_vec_dtype = typename std::conditional<std::is_same<scalar_t, _Float16>::value, half4_t, bf16x4_t>::type;
      if (row_id_valid) { 
        #pragma unroll
        for(int nn = 0; nn < NLoop; nn++) {
            int out_smem_offset = nn*NSizePerWarp*M;
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                out_smem_ptr[out_smem_offset + row_id * NSizePerWarp + i*4 + col_id] = d_vec[nn][i];
            }
        }
      }
      __syncthreads();
            
      if(row_id_valid){
        const float scale_a_value = scale_a[row_id];
        for(int nn = 0; nn < NLoop; nn++) {
          int out_smem_offset = nn*NSizePerWarp*M;
      
          bool n_idx_valid = nn*NSizePerWarp+col_id * 4 < N_lenth;
          if(n_idx_valid)
          {

            out_vec_dtype out_vec_16bit;
            intx4 out_vec_int = *reinterpret_cast<intx4*>(out_smem_ptr + out_smem_offset + row_id * NSizePerWarp + col_id * 4);
            floatx4 scale_b_vec = *reinterpret_cast<floatx4*>(scale_b + nn * NSizePerWarp + col_id * 4);
            #pragma unroll
            for(int i = 0; i < 4; i++) {
              scale_b_vec[i] = scale_a_value * scale_b_vec[i];
            }
            if constexpr(UseBias)
            {
              out_vec_16bit = *reinterpret_cast<out_vec_dtype*>(bias + nn * NSizePerWarp + col_id * 4);
              #pragma unroll
              for(int i = 0; i < 4; i++) {


                out_vec_16bit[i] = f32_to_f16_add<scalar_t>(out_vec_int[i] * scale_b_vec[i], out_vec_16bit[i]);
              }
            }
            else{
              #pragma unroll
              for(int i = 0; i < 4; i++) {
                out_vec_16bit[i] = f32_to_f16<scalar_t>(out_vec_int[i] * scale_b_vec[i]);
              }
            }
            *reinterpret_cast<out_vec_dtype*>(d + row_id * N + nn * NSizePerWarp + col_id * 4) = out_vec_16bit;
          }
        }
      }



}


template <typename scalar_t, int NUM_WARPS=4,int NPerBlock=16,int KPerWarp=64,bool UseBias=false>
__global__ void gemm_w8_a8_smooth_kernel_with_splitk_colmajor(int8_t *a, int8_t *b, scalar_t* d,float* scale_a, float* scale_b, scalar_t* bias, int* splitk_work_space, int* semaphores, int M,int N,int K, int splitk)
{
    constexpr int WarpK=KPerWarp;
    constexpr int tc_number = WarpK/32;
    constexpr int k_per_thread = tc_number * 8;
    constexpr int NSizePerWarp=16;
    constexpr int NLoop=NPerBlock/NSizePerWarp;
    const int bidx = blockIdx.x;  // bid_splitk
    const int bidy = blockIdx.y;  // bid_n
    const int tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;
    const int row_id = lane%NSizePerWarp;  //不用改，warp内各个线程的layout与fp16的tensorcore一致，只是thread持有的元素数量变多了
    const int col_id = lane/NSizePerWarp;    
    const int part_k = ((K/WarpK-1)/splitk+1)*WarpK;

    

    intx4 d_vec[NLoop];
    for(int i=0;i<NLoop;i++){
        d_vec[i]={0,0,0,0};
    }
    a+=      bidx*part_k;
    d+=      NPerBlock*bidy;
    b+=      NPerBlock*bidy*K + bidx*part_k;
    scale_b += NPerBlock*bidy;
    bias += NPerBlock*bidy;
    splitk_work_space += NPerBlock*bidy;
    const int length_per_block = (bidx==splitk-1?K-part_k*bidx:part_k);
    const int N_lenth = N - NPerBlock*bidy;
    
    

    using VecType = int8ABVector<tc_number>;
    VecType a_vec;  // 我们不会让m不符合的thread进行写入，因此无需对其清零
    VecType b_vec;
    for(int i=warp_idx*WarpK;i<length_per_block;i+=WarpK*NUM_WARPS){
    // for(int i=warp_idx*WarpK;i<K;i+=WarpK*NUM_WARPS){
      const int k_idx = i + col_id * k_per_thread;
      if(row_id < M) {
        a_vec = *(VecType*)(a + row_id * K + k_idx);
      }
      
      for(int nn = 0; nn < NLoop; nn++) {
        const int n_idx = NSizePerWarp * nn + row_id;
        int line_offset = n_idx * K + k_idx;
        if(n_idx < N_lenth){
          b_vec = *(VecType*)(b + line_offset);
        }
        #pragma unroll
        for(int ii = 0; ii < tc_number; ++ii)
        {
          builtin_amdgcn_mmac(a_vec.data[ii], b_vec.data[ii], d_vec[nn]);
        }
      }
    }



    extern __shared__ intx4 out_smem[];
    int* out_smem_ptr = reinterpret_cast<int*>(out_smem);

    if constexpr(NUM_WARPS>1){
        int lds_size_n_offset=NUM_WARPS/2*16*M/4;
        if(row_id<M){
            #pragma unroll
            for (int i = NUM_WARPS; i > 1; i /= 2) {
                int mid = i / 2;
                if (warp_idx >= mid && warp_idx < i) {
                    for(int nn=0;nn<NLoop;nn++){
                        out_smem[(warp_idx - mid) * 4 * M+row_id*4+col_id+lds_size_n_offset*nn]=d_vec[nn];
                    }
                }
                __syncthreads();
                if (warp_idx < mid) {
                    for(int nn=0;nn<NLoop;nn++){
                        intx4 tmp=out_smem[warp_idx* 4 * M+row_id*4+col_id+lds_size_n_offset*nn];
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            d_vec[nn][i] += tmp[i];
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    using out_vec_dtype = typename std::conditional<std::is_same<scalar_t, _Float16>::value, half4_t, bf16x4_t>::type;
    if(warp_idx == 0) {
      if(row_id < M) {
        for(int nn=0;nn<NLoop;nn++){
            for(int i=0;i<4;i++){
              if(nn*NSizePerWarp+col_id + i*4 < N_lenth){
                atomicAdd(splitk_work_space+nn*NSizePerWarp+i*4+row_id*N+col_id,d_vec[nn][i]);
              }
            }
        }
      }

  

      int order=-1;
      __syncthreads();
      if(tid==0)order = atomicAdd(semaphores+bidy,1);
      order=__shfl(order,0);


  

      // 注意：这里是所有的线程都要参与进来
      if(order==splitk-1){
          if(tid==0)semaphores[bidy]=0;
          for(int i=tid*4;i<M*NPerBlock;i+=256){
            // Q：i与offset的区别？  A：i是(M,NperBlock)的索引，offset是(M,N)的索引
            
            if(i%NPerBlock<N_lenth)
            {
              int offset=i/NPerBlock*N+i%NPerBlock; // 输出元素的索引
              int scale_a_offset = i/(NPerBlock); // scale_a的索引，
              int scale_b_offset = i%NPerBlock; // scale_b的索引

              intx4 out_vec_int = *(intx4*)(splitk_work_space+offset);
              *(intx4*)(splitk_work_space+offset) = {0,0,0,0};
              float scale_a_value = scale_a[scale_a_offset];
              floatx4 scale_b_vec = *reinterpret_cast<floatx4*>(scale_b + scale_b_offset);


              out_vec_dtype out_vec_16bit;
              if constexpr(UseBias)
              {
                out_vec_16bit = *reinterpret_cast<out_vec_dtype*>(bias + scale_b_offset);
                #pragma unroll
                for(int ii=0;ii<4;ii++){
                  out_vec_16bit[ii] = f32_to_f16_add<scalar_t>(out_vec_int[ii] * scale_a_value * scale_b_vec[ii], out_vec_16bit[ii]);
                }
              }
              else{
                #pragma unroll
                for(int ii=0;ii<4;ii++){
                  out_vec_16bit[ii] = f32_to_f16<scalar_t>(out_vec_int[ii] * scale_a_value * scale_b_vec[ii]);
                }
              }
              *(out_vec_dtype*)(d + offset) = out_vec_16bit;
            }

          }
      }
    }
}



Tensor gemm_w8_a8_smooth_colmajor(const Tensor& x,const Tensor& weight,const Tensor& scale_a,const Tensor& scale_b, const std::optional<torch::Tensor> bias, at::ScalarType out_dtype,int warps,int n_per_block,int k_per_warp, int splitk, bool use_n_warps)
{
    int m=x.sizes()[0];
    AT_ASSERTM(m <= 16, "m must be between 1 and 16");
    AT_ASSERTM(scale_a.scalar_type() == at::ScalarType::Float, "scale must be a float tensor.");
    AT_ASSERTM(scale_b.scalar_type() == at::ScalarType::Float, "scale must be a float tensor.");
    const bool use_bias = bias.has_value();



    int k=x.sizes()[1];
    int n=weight.sizes()[1];
    std::vector<long> size(2);
    size[0]=m;
    size[1]=n;
    int temp;
    Tensor ret=at::empty(size,x.options().dtype(out_dtype));
    auto stream = at::cuda::getCurrentCUDAStream();
// GPU 内存管理结构体
struct GPUWorkSpace {
    int* splitk_work_space;
    int* semaphores;
    int gpu_id;
    int process_id;
    pid_t thread_id;
    
    GPUWorkSpace() : splitk_work_space(nullptr), semaphores(nullptr) {
        hipGetDevice(&gpu_id);
        process_id = getpid();
        thread_id = syscall(SYS_gettid);
        // printf("gemm_w8_a8_smooth_colmajor Current GPU ID: %d, Process ID: %d, Thread ID: %d, hipmalloc\n", gpu_id, process_id, thread_id);
        
        hipMalloc(&splitk_work_space, 10000000); // 10m
        hipMemset(splitk_work_space, 0, 10000000);
        hipMalloc(&semaphores, 100000); // 100k
        hipMemset(semaphores, 0, 100000);
    }
    
    ~GPUWorkSpace() {
        if (splitk_work_space) {
            hipFree(splitk_work_space);
            // printf("gemm_w8_a8_smooth_colmajor GPU ID: %d, Process ID: %d, Thread ID: %d, hipfree splitk_work_space\n", gpu_id, process_id, thread_id);
        }
        if (semaphores) {
            hipFree(semaphores);
            // printf("gemm_w8_a8_smooth_colmajor GPU ID: %d, Process ID: %d, Thread ID: %d, hipfree semaphores\n", gpu_id, process_id, thread_id);
        }
    }
};

#define UseStaticWorkSpace 1
#if UseStaticWorkSpace
    static GPUWorkSpace* workspace = nullptr;
    if(workspace == nullptr){
        workspace = new GPUWorkSpace();
    }
    int* splitk_work_space = workspace->splitk_work_space;
    int* semaphores = workspace->semaphores;
#else
    int* splitk_work_space = (int*)work_space.data_ptr();  // 10MB
    int* semaphores = splitk_work_space + 3*1024*1024;  
#endif
    NPerBlock_SWITCH(n_per_block,[&] {
      NUM_WARPS_SWITCH(warps,[&] {
        KPerWarp_SWITCH(k_per_warp,[&] {
            Output_Type_SWITCH(out_dtype,[&] {
                  scalar_t* bias_ptr = nullptr;
                  if(use_bias){
                    bias_ptr = (scalar_t*)bias.value().data_ptr();
                  }
                  Bias_Type_SWITCH(use_bias,[&] {   
                    if (use_n_warps){
                          int lds_size= NumWarps*NPerBlock*m*sizeof(float);
                          int blocks=(n-1)/(NPerBlock*NumWarps)+1;
                          gemm_w8_a8_smooth_kernel_colmajor_NWarps<scalar_t,NumWarps,NPerBlock,KPerWarp,UseBias><<<dim3(blocks),NumWarps*C10_WARP_SIZE,lds_size,stream>>>
                              ((int8_t*)x.data_ptr(),(int8_t*)weight.data_ptr(),(scalar_t*)ret.data_ptr(),(float*)scale_a.data_ptr(),(float*)scale_b.data_ptr(),bias_ptr,m,n,k);   
                    }
                    else{
                          int blocks=(n-1)/NPerBlock+1;
                          int lds_size= NumWarps > 1 ? NumWarps/2*NPerBlock*m*sizeof(float) : NPerBlock*m*sizeof(float); // 折半归约
                          if(splitk>1){
                              gemm_w8_a8_smooth_kernel_with_splitk_colmajor<scalar_t,NumWarps,NPerBlock,KPerWarp,UseBias><<<dim3(splitk,blocks),NumWarps*C10_WARP_SIZE,lds_size,stream>>>
                                  ((int8_t*)x.data_ptr(),(int8_t*)weight.data_ptr(),(scalar_t*)ret.data_ptr(),(float*)scale_a.data_ptr(),(float*)scale_b.data_ptr(),bias_ptr,splitk_work_space,semaphores,m,n,k,splitk); 
                          }
                          else{ 
                              gemm_w8_a8_smooth_kernel_colmajor<scalar_t,NumWarps,NPerBlock,KPerWarp,UseBias><<<dim3(blocks),NumWarps*C10_WARP_SIZE,lds_size,stream>>>
                                  ((int8_t*)x.data_ptr(),(int8_t*)weight.data_ptr(),(scalar_t*)ret.data_ptr(),(float*)scale_a.data_ptr(),(float*)scale_b.data_ptr(),bias_ptr,m,n,k); 
                          }
                    }
                    });
                  });
          });
      });
    });

    
    return ret;
}
 

}
}
