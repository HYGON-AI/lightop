#include "reduce.h"

namespace at {
namespace native {

static int get_env_(const char *env_var) {
  if (char *value = std::getenv(env_var)) {
    return atoi(value);
  }
  return 0;
}

#ifndef uint32_t
 #define uint32_t unsigned int
#endif
#define fp8 uint8_t
constexpr int GroupSize=128;
using half2_t = __attribute__( (__vector_size__(2 * sizeof(_Float16)) )) _Float16;
using half4_t = __attribute__( (__vector_size__(4 * sizeof(_Float16)) )) _Float16;
using half8_t = __attribute__( (__vector_size__(8 * sizeof(_Float16)) )) _Float16;
using v4bh = __attribute__( (__vector_size__(4 * sizeof(short)) )) short;
using floatx4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
using uintx4 = __attribute__( (__vector_size__(4 * sizeof(uint32_t)) )) uint32_t;
using fp8x16 = __attribute__( (__vector_size__(16 * sizeof(fp8)) )) fp8;


inline __device__ float fp8e4m3_to_fp32(const fp8& input) {

  const uint32_t w = (uint32_t)input << 24;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  uint32_t renorm_shift = __clz(nonsign);
  renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;
  uint32_t result = sign | ((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23));
  return c10::detail::fp32_from_bits(result);
}

struct half4x2{
  half4_t data[2];
};

struct half4x4{
  half4_t data[4];
};

template<bool is_half=true>
inline __device__ void builtin_amdgcn_mmac(const half4_t& reg_a, const half4_t& reg_b, floatx4& reg_c)
{
    #if defined(__gfx936__) || defined(__gfx928__)
        if constexpr (is_half)reg_c=__builtin_amdgcn_mmac_f32_16x16x16f16(reg_a,reg_b,reg_c);
        else reg_c=__builtin_amdgcn_mmac_f32_16x16x16bf16(*(v4bh*)&reg_a,*(v4bh*)&reg_b,reg_c);
    #endif
}

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

#define Input_Type_SWITCH(SRC_DTYPE, ...)     \
  [&] {                                       \
    if (SRC_DTYPE == at::ScalarType::Half) {  \
      using scalar_t=_Float16;                \
      return __VA_ARGS__();                   \
    }else {                                   \
      using scalar_t=hip_bfloat16;          \
      return __VA_ARGS__();                   \
    }                                         \
  }()


static const int W8A16_WARPS = get_env_("W8A16_WARPS");
static const int W8A16_SPLITK = get_env_("W8A16_SPLITK");
static const int W8A16_FORCR_GEMV = get_env_("W8A16_FORCR_GEMV");
static const int W8A16_NPERBLOCK = get_env_("W8A16_NPERBLOCK");
// static const int W8A16_MPERBLOCK = get_env_("W8A16_MPERBLOCK");
static const int W8A16_PRINT_PARAM = get_env_("W8A16_PRINT_PARAM");


template <typename scalar_t, int NUM_WARPS=4,int NPerBlock=16>
__global__ void gemm_w8_a16_kernel_colmajor(scalar_t *a,fp8* b,scalar_t* d,scalar_t* scale_n,int m,int n,int k)
{
    constexpr int NLoop=NPerBlock/16;
    constexpr bool is_half = std::is_same<scalar_t, _Float16>::value;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;
    const int rowid = lane%16;
    const int rows = lane/16;
    static_assert(GroupSize==128,"GroupSize==128");
    floatx4 d_vec[NLoop];
    for(int i=0;i<NLoop;i++){
        d_vec[i]={0,0,0,0};
    }
    d+=      NPerBlock*bid;
    b+=      NPerBlock*bid*k;
    scale_n+=(NPerBlock*bid/GroupSize)*(k/GroupSize);
    half4x4 a_vec;
    for(int i=0;i<4;i++)a_vec.data[i]={0,0,0,0};
    for(int i=warp_idx*GroupSize;i<k;i+=GroupSize*NUM_WARPS){
        scalar_t t_scale;
        if(lane==0)t_scale=scale_n[i/GroupSize];
        t_scale=__shfl((float)t_scale,0);
        #pragma unroll
        for(int i2=0;i2<GroupSize;i2+=GroupSize/2){
            int line_offset = rowid*k + i + i2 + rows*16;
            if(rowid<m)a_vec=*(half4x4*)(a+line_offset);
            #pragma unroll
            for(int nn=0;nn<NLoop;nn++){
                fp8x16 tb = *(fp8x16*)(b+line_offset+16*nn*k);
                #pragma unroll
                for(int ii=0;ii<4;ii++){
                    half4_t b_vec;
                    scalar_t *bp= reinterpret_cast<scalar_t*>(&b_vec);
                    #pragma unroll
                    for(int kk=0;kk<4;kk++){
                      bp[kk] = fp8e4m3_to_fp32(tb[ii*4+kk])*t_scale;
                    }
                    builtin_amdgcn_mmac<is_half>(a_vec.data[ii],b_vec,d_vec[nn]);
                    // printf("tid=%d,a_vec=%.4f,%.4f,%.4f,%.4f,b_vec=%.4f,%.4f,%.4f,%.4f,dec=%.4f,%.4f,%.4f,%.4f\n",lane, (float)a_vec.data[ii][0]
                    // , (float)a_vec.data[ii][1], (float)a_vec.data[ii][2], (float)a_vec.data[ii][3],
                    // (float)b_vec[0], (float)b_vec[1], (float)b_vec[2], (float)b_vec[3],
                    // d_vec[0],d_vec[1],d_vec[2],d_vec[3]);
                }
            }
        }
    }
    if constexpr(NUM_WARPS>1){
        extern __shared__ floatx4 out_smem[];
        int lds_size_n_offset=NUM_WARPS/2*16*m/4;
        if(rowid<m){
            #pragma unroll
            for (int i = NUM_WARPS; i > 1; i /= 2) {
                int mid = i / 2;
                if (warp_idx >= mid && warp_idx < i) {
                    for(int nn=0;nn<NLoop;nn++){
                        out_smem[(warp_idx - mid) * 4 * m+rowid*4+rows+lds_size_n_offset*nn]=d_vec[nn];
                    }
                }
                __syncthreads();
                if (warp_idx < mid) {
                    for(int nn=0;nn<NLoop;nn++){
                        floatx4 tmp=out_smem[warp_idx* 4 * m+rowid*4+rows+lds_size_n_offset*nn];
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
    if(rowid<m&&warp_idx==0){
        for(int nn=0;nn<NLoop;nn++){
            for(int i=0;i<4;i++){
                d[nn*16+i*4+rowid*n+rows]=static_cast<scalar_t>(d_vec[nn][i]);
            }
        }
    }
}

template <typename scalar_t, int NUM_WARPS=4,int NPerBlock=16>
__global__ void gemm_w8_a16_kernel_with_splitk_colmajor(scalar_t *a,fp8* b,scalar_t* d,scalar_t* scale_n, float* splitk_work_space,int *semaphores, int m,int n,int k,int splitk)
{
    constexpr int NLoop=NPerBlock/16;
    constexpr bool is_half = std::is_same<scalar_t, _Float16>::value;
    const int bidy = blockIdx.y;
    const int bidx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;
    const int rowid = lane%16;
    const int rows = lane/16;
    const int part_k=((k/GroupSize-1)/splitk+1)*GroupSize;
    static_assert(GroupSize==128,"GroupSize==128");
    floatx4 d_vec[NLoop];
    for(int i=0;i<NLoop;i++){
        d_vec[i]={0,0,0,0};
    }
    a+=      bidx*part_k;
    d+=      NPerBlock*bidy;
    b+=      NPerBlock*bidy*k+part_k*bidx;
    scale_n+=(NPerBlock*bidy/GroupSize)*(k/GroupSize)+part_k*bidx/GroupSize;
    splitk_work_space+=NPerBlock*bidy;
    const int length_per_block = (bidx==splitk-1?k-part_k*bidx:part_k);
    half4x4 a_vec;
    for(int i=0;i<4;i++)a_vec.data[i]={0,0,0,0};
    for(int i=warp_idx*GroupSize;i<length_per_block;i+=GroupSize*NUM_WARPS){
        scalar_t t_scale;
        if(lane==0)t_scale=scale_n[i/GroupSize];
        t_scale=__shfl((float)t_scale,0);
        #pragma unroll
        for(int i2=0;i2<GroupSize;i2+=GroupSize/2){
            int line_offset = rowid*k + i + i2 + rows*16;
            if(rowid<m)a_vec=*(half4x4*)(a+line_offset);
            #pragma unroll
            for(int nn=0;nn<NLoop;nn++){
                fp8x16 tb = *(fp8x16*)(b+line_offset+16*nn*k);
                #pragma unroll
                for(int ii=0;ii<4;ii++){
                    half4_t b_vec;
                    scalar_t *bp= reinterpret_cast<scalar_t*>(&b_vec);
                    #pragma unroll
                    for(int kk=0;kk<4;kk++){
                      bp[kk] = fp8e4m3_to_fp32(tb[ii*4+kk])*t_scale;
                    }
                    builtin_amdgcn_mmac<is_half>(a_vec.data[ii],b_vec,d_vec[nn]);
                }
            }
        }
    }
    if constexpr(NUM_WARPS>1){
        extern __shared__ floatx4 out_smem[];
        int lds_size_n_offset=NUM_WARPS/2*16*m/4;
        if(rowid<m){
            #pragma unroll
            for (int i = NUM_WARPS; i > 1; i /= 2) {
                int mid = i / 2;
                if (warp_idx >= mid && warp_idx < i) {
                    for(int nn=0;nn<NLoop;nn++){
                        out_smem[(warp_idx - mid) * 4 * m+rowid*4+rows+lds_size_n_offset*nn]=d_vec[nn];
                    }
                }
                __syncthreads();
                if (warp_idx < mid) {
                    for(int nn=0;nn<NLoop;nn++){
                        floatx4 tmp=out_smem[warp_idx* 4 * m+rowid*4+rows+lds_size_n_offset*nn];
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
    if(warp_idx==0){
        if(rowid<m){
            for(int nn=0;nn<NLoop;nn++){
                for(int i=0;i<4;i++){
                    atomicAdd(splitk_work_space+nn*16+i*4+rowid*n+rows,d_vec[nn][i]);
                }
            }
        }
        int order=-1;
        __syncthreads();
        if(tid==0)order = atomicAdd(semaphores+bidy,1);
        order=__shfl(order,0);
        if(order==splitk-1){
            if(tid==0)semaphores[bidy]=0;
            for(int i=tid*4;i<m*NPerBlock;i+=256){
                int offset=i/NPerBlock*n+i%NPerBlock;
                floatx4 tempd=*(floatx4*)(splitk_work_space+offset);
                *(floatx4*)(splitk_work_space+offset)={0,0,0,0};
                half4_t tempd2;
                scalar_t *pd=reinterpret_cast<scalar_t*>(&tempd2);
                for(int ii=0;ii<4;ii++){
                    pd[ii]=static_cast<scalar_t>(tempd[ii]);
                }
                *(half4_t*)(d+offset)=tempd2;
            }
        }
    }
}


static void select_warps(int m,int n,int k,int &warps,int &n_per_block,int& splitk)
{
    if(n_per_block==0||warps==0||splitk==0){
        if(n==576&&k==7168){warps=8;n_per_block=16;splitk=8;return;}
        if(n==1536&&k==7168){warps=4;n_per_block=16;splitk=8;return;}
        if(n==3072&&k==1536){warps=8;n_per_block=32;splitk=1;return;}
        if(n==4096&&k==512){warps=4;n_per_block=16;splitk=1;return;}
        if(n==7168&&k==2048){warps=8;n_per_block=16;splitk=2;return;}
        
        warps=1;
        n_per_block=16;
        splitk=1;
        return;
    }
    
}

Tensor gemm_w8_a16_fp8_colmajor(const Tensor& x,const Tensor& weight,const Tensor& scale_n,int warps,int n_per_block,int splitk)
{
    int m=x.sizes()[0];
    AT_ASSERTM(m <= 16, "m must be between 1 and 16");
    int k=x.sizes()[1];
    int n=weight.sizes()[1];
    std::vector<long> size(2);
    size[0]=m;
    size[1]=n;
    // std::cout<<"weight.sizes()="<<weight.sizes()[0]<<","<<weight.sizes()[1]<<std::endl;
    // std::cout<<"weight.strides()="<<weight.strides()[0]<<","<<weight.strides()[1]<<std::endl;
    Tensor ret=at::empty(size,x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    static float* splitk_work_space = nullptr;
    static int* semaphores = nullptr;
    if(splitk_work_space == nullptr){
        hipMalloc(&splitk_work_space, 10000000); // 10m
        hipMemset(splitk_work_space,0,10000000);
        hipMalloc(&semaphores, 100000); // 100k
        hipMemset(semaphores,0,100000);
    }
    select_warps(m,n,k,warps,n_per_block,splitk);
    //使用环境变量控制，方便训练数据。
    if(W8A16_SPLITK>0&&W8A16_SPLITK<=8)splitk=W8A16_SPLITK;
    if(W8A16_NPERBLOCK!=0)n_per_block=W8A16_NPERBLOCK;
    if(W8A16_WARPS!=0)warps=W8A16_WARPS;
    NPerBlock_SWITCH(n_per_block,[&] {
        NUM_WARPS_SWITCH(warps,[&] {
            Input_Type_SWITCH(x.dtype(),[&] {
                int blocks=(n-1)/NPerBlock+1;
                int lds_size=NumWarps/2*NPerBlock*m*sizeof(float);
                if(W8A16_PRINT_PARAM)std::cout<<"blocks="<<blocks<<" NumWarps="<<NumWarps<<" NPerBlock="<<NPerBlock<<" splitk="<<splitk<<std::endl;
                if(splitk>1){
                    gemm_w8_a16_kernel_with_splitk_colmajor<scalar_t,NumWarps,NPerBlock><<<dim3(splitk,blocks),NumWarps*C10_WARP_SIZE,lds_size,stream>>>
                            ((scalar_t*)x.data_ptr(),(fp8*)weight.data_ptr(),(scalar_t*)ret.data_ptr(),(scalar_t*)scale_n.data_ptr(),splitk_work_space,semaphores,m,n,k,splitk);
                }
                else{
                    gemm_w8_a16_kernel_colmajor<scalar_t,NumWarps,NPerBlock><<<dim3(blocks),NumWarps*C10_WARP_SIZE,lds_size,stream>>>
                        ((scalar_t*)x.data_ptr(),(fp8*)weight.data_ptr(),(scalar_t*)ret.data_ptr(),(scalar_t*)scale_n.data_ptr(),m,n,k); 
                }
            });
        });
    });
 
    return ret;
}

template<typename scalar_t,int blocksize>
__global__ void fp8to16_kernel(scalar_t*out, fp8* x,scalar_t* scale,int m,int n){
  int tid=threadIdx.x;
  int bid=blockIdx.x;
  __shared__ float s_scale;
  if(tid==0)s_scale=scale[bid];
  __syncthreads();
  float v_scale=s_scale;
  int blocks_per_line=n/GroupSize;
  int offset = bid%blocks_per_line*GroupSize+bid/blocks_per_line*n*GroupSize;
  out+=offset;
  x+=offset;
  #pragma unroll
  for(int i=0;i<GroupSize*GroupSize;i+=16*blocksize){
    int offset2=(i/GroupSize+tid/8)*n+tid%8*16;
    if(offset+offset2>m*n)return;
    fp8x16 tx = *reinterpret_cast<fp8x16*>(x+offset2);
    for(int k=0;k<16;k++){
        out[offset2+k] = static_cast<scalar_t>(fp8e4m3_to_fp32(tx[k])*v_scale);
    }
  }
}

Tensor fp8to16(const Tensor&x,const Tensor& scale)
{
  auto size = x.sizes();
  Tensor ret = at::empty(size,scale.options());
  int m = size[0];
  int n = size[1];
  auto stream = at::cuda::getCurrentCUDAStream();
  Input_Type_SWITCH(scale.dtype(),[&]{
    constexpr int blocksize=512;
    fp8to16_kernel<scalar_t,blocksize><<<((m-1)/GroupSize+1)*(n/GroupSize),blocksize,0,stream>>>((scalar_t*)ret.data_ptr(),(fp8*)x.data_ptr(),(scalar_t*)scale.data_ptr(),m,n);
  });
  return ret;
}

}
}