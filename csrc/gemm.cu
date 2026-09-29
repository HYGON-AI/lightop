#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "reduce.h"
namespace at {
namespace native {

#ifndef uint32_t
 #define uint32_t unsigned int
#endif


using half2_t = __attribute__( (__vector_size__(2 * sizeof(_Float16)) )) _Float16;
using half4_t = __attribute__( (__vector_size__(4 * sizeof(_Float16)) )) _Float16;
using v4bh = __attribute__( (__vector_size__(4 * sizeof(short)) )) short;
using float4_t = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
struct half4x2{
  half4_t data[2];
};

struct half2x4{
  half2_t data[4];
};

template<bool is_half=true>
inline __device__ void builtin_amdgcn_mmac(const half4_t& reg_a, const half4_t& reg_b, float4_t& reg_c)
{
    #if defined(__gfx936__) || defined(__gfx928__)
        if constexpr (is_half)reg_c=__builtin_amdgcn_mmac_f32_16x16x16f16(reg_a,reg_b,reg_c);
        else reg_c=__builtin_amdgcn_mmac_f32_16x16x16bf16(*(v4bh*)&reg_a,*(v4bh*)&reg_b,reg_c);
    #elif defined(__gfx938__)
        if constexpr (is_half) reg_c = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(reg_a, reg_b, reg_c, false, false);
        else reg_c = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(v4bh*)&reg_a, *(v4bh*)&reg_b, reg_c, false, false);
    #endif
}

inline __device__ void v_dot2_f32_f16(float& a, const half2_t &  b,const half2_t &  c) {
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %0;": "=v"(a): "v"(b), "v"(c), "0"(a));
}

//假设a矩阵很小，不用过lds,反复读取global不影响性能
// Each block computes a 16x16 output tile, with FP32 split-K accumulation.
template <typename scalar_t, typename output_t, int NUM_WARPS=4>
__global__ void gemm_nt_fp16(const scalar_t *a, const scalar_t *b, output_t *d,
                            int m, int n, int k)
{
    const int warp_idx = threadIdx.x / C10_WARP_SIZE;
    const int lane = threadIdx.x % C10_WARP_SIZE;
    const int row = lane % 16;
    const int col_group = lane / 16;
    const int a_row = blockIdx.y * 16 + row;
    const int b_row = blockIdx.x * 16 + row;
    constexpr bool is_half = std::is_same<scalar_t, at::Half>::value;
    float4_t acc = {0,0,0,0};
    for (int base = warp_idx * 32; base < k; base += 32 * NUM_WARPS) {
        half4x2 av = {}, bv = {};
        const int offset = base + col_group * 8;
        if (a_row < m) av = *reinterpret_cast<const half4x2*>(a + a_row*k + offset);
        if (b_row < n) bv = *reinterpret_cast<const half4x2*>(b + b_row*k + offset);
        builtin_amdgcn_mmac<is_half>(av.data[0], bv.data[0], acc);
        builtin_amdgcn_mmac<is_half>(av.data[1], bv.data[1], acc);
    }
    __shared__ float4_t partial[NUM_WARPS/2 * 64];
    // All lanes participate in barriers, including the last partial M tile.
    for (int active = NUM_WARPS; active > 1; active /= 2) {
        int half = active / 2;
        if (warp_idx >= half && warp_idx < active)
            partial[(warp_idx-half)*64 + lane] = acc;
        __syncthreads();
        if (warp_idx < half) {
            float4_t other = partial[warp_idx*64 + lane];
            for (int j=0; j<4; ++j) acc[j] += other[j];
        }
        __syncthreads();
    }
    if (warp_idx == 0 && a_row < m) {
        for (int j=0; j<4; ++j) {
            int col = blockIdx.x*16 + col_group + j*4;
            if (col < n) d[a_row*n + col] = static_cast<output_t>(acc[j]);
        }
    }
}

Tensor gemm_opt(const Tensor& x, const Tensor& weight,
                c10::optional<at::ScalarType> out_dtype) {
    TORCH_CHECK(x.dim()==2 && weight.dim()==2, "gemm expects 2D inputs");
    TORCH_CHECK(x.is_cuda() && weight.device()==x.device(), "gemm inputs must be on the same GPU");
    TORCH_CHECK(x.is_contiguous(), "gemm input must be contiguous");
    TORCH_CHECK((x.scalar_type()==at::kBFloat16 || x.scalar_type()==at::kHalf) &&
                weight.scalar_type()==x.scalar_type(), "gemm requires matching BF16 or FP16 inputs");
    const int m=x.size(0), k=x.size(1);
    const bool row_major = weight.size(1)==k && weight.is_contiguous();
    TORCH_CHECK(row_major || (weight.size(0)==k && weight.t().is_contiguous()),
                "gemm weight must be contiguous [N,K] or transposed [K,N]");
    const int n=weight.size(row_major ? 0 : 1);
    TORCH_CHECK(weight.size(row_major ? 1 : 0)==k && k>0 && k%32==0,
                "gemm requires matching K, positive and divisible by 32");
    const auto dtype=out_dtype.value_or(x.scalar_type());
    TORCH_CHECK(dtype==x.scalar_type() || dtype==at::kFloat,
                "gemm out_dtype must match input dtype or be float32");
    const c10::cuda::CUDAGuard device_guard(x.device());
    Tensor ret=at::empty({m,n},x.options().dtype(dtype));
    if (m==0 || n==0) return ret;
    auto stream=at::cuda::getCurrentCUDAStream();
    dim3 grid((n+15)/16, (m+15)/16);
    AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "gemm_opt", [&] {
        if (dtype==at::kFloat)
            gemm_nt_fp16<scalar_t,float><<<grid,4*C10_WARP_SIZE,0,stream>>>(
                x.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),ret.data_ptr<float>(),m,n,k);
        else
            gemm_nt_fp16<scalar_t,scalar_t><<<grid,4*C10_WARP_SIZE,0,stream>>>(
                x.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),ret.data_ptr<scalar_t>(),m,n,k);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return ret;
}

template <typename T,int VEC,int NUM_WARPS>
__inline__ __device__ void BlockReduceSumVEC(T* val, T* shared) {
  #pragma unroll
  for(int i=0;i<VEC;i++){
      val[i] = WarpReduceSum(val[i]);
  }
  //if(blockIdx.x==0)printf("tid=%d,val=%.4f\n",threadIdx.x,val[0]);
  if constexpr(1<NUM_WARPS){
    const int tid = threadIdx.x;
    const int lid = tid % C10_WARP_SIZE;
    const int wid = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    if (lid == 0) {
        for(int i=0;i<VEC;i++){
           shared[wid*VEC+i] = val[i];
        }
    }
    __syncthreads();
    // if(blockIdx.x==0&&tid==0)printf("tid=%d,shared=(%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f)\n",threadIdx.x,shared[0],shared[1],shared[2],shared[3],shared[4],shared[5],shared[6],shared[7]);
    if (wid == 0&&lid<NUM_WARPS) {
        for(int i=0;i<VEC;i++){
            val[i] = WarpReduceSum(shared[lid*VEC+i],NUM_WARPS/2);
        }
    }
  }
}

template <int NUM_WARPS,int M>
__global__ void gemv_w4_a16_kernel(_Float16 *a,uint32_t* b,_Float16* d,_Float16* scale_n,_Float16* zero_n,int n,int k)
{
    const int bid=blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float reduce[M*NUM_WARPS];
    constexpr int VEC=8;
    b+=bid*k/8;
    d+=bid;
    float out[M]={0};
    _Float16 scale=scale_n[bid];
    _Float16 zero=zero_n[bid];
    for(int i=tid;i<k/8;i+=NUM_WARPS*C10_WARP_SIZE){
        uint32_t tb=b[i];
        half2x4 a_vec;
        half2x4 b_vec;
        #pragma unroll
        for(int k=0;k<8;k++){
            b_vec.data[k/2][k%2]=static_cast<_Float16>((tb >> (k * 4)) & 0xF)*scale + zero;
        }
        #pragma unroll
        for(int m=0;m<M;m++){
            a_vec=*(half2x4*)(a+k*m+i*8);
            #pragma unroll
            for(int ii=0;ii<4;ii++){
                v_dot2_f32_f16(out[m],a_vec.data[ii],b_vec.data[ii]);
                //if(bid==0)printf("tid=%d,a=(%.4f,%.4f),b=(%.4f,%.4f),out=%f\n",tid,(float)a_vec.data[ii][0],(float)a_vec.data[ii][1],(float)b_vec.data[ii][0],(float)b_vec.data[ii][1],out[m]);
            }
        }
    }
    BlockReduceSumVEC<float,M,NUM_WARPS>(out,reduce);
    if(tid==0){
        for(int i=0;i<M;i++){
            d[i*n]=static_cast<_Float16>(out[i]);
        }
    }
}

template <int NUM_WARPS=4,int NPerBlock=16>
__global__ void gemm_w4_a16_kernel(_Float16 *a,uint32_t* b,_Float16* d,_Float16* scale_n,_Float16* zero_n,int m,int n,int k)
{
    const int bid=blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;
    const int rowid = lane%16;
    const int rows = lane/16;
    constexpr int VEC=8;
    int offset_a=rowid*k+warp_idx*16*2+rows*4*2;
    int offset_b=(offset_a+NPerBlock*bid*k)/8;
    half4x2 a_vec,b_vec;
    float4_t d_vec={0,0,0,0};
    a_vec.data[0]={0,0,0,0};
    a_vec.data[1]={0,0,0,0};
    a+=offset_a;
    b+=offset_b;
    d+=NPerBlock*bid+rowid*n+rows;
    _Float16 scale=scale_n[NPerBlock*bid+rowid];
    _Float16 zero=zero_n[NPerBlock*bid+rowid];
    for(int i=0;i+16*2*warp_idx<k;i+=16*2*NUM_WARPS){
        if(rowid<m){
            a_vec=*(half4x2*)(a+i);
        }
        uint32_t tb=b[i/8];
        #pragma unroll
        for(int k=0;k<8;k++){
            b_vec.data[k/4][k%4]=static_cast<_Float16>((tb >> (k * 4)) & 0xF)*scale + zero;
        }
        builtin_amdgcn_mmac(a_vec.data[0],b_vec.data[0],d_vec);
        builtin_amdgcn_mmac(a_vec.data[1],b_vec.data[1],d_vec);
    }
    if constexpr(NUM_WARPS>1){
        extern __shared__ float4_t out_smem[];
        if(rowid<m){
            #pragma unroll
            for (int i = NUM_WARPS; i > 1; i /= 2) {
                int mid = i / 2;
                if (warp_idx >= mid && warp_idx < i) {
                    out_smem[(warp_idx - mid) * 4 * m+rowid*4+rows]=d_vec;
                }
                __syncthreads();
                if (warp_idx < mid) {
                    float4_t tmp=out_smem[warp_idx* 4 * m+rowid*4+rows];
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        d_vec[i] += tmp[i];
                    }
                }
                __syncthreads();
            }
        }
    }
    if(rowid<m&&warp_idx==0){
        for(int i=0;i<4;i++){
            d[i*4]=static_cast<_Float16>(d_vec[i]);
        }
    }
}

#define M_SWITCH(m,...)[&]{ \
  if (m==1){\
    constexpr int M=1;\
    return __VA_ARGS__();                     \
  }\
  else{\
    constexpr int M=2;\
    return __VA_ARGS__();                     \
  }\
}()

#define NUM_WARPS_SWITCH(warps, ...)    \
  [&] {                                         \
    if (warps == 16) {                   \
      constexpr static int NumWarps = 16;  \
      return __VA_ARGS__();                     \
    }else if (warps == 4) {                 \
      constexpr static int NumWarps = 4;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int NumWarps = 8;  \
      return __VA_ARGS__();                     \
    }                                           \
  }()

bool select_gemv(int m,int n,int k)
{
   if(m>2)return false;
   if(m<=2&&n==24576&&k==6144)return true;
   if(m==1&&(n==3328&&k==6144||n==6144&&k==12288))return true;
   return false;
}

int select_warps(int m,int n,int k)
{
    if(m>=8&&m<=16){
        if(n==6144&&(k==3072||k==12288))return 4;
        if(n==24576&&k==6144)return 16;
    }
    return 8;
}

Tensor gemm_w4_a16(const Tensor& x,const Tensor& weight,const Tensor& scale_n,const Tensor& zero_n)
{
    int m=x.sizes()[0];
    int k=x.sizes()[1];
    int n=weight.sizes()[1];
    std::vector<long> size(2);
    size[0]=m;
    size[1]=n;
    // std::cout<<"weight.sizes()="<<weight.sizes()[0]<<","<<weight.sizes()[1]<<std::endl;
    // std::cout<<"weight.strides()="<<weight.strides()[0]<<","<<weight.strides()[1]<<std::endl;
    Tensor ret=at::empty(size,x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    if(select_gemv(m,n,k)){
        constexpr int NumWarps=2;
        int blocks=n;
        M_SWITCH(m,[&] {
            gemv_w4_a16_kernel<NumWarps,M><<<blocks,NumWarps*C10_WARP_SIZE,0,stream>>>
                    ((_Float16*)x.data_ptr(),(uint32_t*)weight.data_ptr(),(_Float16*)ret.data_ptr(),(_Float16*)scale_n.data_ptr(),(_Float16*)zero_n.data_ptr(),n,k);
        });
    }
    else{
        int warps=select_warps(m,n,k);
        constexpr int NPerBlock=16;
        int blocks=(n-1)/NPerBlock+1;
        NUM_WARPS_SWITCH(warps,[&] {
            int lds_size=NumWarps/2*NPerBlock*m*sizeof(float);
            gemm_w4_a16_kernel<NumWarps,NPerBlock><<<blocks,NumWarps*C10_WARP_SIZE,lds_size,stream>>>
                    ((_Float16*)x.data_ptr(),(uint32_t*)weight.data_ptr(),(_Float16*)ret.data_ptr(),(_Float16*)scale_n.data_ptr(),(_Float16*)zero_n.data_ptr(),m,n,k);
        });
    }

    return ret;
}

__global__ void w4tow16kernel(uint32_t *input,at::Half* output,at::Half* scale_n,at::Half* zero_n,int k)
{
    input+=blockIdx.x*k/8;
    output+=blockIdx.x*k;
    auto scale=scale_n[blockIdx.x];
    auto zero=zero_n[blockIdx.x];
    for(int i=threadIdx.x;i<k/8;i+=blockDim.x){
        uint32_t a=input[i];
        for (int k = 0; k < 8; ++k) {
            uint32_t temp = (a >> (k * 4)) & 0xF;
            output[i*8+k]=static_cast<at::Half>(temp) * scale + zero;
        }
    }
}

Tensor w4_to_w16(const Tensor& x,const Tensor& scale_n,const Tensor& zero_n)
{
    int k = x.sizes()[1]*8;
    int n = x.sizes()[0];
    c10::TensorOptions options(DeviceType::CUDA);
    options=options.dtype(ScalarType::Half);
    Tensor ret=at::empty({n,k},options);
    w4tow16kernel<<<n,256>>>((uint32_t*)x.data_ptr(),ret.data_ptr<at::Half>(),scale_n.data_ptr<at::Half>(),zero_n.data_ptr<at::Half>(),k);
    return ret;
}

}
}