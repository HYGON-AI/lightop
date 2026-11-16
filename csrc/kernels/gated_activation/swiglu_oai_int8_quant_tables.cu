// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// BF16-specialized exact quantization tables and live-row masked scheduling.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

namespace at::native {
void swiglu_oai_int8_quant_validate(const torch::Tensor&,const torch::Tensor&,
                                  const torch::Tensor&,double,double,double);
namespace {
constexpr int QuantTableSize=128*9*128;

// For normal BF16 amax >= eps, x/scale depends only on the two 7-bit
// mantissas and their exponent difference. Power-of-two normalization is
// exact because scale is normal FP32. Differences >=9 always round to zero.
// This table computes the SAME FP32 scale/division/RNE contract, not an
// approximation to ideal x*127/amax.
__global__ void build_int8_mantissa_table_kernel(uint8_t* table) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=QuantTableSize) return;
  const int max_frac=i/(9*128), delta=(i/128)%9, value_frac=i%128;
  const float amax=1.0f+float(max_frac)*(1.0f/128.0f);
  const float value=ldexpf(1.0f+float(value_frac)*(1.0f/128.0f),-delta);
  const float scale=amax/127.0f;
  table[i]=static_cast<uint8_t>(fminf(127.0f,nearbyintf(__fdiv_rn(value,scale))));
}
__device__ __forceinline__ int8_t quantize_bits(
    uint16_t bits,uint16_t max_bits,float maximum,float scale,float inv_scale,
    const uint8_t* quant_table) {
  c10::BFloat16 value; value.x=bits;
  const float scaled=float(value)*inv_scale;
  const float rounded=nearbyintf(scaled);
  if(fabsf(scaled-rounded)<0.4999f)
    return static_cast<int8_t>(fminf(127.0f,fmaxf(-127.0f,rounded)));
  if(maximum>=1.0e-10f) {
    const int delta=int((max_bits>>7)&255)-int((bits>>7)&255);
    // Also reject negative differences from nonfinite lanes before indexing.
    if(static_cast<unsigned>(delta)>=9) return 0;
    // |value|<=amax by construction, so delta cannot be negative.
    const int index=(((max_bits&127)*9+delta)*128)+(bits&127);
    const int magnitude=quant_table[index];
    return static_cast<int8_t>((bits&0x8000)?-magnitude:magnitude);
  }
  if(maximum==0) return 0;
  c10::BFloat16 v;v.x=bits;
  float q=nearbyintf(__fdiv_rn(float(v),scale));
  return static_cast<int8_t>(fminf(127.0f,fmaxf(-127.0f,q)));
}

template <int Bound,int Vec>
__global__ __launch_bounds__(256)
void swiglu_oai_int8_quant_contiguous_tables_kernel(
 const c10::BFloat16* __restrict__ x,int8_t* __restrict__ q,
 float* __restrict__ scales,const float* __restrict__ gate_table,
 const uint8_t* __restrict__ quant_table,int64_t rows,int width,
 float beta,float limit) {
 const int lane=threadIdx.x&63;
 const int64_t row=int64_t(blockIdx.x)*4+threadIdx.x/64;
 if(row>=rows)return;
 constexpr int Segments=(Bound+64*Vec-1)/(64*Vec);
 using In=at::native::memory::aligned_vector<c10::BFloat16,Vec>;
 using Out=at::native::memory::aligned_vector<int8_t,Vec>;
 uint16_t values[Segments*Vec];
 float maximum=0;
#pragma unroll
 for(int seg=0;seg<Segments;++seg) {
  const int col=(seg*64+lane)*Vec;
  if(col<width) {
   In g=*reinterpret_cast<const In*>(x+row*2*width+col);
   In u=*reinterpret_cast<const In*>(x+row*2*width+width+col);
#pragma unroll
   for(int v=0;v<Vec;++v) {
    float up=float(u.val[v]);
    if(limit>0)up=fminf(fmaxf(up,-limit),limit);
    c10::BFloat16 value(gate_table[g.val[v].x]*(up+beta));
    values[seg*Vec+v]=value.x;
    maximum=fmaxf(maximum,fabsf(float(value)));
   }
  }
 }
#pragma unroll
 for(int delta=32;delta;delta>>=1)
  maximum=fmaxf(maximum,__shfl_xor(maximum,delta,64));
 const float scale=fmaxf(maximum,1.0e-10f)/127.0f;
 const uint16_t max_bits=c10::BFloat16(maximum).x;
 const float inv_scale=__fdiv_rn(1.0f,scale);
 if(lane==0)scales[row]=scale;
#pragma unroll
 for(int seg=0;seg<Segments;++seg) {
  const int col=(seg*64+lane)*Vec;
  if(col<width) {
   Out out;
#pragma unroll
   for(int v=0;v<Vec;++v)
    out.val[v]=quantize_bits(values[seg*Vec+v],max_bits,maximum,scale,inv_scale,quant_table);
   *reinterpret_cast<Out*>(q+row*width+col)=out;
  }
 }
}

template <int Bound,int Threads,bool Masked=true,bool Direct=false>
__global__ __launch_bounds__(Threads)
void swiglu_oai_int8_quant_masked_flat_tables_kernel(
 const c10::BFloat16* __restrict__ x,int8_t* __restrict__ q,
 float* __restrict__ scales,const int32_t* __restrict__ counts,
 const float* __restrict__ gate_table,const uint8_t* __restrict__ quant_table,
 int capacity,int width,float beta,float limit) {
 constexpr int Vec=8,Waves=Threads/64,Segments=(Bound+Threads*Vec-1)/(Threads*Vec);
 const int tid=threadIdx.x;
 __shared__ int prefix[17];
 __shared__ float partial[Waves];
 if constexpr(Masked) {
 if(tid<64) {
  int n=0;
  if(tid<16) {int c=counts[tid];n=c<0?0:(c>capacity?capacity:c);}
#pragma unroll
  for(int d=1;d<16;d<<=1) {
   int left=__shfl_up(n,d,64);
   if(tid>=d)n+=left;
  }
  if(tid<16)prefix[tid+1]=n;
  if(tid==0)prefix[0]=0;
 }
 __syncthreads();
 }
 using In=at::native::memory::aligned_vector<c10::BFloat16,Vec>;
 using Out=at::native::memory::aligned_vector<int8_t,Vec>;
 for(int flat=blockIdx.x;flat<(Masked?prefix[16]:capacity);flat+=gridDim.x) {
  int64_t row=flat;
  if constexpr(Masked) {
  int expert=0;
#pragma unroll
  for(int e=0;e<15;++e)if(flat>=prefix[e+1])expert=e+1;
  row=int64_t(expert)*capacity+flat-prefix[expert];
  }
  uint16_t values[Segments*Vec];
  float maximum=0;
#pragma unroll
  for(int seg=0;seg<Segments;++seg) {
   const int col=(seg*Threads+tid)*Vec;
   if(col<width) {
    In g=*reinterpret_cast<const In*>(x+row*2*width+col);
    In u=*reinterpret_cast<const In*>(x+row*2*width+width+col);
#pragma unroll
    for(int v=0;v<Vec;++v) {
     float up=float(u.val[v]);
     if(limit>0)up=fminf(fmaxf(up,-limit),limit);
     float gate_value;
     if constexpr(Direct) {
      float gv=float(g.val[v]);if(limit>0)gv=fminf(gv,limit);
      gate_value=gv*(1.0f/(1.0f+__expf(-1.702f*gv)));
     } else gate_value=gate_table[g.val[v].x];
     c10::BFloat16 value(gate_value*(up+beta));
     values[seg*Vec+v]=value.x;
     maximum=fmaxf(maximum,fabsf(float(value)));
    }
   }
  }
#pragma unroll
  for(int d=32;d;d>>=1)maximum=fmaxf(maximum,__shfl_xor(maximum,d,64));
  if(tid%64==0)partial[tid/64]=maximum;
  __syncthreads();
#pragma unroll
  for(int w=0;w<Waves;++w)maximum=fmaxf(maximum,partial[w]);
  const float scale=fmaxf(maximum,1e-10f)/127.0f;
  const uint16_t max_bits=c10::BFloat16(maximum).x;
 const float inv_scale=__fdiv_rn(1.0f,scale);
  if(tid==0)scales[row]=scale;
#pragma unroll
  for(int seg=0;seg<Segments;++seg) {
   const int col=(seg*Threads+tid)*Vec;
   if(col<width) {
    Out out;
#pragma unroll
    for(int v=0;v<Vec;++v)
     out.val[v]=quantize_bits(values[seg*Vec+v],max_bits,maximum,scale,inv_scale,quant_table);
    *reinterpret_cast<Out*>(q+row*width+col)=out;
   }
  }
  __syncthreads(); // Protect partial[] before the next live row.
 }
}

void check_tables(const torch::Tensor& x,const torch::Tensor& q,const torch::Tensor& s,
                  const torch::Tensor& gt,const torch::Tensor& qt) {
 TORCH_CHECK(gt.device()==x.device()&&qt.device()==x.device()&&
  gt.scalar_type()==at::kFloat&&qt.scalar_type()==at::kByte&&
  gt.numel()==65536&&qt.numel()==QuantTableSize&&
  gt.is_contiguous()&&qt.is_contiguous(),"Invalid immutable quantization tables");
 TORCH_CHECK(!gt.is_alias_of(x)&&!gt.is_alias_of(q)&&!gt.is_alias_of(s)&&
  !qt.is_alias_of(x)&&!qt.is_alias_of(q)&&!qt.is_alias_of(s)&&
  !gt.is_alias_of(qt),"Tables must not alias other buffers");
}
template<int Bound,int Vec>
void launch_contig(const torch::Tensor& x,torch::Tensor& q,torch::Tensor& s,
                   const torch::Tensor& gt,const torch::Tensor& qt,double beta,double limit) {
 const auto rows=x.size(0);
 swiglu_oai_int8_quant_contiguous_tables_kernel<Bound,Vec>
  <<<(rows+3)/4,256,0,at::cuda::getCurrentCUDAStream()>>>(
   x.data_ptr<c10::BFloat16>(),q.data_ptr<int8_t>(),s.data_ptr<float>(),
   gt.data_ptr<float>(),qt.data_ptr<uint8_t>(),rows,x.size(1)/2,beta,limit);
}
template<int Vec>
void dispatch_contig(const torch::Tensor& x,torch::Tensor& q,torch::Tensor& s,
                     const torch::Tensor& gt,const torch::Tensor& qt,double beta,double limit) {
 const auto h=x.size(-1)/2;
#define CONTIG(B) launch_contig<B,Vec>(x,q,s,gt,qt,beta,limit)
 if(h<=512) CONTIG(512);else if(h<=1024) CONTIG(1024);
 else if(h<=2048) CONTIG(2048);else if(h<=3072) CONTIG(3072);
 else if(h<=4096) CONTIG(4096);else CONTIG(8192);
#undef CONTIG
}
template<int Bound,int Threads>
void launch_flat(const torch::Tensor& x,torch::Tensor& q,torch::Tensor& s,
                 const torch::Tensor& counts,const torch::Tensor& gt,
                 const torch::Tensor& qt,double beta,double limit,int blocks) {
 swiglu_oai_int8_quant_masked_flat_tables_kernel<Bound,Threads>
  <<<blocks,Threads,0,at::cuda::getCurrentCUDAStream()>>>(
   x.data_ptr<c10::BFloat16>(),q.data_ptr<int8_t>(),s.data_ptr<float>(),
   counts.data_ptr<int32_t>(),gt.data_ptr<float>(),qt.data_ptr<uint8_t>(),
   x.size(1),x.size(2)/2,beta,limit);
}
template<int Threads>
void dispatch_flat(const torch::Tensor& x,torch::Tensor& q,torch::Tensor& s,
                   const torch::Tensor& counts,const torch::Tensor& gt,
                   const torch::Tensor& qt,double beta,double limit,int blocks) {
 const auto h=x.size(-1)/2;
#define FLAT(B) launch_flat<B,Threads>(x,q,s,counts,gt,qt,beta,limit,blocks)
 if(h<=512) FLAT(512);else if(h<=1024) FLAT(1024);
 else if(h<=2048) FLAT(2048);else if(h<=3072) FLAT(3072);
 else if(h<=4096) FLAT(4096);else FLAT(8192);
#undef FLAT
}

template<int Threads,bool Direct=false>
void launch_contiguous_block(const torch::Tensor& x,torch::Tensor& q,torch::Tensor& s,
 const torch::Tensor& gt,const torch::Tensor& qt,double beta,double limit,int blocks) {
 swiglu_oai_int8_quant_masked_flat_tables_kernel<3072,Threads,false,Direct>
  <<<blocks,Threads,0,at::cuda::getCurrentCUDAStream()>>>(
   x.data_ptr<c10::BFloat16>(),q.data_ptr<int8_t>(),s.data_ptr<float>(),nullptr,
   gt.data_ptr<float>(),qt.data_ptr<uint8_t>(),x.size(0),x.size(1)/2,beta,limit);
}
} // namespace
void swiglu_oai_build_int8_quant_table(torch::Tensor& table) {
 TORCH_CHECK(table.is_cuda()&&table.scalar_type()==at::kByte&&
  table.is_contiguous()&&table.numel()==QuantTableSize,"Invalid quant table buffer");
 const c10::cuda::CUDAGuard guard(table.device());
 build_int8_mantissa_table_kernel<<<(QuantTableSize+255)/256,256,0,
  at::cuda::getCurrentCUDAStream()>>>(table.data_ptr<uint8_t>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void swiglu_oai_int8_quant_contiguous_tables(
 const torch::Tensor& x,torch::Tensor& q,torch::Tensor& s,
 const torch::Tensor& gt,const torch::Tensor& qt,double alpha,double beta,double limit) {
 swiglu_oai_int8_quant_validate(x,q,s,alpha,beta,limit);check_tables(x,q,s,gt,qt);
 TORCH_CHECK(x.dim()==2,"Expected contiguous 2D input");
 if(x.numel()==0)return;
 const c10::cuda::CUDAGuard guard(x.device());
 if(x.size(1)/2%8==0&&reinterpret_cast<uintptr_t>(x.data_ptr())%16==0&&
    reinterpret_cast<uintptr_t>(q.data_ptr())%8==0)dispatch_contig<8>(x,q,s,gt,qt,beta,limit);
 else dispatch_contig<1>(x,q,s,gt,qt,beta,limit);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// Live-only API: output/scales padding is UNTOUCHED. Caller must prove that
// downstream computation does not consume it; not a drop-in padding contract.
void swiglu_oai_int8_quant_masked_flat_tables(
 const torch::Tensor& x,torch::Tensor& q,torch::Tensor& s,const torch::Tensor& counts,
 const torch::Tensor& gt,const torch::Tensor& qt,double alpha,double beta,double limit,
 int64_t threads,int64_t blocks) {
 swiglu_oai_int8_quant_validate(x,q,s,alpha,beta,limit);check_tables(x,q,s,gt,qt);
 TORCH_CHECK(x.dim()==3&&x.size(0)==16&&x.size(1)<(INT32_MAX-512)/16&&
  x.size(2)/2%8==0&&reinterpret_cast<uintptr_t>(x.data_ptr())%16==0&&
  reinterpret_cast<uintptr_t>(q.data_ptr())%8==0,"Invalid flat shape/alignment");
 TORCH_CHECK(counts.device()==x.device()&&counts.scalar_type()==at::kInt&&
  counts.is_contiguous()&&counts.dim()==1&&counts.numel()==16&&
  !counts.is_alias_of(x)&&!counts.is_alias_of(q)&&!counts.is_alias_of(s)&&
  !counts.is_alias_of(gt)&&!counts.is_alias_of(qt),"Invalid counts");
 TORCH_CHECK((threads==64||threads==128||threads==256)&&blocks>0&&blocks<=512,
  "Invalid flat launch configuration");
 if(x.numel()==0)return;
 const c10::cuda::CUDAGuard guard(x.device());
 if(threads==64)dispatch_flat<64>(x,q,s,counts,gt,qt,beta,limit,blocks);
 else if(threads==128)dispatch_flat<128>(x,q,s,counts,gt,qt,beta,limit,blocks);
 else dispatch_flat<256>(x,q,s,counts,gt,qt,beta,limit,blocks);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void swiglu_oai_int8_quant_contiguous_block_tables(
 const torch::Tensor& x,torch::Tensor& q,torch::Tensor& s,
 const torch::Tensor& gt,const torch::Tensor& qt,double alpha,double beta,double limit,
 int64_t threads,int64_t blocks) {
 swiglu_oai_int8_quant_validate(x,q,s,alpha,beta,limit);check_tables(x,q,s,gt,qt);
 TORCH_CHECK(x.dim()==2&&x.size(1)==6144&&
  reinterpret_cast<uintptr_t>(x.data_ptr())%16==0&&
  reinterpret_cast<uintptr_t>(q.data_ptr())%8==0,"Expected aligned H3072");
 TORCH_CHECK((threads==64||threads==128||threads==256)&&blocks>0&&blocks<=65536,"Invalid launch");
 if(x.numel()==0)return;
 const c10::cuda::CUDAGuard guard(x.device());
 if(threads==64)launch_contiguous_block<64>(x,q,s,gt,qt,beta,limit,blocks);
 else if(threads==128)launch_contiguous_block<128>(x,q,s,gt,qt,beta,limit,blocks);
 else launch_contiguous_block<256>(x,q,s,gt,qt,beta,limit,blocks);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void swiglu_oai_int8_quant_contiguous_direct(
 const torch::Tensor& x,torch::Tensor& q,torch::Tensor& s,
 const torch::Tensor& gt,const torch::Tensor& qt,double alpha,double beta,double limit,
 int64_t threads,int64_t blocks) {
 TORCH_CHECK(alpha==1.702,"Direct specialization requires alpha=1.702");
 swiglu_oai_int8_quant_validate(x,q,s,alpha,beta,limit);check_tables(x,q,s,gt,qt);
 TORCH_CHECK(x.dim()==2&&x.size(1)==6144&&
  reinterpret_cast<uintptr_t>(x.data_ptr())%16==0&&
  reinterpret_cast<uintptr_t>(q.data_ptr())%8==0,"Expected aligned H3072");
 TORCH_CHECK((threads==64||threads==128||threads==256)&&blocks>0&&blocks<=65536,"Invalid launch");
 if(x.numel()==0)return;
 const c10::cuda::CUDAGuard guard(x.device());
 if(threads==64)launch_contiguous_block<64,true>(x,q,s,gt,qt,beta,limit,blocks);
 else if(threads==128)launch_contiguous_block<128,true>(x,q,s,gt,qt,beta,limit,blocks);
 else launch_contiguous_block<256,true>(x,q,s,gt,qt,beta,limit,blocks);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace at::native
