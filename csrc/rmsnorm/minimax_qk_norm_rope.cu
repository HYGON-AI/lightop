// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <hip/hip_runtime.h>
#include <array>
#include <cmath>

// Preserve the separate FP32 operations used by the validated QK build.
// FMA contraction can cross a BF16 rounding boundary before rotary encoding.
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace at::native {
namespace {
using Bf16=at::BFloat16;
struct QKParams {
  const Bf16 *x[4], *weight[4];
  Bf16 *out[4];
  int64_t row_stride[4], col_stride[4], rows;
  int heads[4], rotary;
  float eps;
};
template<typename T,int N> struct alignas(16) QKPack{T value[N];};

template<int HeadsPerBlock,bool Neox,typename Cos,typename Pos>
__global__ __launch_bounds__(16*HeadsPerBlock)
void minimax_qk_norm_rope_vec8_kernel(QKParams p,const Cos* cs,const Pos* positions){
  const int lane=threadIdx.x%16;
  const int64_t linear=int64_t(blockIdx.x)*HeadsPerBlock+threadIdx.x/16;
  const int64_t row=linear/73;if(row>=p.rows)return;
  int head=linear-row*73,part=0;
  if(head>=64){head-=64;part=1;if(head>=4){head-=4;part=2;if(head>=4){head-=4;part=3;}}}
  const int count=part==0?64:(part==3?1:4);
  const Bf16* src=p.x[part]+row*p.row_stride[part]+head*128;
  auto x=*reinterpret_cast<const QKPack<Bf16,8>*>(src+lane*8);
  auto w=*reinterpret_cast<const QKPack<Bf16,8>*>(p.weight[part]+lane*8);
  float sum=0.f;
#pragma unroll
  for(int i=0;i<8;++i){float a=float(x.value[i]);sum+=a*a;}
#pragma unroll
  for(int d=8;d;d/=2)sum+=__shfl_xor(sum,d,16);
  const float inv=rsqrtf(sum*(1.f/128.f)+p.eps);
  QKPack<Bf16,8> norm,partner,out;
#pragma unroll
  for(int i=0;i<8;++i)norm.value[i]=Bf16((float(x.value[i])*inv)*(1.f+float(w.value[i])));
  if constexpr(Neox){
#pragma unroll
    for(int i=0;i<4;++i){
      unsigned bits=unsigned(norm.value[2*i].x)|(unsigned(norm.value[2*i+1].x)<<16);
      bits=__shfl_xor(bits,8,16);partner.value[2*i].x=bits&0xffff;partner.value[2*i+1].x=bits>>16;
    }
    const Cos* cache=cs+int64_t(positions[row])*128;
    auto c=*reinterpret_cast<const QKPack<Cos,8>*>(cache+(lane%8)*8);
    auto s=*reinterpret_cast<const QKPack<Cos,8>*>(cache+64+(lane%8)*8);
    const float sign=lane<8?-1.f:1.f;
#pragma unroll
    for(int i=0;i<8;++i)out.value[i]=Bf16(float(norm.value[i])*float(c.value[i])+(sign*float(partner.value[i]))*float(s.value[i]));
  }else{
    const Cos* cache=cs+int64_t(positions[row])*128;
#pragma unroll
    for(int i=0;i<8;++i){int angle=(lane*8+i)/2;float sign=(i&1)?1.f:-1.f;out.value[i]=Bf16(float(norm.value[i])*float(cache[angle])+(sign*float(norm.value[i^1]))*float(cache[64+angle]));}
  }
  *reinterpret_cast<QKPack<Bf16,8>*>(p.out[part]+(row*count+head)*128+lane*8)=out;
}


// Candidate v6: actual M3.1 rotary_dim=64, head_dim=128. Sixteen lanes
// own a head, eight contiguous BF16 values per lane. Only the first eight
// lanes rotate; lanes 8..15 preserve the BF16-rounded normalized values.
template<int HeadsPerBlock,bool Neox,typename Cos,typename Pos>
__global__ __launch_bounds__(16*HeadsPerBlock)
void minimax_qk_norm_rope_rotary64_vec8_kernel(QKParams p,const Cos* cs,const Pos* positions){
  const int lane=threadIdx.x%16;
  const int64_t linear=int64_t(blockIdx.x)*HeadsPerBlock+threadIdx.x/16;
  const int64_t row=linear/73;if(row>=p.rows)return;
  int head=linear-row*73,part=0;
  if(head>=64){head-=64;part=1;if(head>=4){head-=4;part=2;if(head>=4){head-=4;part=3;}}}
  const int count=part==0?64:(part==3?1:4);
  const Bf16* src=p.x[part]+row*p.row_stride[part]+head*128;
  const auto x=*reinterpret_cast<const QKPack<Bf16,8>*>(src+lane*8);
  const auto w=*reinterpret_cast<const QKPack<Bf16,8>*>(p.weight[part]+lane*8);
  float sum=0.f;
#pragma unroll
  for(int i=0;i<8;++i){const float a=float(x.value[i]);sum+=a*a;}
#pragma unroll
  for(int d=8;d;d/=2)sum+=__shfl_xor(sum,d,16);
  const float inv=rsqrtf(sum*(1.f/128.f)+p.eps);
  QKPack<Bf16,8> norm,out;
#pragma unroll
  for(int i=0;i<8;++i)norm.value[i]=Bf16((float(x.value[i])*inv)*(1.f+float(w.value[i])));
  out=norm;
  if constexpr(Neox){
    QKPack<Bf16,8> partner;
    // Keep all 16 lanes participating in the shuffle; only lanes 0..7
    // consume partners. Within that half, xor4 exchanges columns 0..31
    // with 32..63 without the generic kernel's dynamic value selection.
#pragma unroll
    for(int i=0;i<4;++i){
      unsigned bits=unsigned(norm.value[2*i].x)|(unsigned(norm.value[2*i+1].x)<<16);
      bits=__shfl_xor(bits,4,16);
      partner.value[2*i].x=bits&0xffff;partner.value[2*i+1].x=bits>>16;
    }
    if(lane<8){
      const Cos* cache=cs+int64_t(positions[row])*64;
      const auto c=*reinterpret_cast<const QKPack<Cos,8>*>(cache+(lane%4)*8);
      const auto s=*reinterpret_cast<const QKPack<Cos,8>*>(cache+32+(lane%4)*8);
      const float sign=lane<4?-1.f:1.f;
#pragma unroll
      for(int i=0;i<8;++i)out.value[i]=Bf16(float(norm.value[i])*float(c.value[i])+(sign*float(partner.value[i]))*float(s.value[i]));
    }
  }else if(lane<8){
    const Cos* cache=cs+int64_t(positions[row])*64;
#pragma unroll
    for(int i=0;i<8;++i){
      const int angle=(lane*8+i)/2;
      const float sign=(i&1)?1.f:-1.f;
      out.value[i]=Bf16(float(norm.value[i])*float(cache[angle])+(sign*float(norm.value[i^1]))*float(cache[32+angle]));
    }
  }
  *reinterpret_cast<QKPack<Bf16,8>*>(p.out[part]+(row*count+head)*128+lane*8)=out;
}

// A subwave owns one complete head. Its normed values stay in registers,
// including the BF16 rounding required before the RoPE partner exchange.
template<int Lanes,int HeadsPerBlock,bool Neox,typename Cos,typename Pos>
__global__ __launch_bounds__(Lanes*HeadsPerBlock)
void minimax_qk_norm_rope_kernel(QKParams p,const Cos* cs,const Pos* positions) {
  constexpr int Values=128/Lanes;
  const int lane=threadIdx.x%Lanes;
  constexpr int total_heads=73; // Public launcher guards the M3.1 head geometry.
  const int64_t linear=int64_t(blockIdx.x)*HeadsPerBlock+threadIdx.x/Lanes;
  const int64_t row=linear/total_heads;
  if(row>=p.rows)return;
  int head=linear-row*total_heads,part=0;
#pragma unroll
  for(int i=0;i<3;++i){if(head>=p.heads[i] && part==i){head-=p.heads[i];++part;}}
  const Bf16* src=p.x[part]+row*p.row_stride[part]+head*128*p.col_stride[part];
  const Bf16* wt=p.weight[part];
  float value[Values],norm[Values],sum=0.f;
#pragma unroll
  for(int i=0;i<Values;++i){value[i]=float(src[(lane+i*Lanes)*p.col_stride[part]]);sum+=value[i]*value[i];}
#pragma unroll
  for(int d=Lanes/2;d;d/=2)sum+=__shfl_xor(sum,d,Lanes);
  const float inv=rsqrtf(sum*(1.f/128.f)+p.eps);
#pragma unroll
  for(int i=0;i<Values;++i){
    float w=float(wt[lane+i*Lanes]);
    norm[i]=float(Bf16((value[i]*inv)*(1.f+w)));
  }
  const int half=p.rotary/2;
  const int64_t pos=int64_t(positions[row]);
  const Cos* cache=cs+pos*p.rotary;
  Bf16* dst=p.out[part]+(row*p.heads[part]+head)*128;
#pragma unroll
  for(int i=0;i<Values;++i){
    const int col=lane+i*Lanes;
    float result=norm[i];
    // All lanes execute the shuffles even for partial rotary dimensions.
    const int partner=Neox ? (col<half?col+half:col-half) : (col^1);
    const int source_lane=(partner>=0?partner:0)%Lanes;
    const int source_value=(partner>=0?partner:0)/Lanes;
    float other=0.f;
    if constexpr(Neox){
      if(p.rotary==128){other=norm[(i+Values/2)%Values];}
      else {
#pragma unroll
        for(int j=0;j<Values;++j){float v=__shfl(norm[j],source_lane,Lanes);if(source_value==j)other=v;}
      }
    } else {other=__shfl_xor(norm[i],1,Lanes);}
    if(col<p.rotary){
      const int angle=Neox?(col<half?col:col-half):col/2;
      const float c=float(cache[angle]),s=float(cache[half+angle]);
      const float sign=Neox?(col<half?-1.f:1.f):((col&1)?1.f:-1.f);
      result=norm[i]*c+(sign*other)*s;
    }
    dst[col]=Bf16(result);
  }
}

template<typename Cos,typename Pos>
void launch_qk(QKParams p,const Cos* cs,const Pos* positions,bool neox,int variant,hipStream_t stream){
  const int64_t count=p.rows*(p.heads[0]+p.heads[1]+p.heads[2]+1);
#define RUN(L,H) if(neox){hipLaunchKernelGGL((minimax_qk_norm_rope_kernel<L,H,true>),dim3((count+H-1)/H),dim3(L*H),0,stream,p,cs,positions);}else{hipLaunchKernelGGL((minimax_qk_norm_rope_kernel<L,H,false>),dim3((count+H-1)/H),dim3(L*H),0,stream,p,cs,positions);}
#define VECTOR(H) if(neox){hipLaunchKernelGGL((minimax_qk_norm_rope_vec8_kernel<H,true>),dim3((count+H-1)/H),dim3(16*H),0,stream,p,cs,positions);}else{hipLaunchKernelGGL((minimax_qk_norm_rope_vec8_kernel<H,false>),dim3((count+H-1)/H),dim3(16*H),0,stream,p,cs,positions);}
  switch(variant){case 0:{RUN(64,4);break;}case 1:{RUN(32,4);break;}case 2:{RUN(64,8);break;}case 3:{RUN(32,8);break;}case 4:{VECTOR(8);break;}case 5:{VECTOR(16);break;}case 6:{if(neox){hipLaunchKernelGGL((minimax_qk_norm_rope_rotary64_vec8_kernel<16,true>),dim3((count+15)/16),dim3(256),0,stream,p,cs,positions);}else{hipLaunchKernelGGL((minimax_qk_norm_rope_rotary64_vec8_kernel<16,false>),dim3((count+15)/16),dim3(256),0,stream,p,cs,positions);}break;}default:TORCH_CHECK(false,"unknown QK kernel variant");}
#undef RUN
#undef VECTOR
}
} // namespace

std::vector<torch::Tensor> minimax_qk_norm_rope(
    torch::Tensor q,torch::Tensor k,torch::Tensor iq,torch::Tensor ik,
    torch::Tensor qw,torch::Tensor kw,torch::Tensor iqw,torch::Tensor ikw,
    torch::Tensor positions,torch::Tensor cs,double epsilon,int64_t rotary,bool neox,int64_t variant){
  const std::array<torch::Tensor,4> xs={q,k,iq,ik},ws={qw,kw,iqw,ikw};
  QKParams p{};p.rows=q.size(0);p.rotary=rotary;p.eps=float(epsilon);
  TORCH_CHECK(q.is_cuda() && q.dim()==2 && q.scalar_type()==at::kBFloat16,"Q must be CUDA BF16 [rows, heads*128]");
  TORCH_CHECK(rotary>=0 && rotary<=128 && rotary%2==0,"rotary dimension must be even and within [0,128]");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon>0 && std::isfinite(p.eps) && p.eps>0,"epsilon must be positive finite FP32");
  TORCH_CHECK(variant>=0 && variant<7,"unsupported kernel variant");
  std::vector<torch::Tensor> outputs;
  for(int i=0;i<4;++i){
    TORCH_CHECK(xs[i].device()==q.device() && xs[i].dtype()==q.dtype() && xs[i].dim()==2 && xs[i].size(0)==p.rows && xs[i].size(1)>0 && xs[i].size(1)%128==0,"Q/K/index inputs must share BF16 device and rows, with head_dim=128");
    TORCH_CHECK(ws[i].device()==q.device() && ws[i].scalar_type()==at::kBFloat16 && ws[i].dim()==1 && ws[i].numel()==128 && ws[i].is_contiguous(),"Gemma weights must be contiguous BF16 [128]");
    p.heads[i]=xs[i].size(1)/128;p.row_stride[i]=xs[i].stride(0);p.col_stride[i]=xs[i].stride(1);
    p.x[i]=xs[i].data_ptr<Bf16>();p.weight[i]=ws[i].data_ptr<Bf16>();
    outputs.push_back(torch::empty(xs[i].sizes(),xs[i].options()));p.out[i]=outputs.back().data_ptr<Bf16>();
  }
  TORCH_CHECK(p.heads[3]==1,"index K must have one head");
  TORCH_CHECK(p.heads[0]==64 && p.heads[1]==4 && p.heads[2]==4,"optimized M3.1 path requires Q64/K4/indexQ4/indexK1; other geometries use caller fallback");
  if(variant>=4){
    bool vector_ok=(variant==6?rotary==64:rotary==128);
    for(int i=0;i<4;++i)vector_ok=vector_ok && p.col_stride[i]==1 && p.row_stride[i]%8==0 && reinterpret_cast<uintptr_t>(p.x[i])%16==0 && reinterpret_cast<uintptr_t>(p.weight[i])%16==0;
    if(variant==6)vector_ok=vector_ok && reinterpret_cast<uintptr_t>(cs.data_ptr())%16==0;
    if(!vector_ok)variant=3;
  }
  TORCH_CHECK(positions.device()==q.device() && positions.dim()==1 && positions.numel()==p.rows && positions.is_contiguous() && (positions.scalar_type()==at::kLong || positions.scalar_type()==at::kInt),"positions must be contiguous device int32/int64 [rows]");
  TORCH_CHECK(cs.device()==q.device() && cs.dim()==2 && cs.size(1)==rotary && cs.is_contiguous() && (cs.scalar_type()==at::kFloat || cs.scalar_type()==at::kBFloat16),"cos/sin cache must be contiguous FP32/BF16 [positions,rotary_dim]");
  if(!p.rows)return outputs;
  c10::cuda::CUDAGuard guard(q.device());auto stream=at::cuda::getCurrentCUDAStream();
#define POSITION(C) if(positions.scalar_type()==at::kLong)launch_qk<C,int64_t>(p,cs.data_ptr<C>(),positions.data_ptr<int64_t>(),neox,variant,stream);else launch_qk<C,int32_t>(p,cs.data_ptr<C>(),positions.data_ptr<int32_t>(),neox,variant,stream)
  if(cs.scalar_type()==at::kFloat){POSITION(float);}else{POSITION(Bf16);}
#undef POSITION
  C10_CUDA_KERNEL_LAUNCH_CHECK();return outputs;
}
} // namespace at::native
