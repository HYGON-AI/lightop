// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Exact local E4M3FN KV quantization before bytewise CP all-gather.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Float8_e4m3fn.h>
#include <hip/hip_runtime.h>
#include <cmath>
namespace at::native {
namespace {
using Bf16=at::BFloat16;
template<typename T,int N>struct alignas(16) Pack{T x[N];};
__global__ __launch_bounds__(256)
void minimax_cp_pack_fp8_kv_bf16_index_kernel(const Bf16* k,const Bf16* v,const Bf16* ik,
    uint8_t* out,int64_t rows,int64_t ks,int64_t vs,int64_t is,float ki,float vi,bool apply_k,bool apply_v){
 const int lane=threadIdx.x%64;
 const int64_t row=int64_t(blockIdx.x)*4+threadIdx.x/64;
 if(row>=rows)return;
 const auto a=*reinterpret_cast<const Pack<Bf16,8>*>(k+row*ks+lane*8);
 const auto b=*reinterpret_cast<const Pack<Bf16,8>*>(v+row*vs+lane*8);
 uint64_t ka=0,va=0;
#pragma unroll
 for(int i=0;i<8;++i){
  const Bf16 ak=apply_k?Bf16(float(a.x[i])*ki):a.x[i];
  const Bf16 av=apply_v?Bf16(float(b.x[i])*vi):b.x[i];
  float x=float(ak),y=float(av);
  if(x==x)x=fminf(448.f,fmaxf(-448.f,x));
  if(y==y)y=fminf(448.f,fmaxf(-448.f,y));
  ka|=uint64_t(c10::Float8_e4m3fn(x).x)<<(8*i);
  va|=uint64_t(c10::Float8_e4m3fn(y).x)<<(8*i);
 }
 *reinterpret_cast<uint64_t*>(out+row*1280+lane*8)=ka;
 *reinterpret_cast<uint64_t*>(out+row*1280+512+lane*8)=va;
 if(lane<16){
  const auto index=*reinterpret_cast<const Pack<Bf16,8>*>(ik+row*is+lane*8);
  *reinterpret_cast<Pack<Bf16,8>*>(out+row*1280+1024+lane*16)=index;
 }
}
}
torch::Tensor minimax_cp_fp8_pack(torch::Tensor k,torch::Tensor v,torch::Tensor ik,double kscale,double vscale,bool apply_k,bool apply_v){
 TORCH_CHECK(k.is_cuda() && k.dim()==2 && k.scalar_type()==at::kBFloat16 && k.size(1)==512,"K must be BF16 [rows,512]");
 for(const auto& x:{v,ik})TORCH_CHECK(x.device()==k.device() && x.dtype()==k.dtype() && x.dim()==2 && x.size(0)==k.size(0),"V/index K must share BF16 device/rows");
 TORCH_CHECK(v.size(1)==512 && ik.size(1)==128,"V512/indexK128 required");
 for(const auto& x:{k,v,ik})TORCH_CHECK(x.stride(1)==1 && x.stride(0)>=x.size(1) && x.stride(0)%8==0 && reinterpret_cast<uintptr_t>(x.data_ptr())%16==0,"aligned row-strided input required");
 const float ki=1.f/float(kscale),vi=1.f/float(vscale);
 TORCH_CHECK(std::isfinite(kscale) && kscale>0 && std::isfinite(vscale) && vscale>0 && std::isfinite(ki) && std::isfinite(vi),"positive finite inverse scales required");
 c10::cuda::CUDAGuard guard(k.device());auto out=torch::empty({k.size(0),1280},k.options().dtype(at::kByte));
 if(k.size(0)){
  auto stream=at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(minimax_cp_pack_fp8_kv_bf16_index_kernel,dim3((k.size(0)+3)/4),dim3(256),0,stream,k.data_ptr<Bf16>(),v.data_ptr<Bf16>(),ik.data_ptr<Bf16>(),out.data_ptr<uint8_t>(),k.size(0),k.stride(0),v.stride(0),ik.stride(0),ki,vi,apply_k,apply_v);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
 }
 return out;
}
}
