// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// LMSlim-compatible BF16 per-token INT8, specialized wide-row schedules.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

namespace at::native {
namespace {
__device__ __forceinline__ uint32_t umax(uint32_t a,uint32_t b){return a>b?a:b;}
template<int D, int Threads>
__global__ __launch_bounds__(Threads)
void per_token_quant_int8_bf16_wide_kernel(
    const c10::BFloat16* __restrict__ x,
    int8_t* __restrict__ q, float* __restrict__ scales) {
  constexpr int V=Threads>=256?2:8, Wave=64, Segments=(D+Threads*V-1)/(Threads*V);
  using In=at::native::memory::aligned_vector<uint32_t,V/2>;
  using Out=at::native::memory::aligned_vector<int8_t,V>;
  const int row=blockIdx.x, tid=threadIdx.x;
  In values[Segments];
  uint32_t max_bits=0;
#pragma unroll
  for(int s=0;s<Segments;s++) {
    const int col=(s*Threads+tid)*V;
    if(col<D) {
      values[s]=*reinterpret_cast<const In*>(x+int64_t(row)*D+col);
#pragma unroll
      for(int i=0;i<V/2;i++) {
        max_bits=umax(max_bits,values[s].val[i]&0x7fffu);
        max_bits=umax(max_bits,(values[s].val[i]>>16)&0x7fffu);
      }
    }
  }
#pragma unroll
  for(int offset=32;offset;offset>>=1)
    max_bits=umax(max_bits,__shfl_xor(max_bits,offset,Wave));
  __shared__ uint32_t partial[Threads/Wave];
  if constexpr(Threads>Wave) {
    if((tid&63)==0) partial[tid/64]=max_bits;
    __syncthreads();
    max_bits=0;
#pragma unroll
    for(int i=0;i<Threads/Wave;i++) max_bits=umax(max_bits,partial[i]);
  }
  const float maximum=max_bits>0x7f80u ? __uint_as_float(0x7fc00000u) : fmaxf(__uint_as_float(max_bits<<16),1.0e-10f);
  const float inverse=127.0f/maximum;
  if(tid==0) scales[row]=maximum/127.0f;
#pragma unroll
  for(int s=0;s<Segments;s++) {
    const int col=(s*Threads+tid)*V;
    if(col<D) {
      Out packed;
#pragma unroll
      for(int i=0;i<V/2;i++) {
        const uint32_t raw=values[s].val[i];
        packed.val[i*2]=static_cast<int8_t>(nearbyintf(__uint_as_float(raw<<16)*inverse));
        packed.val[i*2+1]=static_cast<int8_t>(nearbyintf(__uint_as_float(raw&0xffff0000u)*inverse));
      }
      *reinterpret_cast<Out*>(q+int64_t(row)*D+col)=packed;
    }
  }
  return;
}
bool overlaps(const torch::Tensor&a,const torch::Tensor&b) {
  const auto ap=reinterpret_cast<uintptr_t>(a.data_ptr());
  const auto bp=reinterpret_cast<uintptr_t>(b.data_ptr());
  return ap<bp+b.numel()*b.element_size() && bp<ap+a.numel()*a.element_size();
}
}
void per_token_quant_int8_minimax_out(torch::Tensor& out,
    const torch::Tensor& input,torch::Tensor& scales,int64_t threads) {
  TORCH_CHECK(input.is_cuda()&&out.is_cuda()&&scales.is_cuda(),"GPU tensors required");
  TORCH_CHECK(input.dim()>=2,"input must have at least two dimensions");
  TORCH_CHECK(input.is_contiguous()&&out.is_contiguous()&&scales.is_contiguous(),"contiguous tensors required");
  TORCH_CHECK(input.scalar_type()==at::kBFloat16&&out.scalar_type()==at::kChar&&scales.scalar_type()==at::kFloat,"expected BF16 input, INT8 output, FP32 scales");
  const int64_t d=input.size(-1);
  TORCH_CHECK(d==3072||d==6144||d==8192||d==12288,"unsupported hidden width");
  const int64_t rows=input.numel()/d;
  TORCH_CHECK(out.sizes()==input.sizes(),"output shape must match input");
  TORCH_CHECK(scales.dim()==input.dim()&&scales.size(-1)==1&&scales.numel()==rows,"invalid scale shape");
  for(int i=0;i<input.dim()-1;i++) TORCH_CHECK(scales.size(i)==input.size(i),"invalid scale prefix");
  TORCH_CHECK(input.device()==out.device()&&input.device()==scales.device(),"device mismatch");
  TORCH_CHECK(!overlaps(input,out)&&!overlaps(input,scales)&&!overlaps(out,scales),"output tensors must not alias");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr())%16==0&&reinterpret_cast<uintptr_t>(out.data_ptr())%8==0,"unaligned input/output");
  TORCH_CHECK(threads==64||threads==128||threads==256||threads==512,"invalid thread count");
  if(rows==0)return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const auto stream=at::cuda::getCurrentCUDAStream();
#define RUN(D,T) per_token_quant_int8_bf16_wide_kernel<D,T><<<rows,T,0,stream>>>(input.data_ptr<c10::BFloat16>(),out.data_ptr<int8_t>(),scales.data_ptr<float>())
#define WIDTH(D) case D: if(threads==64){RUN(D,64);}else if(threads==128){RUN(D,128);}else if(threads==256){RUN(D,256);}else{RUN(D,512);} break
  switch(d){WIDTH(3072);WIDTH(6144);WIDTH(8192);WIDTH(12288);}
#undef WIDTH
#undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace at::native
