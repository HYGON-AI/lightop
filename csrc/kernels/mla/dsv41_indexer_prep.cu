// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <hip/hip_runtime.h>

namespace {
template<typename Index>
__global__ void indexed_rope_fp4_kernel(const at::BFloat16* __restrict__ x,
    const float* __restrict__ freqs, const Index* __restrict__ positions,
    at::BFloat16* __restrict__ y, int64_t rows, int heads) {
  // Eight independent heads per CTA, four contiguous BF16 values per lane.
  const int64_t row = int64_t(blockIdx.x) * 8 + threadIdx.x / 32;
  if (row >= rows) return;
  const int column = (threadIdx.x % 32) * 4;
  const int64_t token = row / heads;
  union { uint2 u; at::BFloat16 v[4]; } values;
  if ((reinterpret_cast<uintptr_t>(x) & 7) == 0) {
    values.u = *reinterpret_cast<const uint2*>(x + row * 128 + column);
  } else {
#pragma unroll
    for (int j=0;j<4;++j) values.v[j]=x[row*128+column+j];
  }
  float a[4];
#pragma unroll
  for (int j=0;j<4;++j) a[j]=float(values.v[j]);
  if (column >= 64) {
    const int64_t fbase = int64_t(positions[token]) * 64 + column - 64;
#pragma unroll
    for (int j=0;j<4;j+=2) {
#pragma clang fp contract(off)
      const float real=a[j], imag=a[j+1], fr=freqs[fbase+j], fi=freqs[fbase+j+1];
      a[j] = float(at::BFloat16(real*fr - imag*fi));
      a[j+1] = float(at::BFloat16(fmaf(imag,fr,real*fi)));
    }
  }
  float maximum = fmaxf(fmaxf(fabsf(a[0]),fabsf(a[1])),fmaxf(fabsf(a[2]),fabsf(a[3])));
#pragma unroll
  for(int d=4;d;d>>=1) maximum=fmaxf(maximum,__shfl_xor(maximum,d,8));
  const float raw=fmaxf(maximum,6.f*0x1p-126f)*(1.f/6.f);
  const uint32_t bits=__float_as_uint(raw);
  const uint32_t exponent=((bits>>23)&255)+((bits&0x7fffff)!=0);
  const float scale=__uint_as_float(exponent<<23);
  const float inverse_scale=__uint_as_float((254u-exponent)<<23);
#pragma unroll
  for(int j=0;j<4;++j) {
    const float scaled=exponent < 255 ? a[j]*inverse_scale : a[j]/scale;
    const float z=fminf(6.f,fmaxf(-6.f,scaled));
    const float mag=fabsf(z), step=mag<2.f?.5f:(mag<4.f?1.f:2.f);
    values.v[j]=at::BFloat16(copysignf(nearbyintf(mag/step)*step,z)*scale);
  }
  *reinterpret_cast<uint2*>(y+row*128+column)=values.u;
}
}

at::Tensor dsv41_indexer_rope_fake_quant(const at::Tensor& x,
    const at::Tensor& freqs, const at::Tensor& positions) {
  TORCH_CHECK(x.is_cuda() && freqs.is_cuda() && positions.is_cuda(), "inputs must be on GPU");
  TORCH_CHECK(x.device()==freqs.device() && x.device()==positions.device(),"devices must match");
  TORCH_CHECK(x.scalar_type()==at::kBFloat16 && x.dim()==3 && x.size(2)==128 && x.is_contiguous(),"x must be contiguous BF16 [T,H,128]");
  TORCH_CHECK(freqs.scalar_type()==at::kComplexFloat && freqs.dim()==2 && freqs.size(1)==32 && freqs.is_contiguous(),"freqs must be contiguous complex64 [P,32]");
  TORCH_CHECK(positions.dim()==1 && positions.size(0)==x.size(0) && positions.is_contiguous() && (positions.scalar_type()==at::kInt || positions.scalar_type()==at::kLong),"positions must be contiguous int32/int64 [T]");
  TORCH_CHECK(x.size(1)>0 && x.size(1)<=65536,"invalid head count");
  c10::cuda::CUDAGuard guard(x.device());
  auto y=at::empty_like(x);
  const int64_t rows=x.size(0)*x.size(1);
  if(rows==0) return y;
  TORCH_CHECK(rows<=int64_t(2147483647)*8,"too many rows");
  const auto stream=at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_INDEX_TYPES(positions.scalar_type(),"dsv41_indexer_rope_fake_quant",[&]{
    indexed_rope_fp4_kernel<index_t><<<(rows+7)/8,256,0,stream>>>(x.data_ptr<at::BFloat16>(),reinterpret_cast<const float*>(freqs.data_ptr<c10::complex<float>>()),positions.data_ptr<index_t>(),y.data_ptr<at::BFloat16>(),rows,x.size(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("dsv41_indexer_rope_fake_quant",&dsv41_indexer_rope_fake_quant);
}
