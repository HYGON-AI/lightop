// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <hip/hip_runtime.h>
#include <c10/util/Float8_e4m3fn.h>
#include <array>
#include <cmath>
#include <type_traits>

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


struct QKCacheParams {
  const Bf16* value;
  const int64_t* locations;
  const int32_t* main_mapping;
  uint8_t* key_cache;
  uint8_t* value_cache;
  uint8_t* index_cache;
  uint8_t* query;
  uint8_t* index_query;
  int64_t value_row_stride, value_col_stride, mapping_size, main_slots, index_slots;
};

template<bool Saturate>
__device__ __forceinline__ void store_e4m3(
    uint8_t* dst, const QKPack<Bf16, 8>& input) {
  float values[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float value = float(input.value[i]);
    if constexpr (Saturate) {
      // Ordered comparisons preserve NaN and the sign of zero.
      value = value > 448.f ? 448.f : (value < -448.f ? -448.f : value);
    }
    values[i] = value;
  }
#if defined(__gfx938__)
  auto* packed = reinterpret_cast<uint32_t*>(dst);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int j = i * 4;
    unsigned int bits = __builtin_hcu_cvt_pk_fp8_f32(
        values[j], values[j + 1], 0, false);
    packed[i] = __builtin_hcu_cvt_pk_fp8_f32(
        values[j + 2], values[j + 3], bits, true);
  }
#else
#pragma unroll
  for (int i = 0; i < 8; ++i) dst[i] = c10::Float8_e4m3fn(values[i]).x;
#endif
}

// Candidate v6: actual M3.1 rotary_dim=64, head_dim=128. Sixteen lanes
// own a head, eight contiguous BF16 values per lane. Only the first eight
// lanes rotate; lanes 8..15 preserve the BF16-rounded normalized values.
template<int HeadsPerBlock, bool Neox, typename Cos, typename Pos,
         typename Cache = bool, bool Transposed = false>
__global__ __launch_bounds__(16*HeadsPerBlock)
void minimax_qk_norm_rope_rotary64_vec8_kernel(
    QKParams p, const Cos* cs, const Pos* positions, Cache cache = false) {
  const int lane=threadIdx.x%16;
  const int64_t linear=int64_t(blockIdx.x)*HeadsPerBlock+threadIdx.x/16;
  const int64_t row=Transposed ? int64_t(blockIdx.x)*16+threadIdx.x/16 : linear/73;
  int head=Transposed ? blockIdx.y : linear-row*73,part=0;
  if(head>=64){head-=64;part=1;if(head>=4){head-=4;part=2;if(head>=4){head-=4;part=3;}}}
  const int count=part==0?64:(part==3?1:4);
  QKPack<Bf16,8> x;
  QKPack<Bf16,8> values;
  if constexpr (Transposed) {
    static_assert(HeadsPerBlock == 16);
    __shared__ __align__(16) uint16_t tile[2*16*128];
    const Bf16* src = p.x[part] + int64_t(blockIdx.x)*16 + head*128*p.col_stride[part];
    const int rr = threadIdx.x % 16;
    const int stored_row = rr * 128;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int col = threadIdx.x / 16 + i*16;
      const int lds = stored_row + (col ^ (rr*8));
      tile[lds] = src[col*p.col_stride[part] + rr].x;
      if (part == 1) {
        const Bf16* val = cache.value + int64_t(blockIdx.x)*16 + head*128*cache.value_col_stride;
        tile[16*128+lds] = val[col*cache.value_col_stride + rr].x;
      }
    }
    __syncthreads();
    const int rlocal = threadIdx.x / 16;
    const int lds = rlocal*128 + ((lane*8) ^ (rlocal*8));
    x = *reinterpret_cast<const QKPack<Bf16,8>*>(tile+lds);
    if (part == 1) values = *reinterpret_cast<const QKPack<Bf16,8>*>(tile+16*128+lds);
  } else {
    if (row >= p.rows) return;
    const Bf16* src=p.x[part]+row*p.row_stride[part]+head*128;
    x=*reinterpret_cast<const QKPack<Bf16,8>*>(src+lane*8);
  }
  const auto w=*reinterpret_cast<const QKPack<Bf16,8>*>(p.weight[part]+lane*8);
  float inv;
  if constexpr (std::is_same_v<Cache, QKCacheParams>) {
    // Reproduce the deployed Triton wave64 reduction's 8,4,2,1,16,32,64
    // tree. With eight contiguous values per lane, the distance-eight
    // neighbors live in the adjacent lane and the next three levels are local.
    float paired[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float own = float(x.value[i]);
      const float peer = __shfl_xor(own, 1, 16);
      const float lo = lane & 1 ? peer : own;
      const float hi = lane & 1 ? own : peer;
      paired[i] = fmaf(hi, hi, lo * lo);
    }
    const float odd = (paired[7] + paired[3]) + (paired[5] + paired[1]);
    const float even = (paired[6] + paired[2]) + (paired[4] + paired[0]);
    float sum = odd + even;
#pragma unroll
    for (int d = 2; d < 16; d *= 2) sum += __shfl_xor(sum, d, 16);
    inv = rsqrtf(fmaf(sum, 1.f / 128.f, p.eps));
  } else {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float a = float(x.value[i]);
      sum += a * a;
    }
#pragma unroll
    for (int d = 8; d; d /= 2) sum += __shfl_xor(sum, d, 16);
    inv = rsqrtf(sum * (1.f / 128.f) + p.eps);
  }
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
      for (int i = 0; i < 8; ++i) {
        const float partner_sin = (sign * float(partner.value[i])) * float(s.value[i]);
        if constexpr (std::is_same_v<Cache, QKCacheParams>) {
          out.value[i] = Bf16(fmaf(float(norm.value[i]), float(c.value[i]), partner_sin));
        } else {
          out.value[i] = Bf16(float(norm.value[i]) * float(c.value[i]) + partner_sin);
        }
      }
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
  if constexpr (std::is_same_v<Cache, QKCacheParams>) {
    const int col = lane * 8;
    if (part == 0 || part == 2) {
      uint8_t* dst = part == 0 ? cache.query : cache.index_query;
      store_e4m3<false>(dst + (row * count + head) * 128 + col, out);
    } else {
      const int64_t loc = cache.locations[row];
      if (part == 1 && loc > 0 && loc < cache.mapping_size) {
        const int64_t slot = cache.main_mapping[loc];
        if (slot > 0 && slot < cache.main_slots) {
          const int64_t offset = (slot * 4 + head) * 128 + col;
          const Bf16* value = cache.value + row * cache.value_row_stride;
          store_e4m3<true>(cache.key_cache + offset, out);
          if constexpr (Transposed) store_e4m3<true>(cache.value_cache + offset, values);
          else store_e4m3<true>(cache.value_cache + offset,
              *reinterpret_cast<const QKPack<Bf16, 8>*>(value + head * 128 + col));
        }
      } else if (part == 3 && loc > 0 && loc < cache.index_slots) {
        store_e4m3<false>(cache.index_cache + loc * 128 + col, out);
      }
    }
  } else {
    *reinterpret_cast<QKPack<Bf16,8>*>(p.out[part]+(row*count+head)*128+lane*8)=out;
  }
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
  switch(variant){case 0:{RUN(64,4);break;}case 1:{RUN(32,4);break;}case 2:{RUN(64,8);break;}case 3:{RUN(32,8);break;}case 4:{VECTOR(8);break;}case 5:{VECTOR(16);break;}case 6:{if(neox){hipLaunchKernelGGL((minimax_qk_norm_rope_rotary64_vec8_kernel<16,true>),dim3((count+15)/16),dim3(256),0,stream,p,cs,positions,false);}else{hipLaunchKernelGGL((minimax_qk_norm_rope_rotary64_vec8_kernel<16,false>),dim3((count+15)/16),dim3(256),0,stream,p,cs,positions,false);}break;}default:TORCH_CHECK(false,"unknown QK kernel variant");}
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
    // All vector variants load aligned QKPack<Cos,8> values in the NeoX path.
    vector_ok=vector_ok && reinterpret_cast<uintptr_t>(cs.data_ptr())%16==0;
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

std::vector<torch::Tensor> minimax_qk_norm_rope_fp8_cache(
    torch::Tensor qkv, std::vector<torch::Tensor> weights,
    torch::Tensor positions, torch::Tensor cs, double epsilon,
    torch::Tensor locations, torch::Tensor main_mapping,
    torch::Tensor key_cache, torch::Tensor value_cache, torch::Tensor index_cache,
    int64_t heads_per_block) {
  TORCH_CHECK(heads_per_block == 4 || heads_per_block == 8 ||
                  heads_per_block == 16 || heads_per_block == 32,
              "Unsupported QK heads per block");
  constexpr int packed_width = 8192 + 512 + 512 + 512 + 128;
  constexpr auto fp8 = at::ScalarType::Float8_e4m3fn;
  const auto aligned = [](const torch::Tensor& t) {
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
  };
  const auto device = qkv.device();
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 &&
              qkv.dim() == 2 && qkv.size(1) == packed_width && aligned(qkv),
              "expected aligned BF16 [rows, 9856] packed M3.1 QKV");
  const bool transposed = qkv.size(0) > 0 && qkv.size(0) % 16 == 0 &&
                          qkv.stride(0) == 1 && qkv.stride(1) == qkv.size(0);
  TORCH_CHECK((qkv.stride(1) == 1 && qkv.stride(0) % 8 == 0) ||
                  (transposed && heads_per_block == 16),
              "expected aligned row-major QKV or dense transposed QKV "
              "with a multiple of 16 rows and heads_per_block=16");
  QKParams p{};
  p.rows = qkv.size(0);
  p.rotary = 64;
  p.eps = float(epsilon);
  TORCH_CHECK(std::isfinite(p.eps) && p.eps > 0, "epsilon must be positive and finite");
  TORCH_CHECK(weights.size() == 4, "expected four Gemma RMSNorm weights");
  constexpr int starts[] = {0, 8192, 9216, 9728};
  for (int i = 0; i < 4; ++i) {
    const auto& weight = weights[i];
    TORCH_CHECK(weight.device() == device && weight.scalar_type() == at::kBFloat16 &&
                weight.dim() == 1 && weight.numel() == 128 && weight.is_contiguous() &&
                aligned(weight), "expected aligned BF16 Gemma weights [128]");
    p.x[i] = p.rows ? qkv.data_ptr<Bf16>() + starts[i] * qkv.stride(1) : nullptr;
    p.weight[i] = weight.data_ptr<Bf16>();
    p.row_stride[i] = qkv.stride(0);
    p.col_stride[i] = qkv.stride(1);
  }
  const auto check_meta = [&](const torch::Tensor& t, at::ScalarType dtype) {
    TORCH_CHECK(t.device() == device && t.scalar_type() == dtype &&
                t.dim() == 1 && t.is_contiguous(), "invalid M3.1 cache metadata");
  };
  check_meta(positions, at::kLong);
  check_meta(locations, at::kLong);
  check_meta(main_mapping, at::kInt);
  TORCH_CHECK(positions.numel() == p.rows && locations.numel() == p.rows,
              "positions and locations must match QKV rows");
  TORCH_CHECK(cs.device() == device && cs.dim() == 2 && cs.size(1) == 64 &&
              (cs.scalar_type() == at::kFloat || cs.scalar_type() == at::kBFloat16) &&
              cs.is_contiguous() && aligned(cs), "invalid rotary64 cos/sin cache");
  const auto check_cache = [&](const torch::Tensor& t, int heads) {
    TORCH_CHECK(t.device() == device && t.scalar_type() == fp8 && t.dim() == 3 &&
                t.size(1) == heads && t.size(2) == 128 && t.is_contiguous() &&
                aligned(t), "invalid contiguous E4M3 M3.1 cache");
  };
  check_cache(key_cache, 4);
  check_cache(value_cache, 4);
  check_cache(index_cache, 1);
  TORCH_CHECK(key_cache.size(0) == value_cache.size(0), "main cache sizes must match");
  c10::cuda::CUDAGuard guard(device);
  auto query = torch::empty({p.rows, 8192}, qkv.options().dtype(fp8));
  auto index_query = torch::empty({p.rows, 512}, qkv.options().dtype(fp8));
  if (!p.rows) return {query, index_query};
  const auto bytes = [](const torch::Tensor& t) {
    return reinterpret_cast<uint8_t*>(t.data_ptr<c10::Float8_e4m3fn>());
  };
  QKCacheParams cache{
      qkv.data_ptr<Bf16>() + 8704 * qkv.stride(1), locations.data_ptr<int64_t>(),
      main_mapping.data_ptr<int32_t>(), bytes(key_cache), bytes(value_cache),
      bytes(index_cache), bytes(query), bytes(index_query), qkv.stride(0), qkv.stride(1),
      main_mapping.numel(), key_cache.size(0), index_cache.size(0)};
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto launch = [&](auto* cos_sin) {
    using Cos = std::remove_pointer_t<decltype(cos_sin)>;
    if (transposed) {
      hipLaunchKernelGGL((minimax_qk_norm_rope_rotary64_vec8_kernel<
          16, true, Cos, int64_t, QKCacheParams, true>),
          dim3(p.rows / 16, 73), dim3(256), 0, stream, p, cos_sin,
          positions.data_ptr<int64_t>(), cache);
      return;
    }
    const dim3 grid((p.rows * 73 + heads_per_block - 1) / heads_per_block);
#define QK_H(H) hipLaunchKernelGGL((minimax_qk_norm_rope_rotary64_vec8_kernel<H, true, Cos, int64_t, QKCacheParams>), grid, dim3(16 * H), 0, stream, p, cos_sin, positions.data_ptr<int64_t>(), cache)
    switch (heads_per_block) {
      case 4: QK_H(4); break;
      case 8: QK_H(8); break;
      case 16: QK_H(16); break;
      case 32: QK_H(32); break;
      default: TORCH_CHECK(false, "Unsupported QK heads per block");
    }
#undef QK_H
  };
  if (cs.scalar_type() == at::kFloat) launch(cs.data_ptr<float>());
  else launch(cs.data_ptr<Bf16>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {query, index_query};
}
} // namespace at::native
