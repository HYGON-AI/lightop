// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include "prefill_sparse_fusion.h"
#include "prefill_sparse_sort.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace {
constexpr uint32_t invalid = INT_MAX;
__device__ __forceinline__ int64_t integer(const void* p, int64_t i, bool wide) {
    return wide ? static_cast<const int64_t*>(p)[i] : static_cast<const int32_t*>(p)[i];
}
__device__ __forceinline__ int64_t floor_div(int64_t a, int64_t b) {
    if (b==1) return a;
    if ((b&(b-1))==0) return a >> __builtin_ctzll(b);
    return a / b - (a < 0 && a % b != 0);
}
__device__ __forceinline__ float bf16_round(float x) {
    uint32_t b = __float_as_uint(x);
    if ((b & 0x7fffffff) > 0x7f800000) return __uint_as_float((b & 0xffff0000) | 0x00400000);
    return __uint_as_float((b + 0x7fff + ((b >> 16) & 1)) & 0xffff0000);
}
template<bool Fast>
__device__ __forceinline__ unsigned fp4(float v, float scale, float inverse) {
    float z;
    if constexpr(Fast) z=v*inverse;
    else z=v/scale;
    z = fminf(fmaxf(z, -6.0f), 6.0f);
    // E2M1 bins have steps 0.5, 1 and 2. One hardware RNE conversion
    // preserves alternating even tie endpoints with fewer comparisons.
    float m = fabsf(z);
    float multiplier = m < 2.0f ? 2.0f : (m < 4.0f ? 1.0f : .5f);
    unsigned offset = m < 2.0f ? 0u : (m < 4.0f ? 2u : 4u);
    unsigned code = __float2int_rn(m * multiplier) + offset;
    return code | ((z < 0 && code != 0) ? 8u : 0u);
}
struct Geometry {
    const int32_t* source = nullptr;
    const void *map = nullptr, *request = nullptr;
    int32_t *blocks = nullptr, *pages = nullptr, *flags = nullptr;
    int width = 0, input_width = 0, page_count = 0, ratio = 1, page_size = 64;
    int64_t source_stride = 0, map_stride = 0;
    bool map64 = false, request64 = false;
};
__device__ __forceinline__ uint32_t candidate(Geometry a, int row, int col) {
    int32_t x = col < a.input_width ? a.source[int64_t(row)*a.source_stride+col] : -1;
    return x < 0 ? invalid : uint32_t(x);
}
template<int Threads, bool Dense=false>
__device__ __forceinline__ void prepare_geometry(Geometry a, int row) {
    if constexpr(Dense){a.width=2048;a.input_width=2048;a.source_stride=2048;a.page_size=64;}
    // Device certificates are refreshed each call, including graph replay.
    __shared__ uint32_t state;
    if (threadIdx.x == 0) state = 0;
    __syncthreads();
    constexpr int Items=(Dense ? 2048 : 4096)/Threads;
    uint32_t keys[Items],bits = 0;
    #pragma unroll
    for(int i=0;i<Items;++i)keys[i]=candidate(a,row,threadIdx.x+i*Threads);
    #pragma unroll
    for (int i=0;i<Items;++i) {
        int col=threadIdx.x+i*Threads;
        uint32_t x=keys[i], next=__shfl_down(x,1,64);
        if((threadIdx.x&63)==63)next=col+1<a.width ? candidate(a,row,col+1) : invalid;
        if (col+1<a.width && x>next) bits|=1;
        if (col+1<a.input_width && x<next) bits|=2;
        if (col+1<a.width && x==next && x!=invalid) bits|=4;
    }
    uint64_t u=__ballot((bits&1)!=0), d=__ballot((bits&2)!=0), dup=__ballot((bits&4)!=0);
    if ((threadIdx.x&63)==0) atomicOr(&state, (u ? 1u:0u)|(d ? 2u:0u)|(dup ? 4u:0u));
    __syncthreads();
    bool ordered=(state&1)==0, reverse=!ordered && (state&2)==0;
    if (ordered || reverse) {
        #pragma unroll
        for (int i=0;i<Items;++i) {
            int col=threadIdx.x+i*Threads;
            uint32_t x=reverse && col<a.input_width ? candidate(a,row,a.input_width-1-col) : keys[i];
            if(col<a.width)a.blocks[int64_t(row)*a.width+col]=x;
        }
    }
    if (threadIdx.x==0) a.flags[row]=(ordered || reverse) ? ((state&4) ? 0 : 1) : 2;
    if (row==0) {
        int64_t request=integer(a.request,0,a.request64);
        for (int p=threadIdx.x; p<a.page_count; p+=blockDim.x) {
            int64_t slot=integer(a.map,request*a.map_stride+int64_t(p)*a.page_size*a.ratio,a.map64);
            a.pages[p]=floor_div(slot,int64_t(a.ratio)*a.page_size);
        }
    }
}

template<int Lanes, bool Combined, bool Fast, bool Dense=false, bool Direct=false>
__global__ __launch_bounds__(256) void sparse_fp8_fp4_paged_mqa_logits__prepare_query(
    const at::BFloat16* q, const at::BFloat16* raw, const float* frequency,
    const void* positions, bool position64, bool aligned_q, int rows, int heads, int rope,
    float weight_scale, int ratio, int width, bool fast,
    int64_t qs0, int64_t qs1, int64_t ws0, int64_t ws1, int64_t pos_stride,
    int64_t fs0, int64_t fs1, int8_t* payload, int32_t* scales, float* weights,
    int32_t* lengths, int32_t* valid, int32_t* pairs, int32_t* zeros, Geometry geometry) {
    if constexpr(Dense) {heads=32;rope=64;width=2048;qs0=4096;qs1=128;ws0=32;ws1=1;pos_stride=1;fs0=64;fs1=2;aligned_q=true;}
    if constexpr(Direct) {ratio=1;position64=false;geometry.ratio=1;geometry.map64=false;geometry.request64=false;}
    constexpr int GroupLanes=Lanes/4, Pairs=64/Lanes, Heads=Lanes<=8 ? 32 : 16;
    int lane=threadIdx.x%Lanes, qh=blockIdx.x*Heads+threadIdx.x/Lanes;
    if (qh<rows*heads) {
        int row,head;
        if constexpr(Combined) {row=blockIdx.x/(32/Heads);head=(blockIdx.x%(32/Heads))*Heads+threadIdx.x/Lanes;}
        else {row=qh/heads;head=qh%heads;}
        int64_t position=integer(positions,int64_t(row)*pos_stride,position64);
        float re[Pairs],im[Pairs],amax=6.0f*0x1p-126f;
        const auto* qbase=q+int64_t(row)*qs0+head*qs1;
        uint4 query_words[Pairs/4];
        float4 frequency_words[Pairs/2];
        if constexpr(Dense) {
            #pragma unroll
            for(int i=0;i<Pairs/4;++i)
                query_words[i]=reinterpret_cast<const uint4*>(qbase)[lane*(Pairs/4)+i];
            if(lane*Pairs>=32) {
                const float* freq=frequency+position*fs0+(lane*Pairs-32)*2;
                #pragma unroll
                for(int i=0;i<Pairs/2;++i)
                    frequency_words[i]=reinterpret_cast<const float4*>(freq)[i];
            }
        }
        #pragma unroll
        for(int i=0;i<Pairs;++i) {
            int pair=lane*Pairs+i;
            float r,s;
            if(aligned_q) {
                uint32_t word;
                if constexpr(Dense) word=reinterpret_cast<const uint32_t*>(query_words)[i];
                else word=reinterpret_cast<const uint32_t*>(qbase)[pair];
                r=__uint_as_float(word<<16);s=__uint_as_float(word&0xffff0000);
            } else {r=float(qbase[2*pair]);s=float(qbase[2*pair+1]);}
            if (pair>=64-rope/2) {
                int64_t f=position*fs0+int64_t(pair-(64-rope/2))*fs1;
                float2 freq;
                if constexpr(Dense) {
                    float4 word=frequency_words[i/2];
                    freq=(i&1) ? make_float2(word.z,word.w) : make_float2(word.x,word.y);
                } else freq=reinterpret_cast<const float2*>(frequency+f)[0];
                float c=freq.x,t=freq.y;
                re[i]=bf16_round(r*c-s*t); im[i]=bf16_round(r*t+s*c);
            } else {re[i]=r;im[i]=s;}
            amax=fmaxf(amax,fmaxf(fabsf(re[i]),fabsf(im[i])));
        }
        #pragma unroll
        for(int delta=GroupLanes/2;delta;delta/=2)amax=fmaxf(amax,__shfl_xor(amax,delta,GroupLanes));
        uint32_t bits=__float_as_uint(amax*(1.0f/6.0f));
        int e=((bits>>23)&255)+((bits&0x7fffff)!=0);
        float scale=__uint_as_float(uint32_t(e)<<23);
        // Exact reciprocal for every possible power-of-two encoding. Preserve
        // IEEE subnormal/zero/infinity behavior without a general divide.
        uint32_t inverse_bits=(e>=1 && e<=253) ? uint32_t(254-e)<<23 :
            (e==254 ? 0x00400000u : (e==255 ? 0u : 0x7f800000u));
        float inverse=__uint_as_float(inverse_bits);
        uint32_t packed_words[Pairs/4] = {};
        #pragma unroll
        for(int i=0;i<Pairs;++i) {
            uint32_t byte=fp4<Fast>(re[i],scale,inverse)|(fp4<Fast>(im[i],scale,inverse)<<4);
            packed_words[i/4]|=byte<<((i%4)*8);
        }
        auto* dst=payload+int64_t(qh)*64+lane*Pairs;
        if constexpr(Pairs==4)*reinterpret_cast<uint32_t*>(dst)=packed_words[0];
        else if constexpr(Pairs==8)*reinterpret_cast<uint2*>(dst)=make_uint2(packed_words[0],packed_words[1]);
        else *reinterpret_cast<uint4*>(dst)=make_uint4(packed_words[0],packed_words[1],packed_words[2],packed_words[3]);
        uint32_t packed=uint32_t(__shfl(e,0,Lanes))+(uint32_t(__shfl(e,GroupLanes,Lanes))<<8)
            +(uint32_t(__shfl(e,2*GroupLanes,Lanes))<<16)+(uint32_t(__shfl(e,3*GroupLanes,Lanes))<<24);
        if(lane==0) {
            scales[qh]=packed;weights[qh]=bf16_round(float(raw[int64_t(row)*ws0+head*ws1])*weight_scale);
            if(head==0) {
                int length=floor_div(position+1,ratio),kept=min((length+7)/8,width);
                lengths[row]=length;valid[row]=length>0 ? 8*(kept-1)+(length-1)%8+1 : 0;
                pairs[row]=row/2;zeros[row]=0;
            }
        }
    }
    if constexpr(Combined) {
        if constexpr(Lanes<=8) prepare_geometry<Lanes*32,Dense>(geometry,blockIdx.x);
        else if((blockIdx.x&1)==0) prepare_geometry<256,Dense>(geometry,blockIdx.x/2);
    }
}
__global__ __launch_bounds__(256) void sparse_fp8_fp4_paged_mqa_logits__prepare_candidates(Geometry a) {
    prepare_geometry<256>(a,blockIdx.x);
}

template<int Distance>
__device__ __forceinline__ uint32_t sort_exchange(uint32_t value) {
#if defined(__HIP_DEVICE_COMPILE__)
    // Fixed quad/row exchanges avoid the address generation and LDS waits
    // of a general lane-addressed bpermute in the common sorting stages.
    if constexpr(Distance==1) return __builtin_amdgcn_mov_dpp(value,0xb1,0xf,0xf,false);
    else if constexpr(Distance==2) return __builtin_amdgcn_mov_dpp(value,0x4e,0xf,0xf,false);
    else if constexpr(Distance==4) return __builtin_amdgcn_mov_dpp(
        __builtin_amdgcn_mov_dpp(value,0x141,0xf,0xf,false),0x1b,0xf,0xf,false);
    else if constexpr(Distance==8) return __builtin_amdgcn_mov_dpp(
        __builtin_amdgcn_mov_dpp(value,0x140,0xf,0xf,false),0x141,0xf,0xf,false);
    else if constexpr(Distance==16) return __builtin_amdgcn_ds_swizzle(value,0x401f);
    else return __shfl_xor(value,Distance,64);
#else
    return value;
#endif
}
template<int Size,int Distance,int Items>
__device__ __forceinline__ void bitonic_stage(uint32_t (&keys)[Items],int lane) {
    uint32_t next[Items];
    #pragma unroll
    for(int i=0;i<Items;++i) {
        uint32_t peer;
        if constexpr(Distance<64) peer=sort_exchange<Distance>(keys[i]);
        else peer=keys[i^(Distance/64)];
        int index=lane+i*64;
        bool lower=((index&Size)==0)==((index&Distance)==0);
        next[i]=lower ? min(keys[i],peer) : max(keys[i],peer);
    }
    #pragma unroll
    for(int i=0;i<Items;++i) keys[i]=next[i];
    if constexpr(Distance>1) bitonic_stage<Size,Distance/2>(keys,lane);
}
template<int Size,int N,int Items>
__device__ __forceinline__ void bitonic(uint32_t (&keys)[Items],int lane) {
    bitonic_stage<Size,Size/2>(keys,lane);
    if constexpr(Size<N) bitonic<Size*2,N>(keys,lane);
}
// Certified compact IDs fit in 16 bits (at most 4096 eight-token blocks).
// Pack two independent halves of the sorting network in each VGPR. HIP emits
// v_pk_min_u16/v_pk_max_u16, sharing each DPP exchange between the two halves.
using IndexPair=uint16_t __attribute__((ext_vector_type(2)));
union PackedIndices {uint32_t bits;IndexPair pair;};
template<int Size,int Distance>
__device__ __forceinline__ void packed_bitonic_stage(uint32_t (&keys)[4],int lane) {
    uint32_t next[4];
    #pragma unroll
    for(int i=0;i<4;++i) {
        PackedIndices a,b,lo,hi;a.bits=keys[i];
        if constexpr(Distance<64)b.bits=sort_exchange<Distance>(a.bits);
        else if constexpr(Distance<256)b.bits=keys[i^(Distance/64)];
        else b.bits=(a.bits<<16)|(a.bits>>16);
        lo.pair=__builtin_elementwise_min(a.pair,b.pair);
        hi.pair=__builtin_elementwise_max(a.pair,b.pair);
        int low_index=lane+i*64,high_index=low_index+256;
        bool low=((low_index&Size)==0)==((low_index&Distance)==0);
        bool high=((high_index&Size)==0)==((high_index&Distance)==0);
        uint32_t mask=(low ? 65535u : 0u)|(high ? 0xffff0000u : 0u);
        next[i]=(lo.bits&mask)|(hi.bits&~mask);
    }
    #pragma unroll
    for(int i=0;i<4;++i)keys[i]=next[i];
    if constexpr(Distance>1)packed_bitonic_stage<Size,Distance/2>(keys,lane);
    return;
}
template<int Size=2>
__device__ __forceinline__ void packed_bitonic(uint32_t (&keys)[4],int lane) {
    packed_bitonic_stage<Size,Size/2>(keys,lane);
    if constexpr(Size<512)packed_bitonic<Size*2>(keys,lane);
    return;
}
struct Finalize {
    const int32_t *compact,*blocks,*lengths,*valid,*flags;
    int32_t *logical,*physical;
    const void *map,*request;
    int k,width,ratio,rows;int64_t cs,bs,ls,ps,ms;
    bool map64,request64;
};
template<int N, bool Dense=false, bool Direct=false>
__global__ __launch_bounds__(256) void sparse_fp8_fp4_paged_mqa_logits__finalize_indices(Finalize a) {
    if constexpr(Dense){a.k=512;a.width=2048;a.cs=512;a.bs=2048;a.ls=512;a.ps=512;}
    if constexpr(Direct){a.ratio=1;a.map64=false;a.request64=false;}
    constexpr int Items=N/64;
    int row=blockIdx.x*4+threadIdx.x/64,lane=threadIdx.x%64;
#if defined(__HIP_DEVICE_COMPILE__)
    // Each wave owns one row. Make this uniform fact explicit so row metadata
    // and output-base arithmetic can use the scalar register/address path.
    row=__builtin_amdgcn_readfirstlane(row);
#endif
    if(row>=a.rows)return;
    int length=a.lengths[row],valid=a.valid[row];
    bool contract=a.flags && a.flags[row]==1 && valid<=a.width*8;
    uint32_t keys[Items];
    #pragma unroll
    for(int i=0;i<Items;++i) {
        int col=lane+i*64,c=col<a.k ? a.compact[int64_t(row)*a.cs+col] : -1;
        int64_t logical=INT_MAX;
        if(c>=0 && c<valid) {
            if(contract) logical=c;
            else {
                int slot=min(c/8,a.width-1);
                logical=int64_t(a.blocks[int64_t(row)*a.bs+slot])*8+c%8;
                if(logical<0 || logical>=length) logical=INT_MAX;
            }
        }
        keys[i]=logical;
    }
    bool unordered=false;
    #pragma unroll
    for(int i=0;i<Items;++i) {
        uint32_t previous=__shfl_up(keys[i],1,64);
        if(i>0) {
            uint32_t last=__shfl(keys[i-1],63,64);
            if(lane==0)previous=last;
        }
        if(lane+i*64>0 && keys[i]<previous)unordered=true;
    }
    if(__ballot(unordered)) {
        if constexpr(N==512) {
            if(contract) {
                uint32_t packed[4];
                #pragma unroll
                for(int i=0;i<4;++i)packed[i]=min(keys[i],65535u)|(min(keys[i+4],65535u)<<16);
                packed_bitonic(packed,lane);
                #pragma unroll
                for(int i=0;i<4;++i) {
                    uint32_t low=packed[i]&65535u,high=packed[i]>>16;
                    keys[i]=low==65535u ? invalid : low;
                    keys[i+4]=high==65535u ? invalid : high;
                }
            } else bitonic<2,N>(keys,lane);
        } else bitonic<2,N>(keys,lane);
    }
    int64_t request=a.physical ? integer(a.request,0,a.request64) : 0;
    int32_t physical_keys[Items];
    // Issue all page-map gathers before the first output store. Keeping the
    // loads separate also avoids serial load/store dependencies when alias
    // analysis cannot disambiguate the tensor pointers in Finalize.
    #pragma unroll
    for(int i=0;i<Items;++i) {
        int col=lane+i*64;uint32_t key=keys[i];int32_t logical=-1;
        if(key!=invalid) {
            int64_t pos=contract ? int64_t(a.blocks[int64_t(row)*a.bs+key/8])*8+key%8 : key;
            if(pos>=0 && pos<length)logical=pos;
        }
        keys[i]=uint32_t(logical);
        if(a.physical && col<a.k)physical_keys[i]=logical<0 ? -1 :
            floor_div(integer(a.map,request*a.ms+int64_t(logical)*a.ratio,a.map64),a.ratio);
    }
    #pragma unroll
    for(int i=0;i<Items;++i) {
        int col=lane+i*64;
        if(col<a.k) {
            a.logical[int64_t(row)*a.ls+col]=int32_t(keys[i]);
            if(a.physical)a.physical[int64_t(row)*a.ps+col]=physical_keys[i];
        }
    }
}
void vector_check(const at::Tensor& t,int64_t rows,at::Device d,const char* name) {
    TORCH_CHECK(t.device()==d && t.scalar_type()==at::kInt && t.dim()==1 && t.numel()==rows && t.is_contiguous(),name," must be contiguous GPU int32 vector");
}
void matrix_check(const at::Tensor& t,at::Device d,const char* name) {
    TORCH_CHECK(t.is_cuda() && t.device()==d && t.scalar_type()==at::kInt && t.dim()==2 && t.stride(1)==1,name," must be GPU int32 matrix");
}
Geometry make_geometry(const at::Tensor& source,int64_t width,const at::Tensor& map,
    const at::Tensor& request,int64_t context,int64_t ratio,int64_t page,
    at::Tensor blocks,at::Tensor pages,at::Tensor flags) {
    auto d=source.device();matrix_check(source,d,"source");
    TORCH_CHECK(width>0 && width<=4096 && width%4==0 && context>=0 && ratio>0 && page>0,"invalid candidate geometry");
    TORCH_CHECK(map.device()==d && map.dim()==2 && map.stride(1)==1 && (map.scalar_type()==at::kInt || map.scalar_type()==at::kLong),"invalid request map");
    TORCH_CHECK(request.device()==d && request.numel()==1 && request.is_contiguous() && (request.scalar_type()==at::kInt || request.scalar_type()==at::kLong),"invalid request ID");
    vector_check(flags,source.size(0),d,"flags");
    Geometry a;a.source=source.data_ptr<int32_t>();a.map=map.data_ptr();a.request=request.data_ptr();
    a.blocks=blocks.data_ptr<int32_t>();a.pages=pages.data_ptr<int32_t>();a.flags=flags.data_ptr<int32_t>();
    a.width=width;a.input_width=std::min(width,source.size(1));a.page_count=(context+page-1)/page;
    a.ratio=ratio;a.page_size=page;a.source_stride=source.stride(0);a.map_stride=map.stride(0);
    a.map64=map.scalar_type()==at::kLong;a.request64=request.scalar_type()==at::kLong;return a;
}
void finish_sort(const at::Tensor& source,at::Tensor blocks,at::Tensor flags,int64_t context) {
    prefill_sparse_sort_candidates(source,blocks,flags,std::min<int64_t>(65536,std::max<int64_t>(1,(context+7)/8)),1,512,flags);
}
void launch_check() {TORCH_CHECK(hipGetLastError()==hipSuccess,"native sparse Prefill fusion launch failed");}
}

std::vector<at::Tensor> prefill_sparse_native_prepare_query(
    const at::Tensor& q,const at::Tensor& raw,const at::Tensor& frequency,const at::Tensor& positions,
    int64_t rope,double scale,int64_t ratio,int64_t width,bool fast,c10::optional<at::Tensor> source,
    c10::optional<at::Tensor> map,c10::optional<at::Tensor> request,int64_t context,int64_t page,
    c10::optional<at::Tensor> flags,int64_t lanes) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.dim()==3 && q.size(2)==128 && q.stride(2)==1,"q must be GPU BF16 [R,H,128]");
    auto d=q.device();int64_t rows=q.size(0),heads=q.size(1);
    TORCH_CHECK(heads>0 && heads<=32 && heads%4==0 && raw.device()==d && raw.scalar_type()==at::kBFloat16 && raw.dim()==2 && raw.size(0)==rows && raw.size(1)==heads,"invalid BF16 head weights");
    TORCH_CHECK(frequency.device()==d && frequency.scalar_type()==at::kComplexFloat && frequency.dim()==2,"frequencies must be GPU complex64 matrix");
    TORCH_CHECK(positions.device()==d && positions.dim()==1 && positions.numel()==rows && (positions.scalar_type()==at::kInt || positions.scalar_type()==at::kLong),"invalid position vector");
    TORCH_CHECK(rope>=0 && rope<=128 && rope%2==0 && frequency.size(1)>=rope/2 && ratio>0 && width>0 && width<=4096 && width%4==0,"invalid query geometry");
    TORCH_CHECK(lanes==4 || lanes==8 || lanes==16,"lanes must be 4,8 or16");
    at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(d);
    auto i=q.options().dtype(at::kInt);std::vector<at::Tensor> out;
    out.push_back(at::empty({rows,1,heads,64},q.options().dtype(at::kChar)));
    out.push_back(at::empty({rows,1,heads},i));out.push_back(at::empty({rows,heads},q.options().dtype(at::kFloat)));
    for(int n=0;n<4;++n)out.push_back(at::empty({rows},i));
    Geometry g;at::Tensor f;
    if(source) {
        TORCH_CHECK(context>=0 && page>0,"invalid request context/page size");
        TORCH_CHECK(heads==32 && map && request && source->size(0)==rows,"combined preparation requires H32 and request geometry");
        out.push_back(at::empty({rows,width},i));out.push_back(at::empty({(context+page-1)/page},i));
        f=flags ? *flags : at::empty({rows},i);g=make_geometry(*source,width,*map,*request,context,ratio,page,out[7],out[8],f);
    }
    auto stream=at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    bool aligned_q=(reinterpret_cast<uintptr_t>(q.data_ptr())%4==0 && q.stride(0)%2==0 && q.stride(1)%2==0);
    bool dense=source && heads==32 && rope==64 && width==2048 && q.is_contiguous() && raw.is_contiguous()
        && positions.is_contiguous() && frequency.stride(0)==32 && frequency.stride(1)==1
        && source->size(1)==2048 && source->stride(0)==2048 && page==64 && aligned_q
        && reinterpret_cast<uintptr_t>(q.data_ptr())%16==0 && reinterpret_cast<uintptr_t>(frequency.data_ptr())%16==0;
    #define ARGS q.data_ptr<at::BFloat16>(),raw.data_ptr<at::BFloat16>(),reinterpret_cast<const float*>(frequency.data_ptr()),positions.data_ptr(),positions.scalar_type()==at::kLong,aligned_q,rows,heads,rope,float(scale),ratio,width,fast,q.stride(0),q.stride(1),raw.stride(0),raw.stride(1),positions.stride(0),frequency.stride(0)*2,frequency.stride(1)*2,out[0].data_ptr<int8_t>(),out[1].data_ptr<int32_t>(),out[2].data_ptr<float>(),out[3].data_ptr<int32_t>(),out[4].data_ptr<int32_t>(),out[5].data_ptr<int32_t>(),out[6].data_ptr<int32_t>(),g
    #define LAUNCH(L,C,G,T) if(fast) {hipLaunchKernelGGL((sparse_fp8_fp4_paged_mqa_logits__prepare_query<L,C,true>),dim3(G),dim3(T),0,stream,ARGS);} else {hipLaunchKernelGGL((sparse_fp8_fp4_paged_mqa_logits__prepare_query<L,C,false>),dim3(G),dim3(T),0,stream,ARGS);}
    #define DENSE_QUERY(L,G,T) if(direct) {hipLaunchKernelGGL((sparse_fp8_fp4_paged_mqa_logits__prepare_query<L,true,true,true,true>),dim3(G),dim3(T),0,stream,ARGS);} else {hipLaunchKernelGGL((sparse_fp8_fp4_paged_mqa_logits__prepare_query<L,true,true,true>),dim3(G),dim3(T),0,stream,ARGS);}
    if(rows) {
        if(dense && fast) {
            bool direct=ratio==1 && positions.scalar_type()==at::kInt && map->scalar_type()==at::kInt && request->scalar_type()==at::kInt;
            if(lanes==4){DENSE_QUERY(4,rows,128)}
            else if(lanes==8){DENSE_QUERY(8,rows,256)}
            else {DENSE_QUERY(16,rows*2,256)}
        }
        else if(source && lanes==4) {LAUNCH(4,true,rows,128)}
        else if(source && lanes==8) {LAUNCH(8,true,rows,256)}
        else if(source) {LAUNCH(16,true,rows*2,256)}
        else if(lanes==4) {LAUNCH(4,false,(rows*heads+31)/32,128)}
        else if(lanes==8) {LAUNCH(8,false,(rows*heads+31)/32,256)}
        else {LAUNCH(16,false,(rows*heads+15)/16,256)}
        launch_check();if(source)finish_sort(*source,out[7],f,context);
    }
    #undef ARGS
    #undef LAUNCH
    #undef DENSE_QUERY
    return out;
}
std::vector<at::Tensor> prefill_sparse_native_prepare_candidates(
    const at::Tensor& source,int64_t width,const at::Tensor& map,const at::Tensor& request,
    int64_t context,int64_t ratio,int64_t page,c10::optional<at::Tensor> flags) {
    TORCH_CHECK(source.is_cuda() && source.dim()==2 && context>=0 && page>0,"invalid candidate geometry");
    at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(source.device());auto opt=source.options().dtype(at::kInt);
    auto blocks=at::empty({source.size(0),width},opt),pages=at::empty({(context+page-1)/page},opt);
    auto f=flags ? *flags : at::empty({source.size(0)},opt);
    auto g=make_geometry(source,width,map,request,context,ratio,page,blocks,pages,f);
    if(source.size(0)) {
        hipLaunchKernelGGL(sparse_fp8_fp4_paged_mqa_logits__prepare_candidates,dim3(source.size(0)),dim3(256),0,at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream(),g);
        launch_check();finish_sort(source,blocks,f,context);
    }
    return {blocks,pages};
}
void prefill_sparse_native_finalize(const at::Tensor& compact,const at::Tensor& blocks,const at::Tensor& lengths,const at::Tensor& valid,
    at::Tensor logical,c10::optional<at::Tensor> physical,c10::optional<at::Tensor> map,c10::optional<at::Tensor> request,int64_t ratio,c10::optional<at::Tensor> flags) {
    auto d=compact.device();matrix_check(compact,d,"compact");matrix_check(blocks,d,"blocks");matrix_check(logical,d,"logical");
    int64_t rows=compact.size(0),k=compact.size(1);
    TORCH_CHECK(k>0 && k<=2048 && blocks.size(0)==rows && blocks.size(1)>0 && logical.sizes()==compact.sizes() && ratio>0,"invalid finalize geometry");
    vector_check(lengths,rows,d,"lengths");vector_check(valid,rows,d,"valid");if(flags)vector_check(*flags,rows,d,"flags");
    at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(d);
    Finalize a{};a.compact=compact.data_ptr<int32_t>();a.blocks=blocks.data_ptr<int32_t>();a.lengths=lengths.data_ptr<int32_t>();a.valid=valid.data_ptr<int32_t>();a.logical=logical.data_ptr<int32_t>();
    a.flags=flags ? flags->data_ptr<int32_t>() : nullptr;a.rows=rows;a.k=k;a.width=blocks.size(1);a.ratio=ratio;a.cs=compact.stride(0);a.bs=blocks.stride(0);a.ls=logical.stride(0);
    if(physical) {
        matrix_check(*physical,d,"physical");TORCH_CHECK(physical->sizes()==compact.sizes() && map && request,"invalid physical output");
        TORCH_CHECK(map->device()==d && map->dim()==2 && map->stride(1)==1 && (map->scalar_type()==at::kInt || map->scalar_type()==at::kLong),"invalid request map");
        TORCH_CHECK(request->device()==d && request->numel()==1 && request->is_contiguous() && (request->scalar_type()==at::kInt || request->scalar_type()==at::kLong),"invalid request ID");
        a.physical=physical->data_ptr<int32_t>();a.ps=physical->stride(0);a.map=map->data_ptr();a.request=request->data_ptr();a.ms=map->stride(0);a.map64=map->scalar_type()==at::kLong;a.request64=request->scalar_type()==at::kLong;
    }
    if(rows) {
        auto stream=at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
        bool dense=k==512 && a.width==2048 && a.cs==512 && a.bs==2048 && a.ls==512 && (!a.physical || a.ps==512);
        bool direct=ratio==1 && (!physical || (map->scalar_type()==at::kInt && request->scalar_type()==at::kInt));
        if(dense && direct)hipLaunchKernelGGL((sparse_fp8_fp4_paged_mqa_logits__finalize_indices<512,true,true>),dim3((rows+3)/4),dim3(256),0,stream,a);
        else if(dense)hipLaunchKernelGGL((sparse_fp8_fp4_paged_mqa_logits__finalize_indices<512,true>),dim3((rows+3)/4),dim3(256),0,stream,a);
        else if(k<=512)hipLaunchKernelGGL((sparse_fp8_fp4_paged_mqa_logits__finalize_indices<512>),dim3((rows+3)/4),dim3(256),0,stream,a);
        else if(k<=1024)hipLaunchKernelGGL((sparse_fp8_fp4_paged_mqa_logits__finalize_indices<1024>),dim3((rows+3)/4),dim3(256),0,stream,a);
        else hipLaunchKernelGGL((sparse_fp8_fp4_paged_mqa_logits__finalize_indices<2048>),dim3((rows+3)/4),dim3(256),0,stream,a);
        launch_check();
    }
}
