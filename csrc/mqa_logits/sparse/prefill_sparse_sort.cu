// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include "prefill_sparse_sort.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <rocprim/block/block_radix_sort.hpp>
#include <algorithm>
#include <cstdint>

namespace {
constexpr uint32_t invalid=0x7fffffffu;
struct Args {
 const int32_t *input, *blocks, *lengths, *valid, *contract, *pending;
 int32_t *output, *physical, *flags;
 const void *map, *request;
 int rows,width,input_width,block_width,ratio,ratio_shift,map64,request64;
 int64_t is,bs,os,ps,ms;
};
__device__ __forceinline__ int64_t integer(const void* p,int64_t i,int is64) {
 return is64 ? reinterpret_cast<const int64_t*>(p)[i] : reinterpret_cast<const int32_t*>(p)[i];
}
template<bool Final>
__device__ __forceinline__ uint32_t read_key(const Args& a,int row,int col,bool contract,uint32_t* mapped=nullptr) {
 if(col>=a.input_width) return invalid;
 int32_t c=a.input[int64_t(row)*a.is+col];
 if constexpr(!Final) return c<0 ? invalid : uint32_t(c);
 else {
  if(c<0 || c>=a.valid[row]) return invalid;
  // The preparation certificate proves ascending, unique candidate IDs.
  // Compact-key order is therefore logical-key order. Map only after sorting
  // so candidate reads are ordered and each position is gathered once.
  if(contract) {if(mapped) *mapped=uint32_t(c);return uint32_t(c);}
  int slot=min(c/8,a.block_width-1);
  int64_t b=a.blocks[int64_t(row)*a.bs+slot];
  int64_t logical=b*8+c%8;
  if(logical<0 || logical>=a.lengths[row]) return invalid;
  if(mapped) *mapped=uint32_t(logical);
  return uint32_t(logical);
 }
}
template<bool Final>
__device__ __forceinline__ void write_key(const Args& a,int row,int col,uint32_t key,bool contract) {
 if(col>=a.width) return;
 int32_t logical=-1;
 if(key!=invalid) {
  if constexpr(Final) {
   if(contract) {
    int64_t mapped=int64_t(a.blocks[int64_t(row)*a.bs+key/8])*8+key%8;
    logical=(mapped>=0 && mapped<a.lengths[row]) ? int32_t(mapped) : -1;
   } else logical=key;
  } else logical=key;
 }
 if constexpr(!Final) {
  a.output[int64_t(row)*a.os+col]=key;
 } else {
  a.output[int64_t(row)*a.os+col]=logical;
  if(a.physical) {
   int32_t physical=-1;
   if(logical>=0) {
    int64_t request=integer(a.request,0,a.request64);
    int64_t slot=integer(a.map,request*a.ms+int64_t(logical)*a.ratio,a.map64);
    if(a.ratio==1) physical=slot;
    else if(a.ratio_shift>=0) physical=slot>>a.ratio_shift;
    else physical=slot/a.ratio-int64_t(slot<0 && slot%a.ratio!=0);
   }
   a.physical[int64_t(row)*a.ps+col]=physical;
  }
 }
}
template<int N,int T>
__device__ __forceinline__ void shared_bitonic(uint32_t* values) {
 for(int size=2;size<=N;size*=2) {
  for(int distance=size/2;distance;distance/=2) {
   for(int col=threadIdx.x;col<N;col+=T) {
    int peer=col^distance;
    if(peer>col) {
     uint32_t x=values[col],y=values[peer];
     bool ascend=(col&size)==0;
     if((x>y)==ascend) {values[col]=y;values[peer]=x;}
    }
   }
   __syncthreads();
  }
 }
}
template<int N,int T,int Domain,bool Final>
__global__ __launch_bounds__(T) void prefill_index_bitmap(Args a) {
 const int row=blockIdx.x,tid=threadIdx.x;
 if(a.pending && a.pending[row]!=2) return;
 constexpr int Items=N/T,Words=(Domain+31)/32,Segments=(Words+63)/64;
 __shared__ uint32_t bitmap[Words],prefix[Words],totals[Segments],ordered[N],boundary[N/64];
 __shared__ uint32_t status,total;
 bool contract=Final && a.contract && a.contract[row]==1 && a.valid[row]<=a.block_width*8;
 uint32_t keys[Items],mapped[Items];
 bool mapped_output=false;
 if(tid==0) status=0;
 for(int word=tid;word<Words;word+=T) bitmap[word]=0;
 #pragma unroll
 for(int i=0;i<Items;++i) {
  int col=tid+i*T;
  mapped[i]=invalid;
  keys[i]=read_key<Final>(a,row,col,contract,mapped+i);
  if((tid&63)==63) boundary[col/64]=keys[i];
 }
 __syncthreads();
 uint32_t local=0;
 #pragma unroll
 for(int i=0;i<Items;++i) {
  int col=tid+i*T;uint32_t key=keys[i];
  uint32_t previous=__shfl_up(key,1,64);
  if((tid&63)==0 && col) previous=boundary[col/64-1];
  if(col && key<previous) local|=4;
  if(key!=invalid) {
   if(key>=Domain) local|=2;
   else if(atomicOr(bitmap+(key>>5),1u<<(key&31))&(1u<<(key&31))) local|=1;
  }
 }
 uint32_t warp_state=(__ballot((local&1)!=0) ? 1u:0u)|(__ballot((local&2)!=0) ? 2u:0u)|(__ballot((local&4)!=0) ? 4u:0u);
 if((tid&63)==0 && warp_state) atomicOr(&status,warp_state);
 __syncthreads();
 uint32_t state=status;
 if((state&4)==0) {
  // Write original keys directly without an extra LDS round trip.
  #pragma unroll
  for(int i=0;i<Items;++i) ordered[tid+i*T]=keys[i];
  __syncthreads();
 } else if(state&3) {
  // Duplicate or wide-domain keys retain exact multiplicity and full int32 order.
  #pragma unroll
  for(int i=0;i<Items;++i) ordered[tid+i*T]=keys[i];
  __syncthreads();
  shared_bitonic<N,T>(ordered);
 } else {
  for(int word=tid;word<Words;word+=T) {
   uint32_t count=__popc(bitmap[word]),scan=count;
   #pragma unroll
   for(int delta=1;delta<64;delta*=2) {
    uint32_t previous=__shfl_up(scan,delta,64);
    if((tid&63)>=delta) scan+=previous;
   }
   prefix[word]=scan-count;
   if((tid&63)==63) totals[word/64]=scan;
  }
  __syncthreads();
  if(tid<64) {
   uint32_t count=tid<Segments ? totals[tid] : 0,scan=count;
   #pragma unroll
   for(int delta=1;delta<64;delta*=2) {
    uint32_t previous=__shfl_up(scan,delta,64);
    if(tid>=delta) scan+=previous;
   }
   if(tid<Segments) totals[tid]=scan-count;
   if(tid==Segments-1) total=scan;
  }
  __syncthreads();
  for(int word=tid;word<Words;word+=T) prefix[word]+=totals[word/64];
  __syncthreads();
  #pragma unroll
  for(int i=0;i<Items;++i) {
   uint32_t key=keys[i];
   if(key!=invalid) {
    uint32_t word=key>>5,bit=key&31;
    uint32_t rank=prefix[word]+__popc(bitmap[word]&((1u<<bit)-1));
    ordered[rank]=Final ? mapped[i] : key;
   }
  }
  mapped_output=Final && !contract;
  for(int col=tid+total;col<N;col+=T) ordered[col]=invalid;
  __syncthreads();
 }
 #pragma unroll
 for(int i=0;i<Items;++i) write_key<Final>(a,row,tid+i*T,ordered[tid+i*T],contract && !mapped_output);
 if(a.flags) {
  if((state&2)==0) {if(tid==0) a.flags[row]=(state&1) ? 0 : 1;}
  else {
   uint32_t duplicates=0;
   for(int col=tid;col<a.width;col+=T)
    if(col && ordered[col]!=invalid && ordered[col]==ordered[col-1]) duplicates=1;
   if(tid==0) status=0;
   __syncthreads();
   uint64_t mask=__ballot(duplicates!=0);
   if((tid&63)==0 && mask) atomicOr(&status,1u);
   __syncthreads();
   if(tid==0) a.flags[row]=status ? 0 : 1;
  }
 }
}
template<int N,int T,int Domain,bool Final>
__global__ __launch_bounds__(T) void prefill_index_radix(Args a) {
 const int row=blockIdx.x,tid=threadIdx.x;
 if(a.pending && a.pending[row]!=2) return;
 constexpr int Items=N/T;
 using Sort=rocprim::block_radix_sort<uint32_t,T,Items>;
 __shared__ typename Sort::storage_type storage;
 __shared__ uint32_t state;
 uint32_t keys[Items];bool contract=Final && a.contract && a.contract[row]==1 && a.valid[row]<=a.block_width*8;
 if(tid==0) state=0;
 __syncthreads();
 #pragma unroll
 for(int i=0;i<Items;++i) {
  keys[i]=read_key<Final>(a,row,tid*Items+i,contract);
  if(keys[i]!=invalid && keys[i]>=Domain) atomicOr(&state,1u);
 }
 __syncthreads();
 bool bounded=state==0;
 if(bounded) for(int i=0;i<Items;++i) if(keys[i]==invalid) keys[i]=Domain;
 Sort().sort(keys,storage,0,bounded ? __builtin_ctz(Domain)+1 : 31);
 #pragma unroll
 for(int i=0;i<Items;++i) {
  if(bounded && keys[i]==Domain) keys[i]=invalid;
  write_key<Final>(a,row,tid*Items+i,keys[i],contract);
 }
 // Caller can use radix independently; mark candidate contract conservatively.
 if(a.flags && tid==0) a.flags[row]=0;
}
template<int N,int T,int Domain,bool Final>
void launch(Args a,int algorithm,hipStream_t stream) {
 if(algorithm==0) {
  hipLaunchKernelGGL((prefill_index_radix<N,T,Domain,Final>),dim3(a.rows),dim3(T),0,stream,a);
 } else {
  hipLaunchKernelGGL((prefill_index_bitmap<N,T,Domain,Final>),dim3(a.rows),dim3(T),0,stream,a);
 }
}
template<int N,int T,bool Final>
void domain_launch(Args a,int capacity,int algorithm,hipStream_t stream) {
 if(capacity<=4096) launch<N,T,4096,Final>(a,algorithm,stream);
 else if(capacity<=8192) launch<N,T,8192,Final>(a,algorithm,stream);
 else if(capacity<=16384) launch<N,T,16384,Final>(a,algorithm,stream);
 else if(capacity<=32768) launch<N,T,32768,Final>(a,algorithm,stream);
 else launch<N,T,65536,Final>(a,algorithm,stream);
}
template<bool Final>
void dispatch(Args a,int capacity,int algorithm,int threads,hipStream_t stream) {
 #define L(N,T) domain_launch<N,T,Final>(a,capacity,algorithm,stream)
 #define S(T) if(a.width<=512){L(512,T);}else if(a.width<=1024){L(1024,T);}else if(a.width<=2048){L(2048,T);}else{L(4096,T);}
 if(threads==128) {S(128)} else if(threads==256) {S(256)} else {S(512)}
 #undef S
 #undef L
 TORCH_CHECK(hipGetLastError()==hipSuccess,"Prefill index sort launch failed");
}
void check_matrix(const at::Tensor& t,const char* name) {
 TORCH_CHECK(t.is_cuda() && t.scalar_type()==at::kInt && t.dim()==2 && t.stride(1)==1,name," must be GPU int32 matrix with contiguous columns");
}
void check_vector(const at::Tensor& t,int rows,at::Device device,const char* name) {
 TORCH_CHECK(t.device()==device && t.scalar_type()==at::kInt && t.dim()==1 && t.numel()==rows && t.is_contiguous(),name," must be contiguous int32 per-row vector");
}
}

void prefill_sparse_sort_candidates(const at::Tensor& source,at::Tensor output,at::Tensor flags,
 int64_t capacity,int64_t algorithm,int64_t threads,c10::optional<at::Tensor> pending) {
 check_matrix(source,"source");check_matrix(output,"output");
 TORCH_CHECK(output.device()==source.device() && output.size(0)==source.size(0),"candidate devices/rows differ");
 TORCH_CHECK(output.size(1)>0 && output.size(1)<=4096,"candidate width must be in[1,4096]");
 TORCH_CHECK(capacity>0 && capacity<=65536 && (algorithm==0 || algorithm==1) && (threads==128 || threads==256 || threads==512),"invalid sort configuration");
 check_vector(flags,source.size(0),source.device(),"flags");
 if(pending) check_vector(*pending,source.size(0),source.device(),"pending");
 at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(source.device());
 Args a{};a.input=source.data_ptr<int32_t>();a.output=output.data_ptr<int32_t>();a.flags=flags.data_ptr<int32_t>();
 a.rows=source.size(0);a.width=output.size(1);a.input_width=std::min(source.size(1),output.size(1));a.is=source.stride(0);a.os=output.stride(0);a.pending=pending ? pending->data_ptr<int32_t>() : nullptr;
 if(a.rows) dispatch<false>(a,capacity,algorithm,threads,at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream());
}

void prefill_sparse_finalize_indices(const at::Tensor& compact,const at::Tensor& blocks,
 const at::Tensor& lengths,const at::Tensor& valid,at::Tensor logical,c10::optional<at::Tensor> physical,
 c10::optional<at::Tensor> request_map,c10::optional<at::Tensor> request_id,
 int64_t ratio,int64_t capacity,int64_t algorithm,int64_t threads,c10::optional<at::Tensor> candidate_flags) {
 check_matrix(compact,"compact");check_matrix(blocks,"blocks");check_matrix(logical,"logical");
 TORCH_CHECK(compact.size(1)>0 && compact.size(1)<=2048 && blocks.size(1)>0,"invalid finalize geometry");
 TORCH_CHECK(logical.sizes()==compact.sizes() && logical.device()==compact.device() && blocks.size(0)==compact.size(0) && blocks.device()==compact.device(),"finalize shapes/devices differ");
 check_vector(lengths,compact.size(0),compact.device(),"lengths");check_vector(valid,compact.size(0),compact.device(),"valid");
 TORCH_CHECK(ratio>0 && capacity>0 && capacity<=65536 && (algorithm==0 || algorithm==1) && (threads==128 || threads==256 || threads==512),"invalid finalize configuration");
 if(candidate_flags) check_vector(*candidate_flags,compact.size(0),compact.device(),"candidate_flags");
 at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(compact.device());
 Args a{};a.rows=compact.size(0);a.width=compact.size(1);a.input_width=a.width;a.block_width=blocks.size(1);a.ratio=ratio;a.ratio_shift=(ratio&(ratio-1))==0 ? __builtin_ctzll(ratio) : -1;
 a.input=compact.data_ptr<int32_t>();a.blocks=blocks.data_ptr<int32_t>();a.lengths=lengths.data_ptr<int32_t>();a.valid=valid.data_ptr<int32_t>();a.output=logical.data_ptr<int32_t>();a.is=compact.stride(0);a.bs=blocks.stride(0);a.os=logical.stride(0);a.contract=candidate_flags ? candidate_flags->data_ptr<int32_t>() : nullptr;
 if(physical) {
  check_matrix(*physical,"physical");TORCH_CHECK(physical->device()==compact.device() && physical->sizes()==compact.sizes(),"physical output differs");
  TORCH_CHECK(request_map && request_id,"mapping and request ID required");
  TORCH_CHECK(request_map->device()==compact.device() && request_map->dim()==2 && request_map->stride(1)==1 && (request_map->scalar_type()==at::kInt || request_map->scalar_type()==at::kLong),"invalid request map");
  TORCH_CHECK(request_id->device()==compact.device() && request_id->numel()==1 && request_id->is_contiguous() && (request_id->scalar_type()==at::kInt || request_id->scalar_type()==at::kLong),"invalid request ID");
  a.physical=physical->data_ptr<int32_t>();a.ps=physical->stride(0);a.map=request_map->data_ptr();a.request=request_id->data_ptr();a.map64=request_map->scalar_type()==at::kLong;a.request64=request_id->scalar_type()==at::kLong;a.ms=request_map->stride(0);
 }
 if(a.rows) dispatch<true>(a,capacity,algorithm,threads,at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream());
}
