// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "../mqa_logits_api.h"
#include "dispatch/auto.hpp"
#include "../memory.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <string>

namespace lightop::mqa_int8 {
namespace {
using lightop::mqa_beta::ByteRows;

void check_tensor(const torch::Tensor &t, const torch::Tensor &q, at::ScalarType dtype,
                  const char *name) {
    TORCH_CHECK(t.defined() && t.is_cuda() && t.device() == q.device(), "mqa_logits INT8: ", name,
                " must be on Q's CUDA/HIP device");
    TORCH_CHECK(t.scalar_type() == dtype, "mqa_logits INT8: ", name, " must have dtype ", dtype,
                ", got ", t.scalar_type());
}

ByteRows tensor_rows(const torch::Tensor &t, int64_t rows, int64_t stride, int64_t width,
                     const char *name) {
    const uint64_t bytes = t.element_size();
    const uint64_t limit = uint64_t(std::numeric_limits<int64_t>::max()) / bytes;
    TORCH_CHECK(rows > 0 && width > 0 && stride >= width && uint64_t(width) <= limit &&
                    uint64_t(rows - 1) <= (limit - width) / uint64_t(stride),
                "mqa_logits INT8: ", name, " address span exceeds signed 64-bit range");
    const uint64_t address = reinterpret_cast<uintptr_t>(t.data_ptr());
    TORCH_CHECK((address & 3) == 0, "mqa_logits INT8: ", name,
                " requires 4-byte pointer alignment");
    const ByteRows result{address, uint64_t(rows), uint64_t(rows == 1 ? width : stride) * bytes,
                          uint64_t(width) * bytes};
    TORCH_CHECK(lightop::mqa_beta::valid_rows(result), "mqa_logits INT8: ", name,
                " address span overflows");
    return result;
}

void check_disjoint(const ByteRows &out, const ByteRows &input, const char *name) {
    TORCH_CHECK(!lightop::mqa_beta::rows_overlap(out, input), "mqa_logits INT8: D_out overlaps ",
                name);
}
} // namespace

torch::Tensor mqa_logits_int8(const torch::Tensor &Q, const torch::Tensor &K,
                              const torch::Tensor &weights,
                              const torch::Tensor &ks, const torch::Tensor &ke,
                              int m, int n, int h, int d,
                              const torch::optional<torch::Tensor> &kv_scale, bool clean_logits,
                              const torch::optional<torch::Tensor> &out) {
    TORCH_CHECK(Q.defined() && Q.is_cuda(), "mqa_logits INT8 requires CUDA/HIP Q");
    // try_dispatch selects this adapter only for INT8 Q.
    check_tensor(K, Q, at::kChar, "K");
    check_tensor(weights, Q, at::kFloat, "weights");
    check_tensor(ks, Q, at::kInt, "ks");
    check_tensor(ke, Q, at::kInt, "ke");
    TORCH_CHECK(m >= 0 && n >= 0 && h >= 1 && h <= 128 && !(h & (h - 1)) && d == 128,
                "mqa_logits INT8 requires M,N>=0, D=128 and power-of-two H in [1,128]");
    TORCH_CHECK(Q.dim() == 3 && Q.size(0) == m && Q.size(1) == h && Q.size(2) == d &&
                    K.dim() == 2 && K.size(0) == n && K.size(1) == d,
                "mqa_logits INT8: explicit M/N/H/D must match Q and K shapes");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == m && weights.size(1) == h,
                "mqa_logits INT8: weights must have shape [M,H]");
    TORCH_CHECK(ks.dim() == 1 && ke.dim() == 1 && ks.numel() >= m && ke.numel() >= m &&
                    ks.is_contiguous() && ke.is_contiguous(),
                "mqa_logits INT8: ks/ke must be contiguous int32 vectors with at least M elements");
    TORCH_CHECK(kv_scale.has_value() && kv_scale->defined(), "mqa_logits INT8 requires kv_scale");
    check_tensor(*kv_scale, Q, at::kFloat, "kv_scale");
    TORCH_CHECK(kv_scale->is_contiguous() && kv_scale->numel() >= n,
                "mqa_logits INT8: kv_scale must be contiguous with at least N elements");
    const c10::cuda::CUDAGuard guard(Q.device());
    hipDeviceProp_t prop{};
    auto status = hipGetDeviceProperties(&prop, Q.get_device());
    TORCH_CHECK(status == hipSuccess, "mqa_logits: ", hipGetErrorString(status));
    const std::string arch = std::string(prop.gcnArchName).substr(0, 6);
    TORCH_CHECK((arch == "gfx936" || arch == "gfx938") && prop.warpSize == 64,
                "mqa_logits beta INT8 requires gfx936/gfx938 with wave64; "
                "unset LIGHTOP_MQA_LOGITS_BETA "
                "or set it to 0 to use the original INT8 implementation");
    torch::Tensor result;
    if (out.has_value() && out->defined()) {
        check_tensor(*out, Q, at::kFloat, "D_out");
        TORCH_CHECK(out->dim() == 2 && out->size(0) >= m && out->size(1) >= n,
                    "mqa_logits INT8: D_out must cover [M,N]");
        result =
            (out->size(0) == m && out->size(1) == n) ? *out : out->slice(0, 0, m).slice(1, 0, n);
    } else {
        result = torch::empty({m, n}, Q.options().dtype(at::kFloat));
    }
    if (!m || !n)
        return result;
    TORCH_CHECK(Q.stride(2) == 1 && Q.stride(1) == 128 && Q.stride(0) == int64_t(h) * 128 &&
                    K.stride(1) == 1 && K.stride(0) == 128,
                "mqa_logits INT8: Q/K require contiguous D=128 layout; got strides ", Q.strides(),
                " and ", K.strides());
    TORCH_CHECK(weights.stride(1) == 1 && weights.stride(0) >= h && result.stride(1) == 1 &&
                    result.stride(0) >= n,
                "mqa_logits INT8: weights/D_out need contiguous columns and nonoverlapping rows; "
                "got strides ",
                weights.strides(), " and ", result.strides());
    const auto output_rows = tensor_rows(result, m, result.stride(0), n, "D_out");
    check_disjoint(output_rows, tensor_rows(Q, m, int64_t(h) * 128, int64_t(h) * 128, "Q"), "Q");
    check_disjoint(output_rows, tensor_rows(K, n, 128, 128, "K"), "K");
    check_disjoint(output_rows, tensor_rows(weights, m, weights.stride(0), h, "weights"),
                   "weights");
    check_disjoint(output_rows, tensor_rows(ks, 1, m, m, "ks"), "ks");
    check_disjoint(output_rows, tensor_rows(ke, 1, m, m, "ke"), "ke");
    check_disjoint(output_rows, tensor_rows(*kv_scale, 1, n, n, "kv_scale"), "kv_scale");
    MqaInt8Params p{};
    p.q = static_cast<const uint8_t *>(Q.data_ptr());
    p.kv = static_cast<const uint8_t *>(K.data_ptr());
    p.weights = weights.data_ptr<float>();
    p.kv_scale = kv_scale->data_ptr<float>();
    p.out = result.data_ptr<float>();
    p.m = m;
    p.n = n;
    p.h = h;
    p.d = d;
    p.q_stride_m = Q.stride(0);
    p.q_stride_h = Q.stride(1);
    p.kv_stride_n = K.stride(0);
    p.weights_stride_m = weights.stride(0);
    p.out_stride_m = result.stride(0);
    p.ks = ks.data_ptr<int32_t>();
    p.ke = ke.data_ptr<int32_t>();
    p.clean_logits = clean_logits;
    const auto stream = at::cuda::getCurrentCUDAStream(Q.get_device());
    status = launch(p, resolve(p), stream, prop.maxGridSize[0], prop.maxGridSize[1]);
    TORCH_CHECK(status == hipSuccess,
                "mqa_logits INT8 launch/configuration failed: ", hipGetErrorString(status));
    return result;
}
} // namespace lightop::mqa_int8

#include "kernels/m64.hpp"
#include "../../../topk/adaptive_topk.h"
namespace lightop::mqa_int8 {
template<int Mode,int Bits>
void launch_stage(MqaTopKParams p,hipStream_t stream) {
    const dim3 grid(Mode==1?8:(p.n+255)/256,(p.m+63)/64);
    if constexpr (Mode == 3) {
        if(p.n%256==0) { m64<32,false,true,true,true,Mode,Bits><<<grid,256,0,stream>>>(p); }
        else { m64<32,false,false,true,true,Mode,Bits><<<grid,256,0,stream>>>(p); }
        return;
    }
    if(p.m%64==0) {
        if(p.n%256==0) { m64<32,true,true,true,true,Mode,Bits><<<grid,256,0,stream>>>(p); }
        else { m64<32,true,false,true,true,Mode,Bits><<<grid,256,0,stream>>>(p); }
    } else {
        if(p.n%256==0) { m64<32,false,true,true,true,Mode,Bits><<<grid,256,0,stream>>>(p); }
        else { m64<32,false,false,true,true,Mode,Bits><<<grid,256,0,stream>>>(p); }
    }
}

__global__ __launch_bounds__(256) void compact_mqa_fallback_rows(
    const uint32_t* q, const float* weights, int64_t weights_stride,
    const int* starts, const int* ends, const int* counts, int width,
    uint32_t* packed_q, float* packed_weights, int* packed_starts,
    int* packed_ends, int* row_map, int* total, int* source_rows) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const int start = starts[row], end = ends[row], count = counts[row];
    const bool needed = start >= 0 && end >= start && end <= width &&
        (end - start <= 2048 || count < 2048 || count > 16384);
    if (tid == 0) row_map[row] = -1;
    if (!needed) return;
    __shared__ int slot;
    if (tid == 0) {
        slot = atomicAdd(total, 1);
        row_map[row] = slot;
        if(source_rows) source_rows[slot] = row;
        packed_starts[slot] = start;
        packed_ends[slot] = end;
    }
    __syncthreads();
    for (int col = tid; col < 1024; col += 256)
        packed_q[int64_t(slot) * 1024 + col] = q[int64_t(row) * 1024 + col];
    if (tid < 32)
        packed_weights[int64_t(slot) * 32 + tid] = weights[int64_t(row) * weights_stride + tid];
}

__global__ void prepare_mqa_fallback_batches(const int* total,int* batches,int rows,int batch_rows) {
    const int batch=threadIdx.x+blockIdx.x*blockDim.x;
    const int begin=batch*batch_rows;
    if(begin<rows)batches[batch]=min(batch_rows,max(0,*total-begin));
}

MqaTopKParams checked_params(const torch::Tensor& q,const torch::Tensor& k,
    const torch::Tensor& weights,const torch::Tensor& scale,
    const torch::Tensor& starts,const torch::Tensor& ends) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kChar && q.dim()==3 && q.size(1)==32 && q.size(2)==128 && q.is_contiguous(),"Q must be contiguous device INT8 [M,32,128]");
    const auto check=[&](const torch::Tensor& t,at::ScalarType type) { TORCH_CHECK(t.is_cuda() && t.device()==q.device() && t.scalar_type()==type,"device/dtype mismatch"); };
    check(k,at::kChar);check(weights,at::kFloat);check(scale,at::kFloat);check(starts,at::kInt);check(ends,at::kInt);
    const int64_t m=q.size(0),n=k.size(0);
    TORCH_CHECK(m>0 && m<=16384 && n>=2048 && n<=1179648,"unsupported M/N");
    TORCH_CHECK(k.dim()==2 && k.size(1)==128 && k.is_contiguous(),"K must be contiguous [N,128]");
    TORCH_CHECK(weights.dim()==2 && weights.size(0)==m && weights.size(1)==32 && weights.stride(1)==1 && weights.stride(0)>=32,"weights layout");
    TORCH_CHECK(scale.numel()>=n && scale.is_contiguous(),"scale layout");
    TORCH_CHECK(starts.dim()==1 && ends.dim()==1 && starts.numel()==m && ends.numel()==m && starts.is_contiguous() && ends.is_contiguous(),"bounds layout");
    for(const auto& t:{q,k,weights,scale,starts,ends}) TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr())&3)==0,"unaligned tensor");
    hipDeviceProp_t prop{};TORCH_CHECK(hipGetDeviceProperties(&prop,q.get_device())==hipSuccess,"device properties");
    TORCH_CHECK(std::string(prop.gcnArchName).substr(0,6)=="gfx936" && prop.warpSize==64,"private prototype requires gfx936 wave64");
    MqaTopKParams p{};
    p.q=static_cast<const uint8_t*>(q.data_ptr());p.kv=static_cast<const uint8_t*>(k.data_ptr());p.weights=weights.data_ptr<float>();p.kv_scale=scale.data_ptr<float>();
    p.m=m;p.n=n;p.h=32;p.d=128;p.q_stride_m=q.stride(0);p.q_stride_h=128;p.kv_stride_n=128;p.weights_stride_m=weights.stride(0);p.out_stride_m=n;p.ks=starts.data_ptr<int>();p.ke=ends.data_ptr<int>();p.clean_logits=true;
    return p;
}

__global__ __launch_bounds__(256) void gather_mqa_uniform_sample(
    const uint4* keys, const float* scales, uint4* sampled_keys,
    float* sampled_scales, int count, int stride, int center) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < count*8) { uint4 value;__builtin_memcpy(&value,keys+int64_t(center+(i/8)*stride)*8+i%8,sizeof(value));sampled_keys[i]=value; }
    if (i < count) sampled_scales[i] = scales[center+i*stride];
}
__global__ void mqa_sample_bounds(const int* starts, const int* ends,
    int* sample_starts, int* sample_ends, int rows, int count, int stride, int center) {
    const int row = blockIdx.x*blockDim.x+threadIdx.x;
    if (row < rows) {
        const int begin = max(0,min(count,(starts[row]-center+stride-1)/stride));
        const int end = max(begin,min(count,max(0,(ends[row]-center+stride-1)/stride)));
        sample_starts[row] = begin;
        sample_ends[row] = end;
    }
}
template<bool Parallel>
__global__ __launch_bounds__(256) void mqa_sample_histogram_gate(
    const float* scores, const int* starts, const int* ends,
    const int* sample_starts, const int* sample_ends, int samples, int target,
    float* gates, int* counts) {
    using FloatReduce = hipcub::BlockReduce<float,256>;
    using IntReduce = hipcub::BlockReduce<int,256>;
    using IntScan = hipcub::BlockScan<int,256>;
    __shared__ union { FloatReduce::TempStorage f; IntReduce::TempStorage i; IntScan::TempStorage scan; } temporary;
    __shared__ int selected_bin;
    __shared__ int histogram[2048];
    __shared__ float reference;
    __shared__ int count;
    const int tid=threadIdx.x, row=blockIdx.x;
    const int begin=sample_starts[row],end=sample_ends[row];
    const float* input=scores+int64_t(row)*samples;
    for(int bin=tid;bin<2048;bin+=256) histogram[bin]=0;
    const float first=tid<samples && tid>=begin && tid<end && isfinite(input[tid]) ? input[tid] : -__builtin_inff();
    const float ref=FloatReduce(temporary.f).Reduce(first,hipcub::Max());
    if(tid==0) reference=isfinite(ref)?ref:0.f;
    __syncthreads();
    int local=0;
    for(int i=begin+tid;i<end;i+=256) {
        const float value=input[i];
        if(isfinite(value)) {
            const uint32_t half=__half_as_ushort(__float2half_rd(value-reference));
            const uint32_t ordered=half&0x8000u?(~half&65535u):(half^0x8000u);
            atomicAdd(histogram+(ordered>>5),1);++local;
        }
    }
    const int total=IntReduce(temporary.i).Sum(local);
    if(tid==0) count=total;
    __syncthreads();
    const int visible=ends[row]-starts[row];
    const bool eligible=visible>target && count>=16;
    if constexpr (Parallel) {
        if(eligible) {
            const int rank=max(1,min(count,int((int64_t(count)*target*17/16+visible-1)/visible)));
            int bins[8],local_sum=0;
#pragma unroll
            for(int j=0;j<8;++j){bins[j]=histogram[2047-tid*8-j];local_sum+=bins[j];}
            int inclusive;
            IntScan(temporary.scan).InclusiveSum(local_sum,inclusive);
            __syncthreads();
            int cumulative=inclusive-local_sum;
            if(cumulative<rank && inclusive>=rank) {
                for(int j=0;j<8;++j){cumulative+=bins[j];if(cumulative>=rank){selected_bin=2047-tid*8-j;break;}}
            }
            __syncthreads();
        }
    }
    if(tid==0) {
        float gate=__builtin_inff();
        if(eligible) {
            int chosen=0;
            if constexpr (Parallel) chosen=selected_bin;
            else {
                const int rank=max(1,min(count,int((int64_t(count)*target*17/16+visible-1)/visible)));
                int cumulative=0;
                for(int bin=2047;bin>=0;--bin){cumulative+=histogram[bin];if(cumulative>=rank){chosen=bin;break;}}
            }
            const uint32_t ordered=uint32_t(chosen)<<5;
            const uint16_t half=ordered&0x8000u?ordered^0x8000u:(~ordered&65535u);
            gate=reference+__half2float(__ushort_as_half(half));
        }
        gates[row]=gate;
        counts[row]=isfinite(gate)?0:(1<<28);
    }
}
__device__ void repair_candidate_ties(const float* scores,const int* ids,int n,
    at::native::adaptive_topk_detail::Shared<2048>& s) {
    namespace topk=at::native::adaptive_topk_detail;
    using U32Reduce=hipcub::BlockReduce<uint32_t,1024>;
    __shared__ U32Reduce::TempStorage temporary;
    const int tid=threadIdx.x;
    uint32_t minimum=0xffffffffu;
    for(int j=0;j<2;++j)minimum=min(minimum,topk::key(scores[s.selected[tid*2+j]]));
    const uint32_t boundary=U32Reduce(temporary).Reduce(minimum,hipcub::Min());
    if(tid==0)s.pivot=boundary;
    __syncthreads();
    const uint32_t pivot=s.pivot;
    topk::Counts local{0,0,0};
    for(int at=tid;at<n;at+=1024){const uint32_t key=topk::key(scores[at]);local.positive+=key>pivot;local.zero+=key==pivot;}
    const auto all=topk::CountReduce(s.temp.counts).Reduce(local,topk::AddCounts{});
    if(tid==0){s.rank=2048-all.positive;s.active=all.zero;s.prefix=0;s.mask=0;}
    __syncthreads();
    if(s.active<=s.rank)return;
    for(int shift=24;shift>=0;shift-=8){
        if(tid<256)s.hist[tid]=0;
        __syncthreads();
        const uint32_t prefix=s.prefix,mask=s.mask;
        for(int base=0;base<n;base+=1024){
            const int at=base+tid;
            const uint32_t key=at<n?~uint32_t(ids[at]):0;
            const bool valid=at<n && topk::key(scores[at])==pivot && (key&mask)==prefix;
            topk::histogram_add(s.hist,valid,(key>>shift)&255);
        }
        __syncthreads();
        int cumulative=0,total=0;const int count=tid<256?s.hist[tid]:0;
        topk::Scan(s.temp.scan).InclusiveSum(count,cumulative,total);
        __syncthreads();
        const int above=total-cumulative;
        if(tid<256 && above<s.rank && above+count>=s.rank){s.active=count;s.hist[0]=s.rank-above;s.hist[1]=tid;}
        __syncthreads();
        if(tid==0){s.rank=s.hist[0];s.prefix|=uint32_t(s.hist[1])<<shift;s.mask|=0xffu<<shift;}
        __syncthreads();
        if(s.active==s.rank)break;
    }
    const uint32_t min_id_key=s.prefix;
    __syncthreads();
    if(tid==0)s.emitted=0;
    __syncthreads();
    for(int base=0;base<n;base+=1024){
        const int at=base+tid;const uint32_t key=at<n?topk::key(scores[at]):0;
        const bool keep=at<n && (key>pivot || (key==pivot && ~uint32_t(ids[at])>=min_id_key));
        const int dest=topk::reserve_wave(keep,&s.emitted);
        if(keep && dest<2048)s.selected[dest]=at;
    }
    __syncthreads();
}

template<int Mode, bool SortSelected>
__global__ __launch_bounds__(1024) void sampled_candidate_topk(
    const float* candidates, const int* ids, const int* counts, const float* dense,
    const int* starts, const int* ends, const int* offsets, int width,
    int* output, float* values, float* thresholds, int* stats, const int* fallback_rows,
    const int* source_rows, const int* batch_count, int batch_begin) {
    namespace topk=at::native::adaptive_topk_detail;
    __shared__ topk::SelectionStorage<2048,SortSelected> storage;
    auto& s=storage.selection;
    const int tid=threadIdx.x;
    int row=blockIdx.x;
    if constexpr (Mode==2) {
        if(int(blockIdx.x)>=*batch_count)return;
        row=source_rows[batch_begin+blockIdx.x];
    }
    const int start=starts[row],end=ends[row],n=end-start,count=counts[row];
    const bool candidate=Mode!=2 && n>2048 && count>=2048 && count<=16384;
    if(start<0 || end<start || end>width) {
        for(int i=tid;i<2048;i+=1024) { output[int64_t(row)*2048+i]=-1;if(values)values[int64_t(row)*2048+i]=-__builtin_inff(); }
        if(tid==0 && thresholds)thresholds[row]=-__builtin_inff();
        if(tid==0 && stats){stats[row*2]=-1;stats[row*2+1]=count;}
        return;
    }
    if constexpr (Mode==1) {if(!candidate)return;}
    const int dense_row=Mode==2?int(blockIdx.x):candidate?0:fallback_rows[row];
    const float* input=candidate?candidates+int64_t(row)*16384:dense+int64_t(dense_row)*width+start;
    if(candidate) { topk::coarse_select(input,count,s);repair_candidate_ties(input,ids+int64_t(row)*16384,count,s); }
    else if(n>2048)topk::coarse_select(input,n,s);
    else {for(int i=tid;i<2048;i+=1024)s.selected[i]=i<n?i:INT_MAX;__syncthreads();}
    if(tid==0 && stats){stats[row*2]=candidate?0:n<=2048?1:count>16384?2:3;stats[row*2+1]=count;}
    if constexpr (SortSelected) {
        uint32_t selected_keys[2];int selected_ids[2];
        for(int j=0;j<2;++j) {
            const int selected=s.selected[tid*2+j];
            const int id=candidate?ids[int64_t(row)*16384+selected]-start:selected;
            const bool valid=id>=0 && id<n;
            selected_keys[j]=topk::key(valid?input[selected]:-__builtin_inff());
            selected_ids[j]=valid?id:-1;
        }
        topk::sorted_topk_tail<2048>(selected_keys,selected_ids,storage,row,offsets[row],nullptr,output,values,thresholds);
    } else {
        for(int at=tid;at<2048;at+=1024) {
            const int selected=s.selected[at];
            const int id=candidate?ids[int64_t(row)*16384+selected]-start:selected;
            output[int64_t(row)*2048+at]=id>=0 && id<n?id+offsets[row]:-1;
        }
    }
}
__global__ void sampled_native_stats(int* stats,const int* starts,const int* ends,int rows){
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<rows){stats[row*2]=4;stats[row*2+1]=ends[row]-starts[row];}
}
torch::Tensor mqa_logits_topk_sampled_int8(const torch::Tensor& q,const torch::Tensor& k,
    const torch::Tensor& weights,const torch::Tensor& scale,const torch::Tensor& starts,
    const torch::Tensor& ends,const torch::Tensor& offsets,
    const torch::optional<torch::Tensor>& sorted_values,
    const torch::optional<torch::Tensor>& next_thresholds,
    const torch::optional<torch::Tensor>& stats, bool parallel_histogram, int64_t fallback_workspace_rows, bool reuse_full_window, bool sort_topk, int64_t min_sample_k, bool direct_candidate_emit) {
    c10::cuda::CUDAGuard guard(q.device());auto p=checked_params(q,k,weights,scale,starts,ends);
    TORCH_CHECK(offsets.device()==q.device() && offsets.scalar_type()==at::kInt && offsets.dim()==1 && offsets.numel()==p.m && offsets.is_contiguous(),"offset layout");
    std::vector<torch::Tensor> outputs;
    auto check_output=[&](const torch::optional<torch::Tensor>& t,at::ScalarType type,std::vector<int64_t> shape){
        if(t){TORCH_CHECK(t->device()==q.device() && t->scalar_type()==type && t->sizes().vec()==shape && t->is_contiguous(),"optional output layout");for(const auto& x:{q,k,weights,scale,starts,ends,offsets})TORCH_CHECK(!t->is_alias_of(x),"output aliases input");for(const auto& x:outputs)TORCH_CHECK(!t->is_alias_of(x),"outputs alias");outputs.push_back(*t);}
    };
    TORCH_CHECK(sort_topk || (!sorted_values && !next_thresholds),"sorted optional outputs require sort_topk=true");
    check_output(sorted_values,at::kFloat,{p.m,2048});check_output(next_thresholds,at::kFloat,{p.m});check_output(stats,at::kInt,{p.m,2});
    const auto f=q.options().dtype(at::kFloat),i=q.options().dtype(at::kInt);
    TORCH_CHECK(fallback_workspace_rows==0 || (fallback_workspace_rows>=64 && fallback_workspace_rows<=16384 && fallback_workspace_rows%64==0),"fallback_workspace_rows must be0 or a multiple of64 in[64,16384]");
    TORCH_CHECK(min_sample_k>=32768 && min_sample_k<=131072,"min_sample_k must be in[32768,131072]");
    const bool bounded=p.n>=min_sample_k && fallback_workspace_rows>0 && p.m>fallback_workspace_rows;
    const int dense_rows=bounded?int(fallback_workspace_rows):int(p.m);
    auto dense=torch::empty({dense_rows,p.n},f),output=torch::empty({p.m,2048},i);
    const auto stream=at::cuda::getCurrentCUDAStream(q.get_device());
    if(p.n<min_sample_k) {
        mqa_logits_int8(q,k,weights,starts,ends,p.m,p.n,32,128,scale,true,dense);
        auto lengths=ends-starts;
        if(sort_topk) { at::native::adaptive_topk_kernel<2048,false,true><<<p.m,1024,0,stream>>>(dense.data_ptr<float>(),p.n,p.n,lengths.data_ptr<int>(),starts.data_ptr<int>(),nullptr,0,0,nullptr,p.m,nullptr,output.data_ptr<int>(),nullptr,true,0,offsets.data_ptr<int>(),sorted_values?sorted_values->data_ptr<float>():nullptr,next_thresholds?next_thresholds->data_ptr<float>():nullptr); }
        else { at::native::adaptive_topk_kernel<2048,false,false><<<p.m,1024,0,stream>>>(dense.data_ptr<float>(),p.n,p.n,lengths.data_ptr<int>(),starts.data_ptr<int>(),nullptr,0,0,nullptr,p.m,nullptr,output.data_ptr<int>(),nullptr,true,0,offsets.data_ptr<int>(),sorted_values?sorted_values->data_ptr<float>():nullptr,next_thresholds?next_thresholds->data_ptr<float>():nullptr); }
        if(stats) { sampled_native_stats<<<(p.m+255)/256,256,0,stream>>>(stats->data_ptr<int>(),starts.data_ptr<int>(),ends.data_ptr<int>(),p.m); }
    } else {
        const int target=p.n<=131072?3072:4096;
        const int rank_tiles=(64*p.n+256*target-1)/(256*target);
        const int tiles=p.n<=131072?rank_tiles:min(32,max(16,rank_tiles));
        const int stride=(p.n+256*tiles-1)/(256*tiles),center=stride/2;
        const int samples=(p.n-center+stride-1)/stride;
        auto sk=torch::empty({samples,128},q.options()),ss=torch::empty({samples},f);
        auto sks=torch::empty({p.m},i),ske=torch::empty({p.m},i),sampled=torch::empty({p.m,samples},f),gates=torch::empty({p.m},f);
        auto counts=torch::empty({p.m},i),candidates=torch::empty({p.m,16384},f),ids=torch::empty({p.m,16384},i);
        auto packed_q=torch::empty_like(q),packed_weights=torch::empty({p.m,32},f),packed_starts=torch::empty({p.m},i),packed_ends=torch::empty({p.m},i),fallback_rows=torch::empty({p.m},i),fallback_count=torch::zeros({1},i);
        gather_mqa_uniform_sample<<<(samples*8+255)/256,256,0,stream>>>(reinterpret_cast<const uint4*>(k.data_ptr()),scale.data_ptr<float>(),reinterpret_cast<uint4*>(sk.data_ptr()),ss.data_ptr<float>(),samples,stride,center);
        mqa_sample_bounds<<<(p.m+255)/256,256,0,stream>>>(p.ks,p.ke,sks.data_ptr<int>(),ske.data_ptr<int>(),p.m,samples,stride,center);
        mqa_logits_int8(q,sk,weights,sks,ske,p.m,samples,32,128,ss,true,sampled);
        if(parallel_histogram) { mqa_sample_histogram_gate<true><<<p.m,256,0,stream>>>(sampled.data_ptr<float>(),p.ks,p.ke,sks.data_ptr<int>(),ske.data_ptr<int>(),samples,target,gates.data_ptr<float>(),counts.data_ptr<int>()); }
        else { mqa_sample_histogram_gate<false><<<p.m,256,0,stream>>>(sampled.data_ptr<float>(),p.ks,p.ke,sks.data_ptr<int>(),ske.data_ptr<int>(),samples,target,gates.data_ptr<float>(),counts.data_ptr<int>()); }
        p.thresholds=reinterpret_cast<const uint32_t*>(gates.data_ptr<float>());p.counts=counts.data_ptr<int>();p.candidate_keys=candidates.data_ptr();p.candidate_ids=ids.data_ptr<int>();p.out=dense.data_ptr<float>();p.out_stride_m=p.n;
        if(direct_candidate_emit && reuse_full_window) { launch_stage<7,32>(p,stream); }
        else if(reuse_full_window) { launch_stage<6,32>(p,stream); }
        else { launch_stage<4,32>(p,stream); }
        auto source_rows=bounded?torch::empty({p.m},i):torch::Tensor();
        compact_mqa_fallback_rows<<<p.m,256,0,stream>>>(reinterpret_cast<const uint32_t*>(p.q),p.weights,p.weights_stride_m,p.ks,p.ke,p.counts,p.n,reinterpret_cast<uint32_t*>(packed_q.data_ptr()),packed_weights.data_ptr<float>(),packed_starts.data_ptr<int>(),packed_ends.data_ptr<int>(),fallback_rows.data_ptr<int>(),fallback_count.data_ptr<int>(),bounded?source_rows.data_ptr<int>():nullptr);
        auto fallback=p;fallback.q=static_cast<const uint8_t*>(packed_q.data_ptr());fallback.weights=packed_weights.data_ptr<float>();fallback.weights_stride_m=32;fallback.ks=packed_starts.data_ptr<int>();fallback.ke=packed_ends.data_ptr<int>();fallback.fallback_count=fallback_count.data_ptr<int>();
        if(!bounded) {
            launch_stage<3,32>(fallback,stream);
            if(sort_topk) { sampled_candidate_topk<0,true><<<p.m,1024,0,stream>>>(candidates.data_ptr<float>(),p.candidate_ids,p.counts,p.out,p.ks,p.ke,offsets.data_ptr<int>(),p.n,output.data_ptr<int>(),sorted_values?sorted_values->data_ptr<float>():nullptr,next_thresholds?next_thresholds->data_ptr<float>():nullptr,stats?stats->data_ptr<int>():nullptr,fallback_rows.data_ptr<int>(),nullptr,nullptr,0); }
            else { sampled_candidate_topk<0,false><<<p.m,1024,0,stream>>>(candidates.data_ptr<float>(),p.candidate_ids,p.counts,p.out,p.ks,p.ke,offsets.data_ptr<int>(),p.n,output.data_ptr<int>(),sorted_values?sorted_values->data_ptr<float>():nullptr,next_thresholds?next_thresholds->data_ptr<float>():nullptr,stats?stats->data_ptr<int>():nullptr,fallback_rows.data_ptr<int>(),nullptr,nullptr,0); }
        } else {
            const int batches=(p.m+dense_rows-1)/dense_rows;
            auto batch_counts=torch::empty({batches},i);
            prepare_mqa_fallback_batches<<<(batches+255)/256,256,0,stream>>>(fallback_count.data_ptr<int>(),batch_counts.data_ptr<int>(),p.m,dense_rows);
            if(sort_topk) { sampled_candidate_topk<1,true><<<p.m,1024,0,stream>>>(candidates.data_ptr<float>(),p.candidate_ids,p.counts,p.out,p.ks,p.ke,offsets.data_ptr<int>(),p.n,output.data_ptr<int>(),sorted_values?sorted_values->data_ptr<float>():nullptr,next_thresholds?next_thresholds->data_ptr<float>():nullptr,stats?stats->data_ptr<int>():nullptr,fallback_rows.data_ptr<int>(),nullptr,nullptr,0); }
            else { sampled_candidate_topk<1,false><<<p.m,1024,0,stream>>>(candidates.data_ptr<float>(),p.candidate_ids,p.counts,p.out,p.ks,p.ke,offsets.data_ptr<int>(),p.n,output.data_ptr<int>(),sorted_values?sorted_values->data_ptr<float>():nullptr,next_thresholds?next_thresholds->data_ptr<float>():nullptr,stats?stats->data_ptr<int>():nullptr,fallback_rows.data_ptr<int>(),nullptr,nullptr,0); }
            for(int batch=0;batch<batches;++batch) {
                const int begin=batch*dense_rows;
                auto part=fallback;part.m=std::min(int(p.m)-begin,dense_rows);
                part.q=static_cast<const uint8_t*>(packed_q.data_ptr())+int64_t(begin)*32*128;
                part.weights=packed_weights.data_ptr<float>()+int64_t(begin)*32;
                part.ks=packed_starts.data_ptr<int>()+begin;part.ke=packed_ends.data_ptr<int>()+begin;
                part.fallback_count=batch_counts.data_ptr<int>()+batch;
                launch_stage<3,32>(part,stream);
                if(sort_topk) { sampled_candidate_topk<2,true><<<part.m,1024,0,stream>>>(candidates.data_ptr<float>(),p.candidate_ids,p.counts,p.out,p.ks,p.ke,offsets.data_ptr<int>(),p.n,output.data_ptr<int>(),sorted_values?sorted_values->data_ptr<float>():nullptr,next_thresholds?next_thresholds->data_ptr<float>():nullptr,stats?stats->data_ptr<int>():nullptr,fallback_rows.data_ptr<int>(),source_rows.data_ptr<int>(),part.fallback_count,begin); }
                else { sampled_candidate_topk<2,false><<<part.m,1024,0,stream>>>(candidates.data_ptr<float>(),p.candidate_ids,p.counts,p.out,p.ks,p.ke,offsets.data_ptr<int>(),p.n,output.data_ptr<int>(),sorted_values?sorted_values->data_ptr<float>():nullptr,next_thresholds?next_thresholds->data_ptr<float>():nullptr,stats?stats->data_ptr<int>():nullptr,fallback_rows.data_ptr<int>(),source_rows.data_ptr<int>(),part.fallback_count,begin); }
            }
        }
    }
    TORCH_CHECK(hipGetLastError()==hipSuccess,"sampled MQA TopK launch failed");return output;
}
}
