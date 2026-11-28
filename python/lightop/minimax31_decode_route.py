"""Graph-safe masked weight preparation, expert remap and validation."""
import torch
import triton
import triton.language as tl


@triton.jit
def _prepare(Ids, Weights, Permutation, OutIds, OutWeights, Check, Enabled, Counts,
             N:tl.constexpr, WR:tl.constexpr, WC:tl.constexpr,
             EXPERTS:tl.constexpr, BLOCK:tl.constexpr, RECORD:tl.constexpr, ASSERT:tl.constexpr):
    element=tl.arange(0,BLOCK)
    active=element<N
    logical=tl.load(Ids+element,active,-1)
    valid=active & (logical>=0) & (logical<128)
    physical=tl.load(Permutation+logical,valid,0)
    mapped=tl.where(valid,physical,logical)
    weight=tl.load(Weights+element//4*WR+element%4*WC,active & (logical>=0),0.0)
    tl.store(OutIds+element,mapped,active)
    tl.store(OutWeights+element,weight,active)
    bits=weight.to(tl.uint32,bitcast=True)&0x7FFFFFFF
    invalid=(mapped<-1) | (mapped>=EXPERTS) | (bits>=0x7F800000) | ((mapped<0)&(bits!=0))
    valid_routes=tl.max((active & invalid).to(tl.int32),0)==0
    if ASSERT:
        tl.device_assert(valid_routes, "invalid physical expert ID or route/padding weights")
    else:
        tl.store(Check,valid_routes)
    if RECORD:
        if tl.load(Enabled)!=0:
            histogram=tl.histogram(logical.to(tl.int32),128,mask=valid)
            tl.atomic_add(Counts+tl.arange(0,128),histogram,histogram!=0,sem="relaxed")


def prepare_route(ids, weights, permutation, *, counters=None, experts=128, assert_inline=False):
    if (ids.dtype!=torch.int64 or ids.ndim!=2 or ids.size(1)!=4
            or not ids.is_contiguous() or weights.shape!=ids.shape
            or weights.dtype!=torch.float32 or permutation.shape!=(128,)
            or not permutation.is_contiguous() or permutation.dtype!=torch.int64):
        raise ValueError("fused MiniMax top-4 route preparation requires its decode ABI")
    count=ids.numel()
    out_ids=torch.empty_like(ids)
    out_weights=torch.empty(ids.shape,device=ids.device,dtype=weights.dtype)
    check=True if assert_inline else torch.empty((),device=ids.device,dtype=torch.bool)
    enabled,hist=(None,None) if counters is None else (counters.enabled,counters.counts)
    _prepare[(1,)](ids,weights,permutation,out_ids,out_weights,None if assert_inline else check,enabled,hist,
                  count,*weights.stride(),experts,max(64,triton.next_power_of_2(count)),
                  counters is not None,assert_inline,num_warps=4,debug=assert_inline)
    return out_ids,out_weights,check
