"""Ordered-FP32 reference and replay contracts for native Top512/1024/2048."""
import os

import numpy as np
import pytest
import torch

from lightop.attention import fast_topk_transform_fused, fast_topk_transform_ragged_fused
from lightop.fuse_topk_transform import topk_transform_512, topk_transform_1024

pytestmark = pytest.mark.skipif(
    os.environ.get("SGL_USE_LIGHTOP_TOPK_BACKAND", "") not in ("", "0")
    or not torch.cuda.is_available() or torch.version.hip is None,
    reason="adaptive TopK requires wave64 HIP and backend 0/default",
)
K = 2048


def reference(scores, lengths, pages, cu, starts=None, k=K):
    result = np.full((scores.shape[0], k), -1, dtype=np.int32)
    for row, n in enumerate(lengths.tolist()):
        start = 0 if starts is None else int(starts[row])
        values = scores[row, start:start+n].contiguous().numpy()
        bits = values.view(np.uint32)
        keys = np.where(bits & np.uint32(0x80000000), ~bits,
                        bits | np.uint32(0x80000000))
        chosen = np.sort(np.argsort(~keys, kind="stable")[:k])
        batch = np.searchsorted(cu.numpy()[1:], row, side="right")
        result[row, :len(chosen)] = pages[batch, chosen].numpy()
    return torch.from_numpy(result)


def values(kind, n):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(n, generator=generator)
    if kind == "negative":
        x = -x.abs()-1
    elif kind == "zero":
        x.zero_()
    elif kind == "sparse":
        x.zero_(); x[:min(300,n)] = torch.arange(1, min(300,n)+1)
    elif kind == "narrow":
        x = 1 + (torch.arange(n) % 17).float() * 2**-23
    elif kind in ("overflow", "underfill"):
        x = torch.rand(n, generator=generator)+2
        ids = torch.arange(2048, dtype=torch.int64)*n//2048
        x[ids] = (torch.arange(2048).float()/2048-2 if kind == "overflow"
                  else torch.arange(2048).float()+100)
    elif kind == "special":
        patterns = np.array([0, 0x80000000, 0x7f800000, 0xff800000,
                             0x7fc00001, 0xffc00001, 0x3f800000], dtype=np.uint32)
        x = torch.from_numpy(np.resize(patterns, n).view(np.float32).copy())
    return x


def inputs(kind, n, rows=6, device="cuda:0"):
    starts = torch.tensor(([0,7,3,0,4,2]*((rows+5)//6))[:rows], dtype=torch.int32)
    lengths = torch.tensor([max(0,n-i%6) for i in range(rows)], dtype=torch.int32)
    storage = torch.full((rows,n+24), float("inf"), dtype=torch.float32)
    scores = storage[:,:n+7]
    for row in range(rows):
        start, count = int(starts[row]), int(lengths[row])
        scores[row,start:start+count] = values(kind,count)
    cu = torch.tensor([0,rows//3,rows//3,rows],dtype=torch.int32)
    page_storage = torch.arange(3*(n+19),dtype=torch.int32).reshape(3,n+19)*7+101
    pages = page_storage[:,:n]
    expected = reference(scores,lengths,pages,cu,starts)
    # Preserve row padding when transferring storage to exercise both strides.
    args = (storage.to(device)[:,:n+7], lengths.to(device),
            page_storage.to(device)[:,:n], cu.to(device), starts.to(device))
    return args, expected


def call(args):
    scores,lengths,pages,cu,starts=args
    return fast_topk_transform_fused(scores,lengths,pages,cu,K,starts)


@pytest.mark.parametrize("n", [8192,32768,65539])
@pytest.mark.parametrize("kind", ["dense","negative","zero","sparse","narrow",
                                 "overflow","underfill","special"])
def test_reference_and_strided_query_groups(kind,n):
    args,expected=inputs(kind,n)
    assert torch.equal(call(args).cpu(),expected)


@pytest.mark.parametrize("n", [0,1,2047,2048,2049,32767])
def test_short_lengths_and_padding(n):
    scores=values("dense",max(n,1))[None,:].cuda()
    lengths=torch.tensor([n],dtype=torch.int32,device="cuda")
    pages=torch.arange(max(n,1),dtype=torch.int32,device="cuda")[None,:]
    cu=torch.tensor([0,1],dtype=torch.int32,device="cuda")
    expected=reference(scores.cpu(),lengths.cpu(),pages.cpu(),cu.cpu())
    assert torch.equal(fast_topk_transform_fused(scores,lengths,pages,cu,K).cpu(),expected)


def test_empty_batch():
    out=fast_topk_transform_fused(
        torch.empty((0,100),device="cuda"),
        torch.empty((0,),dtype=torch.int32,device="cuda"),
        torch.empty((0,100),dtype=torch.int32,device="cuda"),
        torch.zeros((1,),dtype=torch.int32,device="cuda"),K)
    assert out.shape == (0,K)


def test_graph_replay_changes_distribution_and_length():
    args,_=inputs("dense",65539)
    call(args)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out=call(args)
    for kind,n in [("zero",65539),("dense",65539),("overflow",65539),
                   ("underfill",65539),("narrow",65539),("sparse",8192)]:
        replacement,expected=inputs(kind,n)
        args[0].fill_(float("inf"))
        args[0][:,:replacement[0].shape[1]].copy_(replacement[0])
        args[1].copy_(replacement[1]);args[4].copy_(replacement[4])
        # The captured page table stays fixed; recompute reference with it.
        expected=reference(args[0].cpu(),args[1].cpu(),args[2].cpu(),args[3].cpu(),args[4].cpu())
        graph.replay()
        assert torch.equal(out.cpu(),expected)


def test_bs6_verify_shape_on_nondefault_stream():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        args,expected=inputs("dense",65539,rows=30)
        out=call(args)
    stream.synchronize()
    assert torch.equal(out.cpu(),expected)


@pytest.mark.skipif(torch.cuda.device_count()<2,reason="two devices required")
def test_score_device_differs_from_current_device():
    with torch.cuda.device(0):
        args,expected=inputs("narrow",65539,device="cuda:1")
        assert torch.equal(call(args).cpu(),expected)
        assert torch.cuda.current_device()==0


def test_reject_mixed_metadata_devices():
    if torch.cuda.device_count()<2:pytest.skip("two devices required")
    args,_=inputs("dense",8192)
    wrong=list(args);wrong[2]=wrong[2].to("cuda:1")
    with pytest.raises(RuntimeError,match="score device"):
        call(wrong)


def fixed_reference(scores,lengths,pages,k,page_size):
    capacity=min(scores.shape[1],pages.shape[1]*page_size)
    valid_lengths=lengths.clamp(0,capacity)
    identity=torch.arange(scores.shape[1],dtype=torch.int32)[None,:].expand(scores.shape[0],-1)
    cu=torch.arange(scores.shape[0]+1,dtype=torch.int32)
    raw=reference(scores,valid_lengths,identity,cu,k=k)
    mapped=torch.full_like(raw,-1)
    for row in range(scores.shape[0]):
        valid=raw[row]>=0
        ids=raw[row,valid].long()
        physical=pages[row,ids//page_size]
        mapped[row,valid]=torch.where(physical<0,-1,physical*page_size+(ids%page_size).int())
    return mapped,raw


def fixed_inputs(kind,n,k,page_size=64):
    rows=6
    storage=torch.full((rows,n+23),float('inf'))
    scores=storage[:,:n]
    for i in range(rows):scores[i].copy_(values(kind,n))
    lengths=torch.tensor([n,n-7,k-1,k,0,n],dtype=torch.int32)
    count=(n+page_size-1)//page_size
    page_storage=(torch.arange(rows*(count+13),dtype=torch.int32).reshape(rows,count+13)*3+10)
    page_storage[:,0]=-1
    pages=page_storage[:,:count]
    expected=fixed_reference(scores,lengths,pages,k,page_size)
    return (storage.cuda()[:,:n],lengths.cuda(),page_storage.cuda()[:,:count]),expected


def fixed_call(args,k,page_size=64,raw=True):
    out=torch.empty((args[0].shape[0],k),dtype=torch.int32,device=args[0].device)
    raw_out=torch.empty_like(out) if raw else None
    fn=topk_transform_512 if k==512 else topk_transform_1024
    fn(*args,out,page_size,raw_out)
    return out,raw_out


@pytest.mark.parametrize('k',[512,1024])
@pytest.mark.parametrize('kind',['dense','negative','zero','sparse','narrow','overflow','underfill','special'])
@pytest.mark.parametrize('n',[8192,65539])
def test_fixed_topk_reference(k,kind,n):
    args,(mapped,raw)=fixed_inputs(kind,n,k)
    got,got_raw=fixed_call(args,k)
    assert torch.equal(got.cpu(),mapped)
    assert torch.equal(got_raw.cpu(),raw)


@pytest.mark.parametrize('k',[512,1024])
@pytest.mark.parametrize('page_size',[1,64,128])
def test_fixed_page_mapping_and_capacity_clamping(k,page_size):
    args,_=fixed_inputs('dense',8192,k,page_size)
    args=(args[0],torch.tensor([-3,9000,8192,511,1024,2048],dtype=torch.int32,device='cuda'),args[2][:,:args[2].shape[1]//2])
    expected,raw=fixed_reference(*(x.cpu() for x in args),k,page_size)
    out,_=fixed_call(args,k,page_size,raw=False)
    assert torch.equal(out.cpu(),expected)


@pytest.mark.parametrize('k',[512,1024])
def test_fixed_graph_replay(k):
    args,_=fixed_inputs('dense',65539,k)
    fixed_call(args,k)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out,raw=fixed_call(args,k)
    for kind in ['zero','dense','overflow','underfill','narrow']:
        replacement,(expected,expected_raw)=fixed_inputs(kind,65539,k)
        for dest,src in zip(args,replacement):dest.copy_(src)
        graph.replay()
        assert torch.equal(out.cpu(),expected)
        assert torch.equal(raw.cpu(),expected_raw)
    args[1].copy_(torch.tensor([0,1,k-1,k,k+1,8192],dtype=torch.int32,device='cuda'))
    expected,expected_raw=fixed_reference(*(x.cpu() for x in args),k,64)
    graph.replay()
    assert torch.equal(out.cpu(),expected)
    assert torch.equal(raw.cpu(),expected_raw)


@pytest.mark.parametrize('kind',['dense','zero','negative','overflow','underfill'])
def test_ragged_top2048_offsets(kind):
    args,_=inputs(kind,65539)
    scores,lengths,pages,cu,starts=args
    offsets=torch.arange(scores.shape[0],dtype=torch.int32,device='cuda')*100000
    identity=torch.arange(scores.shape[1],dtype=torch.int32)[None,:].expand(scores.shape[0],-1)
    raw=reference(scores.cpu(),lengths.cpu(),identity,torch.arange(scores.shape[0]+1,dtype=torch.int32),starts.cpu())
    expected=torch.where(raw<0,-1,raw+offsets.cpu()[:,None])
    out=fast_topk_transform_ragged_fused(scores,lengths,offsets,K,starts)
    assert torch.equal(out.cpu(),expected)
