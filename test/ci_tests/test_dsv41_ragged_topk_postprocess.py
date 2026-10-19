# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Exact ragged top-k postprocessing, untouched inputs and padded output views."""
import argparse
import json
from pathlib import Path
import statistics

import pytest
import torch


def get_op():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires a HIP GPU')
    from lightop.attention import dsv41_ragged_topk_postprocess
    return dsv41_ragged_topk_postprocess


def reference(scores,selected,offsets,slots,pages,raw,mask):
    if mask:
        cols=selected.long()-offsets[:,None]
        values=scores.gather(1,cols.clamp(0,scores.shape[1]-1))
        valid=(cols>=0)&(cols<scores.shape[1])&(values>-torch.inf)
        selected=selected.masked_fill(~valid,-1)
    sentinel=torch.iinfo(torch.int32).max
    selected=selected.masked_fill(selected<0,sentinel).sort(dim=-1).values
    chosen=selected!=sentinel
    pages.copy_(torch.where(chosen,slots[selected.clamp_max(slots.numel()-1)],-1).int())
    if raw is not None:raw.copy_(torch.where(chosen,selected-offsets[:,None],-1))


def make_case(rows=17,k=512,width=2048,mask=True,raw=True,strided=False,slot_dtype=torch.int64):
    scores=torch.randn((rows,width),device='cuda',dtype=torch.float32)
    scores[:,::13]=-torch.inf;scores[:,1::67]=torch.nan;scores[:,2::71]=torch.inf
    offsets=(torch.arange(rows,device='cuda',dtype=torch.int32)%4)*width
    selected=torch.randint(0,width,(rows,k),device='cuda',dtype=torch.int32)+offsets[:,None]
    selected[:,::17]=-1
    if k>3:
        selected[:,1]=torch.iinfo(torch.int32).max
        selected[:,2]=torch.iinfo(torch.int32).min
        selected[:,3]=width*4+7
    slots=torch.arange(max(1,width*4),device='cuda',dtype=slot_dtype)*3
    if slot_dtype==torch.int64:slots.add_(2**31)
    if strided:
        def spaced(tensor):
            backing=torch.empty(tuple(2*s for s in tensor.shape),device='cuda',dtype=tensor.dtype)
            view=backing[(slice(None,None,2),)*tensor.dim()];view.copy_(tensor);return view
        scores,selected,offsets,slots=map(spaced,(scores,selected,offsets,slots))
    backing=torch.full((rows*2,k*2+7),-77,device='cuda',dtype=torch.int32)
    pages=backing[::2,:k*2:2]
    raw_back=torch.full_like(backing,-77) if raw else None
    out_raw=raw_back[::2,:k*2:2] if raw else None
    return (scores,selected,offsets,slots,pages,out_raw,mask),backing,raw_back


def check_case(op,case,backing,raw_back):
    expected=backing.clone();expected_raw=raw_back.clone() if raw_back is not None else None
    selected_before=case[1].clone()
    k=case[1].shape[1]
    reference(*case[:4],expected[::2,:k*2:2],expected_raw[::2,:k*2:2] if expected_raw is not None else None,case[-1])
    op(*case)
    torch.testing.assert_close(backing,expected,rtol=0,atol=0)
    if raw_back is not None:torch.testing.assert_close(raw_back,expected_raw,rtol=0,atol=0)
    torch.testing.assert_close(case[1],selected_before,rtol=0,atol=0)


def test_exact():
    op=get_op();torch.manual_seed(17)
    for k in (0,1,7,127,128,129,256,511,512,513,1024,2048):
        for mask in (False,True):
            for raw in (False,True):
                for strided in (False,True):
                    check_case(op,*make_case(7,k,257,mask,raw,strided))
    for dtype in (torch.int32,torch.int64):
        check_case(op,*make_case(slot_dtype=dtype))
    check_case(op,*make_case(rows=0))
    case,backing,raw_back=make_case()
    case[1].fill_(-7)
    check_case(op,case,backing,raw_back)


def test_topk_integration():
    op=get_op()
    from lightop import topk_transform_ragged_v2
    case,backing,raw_back=make_case(4,512,2048)
    lengths=torch.tensor([512,513,2048,0],device='cuda',dtype=torch.int32)
    topk_transform_ragged_v2(case[0],lengths,out_offsets=case[2],out_indices=case[1])
    check_case(op,case,backing,raw_back)


def test_stream_and_graph():
    op=get_op();case,backing,raw_back=make_case()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):op(*case)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):op(*case)
    for _ in range(3):
        case[1].copy_(case[1].roll(1,-1));case[0][:,::3]=-torch.inf
        expected=backing.clone();expected_raw=raw_back.clone()
        reference(*case[:4],expected[::2,:1024:2],expected_raw[::2,:1024:2],True)
        graph.replay()
        torch.testing.assert_close(backing,expected,rtol=0,atol=0)
        torch.testing.assert_close(raw_back,expected_raw,rtol=0,atol=0)


def test_invalid_metadata():
    op=get_op();case,_,_=make_case()
    for index,bad in ((0,case[0].half()),(1,case[1].long()),(2,case[2].long()),
                       (3,case[3].float()),(4,case[4].cpu()),(5,case[5][:1])):
        wrong=list(case);wrong[index]=bad
        with pytest.raises((RuntimeError,ValueError,TypeError)):op(*wrong)


def benchmark():
    op=get_op();rows=[]
    for count in (128,4096,32768):
        case,_,_=make_case(count,512,1024)
        samples={'torch':[],'lightop':[]}
        for repeat in range(5):
            funcs=(('torch',lambda:reference(*case)),('lightop',lambda:op(*case)))
            for name,run in (funcs if repeat%2==0 else reversed(funcs)):
                for _ in range(10):run()
                torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(100):run()
                b.record();torch.cuda.synchronize();samples[name].append(a.elapsed_time(b)*10)
        means={name:statistics.mean(v) for name,v in samples.items()}
        logical_bytes=count*512*24+count*4
        row=dict(rows=count,k=512,mean_us=means,samples_us=samples,speedup=means['torch']/means['lightop'],
                 logical_bytes=logical_bytes,effective_GB_s={name:logical_bytes/value/1000 for name,value in means.items()})
        rows.append(row);print(json.dumps(row),flush=True)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--benchmark',action='store_true');p.add_argument('--json',type=Path);a=p.parse_args()
    for test in (test_exact,test_topk_integration,test_stream_and_graph,test_invalid_metadata):
        test();print(test.__name__+': PASS',flush=True)
    result=dict(correctness='PASS',rows=benchmark() if a.benchmark else [])
    if a.json:a.json.write_text(json.dumps(result,indent=2))
