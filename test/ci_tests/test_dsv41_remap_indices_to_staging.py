# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Exact staging map tests and optional synchronized eager Torch comparison."""
import argparse
import json
from pathlib import Path
import statistics

import pytest
import torch


def get_op():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires a HIP GPU')
    from lightop.attention import dsv41_remap_indices_to_staging
    return dsv41_remap_indices_to_staging


def reference(indices,pages,page_size,max_pages,static=False):
    if static:
        slots=torch.where(pages>=0,pages.long(),max_pages+torch.arange(pages.numel(),device=pages.device))
        page_map=torch.full((max_pages+pages.numel(),),-1,dtype=torch.int32,device=indices.device)
    else:
        slots=pages.long();page_map=torch.full((max_pages,),-1,dtype=torch.int32,device=indices.device)
    page_map[slots]=torch.arange(pages.numel(),device=indices.device,dtype=torch.int32)
    valid=indices>=0;safe=indices.clamp_min(0)
    ids=torch.div(safe,page_size,rounding_mode='floor')
    offset=safe-ids*page_size
    return torch.where(valid,page_map[ids.long()].to(indices.dtype)*page_size+offset,indices)


def test_exact():
    op=get_op();torch.manual_seed(17)
    for dtype in (torch.int32,torch.int64):
        for page_dtype in (torch.int32,torch.int64):
            for size in (1,7,64,256):
                for static in (False,True):
                    pages=torch.randperm(97,device='cuda')[:43].to(page_dtype)
                    if static:
                        pages=torch.cat([pages[:17],torch.full((3,),-1,device='cuda',dtype=page_dtype),pages[17:]])
                    indices=torch.randint(0,97*size,(19,258),device='cuda',dtype=dtype)
                    indices[:,0]=-1;indices[:,1]=-7;indices[0,2]=torch.iinfo(dtype).min
                    for data in (indices,indices[:,::2],indices.t(),indices.flatten()[::3],indices[0,0],
                                 indices[:0],indices.reshape(19,2,129).transpose(0,2)):
                        backing=torch.empty(pages.numel()*2,device='cuda',dtype=page_dtype)
                        backing[::2]=pages
                        case=(data,backing[::2],size,97,static)
                        torch.testing.assert_close(op(*case),reference(*case),rtol=0,atol=0)
    # Dynamic negative pages wrap, unlike static padding.
    for static in (False,True):
        pages=torch.tensor([-1,-3,0],device='cuda',dtype=torch.int64)
        indices=torch.arange(7*16,device='cuda',dtype=torch.int32)
        case=(indices,pages,16,7,static)
        torch.testing.assert_close(op(*case),reference(*case),rtol=0,atol=0)
    pages=torch.empty(0,device='cuda',dtype=torch.int64)
    for shape in ((0,),(2,0),(13,)):
        ids=torch.zeros(shape,device='cuda',dtype=torch.int32)
        case=(ids,pages,256,97,False)
        torch.testing.assert_close(op(*case),reference(*case),rtol=0,atol=0)


def test_stream_and_graph():
    op=get_op()
    ids=torch.arange(16384,device='cuda',dtype=torch.int64).view(128,128)
    pages=torch.arange(64,device='cuda',dtype=torch.int64)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):actual=op(ids,pages,256,97,False)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=op(ids,pages,256,97,False)
    for _ in range(3):
        pages.copy_(pages.roll(1));ids[:,0]=-7
        graph.replay()
        torch.testing.assert_close(actual,reference(ids,pages,256,97),rtol=0,atol=0)


def test_invalid_metadata():
    op=get_op();ids=torch.zeros(17,device='cuda',dtype=torch.int32)
    pages=torch.arange(7,device='cuda',dtype=torch.int64)
    case=[ids,pages,256,97,False]
    for index,bad in ((0,ids.float()),(0,ids.cpu()),(1,pages.float()),(1,pages.view(1,-1)),(2,0),(3,-1)):
        wrong=case.copy();wrong[index]=bad
        with pytest.raises((RuntimeError,ValueError,TypeError)):op(*wrong)


def benchmark():
    op=get_op();rows=[]
    for count in (4096,32768,524288,4194304):
        ids=torch.arange(count,device='cuda',dtype=torch.int32)%(97*256)
        pages=torch.arange(0,97,2,device='cuda',dtype=torch.int64)
        case=(ids,pages,256,97,False)
        samples={'torch':[],'lightop':[]}
        for run in (lambda:reference(*case),lambda:op(*case)):
            for _ in range(10):run()
        for repeat in range(5):
            order=(('torch',lambda:reference(*case)),('lightop',lambda:op(*case)))
            for name,run in (order if repeat%2==0 else reversed(order)):
                torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):out=run()
                end.record();torch.cuda.synchronize();samples[name].append(start.elapsed_time(end)*10)
        means={name:statistics.mean(v) for name,v in samples.items()}
        logical_bytes=count*12+97*4+pages.numel()*12
        row=dict(count=count,mean_us=means,samples_us=samples,speedup=means['torch']/means['lightop'],
                 logical_bytes=logical_bytes,effective_GB_s={name:logical_bytes/value/1000 for name,value in means.items()})
        rows.append(row);print(json.dumps(row),flush=True)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--benchmark',action='store_true');p.add_argument('--json',type=Path);a=p.parse_args()
    for test in (test_exact,test_stream_and_graph,test_invalid_metadata):
        test();print(test.__name__+': PASS',flush=True)
    result=dict(correctness='PASS',rows=benchmark() if a.benchmark else [])
    if a.json:a.json.write_text(json.dumps(result,indent=2))
