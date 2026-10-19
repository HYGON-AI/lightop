# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Exact SWA metadata/gather checks; optional synchronized Torch comparison."""
import argparse
import json
from pathlib import Path
import statistics

import pytest
import torch


def get_op():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires a HIP GPU')
    from lightop.attention import dsv41_build_swa_token_ids
    return dsv41_build_swa_token_ids


def reference(seq, extend, req, table, mapping, window, total):
    lens = torch.minimum(seq, extend + window - 1).int()
    first = seq - lens
    offsets = torch.zeros(seq.numel() + 1, dtype=torch.int32, device=seq.device)
    offsets[1:] = lens.cumsum(0).int()
    parts = [mapping[table[r, s-n:s].long()].int()
             for r, s, n in zip(req.tolist(), seq.tolist(), lens.tolist())]
    ids = torch.cat(parts) if parts else torch.empty(0, dtype=torch.int32, device=seq.device)
    assert ids.numel() == total
    return ids, first, lens, offsets


def make_case(batch=8, width=8192, window=128, strided=False, empty=False, query_len=None):
    seq = torch.tensor([max(0, width-i*7) for i in range(batch)], device='cuda', dtype=torch.int32)
    query_len = width//2 if query_len is None else query_len
    extend = torch.tensor([max(0, query_len-i*3) for i in range(batch)], device='cuda', dtype=torch.int32)
    req = torch.arange(batch-1, -1, -1, device='cuda', dtype=torch.int32)
    if empty:
        seq.zero_(); extend.zero_()
    table = torch.arange(batch*max(1,width), device='cuda', dtype=torch.int32).view(batch,max(1,width))
    mapping = torch.arange(batch*max(1,width)+1, device='cuda', dtype=torch.int64)*3+2**31
    if strided:
        def spaced(tensor):
            output = torch.empty(tuple(2*s for s in tensor.shape), device='cuda', dtype=tensor.dtype)
            view = output[(slice(None,None,2),)*tensor.dim()]
            view.copy_(tensor)
            return view
        seq, extend, req, table, mapping = map(spaced, (seq,extend,req,table,mapping))
    total = sum(min(s,e+window-1) for s,e in zip(seq.tolist(),extend.tolist()))
    return seq,extend,req,table,mapping,window,total


def test_exact():
    op=get_op()
    for batch,width in ((0,1),(1,1),(1,127),(1,4096),(1,65536),(8,8192),(64,2048)):
        for strided in (False,True):
            for window in (1,128,256):
                args=make_case(batch,width,window,strided)
                for actual,expected in zip(op(*args),reference(*args)):
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for batch in (0,1,8):
        args=make_case(batch,128,empty=True)
        for actual,expected in zip(op(*args),reference(*args)):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    args=make_case(1,1048576,query_len=32768)
    for actual,expected in zip(op(*args),reference(*args)):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    args=list(make_case(8,8192,strided=True))
    args[0][::3].zero_();args[1][::3].zero_()
    args[-1]=sum(min(s,e+127) for s,e in zip(args[0].tolist(),args[1].tolist()))
    for actual,expected in zip(op(*args),reference(*args)):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_stream_and_graph():
    op=get_op()
    args=make_case()
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): actual=op(*args)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): actual=op(*args)
    for _ in range(3):
        args[4].add_(3)
        args[1][0].sub_(1)
        args[1][1].add_(1)
        graph.replay()
        for out,expected in zip(actual,reference(*args)):
            torch.testing.assert_close(out,expected,rtol=0,atol=0)


def test_invalid_metadata():
    op=get_op()
    args=list(make_case())
    for index,bad in ((0,args[0].long()),(1,args[1][:2]),(2,args[2].cpu()),
                       (3,args[3].long()),(4,args[4].int()),(5,0),(6,-1)):
        wrong=args.copy();wrong[index]=bad
        with pytest.raises((RuntimeError,ValueError,TypeError)): op(*wrong)


def benchmark():
    op=get_op();rows=[]
    # Precompute the literal oracle's CPU lists so timing excludes test-only DtoH.
    for batch,width in ((1,8192),(1,65536),(8,8192),(64,2048)):
        args=make_case(batch,width)
        seq,extend,req,table,mapping,window,total=args
        lens_cpu=[min(s,e+window-1) for s,e in zip(seq.tolist(),extend.tolist())]
        spans=list(zip(req.tolist(),seq.tolist(),lens_cpu))
        def unfused():
            lens=torch.minimum(seq,extend+window-1).int();first=seq-lens
            offsets=torch.zeros(batch+1,device='cuda',dtype=torch.int32)
            offsets[1:]=lens.cumsum(0).int()
            ids=torch.cat([mapping[table[r,s-n:s].long()].int() for r,s,n in spans])
            return ids,first,lens,offsets
        samples={'torch':[],'lightop':[]}
        for run in (unfused,lambda:op(*args)):
            for _ in range(10):run()
        for repeat in range(5):
            order=(('torch',unfused),('lightop',lambda:op(*args)))
            for name,run in (order if repeat%2==0 else reversed(order)):
                torch.cuda.synchronize()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):out=run()
                end.record();torch.cuda.synchronize()
                samples[name].append(start.elapsed_time(end)*10)
        means={name:statistics.mean(v) for name,v in samples.items()}
        logical_bytes=total*16+batch*24+4
        row=dict(batch=batch,total_swa=total,mean_us=means,samples_us=samples,
                 logical_bytes=logical_bytes,speedup=means['torch']/means['lightop'],
                 effective_GB_s={name:logical_bytes/value/1000 for name,value in means.items()})
        rows.append(row);print(json.dumps(row),flush=True)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--json',type=Path);args=parser.parse_args()
    for test in (test_exact,test_stream_and_graph,test_invalid_metadata):
        test();print(test.__name__+': PASS',flush=True)
    result=dict(correctness='PASS',rows=benchmark() if args.benchmark else [])
    if args.json:args.json.write_text(json.dumps(result,indent=2))
