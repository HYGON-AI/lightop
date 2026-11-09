# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Run from the repository root with LightOp installed on a HIP system.
# python3 -m pytest -q test/ci_tests/test_per_token_quant_int8_wide.py

"""LMSlim arithmetic contract, public routing, and changed-input graph replay."""
import pytest
import torch
from lightop.quant import int8 as api
from lightop._lmslim_native.layers.gemm.int8_utils import per_token_quant_int8 as baseline
pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None,
                             reason="HIP required")
DIMS=(3072,6144,8192,12288)
def assert_same(actual,expected):
    torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0)
    torch.testing.assert_close(actual[1],expected[1],rtol=0,atol=0,equal_nan=True)
@pytest.mark.parametrize("d",DIMS)
@pytest.mark.parametrize("p",[1,6,8,16,24,32,40,48,56,64,80,2048,6144,8192])
def test_actual_shapes(p,d):
    torch.manual_seed(13)
    x=torch.randn((p,d),device="cuda",dtype=torch.bfloat16)
    x[0].zero_()
    if p>1:x[1].mul_(1e-12)
    assert_same(api.per_token_quant_int8(x),baseline(x))
@pytest.mark.parametrize("d",DIMS)
def test_native_boundary_and_graph(d):
    if api._wide_int8 is None:pytest.skip("wide native extension unavailable")
    x=torch.randn((129,d),device="cuda",dtype=torch.bfloat16)
    # Covers every finite BF16 magnitude/sign, with amax chosen by each row.
    bits=torch.arange(-32768,32768,device="cuda",dtype=torch.int32).to(torch.int16)
    vals=bits.view(torch.bfloat16)
    finite=vals[torch.isfinite(vals)]
    x.flatten()[:finite.numel()]=finite
    x[0].zero_();x[1].fill_(1e-12);x[2].fill_(1e-10)
    x[3].fill_(127);x[3,0]=254;x[3,1]=-127
    q=torch.empty_like(x,dtype=torch.int8)
    s=torch.empty((x.shape[0],1),device="cuda")
    assert_same(api.per_token_quant_int8(x),baseline(x))
    for threads in [64,128,256,512]:
        api._wide_int8(q,x,s,threads)
        assert_same((q,s),baseline(x))
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        api._wide_int8(q,x,s,64)
    public_graph=torch.cuda.CUDAGraph()
    api.per_token_quant_int8(x)
    with torch.cuda.graph(public_graph):
        public_q,public_s=api.per_token_quant_int8(x)
    for mode in ["normal","zero","tiny","outlier"]:
        x.normal_()
        if mode=="zero":x.zero_()
        elif mode=="tiny":x.mul_(1e-12)
        elif mode=="outlier":x[:,0]=10000
        graph.replay()
        assert_same((q,s),baseline(x))
        public_graph.replay()
        assert_same((public_q,public_s),baseline(x))
@pytest.mark.parametrize("d",DIMS)
def test_mantissa_pairs(d):
    if api._wide_int8 is None:pytest.skip("wide native extension unavailable")
    # All BF16 mantissas, exponent deltas 0..10, both signs; independently
    # exercises multiply-reciprocal tie cases rather than the OAI division LUT.
    m=1+torch.arange(128,device="cuda",dtype=torch.float32)/128
    v=(m[None,:]*torch.pow(2.,-torch.arange(11,device="cuda")[:,None])).flatten()
    v=torch.cat((v,-v))
    x=torch.zeros((128,d),device="cuda",dtype=torch.bfloat16)
    x[:,:len(v)]=v
    # Restrict values to each row's amax so all 128 maxima occur.
    x=torch.minimum(x.abs(),m[:,None]).copysign(x).to(torch.bfloat16)
    x[:,0]=m
    q=torch.empty_like(x,dtype=torch.int8);s=torch.empty((128,1),device="cuda")
    api._wide_int8(q,x,s,64)
    assert_same((q,s),baseline(x))
    assert_same(api.per_token_quant_int8(x),baseline(x))
def test_public_routing_and_fallback(monkeypatch):
    if api._wide_int8 is None:pytest.skip("wide native extension unavailable")
    calls=[]
    original=api._wide_int8
    def counted(q,x,s,threads):
        calls.append((tuple(x.shape),threads))
        return original(q,x,s,threads)
    monkeypatch.setattr(api,"_wide_int8",counted)
    monkeypatch.setattr(api,"_wide_int8_enabled",True)
    monkeypatch.setattr(api,"_int8_method","native")
    x=torch.randn((2048,3072),device="cuda",dtype=torch.bfloat16)
    assert_same(api.per_token_quant_int8(x),baseline(x))
    assert len(calls)==1
    for x in [torch.randn((13,257),device="cuda",dtype=torch.bfloat16),
              torch.randn((13,3072),device="cuda",dtype=torch.float16),
              torch.randn((26,3072),device="cuda",dtype=torch.bfloat16)[::2]]:
        assert_same(api.per_token_quant_int8(x),baseline(x))
    assert len(calls)==1
    monkeypatch.setattr(api,"_wide_int8_enabled",False)
    assert_same(api.per_token_quant_int8(torch.ones((8,6144),device="cuda",dtype=torch.bfloat16)),
                baseline(torch.ones((8,6144),device="cuda",dtype=torch.bfloat16)))
    assert len(calls)==1
def test_empty_invalid_and_alias():
    if api._wide_int8 is None:pytest.skip("wide native extension unavailable")
    x=torch.empty((0,3072),device="cuda",dtype=torch.bfloat16)
    q=torch.empty_like(x,dtype=torch.int8);s=torch.empty((0,1),device="cuda")
    api._wide_int8(q,x,s,64)
    x=torch.randn((8,3072),device="cuda",dtype=torch.bfloat16)
    q=torch.empty_like(x,dtype=torch.int8);s=torch.empty((8,1),device="cuda")
    with pytest.raises(RuntimeError):api._wide_int8(q,x,s,17)
    with pytest.raises(RuntimeError):api._wide_int8(q,x,x.view(torch.float32).flatten()[:8,None],64)
    with pytest.raises(RuntimeError):api._wide_int8(q[:,:-1],x,s,64)

def test_segments_selected_and_original_fallback(monkeypatch):
    monkeypatch.setattr(api,"_int8_method","segments")
    monkeypatch.setattr(api,"_wide_int8_enabled",True)
    calls=[]
    kernel=api._segments_int8
    class Spy:
        def __getitem__(self,grid):
            def run(*args,**kwargs):
                calls.append((grid,kwargs))
                return kernel[grid](*args,**kwargs)
            return run
    monkeypatch.setattr(api,"_segments_int8",Spy())
    for d in [3072,6144,12288,8192]:
        x=torch.randn((56,d),device="cuda",dtype=torch.bfloat16)
        assert_same(api.per_token_quant_int8(x),baseline(x))
    assert len(calls)==3
    monkeypatch.setattr(api,"_int8_method","triton")
    x=torch.randn((56,6144),device="cuda",dtype=torch.bfloat16)
    assert_same(api.per_token_quant_int8(x),baseline(x))
    assert len(calls)==3
