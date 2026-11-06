# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Numerical contract and graph replay tests for OAI INT8 quantization."""
import pytest
import torch

from lightop.activation.swiglu_oai_int8_quant import (
    swiglu_oai_int8_quant_contiguous,
    swiglu_oai_int8_quant_masked,
    swiglu_oai_int8_quant_masked_flat,
)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None, reason="HIP required"
)

def reference(x, alpha=1.702, beta=1.0, limit=7.0):
    g,u=x.float().chunk(2,dim=-1)
    if limit is not None:
        g=g.clamp(max=limit)
        u=u.clamp(-limit,limit)
    y=(g*torch.sigmoid(alpha*g)*(u+beta)).to(x.dtype).float()
    s=y.abs().amax(dim=-1,keepdim=True).clamp_min(1e-10)/127
    return (y/s).round().clamp(-127,127).to(torch.int8),s

def assert_numerics(actual,expected,valid=None):
    q,s=actual;rq,rs=expected
    if valid is not None:
        q,s,rq,rs=q[valid],s[valid],rq[valid],rs[valid]
    torch.testing.assert_close(s,rs,rtol=1e-6,atol=1e-12)
    diff=(q.to(torch.int16)-rq.to(torch.int16)).abs()
    # FP32 sigmoid implementations can straddle a BF16 rounding boundary.
    if diff.numel():
        assert diff.max().item()<=1
        assert (diff!=0).sum().item()/diff.numel()<=1e-4

@pytest.mark.parametrize("shape",[(0,6144),(7,2),(13,66),(63,2048),(512,6144),(2048,6144)])
def test_contiguous(shape):
    torch.manual_seed(17)
    x=torch.randn(shape,device="cuda",dtype=torch.bfloat16)
    assert_numerics(swiglu_oai_int8_quant_contiguous(x),reference(x))

@pytest.mark.parametrize("limit",[None,7.0])
def test_zero_and_tiny_rows(limit):
    x=torch.zeros((8,6144),device="cuda",dtype=torch.bfloat16)
    x[1::2]=1e-12
    assert_numerics(swiglu_oai_int8_quant_contiguous(x,limit=limit),reference(x,limit=limit))

def test_masked_padding_and_changed_graph_inputs():
    torch.manual_seed(19)
    x=torch.randn((16,64,6144),device="cuda",dtype=torch.bfloat16)
    counts=torch.tensor([0,1,3,7,11,13,17,23,29,31,37,41,43,47,61,64],
                        device="cuda",dtype=torch.int32)
    valid=torch.arange(64,device="cuda")[None,:]<counts[:,None]
    x[~valid]=float("nan")
    q=torch.full((16,64,3072),99,device="cuda",dtype=torch.int8)
    s=torch.full((16,64,1),-9.,device="cuda")
    def call():
        return swiglu_oai_int8_quant_masked_flat(x,counts,output=q,scales=s)
    assert_numerics(call(),reference(x),valid)
    assert (q[~valid]==99).all() and (s[~valid]==-9).all()
    padded=swiglu_oai_int8_quant_masked(x,counts)
    assert (padded[0][~valid]==0).all() and (padded[1][~valid]==1).all()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):call()
    for count_mode in ["empty","full","skew"]:
        x.normal_()
        if count_mode=="empty":counts.zero_()
        elif count_mode=="full":counts.fill_(64)
        else:counts.copy_(torch.arange(16,device="cuda",dtype=torch.int32)*4)
        valid=torch.arange(64,device="cuda")[None,:]<counts[:,None]
        x[~valid]=float("nan");q.fill_(99);s.fill_(-9.)
        graph.replay()
        assert_numerics((q,s),reference(x),valid)
        assert (q[~valid]==99).all() and (s[~valid]==-9).all()

def test_invalid_output_alias_and_shape():
    x=torch.randn((4,6144),device="cuda",dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        swiglu_oai_int8_quant_contiguous(x,output=torch.empty((4,3071),device="cuda",dtype=torch.int8))
    # Scale aliases input storage even though dimensions and dtype are otherwise valid.
    scale=x.view(torch.float32).flatten()[:4].view(4,1)
    with pytest.raises(RuntimeError):
        swiglu_oai_int8_quant_contiguous(x,scales=scale)
