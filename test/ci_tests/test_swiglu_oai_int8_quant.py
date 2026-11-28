# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Run from the repository root with LightOp installed on a HIP system.
# python3 -m pytest -q test/ci_tests/test_swiglu_oai_int8_quant.py

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


def vllm_reference(x, alpha=1.702, beta=1.0, limit=7.0):
    gate, up = x.float().chunk(2, dim=-1)
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    value = (gate * torch.sigmoid(alpha * gate) * (up + beta)).bfloat16().float()
    maximum = value.abs().amax(dim=-1, keepdim=True)
    inverse = 127.0 / torch.where(maximum == 0, 1.0, maximum)
    output = (value * inverse).round().clamp(-127, 127).to(torch.int8)
    return output, maximum / 127.0


@pytest.mark.parametrize("rows", [0, 1, 127, 128, 129])
@pytest.mark.parametrize("limit", [None, 7.0])
def test_vllm_contiguous(rows, limit):
    torch.manual_seed(57)
    x = (torch.randn((rows, 6144), device="cuda") * 2).bfloat16()
    actual = swiglu_oai_int8_quant_contiguous(x, limit=limit, vllm_compatible=True)
    expected = vllm_reference(x, limit=limit)
    assert_numerics(actual, expected)
    assert actual[0].is_contiguous() and actual[1].is_contiguous()


def test_vllm_zero_tiny_and_ties():
    x = torch.zeros((3, 6144), dtype=torch.bfloat16, device="cuda")
    x[1:, :3072] = 2
    x[1, 3072:3079] = torch.tensor(
        [127, .5, 1.5, 2.5, -.5, -1.5, -2.5], device="cuda", dtype=torch.bfloat16,
    )
    x[2, 3072] = 1e-12
    q, scale = swiglu_oai_int8_quant_contiguous(
        x, alpha=0.0, beta=0.0, limit=None, vllm_compatible=True,
    )
    ref_q, ref_scale = vllm_reference(x, alpha=0.0, beta=0.0, limit=None)
    torch.testing.assert_close(q, ref_q, atol=0, rtol=0)
    torch.testing.assert_close(scale, ref_scale, atol=0, rtol=0)
    assert scale[0].item() == 0 and q[1, :7].tolist() == [127, 0, 2, 2, 0, -2, -2]


def test_vllm_int8_packing_preserves_fp32_ties():
    x = torch.zeros((2, 6144), device="cuda", dtype=torch.bfloat16)
    # gate * sigmoid(0) = 1, so the BF16 up values reach the quantizer exactly.
    x[:, :3072] = 2
    maxima = (1.0078125, 1.1875)
    for row, maximum in enumerate(maxima):
        half = maximum / 2
        x[row, 3072:3080] = torch.tensor(
            [maximum, half, -half, -maximum, 0, -half, half, maximum],
            device="cuda", dtype=torch.bfloat16,
        )
    q, scale = swiglu_oai_int8_quant_contiguous(
        x, alpha=0.0, beta=0.0, limit=None, vllm_compatible=True,
    )
    expected = torch.zeros_like(q)
    # Fusing the bias with the multiply changes 64 to 63 in row 0.
    # Rounding the exact real ratio changes 63 to 64 in row 1.
    expected[:, :8] = torch.tensor(
        [[127, 64, -64, -127, 0, -64, 64, 127],
         [127, 63, -63, -127, 0, -63, 63, 127]],
        dtype=torch.int8, device="cuda",
    )
    expected_scale = torch.tensor(
        [[value / 127] for value in maxima], device="cuda", dtype=torch.float32,
    )
    assert torch.equal(q, expected)
    assert torch.equal(scale, expected_scale)


def test_vllm_changed_graph_inputs_and_shape_guard():
    torch.manual_seed(59)
    x = torch.randn((128, 6144), device="cuda", dtype=torch.bfloat16)
    q = torch.empty((128, 3072), device="cuda", dtype=torch.int8)
    s = torch.empty((128, 1), device="cuda", dtype=torch.float32)

    def call():
        return swiglu_oai_int8_quant_contiguous(
            x, output=q, scales=s, vllm_compatible=True,
        )

    actual_q, actual_s = call()
    assert actual_q is q and actual_s is s
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    x.normal_()
    x[3].zero_()
    graph.replay()
    assert_numerics((q, s), vllm_reference(x))
    assert s[3].item() == 0 and q[3].count_nonzero().item() == 0
    with pytest.raises(RuntimeError, match="6144"):
        swiglu_oai_int8_quant_contiguous(x[:, :6142].contiguous(), vllm_compatible=True)
    with pytest.raises(RuntimeError, match="must not share storage"):
        swiglu_oai_int8_quant_contiguous(
            x, scales=x.view(torch.float32).flatten()[:128].view(128, 1), vllm_compatible=True,
        )


@pytest.mark.parametrize("tokens,width", [(1, 3072), (7, 3072), (128, 6144)])
def test_vllm_oai_transposed_tiles(tokens, width):
    from lightop.activation import (
        swiglu_oai_int8_quant_contiguous,
        swiglu_oai_int8_quant_transposed,
    )
    if not torch.cuda.is_available():
        pytest.skip("HIP required")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if not arch.startswith("gfx938"):
        pytest.skip("the measured MiniMax path uses gfx938")
    torch.manual_seed(871)
    padded = torch.randn((2 * width, tokens + 8), device="cuda", dtype=torch.bfloat16)
    column = padded[:, :tokens].t()
    column[0].zero_()
    reference = swiglu_oai_int8_quant_contiguous(column.contiguous(), vllm_compatible=True)
    for config in range(4):
        output = swiglu_oai_int8_quant_transposed(column, config=config)
        assert all(torch.equal(a, b) for a, b in zip(output, reference, strict=True))
