# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm benchmark and irregular-shape correctness tests."""

import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from lightop.norm import RMSNormTorch
from torch.autograd import Function
from torch import Tensor

from lightop.norm import RMSNorm
from lightop import op

DEVICE = "cuda"
EPS = 1e-5

# from lightop.norm import rmsnorm_forward, rmsnorm_backward

# class RMSNormFunction(Function):
#     @staticmethod
#     def forward(ctx, x, weight, out, eps, train) -> Tensor:
#         ret,rstd = rmsnorm_forward(x,weight,out, eps, train)
#         ctx.save_for_backward(x,rstd,weight)
#         return ret

#     @staticmethod
#     def backward(ctx, grad):
#         x,rstd,gamma = ctx.saved_tensors
#         dx,dgamma = rmsnorm_backward(grad, x,rstd,gamma)
#         return dx,dgamma,None ,None, None


# class RMSNorm(torch.nn.Module):

#     def __init__(self, normalized_shape:int,eps: float = 1e-5,device=None, dtype=None ):
#         super(RMSNorm, self).__init__()
#         self.eps = eps
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
#         torch.nn.init.ones_(self.weight)

#     def forward(self, x,out=None):
#         return RMSNormFunction.apply(x, self.weight,out,self.eps, self.training)

#     def extra_repr(self):
#         return f'eps={round(self.eps,5):0.5f}'


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):

    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def get_data(batchsize=128, seqs=50, cols=128,use_half=False):

    if use_half:
        x = torch.rand((batchsize, seqs, cols)).cuda().half()
        grad = torch.rand((batchsize, seqs, cols)).cuda().half()
    else:
        x = torch.rand((batchsize, seqs, cols)).cuda()
        grad = torch.rand((batchsize, seqs, cols)).cuda()

    x.requires_grad_(True)
    x1=x.clone().detach().requires_grad_()
    x2=x.clone().detach().requires_grad_()
    return x1,x2,grad

def get_model(cols=128,use_half=False):

    if use_half:
        fuse_rmsnorm=RMSNorm(cols).cuda().half()
        torch_rmsnorm=RMSNormTorch(cols).cuda().half()
    else:
        fuse_rmsnorm=RMSNorm(cols).cuda()
        torch_rmsnorm=RMSNormTorch(cols).cuda()
        
    return torch_rmsnorm,fuse_rmsnorm


def test_profile(batchsize, seqs, cols,repeat_num,use_half=False):
    
    x1,x2,grad=get_data(batchsize, seqs, cols,use_half=use_half)
    f1,f2=get_model(cols=cols,use_half=use_half)

    with torch.autograd.profiler.profile(enabled=True, use_device="cuda", record_shapes=False,use_kineto=True,
                                     profile_memory=False) as prof1:
        for i in range(repeat_num):
            y1 = f1(x1)
            y1.backward(grad)
            torch.cuda.synchronize()
    prof1.export_chrome_trace('./profile_torch_rmsnorm.json')
    
    with torch.autograd.profiler.profile(enabled=True, use_device="cuda", record_shapes=False,use_kineto=True,
                                     profile_memory=False) as prof2:
        for i in range(repeat_num):
            y2 = f2(x2)
            y2.backward(grad)
            torch.cuda.synchronize()
    prof2.export_chrome_trace('./profile_fuse_rmsnorm.json')  

def test_speed(batchsize, seqs, cols,repeat_num,use_half=False):
    if use_half:
        print(f"---------------fp16, batchsize={batchsize},seqs={seqs}, cols={cols}---------------------")
    else:
        print(f"---------------fp32, batchsize={batchsize},seqs={seqs}, cols={cols}---------------------")
    x1,x2,grad=get_data(batchsize, seqs, cols,use_half=use_half)
    f1,f2=get_model(cols=cols,use_half=use_half)

    y1 = f1(x1)
    y1.backward(grad)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(repeat_num):
        x1.grad = None
        f1.weight.grad=None
        y1 = f1(x1)
        y1.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()
    fc1_espl = end_time - start_time
    print('torch_rmsnorm', 'averange time is', fc1_espl *1000/ repeat_num, 'ms')
    # y2=torch.empty_like(x2)
    y2 = f2(x2)
    y2.backward(grad)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(repeat_num):
        x2.grad = None
        f2.weight.grad=None
        y2 = f2(x2)
        y2.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()    
    fc2_espl = end_time - start_time
    print('fuse_rmsnorm', 'averange time is', fc2_espl *1000/ repeat_num, 'ms')

    #误差计算
    if use_half:
        print(allclose(y1,y2,rtol=2e-02, atol=2e-02))
        print(allclose(x1.grad,x2.grad,rtol=2e-02, atol=2e-02))
        print(allclose(f1.weight.grad,f2.weight.grad,rtol=2e-02, atol=2e-02))
    else:
        print(allclose(y1,y2,rtol=1e-05, atol=1e-05))
        print(allclose(x1.grad,x2.grad,rtol=1e-05, atol=1e-05))
        print(allclose(f1.weight.grad,f2.weight.grad,rtol=1e-04, atol=1e-04))

    

def run_rmsnorm_benchmark_suite():
    # mid + fused (>=16384) sizes aligned with micro_perf dim coverage
    shape_cases = [
        (1, 2048, 128),
        (1, 2048, 256),
        (1, 2048, 512),
        (1, 2048, 1024),
        (1, 2048, 2048),
        (1, 2048, 4096),
        (1, 2048, 8192),
        (1, 2048, 16384),
        (1, 1024, 32768),
        (1, 512, 65536),
        (1, 256, 131072),
        (1024, 1, 16384),
        (1024, 1, 32768),
        # irregular: odd / non-pow2 / non-1024 batch
        (7, 1, 100),
        (17, 1, 1000),
        (33, 1, 2047),
        (65, 1, 5000),
        (7, 1, 8191),
        (17, 1, 10000),
        (3, 1, 16385),
        (513, 1, 32767),
    ]

    for use_half in [False, True]:
        for batchsize, seqs, cols in shape_cases:
            repeat_num = 200 if cols <= 8192 else 20
            test_speed(batchsize, seqs, cols, repeat_num=repeat_num, use_half=use_half)
    # test_profile(1, 2048, 16384, repeat_num=10, use_half=True)


ODD_DIMS = (
    1,
    3,
    7,
    15,
    31,
    63,
    127,
    255,
    511,
    1023,
    2047,
    4095,
    8191,
    16383,
    16385,
    32767,
)

# Non-power-of-2 / not vec-aligned (4 or 8)
NON_POW2_DIMS = (
    6,
    12,
    48,
    100,
    384,
    768,
    1000,
    1536,
    3000,
    5000,
    6000,
    8208,   # >8192 and %16==0 — LN mid welford stride path
    10000,
    12000,
    20000,
    25000,
    48000,
)

# Batches / rows not used by xpu-perf (which fixes batch=1024)
BATCHES = (1, 3, 7, 17, 33, 65, 128, 256, 513, 1000, 1025)

RMS_IRREGULAR_DTYPES = (
    (torch.float32, 1e-4, 1e-4),
    (torch.float16, 2e-2, 2e-2),
    (torch.bfloat16, 2e-2, 2e-2),
)

# Representative (batch, dim) pairs mixing irregular batch and dim
MIXED_CASES = (
    (1, 16385),
    (3, 16384),
    (7, 8191),
    (17, 10000),
    (33, 5000),
    (65, 4095),
    (128, 3000),
    (256, 1536),
    (513, 1023),
    (1000, 768),
    (1025, 511),
    (7, 32767),
    (3, 48000),
    (17, 20000),
    (1025, 16383),
    (7, 8208),
    (33, 63),
    (65, 100),
    (257, 2047),
    (999, 999),
)


def _rms_max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()

def _check_rmsnorm(batch: int, dim: int, dtype: torch.dtype, rtol: float, atol: float):
    torch.manual_seed(0)
    x = torch.randn(batch, dim, device=DEVICE, dtype=dtype)
    w = torch.randn(dim, device=DEVICE, dtype=dtype)
    out = torch.empty_like(x)
    xf = x.float()
    y_ref = (
        xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + EPS) * w.float()
    ).to(dtype)
    op.rms_norm_opt(out, x, w, EPS)
    return torch.allclose(out, y_ref, rtol=rtol, atol=atol), _rms_max_abs(out, y_ref)
def _iter_rmsnorm_cases():
    # Fixed batch=7 over odd / non-pow2 dims (batch itself irregular)
    for dim in ODD_DIMS + NON_POW2_DIMS:
        yield 7, dim, "odd/non_pow2"
    # Fixed dim=1000 (non-pow2) over irregular batches
    for batch in BATCHES:
        yield batch, 1000, "batch_sweep"
    # Fixed dim=16385 (odd fused+) over irregular batches
    for batch in (1, 3, 7, 17, 33, 65, 128, 256, 513):
        yield batch, 16385, "odd_fused_batch"
    # Hand-picked mixed cases
    for batch, dim in MIXED_CASES:
        yield batch, dim, "mixed"


def run_irregular_rmsnorm_suite():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device required")

    failures = []
    seen = set()
    for batch, dim, tag in _iter_rmsnorm_cases():
        if (batch, dim) in seen:
            continue
        seen.add((batch, dim))
        for dtype, rtol, atol in RMS_IRREGULAR_DTYPES:
            ok, max_abs = _check_rmsnorm(batch, dim, dtype, rtol, atol)
            print(
                f"[{'PASS' if ok else 'FAIL'}] rms_norm {tag} "
                f"dtype={dtype} batch={batch} dim={dim} max_abs={max_abs:.4e}"
            )
            if not ok:
                failures.append((tag, dtype, batch, dim, max_abs))

    if failures:
        raise AssertionError(f"irregular RMSNorm failures: {failures}")
    print("All irregular RMSNorm checks passed.")


def main():
    run_rmsnorm_benchmark_suite()
    run_irregular_rmsnorm_suite()


if __name__ == "__main__":
    main()
