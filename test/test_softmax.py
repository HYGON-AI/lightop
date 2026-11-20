"""Correctness and performance tests for Softmax.

The test includes irregular-shape coverage that is not represented by standard
benchmark shapes.
"""

#  -*-  coding: utf-8   -*-
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import time
import copy
from lightop.attention import FuseLogSoftmax, FuseSoftmax
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

device = torch.device('cuda')
torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)


class SoftmaxTester:
    def __init__(self, softmax_type='default', dim=-1, test_half=False, half_to_float=False):
        torch.backends.cudnn.enabled = False
        self.half_to_float = half_to_float
        self.test_half = test_half
        self.softmax_type = softmax_type
        self.input_dtype = torch.float16 if self.test_half else torch.float32

        if softmax_type == 'default' or softmax_type == 'softmax':
            self.torch_softmax = nn.Softmax(dim=dim).cuda()
            self.fuse_softmax = FuseSoftmax(dim=dim, half_to_float=half_to_float).cuda()
        elif softmax_type == 'log_softmax':
            self.torch_softmax = nn.LogSoftmax(dim=dim).cuda()
            self.fuse_softmax = FuseLogSoftmax(dim=dim, half_to_float=half_to_float).cuda()
        else:
            raise NotImplementedError('not support type {}'.format(softmax_type))

    def allclose(self, a, b, equal_nan=False):
        if self.softmax_type in ['default', 'softmax']:
            if self.test_half:
                rtol, atol = 2e-02, 2e-02
            else:
                rtol, atol = 1e-05, 1e-05
        else:
            if self.test_half:
                rtol, atol = 2e-02, 2e-02
            else:
                rtol, atol = 2e-04, 2e-04

        return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def check_accuracy(self, x_size=(4, 60, 60)):
        # 设置输入数据
        x0 = torch.rand(x_size, dtype=self.input_dtype).cuda()
        x = Variable(x0, requires_grad=True)
        x0_fuse = copy.deepcopy(x0).cuda()
        x_fuse = Variable(x0_fuse, requires_grad=True)
        grad = torch.rand(x0.shape, dtype=torch.float32).cuda() * 1e2

        self.fuse_softmax.train()
        self.torch_softmax.train()

        # 检查准确性
        fuse_out = self.fuse_softmax(x_fuse)
        fuse_out.backward(grad)
        torch.cuda.synchronize()

        out = self.torch_softmax(x)
        out.backward(grad)
        torch.cuda.synchronize()
        out = out.to(fuse_out.dtype)

        # print('out: ', out[:2, :2, :5])
        # print('fuse_out: ', fuse_out[:2, :2, :5])
        if self.allclose(out, fuse_out):
            print("forward Passed")
        else:
            print("forward Failed")

        # print('x.grad: ', x.grad[:2, :2, :5])
        # print('x_fuse.grad: ', x_fuse.grad[:2, :2, :5])
        if self.allclose(x.grad, x_fuse.grad):
            print("backward Passed")
        else:
            print("backward Failed")
            diff = torch.abs(x.grad - x_fuse.grad)
            print('diff', diff.shape)
            # print('x.grad: ', x.grad[:3, :3, :5])
            # print('x_fuse.grad: ', x_fuse.grad[:3, :3, :5])
            print(torch.argmax(diff))
            print('diff max={}, mean={}'.format(diff.max(), diff.mean()))

    def test_single_speed(self, softmax_tool, x_size=(4, 60, 60),
                          forward_only=False, repeat_num=1000):
        # 设置输入数据
        x0 = torch.rand(x_size, dtype=self.input_dtype).cuda()
        x = Variable(x0, requires_grad=True)
        grad = torch.rand(x0.shape, dtype=torch.float32).cuda()

        if forward_only:
            softmax_tool.eval()
        else:
            softmax_tool.train()

        # warm up
        for i in range(10):
            out = softmax_tool(x)
            if not forward_only:
                out.backward(grad)
        torch.cuda.synchronize()

        start_time = time.time()
        for i in range(repeat_num):
            out = softmax_tool(x)
            if not forward_only:
                x.grad = None
                out.backward(grad)
        torch.cuda.synchronize()

        end_time = time.time()
        fc1_espl = end_time - start_time

        return fc1_espl * 1000.0 / repeat_num

    def test_single_speed_v2(self, softmax_tool, x_size=(4, 60, 60), repeat_num=1000):
        # 设置输入数据
        x0 = torch.rand(x_size, dtype=self.input_dtype).cuda()
        x = Variable(x0, requires_grad=True)
        grad = torch.rand(x0.shape, dtype=torch.float32).cuda()

        softmax_tool.train()

        # warm up
        for i in range(10):
            out = softmax_tool(x)
            out.backward(grad)
            torch.cuda.synchronize()

        fwd_time = 0.0
        bwd_time = 0.0
        start_time = time.time()
        for i in range(repeat_num):
            out = softmax_tool(x)
            torch.cuda.synchronize()
            fwd_time += time.time() - start_time

            x.grad = None
            mid_time = time.time()
            out.backward(grad)
            torch.cuda.synchronize()
            bwd_time += time.time() - mid_time
            start_time = time.time()

        return fwd_time * 1000.0 / repeat_num, bwd_time * 1000.0 / repeat_num

    def test_speed(self, x_size=(4, 60, 60), forward_only=True, split_fwd_bwd=True, repeat_num=500):
        """测试速度

        Args:
            x_size (tuple, optional): 输入的tensor大小. Defaults to (4, 60, 60).
            forward_only (bool, optional): 是否运行推理模式，只测试前向. Defaults to True.
            split_fwd_bwd (bool, optional): 测试前向反向时是否分别记录前向和反向的耗时. Defaults to 1000.
            repeat_num (int, optional): 测试重复的次数. Defaults to 1000.

        Returns:
            _type_: 耗时统计
        """
        if forward_only:
            torch_fwd_avg_time = self.test_single_speed(self.torch_softmax, x_size, True, repeat_num)
            fuse_fwd_avg_time = self.test_single_speed(self.fuse_softmax, x_size, True, repeat_num)

            fwd_acc_rate = (torch_fwd_avg_time - fuse_fwd_avg_time) / torch_fwd_avg_time * 100.0
            print('torch fwd averange time={:.4f} ms\nfuse fwd averange time={:.4f} ms, accelerate={:.2f}%'.format(
                torch_fwd_avg_time, fuse_fwd_avg_time, fwd_acc_rate
            ))
            print('=' * 30, '\n')
            return torch_fwd_avg_time, fuse_fwd_avg_time
        else:
            if split_fwd_bwd:
                torch_fwd_avg_time, torch_bwd_avg_time = self.test_single_speed_v2(self.torch_softmax, x_size, repeat_num)
                torch_avg_time = torch_fwd_avg_time + torch_bwd_avg_time
                fuse_fwd_avg_time, fuse_bwd_avg_time = self.test_single_speed_v2(self.fuse_softmax, x_size, repeat_num)
                fuse_avg_time = fuse_fwd_avg_time + fuse_bwd_avg_time

                fwd_acc_rate = (torch_fwd_avg_time - fuse_fwd_avg_time) / torch_fwd_avg_time * 100.0
                bwd_acc_rate = (torch_bwd_avg_time - fuse_bwd_avg_time) / torch_bwd_avg_time * 100.0
                fwd_bwd_acc_rate = (torch_avg_time - fuse_avg_time) / torch_avg_time * 100.0
                print('torch time: fwd={:.4f} ms, bwd ={:.4f} ms, fwd_bwd ={:.4f} ms\n'.
                      format(torch_fwd_avg_time, torch_bwd_avg_time, torch_avg_time),
                      'fuse time: fwd={:.4f} ms, bwd ={:.4f} ms, fwd_bwd ={:.4f} ms\n'.
                      format(fuse_fwd_avg_time, fuse_bwd_avg_time, fuse_avg_time),
                      "fwd accerate={:.2f}%, bwd accerate={:.2f}%, fwd_bwd accerate={:.2f}%".
                      format(fwd_acc_rate, bwd_acc_rate, fwd_bwd_acc_rate)
                      )
                print('=' * 30, '\n')
                return fwd_acc_rate, bwd_acc_rate
            else:
                torch_avg_time = self.test_single_speed(self.torch_softmax, x_size, False, repeat_num)
                fuse_avg_time = self.test_single_speed(self.fuse_softmax, x_size, False, repeat_num)
                fwd_bwd_acc_rate = (torch_avg_time - fuse_avg_time) / torch_avg_time * 100.0
                print('torch averange time={:.4f} ms\nfuse averange time={:.4f} ms, accelerate={:.2f}%'.format(
                    torch_avg_time, fuse_avg_time, fwd_bwd_acc_rate
                ))
                print('=' * 30, '\n')
                return torch_avg_time, fuse_avg_time


def _softmax_last_dim(x_size):
    return x_size[-1]


def test_lastdim_softmax():
    # mid path + fused (>=16384) + unaligned / micro_perf-like shapes
    x_size_lists = [
        (4, 256, 1024),
        (4, 256, 2048),
        (4, 256, 2049),
        (4, 256, 3966),
        (4, 256, 4096),
        (4, 256, 5015),
        (4, 256, 8192),
        (4, 256, 16384),
        (4, 256, 32768),
        (4, 128, 65536),
        (4, 64, 131072),
        (1024, 8192),
        (1024, 16384),
        (1024, 32768),
        (4, 16, 1025, 1025),
        (4, 256, 16385),  # fused threshold + 1 (unaligned)
        # irregular: odd / non-pow2 / non-xpu-perf batches
        (7, 3),
        (7, 63),
        (17, 100),
        (33, 1000),
        (65, 2047),
        (513, 8191),
        (3, 10000),
        (7, 16383),
    ]

    half_to_float = False

    for test_half in [False, True]:
        for forward_only in [False, True]:
            for softmax_type in ['softmax', 'log_softmax']:
                softmax_tester = SoftmaxTester(softmax_type=softmax_type, dim=-1,
                                               test_half=test_half, half_to_float=half_to_float)
                for x_size in x_size_lists:
                    print('last dim test_half={}, forward_only={}, softmax_type={}, x_size={} '.
                          format(test_half, forward_only, softmax_type, x_size))
                    softmax_tester.check_accuracy(x_size)
                    # large fused dims: accuracy only (speed is covered by micro_perf)
                    if _softmax_last_dim(x_size) <= 8192:
                        softmax_tester.test_speed(x_size, forward_only=forward_only, split_fwd_bwd=True)


def test_spatial_softmax():
    x_size_lists = [
        (4, 256, 1024),
        (4, 256, 2048),
        (4, 256, 2049),
        (4, 256, 3966),
        (4, 256, 4096),
        (4, 256, 5015),
        (4, 256, 8192),
        (4, 256, 16384),
        (4, 256, 32768),
        (4, 16, 1025, 1025),
    ]

    half_to_float = False
    test_half = False

    for forward_only in [False, True]:
        for softmax_type in ['softmax', 'log_softmax']:
            softmax_tester = SoftmaxTester(softmax_type=softmax_type, dim=1,
                                           test_half=test_half, half_to_float=half_to_float)
            for x_size in x_size_lists:
                print('spatial dim test_half={}, forward_only={}, softmax_type={}, x_size={} '.
                      format(test_half, forward_only, softmax_type, x_size))
                softmax_tester.check_accuracy(x_size)
                if _softmax_last_dim(x_size) <= 8192:
                    softmax_tester.test_speed(x_size, forward_only=forward_only, split_fwd_bwd=True)



def run_softmax_benchmark_suite():
    test_lastdim_softmax()
    # test_spatial_softmax()


# Irregular-shape Softmax correctness coverage.
"""Accuracy tests for Softmax shapes not covered by xpu-perf micro_perf.

xpu-perf vector_norm workloads use batch=1024 and power-of-2 dims only.
This suite stresses:
  - odd last dims
  - non-power-of-2 dims (incl. not divisible by vec width 4/8)
  - varied batch / token counts (incl. odd batches)
"""

from lightop import op

DEVICE = "cuda"

# Odd dims: mid path + around fused threshold (>=16384)
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
    8208,
    10000,
    12000,
    20000,
    25000,
    48000,
)

# Batches / rows not used by xpu-perf (which fixes batch=1024)
BATCHES = (1, 3, 7, 17, 33, 65, 128, 256, 513, 1000, 1025)

DTYPES = (
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


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _check_softmax(batch: int, dim: int, dtype: torch.dtype, rtol: float, atol: float):
    torch.manual_seed(0)
    x = torch.randn(batch, dim, device=DEVICE, dtype=dtype)
    y_ref = torch.softmax(x.float(), dim=-1).to(dtype)
    y = op.softmax_forward_autograd(x, -1, False, False)
    close = torch.allclose(y, y_ref, rtol=rtol, atol=atol)
    row_sum = y.float().sum(dim=-1)
    sum_ok = torch.allclose(
        row_sum, torch.ones_like(row_sum), rtol=1e-3, atol=1e-3
    )
    return close and sum_ok, _max_abs(y, y_ref)


def _iter_irregular_cases():
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


def run_irregular_softmax_suite():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device required")

    failures = []
    for batch, dim, tag in _iter_irregular_cases():
        for dtype, rtol, atol in DTYPES:
            ok, max_abs = _check_softmax(batch, dim, dtype, rtol, atol)
            print(
                f"[{'PASS' if ok else 'FAIL'}] softmax {tag} "
                f"dtype={dtype} batch={batch} dim={dim} max_abs={max_abs:.4e}"
            )
            if not ok:
                failures.append(("softmax", tag, dtype, batch, dim, max_abs))

    if failures:
        raise AssertionError(f"irregular Softmax failures: {failures}")
    print("All irregular Softmax checks passed.")


def main():
    run_softmax_benchmark_suite()
    run_irregular_softmax_suite()


if __name__ == "__main__":
    main()
