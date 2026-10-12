# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for FP32 residual sums in fused add + RMSNorm."""

import unittest

import torch

from lightop import op


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm device required")
class TestFusedAddRMSNormFP32(unittest.TestCase):
    def check_case(self, rows, hidden, dtype, *, gemma=False, limit=-1.0,
                   offset=0, cancellation=False):
        torch.manual_seed(1517 + hidden)
        # An offset of one element exercises the unaligned fallback.
        x = torch.randn(rows * hidden + offset, device="cuda", dtype=dtype)[offset:]
        x = x.view(rows, hidden)
        r = torch.randn(rows * hidden + offset, device="cuda", dtype=dtype)[offset:]
        r = r.view(rows, hidden)
        if cancellation:
            r.copy_((-x.float() + 0.003).to(dtype))
        w = torch.randn(hidden + offset, device="cuda", dtype=dtype)[offset:]
        x0, r0, w0 = x.clone(), r.clone(), w.clone()
        summed = x0.float() + r0.float()
        reference = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-5)
        reference = reference * (w0.float() + (1.0 if gemma else 0.0))
        if limit >= 0:
            reference = reference.clamp(-limit, limit)
        expected = reference.to(dtype)
        ptrs = (x.data_ptr(), r.data_ptr())
        if gemma:
            op.gemma_fused_add_rmsnorm(x, r, w, 1e-5)
        else:
            op.fused_add_rms_norm_opt(x, r, w, 1e-5, limit)
        self.assertEqual((x.data_ptr(), r.data_ptr()), ptrs)
        torch.testing.assert_close(w, w0, rtol=0, atol=0)
        torch.testing.assert_close(r, summed.to(dtype), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(x).all())
        error = ((x.float() - expected.float()).norm()
                 / expected.float().norm().clamp_min(1e-30)).item()
        # Allow final-cast boundary differences, but reject premature sum rounding.
        tolerance = 1e-4 if hidden in (4096, 8192) else 5e-4
        self.assertLess(error, tolerance, (rows, hidden, dtype, gemma, limit, error))
        if limit >= 0:
            self.assertLessEqual(x.float().abs().max().item(), limit)

    def test_dispatch_shapes(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for hidden in (64, 128, 256, 512, 768, 1024, 1536, 2048,
                           4096, 8192, 16384, 65, 520, 16392):
                with self.subTest(dtype=dtype, hidden=hidden):
                    self.check_case(17, hidden, dtype)

    def test_clamp_and_gemma(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for hidden in (64, 512, 4096, 8192, 65, 520, 16392):
                for mode in ("clamp", "gemma"):
                    with self.subTest(dtype=dtype, hidden=hidden, mode=mode):
                        self.check_case(17, hidden, dtype,
                                        gemma=mode == "gemma",
                                        limit=0.5 if mode == "clamp" else -1.0)

    def test_unaligned_cancellation_and_grid_reuse(self):
        for dtype in (torch.bfloat16, torch.float16):
            for hidden in (65, 512, 4096, 8192):
                with self.subTest(dtype=dtype, hidden=hidden, unaligned=True):
                    self.check_case(17, hidden, dtype, offset=1)
                with self.subTest(dtype=dtype, hidden=hidden, cancellation=True):
                    self.check_case(17, hidden, dtype, cancellation=True)
            # Cover both the general grid-reuse path and specialized tiled paths.
            for _ in range(3):
                self.check_case(288, 2048, dtype)
                self.check_case(288, 4096, dtype)
                self.check_case(288, 8192, dtype)

    def test_graph_replay(self):
        for hidden in (4096, 8192):
            dtype = torch.bfloat16
            torch.manual_seed(42)
            x = torch.randn(17, hidden, device="cuda", dtype=dtype)
            r = torch.randn_like(x)
            w = torch.randn(hidden, device="cuda", dtype=dtype)
            x0, r0 = x.clone(), r.clone()
            op.fused_add_rms_norm_opt(x, r, w, 1e-5, -1.0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                x.copy_(x0)
                r.copy_(r0)
                op.fused_add_rms_norm_opt(x, r, w, 1e-5, -1.0)
            for _ in range(3):
                x0.normal_()
                r0.normal_()
                graph.replay()
                summed = x0.float() + r0.float()
                expected = (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True)
                                                + 1e-5) * w.float()).to(dtype)
                error = ((x.float() - expected.float()).norm()
                         / expected.float().norm()).item()
                self.assertLess(error, 1e-4)
                torch.testing.assert_close(r, summed.to(dtype), rtol=0, atol=0)

    def test_empty(self):
        for hidden in (65, 512, 4096, 8192):
            x = torch.empty(0, hidden, device="cuda", dtype=torch.bfloat16)
            r = torch.empty_like(x)
            w = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
            op.fused_add_rms_norm_opt(x, r, w, 1e-5, -1.0)
            self.assertEqual(x.shape, (0, hidden))
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
