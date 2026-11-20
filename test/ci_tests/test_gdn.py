# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Shape contracts for the GDN operator wrappers."""

from __future__ import annotations

import unittest

import torch

from lightop import gdn


class GdnShapeTest(unittest.TestCase):
    def test_f32_returns_a_float32_view(self):
        x = torch.randn(2, 3, dtype=torch.bfloat16)
        out = gdn.f32(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, torch.float32)

    def test_prenorm_buffers_have_the_kernel_layout(self):
        qn, kn, eg, beta = gdn.prenorm_buffers(
            rows=5, h=4, kdim=8, hv=8, device=torch.device("cpu")
        )
        self.assertEqual(tuple(qn.shape), (5, 4, 8))
        self.assertEqual(tuple(kn.shape), (5, 4, 8))
        self.assertEqual(tuple(eg.shape), (5, 8))
        self.assertEqual(tuple(beta.shape), (5, 8))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a device")
    def test_conv_update_preserves_the_token_shape(self):
        dim, width, rows, batch = 8, 4, 4, 2
        x = torch.randn(batch, dim, dtype=torch.bfloat16, device="cuda")
        state = torch.zeros(rows, dim, width - 1, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(dim, width, dtype=torch.bfloat16, device="cuda")
        indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        reason = gdn.gdn_conv1d_unsupported_reason(
            x, state, weight, conv_state_indices=indices
        )
        if reason is not None:
            self.skipTest(reason)
        out = gdn.gdn_conv1d_update(
            x, state, weight, conv_state_indices=indices
        )
        self.assertEqual(tuple(out.shape), (batch, dim))
        self.assertEqual(out.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
