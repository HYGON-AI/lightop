"""Public API checks for native ``per_token_group_quant_fp8``."""

import inspect

import torch

from lightop import op as lightop_native
from lightop.quant import per_token_group_quant_fp8 as category_api


def test_per_token_group_quant_fp8_uses_native_lightop() -> None:
    assert list(inspect.signature(category_api).parameters) == [
        "x",
        "group_size",
        "eps",
        "dtype",
        "column_major_scales",
        "out_q",
    ]

    torch.manual_seed(3)
    input_tensor = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
    group_size = 128
    eps = 1e-10
    preallocated_q = torch.empty_like(
        input_tensor, dtype=torch.float8_e4m3fn
    )

    category_q, category_scale = category_api(
        input_tensor,
        group_size,
        eps=eps,
        out_q=preallocated_q,
    )
    assert category_q is preallocated_q

    native_q = torch.empty_like(category_q)
    native_scale = torch.empty_like(category_scale)
    lightop_native.per_token_group_quant_fp8(
        native_q,
        input_tensor,
        native_scale,
        group_size,
        eps,
        False,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        category_q.float(), native_q.float(), rtol=0, atol=0
    )
    torch.testing.assert_close(category_scale, native_scale, rtol=0, atol=0)
