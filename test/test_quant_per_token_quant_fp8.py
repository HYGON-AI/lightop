"""Public API checks for native ``per_token_quant_fp8``."""

import inspect

import torch

from lightop import op as lightop_native
from lightop.quant import per_token_quant_fp8 as category_api


def test_per_token_quant_fp8_uses_native_lightop() -> None:
    assert list(inspect.signature(category_api).parameters) == [
        "x",
        "dtype",
        "out_q",
        "out_scale",
    ]

    torch.manual_seed(4)
    input_tensor = torch.randn((3, 257), device="cuda", dtype=torch.float16)

    category_q, category_scale = category_api(input_tensor)
    native_q = torch.empty_like(category_q)
    native_scale = torch.empty_like(category_scale)
    lightop_native.per_token_quant_fp8(
        native_q, input_tensor, native_scale
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        category_q.float(), native_q.float(), rtol=0, atol=0
    )
    torch.testing.assert_close(category_scale, native_scale, rtol=0, atol=0)


def test_per_token_quant_fp8_accepts_preallocated_outputs() -> None:
    torch.manual_seed(5)
    input_tensor = torch.randn((4, 257), device="cuda", dtype=torch.float16)
    out_q = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn)
    out_scale = torch.empty((4, 1), device="cuda", dtype=torch.float32)

    category_q, category_scale = category_api(
        input_tensor,
        dtype=out_q.dtype,
        out_q=out_q,
        out_scale=out_scale,
    )

    assert category_q is out_q
    assert category_scale is out_scale

    native_q = torch.empty_like(out_q)
    native_scale = torch.empty_like(out_scale)
    lightop_native.per_token_quant_fp8(
        native_q, input_tensor, native_scale
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out_q.float(), native_q.float(), rtol=0, atol=0)
    torch.testing.assert_close(out_scale, native_scale, rtol=0, atol=0)
