"""Migration comparison for LMSlim ``per_token_quant_fp8``."""

import inspect

import torch

from lightop.quant import per_token_quant_fp8 as migrated_api
from lmslim.layers.gemm.fp8_utils import per_token_quant_fp8 as source_api


def test_per_token_quant_fp8_matches_lmslim_source() -> None:
    assert inspect.signature(migrated_api) == inspect.signature(source_api)

    torch.manual_seed(4)
    input_tensor = torch.randn((3, 257), device="cuda", dtype=torch.float16)

    migrated_q, migrated_scale = migrated_api(input_tensor)
    source_q, source_scale = source_api(input_tensor)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        migrated_q.float(), source_q.float(), rtol=0, atol=0
    )
    torch.testing.assert_close(migrated_scale, source_scale, rtol=0, atol=0)
