"""Migration tests for ``w4tow16``."""

import torch

import lightop
import lightop.quant
import lightop.quant


def test_w4tow16_export_and_unpack_reference() -> None:
    category_api = lightop.quant.w4tow16
    assert category_api is lightop.quant.w4tow16
    assert "w4tow16" in lightop.quant.__all__

    packed = torch.tensor(
        [[0x76543210, 0xFEDCBA98], [0x01234567, 0x89ABCDEF]],
        device="cuda",
        dtype=torch.uint32,
    )
    scales = torch.tensor([0.5, 2.0], device="cuda", dtype=torch.float16)
    zeros = torch.tensor([1.0, -2.0], device="cuda", dtype=torch.float16)

    output = category_api(packed, scales, zeros)
    torch.cuda.synchronize()

    nibbles = torch.tensor(
        [
            list(range(16)),
            list(range(7, -1, -1)) + list(range(15, 7, -1)),
        ],
        device="cuda",
        dtype=torch.float16,
    )
    expected = nibbles * scales[:, None] + zeros[:, None]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
