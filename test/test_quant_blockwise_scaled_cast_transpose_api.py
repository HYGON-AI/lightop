"""API compatibility checks for the blockwise quantization migration."""

import lightop
import lightop.quant
import lightop.quant


def test_blockwise_scaled_cast_transpose_category_export() -> None:
    assert "blockwise_scaled_cast_transpose" in lightop.quant.__all__
    assert (
        lightop.quant.blockwise_scaled_cast_transpose
        is lightop.quant.blockwise_scaled_cast_transpose
    )
