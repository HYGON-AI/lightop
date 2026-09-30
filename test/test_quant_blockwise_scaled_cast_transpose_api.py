# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""API compatibility checks for the blockwise quantization migration."""

import lightop
import lightop.quant


def test_blockwise_scaled_cast_transpose_category_export() -> None:
    assert "blockwise_scaled_cast_transpose" in lightop.quant.__all__
    assert (
        lightop.quant.blockwise_scaled_cast_transpose
        is lightop.blockwise_scaled_cast_transpose
    )
