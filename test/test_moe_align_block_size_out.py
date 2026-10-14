# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Checks for the categorized preallocated MoE alignment API."""

from unittest.mock import patch

from lightop.moe import routing


def test_moe_align_block_size_out_forwards_ep_and_fill_flags() -> None:
    arguments = [object() for _ in range(9)]
    expected = object()

    with patch.object(
        routing.op,
        "moe_align_block_size",
        return_value=expected,
    ) as native_api:
        result = routing.moe_align_block_size_out(
            *arguments,
            is_ep=True,
            is_fuse_fill=False,
        )

    assert result is expected
    native_api.assert_called_once_with(*arguments, True, False)
