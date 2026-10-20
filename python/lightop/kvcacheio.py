# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from . import op


def hcu_alloc_extend_kernel(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs: int,
    page_size: int,
):
    return op.hcu_alloc_extend_kernel(
        pre_lens_ptr,
        seq_lens_ptr,
        last_loc_ptr,
        free_page_ptr,
        out_indices,
        bs,
        page_size,
    )
