from typing import Optional
from lightop import op
import torch

def topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Paged Top512 (or Top1024 by output width), with optional logical IDs.

    On wave64 HIP, SGL_USE_LIGHTOP_TOPK_BACKAND=0 (also the default) selects
    exact adaptive TopK. Its output IDs are sorted by logical position.
    Backends 1 and 2 retain their previous behavior and supported K values.
    """
    topk = out_page_indices.shape[1]
    if topk == 512:
        op.topk_transform_512(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            page_size,
            out_raw_indices,
        )
    elif topk == 1024:
        op.topk_transform_1024(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            page_size,
            out_raw_indices,
        )
    else:
        raise ValueError(f"unsupported topk={topk}; expected 512 or 1024")

def topk_transform_1024(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Paged Top1024; backend 0/default uses native adaptive selection on HIP."""
    op.topk_transform_1024(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        out_raw_indices,
    )

def fast_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,  # NOTE: page size should be 1
    cu_seqlens_q: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get the topk indices of the score tensor and then transform the topk indices
    to indices to the page table (page_size = 1)
    On wave64 HIP, SGL_USE_LIGHTOP_TOPK_BACKAND=0 or unset selects the exact
    adaptive Top2048 implementation. It requires fully materialized FP32 scores, sorts logical
    token IDs before mapping, and breaks equal ordered-FP32-key ties by smaller
    logical ID. Set the environment variable before the first call.
    Top512/1024 and ragged Top2048 share the adaptive selection core;
    backend 1 retains the previous implementation; backend 2 requires Top512.
    Args:
        score: The score tensor of shape (B, L). The score tensor is the logits
            between the query and the key whose layout is either ragged or paged.
            row_starts is only required when the key is ragged.
        lengths: The lengths tensor of shape (B)
        page_table_size_1: The page table tensor of shape (Batch, topk)
        cu_seqlens_q: The cumulative sequence lengths tensor of shape (Batch + 1)
        topk: The number of topk indices to get
        row_starts: The start index of each row in the score tensor of shape (B).
            For each row i, topk only applies to section [row_starts[i], row_starts[i] + lengths[i]]
            of the score tensor. It's only used for cases where the key is
            ragged, i.e. during extend and draft extend.
    Returns:
        The topk indices tensor of shape (B, topk)
    """
    assert (
        topk == 2048
    ), "fast_topk_transform_fused is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2
    src_page_table = page_table_size_1
    dst_page_table = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    op.fast_topk_transform_interface(
        score, lengths, dst_page_table, src_page_table, cu_seqlens_q, row_starts
    )
    return dst_page_table


def fast_topk_transform_sparse_mask_fused(
    score: torch.Tensor,
    nonzero_mask: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Exact TopK over sparse paged-MQA output.

    ``nonzero_mask`` packs 16 score positions into every int16 element.
    Masked-out positions are reconstructed as positive zero without reading
    the corresponding uninitialized score storage.
    """

    if topk != 2048:
        raise ValueError("sparse-mask TopK requires topk=2048")
    if score.dim() != 2 or score.dtype != torch.float32:
        raise ValueError("score must be float32 [rows, width]")
    expected_mask_width = (score.shape[1] + 15) // 16
    if (nonzero_mask.dtype != torch.int16 or nonzero_mask.dim() != 2 or
            nonzero_mask.shape != (score.shape[0], expected_mask_width)):
        raise ValueError(
            "nonzero_mask must be int16 [rows, ceil(width/16)]")
    dst_page_table = score.new_empty(
        (score.shape[0], topk), dtype=torch.int32)
    op.fast_topk_transform_sparse_mask_interface(
        score,
        nonzero_mask,
        lengths,
        dst_page_table,
        page_table_size_1,
        cu_seqlens_q,
    )
    return dst_page_table

def fast_topk_transform_ragged_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices_offset: torch.Tensor,  # ragged kv
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get the topk indices of the score tensor and then transform the topk indices to
    indices to ragged kv (non-paged). This function is only used for extend,
    not including draft extend.
    On HIP, backend 0/default uses the same exact adaptive selection as the
    paged interface, then adds each row's topk_indices_offset.
    Args:
        score: The score tensor of shape (B, L). The score tensor is the logits
            between the query and the key which can be ragged or paged.
            row_starts is only required when the key is ragged.
        lengths: The lengths tensor of shape (B)
        topk_indices_offset: The offset of topk indices in ragged kv of shape (B)
        topk: The number of topk indices to get
        row_starts: The start index of each row in the score tensor of shape (B).
            For each row i, topk only applies to section [row_starts[i], row_starts[i] + lengths[i]]
            of the score tensor. It can be None if only the fast path is triggered,
            in the case of all values in lengths <= topk (not checked in the kernel,
            guaranteed by the caller).
    Returns:
        The topk indices tensor of shape (B, topk)
    """
    assert (
        topk == 2048
    ), "fast_topk_transform_ragged_fused is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2
    topk_indices_ragged = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    op.fast_topk_transform_ragged_interface(
        score, lengths, topk_indices_ragged, topk_indices_offset, row_starts
    )
    return topk_indices_ragged


def fast_kpool_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    out_rows: Optional[int] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fuse pooled TopK, token expansion, index mapping, and tail append."""
    if pool_size not in (4, 16) or topk != 2048:
        raise ValueError(
            "fast_kpool_topk_transform_fused requires pool_size=4 or 16 "
            "and topk=2048"
        )
    if seq_lens is None:
        raise ValueError("seq_lens is required for the kpool tail")
    if page_table is not None and topk_indices_offset is not None:
        raise ValueError("page_table and topk_indices_offset are mutually exclusive")
    if topk_indices_offset is not None and topk_indices_offset.ndim == 2:
        topk_indices_offset = topk_indices_offset.squeeze(1)

    output_rows = score.shape[0] if out_rows is None else out_rows
    output = torch.empty(
        (output_rows, topk + pool_size - 1),
        dtype=torch.int32,
        device=score.device,
    )
    op.fast_kpool_topk_transform_interface(
        score,
        lengths,
        output,
        pool_size,
        seq_lens,
        page_table,
        topk_indices_offset,
        row_starts,
        page_table_row_index,
    )
    return output
