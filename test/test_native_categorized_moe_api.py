"""Verify categorized exports for native MoE routing and EP operators."""

import torch

import lightop.moe
from lightop import op as lightop_native
from lightop.moe import (
    ep_build_m_indices,
    ep_gather,
    ep_scatter,
    topk_ids_postprocess,
    topk_softmax,
)


def test_native_categorized_moe_exports() -> None:
    expected = {
        "ep_build_m_indices": ep_build_m_indices,
        "ep_gather": ep_gather,
        "ep_scatter": ep_scatter,
        "topk_ids_postprocess": topk_ids_postprocess,
        "topk_softmax": topk_softmax,
    }
    for name, categorized_api in expected.items():
        assert categorized_api is getattr(lightop_native, name)
        assert name in lightop.moe.__all__


def test_ep_build_m_indices_reference() -> None:
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [0, -1]], device="cuda", dtype=torch.int64
    )
    m_indices = torch.empty(6, device="cuda", dtype=torch.int32)

    ep_build_m_indices(topk_ids, m_indices, 3, 2)

    expected = torch.tensor(
        [0, 0, 1, 1, 2, -1], device="cuda", dtype=torch.int32
    )
    torch.testing.assert_close(m_indices, expected, rtol=0, atol=0)


def test_ep_gather_reference() -> None:
    torch.manual_seed(0)
    num_tokens, topk, hidden_size = 4, 2, 16
    source = torch.randn(
        num_tokens * topk,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0]],
        device="cuda",
        dtype=torch.int64,
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, topk, device="cuda"), dim=-1
    )
    inv_perm = torch.tensor(
        [[3, 0], [5, 6], [2, 7], [1, 4]],
        device="cuda",
        dtype=torch.int32,
    )
    output = torch.empty(
        num_tokens, hidden_size, device="cuda", dtype=source.dtype
    )

    ep_gather(
        source, topk_ids, topk_weights, inv_perm, None, output
    )

    expected = (
        source[inv_perm.long()].float()
        * topk_weights.unsqueeze(-1)
    ).sum(dim=1).to(source.dtype)
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


def test_ep_scatter_reference() -> None:
    num_tokens, topk, hidden_size, num_experts = 4, 2, 16, 4
    alignment = 2
    activations = torch.arange(
        num_tokens * hidden_size, device="cuda", dtype=torch.int8
    ).reshape(num_tokens, hidden_size)
    scales = torch.arange(
        1, num_tokens + 1, device="cuda", dtype=torch.float32
    ).reshape(num_tokens, 1)
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0]],
        device="cuda",
        dtype=torch.int64,
    )
    expert_counts = torch.bincount(
        topk_ids.flatten(), minlength=num_experts
    ).to(torch.int32)
    padded_counts = (
        (expert_counts + alignment - 1) // alignment * alignment
    )
    output_rows = int(padded_counts.sum().item())
    output = torch.empty(
        output_rows, hidden_size, device="cuda", dtype=torch.int8
    )
    output_scales = torch.empty(
        output_rows, 1, device="cuda", dtype=torch.float32
    )
    m_indices = torch.full(
        (output_rows,), -1, device="cuda", dtype=torch.int32
    )
    inv_perm = torch.empty(
        num_tokens, topk, device="cuda", dtype=torch.int32
    )

    ep_scatter(
        activations,
        scales,
        topk_ids,
        None,
        expert_counts.clone(),
        output,
        output_scales,
        m_indices,
        inv_perm,
        num_experts,
        alignment,
    )

    gathered_rows = inv_perm.long()
    expected_activations = activations[:, None, :].expand(
        -1, topk, -1
    )
    expected_scales = scales[:, None, :].expand(-1, topk, -1)
    torch.testing.assert_close(
        output[gathered_rows], expected_activations, rtol=0, atol=0
    )
    torch.testing.assert_close(
        output_scales[gathered_rows], expected_scales, rtol=0, atol=0
    )
    torch.testing.assert_close(
        m_indices[gathered_rows], topk_ids.to(torch.int32), rtol=0, atol=0
    )


def test_topk_ids_postprocess_reference() -> None:
    topk_ids = torch.tensor(
        [[0, 2], [1, 3], [2, 0]], device="cuda", dtype=torch.int64
    )
    logical_to_physical_map = torch.tensor(
        [10, 11, 12, 13], device="cuda", dtype=torch.int32
    )
    num_token_non_padded = torch.tensor(
        [2], device="cuda", dtype=torch.int32
    )

    topk_ids_postprocess(
        topk_ids, logical_to_physical_map, num_token_non_padded
    )

    expected = torch.tensor(
        [[10, 12], [11, 13], [-1, -1]],
        device="cuda",
        dtype=torch.int64,
    )
    torch.testing.assert_close(topk_ids, expected, rtol=0, atol=0)


def test_topk_softmax_reference() -> None:
    torch.manual_seed(1)
    num_tokens, num_experts, topk = 8, 16, 4
    gating_output = torch.randn(
        num_tokens, num_experts, device="cuda", dtype=torch.float32
    )
    weights = torch.empty(
        num_tokens, topk, device="cuda", dtype=torch.float32
    )
    indices = torch.empty(
        num_tokens, topk, device="cuda", dtype=torch.int32
    )
    token_expert_indices = torch.empty_like(indices)

    topk_softmax(
        weights, indices, token_expert_indices, gating_output, False
    )

    expected_weights, expected_indices = torch.topk(
        torch.softmax(gating_output, dim=-1), k=topk, dim=-1
    )
    actual_indices, order = torch.sort(indices, dim=-1)
    actual_weights = torch.gather(weights, -1, order)
    expected_indices, order = torch.sort(
        expected_indices.to(torch.int32), dim=-1
    )
    expected_weights = torch.gather(expected_weights, -1, order)
    torch.testing.assert_close(
        actual_indices, expected_indices, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_weights, expected_weights, rtol=1e-5, atol=1e-6
    )
