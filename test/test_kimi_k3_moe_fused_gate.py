import pytest
import torch

from lightop.moe_fused_gate import moe_fused_gate


def k3_reference(logits, bias, routed_scaling_factor, apply_scale):
    probabilities = torch.sigmoid(logits.float())
    selection_scores = probabilities + bias.float().unsqueeze(0)
    topk_ids = torch.topk(selection_scores, k=16, dim=-1, sorted=True).indices
    topk_weights = probabilities.gather(1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if apply_scale:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.float(), topk_ids.int()


@pytest.mark.parametrize("tokens", [1, 7, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("apply_scale", [False, True])
def test_kimi_k3_896_group1_top16(tokens, dtype, apply_scale):
    torch.manual_seed(1234 + tokens)
    logits = torch.randn(tokens, 896, device="cuda", dtype=dtype)
    bias = torch.randn(896, device="cuda", dtype=dtype)
    scale = 2.5

    expected_weights, expected_ids = k3_reference(
        logits, bias, scale, apply_scale
    )
    actual_weights, actual_ids = moe_fused_gate(
        logits,
        bias,
        num_expert_group=1,
        topk_group=1,
        topk=16,
        n_share_experts_fusion=0,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=apply_scale,
    )

    # The vLLM grouped-router contract does not require a particular order for
    # equal selection scores. Compare each (expert ID, weight) pair instead.
    expected_order = torch.argsort(expected_ids, dim=-1)
    actual_order = torch.argsort(actual_ids, dim=-1)
    expected_ids = torch.gather(expected_ids, 1, expected_order)
    actual_ids = torch.gather(actual_ids, 1, actual_order)
    expected_weights = torch.gather(expected_weights, 1, expected_order)
    actual_weights = torch.gather(actual_weights, 1, actual_order)

    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(
        actual_weights,
        expected_weights,
        rtol=5e-3 if dtype is torch.bfloat16 else 2e-5,
        atol=5e-3 if dtype is torch.bfloat16 else 2e-6,
    )


if __name__ == "__main__":
    pytest.main([__file__])
