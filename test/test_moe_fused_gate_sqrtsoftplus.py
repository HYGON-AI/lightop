import torch

from lightop.moe import moe_fused_gate_sqrtsoftplus


def reference_moe_fused_gate_sqrtsoftplus(
    input,
    bias,
    topk,
    num_fused_shared_experts=0,
    renormalize=True,
    routed_scaling_factor=1.0,
    apply_routed_scaling_factor_on_output=False,
):
    scores = torch.sqrt(torch.log1p(torch.exp(input.float())))
    selection_scores = scores + bias.float()
    num_rows, num_experts = input.shape
    topk_routed = topk - num_fused_shared_experts

    _, topk_ids = torch.topk(selection_scores, k=topk_routed, dim=-1)
    topk_weights = torch.gather(scores, dim=1, index=topk_ids)
    routed_sum = topk_weights.sum(dim=-1, keepdim=True)

    if num_fused_shared_experts > 0:
        shared_ids = torch.arange(
            num_experts,
            num_experts + num_fused_shared_experts,
            device=input.device,
            dtype=torch.int64,
        )
        shared_ids = shared_ids.unsqueeze(0).expand(num_rows, -1)
        shared_weights = routed_sum / routed_scaling_factor
        shared_weights = shared_weights.expand(num_rows, num_fused_shared_experts)
        topk_ids = torch.cat([topk_ids, shared_ids], dim=-1)
        topk_weights = torch.cat([topk_weights, shared_weights], dim=-1)

    norm = routed_sum if renormalize else 1.0
    topk_weights = topk_weights / norm
    if apply_routed_scaling_factor_on_output:
        topk_weights = topk_weights * routed_scaling_factor

    return topk_weights.float(), topk_ids.int()


def check_case(
    num_rows,
    topk,
    num_fused_shared_experts,
    renormalize=True,
    routed_scaling_factor=2.5,
    apply_routed_scaling_factor_on_output=False,
):
    torch.manual_seed(0)
    input = torch.randn(num_rows, 256, device="cuda", dtype=torch.float32)
    bias = torch.randn(256, device="cuda", dtype=torch.float32)

    ref_weights, ref_ids = reference_moe_fused_gate_sqrtsoftplus(
        input,
        bias,
        topk,
        num_fused_shared_experts,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )
    opt_weights, opt_ids = moe_fused_gate_sqrtsoftplus(
        input,
        bias,
        topk,
        num_fused_shared_experts,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )

    assert torch.equal(ref_ids, opt_ids), (ref_ids, opt_ids)
    assert torch.allclose(ref_weights, opt_weights, rtol=1e-5, atol=1e-5), (
        ref_weights,
        opt_weights,
    )


def test_moe_fused_gate_sqrtsoftplus():
    assert torch.cuda.is_available(), "CUDA/ROCm is required"
    for num_rows in [1, 16, 256, 513]:
        check_case(num_rows, topk=6, num_fused_shared_experts=0)
        check_case(num_rows, topk=6, num_fused_shared_experts=1)
        check_case(
            num_rows,
            topk=6,
            num_fused_shared_experts=1,
            apply_routed_scaling_factor_on_output=True,
        )


if __name__ == "__main__":
    test_moe_fused_gate_sqrtsoftplus()
    print("moe_fused_gate_sqrtsoftplus passed")
