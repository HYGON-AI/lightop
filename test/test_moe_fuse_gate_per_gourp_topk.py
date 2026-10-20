# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch

from lightop import moe_fuse_gate_per_gourp_topk


TOKEN_SIZES = [
    1,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
]


CONFIGS = [
    {
        "name": "single-group",
        "num_expert_slots": 128,
        "num_expert_group": 1,
        "topk_per_group": 7,
        "num_pad_experts": 0,
        "shared_variants": (0, 1),
    },
    {
        "name": "two-group",
        "num_expert_slots": 256,
        "num_expert_group": 2,
        "topk_per_group": 7,
        "num_pad_experts": 2,
        "shared_variants": (0, 2),
    },
]


def torch_per_group_topk_reference(
    logits,
    correction_bias,
    num_expert_group,
    topk_per_group,
    num_pad_experts,
    num_shared_experts,
    routed_scaling_factor,
    apply_routed_scaling_factor_on_output,
):
    """Torch baseline corresponding to the generated per-group router graph."""
    num_tokens, num_expert_slots = logits.shape
    slots_per_group = num_expert_slots // num_expert_group
    assert slots_per_group * num_expert_group == num_expert_slots

    # Keep the reference as a normal PyTorch implementation. Packed score/id
    # keys are an internal kernel optimization and must not affect the baseline.
    probabilities = logits.sigmoid()
    choice_scores = probabilities + correction_bias.unsqueeze(0)
    choice_scores = choice_scores.view(
        num_tokens, num_expert_group, slots_per_group
    )
    grouped_probabilities = probabilities.view(
        num_tokens, num_expert_group, slots_per_group
    )

    if num_pad_experts:
        assert num_pad_experts % num_expert_group == 0
        num_pad_experts_per_group = (
            num_pad_experts // num_expert_group
        )
        choice_scores = choice_scores.clone()
        choice_scores[..., -num_pad_experts_per_group:] = float(
            "-inf"
        )
    group_topk_ids = torch.topk(
        choice_scores,
        k=topk_per_group,
        dim=-1,
        sorted=False,
    ).indices
    group_topk_weights = torch.gather(
        grouped_probabilities,
        dim=-1,
        index=group_topk_ids,
    )
    group_topk_weights = (
        group_topk_weights
        / group_topk_weights.sum(dim=-1, keepdim=True)
    )
    if apply_routed_scaling_factor_on_output:
        group_topk_weights = (
            group_topk_weights * routed_scaling_factor
        )

    group_offsets = (
        torch.arange(
            num_expert_group,
            device=logits.device,
            dtype=group_topk_ids.dtype,
        ).view(1, num_expert_group, 1)
        * slots_per_group
    )
    routed_topk = num_expert_group * topk_per_group
    routed_ids = (
        group_topk_ids + group_offsets
    ).reshape(num_tokens, routed_topk)
    routed_weights = group_topk_weights.reshape(
        num_tokens, routed_topk
    ).float()

    if num_shared_experts == 0:
        return routed_weights, routed_ids.to(torch.int32)

    if (
        num_expert_group == 2
        and num_pad_experts == num_shared_experts
    ):
        # The two shared experts occupy the last physical
        # slot of each 128-slot group: IDs 127 and 255.
        shared_ids = (
            torch.arange(
                1,
                num_expert_group + 1,
                device=logits.device,
                dtype=torch.int32,
            )
            * slots_per_group
            - 1
        ).view(1, -1).expand(num_tokens, -1)
    else:
        shared_ids = torch.arange(
            num_expert_slots,
            num_expert_slots + num_shared_experts,
            device=logits.device,
            dtype=torch.int32,
        ).view(1, -1).expand(num_tokens, -1)
    shared_weights = torch.ones(
        (num_tokens, num_shared_experts),
        device=logits.device,
        dtype=torch.float32,
    )
    return (
        torch.cat((routed_weights, shared_weights), dim=-1),
        torch.cat((routed_ids.to(torch.int32), shared_ids), dim=-1),
    )


def canonicalize(weights, expert_ids):
    order = torch.argsort(expert_ids, dim=-1)
    return (
        torch.gather(weights, 1, order),
        torch.gather(expert_ids, 1, order),
    )


def assert_matches_reference(
    model_name,
    weights,
    expert_ids,
    reference_weights,
    reference_ids,
    num_expert_slots,
    num_expert_group,
    num_pad_experts,
    num_shared_experts,
    rtol,
    atol,
):
    weights, expert_ids = canonicalize(weights, expert_ids)
    reference_weights, reference_ids = canonicalize(
        reference_weights, reference_ids
    )

    if not torch.equal(expert_ids, reference_ids):
        different = expert_ids != reference_ids
        first_positions = torch.nonzero(different)[:10].tolist()
        raise AssertionError(
            f"{model_name} expert IDs differ at "
            f"{different.sum().item()} positions; "
            f"first positions: {first_positions}"
        )

    if num_pad_experts:
        assert num_pad_experts % num_expert_group == 0
        slots_per_group = num_expert_slots // num_expert_group
        num_pad_experts_per_group = (
            num_pad_experts // num_expert_group
        )
        first_pad_offset = slots_per_group - num_pad_experts_per_group
        selected_padding = (
            expert_ids.remainder(slots_per_group) >= first_pad_offset
        )
        if (
            num_expert_group == 2
            and num_pad_experts == num_shared_experts
        ):
            shared_ids = torch.tensor(
                [127, 255],
                device=expert_ids.device,
                dtype=expert_ids.dtype,
            )
            selected_padding &= ~torch.isin(expert_ids, shared_ids)
        if torch.any(selected_padding):
            raise AssertionError(
                f"{model_name} selected a padding expert"
            )

    max_abs = (
        reference_weights - weights
    ).abs().max().item()
    max_rel = (
        (reference_weights - weights).abs()
        / (weights.abs() + 1e-12)
    ).max().item()
    if not torch.allclose(
        reference_weights,
        weights,
        rtol=rtol,
        atol=atol,
    ):
        raise AssertionError(
            f"{model_name} weights differ: "
            f"max_abs={max_abs:.4e}, max_rel={max_rel:.4e}, "
            f"rtol={rtol}, atol={atol}"
        )
    return max_abs, max_rel


def run_case(
    config,
    num_tokens,
    num_shared_experts,
    input_dtype=torch.float32,
    bias_dtype=torch.float32,
    routed_scaling_factor=2.5,
    apply_routed_scaling_factor_on_output=True,
    warmup=20,
    repeat=100,
    rtol=1e-6,
    atol=1e-6,
):
    device = torch.device("cuda")
    num_expert_slots = config["num_expert_slots"]
    logits = torch.randn(
        (num_tokens, num_expert_slots),
        device=device,
        dtype=input_dtype,
    )
    correction_bias = torch.randn(
        num_expert_slots,
        device=device,
        dtype=bias_dtype,
    )

    reference_weights, reference_ids = (
        torch_per_group_topk_reference(
            logits,
            correction_bias,
            config["num_expert_group"],
            config["topk_per_group"],
            config["num_pad_experts"],
            num_shared_experts,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        )
    )
    weights, expert_ids = moe_fuse_gate_per_gourp_topk(
        logits,
        correction_bias,
        config["num_expert_group"],
        config["topk_per_group"],
        config["num_pad_experts"],
        num_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )

    total_topk = (
        config["num_expert_group"] * config["topk_per_group"]
        + num_shared_experts
    )
    assert weights.shape == (num_tokens, total_topk)
    assert expert_ids.shape == (num_tokens, total_topk)
    assert weights.dtype == torch.float32
    assert expert_ids.dtype == torch.int32

    max_abs, max_rel = assert_matches_reference(
        config["name"],
        weights,
        expert_ids,
        reference_weights,
        reference_ids,
        num_expert_slots,
        config["num_expert_group"],
        config["num_pad_experts"],
        num_shared_experts,
        rtol,
        atol,
    )

    for _ in range(warmup):
        moe_fuse_gate_per_gourp_topk(
            logits,
            correction_bias,
            config["num_expert_group"],
            config["topk_per_group"],
            config["num_pad_experts"],
            num_shared_experts,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        moe_fuse_gate_per_gourp_topk(
            logits,
            correction_bias,
            config["num_expert_group"],
            config["topk_per_group"],
            config["num_pad_experts"],
            num_shared_experts,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        )
    end.record()
    end.synchronize()
    latency_us = start.elapsed_time(end) * 1000.0 / repeat
    return max_abs, max_rel, latency_us


def run_dtype_coverage():
    dtype_pairs = (
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.float16),
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
    )
    count = 0
    for config in CONFIGS:
        num_shared_experts = config["shared_variants"][-1]
        for input_dtype, bias_dtype in dtype_pairs:
            max_abs, max_rel, _ = run_case(
                config,
                num_tokens=32,
                num_shared_experts=num_shared_experts,
                input_dtype=input_dtype,
                bias_dtype=bias_dtype,
                warmup=1,
                repeat=1,
                rtol=1e-3 if input_dtype != torch.float32 else 1e-6,
                atol=1e-3 if input_dtype != torch.float32 else 1e-6,
            )
            count += 1
            print(
                f"[PASS][dtype][{config['name']}] "
                f"input={input_dtype} bias={bias_dtype} "
                f"max_abs={max_abs:.4e} max_rel={max_rel:.4e}"
            )
    print(f"[Dtype] {count}/{len(CONFIGS) * len(dtype_pairs)} passed")


def run_padding_layout_regression():
    """The two-group layout pads the last slot of each physical group."""
    config = CONFIGS[1]
    device = torch.device("cuda")
    num_expert_slots = config["num_expert_slots"]
    base_logits = torch.linspace(
        -4.0,
        4.0,
        num_expert_slots,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    # IDs 127 and 255 are padding. ID 254 is a real expert and must remain
    # selectable; this distinguishes per-group padding from a global tail mask.
    base_logits[0, 127] = 100.0
    base_logits[0, 254] = 90.0
    base_logits[0, 255] = 100.0

    dtype_pairs = (
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
    )
    for input_dtype, bias_dtype in dtype_pairs:
        logits = base_logits.to(input_dtype)
        correction_bias = torch.zeros(
            num_expert_slots, device=device, dtype=bias_dtype
        )
        reference_weights, reference_ids = torch_per_group_topk_reference(
            logits,
            correction_bias,
            config["num_expert_group"],
            config["topk_per_group"],
            config["num_pad_experts"],
            0,
            2.5,
            True,
        )
        # Exercise the public wrapper's plugin-facing [T, G, slots] input.
        weights, expert_ids = moe_fuse_gate_per_gourp_topk(
            logits.view(1, config["num_expert_group"], -1),
            correction_bias,
            config["num_expert_group"],
            config["topk_per_group"],
            config["num_pad_experts"],
            0,
            2.5,
            True,
        )
        tolerance_value = 1e-6 if input_dtype == torch.float32 else 1e-3
        assert_matches_reference(
            config["name"],
            weights,
            expert_ids,
            reference_weights,
            reference_ids,
            num_expert_slots,
            config["num_expert_group"],
            config["num_pad_experts"],
            0,
            tolerance_value,
            tolerance_value,
        )
        assert not torch.any((expert_ids == 127) | (expert_ids == 255))
        assert torch.any(expert_ids == 254)
        print(
            "[PASS][two-group] per-group padding IDs 127/255 "
            f"input={input_dtype} bias={bias_dtype}"
        )


def run_fp32_sigmoid_ordering_regression():
    """FP32 sigmoid numerics must not change a near-tie expert choice."""
    config = CONFIGS[1]
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(43)
    logits = torch.randn(
        (1, config["num_expert_slots"]),
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    probabilities = logits.sigmoid()

    # Build distinct adjacent FP32 target scores. The 7th/8th scores are one
    # ULP apart, so an exp2 sigmoid approximation changes the selected expert.
    value = torch.tensor(0.25, device=device, dtype=torch.float32)
    infinity = torch.tensor(float("inf"), device=device, dtype=torch.float32)
    target_values = []
    for _ in range(128):
        target_values.append(value)
        value = torch.nextafter(torch.nextafter(value, infinity), infinity)
    target_values = torch.stack(target_values)
    target_scores = torch.cat(
        (
            target_values[
                torch.randperm(128, generator=generator).to(device)
            ],
            target_values[
                torch.randperm(128, generator=generator).to(device)
            ],
        )
    )
    correction_bias = (target_scores - probabilities[0]).contiguous()

    reference_weights, reference_ids = torch_per_group_topk_reference(
        logits,
        correction_bias,
        config["num_expert_group"],
        config["topk_per_group"],
        config["num_pad_experts"],
        0,
        2.5,
        True,
    )
    weights, expert_ids = moe_fuse_gate_per_gourp_topk(
        logits,
        correction_bias,
        config["num_expert_group"],
        config["topk_per_group"],
        config["num_pad_experts"],
        0,
        2.5,
        True,
    )
    assert_matches_reference(
        config["name"],
        weights,
        expert_ids,
        reference_weights,
        reference_ids,
        config["num_expert_slots"],
        config["num_expert_group"],
        config["num_pad_experts"],
        0,
        1e-6,
        1e-6,
    )
    print("[PASS][two-group] FP32 sigmoid near-tie ordering")


def run_shared_pad_layout_regression():
    """The two-group shared experts must occupy physical IDs 127/255."""
    config = CONFIGS[1]
    device = torch.device("cuda")
    logits = torch.randn(
        (8, config["num_expert_slots"]),
        device=device,
        dtype=torch.float32,
    )
    correction_bias = torch.randn(
        config["num_expert_slots"],
        device=device,
        dtype=torch.float32,
    )
    weights, expert_ids = moe_fuse_gate_per_gourp_topk(
        logits,
        correction_bias,
        config["num_expert_group"],
        config["topk_per_group"],
        config["num_pad_experts"],
        2,
        2.5,
        True,
    )
    assert torch.equal(
        expert_ids[:, -2:],
        torch.tensor([127, 255], device=device, dtype=torch.int32)
        .view(1, 2)
        .expand(8, 2),
    )
    assert torch.equal(weights[:, -2:], torch.ones_like(weights[:, -2:]))
    assert not torch.any((expert_ids[:, :-2] == 127) | (expert_ids[:, :-2] == 255))
    print("[PASS][two-group] shared experts use physical IDs 127/255")


def run_single_group_shared_id_regression():
    """The single-group layout has no in-group pad, so shared remains ID 128."""
    config = CONFIGS[0]
    device = torch.device("cuda")
    logits = torch.randn(
        (8, config["num_expert_slots"]),
        device=device,
        dtype=torch.float32,
    )
    correction_bias = torch.randn(
        config["num_expert_slots"],
        device=device,
        dtype=torch.float32,
    )
    weights, expert_ids = moe_fuse_gate_per_gourp_topk(
        logits,
        correction_bias,
        config["num_expert_group"],
        config["topk_per_group"],
        config["num_pad_experts"],
        1,
        1.0,
        True,
    )
    assert torch.equal(
        expert_ids[:, -1],
        torch.full((8,), 128, device=device, dtype=torch.int32),
    )
    assert torch.equal(weights[:, -1], torch.ones_like(weights[:, -1]))
    print("[PASS][single-group] shared expert remains physical ID 128")


def run_size_coverage():
    results = []
    for config in CONFIGS:
        for num_shared_experts in config["shared_variants"]:
            for num_tokens in TOKEN_SIZES:
                max_abs, max_rel, latency_us = run_case(
                    config,
                    num_tokens,
                    num_shared_experts,
                )
                total_topk = (
                    config["num_expert_group"]
                    * config["topk_per_group"]
                    + num_shared_experts
                )
                results.append(
                    (
                        config["name"],
                        num_tokens,
                        num_shared_experts,
                        max_abs,
                        max_rel,
                        latency_us,
                    )
                )
                print(
                    f"[PASS][{config['name']}] "
                    f"tokens={num_tokens:5d} "
                    f"slots={config['num_expert_slots']} "
                    f"groups={config['num_expert_group']} "
                    f"per_group_topk={config['topk_per_group']} "
                    f"pad={config['num_pad_experts']} "
                    f"shared={num_shared_experts} "
                    f"total_topk={total_topk:2d} "
                    f"max_abs={max_abs:.4e} "
                    f"max_rel={max_rel:.4e} "
                    f"time_us={latency_us:.3f}"
                )

    expected = sum(
        len(config["shared_variants"]) * len(TOKEN_SIZES)
        for config in CONFIGS
    )
    print(
        f"[Size coverage] {len(results)}/{expected} passed; "
        "timing warmup=20 repeat=100"
    )
    return results


if __name__ == "__main__":
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "A CUDA/ROCm device is required"
    run_padding_layout_regression()
    run_shared_pad_layout_regression()
    run_single_group_shared_id_regression()
    run_fp32_sigmoid_ordering_regression()
    run_dtype_coverage()
    run_size_coverage()
