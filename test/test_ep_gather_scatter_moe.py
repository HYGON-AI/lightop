# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch

from lightop import moe_fuse_gate_per_gourp_topk, op
from lightop.quant import per_token_dynamic_quant_int8


DEVICE = "cuda"
TOKENS = 32
HIDDEN = 256
ALIGNMENT = 16

CONFIGS = (
    {
        "name": "single-group-routed",
        "num_experts": 128,
        "num_slots": 128,
        "num_groups": 1,
        "per_group_topk": 7,
        "num_pad_experts": 0,
        "num_shared_experts": 0,
    },
    {
        "name": "single-group-shared",
        "num_experts": 128,
        "num_slots": 128,
        "num_groups": 1,
        "per_group_topk": 7,
        "num_pad_experts": 0,
        "num_shared_experts": 1,
    },
    {
        "name": "two-group-routed",
        "num_experts": 254,
        "num_slots": 256,
        "num_groups": 2,
        "per_group_topk": 7,
        "num_pad_experts": 2,
        "num_shared_experts": 0,
    },
    {
        "name": "two-group-shared",
        "num_experts": 254,
        "num_slots": 256,
        "num_groups": 2,
        "per_group_topk": 7,
        "num_pad_experts": 2,
        "num_shared_experts": 2,
    },
)


def make_gate_outputs(config):
    torch.manual_seed(20260731)
    shape = (
        (TOKENS, config["num_slots"])
        if config["num_groups"] == 1
        else (TOKENS, config["num_groups"], config["num_slots"] // config["num_groups"])
    )
    logits = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    bias = torch.randn(config["num_slots"], dtype=torch.float32, device=DEVICE)
    weights, ids = moe_fuse_gate_per_gourp_topk(
        logits,
        bias,
        config["num_groups"],
        config["per_group_topk"],
        config["num_pad_experts"],
        config["num_shared_experts"],
        1.0,
        True,
    )
    expected_k = (
        config["num_groups"] * config["per_group_topk"]
        + config["num_shared_experts"]
    )
    assert ids.shape == (TOKENS, expected_k)
    assert ids.dtype == torch.int32
    assert weights.shape == ids.shape
    assert weights.dtype == torch.float32
    return weights.contiguous(), ids.contiguous()


def resolve_local_ids(ids, expert_map, local_num_experts):
    if expert_map is None:
        mapped = ids.to(torch.int64)
        valid = (mapped >= 0) & (mapped < local_num_experts)
        return mapped, valid

    mapped = torch.full_like(ids, -1, dtype=torch.int64)
    in_map = (ids >= 0) & (ids < expert_map.numel())
    mapped[in_map] = expert_map[ids[in_map].to(torch.int64)].to(torch.int64)
    valid = (mapped >= 0) & (mapped < local_num_experts)
    return mapped, valid


def local_counts(ids, expert_map, local_num_experts):
    mapped, valid = resolve_local_ids(ids, expert_map, local_num_experts)
    return torch.bincount(
        mapped[valid], minlength=local_num_experts
    ).to(torch.int32)


def allocate_outputs(ids, expert_map, local_num_experts):
    counts = local_counts(ids, expert_map, local_num_experts)
    padded_counts = (counts + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    rows = int(padded_counts.sum().item())
    aq_out = torch.empty((rows, HIDDEN), dtype=torch.int8, device=DEVICE)
    scale_out = torch.empty((rows, 1), dtype=torch.float32, device=DEVICE)
    m_indices = torch.full((rows,), -1, dtype=torch.int32, device=DEVICE)
    inv_perm = torch.full(ids.shape, -2, dtype=torch.int32, device=DEVICE)
    return counts, aq_out, scale_out, m_indices, inv_perm


def check_scatter_result(
    source_q, source_scale, ids, expert_map, local_num_experts,
    aq_out, scale_out, m_indices, inv_perm,
):
    mapped, valid = resolve_local_ids(ids, expert_map, local_num_experts)
    assert torch.all(inv_perm[~valid] == -1)

    dst = inv_perm[valid].to(torch.int64)
    token_index = (
        torch.arange(ids.size(0), device=DEVICE)
        .unsqueeze(1)
        .expand_as(ids)[valid]
    )
    torch.testing.assert_close(aq_out[dst], source_q[token_index], rtol=0, atol=0)
    torch.testing.assert_close(
        scale_out[dst, 0], source_scale[token_index, 0], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        m_indices[dst], mapped[valid].to(torch.int32), rtol=0, atol=0
    )


def run_case(config, id_dtype, use_expert_map):
    weights, gate_ids = make_gate_outputs(config)
    ids = gate_ids.to(id_dtype).contiguous()
    num_experts = config["num_experts"]

    # The map uses physical router slots. For the two-group layout, padding slots are
    # 127/255 and physical expert 254 maps to logical routed expert 253.
    if use_expert_map:
        local_globals = torch.arange(
            1, num_experts, 2, dtype=torch.int64, device=DEVICE
        )
        local_num_experts = local_globals.numel()
        logical_to_local = torch.full(
            (num_experts,), -1, dtype=torch.int32, device=DEVICE
        )
        logical_to_local[local_globals] = torch.arange(
            local_num_experts, dtype=torch.int32, device=DEVICE
        )
        physical_to_logical = torch.arange(
            config["num_slots"], dtype=torch.int64, device=DEVICE
        )
        if config["num_pad_experts"]:
            physical_to_logical = torch.full(
                (config["num_slots"],),
                -1,
                dtype=torch.int64,
                device=DEVICE,
            )
            slots_per_group = config["num_slots"] // config["num_groups"]
            real_per_group = (
                slots_per_group
                - config["num_pad_experts"] // config["num_groups"]
            )
            for group in range(config["num_groups"]):
                physical_start = group * slots_per_group
                logical_start = group * real_per_group
                physical_to_logical[
                    physical_start : physical_start + real_per_group
                ] = torch.arange(
                    logical_start,
                    logical_start + real_per_group,
                    dtype=torch.int64,
                    device=DEVICE,
                )
        expert_map = torch.full(
            (config["num_slots"],), -1, dtype=torch.int32, device=DEVICE
        )
        valid_physical = physical_to_logical >= 0
        expert_map[valid_physical] = logical_to_local[
            physical_to_logical[valid_physical]
        ]
    else:
        # Without a map, EP consumes physical slot IDs directly.
        local_num_experts = config["num_slots"]
        expert_map = None

    if not use_expert_map:
        routed_k = config["num_groups"] * config["per_group_topk"]
        _, routed_valid = resolve_local_ids(
            ids[:, :routed_k], expert_map, local_num_experts
        )
        assert torch.all(routed_valid), "a routed physical expert was dropped"

    torch.manual_seed(1234)
    x = torch.randn(
        (TOKENS, HIDDEN), dtype=torch.bfloat16, device=DEVICE
    ).contiguous()
    source_q, source_scale = per_token_dynamic_quant_int8(
        x, torch.ones(HIDDEN, dtype=torch.float32, device=DEVICE)
    )
    source_scale = source_scale.unsqueeze(1).contiguous()

    # Input is already INT8: ep_scatter.
    counts, aq_out, scale_out, m_indices, inv_perm = allocate_outputs(
        ids, expert_map, local_num_experts
    )
    op.ep_scatter(
        source_q,
        source_scale,
        ids,
        expert_map,
        counts.clone(),
        aq_out,
        scale_out,
        m_indices,
        inv_perm,
        local_num_experts,
        ALIGNMENT,
    )
    torch.cuda.synchronize()
    check_scatter_result(
        source_q, source_scale, ids, expert_map, local_num_experts,
        aq_out, scale_out, m_indices, inv_perm,
    )

    # BF16 input with fused per-token INT8 quantization and scatter.
    counts_f, aq_out_f, scale_out_f, m_indices_f, inv_perm_f = allocate_outputs(
        ids, expert_map, local_num_experts
    )
    op.ep_fused_quant_scatter(
        x,
        ids,
        expert_map,
        counts_f.clone(),
        aq_out_f,
        scale_out_f,
        m_indices_f,
        inv_perm_f,
        local_num_experts,
        ALIGNMENT,
    )
    torch.cuda.synchronize()
    check_scatter_result(
        source_q, source_scale, ids, expert_map, local_num_experts,
        aq_out_f, scale_out_f, m_indices_f, inv_perm_f,
    )

    # Gather only routed experts. Shared experts are computed on their separate
    # model path and have inv_perm == -1 here.
    torch.manual_seed(5678)
    expert_output = torch.randn_like(aq_out_f, dtype=torch.bfloat16)
    output = torch.empty(
        (TOKENS, HIDDEN), dtype=torch.bfloat16, device=DEVICE
    )
    op.ep_gather(
        expert_output,
        ids,
        weights,
        inv_perm_f,
        expert_map,
        output,
    )
    torch.cuda.synchronize()

    reference = torch.zeros(
        (TOKENS, HIDDEN), dtype=torch.float32, device=DEVICE
    )
    _, valid_all = resolve_local_ids(ids, expert_map, local_num_experts)
    for k in range(ids.size(1)):
        valid = valid_all[:, k]
        src = inv_perm_f[valid, k].to(torch.int64)
        reference[valid] += (
            expert_output[src].float() * weights[valid, k].unsqueeze(1)
        )
    torch.testing.assert_close(
        output, reference.to(torch.bfloat16), rtol=2e-2, atol=2e-2
    )

    print(
        f"[PASS] {config['name']} ids={str(id_dtype).split('.')[-1]} "
        f"map={'yes' if use_expert_map else 'no'} K={ids.size(1)}"
    )


def main():
    for config in CONFIGS:
        for id_dtype in (torch.int32, torch.int64):
            for use_expert_map in (False, True):
                run_case(config, id_dtype, use_expert_map)
    print("[PASS] all EP scatter/INT8-quant-scatter/gather cases")


if __name__ == "__main__":
    main()
