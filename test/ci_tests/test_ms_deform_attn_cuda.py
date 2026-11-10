# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import os
from lightop import op as ops


# ============================================================
# 纯 PyTorch 参考实现：multi_scale_deformable_attn_pytorch
# （替换为 mmcv 官方等价实现）
# ============================================================
def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    CPU version of Multi-Scale Deformable Attention
    与 mmcv.ops.multi_scale_deform_attn_pytorch 逻辑等价
    用于正确性验证
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode='bilinear',
            padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(
                  bs, num_heads * embed_dims, num_queries)

    return output.transpose(1, 2).contiguous()


# ============================================================
# GPU性能基准测试 + 正确性验证
# ============================================================
def benchmark_gpu_timing_and_correctness(num_runs=50):
    device = torch.device("cuda")

    # 使用实际测试尺寸 1
    B = 2
    H = 8
    C_per_head = int(os.getenv("C_PER_HEAD", 32))
    print(f"当前使用的 C_per_head = {C_per_head}")
    embed_dim = H * C_per_head  # 256

    # 只有一个 level, 大小 200x200
    value_spatial_shapes = torch.tensor([[200, 200]], dtype=torch.long)
    level_start_index = torch.tensor([0], dtype=torch.long)

    S = (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).sum().item()  # 40000
    L = value_spatial_shapes.shape[0]  # 1
    P = 4
    Lq = S  # 40000
    im2col_step = 64


    # # 使用实际测试尺寸 2
    # B = 2
    # H = 8
    # C_per_head = int(os.getenv("C_PER_HEAD", 32))
    # print(f"当前使用的 C_per_head = {C_per_head}")
    # embed_dim = H * C_per_head  # 256

    # # 多尺度：3 个 level
    # value_spatial_shapes = torch.tensor([[32, 32], [64, 64], [128, 128]], dtype=torch.long, device=device)
    # level_start_index = torch.cat([
    #     value_spatial_shapes.new_zeros(1),
    #     (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).cumsum(0)[:-1]
    # ])

    # S = (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).sum().item()  # 21504
    # L = value_spatial_shapes.shape[0]  # 3
    # P = 4
    # Lq = S  # 21504
    # im2col_step = 64

    # # 使用实际测试尺寸 3
    # B = 6
    # H = 8
    # C_per_head = 32
    # embed_dim = H * C_per_head  # 256

    # # 多尺度：4 个 level
    # value_spatial_shapes = torch.tensor([
    #     [32, 32], 
    #     [64, 64], 
    #     [128, 128], 
    #     [49, 49]   # 举例，保证 sum(H*W)=30825
    # ], dtype=torch.long, device=device)

    # level_start_index = torch.cat([
    #     value_spatial_shapes.new_zeros(1),
    #     (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).cumsum(0)[:-1]
    # ])

    # S = (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).sum().item()  # 30825
    # L = value_spatial_shapes.shape[0]  # 4

    # P = 4
    # Lq = 9703
    # im2col_step = 64

    # 数据准备
    torch.manual_seed(3)
    value_base = torch.rand(B, S, H, C_per_head) * 0.01
    sampling_locations_base = torch.rand(B, Lq, H, L, P, 2)
    attention_weights_base = torch.rand(B, Lq, H, L, P) + 1e-5
    attention_weights_base = attention_weights_base / (
        attention_weights_base.sum(-1, keepdim=True).sum(-2, keepdim=True)
    )

    value = value_base.clone().to(device).requires_grad_(True)
    sampling_locations = sampling_locations_base.clone().to(device).requires_grad_(True)
    attention_weights = attention_weights_base.clone().to(device).requires_grad_(True)
    shapes_gpu = value_spatial_shapes.to(device)
    level_start_index_gpu = level_start_index.to(device)

    # 前向预热
    for _ in range(5):
        _ = ops.ms_deform_attn_forward(value, shapes_gpu, level_start_index_gpu,
                                       sampling_locations, attention_weights, im2col_step)
    torch.cuda.synchronize()

    # ---------------- ops 前向计时 ----------------
    forward_times = []
    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        _ = ops.ms_deform_attn_forward(value, shapes_gpu, level_start_index_gpu,
                                       sampling_locations, attention_weights, im2col_step)
        end_event.record()
        torch.cuda.synchronize()
        forward_times.append(start_event.elapsed_time(end_event))
    print(f"[ops] 前向平均时间: {sum(forward_times)/len(forward_times):.3f} ms")

    # ---------------- ops 反向计时 ----------------
    backward_times = []
    for _ in range(num_runs):
        if value.grad is not None:
            value.grad.zero_()
        if sampling_locations.grad is not None:
            sampling_locations.grad.zero_()
        if attention_weights.grad is not None:
            attention_weights.grad.zero_()
        output = ops.ms_deform_attn_forward(value, shapes_gpu, level_start_index_gpu,
                                           sampling_locations, attention_weights, im2col_step)
        grad_output = torch.ones_like(output)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        ops.ms_deform_attn_backward(value, shapes_gpu, level_start_index_gpu,
                                    sampling_locations, attention_weights,
                                    grad_output,
                                    grad_value := torch.zeros_like(value),
                                    grad_sampling_loc := torch.zeros_like(sampling_locations),
                                    grad_attn_weight := torch.zeros_like(attention_weights),
                                    im2col_step)
        end_event.record()
        torch.cuda.synchronize()
        backward_times.append(start_event.elapsed_time(end_event))
    print(f"[ops] 反向平均时间: {sum(backward_times)/len(backward_times):.3f} ms")

    # ==================== 前向正确性对比 ====================
    gpu_output = ops.ms_deform_attn_forward(value, shapes_gpu, level_start_index_gpu,
                                            sampling_locations, attention_weights, im2col_step).detach().cpu()
    pytorch_output = multi_scale_deformable_attn_pytorch(
        value.detach().cpu(), value_spatial_shapes,
        sampling_locations.detach().cpu(),
        attention_weights.detach().cpu())

    print(f"前向输出匹配: {'✓' if torch.allclose(gpu_output, pytorch_output, rtol=1e-4, atol=1e-4) else '✗'}")

    # ==================== 反向正确性对比 ====================
    grad_value = torch.zeros_like(value)
    grad_sampling_loc = torch.zeros_like(sampling_locations)
    grad_attn_weight = torch.zeros_like(attention_weights)

    output = ops.ms_deform_attn_forward(
        value, shapes_gpu, level_start_index_gpu,
        sampling_locations, attention_weights, im2col_step)
    grad_output = torch.ones_like(output)

    ops.ms_deform_attn_backward(
        value, shapes_gpu, level_start_index_gpu,
        sampling_locations, attention_weights,
        grad_output,
        grad_value, grad_sampling_loc, grad_attn_weight,
        im2col_step)

    value_ref = value.detach().cpu().clone().requires_grad_(True)
    sampling_ref = sampling_locations.detach().cpu().clone().requires_grad_(True)
    attn_ref = attention_weights.detach().cpu().clone().requires_grad_(True)

    out_ref = multi_scale_deformable_attn_pytorch(
        value_ref, value_spatial_shapes,
        sampling_ref, attn_ref)
    out_ref.backward(torch.ones_like(out_ref))

    def check_grad(name, grad, grad_ref):
        diff = grad.cpu() - grad_ref
        max_err = diff.abs().max().item()
        mean_err = diff.abs().mean().item()
        ok = torch.allclose(grad.cpu(), grad_ref, rtol=1e-4, atol=1e-4)
        print(f"{name}: {'✓' if ok else '✗'} "
              f"(max_err={max_err:.3e}, mean_err={mean_err:.3e})")
        return ok

    ok_value = check_grad("grad_value", grad_value, value_ref.grad)
    ok_sampling = check_grad("grad_sampling_loc", grad_sampling_loc, sampling_ref.grad)
    ok_attn = check_grad("grad_attn_weight", grad_attn_weight, attn_ref.grad)

    print(f"反向总体结果: {'✓' if (ok_value and ok_sampling and ok_attn) else '✗'}")



if __name__ == "__main__":
    benchmark_gpu_timing_and_correctness()
