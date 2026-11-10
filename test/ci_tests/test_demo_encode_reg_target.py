# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.profiler import ProfilerActivity
from lightop import op 
import time

def encode_reg_target(box_target, device=None):
    for box in box_target:
        output = torch.cat(
                [
                    box[..., [X, Y, Z]],
                    box[..., [W, L, H]].log(),
                    torch.sin(box[..., YAW]).unsqueeze(-1),
                    torch.cos(box[..., YAW]).unsqueeze(-1),
                    box[..., YAW + 1 :],
                ],
                dim=-1,
            )
        if device is not None:
            output = output.to(device=device)
        outputs_ref.append(output)
    return outputs_ref

def lightop_encode_reg_target(box_target_list, device=None):
    if len(box_target_list) == 0:
        return []

    dev = device if device is not None else box_target_list[0].device
    box_target_list = [t.to(dev) if t.device != dev else t for t in box_target_list]

    B = len(box_target_list)
    N = box_target_list[0].shape[0]
    D = box_target_list[0].shape[1]

    input_tensor = torch.stack(box_target_list, dim=0)  # [B, N, D]

    lengths_tensor = torch.tensor([N]*B, dtype=torch.int64, device=dev)

    out_all = op.encode_reg_t(input_tensor, lengths_tensor)

    outputs_lightop = [out_all[i, :N, :].contiguous() for i in range(B)]
    return outputs_lightop

if __name__ == '__main__':

    X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))
    YAW = 6

    box_target_len = 100
    box_pred_shape0 = 100
    N_dim = 64
    D_dim = 10

    box_pred = torch.randn(box_target_len, N_dim, D_dim, device='cuda:0')
    box_target = [torch.rand(32, D_dim, dtype=torch.float32, device='cuda:0') for _ in range(box_target_len)]

    num_warmup = 10
    num_runs = 100

    outputs_ref = []
    outputs_lightop = []

    for _ in range(num_warmup):
        out_ref = encode_reg_target(box_target, box_pred.device)

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_runs):
        out_ref = encode_reg_target(box_target, box_pred.device)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms_ref = start_event.elapsed_time(end_event)
    avg_time_ms_ref = elapsed_time_ms_ref / num_runs
    print(f"Running Python reference encode_reg_target 耗时: {avg_time_ms_ref:.4f} ms")

    for _ in range(num_warmup):
        out_lightop = lightop_encode_reg_target(box_target, box_pred.device)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_runs):
        out_lightop = lightop_encode_reg_target(box_target, box_pred.device)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms_lightop = start_event.elapsed_time(end_event)
    avg_time_ms_lightop = elapsed_time_ms_lightop / num_runs
    print(f"Lightop lightop_encode_reg_target 耗时: {avg_time_ms_lightop:.4f} ms")

    all_close = True
    max_diff = 0.0
    mean_diff = 0.0
    total_items = 0

    for i in range(box_target_len):
        diff_i = torch.abs(out_ref[i] - out_lightop[i])
        if not torch.allclose(out_ref[i], out_lightop[i], atol=1e-5):
            all_close = False
            print(f" Mismatch at batch item {i}!")
        max_diff = max(max_diff, diff_i.max().item())
        mean_diff += diff_i.mean().item()
        total_items += 1

    if all_close:
        print(" 精度验证通过！")
    else:
        print(" 精度验证失败！")
    print(f" 最大误差: {max_diff}")
    print(f" 平均误差: {mean_diff / total_items}")

    print("\nTest end.")