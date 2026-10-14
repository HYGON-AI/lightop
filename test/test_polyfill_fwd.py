# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import pandas as pd
import time
from lightop import op
#给的基准代码，20000个顶点要用大约150kb的seme，HCU只有64KBseme可用，NUM_VERTICES大于8000，基准跑不通。lightop优化版本不会
# ================= 配置参数 =================
BATCH_SIZE = 1


NUM_VERTICES = 8000 

RESOLUTIONS = [
    (256, 256),
    (512, 512),
    (1024, 1024)
]
# 0: BOUNDARY, 1: MASK, 2: HARD_MASK
MODES = [0, 1, 2] 
MODE_NAMES = {0: "Boundary", 1: "Mask", 2: "Hard_Mask"}

INV_SMOOTHNESS = 1.0
WARMUP_ROUNDS = 10
TEST_ROUNDS = 100

def run_benchmark():
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("Error: CUDA is not available.")
        return

    device = torch.device("cuda:0")
    print(f"Running on device: {torch.cuda.get_device_name(device)}")
    print(f"Vertices: {NUM_VERTICES}, Batch Size: {BATCH_SIZE}")
    print("-" * 80)

    results = []

    for height, width in RESOLUTIONS:
        for mode in MODES:
            mode_name = MODE_NAMES[mode]
            print(f"Res: [{height}x{width}], Mode: {mode_name:<10} | ", end="")

            # ================= 1. 准备数据 =================
            # 顶点坐标 (B, N, 2)
            vertices = torch.rand((BATCH_SIZE, NUM_VERTICES, 2), device=device, dtype=torch.float32)
            vertices[:, :, 0] *= width
            vertices[:, :, 1] *= height
            
            # 为两个接口准备独立的输出 buffer，防止互相污染
            out_cuda = torch.zeros((BATCH_SIZE, height, width), device=device, dtype=torch.float32)
            map_cuda = torch.zeros((BATCH_SIZE, height, width), device=device, dtype=torch.int32)
            
            out_opt = torch.zeros((BATCH_SIZE, height, width), device=device, dtype=torch.float32)
            map_opt = torch.zeros((BATCH_SIZE, height, width), device=device, dtype=torch.int32)

            # ================= 2. 正确性校验 (Validation) =================
            # 运行基准版本
            if NUM_VERTICES <= 8000:
                op.forward_polyfill_cuda(
                    vertices, out_cuda, map_cuda, width, height, INV_SMOOTHNESS, mode
                )
            # 运行优化版本
            op.forward_polyfill_lightop(
                vertices, out_opt, map_opt, width, height, INV_SMOOTHNESS, mode
            )
            
            # 检查误差
            diff_abs = (out_cuda - out_opt).abs()
            max_diff = diff_abs.max().item()
            mean_diff = diff_abs.mean().item()
            is_correct = max_diff < 1e-4  
            # import pdb
            # pdb.set_trace()
            
            status_str = "PASS" if is_correct else "FAIL"
            print(f"Check: {status_str} (Diff: {max_diff:.1e}) | ", end="")

            # ================= 3. 性能测试: CUDA Base =================
            # 预热
            if NUM_VERTICES <= 8000:
                for _ in range(WARMUP_ROUNDS):
                    op.forward_polyfill_cuda(vertices, out_cuda, map_cuda, width, height, INV_SMOOTHNESS, mode)
                
                torch.cuda.synchronize()
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                
                start_evt.record()
                for _ in range(TEST_ROUNDS):
                    op.forward_polyfill_cuda(vertices, out_cuda, map_cuda, width, height, INV_SMOOTHNESS, mode)
                end_evt.record()
                torch.cuda.synchronize()
                
                cuda_time_us = (start_evt.elapsed_time(end_evt) / TEST_ROUNDS) * 1000
            else:
                cuda_time_us = 0  

            # ================= 4. 性能测试: Opt =================
            # 预热
            for _ in range(WARMUP_ROUNDS):
                op.forward_polyfill_lightop(vertices, out_opt, map_opt, width, height, INV_SMOOTHNESS, mode)
            
            torch.cuda.synchronize()
            start_evt_opt = torch.cuda.Event(enable_timing=True)
            end_evt_opt = torch.cuda.Event(enable_timing=True)
            
            start_evt_opt.record()
            for _ in range(TEST_ROUNDS):
                op.forward_polyfill_lightop(vertices, out_opt, map_opt, width, height, INV_SMOOTHNESS, mode)
            end_evt_opt.record()
            torch.cuda.synchronize()
            
            opt_time_us = (start_evt_opt.elapsed_time(end_evt_opt) / TEST_ROUNDS) * 1000

            print(f"Base: {cuda_time_us:.1f}us, Opt: {opt_time_us:.1f}us")

            # ================= 5. 记录结果 =================
            results.append({
                "Resolution": f"{height}x{width}",
                "H": height,
                "W": width,
                "Mode": mode_name,
                "Valid": is_correct,
                "Max Diff": max_diff,
                "Base Time (us)": cuda_time_us,
                "Opt Time (us)": opt_time_us,
                "Speedup": cuda_time_us / opt_time_us if opt_time_us > 0 else 0
            })

    # ================= 6. 保存到 Excel =================
    df = pd.DataFrame(results)
    output_file = "polyfill_benchmark_compare.xlsx"
    df.to_excel(output_file, index=False)
    print("-" * 80)
    print(f"Benchmark finished. Results saved to {output_file}")
    
  
    print("\nSummary (Average Speedup):")
    print(df.groupby("Mode")["Speedup"].mean())

if __name__ == "__main__":
    run_benchmark()