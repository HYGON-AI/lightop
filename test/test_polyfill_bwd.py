import torch
import pandas as pd
import time
import os

#给的基准代码，20000个顶点要用大约150kb的seme，HCU只有64KBseme可用，NUM_VERTICES大于8000，基准跑不通。lightop优化版本不会
try:
    from lightop import op
    print("Successfully imported lightop.")
except ImportError:
    print("Error: Could not import 'lightop'. Please ensure your extension is installed/compiled.")
    exit(1)

def run_benchmark():
    # ================= 配置参数 =================
    BATCH_SIZE = 1            
    NUM_VERTICES = 8000       
    RESOLUTIONS = [           
        (256, 256),
        (512, 512),
        (1024, 1024)
    ]
    INV_SMOOTHNESS = 10.0
    MODE = 1                  # MODE_MASK
    
    WARMUP_ROUNDS = 10
    TEST_ROUNDS = 100
    OUTPUT_FILE = "benchmark_results.xlsx"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("Error: CUDA not available.")
        return

    results = []

    print(f"\nConfiguration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Num Vertices: {NUM_VERTICES}")
    print(f"  Warmup: {WARMUP_ROUNDS}, Test: {TEST_ROUNDS}")
    print(f"  Device: {torch.cuda.get_device_name(0)}\n")

    for height, width in RESOLUTIONS:
        print(f"Running benchmark for Resolution: {height}x{width} ...")

        # ================= 准备数据 =================
        # Vertices: [B, N, 2]
        vertices = torch.rand((BATCH_SIZE, NUM_VERTICES, 2), device=device, dtype=torch.float32)
        vertices[..., 0] *= width
        vertices[..., 1] *= height
        
        # Output map: [B, H*W] (flattened as per kernel logic usually, or [B, H, W])
        # C++ kernel treats it linearly, PyTorch tensor can be [B, H, W]
        output_map = torch.rand((BATCH_SIZE, height, width), device=device, dtype=torch.float32)
        
        # Nearest Edge Map: [B, H, W], indices
        nearest_edge_map = torch.randint(0, NUM_VERTICES, (BATCH_SIZE, height, width), device=device, dtype=torch.int32)
        
        # Grad Output: [B, H, W]
        grad_output = torch.randn((BATCH_SIZE, height, width), device=device, dtype=torch.float32)

        # Grad Vertices buffers (Need to be zeroed before accumulation if the kernel adds to it)
        # Based on your C++ code: atomicAdd is used, so we must pass a zero-filled tensor.
        grad_vertices_base = torch.zeros_like(vertices)
        grad_vertices_opt = torch.zeros_like(vertices)

        # ================= 1. 精度验证 (Correctness Check) =================
        
        # Run Baseline
        if NUM_VERTICES <= 8000:
            op.backward_polyfill_cuda(
                vertices, output_map, nearest_edge_map, grad_output, grad_vertices_base, 
                width, height, INV_SMOOTHNESS, MODE
            )
        
        # Run Optimized
        op.backward_polyfill_lightop(
            vertices, output_map, nearest_edge_map, grad_output, grad_vertices_opt, 
            width, height, INV_SMOOTHNESS, MODE
        )

        # Compare
        diff = (grad_vertices_base - grad_vertices_opt).abs()
        relative_error = diff / (torch.abs(grad_vertices_base) + 1e-8)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        is_close = torch.allclose(grad_vertices_base, grad_vertices_opt, atol=1e-4, rtol=1e-4)
        # import pdb
        # pdb.set_trace()
        print(f"  Accuracy Check -> Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}, Passed: {is_close}")
        print(f"Max Relative Diff: {relative_error.max().item()}")
        # ================= 2. 性能测试 (Performance Benchmark) =================
        
        # Helper to time function
        def benchmark_op(func, target_grad_tensor, name):
            # Warmup
            for _ in range(WARMUP_ROUNDS):
                target_grad_tensor.zero_() # Reset grad
                func(vertices, output_map, nearest_edge_map, grad_output, target_grad_tensor, width, height, INV_SMOOTHNESS, MODE)
            
            torch.cuda.synchronize()
            
            # Timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(TEST_ROUNDS):
                target_grad_tensor.zero_() # Reset grad mainly to simulate real usage, though cost is small
                func(vertices, output_map, nearest_edge_map, grad_output, target_grad_tensor, width, height, INV_SMOOTHNESS, MODE)
            end_event.record()
            
            torch.cuda.synchronize()
            total_time_ms = start_event.elapsed_time(end_event)
            avg_time_us = (total_time_ms / TEST_ROUNDS) * 1000.0
            return avg_time_us

        if NUM_VERTICES <= 8000:
            time_base = benchmark_op(op.backward_polyfill_cuda, grad_vertices_base, "Baseline")
        else:
            time_base = 0   
        
        # Measure Optimized
        time_opt = benchmark_op(op.backward_polyfill_lightop, grad_vertices_opt, "Optimized")
        
        speedup = time_base / time_opt if time_opt > 0 else 0.0
        
        print(f"  Time (us) -> Baseline: {time_base:.2f}, Opt: {time_opt:.2f}, Speedup: {speedup:.2f}x")

        results.append({
            "Resolution": f"{height}x{width}",
            "Vertices": NUM_VERTICES,
            "Baseline (us)": time_base,
            "Optimized (us)": time_opt,
            "Speedup": speedup,
            "Max Diff": max_diff,
            "Passed": is_close
        })

    # ================= 3. 保存结果 (Save to Excel) =================
    df = pd.DataFrame(results)
    
    # 格式化一下显示
    print("\nSummary:")
    print(df)
    
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_benchmark()