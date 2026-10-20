# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl
from lightop import op  # 导入你的自定义 CUDA 算子

@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr, block_table_ptr, token_indices_ptr, out_ptr,
    prefill_request_id_ptr, workspace_starts_ptr,
    max_num_blocks_per_req: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr, HAS_PREFILL: tl.constexpr,
    bt_stride0, bt_stride1, ti_stride0, ti_stride1, out_stride0, out_stride1,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    req = tl.load(req_id_ptr + token_id)

    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)

    is_invalid_tok = tok < 0
    is_prefill = False
    if HAS_PREFILL:
        prefill_req_id = tl.load(prefill_request_id_ptr + token_id)
        is_prefill = prefill_req_id >= 0

    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    is_invalid_tok |= ~valid_block
    base = tl.load(bt_ptr, mask=valid_block & ~is_prefill, other=0)
    out_val = base * BLOCK_SIZE + inblock_off

    if HAS_PREFILL:
        workspace_start = tl.load(
            workspace_starts_ptr + prefill_req_id, mask=is_prefill, other=0
        )
        prefill_out = workspace_start + tok
        out_val = tl.where(is_prefill, prefill_out, out_val)
    
    out_val = tl.where(is_invalid_tok, -1, out_val)

    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)

def triton_convert_req_index_to_global_index(
    req_id, block_table, token_indices, BLOCK_SIZE=64, NUM_TOPK_TOKENS=2048, 
    BLOCK_N=128, HAS_PREFILL_WORKSPACE=False, prefill_workspace_request_ids=None, prefill_workspace_starts=None
):
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    grid = (num_tokens, tiles_per_row)
    _convert_req_index_to_global_index_kernel[grid](
        req_id_c, block_table_c, token_indices_c, out,
        prefill_workspace_request_ids, prefill_workspace_starts,
        max_num_blocks_per_req, BLOCK_SIZE, BLOCK_N, HAS_PREFILL_WORKSPACE,
        bt_stride0, bt_stride1, ti_stride0, ti_stride1, out_stride0, out_stride1,
    )
    return out

# =========================================================================
# 2. 性能测试工具函数
# =========================================================================
def benchmark_latency(func, *args, **kwargs):
    # 预热 10 轮
    for _ in range(10):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    
    # 计测 100 轮
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    
    start_evt.record()
    for _ in range(100):
        func(*args, **kwargs)
    end_evt.record()
    torch.cuda.synchronize()
    
    # ms 转换为 us，并取平均
    return (start_evt.elapsed_time(end_evt) * 1000) / 100.0


def run_benchmark():
    print("### DeepSeek V3.2 算子性能与精度验证报告\n")
    print("| 阶段 | 场景/规模 (`num_tokens`) | `top_k` | 精度验证 | Triton 基准 (us) | CUDA 优化 (us) | 加速比 |")
    print("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |")


    BLOCK_SIZE = 64
    MAX_NUM_BLOCKS = 2048 
    NUM_TOPK = 2048        


    test_cases = [
        {"phase": "Prefill", "num_tokens": 4096, "num_requests": 1, "has_prefill": True},
        {"phase": "Prefill", "num_tokens": 8192, "num_requests": 1, "has_prefill": True},
        {"phase": "Decode", "num_tokens": 128, "num_requests": 128, "has_prefill": False},
        {"phase": "Decode", "num_tokens": 256, "num_requests": 256, "has_prefill": False},
    ]

    for tc in test_cases:
        phase = tc["phase"]
        num_tokens = tc["num_tokens"]
        num_requests = tc["num_requests"]
        has_prefill = tc["has_prefill"]

        req_id = torch.randint(0, num_requests, (num_tokens,), dtype=torch.int32, device='cuda')
        block_table = torch.randint(0, 50000, (num_requests, MAX_NUM_BLOCKS), dtype=torch.int32, device='cuda')
        
        if has_prefill:
            token_indices = torch.randint(-1, num_tokens, (num_tokens, NUM_TOPK), dtype=torch.int32, device='cuda')
            prefill_workspace_request_ids = torch.zeros((num_tokens,), dtype=torch.int32, device='cuda')
            prefill_workspace_starts = torch.zeros((1,), dtype=torch.int32, device='cuda')
        else:
            token_indices = torch.randint(-1, MAX_NUM_BLOCKS * BLOCK_SIZE, (num_tokens, NUM_TOPK), dtype=torch.int32, device='cuda')
            prefill_workspace_request_ids = None
            prefill_workspace_starts = None

        out_triton = triton_convert_req_index_to_global_index(
            req_id, block_table, token_indices, 
            BLOCK_SIZE=BLOCK_SIZE, NUM_TOPK_TOKENS=NUM_TOPK, BLOCK_N=128,
            HAS_PREFILL_WORKSPACE=has_prefill,
            prefill_workspace_request_ids=prefill_workspace_request_ids,
            prefill_workspace_starts=prefill_workspace_starts
        )
        
        out_cuda = op.convert_req_index_to_global_index(
            req_id, block_table, token_indices, 
            BLOCK_SIZE, NUM_TOPK,
            has_prefill, 
            prefill_workspace_request_ids if has_prefill else None, 
            prefill_workspace_starts if has_prefill else None
        )

        is_correct = torch.equal(out_triton, out_cuda)
        status_icon = "✅ 通过" if is_correct else "❌ 失败"

        if not is_correct:
            print(f"| {phase} | {num_tokens} | {NUM_TOPK} | {status_icon} | N/A | N/A | N/A |")
            continue

        def call_triton():
            return triton_convert_req_index_to_global_index(
                req_id, block_table, token_indices, 
                BLOCK_SIZE=BLOCK_SIZE, NUM_TOPK_TOKENS=NUM_TOPK, BLOCK_N=128,
                HAS_PREFILL_WORKSPACE=has_prefill,
                prefill_workspace_request_ids=prefill_workspace_request_ids,
                prefill_workspace_starts=prefill_workspace_starts
            )

        def call_cuda():
            return op.convert_req_index_to_global_index(
                req_id, block_table, token_indices, 
                BLOCK_SIZE, NUM_TOPK,
                has_prefill, 
                prefill_workspace_request_ids if has_prefill else None, 
                prefill_workspace_starts if has_prefill else None
            )

        time_triton = benchmark_latency(call_triton)
        time_cuda = benchmark_latency(call_cuda)
        
        speedup = time_triton / time_cuda if time_cuda > 0 else 0

        phase_icon = "🟢 Prefill" if phase == "Prefill" else "🔵 Decode"
        
        print(f"| {phase_icon} | `{num_tokens}` | `{NUM_TOPK}` | {status_icon} | {time_triton:.2f} | **{time_cuda:.2f}** | **{speedup:.2f}x** |")

if __name__ == "__main__":
    run_benchmark()