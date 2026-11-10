# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch import cuda
from lightop import op as ops

# --------------------------
# 工具函数
# --------------------------
def allclose(a, b, rtol=1e-2, atol=1e-2):
    return torch.allclose(a, b, rtol=rtol, atol=atol)

def rotate_half(x):
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)

def reshape_for_broadcast(freqs_cis, x, head_first=False):
    ndim = x.ndim
    if isinstance(freqs_cis, tuple):
        if head_first:
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)

# --------------------------
# 原始实现
# --------------------------
def original_mha(input_data, freqs_cis_img, Wqkv, 
                 gamma_q, beta_q, gamma_k, beta_k, eps=1e-6,
                 dim=1408, num_heads=16):
    #参数
    dim = dim
    num_heads = num_heads
    head_dim = dim // num_heads
    factory_kwargs = {'device': "cuda", 'dtype': torch.float16}

    q_norm=nn.LayerNorm(head_dim,  eps=1e-6,**factory_kwargs)
    k_norm=nn.LayerNorm(head_dim,  eps=1e-6,**factory_kwargs)

    #qkv获取 + norm计算
    b, s, d = input_data.shape
    qkv = Wqkv(input_data)
    
    qkv = qkv.view(b, s, 3, num_heads, head_dim)  # [b, s, 3, h, d]
    q, k, v = qkv.unbind(dim=2)  # [b, s, h, d]
    
    q = q_norm(q).half()  # [b, s, h, d]
    k = k_norm(k).half()
    
    head_first=False

    ndim = q.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis_img[0].shape == (
        q.shape[1],
        q.shape[-1],
        ), f"freqs_cis shape {freqs_cis_img[0].shape} does not match x shape {q.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(q.shape)]

    cos = freqs_cis_img[0].view(*shape)
    sin = freqs_cis_img[1].view(*shape)

    cos, sin = cos.to(q.device), sin.to(q.device)
    x_real, x_imag = (q.float().reshape(*q.shape[:-1], -1, 2).unbind(-1)) 
    rotate_data_q=torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        #rotate_half(q.float())
        #>

    q_out = (q.float() * cos +  rotate_data_q * sin).type_as(q)

    ##rotate_half(k.float())
    #<
    x_real, x_imag = (k.float().reshape(*k.shape[:-1], -1, 2).unbind(-1)) 
    rotate_data_k=torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    ##rotate_half(k.float())
    #>
    k_out = (k.float() * cos +  rotate_data_k * sin).type_as(k)
    
    assert (q_out.shape == q.shape and k_out.shape == k.shape
        ), f"qq: {q_out.shape}, q: {q.shape}, kk: {k_out.shape}, k: {k.shape}"
    q, k = q_out, k_out
  
    qkv = torch.stack([q, k, v], dim=2)  # [b, s, 3, h, d]
    return qkv

# --------------------------
# Custom 实现
# --------------------------
def custom_mha(input_data, freqs_cis_img, Wqkv,
               gamma_q, beta_q, gamma_k, beta_k, eps=1e-6,
               dim=1408, num_heads=16):
    head_dim = dim // num_heads
    b, s, d = input_data.shape

    qkv = Wqkv(input_data).view(b, s, 3, num_heads, head_dim).contiguous()

    cos, sin = freqs_cis_img
    cos = cos.to(qkv.device).contiguous()  # [S, D]
    sin = sin.to(qkv.device).contiguous()  # [S, D]

    ops.mha_norm_rope_qkv_inplace_cuda(
        qkv,          # arg0: Tensor [B,S,3,H,D], contiguous
        gamma_q,      # arg1: Optional[Tensor] or None, [D]
        beta_q,       # arg2: Optional[Tensor] or None, [D]
        gamma_k,      # arg3: Optional[Tensor] or None, [D]
        beta_k,       # arg4: Optional[Tensor] or None, [D]
        cos,          # arg5: Tensor float32 [S,D], CUDA, contiguous
        sin,          # arg6: Tensor float32 [S,D], CUDA, contiguous
        eps           # arg7: float
    )

    return qkv

# --------------------------
# 测试函数
# --------------------------
def benchmark_batchsizes(batchsizes, seqs=1024, dim=1408, repeat=50):
    head_dim = dim // 16
    factory_kwargs = {'device': "cuda", 'dtype': torch.float16}

    # 初始化 Wqkv 和 gamma/beta
    Wqkv = nn.Linear(dim, 3 * dim, bias=True, **factory_kwargs).cuda()
    gamma_q = torch.ones(head_dim, device='cuda', dtype=torch.float16)
    beta_q = torch.zeros(head_dim, device='cuda', dtype=torch.float16)
    gamma_k = torch.ones(head_dim, device='cuda', dtype=torch.float16)
    beta_k = torch.zeros(head_dim, device='cuda', dtype=torch.float16)
    eps = 1e-6

    # --- 预热部分 ---
    warmup_batch = 2
    input_data = torch.randn(warmup_batch, seqs, dim, device='cuda', dtype=torch.float16)
    freqs_cis_img = (
        torch.randn(seqs, head_dim, device='cuda', dtype=torch.float32),
        torch.randn(seqs, head_dim, device='cuda', dtype=torch.float32)
    )

    for _ in range(10):
        _ = original_mha(input_data.clone(), freqs_cis_img, Wqkv,  # <<< clone 确保一致 <<<
                         gamma_q, beta_q, gamma_k, beta_k, eps)
        _ = custom_mha(input_data.clone(), freqs_cis_img, Wqkv,   # <<< clone 确保一致 <<<
                       gamma_q, beta_q, gamma_k, beta_k, eps)
    torch.cuda.synchronize()
    # --- 预热结束 ---

    for batchsize in batchsizes:
        print(f"\nBatch size = {batchsize}")
        input_data = torch.randn(batchsize, seqs, dim, device='cuda', dtype=torch.float16)
        freqs_cis_img = (
            torch.randn(seqs, head_dim, device='cuda', dtype=torch.float32),
            torch.randn(seqs, head_dim, device='cuda', dtype=torch.float32)
        )

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # 原始实现
        start_event.record()
        for _ in range(repeat):
            out_orig = original_mha(input_data.clone(), freqs_cis_img, Wqkv,   # <<< clone 确保一致 <<<
                                    gamma_q, beta_q, gamma_k, beta_k, eps)
        end_event.record()
        end_event.synchronize()
        time_orig = start_event.elapsed_time(end_event) / repeat

        # 自定义实现
        start_event.record()
        for _ in range(repeat):
            out_custom = custom_mha(input_data.clone(), freqs_cis_img, Wqkv,   # <<< clone 确保一致 <<<
                                    gamma_q, beta_q, gamma_k, beta_k, eps)
        end_event.record()
        end_event.synchronize()
        time_custom = start_event.elapsed_time(end_event) / repeat

        # 精度检查（用同一个 input 生成的 out_orig/out_custom，因 clone 一致）
        fwd_close = allclose(out_orig, out_custom)
        print(f"  torch time: {time_orig:.3f} ms")
        print(f"  ops time: {time_custom:.3f} ms")
        print(f"  Forward close: {fwd_close}")


# --------------------------
# 主程序
# --------------------------
if __name__ == "__main__":
    batchsizes = [1, 2, 4, 8]
    benchmark_batchsizes(batchsizes)
