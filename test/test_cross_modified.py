from typing import Tuple, Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightop import op as ops


# ---- 工具函数 ----
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    xr, xi = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-xi, xr], dim=-1).flatten(3)

def reshape_for_broadcast(
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    x: torch.Tensor,
    head_first: bool = False,
):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if isinstance(freqs_cis, tuple):
        if head_first:
            assert freqs_cis[0].shape == (x.shape[-2], x.shape[-1]), \
                f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1
                     for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis[0].shape == (x.shape[1], x.shape[-1]), \
                f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1
                     for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    raise ValueError("This helper expects (cos, sin) tuple for freqs_cis.")

def original_cross_mha(
    x: torch.Tensor,                    # [B, S1, qdim]
    y: torch.Tensor,                    # [B, S2, kdim]
    freqs_cis_img: Tuple[torch.Tensor, torch.Tensor],  # (cos[S1,D], sin[S1,D]) fp32
    q_proj: nn.Linear,
    kv_proj: nn.Linear,
    gamma_q: torch.Tensor, beta_q: torch.Tensor,       # [D]
    gamma_k: torch.Tensor, beta_k: torch.Tensor,       # [D]
    eps: float = 1e-6,
    qdim: int = 1408, kdim: int = 1024, num_heads: int = 16,
):
    head_dim = qdim // num_heads
    B, S1, _ = x.shape
    _, S2, _ = y.shape

    q = q_proj(x).view(B, S1, num_heads, head_dim)      # [B,S1,H,D]
    kv = kv_proj(y).view(B, S2, 2, num_heads, head_dim) # [B,S2,2,H,D]
    k, v = kv.unbind(dim=2)                              # [B,S2,H,D], [B,S2,H,D]

    # LayerNorm（用 1D γ/β，与内核一致）
    def ln_affine(t, g, b):
        return F.layer_norm(t.float(), (head_dim,),
                            weight=g.float(), bias=b.float(),
                            eps=eps).to(t.dtype)

    q = ln_affine(q, gamma_q, beta_q)
    k = ln_affine(k, gamma_k, beta_k)

    # RoPE 只用于 Q（cos/sin: [S1,D] -> [1,S1,1,D]）
    cos, sin = freqs_cis_img
    cos = cos.to(q.device, dtype=torch.float32).view(1, S1, 1, head_dim)
    sin = sin.to(q.device, dtype=torch.float32).view(1, S1, 1, head_dim)
    q_rot = (q.float() * cos + rotate_half(q) * sin).to(q.dtype)

    kv_out = torch.stack([k, v], dim=2)  # [B,S2,2,H,D]
    return kv_out, q_rot                 # (kv, q)

def custom_cross_mha(
    x: torch.Tensor,
    y: torch.Tensor,
    freqs_cis_img: Tuple[torch.Tensor, torch.Tensor],
    q_proj: nn.Linear,
    kv_proj: nn.Linear,
    gamma_q: torch.Tensor, beta_q: torch.Tensor,
    gamma_k: torch.Tensor, beta_k: torch.Tensor,
    eps: float = 1e-6,
    qdim: int = 1408, kdim: int = 1024, num_heads: int = 16,
):
    head_dim = qdim // num_heads
    B, S1, _ = x.shape
    _, S2, _ = y.shape

    q  = q_proj(x).view(B, S1, num_heads, head_dim).contiguous()          # [B,S1,H,D]
    kv = kv_proj(y).view(B, S2, 2, num_heads, head_dim).contiguous()      # [B,S2,2,H,D]

    cos, sin = freqs_cis_img
    cos = cos.to(q.device).contiguous()  # [S1,D]
    sin = sin.to(q.device).contiguous()  # [S1,D]

    ops.cross_norm_rope_q_kv_inplace_cuda(
        q,     # [B,S1,H,D]
        kv,    # [B,S2,2,H,D]
        gamma_q, beta_q,
        gamma_k, beta_k,
        cos, sin,
        eps
    )
    return kv, q

# ---- 基准测试：比较原始 vs 融合 的耗时 & 精度 ----
def benchmark_batchsizes(
    batchsizes: List[int],
    S1: int = 1024, S2: int = 333,
    qdim: int = 1408, kdim: int = 1024,
    num_heads: int = 16, repeat: int = 50,
):
    device = "cuda"
    dtype = torch.float16
    torch.manual_seed(0)

    head_dim = qdim // num_heads
    factory_kwargs = {'device': "cuda", 'dtype': torch.float16}

    # 统一的投影层（两边共享权重）
    q_proj = nn.Linear(qdim, qdim, bias=True, **factory_kwargs).to(device)
    kv_proj = nn.Linear(kdim, 2 * qdim, bias=True, **factory_kwargs).to(device)

    # LN 仿射参数（1D 向量）
    gamma_q = torch.ones(head_dim, device=device, dtype=dtype)
    beta_q  = torch.zeros(head_dim, device=device, dtype=dtype)
    gamma_k = torch.ones(head_dim, device=device, dtype=dtype)
    beta_k  = torch.zeros(head_dim, device=device, dtype=dtype)

    # 预热
    warm_bs = 2
    x = torch.randn(warm_bs * 2, S1, qdim, device=device, dtype=dtype)
    y = torch.randn(warm_bs * 2, S2, kdim, device=device, dtype=dtype)
    freqs_cis_img = (torch.randn([1024, 88],dtype=torch.float32,device="cuda"),torch.randn([1024, 88],dtype=torch.float32,device="cuda"))

    for _ in range(10):
        _ = original_cross_mha(x, y, freqs_cis_img,
                               q_proj, kv_proj,
                               gamma_q, beta_q, gamma_k, beta_k,
                               eps=1e-6, qdim=qdim, kdim=kdim, num_heads=num_heads)
        _ = custom_cross_mha(x, y, freqs_cis_img,
                             q_proj, kv_proj,
                             gamma_q, beta_q, gamma_k, beta_k,
                             eps=1e-6, qdim=qdim, kdim=kdim, num_heads=num_heads)
    torch.cuda.synchronize()

    for bs in batchsizes:
        B = bs * 2  # 第一维为 bs*2
        print(f"\nBatch size= {bs}")
        x = torch.randn(B, S1, qdim, device=device, dtype=dtype)
        y = torch.randn(B, S2, kdim, device=device, dtype=dtype)
        freqs_cis_img = (torch.randn([1024, 88],dtype=torch.float32,device="cuda"),torch.randn([1024, 88],dtype=torch.float32,device="cuda"))


        # 原始
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        ed = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(repeat):
            kv_ref, q_ref = original_cross_mha(x.clone(), y.clone(), freqs_cis_img,
                                               q_proj, kv_proj,
                                               gamma_q, beta_q, gamma_k, beta_k,
                                               eps=1e-6, qdim=qdim, kdim=kdim, num_heads=num_heads)
        ed.record(); ed.synchronize()
        t_ref = st.elapsed_time(ed) / repeat

        # 融合
        st.record()
        for _ in range(repeat):
            kv_fused, q_fused = custom_cross_mha(x.clone(), y.clone(), freqs_cis_img,
                                                 q_proj, kv_proj,
                                                 gamma_q, beta_q, gamma_k, beta_k,
                                                 eps=1e-6, qdim=qdim, kdim=kdim, num_heads=num_heads)
        ed.record(); ed.synchronize()
        t_fused = st.elapsed_time(ed) / repeat

        # 精度
        close_q  = torch.allclose(q_ref, q_fused, rtol=1e-2, atol=1e-2)
        close_kv = torch.allclose(kv_ref, kv_fused, rtol=1e-2, atol=1e-2)

        print(f"  torch time: {t_ref:.3f} ms")
        print(f"  ops time: {t_fused:.3f} ms")
        print(f"  allclose -> q:{close_q}  kv:{close_kv}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available; skip.")
    else:
        batchsizes = [1, 2, 4, 8]
        benchmark_batchsizes(batchsizes)