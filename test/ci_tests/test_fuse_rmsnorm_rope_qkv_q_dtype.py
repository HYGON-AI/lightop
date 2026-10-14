# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""单测 + 跑分：commit 6d99d17 "derive q and KV cache fp8 formats from their own buffers"。

该 commit 修的是两个**互相独立**的 bug，本文件对两个都给出判定性用例：

Bug 1 —— q 的 dtype 跟着 KV cache 走
    q 直接交给 attention，从不进 KV cache，dtype 本应由 output_dtype 决定。
    修复前 `q_rope_out` 按 `v_buffer.dtype` 分配，kernel 侧也用 cache_fp8_type
    决定 q 的输出格式，于是「fp8 cache」强行把 q 也变成 fp8。消费端若按模型
    dtype 读（如 gfx936，sglang 的 is_nmz_fp8() 为 False），就会把 fp8 字节
    当 bf16 解释 —— 不是精度损失，是整片数值全错。
    判据见 test_q_dtype_follows_output_dtype / test_q_values_are_not_fp8_quantised。

Bug 2 —— e5m2 的 cache 被写成 e4m3 的位模式
    wrapper 里的 `use_fp8_e4m3` 语义只是「cache 是不是 fp8」（它在 Python 侧
    算的是「不是 fp8」，传参时取反），kernel 却用 `cache_fp8_type =
    use_fp8_e4m3 ? 1 : 0`，等价于「只要是 fp8 就按 e4m3 编码」。于是 e5m2
    的 buffer 里装的是 e4m3 的字节。两种格式指数位宽不同（e5m2 max=57344，
    e4m3 max=448），同一个数的编码完全不同：
        1024.0 -> e5m2 字节 0x64，e4m3 字节 0x7e；用 e5m2 读 0x7e 得到 NaN
         256.0 -> e5m2 字节 0x5c，e4m3 字节 0x78；用 e5m2 读 0x78 得到 32768.0
    判据见 test_e5m2_cache_is_not_encoded_as_e4m3 —— 用落在 (448, 57344] 区间的
    值，e5m2 能表示而 e4m3 不能，修复前必然溢出成 NaN。

跑分部分：6 组配置 = 3 种 KV cache dtype (bf16 / fp8_e5m2 / fp8_e4m3fn) × 2 种
rot_dim (128 / 64)，q 固定 bf16（node1 不支持 fp8 的 q 输出）。每组都拿融合算子
对**纯 torch 参考实现** baseline_rms_norm_rope_kv_store 比误差和耗时。

关于 CSV 的误差列：
    误差是要被报告的量，不是只在失败时才可见的量 —— 只写一个 Pass/Fail，读表的人
    无从判断 max_diff 是 1e-3 还是正好卡在容差边上。所以每行都带上 max/mean 以及
    当时生效的 atol/rtol。

    Q 和 KV 分开两组：q 是 bf16，k/v 落 cache 走 cache dtype，两者容差不同
    （见 tolerance_for），混在一列会让人拿 fp8 的尺度去读 bf16 的数。
    KV 列取 k/v/k_buf/v_buf 四者中最差的一个，任何一处退化都不会被平均掉。

    同时给 max 和 mean：max 对单个异常点敏感（fp8 的 1-ULP 舍入方向差异就能把它
    顶起来），mean 反映整体分布，两个一起看才判断得了「是回归还是正常量化误差」。

关于 CSV 的 Path 列（重要）：
    wrapper 里 `if (head_dim > 128 or 非fp8cache)` 走的是【非 fused 回退分支】
    （单独调 rms_rotary_embedding_fuse，再用 torch 索引写 KV）。所以 bf16 cache
    根本进不了 fused kernel，只有 fp8 cache 才走 fused。两条路径耗时不可直接
    比较，CSV 用 Path 列标注。本文件 head_dim 固定 128，判别条件退化为
    「cache 是 fp8 -> fused」。

Run:
    pytest test/ci_tests/test_fuse_rmsnorm_rope_qkv_q_dtype.py -v   # 只跑单测
    python3 test/ci_tests/test_fuse_rmsnorm_rope_qkv_q_dtype.py     # 单测 CSV + 跑分 CSV
    python3 test/ci_tests/test_fuse_rmsnorm_rope_qkv_q_dtype.py --regression-only
    python3 test/ci_tests/test_fuse_rmsnorm_rope_qkv_q_dtype.py --tokens 128 4096

输出两个 CSV：
    --regression-csv  单测结果：逐 case、逐 tensor 的误差（对应原
                      test/test_fuse_rmsnorm_rope_qkv_q_dtype.py 的 8 个 case）
    --csv             跑分结果：6 组配置 × 70 个 token 数的误差与耗时
"""

import argparse
import csv
import sys

import pytest
import torch

from lightop.attention import (
    rms_rotary_embedding_fuse_with_kv_store,
    rms_rotary_embedding_fuse_with_kv_store_fake,
)

torch.manual_seed(0)

DEVICE = "cuda"
MODEL_DTYPE = torch.bfloat16
RMS_EPS = 1e-6
ROPE_THETA = 10000.0
PAGE_SIZE = 64

H_Q = 8
H_K = 1
HEAD_DIM = 128
IS_NEOX = True

WARMUP_ITERS = 10
TEST_ITERS = 100

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# e4m3 max = 448，e5m2 max = 57344。取一个夹在中间的值：e5m2 能精确表示，
# e4m3 表示不了。Bug 2 若复现（e5m2 buffer 按 e4m3 编码），这里必然溢出成 NaN。
E4M3_MAX = 448.0
BEYOND_E4M3 = 1024.0

# (batch, seq, h_q, h_k, head_dim, rot_dim) -- fused kernel 支持的两种 rot_dim，
# 对应 opt_rms_rope_qwen3_with_qkv 和 opt_rms_rope_qwen3_rotDim64_with_qkv 两个
# kernel，commit 对两个都改了，所以两个都要覆盖。
SHAPES = [(128, 1, 8, 1, 128, 128), (128, 1, 8, 1, 128, 64)]

# (cache_dtype, q_output_dtype, label)
# 前两组是 Bug 1 的判定场景：cache 是 fp8，但 q 必须留在 bf16。
# 后两组是 gfx938 那类「q 本来就要 fp8」的场景，不能被修坏 —— 与原单测
# test/test_fuse_rmsnorm_rope_qkv_q_dtype.py 的 DTYPE_CASES 保持一致。
# 注：此前按「node1 不支持 fp8 的 q 输出」删掉过后两组，但在 hy3_lcy 上实测
# 这两组是通过的（max_diff 全 0），故保留；若换机器出现不支持，再按机器裁剪。
DTYPE_CASES = [
    (torch.float8_e5m2, torch.bfloat16, "fp8_e5m2 cache + bf16 q"),
    (torch.float8_e4m3fn, torch.bfloat16, "fp8_e4m3 cache + bf16 q"),
    (torch.float8_e5m2, torch.float8_e5m2, "fp8_e5m2 cache + fp8_e5m2 q"),
    (torch.float8_e4m3fn, torch.float8_e4m3fn, "fp8_e4m3 cache + fp8_e4m3 q"),
]

# (cache_dtype, rot_dim, label) -- 6 组跑分配置。
CFG_CASES = [
    (torch.bfloat16, 128, "bf16_r128"),
    (torch.bfloat16, 64, "bf16_r64"),
    (torch.float8_e5m2, 128, "e5m2_r128"),
    (torch.float8_e5m2, 64, "e5m2_r64"),
    (torch.float8_e4m3fn, 128, "e4m3_r128"),
    (torch.float8_e4m3fn, 64, "e4m3_r64"),
]

NUM_TOKENS_LIST = [
    32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
    144, 160, 176, 192, 208, 224, 240, 256,
    288, 320, 352, 384, 416, 448, 480, 512,
    640, 768, 896, 1024,
    1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
    2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072,
    3200, 3328, 3456, 3584, 3712, 3840, 3968, 4096,
    6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480,
    22528, 24576, 26624, 28672, 32768
]


# =====================================================================
# 1. PyTorch 基准实现
# =====================================================================
def baseline_rms_norm_rope_kv_store(
    positions, query, key, v, cos_sin_cache, k_buffer, v_buffer, kv_cache_loc,
    page_size, is_neox, weight_q, weight_k, output_dtype, epsilon=RMS_EPS,
):
    """纯 torch 参考：RMSNorm(整个 head_dim) -> RoPE(前 rot_dim) -> 写 KV cache。

    v 不做 norm 也不做 rope，只跟着 cache dtype 转换后落盘（与 kernel 一致）。

    RoPE 用 dual-half 配对：half = rot_dim // 2，配对的是 (i, i + half)，
    而不是相邻的 (2j, 2j+1)：

        out[i]       = x[i] * cos[i] - x[i + half] * sin[i]
        out[i+half]  = x[i] * sin[i] + x[i + half] * cos[i]

    cos_sin_cache 这里传的是扁平的 [max_pos, rot_dim]，前一半 cos、后一半 sin。

    注意 q 用 output_dtype、k/v 用 v_buffer.dtype —— 这正是 commit 确立的规则，
    参考实现必须照此写，否则它自己就复现了 Bug 1，测不出东西。
    """
    rot_dim = cos_sin_cache.shape[1]
    half = rot_dim // 2

    def rms(x, w):
        xf = x.float()
        return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + epsilon) * w.float()

    cos = cos_sin_cache[positions, :half].float().unsqueeze(-2)  # [n, 1, half]
    sin = cos_sin_cache[positions, half:].float().unsqueeze(-2)

    def rope(x):
        out = x.clone()
        if is_neox:
            lo, hi = x[..., :half], x[..., half:rot_dim]
            out[..., :half] = lo * cos - hi * sin
            out[..., half:rot_dim] = lo * sin + hi * cos
        else:
            pair = x[..., :rot_dim].unflatten(-1, (-1, 2))
            even, odd = pair[..., 0], pair[..., 1]
            out[..., 0:rot_dim:2] = even * cos - odd * sin
            out[..., 1:rot_dim:2] = even * sin + odd * cos
        return out

    q_out = rope(rms(query, weight_q)).to(output_dtype).contiguous()
    k_out = rope(rms(key, weight_k)).to(v_buffer.dtype)
    v_out = v.to(v_buffer.dtype)

    loc_id = kv_cache_loc // page_size
    loc_offset = kv_cache_loc % page_size
    k_buffer[loc_id, :, loc_offset, :] = k_out
    v_buffer[loc_id, :, :, loc_offset] = v_out
    return q_out, k_out, v_out


# =====================================================================
# 2. 工具函数
# =====================================================================
def build_cos_sin_cache(max_pos, dim, base=ROPE_THETA, dtype=torch.float32, device=DEVICE):
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=dtype, device=device) / half))
    t = torch.arange(0, max_pos, dtype=dtype, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).contiguous()


def make_weights(dim, gen):
    """[dim] —— kernel 里 gamma 只按 head 内偏移索引，各 head 共用一份 weight。

    用非全 1 的 weight：全 1 的乘法是 no-op，weight 相关的问题就测不出来了。
    """
    return (torch.rand((dim,), generator=gen, dtype=torch.float32, device=DEVICE)
            .to(MODEL_DTYPE) + 0.5)


def make_inputs(num_tokens, h_q, h_k, dim, rot_dim, cache_dtype, seed=0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    num_blocks = max(1, (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE)
    rand = lambda *s: torch.randn(*s, generator=gen, device=DEVICE, dtype=MODEL_DTYPE)
    q = rand(num_tokens, h_q, dim)
    k = rand(num_tokens, h_k, dim)
    v = rand(num_tokens, h_k, dim)
    positions = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    w_q = make_weights(dim, gen)
    w_k = make_weights(dim, gen)
    k_buf = torch.zeros(num_blocks, h_k, PAGE_SIZE, dim, device=DEVICE, dtype=cache_dtype)
    v_buf = torch.zeros(num_blocks, h_k, dim, PAGE_SIZE, device=DEVICE, dtype=cache_dtype)
    kv_loc = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    cos_sin = build_cos_sin_cache(num_tokens, rot_dim).to(MODEL_DTYPE).contiguous()
    return q, k, v, positions, w_q, w_k, k_buf, v_buf, kv_loc, cos_sin


def call_kernel(positions, q, k, v, cos_sin, k_buf, v_buf, kv_loc,
                dim, w_q, w_k, output_dtype, is_neox=IS_NEOX):
    return rms_rotary_embedding_fuse_with_kv_store(
        positions, q.clone(), k.clone(), v.clone(), cos_sin,
        head_dim=dim, page_size=PAGE_SIZE, k_buffer=k_buf, v_buffer=v_buf,
        kv_cache_loc=kv_loc, is_neox=is_neox, weight_q=w_q, weight_k=w_k,
        output_dtype=output_dtype, residual_q=None, residual_k=None,
        k_scale=None, v_scale=None, epsilon=RMS_EPS,
    )


def gather_cache(k_buf, v_buf, kv_loc, page_size=PAGE_SIZE):
    """从 paged buffer 里按 kv_cache_loc 取回 [n, h_k, dim]。

    注意 v_buffer 的后两维是转置的：k 是 [blocks, h_k, page, dim]，
    v 是 [blocks, h_k, dim, page]。
    """
    loc_id = kv_loc // page_size
    loc_offset = kv_loc % page_size
    return k_buf[loc_id, :, loc_offset, :], v_buf[loc_id, :, :, loc_offset]


def fp8_tolerance(dtype):
    """按 fp8 格式的量化粒度给容差，而不是拍一个固定数字。

    kernel 在 bf16 下做 RMSNorm/RoPE，参考实现在 fp32 下做；两者微小的中间差异
    一旦跨过 fp8 的舍入边界，结果就会整整差 1 个 ULP。e5m2 尾数 2 位、
    e4m3 尾数 3 位，对应的相对间距是 2^-2 / 2^-3，所以 rtol 取该值才能容下
    「同一个数、舍入方向不同」这种合法差异（实测 16384 个元素里只有 1 个
    落在这个边界上：kernel -4.0 vs ref -5.0，该幅值处 e5m2 的 ULP 正好是 1.0）。

    atol 兜住近零值：那里相对误差可以很大，但绝对误差只有 ~3e-3，无实际影响。
    """
    mantissa_bits = 2 if dtype is torch.float8_e5m2 else 3
    return 0.25, 2.0 ** -mantissa_bits


def tolerance_for(fp8_dtype):
    """按输出的实际数值格式给容差。

    fp8（k/v 落 cache）：只有 2~3 位尾数，量化误差占主导，见 fp8_tolerance。

    bf16（q）：尾数 8 位，且 RoPE 的 `lo*cos - hi*sin` 在结果接近零时会发生
    相消，那里绝对误差可以到 ~0.05 而相对意义为零（kernel 在 bf16 下算中间值，
    参考实现在 fp32 下算）。实测 1677 万个元素：mean=1.7e-3、p99.9=0.0156，
    分布干净，极值点都落在 ref≈0 处。atol 取 0.06 兜住相消，rtol 2e-2 仍能
    抓住真正的数值错误（错一位尾数以上的偏差会立刻越界）。
    """
    return fp8_tolerance(fp8_dtype) if fp8_dtype is not None else (6e-2, 2e-2)


def check_close(actual, ref, name, fp8_dtype, label=""):
    """比对并返回误差统计。

    误差是要被**报告**的量，不是只在失败时才可见的量：只回一个 Pass/Fail，
    读表的人无从判断 max_diff 是 1e-3 还是正好卡在容差边上。所以这里把
    max/mean 和当时用的容差一起带出去，写进 CSV。

    同时看 max 和 mean：max 对单个异常点敏感（fp8 的 1-ULP 舍入方向差异就能
    把它顶起来），mean 反映整体分布。只看 max 容易把正常的量化误差误判成回归。
    """
    atol, rtol = tolerance_for(fp8_dtype)
    a, b = actual.float(), ref.float()
    err = (a - b).abs()
    max_diff, mean_diff = err.max().item(), err.mean().item()
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"[{label}] {name} mismatch: max_diff={max_diff:.6e} mean_diff={mean_diff:.6e} "
        f"(atol={atol}, rtol={rtol})"
    )
    return {"max": max_diff, "mean": mean_diff, "atol": atol, "rtol": rtol}


def assert_dtype_contract(q_out, k_out, v_out, q_output_dtype, cache_dtype, label):
    """commit 确立的 dtype 契约：q 跟 output_dtype，k/v 跟 cache。"""
    assert q_out.dtype == q_output_dtype, (
        f"[{label}] q dtype 必须跟随 output_dtype：期望 {q_output_dtype}，"
        f"实际 {q_out.dtype}（cache dtype 是 {cache_dtype}）—— Bug 1 复现"
    )
    assert k_out.dtype == cache_dtype, (
        f"[{label}] k dtype 应跟随 cache：期望 {cache_dtype}，实际 {k_out.dtype}"
    )
    assert v_out.dtype == cache_dtype, (
        f"[{label}] v dtype 应跟随 cache：期望 {cache_dtype}，实际 {v_out.dtype}"
    )


def time_us(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


# =====================================================================
# 3. 单个配置的精度 + 性能
# =====================================================================
def run_case(num_tokens, cfg, timed=True, warmup=WARMUP_ITERS, iters=TEST_ITERS):
    cache_dtype, rot_dim, label = cfg
    h_q, h_k, dim = H_Q, H_K, HEAD_DIM

    (
        q, k, v, positions, w_q, w_k, k_buf_org, v_buf_org, kv_loc, cos_sin,
    ) = make_inputs(num_tokens, h_q, h_k, dim, rot_dim, cache_dtype)

    cache_is_fp8 = cache_dtype in FP8_DTYPES
    # wrapper 的判别条件：if (head_dim > 128 or 非fp8cache) -> 回退分支
    path = "fused" if (dim <= 128 and cache_is_fp8) else "fallback"

    # ---- 精度（对纯 torch 参考实现）----
    k_buf_ref, v_buf_ref = torch.zeros_like(k_buf_org), torch.zeros_like(v_buf_org)
    q_ref, k_ref, v_ref = baseline_rms_norm_rope_kv_store(
        positions, q, k, v, cos_sin, k_buf_ref, v_buf_ref, kv_loc,
        PAGE_SIZE, IS_NEOX, w_q, w_k, MODEL_DTYPE,
    )

    k_buf_tgt, v_buf_tgt = torch.zeros_like(k_buf_org), torch.zeros_like(v_buf_org)
    q_out, k_out, v_out = call_kernel(
        positions, q, k, v, cos_sin, k_buf_tgt, v_buf_tgt, kv_loc, dim, w_q, w_k,
        MODEL_DTYPE,
    )

    assert_dtype_contract(q_out, k_out, v_out, MODEL_DTYPE, cache_dtype, label)

    fp8_or_none = cache_dtype if cache_is_fp8 else None
    diffs = {
        "q": check_close(q_out, q_ref, "q", None, label),
        "k": check_close(k_out, k_ref, "k", fp8_or_none, label),
        "v": check_close(v_out, v_ref, "v", fp8_or_none, label),
        "k_buf": check_close(k_buf_tgt, k_buf_ref, "k_buf", fp8_or_none, label),
        "v_buf": check_close(v_buf_tgt, v_buf_ref, "v_buf", fp8_or_none, label),
    }

    if not timed:
        return {"num_tokens": num_tokens, "label": label, "path": path, "diffs": diffs}

    # ---- 性能 ----
    def run_baseline():
        return baseline_rms_norm_rope_kv_store(
            positions, q, k, v, cos_sin, k_buf_ref, v_buf_ref, kv_loc,
            PAGE_SIZE, IS_NEOX, w_q, w_k, MODEL_DTYPE,
        )

    def run_kernel():
        return call_kernel(positions, q, k, v, cos_sin, k_buf_tgt, v_buf_tgt,
                           kv_loc, dim, w_q, w_k, MODEL_DTYPE)

    base_us = time_us(run_baseline, warmup, iters)
    cuda_us = time_us(run_kernel, warmup, iters)

    # ---- 带宽 ----
    # 读：q/k/v 的 token 数据、weight、cos_sin。写：q_out，以及 k/v 落进 cache。
    # k_rope_out / v_rope_out 与写进 cache 的是同一份数据，只计一次，不重复计。
    in_bytes = num_tokens * (h_q + 2 * h_k) * dim * q.element_size()
    in_bytes += num_tokens * rot_dim * cos_sin.element_size()
    in_bytes += (h_q + h_k) * dim * w_q.element_size()
    out_bytes = num_tokens * h_q * dim * q_out.element_size()
    out_bytes += 2 * num_tokens * h_k * dim * k_out.element_size()
    io_bytes = in_bytes + out_bytes
    bandwidth_gb_s = (io_bytes / 1e9) / (cuda_us / 1e6)

    return {
        "num_tokens": num_tokens,
        "label": label,
        "path": path,
        "qkv_heads": f"{h_q}_{h_k}",
        "rot_dim": rot_dim,
        "base_time_us": base_us,
        "cuda_time_us": cuda_us,
        "speedup": base_us / cuda_us if cuda_us > 0 else float("inf"),
        "bandwidth_gb_s": bandwidth_gb_s,
        "diffs": diffs,
    }


# =====================================================================
# 4. 全量跑分 + CSV
# =====================================================================
CSV_HEADER = [
    "Tokens", "Case", "Path", "Q/KV Heads", "RotDim", "Result",
    # 误差列：融合算子 vs 纯 torch 参考实现。q 是 bf16，k/v 落 cache 走 cache dtype，
    # 所以两者的容差不同（见 tolerance_for），把 atol/rtol 一并写出来，
    # 免得读表时拿 fp8 的 max_diff 去对 bf16 的尺度。
    "Q MaxDiff", "Q MeanDiff", "Q atol", "Q rtol",
    "KV MaxDiff", "KV MeanDiff", "KV atol", "KV rtol",
    "PyTorch(us)", "CUDA(us)", "Speedup", "Bandwidth(GB/s)",
]


def run_benchmark(save_to_csv=True, csv_filename="rmsnorm_rope_qkv_q_dtype_benchmark.csv",
                  warmup=WARMUP_ITERS, iters=TEST_ITERS, tokens=None):
    tokens = tokens or NUM_TOKENS_LIST
    width = 176
    print(f"\n{'=' * width}")
    print(
        f"{'Tokens':<8} | {'Case':<10} | {'Path':<9} | {'Result':<8} | "
        f"{'Q MaxDiff':<11} | {'Q MeanDiff':<11} | {'KV MaxDiff':<11} | {'KV MeanDiff':<11} | "
        f"{'PyTorch(us)':<12} | {'CUDA(us)':<10} | {'Speedup':<8} | {'Bandwidth(GB/s)':<15}"
    )
    print(f"{'-' * width}")

    csv_file, csv_writer = None, None
    if save_to_csv:
        csv_file = open(csv_filename, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_HEADER)

    for cfg in CFG_CASES:
        for num_tokens in tokens:
            result = run_case(num_tokens, cfg, timed=True, warmup=warmup, iters=iters)
            q_err = result["diffs"]["q"]
            # KV 侧取 k/v/k_buf/v_buf 里最差的一个：它们同属 cache dtype、同一套
            # 容差，逐个开列没有信息量，但取最大值能保证任何一处退化都不会被平均掉。
            kv_err = max(
                (result["diffs"][n] for n in ("k", "v", "k_buf", "v_buf")),
                key=lambda d: d["max"],
            )

            print(
                f"{result['num_tokens']:<8} | {result['label']:<10} | {result['path']:<9} | "
                f"{'Pass ✅':<8} | {q_err['max']:<11.3e} | {q_err['mean']:<11.3e} | "
                f"{kv_err['max']:<11.3e} | {kv_err['mean']:<11.3e} | "
                f"{result['base_time_us']:<12.2f} | "
                f"{result['cuda_time_us']:<10.2f} | {result['speedup']:<8.2f} | "
                f"{result['bandwidth_gb_s']:<15.2f}"
            )

            if csv_writer is not None:
                csv_writer.writerow([
                    result["num_tokens"],
                    result["label"],
                    result["path"],
                    result["qkv_heads"],
                    result["rot_dim"],
                    "Pass",
                    f"{q_err['max']:.6e}",
                    f"{q_err['mean']:.6e}",
                    q_err["atol"],
                    q_err["rtol"],
                    f"{kv_err['max']:.6e}",
                    f"{kv_err['mean']:.6e}",
                    kv_err["atol"],
                    kv_err["rtol"],
                    f"{result['base_time_us']:.2f}",
                    f"{result['cuda_time_us']:.2f}",
                    f"{result['speedup']:.2f}",
                    f"{result['bandwidth_gb_s']:.2f}",
                ])

    if csv_file is not None:
        csv_file.close()
        print(f"\n[INFO] Benchmark results have been successfully saved to: {csv_filename}")

    print(f"{'=' * width}\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iters", type=int, default=TEST_ITERS)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--tokens", type=int, nargs="+", default=None,
                        help="只跑指定的 token 数（默认跑完整 NUM_TOKENS_LIST）")
    parser.add_argument("--csv", default="rmsnorm_rope_qkv_q_dtype_benchmark.csv")
    parser.add_argument("--regression-csv",
                        default="rmsnorm_rope_qkv_q_dtype_regression.csv",
                        help="单测结果（逐 case、逐 tensor 的误差）导出路径")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--regression-only", action="store_true",
                        help="只跑单测并导 CSV，跳过耗时的全量跑分")
    return parser.parse_args(argv)


# =====================================================================
# 5. Bug 1 的回归用例：q 的 dtype 与 KV cache 无关
# =====================================================================
def run_shape_case(shape, cache_dtype, q_output_dtype, label, is_neox=True):
    batch, seq, h_q, h_k, dim, rot_dim = shape
    num_tokens = batch * seq
    (
        q_org, k_org, v_org, positions, w_q, w_k, k_buf_org, v_buf_org, kv_loc, cos_sin,
    ) = make_inputs(num_tokens, h_q, h_k, dim, rot_dim, cache_dtype)

    common = dict(
        head_dim=dim, page_size=PAGE_SIZE, kv_cache_loc=kv_loc, is_neox=is_neox,
        weight_q=w_q, weight_k=w_k, output_dtype=q_output_dtype,
        residual_q=None, residual_k=None, k_scale=None, v_scale=None, epsilon=RMS_EPS,
    )

    k_buf_ref, v_buf_ref = torch.zeros_like(k_buf_org), torch.zeros_like(v_buf_org)
    q_ref, k_ref, v_ref = rms_rotary_embedding_fuse_with_kv_store_fake(
        positions, q_org.clone(), k_org.clone(), v_org.clone(), cos_sin,
        k_buffer=k_buf_ref, v_buffer=v_buf_ref, **common,
    )

    k_buf_tgt, v_buf_tgt = torch.zeros_like(k_buf_org), torch.zeros_like(v_buf_org)
    q_out, k_out, v_out = rms_rotary_embedding_fuse_with_kv_store(
        positions, q_org.clone(), k_org.clone(), v_org.clone(), cos_sin,
        k_buffer=k_buf_tgt, v_buffer=v_buf_tgt, **common,
    )

    assert_dtype_contract(q_out, k_out, v_out, q_output_dtype, cache_dtype, label)

    q_fp8 = q_output_dtype if q_output_dtype in FP8_DTYPES else None
    cache_fp8 = cache_dtype if cache_dtype in FP8_DTYPES else None

    # 对 ..._fake：两条路径都调同一个 op.rms_rotary_embedding_fuse，只有落盘方式不同，
    # 所以这里期望 max_diff 恒为 0 —— 它查的是「wrapper 的两条分支是否一致」，
    # 不是数值精度。实测确实全是 0.0。
    vs_fake = {
        "q": check_close(q_out, q_ref, "q", q_fp8, label),
        "k": check_close(k_out, k_ref, "k", cache_fp8, label),
        "v": check_close(v_out, v_ref, "v", cache_fp8, label),
        "k_buf": check_close(k_buf_tgt, k_buf_ref, "k_buf", cache_fp8, label),
        "v_buf": check_close(v_buf_tgt, v_buf_ref, "v_buf", cache_fp8, label),
    }

    # 对纯 torch 参考实现：这才是真正的数值精度。上面那组恒为 0，单独导出去没有
    # 信息量，所以两组一起报，CSV 里也分开两列。
    k_buf_t, v_buf_t = torch.zeros_like(k_buf_org), torch.zeros_like(v_buf_org)
    q_t, k_t, v_t = baseline_rms_norm_rope_kv_store(
        positions, q_org, k_org, v_org, cos_sin, k_buf_t, v_buf_t, kv_loc,
        PAGE_SIZE, is_neox, w_q, w_k, q_output_dtype,
    )
    vs_torch = {
        "q": check_close(q_out, q_t, "q(vs torch)", q_fp8, label),
        "k": check_close(k_out, k_t, "k(vs torch)", cache_fp8, label),
        "v": check_close(v_out, v_t, "v(vs torch)", cache_fp8, label),
        "k_buf": check_close(k_buf_tgt, k_buf_t, "k_buf(vs torch)", cache_fp8, label),
        "v_buf": check_close(v_buf_tgt, v_buf_t, "v_buf(vs torch)", cache_fp8, label),
    }

    return {
        "shape": shape, "rot_dim": rot_dim, "num_tokens": num_tokens,
        "cache_dtype": cache_dtype, "q_dtype": q_output_dtype, "label": label,
        "q_out_dtype": q_out.dtype, "k_out_dtype": k_out.dtype,
        "vs_fake": vs_fake, "vs_torch": vs_torch,
    }


@pytest.mark.parametrize("shape", SHAPES, ids=[f"rot{s[5]}" for s in SHAPES])
@pytest.mark.parametrize("case", DTYPE_CASES, ids=[c[2] for c in DTYPE_CASES])
def test_q_dtype_follows_output_dtype(shape, case):
    """Bug 1：fp8 的 KV cache 不能把 q 也拖成 fp8，q 必须是 bf16。"""
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA/ROCm 设备")
    cache_dtype, q_dtype, label = case
    run_shape_case(shape, cache_dtype, q_dtype, f"{label}, shape={shape}")


# =====================================================================
# 5b. 单测结果导出 CSV（对应原 test/test_fuse_rmsnorm_rope_qkv_q_dtype.py）
# =====================================================================
REGRESSION_CSV_HEADER = [
    "Case", "Shape", "Tokens", "RotDim", "CacheDtype", "QDtype",
    "QDtypeOut", "KDtypeOut", "Result", "Tensor",
    # 两个参照物，含义不同，分开列：
    #   vs_fake  —— wrapper 两条分支是否一致，期望恒 0（两边同调一个 kernel）
    #   vs_torch —— 对纯 torch 参考实现的真实数值误差
    "MaxDiff(vs fake)", "MaxDiff(vs torch)", "MeanDiff(vs torch)", "atol", "rtol",
]

TENSOR_ORDER = ("q", "k", "v", "k_buf", "v_buf")


def dtype_name(dt):
    return str(dt).replace("torch.", "")


def run_regression_suite(save_to_csv=True,
                         csv_filename="rmsnorm_rope_qkv_q_dtype_regression.csv"):
    """跑原单测的全部 case（4 dtype 组合 × 2 shape），逐 tensor 导出误差。

    原单测只把 max_diff 打在 stdout 上，且它比的是 kernel vs ..._fake —— 两条
    路径同调一个 op.rms_rotary_embedding_fuse，所以那组数恒为 0。只导这一组的话
    CSV 会是一整列 0.0，没有信息量，因此同时给出对纯 torch 参考实现的误差。
    """
    width = 150
    print(f"\n{'=' * width}")
    print(
        f"{'Case':<28} | {'Rot':<4} | {'Tensor':<6} | {'Result':<8} | "
        f"{'vs fake':<11} | {'vs torch max':<12} | {'vs torch mean':<13} | "
        f"{'atol':<7} | {'rtol':<7}"
    )
    print(f"{'-' * width}")

    csv_file, csv_writer = None, None
    if save_to_csv:
        csv_file = open(csv_filename, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(REGRESSION_CSV_HEADER)

    failures = []
    for shape in SHAPES:
        for cache_dtype, q_dtype, label in DTYPE_CASES:
            try:
                res = run_shape_case(shape, cache_dtype, q_dtype, label)
                result = "Pass"
            except AssertionError as exc:
                failures.append((label, shape, str(exc)))
                print(f"{label:<28} | {shape[5]:<4} | {'-':<6} | {'Fail ❌':<8} | {exc}")
                if csv_writer is not None:
                    csv_writer.writerow([
                        label, "x".join(str(s) for s in shape), shape[0] * shape[1],
                        shape[5], dtype_name(cache_dtype), dtype_name(q_dtype),
                        "", "", "Fail", "", "", "", "", "", "",
                    ])
                continue

            for tensor in TENSOR_ORDER:
                f_err, t_err = res["vs_fake"][tensor], res["vs_torch"][tensor]
                print(
                    f"{label:<28} | {res['rot_dim']:<4} | {tensor:<6} | "
                    f"{'Pass ✅':<8} | {f_err['max']:<11.3e} | {t_err['max']:<12.3e} | "
                    f"{t_err['mean']:<13.3e} | {t_err['atol']:<7} | {t_err['rtol']:<7}"
                )
                if csv_writer is not None:
                    csv_writer.writerow([
                        label,
                        "x".join(str(s) for s in shape),
                        res["num_tokens"],
                        res["rot_dim"],
                        dtype_name(cache_dtype),
                        dtype_name(q_dtype),
                        dtype_name(res["q_out_dtype"]),
                        dtype_name(res["k_out_dtype"]),
                        result,
                        tensor,
                        f"{f_err['max']:.6e}",
                        f"{t_err['max']:.6e}",
                        f"{t_err['mean']:.6e}",
                        t_err["atol"],
                        t_err["rtol"],
                    ])

    if csv_file is not None:
        csv_file.close()
        print(f"\n[INFO] Regression results have been successfully saved to: {csv_filename}")
    print(f"{'=' * width}\n")

    if failures:
        raise AssertionError("\n".join(f"{l} {s}: {e}" for l, s, e in failures))


@pytest.mark.parametrize("cache_dtype", FP8_DTYPES,
                         ids=["e4m3_cache", "e5m2_cache"])
def test_q_values_are_not_fp8_quantised(cache_dtype):
    """Bug 1 的值级判据：只查 dtype 不够，值也不能被 fp8 量化过。

    光断言 `q_out.dtype == bf16` 会漏掉「kernel 先量化成 fp8、再扩回 bf16」这种
    修法 —— dtype 对了，精度已经没了。fp8 只有 2~3 位尾数，网格很稀疏：如果 q
    真的被量化过，它每一个元素都会正好落在 fp8 的可表示值上。

    实测未量化的 q 落在网格上的比例仅 5%(e5m2) / 9%(e4m3)，纯属巧合命中；
    Bug 1 复现时该比例是 100%。阈值取 50%，离两侧都很远。
    """
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA/ROCm 设备")
    num_tokens, dim, rot_dim = 128, HEAD_DIM, 128
    (
        q, k, v, positions, w_q, w_k, k_buf, v_buf, kv_loc, cos_sin,
    ) = make_inputs(num_tokens, H_Q, H_K, dim, rot_dim, cache_dtype)

    q_out, _, _ = call_kernel(positions, q, k, v, cos_sin, k_buf, v_buf, kv_loc,
                              dim, w_q, w_k, MODEL_DTYPE)

    qf = q_out.float()
    on_grid = (qf == qf.to(cache_dtype).float()).float().mean().item()
    assert on_grid < 0.5, (
        f"q 的 {on_grid:.1%} 个元素正好落在 {cache_dtype} 的网格上，"
        f"说明 q 被按 cache 的 fp8 格式量化过（Bug 1 复现）——"
        f"dtype 虽是 {q_out.dtype}，精度已经损失"
    )


# =====================================================================
# 6. Bug 2 的回归用例：cache 的 fp8 格式取自 buffer 本身，而非「是不是 fp8」
# =====================================================================
@pytest.mark.parametrize("rot_dim", [128, 64], ids=["rot128", "rot64"])
def test_e5m2_cache_is_not_encoded_as_e4m3(rot_dim):
    """Bug 2 的判定性用例：e5m2 的 buffer 必须用 e5m2 编码。

    构造在 (e4m3_max=448, e5m2_max=57344] 区间内的 v —— e5m2 能精确表示，
    e4m3 表示不了。修复前 cache_fp8_type 对任何 fp8 cache 都取 1(e4m3)，
    这些值会溢出；用 e5m2 读回来就是 NaN。

    v 不过 norm 也不过 rope，是唯一能原值送达 cache 的通路，所以用 v 来验
    编码格式最干净（k 会被 RMSNorm 归一化掉，幅值不可控）。

    两个 rot_dim 对应两个不同的 kernel，commit 对两个都改了，都要覆盖。
    """
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA/ROCm 设备")
    cache_dtype = torch.float8_e5m2
    num_tokens, dim = 128, HEAD_DIM
    (
        q, k, _, positions, w_q, w_k, k_buf, v_buf, kv_loc, cos_sin,
    ) = make_inputs(num_tokens, H_Q, H_K, dim, rot_dim, cache_dtype)

    # 确认前提成立，免得常量哪天被改坏了测试还在空转
    assert torch.finfo(torch.float8_e4m3fn).max == E4M3_MAX
    assert BEYOND_E4M3 > E4M3_MAX
    assert BEYOND_E4M3 <= torch.finfo(torch.float8_e5m2).max

    v = torch.full((num_tokens, H_K, dim), BEYOND_E4M3, device=DEVICE, dtype=MODEL_DTYPE)
    _, _, v_out = call_kernel(positions, q, k, v, cos_sin, k_buf, v_buf, kv_loc,
                              dim, w_q, w_k, MODEL_DTYPE)

    _, v_cached = gather_cache(k_buf, v_buf, kv_loc)
    for name, got in (("v_out", v_out), ("v_buffer", v_cached)):
        gf = got.float()
        assert torch.isfinite(gf).all(), (
            f"{name} 里出现了 {torch.isnan(gf).sum().item()} 个 NaN："
            f"e5m2 的 cache 被按 e4m3 编码了（{BEYOND_E4M3} 超出 e4m3 的 "
            f"{E4M3_MAX} 上限）—— Bug 2 复现"
        )
        assert (gf == BEYOND_E4M3).all(), (
            f"{name} 期望全为 {BEYOND_E4M3}（e5m2 可精确表示），"
            f"实际 min={gf.min().item()} max={gf.max().item()}"
        )


@pytest.mark.parametrize("cache_dtype", FP8_DTYPES, ids=["e4m3_cache", "e5m2_cache"])
@pytest.mark.parametrize("rot_dim", [128, 64], ids=["rot128", "rot64"])
def test_cache_encoding_is_bit_exact(cache_dtype, rot_dim):
    """Bug 2 的一般性判据：v 落盘必须与 torch 的同格式转换逐位相同。

    v 不做 norm 也不做 rope，kernel 对它只有一次 dtype 转换，所以这里可以要求
    bit-exact —— 不需要任何容差。编码格式一旦选错，字节就对不上。
    """
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA/ROCm 设备")
    num_tokens, dim = 128, HEAD_DIM
    (
        q, k, v, positions, w_q, w_k, k_buf, v_buf, kv_loc, cos_sin,
    ) = make_inputs(num_tokens, H_Q, H_K, dim, rot_dim, cache_dtype)

    call_kernel(positions, q, k, v, cos_sin, k_buf, v_buf, kv_loc,
                dim, w_q, w_k, MODEL_DTYPE)

    _, v_cached = gather_cache(k_buf, v_buf, kv_loc)
    expected = v.to(cache_dtype)
    got_bytes = v_cached.view(torch.uint8)
    exp_bytes = expected.view(torch.uint8)
    mismatched = (got_bytes != exp_bytes).sum().item()
    assert mismatched == 0, (
        f"v_buffer 有 {mismatched}/{exp_bytes.numel()} 字节与 v.to({cache_dtype}) 不符，"
        f"cache 的编码格式选错了（Bug 2）"
    )


def test_mixed_kv_buffer_dtypes_are_rejected():
    """k_buffer 与 v_buffer 的 dtype 必须一致。

    cache 的编码格式是从 v_buffer 推出来、同时用于 k 和 v 的，两者 dtype 不一致
    就无从推起。commit 为此加了 TORCH_CHECK —— 必须报错，不能默默写坏一个。
    """
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA/ROCm 设备")
    num_tokens, dim, rot_dim = 128, HEAD_DIM, 128
    (
        q, k, v, positions, w_q, w_k, k_buf, _, kv_loc, cos_sin,
    ) = make_inputs(num_tokens, H_Q, H_K, dim, rot_dim, torch.float8_e5m2)
    num_blocks = k_buf.shape[0]
    v_buf_other = torch.zeros(num_blocks, H_K, dim, PAGE_SIZE,
                              device=DEVICE, dtype=torch.float8_e4m3fn)

    with pytest.raises(RuntimeError, match="must share a dtype"):
        call_kernel(positions, q, k, v, cos_sin, k_buf, v_buf_other, kv_loc,
                    dim, w_q, w_k, MODEL_DTYPE)


def test_q_dtype_mismatch_is_rejected():
    """q buffer 既不是 fp8、也不等于 query 的 dtype 时必须报错，而不是写坏数据。

    kernel 里 q_out_fp8_type==2 这条分支是按 scalar_t 的宽度算地址的，
    传个 fp32 的 output_dtype 进来它无法安全表达，commit 加了 TORCH_CHECK 兜住。
    """
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA/ROCm 设备")
    num_tokens, dim, rot_dim = 128, HEAD_DIM, 128
    (
        q, k, v, positions, w_q, w_k, k_buf, v_buf, kv_loc, cos_sin,
    ) = make_inputs(num_tokens, H_Q, H_K, dim, rot_dim, torch.float8_e5m2)

    with pytest.raises(RuntimeError):
        call_kernel(positions, q, k, v, cos_sin, k_buf, v_buf, kv_loc,
                    dim, w_q, w_k, torch.float32)


# =====================================================================
# 7. 6 组配置的精度自查（跑分同款配置，不计时）
# =====================================================================
SMOKE_TOKENS = 4096


@pytest.mark.parametrize("cfg", CFG_CASES, ids=[c[2] for c in CFG_CASES])
def test_fuse_rmsnorm_rope_qkv_q_dtype(cfg):
    """6 组配置对纯 torch 参考实现的精度自查；全量跑分走 __main__。"""
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA/ROCm 设备")
    run_case(SMOKE_TOKENS, cfg, timed=False)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no GPU available")
        sys.exit(0)

    args = parse_args()

    # 先跑单测并导出结果 CSV，再跑全量跑分 —— 单测挂了就没必要跑分了。
    run_regression_suite(
        save_to_csv=not args.no_csv,
        csv_filename=args.regression_csv,
    )

    if args.regression_only:
        sys.exit(0)

    run_benchmark(
        save_to_csv=not args.no_csv,
        csv_filename=args.csv,
        warmup=args.warmup,
        iters=args.iters,
        tokens=args.tokens,
    )
