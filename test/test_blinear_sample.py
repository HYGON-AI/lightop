# -*- coding: utf-8 -*-
"""对比 torch nn.Upsample(bilinear) 与 lightop.bilinear_upsample 的精度、速度和带宽。
结果同时打印到终端并保存为 Excel 文件。
"""
import os
import torch
import torch.nn as nn
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

import lightop

# ============================================================
#  测试用例：(dtype, N, C, H_in, W_in, scale_factor)
# ============================================================
TEST_CASES = [
    (torch.float32,  6, 256, 128, 248, 2),
    (torch.float16,  6, 256, 128, 248, 2),
    (torch.float32, 48, 384,  16,  32, 2),
    (torch.float16,  6, 512,  16,  31, 4),
    (torch.float16,  6, 512,  64, 124, 2),
    (torch.float32, 48,  64, 128, 256, 2),
]

REPEAT    = 100
WARMUP    = 10
RTOL      = 1e-2
ATOL      = 1e-2
EXCEL_OUT = os.path.join(os.path.dirname(__file__), "bilinear_upsample_bench.xlsx")

device     = torch.device("cuda")
DTYPE_NAME = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
BYTES_ELEM = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}

torch.manual_seed(0)


# ============================================================
#  lightop 可微分包装
# ============================================================
class _LightopBilinearUpsample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output_size):
        ctx.save_for_backward(torch.tensor(list(x.shape), dtype=torch.long))
        ctx.output_size = list(output_size)
        return lightop.bilinear_upsample(x, output_size)

    @staticmethod
    def backward(ctx, grad_output):
        (input_size_t,) = ctx.saved_tensors
        grad_input = lightop.bilinear_upsample_backward(
            grad_output.contiguous(),
            input_size_t.tolist(),
        )
        return grad_input, None


def lightop_upsample(x: torch.Tensor, output_size) -> torch.Tensor:
    return _LightopBilinearUpsample.apply(x, list(output_size))


# ============================================================
#  带宽计算
#    前向：读 input + 写 output
#    反向：读 grad_output + 写 grad_input（+ float32 缓冲区 for fp16/bf16）
# ============================================================
def fwd_bw_gbps(N, C, H_in, W_in, H_out, W_out, dtype, ms):
    bpe   = BYTES_ELEM[dtype]
    total = (N * C * H_in * W_in + N * C * H_out * W_out) * bpe
    return total / (ms * 1e-3) / 1e9


def bwd_bw_gbps_torch(N, C, H_in, W_in, H_out, W_out, dtype, ms):
    """torch backward: 读 grad_output + 原子写 grad_input（fp32 中间）"""
    bpe    = BYTES_ELEM[dtype]
    bpe_f32 = 4
    # 读 grad_output(dtype) + 写 grad_input(dtype) + 读写 float32 buffer
    total = (N * C * H_out * W_out * bpe          # read grad_output
           + N * C * H_in  * W_in  * bpe          # write grad_input
           + N * C * H_in  * W_in  * bpe_f32 * 2) # float32 buffer rw
    return total / (ms * 1e-3) / 1e9


def bwd_bw_gbps_lightop(N, C, H_in, W_in, H_out, W_out, dtype, ms):
    """lightop backward: 与 torch 相同逻辑"""
    return bwd_bw_gbps_torch(N, C, H_in, W_in, H_out, W_out, dtype, ms)


# ============================================================
#  计时工具
# ============================================================
def bench_fn(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeat):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / repeat


def bench_backward(fwd_fn, x_base, grad_out, warmup, repeat):
    x = x_base.detach().requires_grad_(True)
    y = fwd_fn(x)
    for _ in range(warmup):
        x.grad = None
        y.backward(grad_out, retain_graph=True)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeat):
        x.grad = None
        y.backward(grad_out, retain_graph=True)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / repeat


# ============================================================
#  精度验证
# ============================================================
def check_accuracy(x, H_out, W_out):
    torch_up = nn.Upsample(size=(H_out, W_out), mode="bilinear", align_corners=False)
    fwd_ok = torch.allclose(
        torch_up(x).float(),
        lightop.bilinear_upsample(x, [H_out, W_out]).float(),
        rtol=RTOL, atol=ATOL
    )
    grad_out = torch.ones(x.shape[0], x.shape[1], H_out, W_out,
                          device=x.device, dtype=x.dtype)
    x_t = x.clone().requires_grad_(True)
    torch_up(x_t).backward(grad_out)
    x_l = x.clone().requires_grad_(True)
    lightop_upsample(x_l, [H_out, W_out]).backward(grad_out)
    bwd_ok = torch.allclose(x_t.grad.float(), x_l.grad.float(), rtol=RTOL, atol=ATOL)
    return fwd_ok, bwd_ok


# ============================================================
#  Excel 写入工具
# ============================================================
def make_excel(rows, path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "bilinear_upsample"

    headers = [
        "dtype", "input_size", "scale", "output_size", "impl",
        "fwd_ms", "fwd_BW(GB/s)",
        "bwd_ms", "bwd_BW(GB/s)",
        "fwd_speedup", "bwd_speedup",
        "fwd_acc", "bwd_acc",
    ]

    # --- 颜色 ---
    hdr_fill   = PatternFill("solid", fgColor="1F4E79")   # 深蓝
    torch_fill = PatternFill("solid", fgColor="D6E4F0")   # 浅蓝
    light_fill = PatternFill("solid", fgColor="E8F5E9")   # 浅绿
    ok_fill    = PatternFill("solid", fgColor="C8E6C9")   # 绿
    fail_fill  = PatternFill("solid", fgColor="FFCDD2")   # 红

    thin = Side(style="thin", color="AAAAAA")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def cell_style(cell, fill=None, bold=False, align="center"):
        if fill:
            cell.fill = fill
        cell.font      = Font(bold=bold,
                              color="FFFFFF" if fill == hdr_fill else "000000",
                              size=10)
        cell.alignment = Alignment(horizontal=align, vertical="center",
                                   wrap_text=True)
        cell.border    = border

    # --- 表头 ---
    ws.append(headers)
    for c_idx, _ in enumerate(headers, 1):
        cell_style(ws.cell(1, c_idx), fill=hdr_fill, bold=True)

    # --- 数据行 ---
    for row in rows:
        ws.append(row)
        r = ws.max_row
        is_torch = (row[4] == "torch")
        fill = torch_fill if is_torch else light_fill
        for c_idx in range(1, len(headers) + 1):
            cell = ws.cell(r, c_idx)
            # 精度列特殊着色
            if c_idx == 12:   # fwd_acc
                cell_style(cell, fill=ok_fill if row[11] == "OK" else fail_fill)
            elif c_idx == 13: # bwd_acc
                cell_style(cell, fill=ok_fill if row[12] == "OK" else fail_fill)
            else:
                cell_style(cell, fill=fill)

    # --- 列宽 ---
    col_widths = [7, 22, 7, 22, 9, 10, 14, 10, 14, 12, 12, 9, 9]
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ws.row_dimensions[1].height = 22
    ws.freeze_panes = "A2"

    wb.save(path)
    print(f"\n[Excel] 已保存到: {path}")


# ============================================================
#  主流程
# ============================================================
if __name__ == "__main__":
    # ---- 终端表头 ----
    W = [6, 22, 6, 22, 8, 10, 14, 10, 14, 11, 11, 5, 5]

    def fmt_row(*vals):
        return " ".join(f"{str(v):<{w}}" for v, w in zip(vals, W))

    hdr = fmt_row("dtype", "input_size", "scale", "output_size", "impl",
                  "fwd(ms)", "fwd_BW(GB/s)", "bwd(ms)", "bwd_BW(GB/s)",
                  "fwd_spd", "bwd_spd", "fwdA", "bwdA")
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    excel_rows = []

    for dtype, N, C, H_in, W_in, scale in TEST_CASES:
        H_out = H_in * scale
        W_out = W_in * scale

        in_str  = f"{N},{C},{H_in},{W_in}"
        out_str = f"{N},{C},{H_out},{W_out}"
        dt_str  = DTYPE_NAME[dtype]

        x        = torch.rand(N, C, H_in, W_in, device=device, dtype=dtype)
        grad_out = torch.ones(N, C, H_out, W_out, device=device, dtype=dtype)
        torch_up = nn.Upsample(size=(H_out, W_out), mode="bilinear", align_corners=False)

        # 精度
        fwd_ok, bwd_ok = check_accuracy(x, H_out, W_out)
        acc_fwd = "OK" if fwd_ok else "FAIL"
        acc_bwd = "OK" if bwd_ok else "FAIL"

        # 计时
        ms_t_fwd = bench_fn(lambda: torch_up(x), WARMUP, REPEAT)
        ms_t_bwd = bench_backward(torch_up, x, grad_out, WARMUP, REPEAT)
        ms_l_fwd = bench_fn(lambda: lightop.bilinear_upsample(x, [H_out, W_out]), WARMUP, REPEAT)
        ms_l_bwd = bench_backward(lambda xi: lightop_upsample(xi, [H_out, W_out]),
                                  x, grad_out, WARMUP, REPEAT)

        # 带宽
        bw_t_fwd = fwd_bw_gbps(N, C, H_in, W_in, H_out, W_out, dtype, ms_t_fwd)
        bw_l_fwd = fwd_bw_gbps(N, C, H_in, W_in, H_out, W_out, dtype, ms_l_fwd)
        bw_t_bwd = bwd_bw_gbps_torch(N, C, H_in, W_in, H_out, W_out, dtype, ms_t_bwd)
        bw_l_bwd = bwd_bw_gbps_lightop(N, C, H_in, W_in, H_out, W_out, dtype, ms_l_bwd)

        spd_fwd = ms_t_fwd / ms_l_fwd
        spd_bwd = ms_t_bwd / ms_l_bwd

        # ---- 终端输出 ----
        print(fmt_row(dt_str, in_str, scale, out_str, "torch",
                      f"{ms_t_fwd:.3f}", f"{bw_t_fwd:.2f}",
                      f"{ms_t_bwd:.3f}", f"{bw_t_bwd:.2f}",
                      "--", "--", acc_fwd, acc_bwd))
        print(fmt_row("", "", "", "", "lightop",
                      f"{ms_l_fwd:.3f}", f"{bw_l_fwd:.2f}",
                      f"{ms_l_bwd:.3f}", f"{bw_l_bwd:.2f}",
                      f"{spd_fwd:.2f}x", f"{spd_bwd:.2f}x", "", ""))
        print(sep)

        # ---- Excel 行数据（保留数值方便筛选）----
        excel_rows.append([dt_str, in_str, scale, out_str, "torch",
                           round(ms_t_fwd, 4), round(bw_t_fwd, 2),
                           round(ms_t_bwd, 4), round(bw_t_bwd, 2),
                           None, None, acc_fwd, acc_bwd])
        excel_rows.append([dt_str, in_str, scale, out_str, "lightop",
                           round(ms_l_fwd, 4), round(bw_l_fwd, 2),
                           round(ms_l_bwd, 4), round(bw_l_bwd, 2),
                           round(spd_fwd, 2), round(spd_bwd, 2), "", ""])

    make_excel(excel_rows, EXCEL_OUT)
    print("Done.")
