"""
混合精度 LayerNorm 验证：激活为 Half / BFloat16，weight / bias 为 float32。

参考实现：在 float32 上做 torch.nn.functional.layer_norm，再 cast 回输入 dtype，
与 lightop.LayerNormMixedFp32Weight 对比前向、dx、dweight、dbias。
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightop import LayerNormMixedFp32Weight

# ------------------------------------------------------------------------------------
# 1. 模块封装
# ------------------------------------------------------------------------------------


class TorchMixedLayerNormModule(nn.Module):
    """gamma/beta 始终 float32；前向在 fp32 中归一化后 cast 回 x.dtype。"""

    def __init__(self, normalized_shape: int, elementwise_affine: bool = True, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=torch.float32, device="cuda"))
            if bias:
                self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.float32, device="cuda"))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        return F.layer_norm(x.float(), self.normalized_shape, self.weight, self.bias, self.eps).to(dtype=x.dtype)


class FuseMixedLayerNormModule(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        elementwise_affine: bool = True,
        bias: bool = True,
        activation_dtype: torch.dtype = torch.float16,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.ln = LayerNormMixedFp32Weight(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias and elementwise_affine,
            device="cuda",
            dtype=activation_dtype,
        )

    def forward(self, x):
        return self.ln(x)


# ------------------------------------------------------------------------------------
# 2. 辅助函数
# ------------------------------------------------------------------------------------


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def get_data(batchsize, seqs, cols, activation_dtype: torch.dtype):
    device = "cuda"
    x = torch.randn((batchsize, seqs, cols), device=device, dtype=activation_dtype, requires_grad=True)
    grad_y = torch.randn_like(x)
    x1 = x.clone().detach().requires_grad_(True)
    x2 = x.clone().detach().requires_grad_(True)
    return x1, x2, grad_y


def get_model(cols, activation_dtype: torch.dtype, mode: str = "full"):
    """
    mode:
      - 'full': weight + bias (fp32)
      - 'no_bias': 仅 weight
      - 'no_affine': 无仿射参数
    """
    elementwise_affine = mode != "no_affine"
    use_bias = mode == "full"

    torch_model = TorchMixedLayerNormModule(cols, elementwise_affine=elementwise_affine, bias=use_bias).cuda()
    fuse_model = FuseMixedLayerNormModule(
        cols, elementwise_affine=elementwise_affine, bias=use_bias, activation_dtype=activation_dtype
    ).cuda()

    if mode == "full":
        with torch.no_grad():
            fuse_model.ln.weight.copy_(torch_model.weight)
            fuse_model.ln.bias.copy_(torch_model.bias)
    elif mode == "no_bias":
        with torch.no_grad():
            fuse_model.ln.weight.copy_(torch_model.weight)

    torch_model.train()
    fuse_model.train()
    return torch_model, fuse_model


# ------------------------------------------------------------------------------------
# 3. 基准与精度
# ------------------------------------------------------------------------------------


def run_benchmark(mode, batchsize, seqs, cols, repeat_num, activation_dtype: torch.dtype):
    mode_title = {
        "full": "Weight + Bias fp32, activ mixed (Fwd+Bwd)",
        "no_bias": "仅 Weight fp32 (Fwd+Bwd)",
        "no_affine": "无仿射 (Fwd+Bwd)",
    }[mode]
    prec_name = "fp16" if activation_dtype == torch.float16 else "bf16"
    print(f"--- Mixed LN: {mode_title} x:{prec_name}, batch={batchsize}, seqs={seqs}, cols={cols} ---")

    x1, x2, grad_y = get_data(batchsize, seqs, cols, activation_dtype)
    f1, f2 = get_model(cols, activation_dtype, mode=mode)

    y1_check = f1(x1)
    y1_check.backward(grad_y)
    y2_check = f2(x2)
    y2_check.backward(grad_y)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(repeat_num):
        f1.zero_grad()
        x1.grad = None
        y1 = f1(x1)
        y1.backward(grad_y)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) * 1000 / repeat_num
    print(f"  Torch (fp32 ref) average time: {torch_time:.4f} ms")

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(repeat_num):
        f2.zero_grad()
        x2.grad = None
        y2 = f2(x2)
        y2.backward(grad_y)
    torch.cuda.synchronize()
    fuse_time = (time.time() - start_time) * 1000 / repeat_num
    print(f"  Fuse mixed-fp32w average time:  {fuse_time:.4f} ms")

    print("\n  Correctness Check:")
    rtol, atol = (2e-2, 2e-2) if activation_dtype in (torch.float16, torch.bfloat16) else (1e-5, 1e-5)

    x1_final, x2_final, grad_y_final = get_data(batchsize, seqs, cols, activation_dtype)
    f1_check, f2_check = get_model(cols, activation_dtype, mode)
    y1_final = f1_check(x1_final)
    y2_final = f2_check(x2_final)
    y1_final.backward(grad_y_final)
    y2_final.backward(grad_y_final)

    print(f"    Forward output close: {allclose(y1_final, y2_final, rtol=rtol, atol=atol)}")
    print(f"    Input grad close:     {allclose(x1_final.grad, x2_final.grad, rtol=rtol, atol=atol)}")

    if mode in ("full", "no_bias"):
        w_rtol, w_atol = (5e-3, 5e-4)  # fp32 参数梯度
        wg = allclose(f1_check.weight.grad, f2_check.ln.weight.grad, rtol=w_rtol, atol=w_atol)
        print(f"    Weight grad close:    {wg}")
    else:
        wc = f1_check.weight is None and f2_check.ln.weight is None
        print(f"    Weight is None:       {wc}")

    if mode == "full":
        w_rtol, w_atol = (5e-3, 5e-4)
        bg = allclose(f1_check.bias.grad, f2_check.ln.bias.grad, rtol=w_rtol, atol=w_atol)
        print(f"    Bias grad close:      {bg}")
    else:
        bc = f1_check.bias is None and f2_check.ln.bias is None
        print(f"    Bias is None:         {bc}")


# ------------------------------------------------------------------------------------
# 4. 入口
# ------------------------------------------------------------------------------------

if __name__ == "__main__":
    # cols>=64 且 cols%8==0 时走 fused CUDA；与 test_layernorm.py 默认一致
    activation_dtype = torch.float16
    # activation_dtype = torch.bfloat16
    batchsize, seqs, cols = 1, 4096, 1024
    repeat_num = 1000
    sep = "\n" + "=" * 80 + "\n"

    print(sep)
    run_benchmark("full", batchsize, seqs, cols, repeat_num, activation_dtype)
    print(sep)

    # run_benchmark("no_bias", batchsize, seqs, cols, repeat_num, activation_dtype)
    # print(sep)

    run_benchmark("no_affine", batchsize, seqs, cols, repeat_num, activation_dtype)
    print(sep)
