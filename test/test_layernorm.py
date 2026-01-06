import torch
import torch.nn as nn
import time
from lightop import LayerNorm

# ------------------------------------------------------------------------------------
# 1. 统一的模块封装
# ------------------------------------------------------------------------------------

class TorchLayerNormModule(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine=True):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, elementwise_affine=elementwise_affine)

    def forward(self, x):
        return self.ln(x)

class FuseLayerNormModule(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine=True):
        super().__init__()
        self.ln = LayerNorm(normalized_shape, elementwise_affine=elementwise_affine)

    def forward(self, x):
        return self.ln(x)

# ------------------------------------------------------------------------------------
# 2. 统一的辅助函数
# ------------------------------------------------------------------------------------

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def get_data(batchsize, seqs, cols, use_half=False):
    dtype = torch.float16 if use_half else torch.float32
    device = 'cuda'
    
    x = torch.randn((batchsize, seqs, cols), device=device, dtype=dtype, requires_grad=True)
    grad_y = torch.randn_like(x)
    
    x1 = x.clone().detach().requires_grad_(True)
    x2 = x.clone().detach().requires_grad_(True)
    return x1, x2, grad_y

def get_model(cols, use_half=False, mode='full'):
    """
    模式 (mode):
      - 'full': 包含 weight 和 bias
      - 'no_bias': 只包含 weight
      - 'no_affine': 不包含 weight 和 bias
    """
    elementwise_affine = (mode != 'no_affine')

    torch_model = TorchLayerNormModule(cols, elementwise_affine=elementwise_affine).cuda()
    fuse_model = FuseLayerNormModule(cols, elementwise_affine=elementwise_affine).cuda()

    if use_half:
        torch_model.half()
        fuse_model.half()

    # 根据模式同步参数
    if mode == 'full':
        with torch.no_grad():
            fuse_model.ln.weight.copy_(torch_model.ln.weight)
            fuse_model.ln.bias.copy_(torch_model.ln.bias)
    elif mode == 'no_bias':
        torch_model.ln.bias = None
        fuse_model.ln.bias = None
        with torch.no_grad():
            fuse_model.ln.weight.copy_(torch_model.ln.weight)
    
    torch_model.train()
    fuse_model.train()

    return torch_model, fuse_model

# ------------------------------------------------------------------------------------
# 3. 统一的基准测试函数
# ------------------------------------------------------------------------------------

def run_benchmark(mode, batchsize, seqs, cols, repeat_num, use_half=False):
    """
    执行指定模式的基准测试，包括速度和精度。
    """
    # --- 打印测试标题 ---
    mode_title = {
        'full': "包含 Weight 和 Bias (Fwd+Bwd)",
        'no_bias': "仅包含 Weight (Fwd+Bwd)",
        'no_affine': "无 Weight 和 Bias (Fwd+Bwd)"
    }[mode]
    precision = "fp16" if use_half else "fp32"
    print(f"--- Running Test: {mode_title} | {precision}, batch={batchsize}, seqs={seqs}, cols={cols} ---")

    # --- 数据和模型准备 ---
    x1, x2, grad_y = get_data(batchsize, seqs, cols, use_half=use_half)
    f1, f2 = get_model(cols, use_half=use_half, mode=mode)

    # --- 预热 (Warm-up) ---
    y1_check = f1(x1)
    y1_check.backward(grad_y)
    y2_check = f2(x2)
    y2_check.backward(grad_y)
    torch.cuda.synchronize()

    # --- 速度测试 ---
    # PyTorch LayerNorm
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(repeat_num):
        f1.zero_grad()
        x1.grad = None
        y1 = f1(x1)
        y1.backward(grad_y)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) * 1000 / repeat_num
    print(f'  Torch LayerNorm average time: {torch_time:.4f} ms')

    # Fuse LayerNorm
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(repeat_num):
        f2.zero_grad()
        x2.grad = None
        y2 = f2(x2)
        y2.backward(grad_y)
    torch.cuda.synchronize()
    fuse_time = (time.time() - start_time) * 1000 / repeat_num
    print(f'  Fuse LayerNorm average time:  {fuse_time:.4f} ms')
    
    # --- 精度对比 (Correctness Check) ---
    print("\n  Correctness Check:")
    rtol, atol = (2e-2, 2e-2) if use_half else (1e-5, 1e-5)
    
    # 为了获得最新的、用于对比的梯度值，在检查前单独执行一次前向和反向传播
    x1_final, x2_final, grad_y_final = get_data(batchsize, seqs, cols, use_half)
    f1_check, f2_check = get_model(cols, use_half, mode)
    y1_final = f1_check(x1_final)
    y2_final = f2_check(x2_final)
    y1_final.backward(grad_y_final)
    y2_final.backward(grad_y_final)
        
    # 1. 检查前向输出
    print(f"    Forward output close: {allclose(y1_final, y2_final, rtol=rtol, atol=atol)}")
    
    # 2. 检查输入梯度
    print(f"    Input grad close:     {allclose(x1_final.grad, x2_final.grad, rtol=rtol, atol=atol)}")
    # print(f"      └─ Max Abs Error:  {torch.max(torch.abs(x1_final.grad - x2_final.grad))}")

    # 3. 检查权重梯度 (根据模式判断)
    if mode in ['full', 'no_bias']:
        w_rtol, w_atol = (2e-1, 2e-1) if use_half else (1e-4, 1e-4)
        weight_grad_close = allclose(f1_check.ln.weight.grad, f2_check.ln.weight.grad, rtol=w_rtol, atol=w_atol)
        print(f"    Weight grad close:    {weight_grad_close}")
        # print(f"      └─ Max Abs Error:  {torch.max(torch.abs(f1_check.ln.weight.grad - f2_check.ln.weight.grad))}")
    else: # no_affine
        weight_check = f1_check.ln.weight is None and f2_check.ln.weight is None
        print(f"    Weight grad is None:  {weight_check}")

    # 4. 检查偏置梯度 (根据模式判断)
    if mode == 'full':
        bias_grad_close = allclose(f1_check.ln.bias.grad, f2_check.ln.bias.grad, rtol=w_rtol, atol=w_atol)
        print(f"    Bias grad close:      {bias_grad_close}")
        # print(f"      └─ Max Abs Error:  {torch.max(torch.abs(f1_check.ln.bias.grad - f2_check.ln.bias.grad))}")
    else: # no_bias or no_affine
        bias_check = f1_check.ln.bias is None and f2_check.ln.bias is None
        print(f"    Bias grad is None:    {bias_check}")

# ------------------------------------------------------------------------------------
# 4. 主程序入口
# ------------------------------------------------------------------------------------

if __name__ == "__main__":
    # --- 全局测试参数 ---
    use_half = True
    # use_half = False
    batchsize, seqs, cols = 1, 4096, 1024
    # batchsize, seqs, cols = 1, 2048, 16384
    repeat_num = 1000
    separator = "\n" + "="*80 + "\n"

    # --- 运行所有测试用例 ---
    # 案例 1: 包含 Weight 和 Bias
    print(separator)
    run_benchmark('full', batchsize, seqs, cols, repeat_num, use_half)
    print(separator)

    # # 案例 2: 仅包含 Weight
    # run_benchmark('no_bias', batchsize, seqs, cols, repeat_num, use_half)
    # print(separator)

    # 案例 3: 无 Weight 和 Bias (包含反向传播)
    run_benchmark('no_affine', batchsize, seqs, cols, repeat_num, use_half)
    print(separator)