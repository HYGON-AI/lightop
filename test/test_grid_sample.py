"""
测试 grid_sample 的脚本，参考 test_rmsnorm.py 的调用流程。

对比:
- GridSampleTorch: 使用 torch.nn.functional.grid_sample (PyTorch 参考实现)
- GridSample: 使用 lightop 的 grid_sample (fuse 实现)

GridSamplerInterpolation: Bilinear=0, Nearest=1, Bicubic=2
GridSamplerPadding: Zeros=0, Border=1, Reflection=2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from lightop import GridSample


# PyTorch grid_sample 的 mode/padding_mode 字符串与 lightop 枚举的映射
INTERP_BILINEAR = 0
INTERP_NEAREST = 1
INTERP_BICUBIC = 2
PADDING_ZEROS = 0
PADDING_BORDER = 1
PADDING_REFLECTION = 2


class GridSampleTorch(nn.Module):
    """使用 torch.nn.functional.grid_sample 的参考实现"""

    def __init__(self, mode='bilinear', padding_mode='zeros', align_corners=False):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, x, grid):
        return F.grid_sample(
            x, grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners
        )


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def get_data(batchsize=48, channels=64, h_in=32, w_in=64, h_out=None, w_out=None, align_corners=False):
    """生成 input、grid、grad"""
    h_out = h_out or h_in
    w_out = w_out or w_in

    x = torch.rand((batchsize, channels, h_in, w_in)).cuda()
    grad = torch.rand((batchsize, channels, h_out, w_out)).cuda()

    # 创建采样网格: (N, H_out, W_out, 2)，坐标范围 [-1, 1]
    y = torch.linspace(-1, 1, h_out).cuda()
    x_coord = torch.linspace(-1, 1, w_out).cuda()
    grid_y, grid_x = torch.meshgrid(y, x_coord, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    grid = grid.expand(batchsize, -1, -1, -1)

    x.requires_grad_(True)
    grid.requires_grad_(True)

    x1 = x.clone().detach().requires_grad_(True)
    x2 = x.clone().detach().requires_grad_(True)
    grid1 = grid.clone().detach().requires_grad_(True)
    grid2 = grid.clone().detach().requires_grad_(True)

    return x1, x2, grid1, grid2, grad


def get_model(mode='bilinear', padding_mode='zeros', align_corners=False):
    """返回 (torch 参考, lightop fuse)"""
    torch_grid_sample = GridSampleTorch(
        mode=mode, padding_mode=padding_mode, align_corners=align_corners
    ).cuda()
    fuse_grid_sample = GridSample().cuda()

    return torch_grid_sample, fuse_grid_sample


def _get_interp_padding_map():
    interp_map = {'bilinear': INTERP_BILINEAR, 'nearest': INTERP_NEAREST, 'bicubic': INTERP_BICUBIC}
    padding_map = {'zeros': PADDING_ZEROS, 'border': PADDING_BORDER, 'reflection': PADDING_REFLECTION}
    return interp_map, padding_map


def test_speed_forward(batchsize, channels, h_in, w_in, h_out=None, w_out=None,
                       repeat_num=100,
                       mode='bilinear', padding_mode='zeros', align_corners=False):
    """仅测试前向：速度对比 + 输出正确性对比"""
    h_out = h_out or h_in
    w_out = w_out or w_in

    print(f"===== 前向对比 | fp32, N={batchsize}, C={channels}, {h_in}x{w_in} -> {h_out}x{w_out} =====")

    x1, x2, grid1, grid2, grad = get_data(
        batchsize, channels, h_in, w_in, h_out, w_out, align_corners=align_corners
    )
    f1, f2 = get_model(mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    interp_map, padding_map = _get_interp_padding_map()

    # torch 参考
    y1 = f1(x1, grid1)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(repeat_num):
        y1 = f1(x1, grid1)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) * 1000 / repeat_num
    print('torch_grid_sample (forward)', 'average time:', torch_time, 'ms')

    # lightop fuse
    y2 = f2(x2, grid2, interp_map[mode], padding_map[padding_mode], align_corners)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(repeat_num):
        y2 = f2(x2, grid2, interp_map[mode], padding_map[padding_mode], align_corners)
    torch.cuda.synchronize()
    fuse_time = (time.time() - start_time) * 1000 / repeat_num
    print('fuse_grid_sample (forward)', 'average time:', fuse_time, 'ms')

    print('output allclose:', allclose(y1, y2, rtol=1e-05, atol=1e-05))
    print()


def test_speed_backward(batchsize, channels, h_in, w_in, h_out=None, w_out=None,
                        repeat_num=100,
                        mode='bilinear', padding_mode='zeros', align_corners=False):
    """仅测试反向：速度对比 + 梯度正确性对比"""
    h_out = h_out or h_in
    w_out = w_out or w_in

    print(f"===== 反向对比 | fp32, N={batchsize}, C={channels}, {h_in}x{w_in} -> {h_out}x{w_out} =====")

    x1, x2, grid1, grid2, grad = get_data(
        batchsize, channels, h_in, w_in, h_out, w_out, align_corners=align_corners
    )
    f1, f2 = get_model(mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    interp_map, padding_map = _get_interp_padding_map()

    # torch 参考：先 forward 一次得到 y1，再 backward
    y1 = f1(x1, grid1)
    y1.backward(grad)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(repeat_num):
        x1.grad = None
        grid1.grad = None
        y1 = f1(x1, grid1)
        y1.backward(grad)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) * 1000 / repeat_num
    print('torch_grid_sample (backward)', 'average time:', torch_time, 'ms')

    # lightop fuse
    y2 = f2(x2, grid2, interp_map[mode], padding_map[padding_mode], align_corners)
    y2.backward(grad)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(repeat_num):
        x2.grad = None
        grid2.grad = None
        y2 = f2(x2, grid2, interp_map[mode], padding_map[padding_mode], align_corners)
        y2.backward(grad)
    torch.cuda.synchronize()
    fuse_time = (time.time() - start_time) * 1000 / repeat_num
    print('fuse_grid_sample (backward)', 'average time:', fuse_time, 'ms')

    print('x.grad allclose:', allclose(x1.grad, x2.grad, rtol=1e-05, atol=1e-05))
    print('grid.grad allclose:', allclose(grid1.grad, grid2.grad, rtol=1e-05, atol=1e-05))
    print()


def test_speed(batchsize, channels, h_in, w_in, h_out=None, w_out=None,
               repeat_num=100,
               mode='bilinear', padding_mode='zeros', align_corners=False):
    """前向+反向整体耗时测试 + 全量正确性对比"""
    h_out = h_out or h_in
    w_out = w_out or w_in

    print(f"---------------fp32, N={batchsize}, C={channels}, {h_in}x{w_in} -> {h_out}x{w_out}---------------------")

    x1, x2, grid1, grid2, grad = get_data(
        batchsize, channels, h_in, w_in, h_out, w_out, align_corners=align_corners
    )
    f1, f2 = get_model(mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    interp_map, padding_map = _get_interp_padding_map()

    # torch 参考
    y1 = f1(x1, grid1)
    y1.backward(grad)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(repeat_num):
        x1.grad = None
        grid1.grad = None
        y1 = f1(x1, grid1)
        y1.backward(grad)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) * 1000 / repeat_num
    print('torch_grid_sample', 'average time is', torch_time, 'ms')

    # lightop fuse
    y2 = f2(x2, grid2, interp_map[mode], padding_map[padding_mode], align_corners)
    y2.backward(grad)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(repeat_num):
        x2.grad = None
        grid2.grad = None
        y2 = f2(x2, grid2, interp_map[mode], padding_map[padding_mode], align_corners)
        y2.backward(grad)
    torch.cuda.synchronize()
    fuse_time = (time.time() - start_time) * 1000 / repeat_num
    print('fuse_grid_sample', 'average time is', fuse_time, 'ms')

    # 误差检查
    print('output allclose:', allclose(y1, y2, rtol=1e-05, atol=1e-05))
    print('x.grad allclose:', allclose(x1.grad, x2.grad, rtol=1e-05, atol=1e-05))
    print('grid.grad allclose:', allclose(grid1.grad, grid2.grad, rtol=1e-05, atol=1e-05))


def test_profile(batchsize, channels, h_in, w_in, h_out, w_out,
                 repeat_num=10, use_half=False,
                 mode='bilinear', padding_mode='zeros', align_corners=False):
    x1, x2, grid1, grid2, grad = get_data(
        batchsize, channels, h_in, w_in, h_out, w_out, use_half=use_half, align_corners=align_corners
    )
    f1, f2 = get_model(mode=mode, padding_mode=padding_mode, align_corners=align_corners, use_half=use_half)
    interp_map = {'bilinear': INTERP_BILINEAR, 'nearest': INTERP_NEAREST, 'bicubic': INTERP_BICUBIC}
    padding_map = {'zeros': PADDING_ZEROS, 'border': PADDING_BORDER, 'reflection': PADDING_REFLECTION}

    with torch.autograd.profiler.profile(enabled=True, use_device="cuda", record_shapes=False,
                                         use_kineto=True, profile_memory=False) as prof1:
        for i in range(repeat_num):
            y1 = f1(x1, grid1)
            y1.backward(grad)
            torch.cuda.synchronize()
    prof1.export_chrome_trace('./profile_torch_grid_sample.json')

    with torch.autograd.profiler.profile(enabled=True, use_device="cuda", record_shapes=False,
                                         use_kineto=True, profile_memory=False) as prof2:
        for i in range(repeat_num):
            y2 = f2(x2, grid2, interp_map[mode], padding_map[padding_mode], align_corners)
            y2.backward(grad)
            torch.cuda.synchronize()
    prof2.export_chrome_trace('./profile_fuse_grid_sample.json')


if __name__ == "__main__":
    use_half = True
    batchsize, channels = 48, 64
    h_in, w_in = 32, 64
    h_out, w_out = 128, 248

    # 前向对比（速度 + 输出正确性）
    test_speed_forward(batchsize, channels, h_in, w_in, h_out, w_out,
                      repeat_num=100,
                      mode='bilinear', padding_mode='zeros', align_corners=True)

    # 反向对比（速度 + 梯度正确性）
    test_speed_backward(batchsize, channels, h_in, w_in, h_out, w_out,
                       repeat_num=100,
                       mode='bilinear', padding_mode='zeros', align_corners=True)

    # 前向+反向整体耗时（可选）
    # test_speed(batchsize, channels, h_in, w_in, h_out, w_out,
    #            repeat_num=100,
    #            mode='bilinear', padding_mode='zeros', align_corners=True)

    # 性能分析 (按需取消注释)
    # test_profile(batchsize, channels, h_in, w_in, h_out, w_out,
    #              repeat_num=10, use_half=use_half,
    #              mode='bilinear', padding_mode='zeros', align_corners=False)
