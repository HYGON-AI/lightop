import os
import time
import torch
import torch.nn.functional as F

from lightop.fuse_norms_add_drops import FusedLnRmsNormsAddDropout
from apex.normalization import FusedLayerNorm, MixedFusedRMSNorm

torch.set_printoptions(precision=7, )


# ################################## 准确性检测和速度测试 ##################################
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """ 检测两个tensor之间的误差 """
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def get_dropout_fraction(dropout_mask, dst_shape, drop_rate: float = 0.0):
    """
    用于检测dropout，使用大数定律检测其dropout的比例
    dropout_mask: bool. True means keep, False means drop.
    """
    assert dropout_mask is not None
    assert dropout_mask.shape == dst_shape

    dropped_total = dropout_mask.sum()  # all keep number
    tensor_total = dropout_mask.flatten().shape[0]
    if tensor_total <= 1:  # 没有使用dropmask
        return

    drop_actual_rate = 1.0 - dropped_total / tensor_total
    print(bool(torch.abs(drop_rate-drop_actual_rate) < 0.01), '-- drop_rate = {:.6f}, drop_actual_rate = {:.6f}'.format(drop_rate, drop_actual_rate))


def test_op1in_speed(op, op_name, use_half=False):
    if use_half:
        x1 = x0.half().clone().detach().requires_grad_()
        grad1 = grad.half().clone().detach().requires_grad_()
        op.half()
    else:
        x1 = x0.clone().detach().requires_grad_()
        grad1 = grad.clone().detach().requires_grad_()
        op.float()

    if do_test_speed:
        # warmup
        for i in range(10):
            out = op(x1)
            out.backward(grad1)
            torch.cuda.synchronize()

        forward_time, backward_time = 0, 0
        for i in range(int(repeat_num)):
            time_t1 = time.time()
            out = op(x1)
            torch.cuda.synchronize()
            time_t2 = time.time()
            forward_time += time_t2 - time_t1

            out.backward(grad1)
            torch.cuda.synchronize()
            time_t3 = time.time()
            backward_time += time_t3 - time_t2

        coef = 1000.0 / repeat_num
        print("test_op1in_speed {} forward avg time: {:.6f}ms, backward avg time: {:.6f}ms, total avg time: {:.6f}ms".
              format(op_name, coef * forward_time, coef * backward_time, coef * (forward_time + backward_time)))

    # #### test accuracy
    x1.grad=None
    out = op(x1)
    out.backward(grad1)
    torch.cuda.synchronize()

    return out, x1, op


def test_op2in_speed(op, op_name, use_half=False):
    if use_half:
        x1 = x0.half().clone().detach().requires_grad_()
        y1 = y0.half().clone().detach().requires_grad_()
        grad1 = grad.half().clone().detach().requires_grad_()
        op.half()
    else:
        x1 = x0.clone().detach().requires_grad_()
        y1 = y0.clone().detach().requires_grad_()
        grad1 = grad.clone().detach().requires_grad_()
        op.float()

    # #### test speed
    if do_test_speed:
        # warmup
        for i in range(10):
            out = op(x1, y1)
            out.backward(grad1)
            torch.cuda.synchronize()

        forward_time, backward_time = 0, 0
        for i in range(int(repeat_num)):
            # forward
            time_t1 = time.time()
            out = op(x1, y1)
            torch.cuda.synchronize()
            time_t2 = time.time()
            forward_time += time_t2 - time_t1
            # backward
            out.backward(grad1)
            torch.cuda.synchronize()
            time_t3 = time.time()
            backward_time += time_t3 - time_t2

        coef = 1000.0 / repeat_num
        print("test_op2in_speed {} forward avg time: {:.6f}ms, backward avg time: {:.6f}ms, total avg time: {:.6f}ms".
              format(op_name, coef * forward_time, coef * backward_time, coef * (forward_time + backward_time)))

    # #### test accuracy
    x1.grad=None
    y1.grad=None
    out = op(x1, y1)
    out.backward(grad1)
    torch.cuda.synchronize()

    return out, x1, y1, op


class TorchLnAddDropout(torch.nn.Module):
    """ torch的结构"""

    def __init__(self, col_size, norm_type='ln', use_drop=True, dropout_p=0.2):
        super(TorchLnAddDropout, self).__init__()
        assert norm_type in ['ln', 'none']
        self.norm_type = norm_type
        self.weight = None
        self.bias = None

        if self.norm_type == 'ln':
            self.torch_ln = torch.nn.LayerNorm(col_size)
        else:
            self.torch_ln = None

        self.torch_dropout = torch.nn.Dropout(p=dropout_p)
        self.use_drop = use_drop

    def forward(self, x, residual=None):
        if self.use_drop:
            x = self.torch_dropout(x)

        if residual is not None:
            out = torch.add(x, residual)
        else:
            out = x

        if self.norm_type == 'ln':
            out = self.torch_ln(out)

        return out


class ApexLnRmsAddDropout(torch.nn.Module):
    """ torch的结构"""

    def __init__(self, col_size, norm_type='ln', use_drop=True, dropout_p=0.2):
        super(ApexLnRmsAddDropout, self).__init__()
        assert norm_type in ['ln', 'rms', 'none']
        self.norm_type = norm_type
        if self.norm_type == 'ln':
            self.torch_norm = FusedLayerNorm(col_size)
        elif self.norm_type == 'rms':
            self.torch_norm = MixedFusedRMSNorm(col_size)
        else:
            self.torch_norm = None

        self.torch_dropout = torch.nn.Dropout(p=dropout_p)
        self.use_drop = use_drop

    def forward(self, x, residual=None):
        if self.use_drop:
            x = self.torch_dropout(x)

        if residual is not None:
            out = torch.add(x, residual)
        else:
            out = x

        if self.norm_type != 'none':
            out = self.torch_norm(out)

        return out


def test_speed_accuracy_ln_rms_add(norm_type: str = 'rms', use_residual=True,  use_drop = True, use_half=True):
    print('testing param: norm_type={}, use_residual={}, use_drop={}, use_half={} ...'.format(norm_type, use_residual,use_drop, use_half))

    # #### 设置各种需要的函数 ####
    drop_rate = 0.0  # 准确性测试不考虑dropout
    fused_module = FusedLnRmsNormsAddDropout(norm_type=norm_type, normalized_shape=cols, normalized_affine=True,
                                             use_residual=use_residual, use_dropout=use_drop, drop_prob=drop_rate).to(device)

    # #### 运行并测试速度
    if norm_type == 'ln':
        apex_module = ApexLnRmsAddDropout(cols, norm_type=norm_type, use_drop=use_drop, dropout_p=drop_rate).to(device)
        torch_module = TorchLnAddDropout(cols, norm_type=norm_type, use_drop=use_drop, dropout_p=drop_rate).to(device)
        if use_residual:  # ln+add
            apex_out, apex_x, apex_y, apex_op = test_op2in_speed(apex_module, op_name='apex {}+add'.format(norm_type), use_half=use_half)
            torch_out, torch_x, torch_y, torch_op = test_op2in_speed(torch_module, op_name='torch {}+add'.format(norm_type), use_half=use_half)
            fused_out, fused_x, fused_y, fused_op = test_op2in_speed(fused_module, op_name='fused {}+add'.format(norm_type), use_half=use_half)
            # 误差计算
            if use_half:
                print(allclose(fused_out, torch_out, rtol=half_rtol, atol=half_atol))
                print(allclose(fused_x.grad, torch_x.grad, rtol=half_rtol, atol=half_atol))
                print(allclose(fused_y.grad, torch_y.grad, rtol=half_rtol, atol=half_atol))
                print(allclose(fused_op.weight.grad, torch_op.torch_ln.weight.grad, rtol=half_rtol, atol=half_atol))
                print(allclose(fused_op.bias.grad, torch_op.torch_ln.bias.grad, rtol=half_rtol, atol=half_atol))
            else:
                print(allclose(fused_out, torch_out, rtol=float_rtol, atol=float_atol))
                print(allclose(fused_x.grad, torch_x.grad, rtol=float_rtol, atol=float_atol))
                print(allclose(fused_y.grad, torch_y.grad, rtol=float_rtol, atol=float_atol))
                print(allclose(fused_op.weight.grad, torch_op.torch_ln.weight.grad, rtol=float_rtol_wb_grad, atol=float_atol_wb_grad))
                print(allclose(fused_op.bias.grad, torch_op.torch_ln.bias.grad, rtol=float_rtol_wb_grad, atol=float_atol_wb_grad))
        else:  # ln
            apex_out, apex_x, apex_op = test_op1in_speed(apex_module, op_name='apex {}'.format(norm_type), use_half=use_half)
            torch_out, torch_x, torch_op = test_op1in_speed(torch_module, op_name='torch {}'.format(norm_type), use_half=use_half)
            fused_out, fused_x, fused_op = test_op1in_speed(fused_module, op_name='fused {}'.format(norm_type), use_half=use_half)
            # 误差计算
            if use_half:
                print(allclose(fused_out, torch_out, rtol=half_rtol, atol=half_atol))
                print(allclose(fused_x.grad, torch_x.grad, rtol=half_rtol, atol=half_atol))
                print(allclose(fused_op.weight.grad, torch_op.torch_ln.weight.grad, rtol=half_rtol, atol=half_atol))
                print(allclose(fused_op.bias.grad, torch_op.torch_ln.bias.grad, rtol=half_rtol, atol=half_atol))
            else:
                print(allclose(fused_out, torch_out, rtol=float_rtol, atol=float_atol))
                print(allclose(fused_x.grad, torch_x.grad, rtol=float_rtol, atol=float_atol))
                print(allclose(fused_op.weight.grad, torch_op.torch_ln.weight.grad, rtol=float_rtol_wb_grad, atol=float_atol_wb_grad))
                print(allclose(fused_op.bias.grad, torch_op.torch_ln.bias.grad, rtol=float_rtol_wb_grad, atol=float_atol_wb_grad))
    elif norm_type == 'rms':
        apex_module = ApexLnRmsAddDropout(cols, norm_type=norm_type, use_drop=use_drop, dropout_p=drop_rate).to(device)
        if use_residual:  # rms+add
            apex_out, apex_x, apex_y, apex_op = test_op2in_speed(apex_module, op_name='apex {}+add'.format(norm_type), use_half=use_half)
            fused_out, fused_x, fused_y, fused_op = test_op2in_speed(fused_module, op_name='fused {}+add'.format(norm_type), use_half=use_half)
            # 误差计算
            if use_half:
                print(allclose(fused_out, apex_out, rtol=half_rtol, atol=half_atol))
                print(allclose(fused_x.grad, apex_x.grad, rtol=half_rtol, atol=half_atol))
                print(allclose(fused_y.grad, apex_y.grad, rtol=half_rtol, atol=half_atol))
                print(allclose(fused_op.weight.grad, apex_op.torch_norm.weight.grad, rtol=half_rtol, atol=half_atol))
            else:
                print(allclose(fused_out, apex_out, rtol=float_rtol, atol=float_atol))
                print(allclose(fused_x.grad, apex_x.grad, rtol=float_rtol, atol=float_atol))
                print(allclose(fused_y.grad, apex_y.grad, rtol=float_rtol, atol=float_atol))
                print(allclose(fused_op.weight.grad, apex_op.torch_norm.weight.grad, rtol=float_rtol_wb_grad, atol=float_atol_wb_grad))
        else:  # rms
            apex_out, apex_x, apex_op = test_op1in_speed(apex_module, op_name='apex {}'.format(norm_type), use_half=use_half)
            fused_out, fused_x, fused_op = test_op1in_speed(fused_module, op_name='fused {}'.format(norm_type), use_half=use_half)
            # 误差计算
            if use_half:
                print(allclose(fused_out, apex_out, rtol=half_rtol, atol=half_atol))
                print(allclose(fused_x.grad, apex_x.grad, rtol=half_rtol, atol=half_atol))
                print(allclose(fused_op.weight.grad, apex_op.torch_norm.weight.grad, rtol=half_rtol, atol=half_atol))
            else:
                print(allclose(fused_out, apex_out, rtol=float_rtol, atol=float_atol))
                print(allclose(fused_x.grad, apex_x.grad, rtol=float_rtol, atol=float_atol))
                print(allclose(fused_op.weight.grad, apex_op.torch_norm.weight.grad, rtol=float_rtol_wb_grad, atol=float_atol_wb_grad))
    else:
        torch_module = TorchLnAddDropout(cols, norm_type=norm_type, use_drop=use_drop, dropout_p=drop_rate).to(device)
        if use_residual:
            fused_out, fused_x, fused_y, fused_op = test_op2in_speed(fused_module, op_name='fused add'.format(norm_type), use_half=use_half)
            torch_out, torch_x, torch_y, torch_op = test_op2in_speed(torch_module, op_name='torch add'.format(norm_type), use_half=use_half)
            # 误差计算
            if use_half:
                print(allclose(fused_out, torch_out, rtol=half_rtol, atol=half_atol))
                print(allclose(fused_x.grad, torch_x.grad, rtol=half_rtol, atol=half_atol))
                print(allclose(fused_y.grad, torch_y.grad, rtol=half_rtol, atol=half_atol))
            else:
                print(allclose(fused_out, torch_out, rtol=float_rtol, atol=float_atol))
                print(allclose(fused_x.grad, torch_x.grad, rtol=float_rtol, atol=float_atol))
                print(allclose(fused_y.grad, torch_y.grad, rtol=float_rtol, atol=float_atol))
        else:
            print('do nothing about norm add dropout')


def test_dropout_accuracy_ln_rms_add_drops(norm_type: str = 'rms', use_residual=True, drop_rate=0.2, use_half=True):
    print('testing param: norm_type={}, use_residual={}, use_half={}, use_drop={} ...'.format(norm_type, use_residual, True, use_half))

    fused_module = FusedLnRmsNormsAddDropout(norm_type=norm_type, normalized_shape=cols, normalized_affine=True,
                                             use_residual=use_residual, use_dropout=True, drop_prob=drop_rate).to(device)
    if use_half:
        x1 = x0.half().clone().detach().requires_grad_()
        y1 = y0.half().clone().detach().requires_grad_()
        fused_module = fused_module.half()
    else:
        x1 = x0.clone().detach().requires_grad_()
        y1 = y0.clone().detach().requires_grad_()
        fused_module = fused_module.float()

    if use_residual:
        out = fused_module(x1,y1)
    else:
        out = fused_module(x1)

    drop_mask = fused_module.drop_mask
    get_dropout_fraction(drop_mask, dst_shape=x1.shape , drop_rate=drop_rate)


def test_speed_rms_add_drop_by_profiler(norm_type: str = 'rms'):
    x1 = x0.clone().detach().requires_grad_()
    x2 = x0.clone().detach().requires_grad_()
    y1 = y0.clone().detach().requires_grad_()
    y2 = y0.clone().detach().requires_grad_()

    assert norm_type in ['ln', 'rms']
    dropout_rate = 0.2
    fuse_module = FusedLnRmsNormsAddDropout(norm_type=norm_type, normalized_shape=cols, normalized_affine=True,
                                            use_residual=True, use_dropout=True, drop_prob=dropout_rate).to(device)
    fuse_module_name = 'fused {}_add_dropout'.format(norm_type)
    if norm_type == 'rms':
        op, op_name = ApexLnRmsAddDropout(cols, norm_type=norm_type, use_drop=True, dropout_p=dropout_rate).to(device), 'apex rms_add_dropout'
    elif norm_type == 'ln':
        op, op_name = TorchLnAddDropout(cols, norm_type=norm_type, use_drop=True, dropout_p=dropout_rate).to(device), 'torch ln_add_dropout'
    else:
        raise ValueError

    dst_dir = './profiler_results'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                         profile_memory=False) as prof:
        for _ in range(20):
            out1 = op(x1, y1)
            out1.backward(grad)
            time.sleep(0.1)
    prof.export_chrome_trace('{}/{}_profile_origin.json'.format(dst_dir, op_name))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                         profile_memory=False) as prof2:
        for _ in range(20):
            out2 = fuse_module(x2, y2)
            out2.backward(grad)
            time.sleep(0.1)
    prof2.export_chrome_trace('{}/{}_profile_fuse.json'.format(dst_dir, fuse_module_name))


def main():
    # #################### test speed accuracy ####################
    for norm_type in ['rms', 'ln', 'none']:
        for use_residual in [True, False]:
            for use_half in [True, False]:
                test_speed_accuracy_ln_rms_add(norm_type=norm_type, use_residual=use_residual, use_half=use_half, use_drop=True)
                print('=' * 30)

    # #################### test dropout accuracy ####################
    for norm_type in ['rms', 'ln', 'none']:
        for use_residual in [True, False]:
            for use_half in [True, False]:
                test_dropout_accuracy_ln_rms_add_drops(norm_type=norm_type, use_residual=False, drop_rate=0.5, use_half=True)
                print('=' * 30)

    # #################### test by profiler ####################
    for norm_type in ['rms', 'ln']:
        test_speed_rms_add_drop_by_profiler(norm_type=norm_type)


if __name__ == "__main__":
    repeat_num = 1e2
    batchsize, seqs, cols = 64, 1000, 1024
    # batchsize, seqs, cols = 32, 64, 512
    # batchsize, seqs, cols = 128, 100, 132

    device = torch.device('cuda')
    torch.manual_seed(1)
    x0 = torch.rand([batchsize, seqs, cols]).cuda()
    y0 = torch.rand([batchsize, seqs, cols]).cuda()
    grad = torch.rand([batchsize, seqs, cols]).cuda()

    half_rtol, half_atol = 2e-02, 2e-02
    float_rtol, float_atol = 1e-05, 1e-05
    float_rtol_wb_grad, float_atol_wb_grad = 1e-04, 1e-04

    do_test_speed = True

    main()
