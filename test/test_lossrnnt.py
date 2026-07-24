import os
import time

import numpy as np
import torch
try:
    import torchaudio
except ModuleNotFoundError:
    torchaudio = None

from lightop.loss import RNNTLoss as lightop_RNNTLoss

torch.set_printoptions(precision=10)

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

# ################################### test speed ###################################

def get_data_v2(use_cuda=True, use_half=False, **args):
    def get_data(N_val=6, T_val=50, L_val=20, C_val=10, solid_value=True):
        # # Target are to be un-padded
        # N_val = 6  # Batch size
        # T_val = 50  # probs sequence length max target length
        # L_val = 20  # max target length
        # C_val = 10  # Number of classes (including blank)
        #
        probs = torch.rand(N_val, T_val, L_val + 1, C_val).detach()  # (batch, max seq length, max target length + 1, class)
        # Initialize random batch of targets (0 = blank, 1:C_val = classes)
        target = torch.randint(low=1, high=C_val, size=(N_val, L_val), dtype=torch.int)  # (batch, max target length)
        if solid_value:
            # #### 固定长度
            probs_lengths = torch.full(size=(N_val,), fill_value=T_val, dtype=torch.int)  # (batch)
            target_lengths = torch.full(size=(N_val,), fill_value=L_val, dtype=torch.int)  # (batch)
        else:
            # #### 非定长
            probs_lengths = torch.randint(low=max(1, T_val // 2), high=T_val, size=(N_val,), dtype=torch.int)  # (batch)
            target_lengths = torch.randint(low=L_val // 2 + 1, high=L_val, size=(N_val,), dtype=torch.int)  # (batch)
            for i in range(N_val):
                target_lengths[i] = torch.max(target_lengths[i], probs_lengths[i] + 1)
            probs_lengths[1] = T_val
            target_lengths[1] = L_val

        return probs, target, probs_lengths, target_lengths

    probs, target, probs_lengths, target_lengths = get_data(**args)
    if use_half:
        probs = probs.half()
    if use_cuda:
        probs = probs.cuda()
        target = target.cuda()
        probs_lengths = probs_lengths.cuda()
        target_lengths = target_lengths.cuda()

    return probs, target, probs_lengths, target_lengths


def single_op_test(repeat_num, transform, p, gt, p_len, gt_len, name='torch'):
    for i in range(10):
        p1 = p.clone().requires_grad_()
        loss = transform(p1, gt, p_len, gt_len)
        loss.backward()

    torch.cuda.synchronize()
    time_st = time.perf_counter()
    for i in range(repeat_num):
        p1 = p.clone().requires_grad_()
        loss = transform(p1, gt, p_len, gt_len)
        loss.backward()
    torch.cuda.synchronize()
    time_used = (time.perf_counter() - time_st) * 1e6 / repeat_num
    print('\ngpu ctc {} using time:{:.4f}us'.format(name, time_used))

    p1_grad = p1.grad.detach().cpu()
    print('torch {} loss='.format(name), loss, 'grad shape={}'.format(p1_grad.shape))
    return p1_grad, loss


def test_compare_gpu(repeat_num=100, use_half=False, **args):
    if torchaudio is None:
        print("SKIP: torchaudio is not installed; RNNT correctness is covered by test_loss_existing_api.py")
        return
    p, gt, p_len, gt_len = get_data_v2(use_cuda=True, use_half=use_half, **args)
    print('p shape ', p.shape, p.dtype, ' gt shape ', gt.shape)
    print('p_len ', p_len, p_len.shape)
    print('gt_len ', gt_len, gt_len.shape)

    print('#' * 20)
    transform_torchaudio = torchaudio.transforms.RNNTLoss(blank=0, reduction='mean')
    grad_torch, loss_torch = single_op_test(repeat_num, transform_torchaudio, p, gt, p_len, gt_len, name='torchaudio')
    print('#' * 20)
    transform_lightop = lightop_RNNTLoss(blank=0, reduction='mean')
    grad_lightop, loss_lightop = single_op_test(repeat_num, transform_lightop, p, gt, p_len, gt_len, name='lightop')
    print('#' * 20)
    if use_half:
        print(allclose(loss_torch,loss_lightop,rtol=2e-02, atol=2e-02))
        print(allclose(grad_torch,grad_lightop,rtol=2e-02, atol=2e-02))
    else:
        print(allclose(loss_torch,loss_lightop,rtol=1e-05, atol=1e-05))
        print(allclose(grad_torch,grad_lightop,rtol=1e-05, atol=1e-05))
    print('#' * 20)


def run_gpu_speed(repeat_num=20, use_half=False):
    """ test speed and check result """
    test_compare_gpu(repeat_num=repeat_num, use_half=use_half, N_val=1, T_val=3, L_val=6, C_val=10, solid_value=True)
    print('*' * 20, '\n')
    test_compare_gpu(repeat_num=repeat_num, use_half=use_half, N_val=10, T_val=30, L_val=20, C_val=30, solid_value=True)
    print('*' * 20, '\n')
    test_compare_gpu(repeat_num=repeat_num, use_half=use_half, N_val=10, T_val=30, L_val=20, C_val=4000, solid_value=True)
    print('*' * 20, '\n')
    test_compare_gpu(repeat_num=repeat_num, use_half=use_half, N_val=20, T_val=80, L_val=60, C_val=400, solid_value=True)
    print('*' * 20, '\n')
    test_compare_gpu(repeat_num=repeat_num, use_half=use_half, N_val=20, T_val=100, L_val=60, C_val=400, solid_value=True)
    print('*' * 20, '\n')


def main():
    repeat_num = 100
    run_gpu_speed(repeat_num=repeat_num, use_half=False)
    print('*' * 20, '\n')
    # run_gpu_speed(repeat_num=repeat_num, use_half=True)


if __name__ == "__main__":
    main()

# HIP_VISIBLE_DEVICES=2 python3 test_lossrnnt.py
