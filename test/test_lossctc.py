import os
import time
import torch

from lightop.lossctc import CTCLoss as lightop_CTCLoss

torch.set_printoptions(precision=10)


def get_data_by_vesion(version=1, use_half=False, use_cuda=True, **args):
    def get_data_v1(N=16, T=50, S=10, C=30):
        # # Target are to be un-padded
        # T = 50  # probs sequence length
        # C = 30  # Number of classes (including blank)
        # N = 16  # Batch size
        # Initialize random batch of probs vectors, for *size = (T,N,C)
        probs = torch.randn(T, N, C).detach()
        probs_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int)
        # Initialize random batch of targets (0 = blank, 1:C = classes)
        # target_lengths = torch.randint(low=1, high=T // 2, size=(N,), dtype=torch.int)
        target_lengths = torch.full(size=(N,), fill_value=S, dtype=torch.int)
        target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.int)
        return probs, target, probs_lengths, target_lengths

    def get_data_v2():
        probs = torch.FloatTensor([[
            [0.3, 0.6, 0.1, 0.5, 0.9], [0.5, 1.2, 0.6, 0.1, 0.3]
        ]]).transpose(0, 1).contiguous()
        target = torch.IntTensor([[1, 2]])
        target_lengths = torch.IntTensor([2])
        probs_lengths = torch.IntTensor([2])
        return probs, target, probs_lengths, target_lengths

    data_dict = {1: get_data_v1(**args), 2: get_data_v2()}
    probs, target, probs_lengths, target_lengths = data_dict[version]
    if use_half:
        probs = probs.half()

    if use_cuda:
        probs = probs.cuda()
        target = target.cuda()
        probs_lengths = probs_lengths.cuda()
        target_lengths = target_lengths.cuda()

    return probs, target, probs_lengths, target_lengths


# ################################### test speed ###################################
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
        if i < 1:
            p1_grad = p1.grad.detach().cpu()
            print('torch {} loss='.format(name), loss, 'grad shape={}'.format(p1_grad.shape))
    torch.cuda.synchronize()
    time_used = (time.perf_counter() - time_st) * 1e6 / repeat_num
    print('\ngpu ctc {} using time:{:.4f}us'.format(name, time_used))
    return p1_grad


def test_compare_gpu(repeat_num=50, use_half=False, **args):
    p, gt, p_len, gt_len = get_data_by_vesion(version=1, use_half=use_half, use_cuda=True, **args)
    print('p shape ', p.shape, p.dtype, ' gt shape ', gt.shape)
    print('p_len ', p_len.shape)
    print('gt_len ', gt_len.shape)
    print('#' * 20, '\n')

    #
    transform_lightop = lightop_CTCLoss(blank=0, reduction='mean', logits_time_major=True)
    for i in range(10):
        p1 = p.clone().requires_grad_()
        loss_lightop = transform_lightop(p1, gt, p_len, gt_len)
        loss_lightop.backward()

    torch.cuda.synchronize()
    time_st = time.perf_counter()
    for i in range(repeat_num):
        p1 = p.clone().requires_grad_()
        loss_lightop = transform_lightop(p1, gt, p_len, gt_len)
        loss_lightop.backward()

    torch.cuda.synchronize()
    time_used = (time.perf_counter() - time_st) * 1e6 / repeat_num
    print('\ngpu ctc lightop using time:{:.4f}us'.format(time_used))
    print('#' * 20, '\n')
    #
    #
    if not use_half:
        transform_torch = torch.nn.CTCLoss(blank=0, reduction='mean')
        for i in range(10):
            p2 = p.clone().log_softmax(-1).requires_grad_()
            loss_torch = transform_torch(p2, gt, p_len, gt_len)
            loss_torch.backward()

        torch.cuda.synchronize()
        time_st = time.perf_counter()
        for i in range(repeat_num):
            p2 = p.clone().log_softmax(-1).requires_grad_()
            loss_torch = transform_torch(p2, gt, p_len, gt_len)
            loss_torch.backward()

        torch.cuda.synchronize()
        time_used = (time.perf_counter() - time_st) * 1e6 / repeat_num
        print('\ngpu ctc torch using time:{:.4f}us'.format(time_used))
        print('#' * 20, '\n')

        grad_lightop = p1.grad.detach().cpu()
        grad_torch = p2.grad.detach().cpu()
        print('torch torch loss_torch=', loss_torch, 'grad shape={}'.format(grad_torch.shape))
        print('torch lightop loss_lightop=', loss_lightop, 'grad shape={}'.format(grad_lightop.shape))
        print('#' * 20, '\n')
        grad_diff = torch.nn.functional.l1_loss(grad_torch, grad_lightop, reduction='none')
        print('grad_diff max mean=', torch.max(grad_diff).data, torch.mean(grad_diff).data, grad_diff.shape)
        loss_diff = torch.nn.functional.l1_loss(loss_torch, loss_lightop, reduction='sum')
        print("loss_diff=", loss_diff)

    print('#' * 20)
    print('#' * 20)
    print('#' * 20, '\n')


def run_test_gpu(repeat_num=50, use_half=False):
    test_compare_gpu(repeat_num=repeat_num, use_half=use_half, N=1, T=6, S=3, C=4)
    print('*' * 20, '\n' * 2)
    test_compare_gpu(repeat_num=repeat_num, use_half=use_half, N=5, T=60, S=30, C=20)
    print('*' * 20, '\n' * 2)
    test_compare_gpu(repeat_num=repeat_num, use_half=use_half, N=10, T=20, S=10, C=400)
    print('*' * 20, '\n' * 2)
    test_compare_gpu(repeat_num=repeat_num, use_half=use_half, N=30, T=80, S=50, C=200)
    print('*' * 20, '\n' * 2)
    test_compare_gpu(repeat_num=repeat_num, use_half=use_half, N=60, T=120, S=80, C=240)
    print('*' * 20, '\n' * 2)
    test_compare_gpu(repeat_num=repeat_num, use_half=use_half, N=100, T=200, S=120, C=400)
    print('*' * 20, '\n' * 2)
    test_compare_gpu(repeat_num=repeat_num, use_half=use_half, N=100, T=400, S=320, C=600)
    print('*' * 20, '\n' * 2)


def main():
    repeat_num = 100
    run_test_gpu(repeat_num, use_half=False)
    # run_test_gpu(repeat_num, use_half=True)


if __name__ == "__main__":
    main()

# HIP_VISIBLE_DEVICES=2 python3 test_lossctc.py
