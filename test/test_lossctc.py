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


def test_compare_gpu(pre_repeat_num=10, repeat_num=50, use_half=False, **args):
    p, gt, p_len, gt_len = get_data_by_vesion(version=1, use_half=use_half, use_cuda=True, **args)
    print('p shape ', p.shape, p.dtype, ' gt shape ', gt.shape)
    print('p_len ', p_len.shape)
    print('gt_len ', gt_len.shape)
    print('#' * 20, '\n')

    logits_time_major = False
    reduction = 'mean'
    p1_trans = p.clone().permute(1, 0, 2).contiguous()
    print('p shape ', p.shape, ', p1_trans shape ', p1_trans.shape)
    #
    transform_lightop = lightop_CTCLoss(blank=0, reduction=reduction, logits_time_major=logits_time_major)
    for i in range(pre_repeat_num):
        p1 = p.clone().requires_grad_() if logits_time_major else p1_trans.clone().requires_grad_()
        loss_lightop = transform_lightop(p1, gt, p_len, gt_len)
        loss_lightop.backward()

    torch.cuda.synchronize()
    time.sleep(0.01)
    time_st = time.perf_counter()
    for i in range(repeat_num):
        p1 = p.clone().requires_grad_() if logits_time_major else p1_trans.clone().requires_grad_()
        loss_lightop = transform_lightop(p1, gt, p_len, gt_len)
        loss_lightop.backward()

    torch.cuda.synchronize()
    time_used = (time.perf_counter() - time_st) * 1e6 / repeat_num
    print('\ngpu ctc lightop using time:{:.4f}us'.format(time_used))
    print('#' * 20, '\n')
    #
    #
    if not use_half:
        transform_torch = torch.nn.CTCLoss(blank=0, reduction=reduction)
        for i in range(pre_repeat_num):
            p2 = p.clone().log_softmax(-1).requires_grad_()
            loss_torch = transform_torch(p2, gt, p_len, gt_len)
            loss_torch.backward()

        torch.cuda.synchronize()
        time.sleep(0.01)
        time_st = time.perf_counter()
        for i in range(repeat_num):
            p2 = p.clone().log_softmax(-1).requires_grad_()
            loss_torch = transform_torch(p2, gt, p_len, gt_len)
            loss_torch.backward()

        torch.cuda.synchronize()
        time_used = (time.perf_counter() - time_st) * 1e6 / repeat_num
        print('\ngpu ctc torch using time:{:.4f}us'.format(time_used))
        print('#' * 20, '\n')

        grad_lightop = p1.grad.detach().cpu() if logits_time_major else p1.grad.detach().cpu().permute(1, 0, 2)
        grad_torch = p2.grad.detach().cpu()
        print('torch torch loss_torch=', loss_torch, ', grad shape={}'.format(grad_torch.shape))
        print('torch lightop loss_lightop=', loss_lightop, ', grad shape={}'.format(grad_lightop.shape))
        print('#' * 20, '\n')
        grad_diff = torch.nn.functional.l1_loss(grad_torch, grad_lightop, reduction='none')
        print('grad_diff max mean=', torch.max(grad_diff).data, torch.mean(grad_diff).data, grad_diff.shape)
        loss_diff = torch.nn.functional.l1_loss(loss_torch, loss_lightop, reduction='sum')
        print("loss_diff=", loss_diff)

    print('#' * 20, '\n')


def make_dir_for_file(file_path):
    file_dir = os.path.abspath(os.path.dirname(file_path))
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


def test_record_by_profiler(use_half=False, **args):
    p, gt, p_len, gt_len = get_data_by_vesion(version=1, use_half=use_half, use_cuda=True, **args)
    print('p shape ', p.shape, p.dtype, ' gt shape ', gt.shape)
    print('p_len ', p_len.shape)
    print('gt_len ', gt_len.shape)
    print('#' * 20, '\n')

    logits_time_major = False
    p1_trans = p.clone().permute(1, 0, 2).contiguous()
    print('p shape ', p.shape, ', p1_trans shape ', p1_trans.shape)
    #
    transform_lightop = lightop_CTCLoss(blank=0, reduction='mean', logits_time_major=logits_time_major)
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        for i in range(20):
            p1 = p.clone().requires_grad_() if logits_time_major else p1_trans.clone().requires_grad_()
            loss_lightop = transform_lightop(p1, gt, p_len, gt_len)
            loss_lightop.backward()
            time.sleep(0.2)

    dst_json_path = './test_result/profile_lightop_CTCLoss.json'
    make_dir_for_file(dst_json_path)
    prof.export_chrome_trace(dst_json_path)
    print('saving to {} done'.format(dst_json_path))

    if not use_half:
        transform_torch = torch.nn.CTCLoss(blank=0, reduction='mean')
        with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
            for i in range(20):
                p2 = p.clone().log_softmax(-1).requires_grad_()
                loss_torch = transform_torch(p2, gt, p_len, gt_len)
                loss_torch.backward()
                time.sleep(0.2)

        dst_json_path = './test_result/profile_torch_CTCLoss.json'
        make_dir_for_file(dst_json_path)
        prof.export_chrome_trace(dst_json_path)
        print('saving to {} done'.format(dst_json_path))


def run_test_gpu(pre_repeat_num, repeat_num=100, use_half=False):
    test_compare_gpu(pre_repeat_num=pre_repeat_num, repeat_num=repeat_num, use_half=use_half, N=2, T=6, S=3, C=4)
    print('*' * 50, '\n' * 2)
    test_compare_gpu(pre_repeat_num=pre_repeat_num, repeat_num=repeat_num, use_half=use_half, N=5, T=60, S=30, C=20)
    print('*' * 50, '\n' * 2)
    test_compare_gpu(pre_repeat_num=pre_repeat_num, repeat_num=repeat_num, use_half=use_half, N=10, T=20, S=10, C=400)
    print('*' * 50, '\n' * 2)
    test_compare_gpu(pre_repeat_num=pre_repeat_num, repeat_num=repeat_num, use_half=use_half, N=30, T=80, S=50, C=200)
    print('*' * 50, '\n' * 2)
    test_compare_gpu(pre_repeat_num=pre_repeat_num, repeat_num=repeat_num, use_half=use_half, N=60, T=120, S=80, C=240)
    print('*' * 50, '\n' * 2)
    test_compare_gpu(pre_repeat_num=pre_repeat_num, repeat_num=repeat_num, use_half=use_half, N=100, T=200, S=120, C=400)
    print('*' * 50, '\n' * 2)


def main():
    pre_repeat_num = 10
    repeat_num = 100
    # run_test_gpu(pre_repeat_num, repeat_num, use_half=True)
    run_test_gpu(pre_repeat_num, repeat_num, use_half=False)
    # test_record_by_profiler(use_half=False, N=100, T=400, S=320, C=600)


if __name__ == "__main__":
    main()

# HIP_VISIBLE_DEVICES=2 python3 test_lossctc.py
