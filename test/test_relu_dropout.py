import time

import torch
import copy
from torch.autograd import Variable
import torch.nn.functional as F

from lightop.fusereludropout import ReluDropout


def get_model(do_relu=1, do_eval=1, dropout_p=0.5):
    if do_relu:
        torch_reluDropout = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(dropout_p)).cuda()
        fuse_reluDropout = ReluDropout(use_relu=True, droprate=dropout_p).cuda()
    else:
        torch_reluDropout = torch.nn.Sequential(torch.nn.Dropout(dropout_p)).cuda()
        fuse_reluDropout = ReluDropout(use_relu=False, droprate=dropout_p).cuda()

    # eval
    if do_eval:
        torch_reluDropout = torch_reluDropout.eval()
        fuse_reluDropout = fuse_reluDropout.eval()

    return torch_reluDropout, fuse_reluDropout


def check_test():
    # check_test_running(do_relu=1, do_eval=0, dropout_p=0.5, verbose=True)

    # # ####
    # for do_eval in [0, 1]:
    #     check_test_running(do_relu=0, do_eval=do_eval, dropout_p=0.5, verbose=True)

    # #####
    for dropout_p in [0.3, 0.5]:
        for do_relu in [0, 1]:
            for do_eval in [0, 1]:
                check_test_running(do_relu, do_eval, dropout_p, verbose=False)
                print('--------------')


def check_test_running(do_relu=1, do_eval=1, dropout_p=0.5, verbose=True):
    torch_reluDropout, fuse_reluDropout = get_model(do_relu=do_relu, do_eval=do_eval, dropout_p=dropout_p)
    print("check_test start ..., do_relu={}, do_eval={}, dropout_p={}".format(do_relu, do_eval, dropout_p))

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x = Variable(x0, requires_grad=True)
    out = torch_reluDropout(x)
    out.backward(out.clone().detach())
    if verbose:
        print('torch_reluDropout out: ', out[:3, :3, :3])
        print('x.grad: ', x.grad[:3, :3, :3])
        print("check_test torch done ...\n\n")

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    fuse_x = Variable(fuse_x0, requires_grad=True)
    fuse_out = fuse_reluDropout(fuse_x)
    fuse_out.backward(fuse_out.clone().detach())
    if verbose:
        print('fuse_reluDropout fuse_out: ', fuse_out[:3, :3, :3])
        print('fuse_x.grad: ', fuse_x.grad[:3, :3, :3])
        print("check_test fuse_reluDropout done ...")

    # check result
    if out.equal(fuse_out):
        print("result check_test:TRUE")
    else:
        print("result check_test:", out.equal(fuse_out))

    if x.grad.equal(fuse_x.grad):
        print("grad check_test:TRUE")
    else:
        print("grad check_test:", x.grad.equal(fuse_x.grad))


def test_profile(repeat_num=30, do_relu=1, do_eval=0, dropout_p=0.3):
    torch_reluDropout, fuse_reluDropout = get_model(do_relu=do_relu, do_eval=do_eval, dropout_p=dropout_p)

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for _ in range(repeat_num):
            x1 = x0.requires_grad_()
            fuseout = fuse_reluDropout(x1)
            fuseout.backward(fuseout.clone().detach())
            time.sleep(0.02)
    prof.export_chrome_trace('./test_result/resnet_profile_fuse.json')

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof2:
        for _ in range(repeat_num):
            x2 = x0.requires_grad_()
            out = torch_reluDropout(x2)
            out.backward(out.clone().detach())
            time.sleep(0.02)
    prof2.export_chrome_trace('./test_result/resnet_profile_torch.json')


def test_time_compare(repeat_num=30):
    # print("test_time_compare ...")
    # # #####
    # for dropout_p in [0.3, 0.5]:
    #     for do_relu in [0, 1]:
    #         for do_eval in [0, 1]:
    #             test_time_compare_runnint(repeat_num, do_relu, do_eval, dropout_p)
    #             print('--------------')
    # # #####
    test_time_compare_runnint(repeat_num, 1, 0, 0.5)


def test_time_compare_runnint(repeat_num=30, do_relu=1, do_eval=1, dropout_p=0.5):
    torch_reluDropout, fuse_reluDropout = get_model(do_relu=do_relu, do_eval=do_eval, dropout_p=dropout_p)
    print("test_time_compare_runnint..., do_relu={}, do_eval={}, dropout_p={}".format(do_relu, do_eval, dropout_p))

    time_st = time.perf_counter()  # 单位为us
    for i in range(repeat_num):
        x = Variable(x0, requires_grad=True)
        torch.manual_seed(0)
        out = torch_reluDropout(x)
        out.backward(out.clone().detach())
    time_using = (time.perf_counter() - time_st) * 1000 / repeat_num
    print("torch use time: {} ms".format(time_using))
    time.sleep(0.5)
    print('================')

    time_st2 = time.perf_counter()
    for i in range(repeat_num):
        fuse_x = Variable(fuse_x0, requires_grad=True)
        torch.manual_seed(0)
        fuse_out = fuse_reluDropout(fuse_x)
        fuse_out.backward(fuse_out.clone().detach())
    time_using2 = (time.perf_counter() - time_st2) * 1000 / repeat_num
    print("ReluDropout use time: {} ms".format(time_using2))
    print('========================================================\n')


def compare_cuda_detail(do_relu=1, do_eval=1, dropout_p=0.5):
    # from torch.profiler import profile, record_function, ProfilerActivity
    from torch.autograd.profiler import profile

    torch_reluDropout, fuse_reluDropout = get_model(do_relu=do_relu, do_eval=do_eval, dropout_p=dropout_p)
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x = Variable(x0, requires_grad=True)
    with profile(enabled=True, use_cuda=True, record_shapes=True, profile_memory=False) as prof:
        out = torch_reluDropout(x)
        out.backward(grad)
        time.sleep(0.5)
    print(prof.key_averages().table(sort_by="cuda_time_total"))

    fuse_x = Variable(fuse_x0, requires_grad=True)
    with profile(enabled=True, use_cuda=True, record_shapes=True, profile_memory=False) as prof:
        fuse_out = fuse_reluDropout(fuse_x)
        fuse_out.backward(grad)
        time.sleep(0.5)
    print(prof.key_averages().table(sort_by="cuda_time_total"))


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False

    # 设置seed
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置基础参数
    batchsize, seqs, cols, hols = 128, 200, 256, 2
    # x0 = torch.rand([128, 50, 768]).cuda()
    x0 = torch.randn([batchsize, seqs, cols, hols]).cuda()
    grad = torch.randn([batchsize, seqs, cols, hols]).cuda()
    fuse_x0 = copy.deepcopy(x0).cuda()
    print('x0: ', x0.shape)

    # check_test()
    test_profile()
    # test_time_compare(repeat_num=100)
    # compare_cuda_detail()

# HIP_VISIBLE_DEVICES=1 python3 test/test_relu_dropout.py
