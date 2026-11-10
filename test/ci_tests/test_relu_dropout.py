# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import time

import torch
import copy
from torch.autograd import Variable
import torch.nn.functional as F

from lightop.activation import ReluDropout

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):

    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def get_data(batchsize=2, seqs=128, cols=64,hols=64,use_half=False):
    if use_half:
        x = torch.rand((batchsize, seqs, cols,hols)).cuda().half()
        grad = torch.rand((batchsize, seqs, cols,hols)).cuda().half()

    else:
        x = torch.rand((batchsize, seqs, cols,hols)).cuda()
        grad = torch.rand((batchsize, seqs, cols,hols)).cuda()

    x.requires_grad_(True)
    x1=x.clone().detach().requires_grad_()
    x2=x.clone().detach().requires_grad_()
    return x1,x2,grad

def get_model(do_relu=1, dropout_p=0.5,use_half=False):

    if use_half:
        if do_relu:
            torch_reluDropout = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(dropout_p)).cuda().half()
            fuse_reluDropout = ReluDropout(use_relu=True, droprate=dropout_p).cuda().half()
        else:
            torch_reluDropout = torch.nn.Sequential(torch.nn.Dropout(dropout_p)).cuda().half()
            fuse_reluDropout = ReluDropout(use_relu=False, droprate=dropout_p).cuda().half()
    else:
        if do_relu:
            torch_reluDropout = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(dropout_p)).cuda()
            fuse_reluDropout = ReluDropout(use_relu=True, droprate=dropout_p).cuda()
        else:
            torch_reluDropout = torch.nn.Sequential(torch.nn.Dropout(dropout_p)).cuda()
            fuse_reluDropout = ReluDropout(use_relu=False, droprate=dropout_p).cuda()
    return torch_reluDropout, fuse_reluDropout

def test_profile(batchsize,seqs,cols,hols,dropout_p,repeat_num,use_half=False,do_relu=1):

    x1,x2,grad=get_data(batchsize, seqs, cols,hols,use_half=use_half)
    torch_reluDropout, fuse_reluDropout = get_model(do_relu=do_relu, dropout_p=dropout_p,use_half=use_half)
   
    torch.manual_seed(0)
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for _ in range(repeat_num):
            fuseout = fuse_reluDropout(x1)
            fuseout.backward(grad)
            time.sleep(0.02)
    prof.export_chrome_trace('./profile_torch_relu_dropout.json')

    torch.manual_seed(0)
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof2:
        for _ in range(repeat_num):
            out = torch_reluDropout(x2)
            out.backward(grad)
            time.sleep(0.02)
    prof2.export_chrome_trace('./profile_fuse_relu_dropout.json')

def test_speed(batchsize,seqs,cols,hols,dropout_p,repeat_num,use_half=False,do_relu=1):

    x1,x2,grad=get_data(batchsize, seqs, cols,hols,use_half=use_half)
    torch_reluDropout, fuse_reluDropout = get_model(do_relu=do_relu, dropout_p=dropout_p,use_half=use_half)
    torch.manual_seed(0)
    time_st = time.perf_counter()  # 单位为us
    for i in range(repeat_num):
        y1 = torch_reluDropout(x1)
        y1.backward(grad)
    time_using = (time.perf_counter() - time_st) * 1000 / repeat_num
    print("torch use time: {} ms".format(time_using))
    print('========================================================\n')
    time.sleep(0.5)

    torch.manual_seed(0)
    time_st2 = time.perf_counter()
    for i in range(repeat_num):
        y2 = fuse_reluDropout(x2)
        y2.backward(grad)
    time_using2 = (time.perf_counter() - time_st2) * 1000 / repeat_num
    print("ReluDropout use time: {} ms".format(time_using2))
    print('========================================================\n')
    #误差计算
    if use_half:
        print(allclose(y1,y2,rtol=2e-02, atol=2e-02))
        print(allclose(x1.grad,x2.grad,rtol=2e-02, atol=2e-02))
    else:
        print(allclose(y1,y2,rtol=1e-05, atol=1e-05))
        print(allclose(x1.grad,x2.grad,rtol=1e-05, atol=1e-05))
       
if __name__ == "__main__":
    torch.backends.cudnn.enabled = False

    # 设置基础参数
    batchsize, seqs, cols, hols = 128, 200, 256, 2
    dropout_p=0.5
    #耗时测试
    test_speed(batchsize, seqs,cols,hols,dropout_p,repeat_num=1000,use_half=False,do_relu=1)
    #性能分析
    test_profile(batchsize, seqs,cols,hols,dropout_p,repeat_num=100,use_half=False,do_relu=1)
  

# HIP_VISIBLE_DEVICES=1 python3 test/test_relu_dropout.py
