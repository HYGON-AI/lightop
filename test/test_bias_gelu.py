import torch
import torch.nn.functional as F
import os
from lightop import FuseBiasGelu
import time
import random

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):

    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def get_data(batchsize=128, seqs=50, cols=128,use_half=False):

    if use_half:
        bias = torch.rand(cols).cuda().half()
        x = torch.rand((batchsize, seqs, cols)).cuda().half()
        grad = torch.rand((batchsize, seqs, cols)).cuda().half()
    else:
        bias = torch.rand(cols).cuda()
        x = torch.rand((batchsize, seqs, cols)).cuda()
        grad = torch.rand((batchsize, seqs, cols)).cuda()

    bias.requires_grad_(True)
    x.requires_grad_(True)
    x1=x.clone().detach().requires_grad_()
    x2=x.clone().detach().requires_grad_()
    bias1=bias.clone().detach().requires_grad_()
    bias2=bias.clone().detach().requires_grad_()
    return x1,x2,bias1,bias2,grad

def get_model(use_half=False):

    if use_half:
        fuse_bias_gelu=FuseBiasGelu().cuda().half()
        torch_bias_gelu=lambda x, b:F.gelu(x + b).half()
    else:
        fuse_bias_gelu=FuseBiasGelu().cuda()
        torch_bias_gelu=lambda x, b:F.gelu(x + b)
        
    return torch_bias_gelu,fuse_bias_gelu


def test_profile(batchsize, seqs, cols,repeat_num,use_half=False):
    
    x1,x2,bias1,bias2,grad=get_data(batchsize, seqs, cols,use_half=use_half)
    f1,f2=get_model(use_half=use_half)

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof1:
        for i in range(repeat_num):
            bias1.grad = None
            x1.grad = None
            y1 = f1(x1, bias1)
            y1.backward(grad)
            torch.cuda.synchronize()
    prof1.export_chrome_trace('./profile_torch_bias_gelu.json')
    
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof2:
        for i in range(repeat_num):
            bias2.grad = None
            x2.grad = None
            y2 = f2(x2, bias2)
            y2.backward(grad)
            torch.cuda.synchronize()
    prof2.export_chrome_trace('./profile_fuse_bias_gelu.json')  

def test_speed(batchsize, seqs, cols,repeat_num,use_half=False):

    x1,x2,bias1,bias2,grad=get_data(batchsize, seqs, cols,use_half=use_half)
    f1,f2=get_model(use_half=use_half)

    start_time = time.time()
    for i in range(repeat_num):
        bias1.grad = None
        x1.grad = None
        y1 = f1(x1, bias1)
        y1.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()
    fc1_espl = end_time - start_time
    print('torch_bias_gelu', 'averange time is', fc1_espl *1000/ repeat_num, 'ms')
    
    start_time = time.time()
    for i in range(repeat_num):
        bias2.grad = None
        x2.grad = None
        y2 = f2(x2, bias2)
        y2.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()    
    fc2_espl = end_time - start_time
    print('fuse_bias_gelu', 'averange time is', fc2_espl *1000/ repeat_num, 'ms')

    #误差计算
    if use_half:
        print(allclose(y1,y2,rtol=2e-02, atol=2e-02))
        print(allclose(x1.grad,x2.grad,rtol=2e-02, atol=2e-02))
        print(allclose(bias1.grad,bias2.grad,rtol=2e-02, atol=2e-02))
    else:
        print(allclose(y1,y2,rtol=1e-05, atol=1e-05))
        print(allclose(x1.grad,x2.grad,rtol=1e-05, atol=1e-05))
        print(allclose(bias1.grad,bias2.grad,rtol=1e-05, atol=1e-05))
    
if __name__ == "__main__":

    #设置基础参数
    batchsize, seqs, cols = 1, 2048, 3072
    #耗时测试
    test_speed(batchsize, seqs, cols,repeat_num=1000,use_half=False)
    #性能分析
    test_profile(batchsize, seqs, cols,repeat_num=100,use_half=False)



