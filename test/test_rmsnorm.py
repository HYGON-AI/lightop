import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from lightop import RMSNorm,RMSNormTorch


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):

    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def get_data(batchsize=128, seqs=50, cols=128,use_half=False):

    if use_half:
        x = torch.rand((batchsize, seqs, cols)).cuda().half()
        grad = torch.rand((batchsize, seqs, cols)).cuda().half()
    else:
        x = torch.rand((batchsize, seqs, cols)).cuda()
        grad = torch.rand((batchsize, seqs, cols)).cuda()

    x.requires_grad_(True)
    x1=x.clone().detach().requires_grad_()
    x2=x.clone().detach().requires_grad_()
    return x1,x2,grad

def get_model(cols=128,use_half=False):

    if use_half:
        fuse_rmsnorm=RMSNorm(cols).cuda().half()
        torch_rmsnorm=RMSNormTorch(cols).cuda().half()
    else:
        fuse_rmsnorm=RMSNorm(cols).cuda()
        torch_rmsnorm=RMSNormTorch(cols).cuda()
        
    return torch_rmsnorm,fuse_rmsnorm


def test_profile(batchsize, seqs, cols,repeat_num,use_half=False):
    
    x1,x2,grad=get_data(batchsize, seqs, cols,use_half=use_half)
    f1,f2=get_model(cols=cols,use_half=use_half)

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof1:
        for i in range(repeat_num):
            y1 = f1(x1)
            y1.backward(grad)
            torch.cuda.synchronize()
    prof1.export_chrome_trace('./profile_torch_layernorm.json')
    
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof2:
        for i in range(repeat_num):
            y2 = f2(x2)
            y2.backward(grad)
            torch.cuda.synchronize()
    prof2.export_chrome_trace('./profile_fuse_layernorm.json')  

def test_speed(batchsize, seqs, cols,repeat_num,use_half=False):

    x1,x2,grad=get_data(batchsize, seqs, cols,use_half=use_half)
    f1,f2=get_model(cols=cols,use_half=use_half)

    y1 = f1(x1)
    y1.backward(grad)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(repeat_num):
        x1.grad = None
        f1.weight.grad=None
        y1 = f1(x1)
        y1.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()
    fc1_espl = end_time - start_time
    print('torch_rmsnorm', 'averange time is', fc1_espl *1000/ repeat_num, 'ms')
    y2 = f2(x2)
    y2.backward(grad)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(repeat_num):
        x2.grad = None
        f2.weight.grad=None
        y2 = f2(x2)
        y2.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()    
    fc2_espl = end_time - start_time
    print('fuse_rmsnorm', 'averange time is', fc2_espl *1000/ repeat_num, 'ms')

    #误差计算
    if use_half:
        print(allclose(y1,y2,rtol=2e-02, atol=2e-02))
        print(allclose(x1.grad,x2.grad,rtol=2e-02, atol=2e-02))
        print(allclose(f1.weight.grad,f2.weight.grad,rtol=2e-02, atol=2e-02))
    else:
        print(allclose(y1,y2,rtol=1e-05, atol=1e-05))
        print(allclose(x1.grad,x2.grad,rtol=1e-05, atol=1e-05))
        print(allclose(f1.weight.grad,f2.weight.grad,rtol=1e-04, atol=1e-04))

    
if __name__ == "__main__":

    #设置基础参数
    batchsize, seqs, cols = 1, 2048, 5000
    #耗时测试
    test_speed(batchsize, seqs, cols,repeat_num=1000,use_half=False)
    #性能分析
    test_profile(batchsize, seqs, cols,repeat_num=10,use_half=False)