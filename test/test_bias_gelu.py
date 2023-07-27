import torch
import torch.nn.functional as F
import os
from lightop.fusebiasgelu import FuseBiasGelu
import time
import random

def test_speed(name, seq_len, bsz, dim, n, f1,f2):
    
    bias = torch.rand(dim).cuda()
    x = torch.rand((seq_len, bsz, dim)).cuda()
    grad = torch.rand((seq_len, bsz, dim)).cuda()
    bias.requires_grad_(True)
    x.requires_grad_(True)
    x1=x.clone().detach().requires_grad_()
    x2=x.clone().detach().requires_grad_()
    bias1=bias.clone().detach().requires_grad_()
    bias2=bias.clone().detach().requires_grad_()
    for i in range(10):
        bias1.grad = None
        x1.grad = None
        y1 = f1(x1, bias1)
        y1.backward(grad)
        torch.cuda.synchronize()

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof1:
        start_time = time.time()
        for i in range(n):
            bias1.grad = None
            x1.grad = None
            y1 = f1(x1, bias1)
            y1.backward(grad)
            torch.cuda.synchronize()
        end_time = time.time()
        fc1_espl = end_time - start_time
        print('torch_bias_gelu', 'averange time is', fc1_espl *1000/ n, 'ms')
    prof1.export_chrome_trace('./profile_torch_bias_relu1111.json')
    
    for i in range(10):
        bias2.grad = None
        x2.grad = None
        y2 = f2(x2, bias2)
        y2.backward(grad)
        torch.cuda.synchronize()
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof2:
        start_time = time.time()
        for i in range(n):
            bias2.grad = None
            x2.grad = None
            y2 = f2(x2, bias2)
            y2.backward(grad)
            torch.cuda.synchronize()
        end_time = time.time()    
        fc2_espl = end_time - start_time
        print('fuse_bias_gelu', 'averange time is', fc2_espl *1000/ n, 'ms')
    prof2.export_chrome_trace('./profile_fuse_bias_relu1111.json')  
    print(y1.equal(y2))
    print(x1.grad.equal(x2.grad))
    #相对误差小于1e-5
    print((abs(bias1.grad-bias2.grad)/(abs(bias1.grad)+1e-5)).max().item()<1e-5)

# for _ in range(100):
#     seq_len = random.randint(1, 5)*16
#     bsz = random.randint(1, 10)*64
#     dim = random.randint(1, 5)*512
#     torch.manual_seed(0)
#     fuse_bias_gelu=FuseBiasGelu().cuda()
#     test_speed('bias_gelu', seq_len, bsz, dim, 1000,lambda x, b:F.gelu(x + b) ,fuse_bias_gelu )
fuse_bias_gelu=FuseBiasGelu().cuda()
test_speed('bias_gelu', 2046, 16, 64, 1000,lambda x, b:F.gelu(x + b) ,fuse_bias_gelu )