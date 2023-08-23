import torch
import time
import torch.nn.functional as F
from torch.autograd import Variable
from lightop.fusebiasdropoutadd import FuseBiasDropoutAdd

def rerr(x1, x2):
        return ((x1.float() - x2.float()) / (x1.float() + x2.float() + 1e-6)).abs().max()

def test_speed(name, seq_len, bsz, dim, n, f1,f2):

    torch.manual_seed(0)
    bias = torch.rand(dim).cuda()
    x = torch.rand((seq_len, bsz, dim)).cuda()
    grad = torch.rand((seq_len, bsz, dim)).cuda()
    residual = torch.rand((seq_len, bsz, dim)).cuda()
    bias.requires_grad_(True)
    x.requires_grad_(True)
    residual.requires_grad_(True)
    x1=x.clone().detach().requires_grad_()
    x2=x.clone().detach().requires_grad_()
    bias1=bias.clone().detach().requires_grad_()
    bias2=bias.clone().detach().requires_grad_()
    residual1=residual.clone().detach().requires_grad_()
    residual2=residual.clone().detach().requires_grad_()
    for i in range(10):
        torch.manual_seed(0)
        bias1.grad = None
        x1.grad = None
        residual1.grad = None
        y1 = f1(x1, bias1,residual1)
        y1.backward(grad)
        torch.cuda.synchronize()

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof1:
        start_time = time.time()
        for i in range(n):
            torch.manual_seed(0)
            bias1.grad = None
            x1.grad = None
            residual1.grad = None
            y1 = f1(x1, bias1,residual1)
            y1.backward(grad)
            torch.cuda.synchronize()
        end_time = time.time()
        fc1_espl = end_time - start_time
        print('torch_bias_dropout_add', 'averange time is', fc1_espl *1000/ n, 'ms')
    prof1.export_chrome_trace('./profile_torch_bias_dropout_add.json')
    
    for i in range(10):
        torch.manual_seed(0)
        bias2.grad = None
        x2.grad = None
        residual2.grad = None
        y2 = f2(x2, bias2,residual2)
        y2.backward(grad)
        torch.cuda.synchronize()
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof2:
        start_time = time.time()
        for i in range(n):
            torch.manual_seed(0)
            bias2.grad = None
            x2.grad = None
            residual2.grad = None
            y2 = f2(x2, bias2,residual2)
            y2.backward(grad)
            torch.cuda.synchronize()
        end_time = time.time()    
        fc2_espl = end_time - start_time
        print('fuse_bias_dropout_add', 'averange time is', fc2_espl *1000/ n, 'ms')
    prof2.export_chrome_trace('./profile_bias_dropout_add.json')  
    print(y1.equal(y2))
    print(x1.grad.equal(x2.grad))
    #相对误差
    print( f"error max: bias.grad: {rerr(bias1.grad, bias2.grad)}")
 

fuse_bias_dropout_add=FuseBiasDropoutAdd(droprate=0.5).cuda()
test_speed('bias_dropout_add', 2048, 16, 512, 1000,lambda x, b, r: F.dropout(x + b, 0.5) + r ,fuse_bias_dropout_add)