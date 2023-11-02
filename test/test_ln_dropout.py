import torch
import time
import os
import torch.nn.functional as F
from lightop.fuselndropout import LnDropout


class Torch_LnDropout(torch.nn.Module):

    def __init__(self,cols,dropout_p):
        super(Torch_LnDropout, self).__init__()
        self.cols = cols
        self.LayerNorm = torch.nn.LayerNorm(cols)
        self.Dropout =  torch.nn.Dropout(p=dropout_p)

    def forward(self,x):
        out=self.Dropout(x)
        out=self.LayerNorm(out)
        return out

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):

    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def get_data(batchsize=2, seqs=128, cols=128,use_half=False):
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

def get_model(cols=1,dropout_p=0.5,use_half=False):

    if use_half:
        fuse_ln_dropout=LnDropout(cols,dropout_p).cuda().half()
        torch_ln_dropout=Torch_LnDropout(cols,dropout_p).cuda().half()

    else:
       
        fuse_ln_dropout=LnDropout(cols,dropout_p).cuda()
        torch_ln_dropout=Torch_LnDropout(cols,dropout_p).cuda()
       
    return torch_ln_dropout,fuse_ln_dropout


def test_profile(batchsize, seqs, cols,dropout_p,repeat_num,use_half=False):
    
    x1,x2,grad=get_data(batchsize, seqs, cols,use_half=use_half)
    f1,f2=get_model(cols=cols,dropout_p=dropout_p,use_half=use_half)

    torch.manual_seed(0)
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof1:
        for i in range(repeat_num):
            x1.grad = None
            y1 = f1(x1)
            y1.backward(grad)
            torch.cuda.synchronize()
    prof1.export_chrome_trace('./profile_torch_ln_dropout.json')

    torch.manual_seed(0)
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof2:
        for i in range(repeat_num):
            x2.grad = None
            y2 = f2(x2)
            y2.backward(grad)
            torch.cuda.synchronize()
    prof2.export_chrome_trace('./profile_fuse_ln_dropout.json')  

def test_speed(batchsize,seqs,cols,dropout_p,repeat_num,use_half=False):
    # torch.manual_seed(0)
    x1,x2,grad=get_data(batchsize, seqs, cols,use_half=use_half)
    f1,f2=get_model(cols=cols,dropout_p=dropout_p,use_half=use_half)
    y1 = f1(x1)
    y1.backward(grad)
    torch.cuda.synchronize()
    torch.manual_seed(0)
    start_time = time.time()
    for i in range(repeat_num):
        x1.grad = None
        f1.LayerNorm.weight.grad=None
        f1.LayerNorm.bias.grad=None
        y1 = f1(x1)
        y1.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()
    fc1_espl = end_time - start_time
    print('torch_ln_dropout', 'averange time is', fc1_espl *1000/ repeat_num, 'ms')
    y2 = f2(x2)
    y2.backward(grad)
    torch.cuda.synchronize()

    torch.manual_seed(0)
    start_time = time.time()
    for i in range(repeat_num):
        x2.grad = None
        f2.weight.grad=None
        f2.bias.grad=None
        y2 = f2(x2)
        y2.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()    
    fc2_espl = end_time - start_time
    print('fuse_ln_dropout', 'averange time is', fc2_espl *1000/ repeat_num, 'ms')

    #误差计算
    if use_half:
        print(allclose(y1,y2,rtol=2e-02, atol=2e-02))
        print(allclose(x1.grad,x2.grad,rtol=2e-02, atol=2e-02))
        print(allclose(f1.LayerNorm.weight.grad,f2.weight.grad,rtol=2e-02, atol=2e-02))
        print(allclose(f1.LayerNorm.bias.grad,f2.bias.grad,rtol=2e-02, atol=2e-02))
        
    else:
        print(allclose(y1,y2,rtol=1e-05, atol=1e-05))
        print(allclose(x1.grad,x2.grad,rtol=1e-05, atol=1e-05))
        print(allclose(f1.LayerNorm.weight.grad,f2.weight.grad,rtol=1e-04, atol=1e-04))
        print(allclose(f1.LayerNorm.bias.grad,f2.bias.grad,rtol=1e-05, atol=1e-05))
    
if __name__ == "__main__":

    #设置基础参数
    batchsize,seqs, cols = 1,2048,1536
    dropout_p=0.1
    #耗时测试
    test_speed(batchsize, seqs,cols,dropout_p,repeat_num=100,use_half=False)
    #性能分析
    test_profile(batchsize, seqs,cols,dropout_p,repeat_num=10,use_half=False)
