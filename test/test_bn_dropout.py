import torch
import time
import os
import torch.nn.functional as F
from lightop import FuseBatchNormDropout2d

class torch_bn2d_dropout(torch.nn.Module):

    def __init__(self,channel,dropout_p):
        super(torch_bn2d_dropout, self).__init__()
        self.channel = channel
        self.BatchNorm = torch.nn.BatchNorm2d(channel)
        self.Dropout =  torch.nn.Dropout(p=dropout_p)

    def forward(self,x):
        out=self.BatchNorm(x)
        out=self.Dropout(out)
        return out

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):

    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def get_data(batchsize=2, channel=3,cols=8,rows=8,use_half=False):
    if use_half:
        x = torch.rand((batchsize, channel, cols,rows)).cuda().half()
        grad = torch.rand((batchsize, channel, cols,rows)).cuda().half()

    else:
        x = torch.rand((batchsize, channel, cols,rows)).cuda()
        grad = torch.rand((batchsize, channel, cols,rows)).cuda()

    x.requires_grad_(True)
    x1=x.clone().detach().requires_grad_()
    x2=x.clone().detach().requires_grad_()
    return x1,x2,grad

def get_model(channel=1,dropout_p=0.5,use_half=False):

    if use_half:
        fuse_bn_dropout=FuseBatchNormDropout2d(channel,p = dropout_p).cuda().half()
        torch_bn_dropout=torch_bn2d_dropout(channel,dropout_p).cuda().half()

    else:
       
        fuse_bn_dropout=FuseBatchNormDropout2d(channel,p = dropout_p).cuda()
        torch_bn_dropout=torch_bn2d_dropout(channel,dropout_p).cuda()
       
    return torch_bn_dropout,fuse_bn_dropout


def test_profile(batchsize,channel,cols,rows,dropout_p,repeat_num,use_half=False):
    
    x1,x2,grad=get_data(batchsize, channel, cols,rows,use_half=use_half)
    f1,f2=get_model(channel=channel,dropout_p=dropout_p,use_half=use_half)

    torch.manual_seed(1)
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,use_kineto=True,
                                     profile_memory=False) as prof1:
        for i in range(repeat_num):
            x1.grad = None
            y1 = f1(x1)
            y1.backward(grad)
            torch.cuda.synchronize()
    prof1.export_chrome_trace('./profile_torch_bn_dropout.json')

    torch.manual_seed(1)
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,use_kineto=True,
                                     profile_memory=False) as prof2:
        for i in range(repeat_num):
            x2.grad = None
            y2 = f2(x2)
            y2.backward(grad)
            torch.cuda.synchronize()
    prof2.export_chrome_trace('./profile_fuse_bn_dropout.json')  

def test_speed(batchsize,channel,cols,rows,dropout_p,repeat_num,use_half=False):

    x1,x2,grad=get_data(batchsize, channel, cols,rows,use_half=use_half)
    f1,f2=get_model(channel=channel,dropout_p=dropout_p,use_half=use_half)

    torch.manual_seed(1)
    start_time = time.time()
    for i in range(repeat_num):
        x1.grad = None
        y1 = f1(x1)
        y1.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()
    fc1_espl = end_time - start_time
    print('torch_bn_dropout', 'averange time is', fc1_espl *1000/ repeat_num, 'ms')

    torch.manual_seed(1)
    start_time = time.time()
    for i in range(repeat_num):
        x2.grad = None
        y2 = f2(x2)
        y2.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()    
    fc2_espl = end_time - start_time
    print('fuse_bn_dropout', 'averange time is', fc2_espl *1000/ repeat_num, 'ms')
    #误差计算
    if use_half:
        print(allclose(y1,y2,rtol=2e-02, atol=2e-02))
        print(allclose(x1.grad,x2.grad,rtol=2e-02, atol=2e-02))
        print(allclose(f1.BatchNorm.weight.grad,f2.weight.grad,rtol=2e-02, atol=2e-02))
        print(allclose(f1.BatchNorm.bias.grad,f2.bias.grad,rtol=2e-02, atol=2e-02))
        
    else:
        print(allclose(y1,y2,rtol=1e-05, atol=1e-05))
        print(allclose(x1.grad,x2.grad,rtol=1e-05, atol=1e-05))
        print(allclose(f1.BatchNorm.weight.grad,f2.weight.grad,rtol=1e-05, atol=1e-05))
        print(allclose(f1.BatchNorm.bias.grad,f2.bias.grad,rtol=1e-05, atol=1e-05))
    
if __name__ == "__main__":

    #设置基础参数
    batchsize, channel,cols,rows = 2,3,8,8
    dropout_p=0.4
    #耗时测试
    test_speed(batchsize, channel,cols,rows,dropout_p,repeat_num=1000,use_half=False)
    #性能分析
    #test_profile(batchsize, channel,cols,rows,dropout_p,repeat_num=100,use_half=False)
