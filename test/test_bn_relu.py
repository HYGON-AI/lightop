import torch
import time
import os
from lightop.fusebnrelu import FuseBatchNormRelu1d
from lightop.fusebnrelu import FuseBatchNormRelu2d
from lightop.fusebnrelu import FuseBatchNormRelu3d


class torch_bn1d_relu(torch.nn.Module):

    def __init__(self,channel):
        super(torch_bn1d_relu, self).__init__()
        self.channel = channel
        self.BatchNorm = torch.nn.BatchNorm1d(channel)
        self.Relu =  torch.nn.ReLU()

    def forward(self,x):
        out=self.BatchNorm(x)
        out=self.Relu(out)
        return out

class torch_bn2d_relu(torch.nn.Module):

    def __init__(self,channel):
        super(torch_bn2d_relu, self).__init__()
        self.channel = channel
        self.BatchNorm = torch.nn.BatchNorm2d(channel)
        self.Relu =  torch.nn.ReLU()

    def forward(self,x):
        out=self.BatchNorm(x)
        out=self.Relu(out)
        return out
     

class torch_bn3d_relu(torch.nn.Module):

    def __init__(self,channel):
        super(torch_bn3d_relu, self).__init__()
        self.channel = channel
        self.BatchNorm = torch.nn.BatchNorm3d(channel)
        self.Relu =  torch.nn.ReLU()

    def forward(self,x):
        out=self.BatchNorm(x)
        out=self.Relu(out)
        return out

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):

    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def get_data(batchsize=2, channel=3,depth=3, cols=8,rows=8,d_num=3,use_half=False):

    if use_half:
        if d_num==1 :
            x = torch.rand((batchsize,channel)).cuda().half()
            grad = torch.rand((batchsize, channel)).cuda().half()
        elif d_num==2 :
            x = torch.rand((batchsize, channel, cols,rows)).cuda().half()
            grad = torch.rand((batchsize, channel, cols,rows)).cuda().half()
        else:
            x = torch.rand((batchsize, channel, depth,cols,rows)).cuda().half()
            grad = torch.rand((batchsize, channel,depth, cols,rows)).cuda().half()
    else:
        if d_num==1 :
            x = torch.rand((batchsize,channel)).cuda()
            grad = torch.rand((batchsize, channel)).cuda()
        elif d_num==2 :
            x = torch.rand((batchsize, channel, cols,rows)).cuda()
            grad = torch.rand((batchsize, channel, cols,rows)).cuda()
        else:
            x = torch.rand((batchsize, channel,depth, cols,rows)).cuda()
            grad = torch.rand((batchsize, channel,depth, cols,rows)).cuda()
 
    x.requires_grad_(True)
    x1=x.clone().detach().requires_grad_()
    x2=x.clone().detach().requires_grad_()
    return x1,x2,grad

def get_model(channel=1,d_num=3,use_half=False):

    if use_half:
        if d_num==1 :
            fuse_bn_relu=FuseBatchNormRelu1d(channel).cuda().half()
            torch_bn_relu=torch_bn1d_relu(channel).cuda().half()
     
        elif d_num==2 :
            fuse_bn_relu=FuseBatchNormRelu2d(channel).cuda().half()
            torch_bn_relu=torch_bn2d_relu(channel).cuda().half()

        else:
            fuse_bn_relu=FuseBatchNormRelu3d(channel).cuda().half()
            torch_bn_relu=torch_bn3d_relu(channel).cuda().half()
    else:
        if d_num==1 :
            fuse_bn_relu=FuseBatchNormRelu1d(channel).cuda()
            torch_bn_relu=torch_bn1d_relu(channel).cuda()
        elif d_num==2 :
            fuse_bn_relu=FuseBatchNormRelu2d(channel).cuda()
            torch_bn_relu=torch_bn2d_relu(channel).cuda()
        else:
            fuse_bn_relu=FuseBatchNormRelu3d(channel).cuda()
            torch_bn_relu=torch_bn3d_relu(channel).cuda()
    return torch_bn_relu,fuse_bn_relu


def test_profile(batchsize,channel,depth, cols,rows,repeat_num,d_num,use_miopen=False,use_half=False):
    
    x1,x2,grad=get_data(batchsize, channel,depth, cols,rows,d_num=d_num,use_half=use_half)
    f1,f2=get_model(channel=channel,d_num=d_num,use_half=use_half)
    if use_miopen:
      os.environ['HIP_NATIVE_FUSE']="0"
      os.environ['HIP_MIOPEN_FUSE']="1"
    else:
      os.environ['HIP_NATIVE_FUSE']="1"
      os.environ['HIP_MIOPEN_FUSE']="0"
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof1:
        for i in range(repeat_num):
            x1.grad = None
            y1 = f1(x1)
            y1.backward(grad)
            torch.cuda.synchronize()
    prof1.export_chrome_trace('./profile_torch_bn_relu.json')
    
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof2:
        for i in range(repeat_num):
            x2.grad = None
            y2 = f2(x2)
            y2.backward(grad)
            torch.cuda.synchronize()
    prof2.export_chrome_trace('./profile_fuse_bn_relu.json')  

def test_speed(batchsize,channel,depth, cols,rows,repeat_num,d_num,use_miopen=False,use_half=False):

    x1,x2,grad=get_data(batchsize, channel,depth, cols,rows,d_num=d_num,use_half=use_half)
    f1,f2=get_model(channel=channel,d_num=d_num,use_half=use_half)
    if use_miopen:
      os.environ['HIP_NATIVE_FUSE']="0"
      os.environ['HIP_MIOPEN_FUSE']="1"
    else:
      os.environ['HIP_NATIVE_FUSE']="1"
      os.environ['HIP_MIOPEN_FUSE']="0"
    start_time = time.time()
    for i in range(repeat_num):
        x1.grad = None
        y1 = f1(x1)
        y1.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()
    fc1_espl = end_time - start_time
    print('torch_bn_relu', 'averange time is', fc1_espl *1000/ repeat_num, 'ms')
    
    start_time = time.time()
    for i in range(repeat_num):
        x2.grad = None
        y2 = f2(x2)
        y2.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()    
    fc2_espl = end_time - start_time
    print('fuse_bn_relu', 'averange time is', fc2_espl *1000/ repeat_num, 'ms')
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
    batchsize, channel,depth, cols,rows = 2,3,3,8,8
    #输入维度，batchnorm1d(batchsize, channel) 1;batchnorm2d(batchsize, channel,cols,rows)：2，
    # batchnorm3d(batchsize, channel,depth, cols,rows)：3
    d_num=1
    #耗时测试
    test_speed(batchsize, channel,depth, cols,rows,repeat_num=1000,d_num=d_num,use_miopen=False,use_half=False)
    #性能分析
    #test_profile(batchsize, channel,depth, cols,rows,repeat_num=100,d_num=d_num,use_miopen=False,use_half=False)

