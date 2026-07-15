import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from lightop.norm import GroupNorm


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):

    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def get_data(batchsize=128, seqs=50, cols=128,rows=128,use_half=False):

    if use_half:
        #x = torch.rand((batchsize, seqs)).cuda().half()
        #grad = torch.rand((batchsize, seqs)).cuda().half()
        x = torch.rand((batchsize, seqs, cols,rows)).cuda().half()
        grad = torch.rand((batchsize, seqs, cols,rows)).cuda().half()
    else:
        #x = torch.rand((batchsize, seqs)).cuda()
        #grad = torch.rand((batchsize, seqs)).cuda()
        # x = torch.ones((batchsize, seqs, cols,rows)).cuda()
        x = torch.rand((batchsize, seqs, cols,rows)).cuda()
        grad = torch.rand((batchsize, seqs, cols,rows)).cuda()

    x.requires_grad_(True)
    x1=x.clone().detach().requires_grad_()
    x2=x.clone().detach().requires_grad_()
    return x1,x2,grad

def get_model(seqs=1024,num_groups=2,use_half=False):

    if use_half:
        fuse_groupnorm=GroupNorm(num_groups,seqs).cuda().half()
        torch_groupnorm=torch.nn.GroupNorm(num_groups,seqs).cuda().half()
    else:
        fuse_groupnorm=GroupNorm(num_groups,seqs).cuda()
        torch_groupnorm=torch.nn.GroupNorm(num_groups,seqs).cuda()
        
    return torch_groupnorm,fuse_groupnorm


def test_profile(batchsize, seqs, cols,rows,num_groups,repeat_num,use_half=False):
    
    x1,x2,grad=get_data(batchsize, seqs, cols,rows,use_half=use_half)
    f1,f2=get_model(seqs=seqs,num_groups=num_groups,use_half=use_half)

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,use_kineto=True,
                                     profile_memory=False) as prof1:
        for i in range(repeat_num):
            y1 = f1(x1)
            y1.backward(grad)
            torch.cuda.synchronize()
    prof1.export_chrome_trace('./profile_torch_groupnorm.json')
    
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,use_kineto=True,
                                     profile_memory=False) as prof2:
        for i in range(repeat_num):
            y2 = f2(x2)
            y2.backward(grad)
            torch.cuda.synchronize()
    prof2.export_chrome_trace('./profile_fuse_groupnorm.json')  

def test_speed(batchsize, seqs, cols,rows,num_groups,repeat_num,use_half=False,forward_only=False):
    torch.manual_seed(0)
    x1,x2,grad=get_data(batchsize, seqs, cols,rows,use_half=use_half)
    f1,f2=get_model(seqs=seqs,num_groups=num_groups,use_half=use_half)
    if forward_only:
        f1=f1.eval()
        f2=f2.eval()
    torch.manual_seed(0)
    weight=torch.rand(seqs).cuda()
    f1.weight.data=weight.clone().detach()
    f2.weight.data=weight.clone().detach()
    print("size=(",batchsize,",",seqs,",",cols,",",rows,")")
    for i in range(10):
        y1 = f1(x1)
        if not forward_only:
            y1.backward(grad)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(repeat_num):
        x1.grad = None
        f1.weight.grad=None
        f1.bias.grad=None
        y1 = f1(x1)
        if not forward_only:
            y1.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()
    fc1_espl = end_time - start_time

    for i in range(10):
        y2 = f2(x2)
        if not forward_only:
            y2.backward(grad)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(repeat_num):
        x2.grad = None
        f2.weight.grad=None
        f2.bias.grad=None
        y2 = f2(x2)
        if not forward_only:
            y2.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()    
    fc2_espl = end_time - start_time
    print('torch time:', format(fc1_espl *1000/ repeat_num,'.3f'), 'ms. lightop time:', format(fc2_espl *1000/ repeat_num,'.3f'), 'ms speedup:', format(fc1_espl/fc2_espl-1,'.2f'))
    # print(y1)
    # print(y2)
    # #误差计算
    if use_half:
        print(allclose(y1,y2,rtol=2e-02, atol=2e-02))
        print(allclose(x1.grad,x2.grad,rtol=2e-02, atol=2e-02))
        print(allclose(f1.weight.grad,f2.weight.grad,rtol=2e-02, atol=2e-02))
        print(allclose(f1.bias.grad,f2.bias.grad,rtol=2e-02, atol=2e-02))
    else:
        rtol=1e-3
        atol=1e-3
        print(allclose(y1,y2,rtol=rtol, atol=atol))
        # print(y1.view(-1))
        # print(y2.view(-1))
        if not forward_only:
            print(allclose(x1.grad,x2.grad,rtol=rtol, atol=atol))
            # print(x1.grad.view(-1))
            # print(x2.grad.view(-1))
            print(allclose(f1.weight.grad,f2.weight.grad,rtol=2e-2, atol=2e-2))
            print(allclose(f1.bias.grad,f2.bias.grad,rtol=rtol, atol=atol))
            # print(f1.bias.grad)
            # print(f2.bias.grad)

def group_norm(x, gamma, beta, G, eps=1e-5):
    # x: 输入数据, [N, C, H, W]
    # gamma, beta: 缩放和偏移参数, [C]
    # G: 组的数量
    # eps: 分母初始值防止除以零
    
    N, C, H, W = x.shape
    x = x.reshape(N, G, C // G, H, W)
    mean = x.mean(dim=(2, 3, 4), keepdim=True)
    var = x.var(dim=(2, 3, 4), keepdim=True)
    x = (x - mean) / torch.sqrt(var + eps)
    x = x.reshape(N, C, H, W)
    return x * gamma.reshape(1, C, 1, 1) + beta.reshape(1, C, 1, 1)

if __name__ == "__main__":
    #设置基础参数
    forward_only=True
    use_half=False
    num_groups=32
    batchsize=[1,2]
    seq=[160,640,2560]
    cols=[8,16,32,64]
    for b in batchsize:
        for s in seq:
            for c in cols:
                if b*s*c*c <=2560*64*64:
                    test_speed(b, s, c,c,num_groups,repeat_num=100,use_half=use_half,forward_only=forward_only)

    batchsize, seqs, cols,rows = 1,512,128,128
    test_speed(batchsize, seqs, cols,rows,num_groups,repeat_num=100,use_half=use_half,forward_only=forward_only)
    batchsize, seqs, cols,rows = 1,512,256,256
    test_speed(batchsize, seqs, cols,rows,num_groups,repeat_num=100,use_half=use_half,forward_only=forward_only)
    batchsize, seqs, cols,rows = 1,256,512,512
    test_speed(batchsize, seqs, cols,rows,num_groups,repeat_num=100,use_half=use_half,forward_only=forward_only)
    batchsize, seqs, cols,rows = 128,1024,1,1
    test_speed(batchsize, seqs, cols,rows,num_groups,repeat_num=100,use_half=use_half,forward_only=forward_only)

    
    #耗时测试
    # test_speed(batchsize, seqs, cols,rows,num_groups,repeat_num=100,use_half=False)
    #性能分析
    # test_profile(batchsize, seqs, cols,rows,num_groups,repeat_num=10,use_half=False)
