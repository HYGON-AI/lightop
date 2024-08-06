import torch
import time
import os
import torch.nn.functional as F
from lightop import RMSNormTorch, RMSNormAdd


class Torch_RNAdd(torch.nn.Module):

    def __init__(self,cols):
        super(Torch_RNAdd, self).__init__()
        self.cols = cols
        self.RMSNormTorch = RMSNormTorch(cols)
   
    def forward(self,x,residual):
        out=self.RMSNormTorch(x + residual)
        return out

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):

    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def get_data(batchsize=2, seqs=128, cols=128,use_half=False):
    if use_half:
        x = torch.rand((batchsize, seqs, cols)).cuda().half()
        residual = torch.rand((batchsize, seqs, cols)).cuda().half()
        grad = torch.rand((batchsize, seqs, cols)).cuda().half()

    else:
        x = torch.rand((batchsize, seqs, cols)).cuda()
        residual = torch.rand((batchsize, seqs, cols)).cuda()
        grad = torch.rand((batchsize, seqs, cols)).cuda()

    x.requires_grad_(True)
    residual.requires_grad_(True)
    x1=x.clone().detach().requires_grad_()
    x2=x.clone().detach().requires_grad_()
    residual1=residual.clone().detach().requires_grad_()
    residual2=residual.clone().detach().requires_grad_()
    return x1,x2,residual1,residual2,grad

def get_model(cols=1,use_half=False, inplace = False):

    if use_half:
        fuse_rn_add=RMSNormAdd(cols,inplace=inplace).cuda().half()
        torch_rn_add=Torch_RNAdd(cols).cuda().half()

    else:
       
        fuse_rn_add=RMSNormAdd(cols, inplace=inplace).cuda()
        torch_rn_add=Torch_RNAdd(cols).cuda()
       
    return torch_rn_add, fuse_rn_add


def test_profile(batchsize, seqs, cols,repeat_num,use_half=False):
    
    x1,x2,residual1,residual2,grad=get_data(batchsize, seqs, cols,use_half=use_half)
    f1,f2=get_model(cols=cols,use_half=use_half)

    torch.manual_seed(0)
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof1:
        for i in range(repeat_num):
            y1 = f1(x1,residual1)
            y1.backward(grad)
            torch.cuda.synchronize()
    prof1.export_chrome_trace('./profile_torch_rn_add.json')

    torch.manual_seed(0)
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof2:
        for i in range(repeat_num):
            y2 = f2(x2,residual2)
            y2.backward(grad)
            torch.cuda.synchronize()
    prof2.export_chrome_trace('./profile_fuse_rn_add.json')  

def test_speed(batchsize,seqs,cols,repeat_num,use_half=False):
    if use_half:
        print(f"---------------fp16, batchsize={batchsize},seqs={seqs}, cols={cols}---------------------")
    else:
        print(f"---------------fp32, batchsize={batchsize},seqs={seqs}, cols={cols}---------------------")
    
    x1,x2,residual1,residual2,grad=get_data(batchsize, seqs, cols,use_half=use_half)
    f1,f2=get_model(cols=cols,use_half=use_half)
    y1 = f1(x1,residual1)
    y1.backward(grad)
    y2 = f2(x2,residual2)
    y2.backward(grad)
    torch.cuda.synchronize()
    torch.manual_seed(0)
    start_time = time.time()
    # torch版
    for i in range(repeat_num):
        x1.grad = None
        residual1.grad=None
        f1.RMSNormTorch.weight.grad=None
        y1 = f1(x1,residual1)
        y1.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()
    fc1_espl = end_time - start_time
    print('torch_rn_add', 'averange time is', fc1_espl *1000/ repeat_num, 'ms')
    # hip版
    y2 = f2(x2,residual2)
    y2.backward(grad)
    torch.cuda.synchronize()
    torch.manual_seed(0)
    start_time = time.time()
    for i in range(repeat_num):
        x2.grad = None
        residual2.grad=None
        f2.weight.grad=None
        y2 = f2(x2,residual2)
        y2.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()    
    fc2_espl = end_time - start_time
    print('fuse_rn_add', 'averange time is', fc2_espl *1000/ repeat_num, 'ms')

    #误差计算
    if use_half:
        print(allclose(y1,y2,rtol=2e-02, atol=2e-02))  # 训练模式前传误差 
        print(allclose(x1.grad,x2.grad,rtol=2e-02, atol=2e-02)) # 训练模式反传中的x1、x2梯度误差
        print(allclose(residual1.grad,residual2.grad,rtol=2e-02, atol=2e-02)) # 训练模式反传中的residual误差
        print(allclose(f1.RMSNormTorch.weight.grad,f2.weight.grad,rtol=2e-02, atol=2e-02)) # 训练模式反传中的gamma梯度误差
        
    else:
        print(allclose(y1,y2,rtol=1e-05, atol=1e-05))
        print(allclose(x1.grad,x2.grad,rtol=1e-05, atol=1e-05))
        print(allclose(residual1.grad,residual2.grad,rtol=1e-05, atol=1e-05))
        print(allclose(f1.RMSNormTorch.weight.grad,f2.weight.grad,rtol=1e-04, atol=1e-04))

def test_speed_eval(batchsize, seqs, cols, repeat_num, use_half=False, inplace = False):
    if use_half:
        print(f"---------------inplace={inplace},fp16, batchsize={batchsize},seqs={seqs}, cols={cols}---------------------")
    else:
        print(f"---------------fp32, batchsize={batchsize},seqs={seqs}, cols={cols}---------------------")
    if inplace ==True:
        y1, x2, residual1, residual2, grad = get_data(batchsize, seqs, cols, use_half=use_half)
        f1, f2 = get_model(cols=cols, use_half=use_half, inplace = inplace)
        # 将模型设置为评估模式
        f1.eval()
        f2.eval()
        y1 = f1(y1,residual1)
        y2 = f2(x2,residual2)
        # CUDA 模型和数据准备
        torch.cuda.synchronize()
        torch.manual_seed(0)

        # 记录开始时间
        start_time = time.time()

        # 测试 torch 版前向传播速度
        for i in range(repeat_num):
            with torch.no_grad():  # 在评估模式中无需计算梯度
                y1 = f1(y1, residual1)
        torch.cuda.synchronize()
        end_time = time.time()
        fc1_espl = end_time - start_time
        print('torch_rn_add', 'average time is', fc1_espl * 1000 / repeat_num, 'ms')

        # 记录开始时间
        start_time = time.time()

        # 测试 hip 版前向传播速度
        for i in range(repeat_num):
            with torch.no_grad():  # 在评估模式中无需计算梯度
                y2 = f2(x2, residual2)
        
        torch.cuda.synchronize()
        end_time = time.time()
        fc2_espl = end_time - start_time
        print('fuse_rn_add', 'average time is', fc2_espl * 1000 / repeat_num, 'ms')

        # 误差计算 (前向传播输出的误差)
        if use_half:
            print(allclose(y1, y2, rtol=2e-02, atol=2e-02))
        else:
            print(allclose(y1, y2, rtol=1e-05, atol=1e-05))
    else:
        x1, x2, residual1, residual2, grad = get_data(batchsize, seqs, cols, use_half=use_half)
        f1, f2 = get_model(cols=cols, use_half=use_half, inplace = inplace)
        # 将模型设置为评估模式
        f1.eval()
        f2.eval()
        y1 = f1(x1,residual1)
        y2 = f2(x2,residual2)
        # CUDA 模型和数据准备
        torch.cuda.synchronize()
        torch.manual_seed(0)

        # 记录开始时间
        start_time = time.time()

        # 测试 torch 版前向传播速度
        for i in range(repeat_num):
            with torch.no_grad():  # 在评估模式中无需计算梯度
                y1 = f1(x1, residual1)
        torch.cuda.synchronize()
        end_time = time.time()
        fc1_espl = end_time - start_time
        print('torch_rn_add', 'average time is', fc1_espl * 1000 / repeat_num, 'ms')

        # 记录开始时间
        start_time = time.time()

        # 测试 hip 版前向传播速度
        for i in range(repeat_num):
            with torch.no_grad():  # 在评估模式中无需计算梯度
                y2 = f2(x2, residual2)
        
        torch.cuda.synchronize()
        end_time = time.time()
        fc2_espl = end_time - start_time
        print('fuse_rn_add', 'average time is', fc2_espl * 1000 / repeat_num, 'ms')

        # 误差计算 (前向传播输出的误差)
        if use_half:
            print(allclose(y1, y2, rtol=2e-02, atol=2e-02))
        else:
            print(allclose(y1, y2, rtol=1e-05, atol=1e-05))       





if __name__ == "__main__":
    # 设置环境变量
    import os  
    #设置基础参数
    # batchsize,seqs, cols = 1,2048,2048
    # batchsize,seqs, cols = 1,2048,3072
    # batchsize,seqs, cols = 1,2048,8192
    batchsize,seqs, cols = 1,2048,16384
    use_half =  True
    inplace = True
    #耗时测试
    # test_speed_eval(batchsize, seqs,cols,repeat_num=100,use_half=use_half, inplace=inplace)
    test_speed(batchsize, seqs,cols,repeat_num=100,use_half=use_half)
    #性能分析
    # test_profile(batchsize, seqs,cols,repeat_num=10,use_half=False)
