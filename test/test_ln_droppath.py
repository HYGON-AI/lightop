
import torch
import time
import torch.nn.functional as F
from torch.autograd import Variable
from lightop import LnDropPath


def drop_path(x, drop_prob, training, scale_by_keep):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class TorchLnDropPath(torch.nn.Module):
    def __init__(self, cols:int, drop_prob: float = 0., scale_by_keep: bool = True, device=None, dtype=None):
        super(TorchLnDropPath, self).__init__()
        self.norm = torch.nn.LayerNorm(cols, dtype=dtype).to(device)
        self.weight = self.norm.weight
        self.bias = self.norm.bias
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        res = self.norm(x)
        if (not self.training) or self.drop_prob==1.:
            return res
        return drop_path(res, self.drop_prob, self.training, self.scale_by_keep)
    
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
        fuse_ln_dropoath=LnDropPath(cols,dropout_p).cuda().half()
        torch_ln_dropoath=TorchLnDropPath(cols,dropout_p).cuda().half()

    else:
       
        fuse_ln_dropoath=LnDropPath(cols,dropout_p).cuda()
        torch_ln_dropoath=TorchLnDropPath(cols,dropout_p).cuda()
       
    return torch_ln_dropoath,fuse_ln_dropoath


def test_profile(batchsize, seqs, cols,dropout_p,repeat_num,use_half=False):
    
    x1,x2,grad=get_data(batchsize, seqs, cols,use_half=use_half)
    f1,f2=get_model(cols=cols,dropout_p=dropout_p,use_half=use_half)

    torch.manual_seed(0)
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,use_kineto=True,
                                     profile_memory=False) as prof1:
        for i in range(repeat_num):
            x1.grad = None
            y1 = f1(x1)
            y1.backward(grad)
            torch.cuda.synchronize()
    prof1.export_chrome_trace('./profile_torch_ln_dropoath.json')

    torch.manual_seed(0)
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,use_kineto=True,
                                     profile_memory=False) as prof2:
        for i in range(repeat_num):
            x2.grad = None
            y2 = f2(x2)
            y2.backward(grad)
            torch.cuda.synchronize()
    prof2.export_chrome_trace('./profile_fuse_ln_dropoath.json')  

def test_speed(batchsize,seqs,cols,dropout_p,repeat_num,use_half=False):
    torch.manual_seed(0)
    x1,x2,grad=get_data(batchsize, seqs, cols,use_half=use_half)
    f1,f2=get_model(cols=cols,dropout_p=dropout_p,use_half=use_half)
    torch.manual_seed(0)
    y1 = f1(x1)
    y1.backward(grad)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(repeat_num):
        x1.grad = None
        f1.weight.grad=None
        f1.bias.grad=None
        y1 = f1(x1)
        y1.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()
    fc1_espl = end_time - start_time
    print('torch_ln_dropoath', 'averange time is', fc1_espl *1000/ repeat_num, 'ms')

    torch.manual_seed(0)
    y2 = f2(x2)
    y2.backward(grad)
    torch.cuda.synchronize()
    
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
    print('fuse_ln_dropoath', 'averange time is', fc2_espl *1000/ repeat_num, 'ms')
    #误差计算
    if use_half:
        print(allclose(y1,y2,rtol=2e-02, atol=2e-02))
        print(allclose(x1.grad,x2.grad,rtol=2e-02, atol=2e-02))
        print(allclose(f1.weight.grad,f2.weight.grad,rtol=2e-02, atol=2e-02))
        print(allclose(f1.bias.grad,f2.bias.grad,rtol=2e-02, atol=2e-02))
        
    else:
        print(allclose(y1,y2,rtol=1e-05, atol=1e-05))
        print(allclose(x1.grad,x2.grad,rtol=1e-05, atol=1e-05))
        print(allclose(f1.weight.grad,f2.weight.grad,rtol=1e-02, atol=1e-02))
        print(allclose(f1.bias.grad,f2.bias.grad,rtol=1e-05, atol=1e-05))
    
if __name__ == "__main__":

    #设置基础参数
    batchsize,seqs, cols = 128, 3136, 96
    #batchsize,seqs, cols = 2, 512, 1024
    dropout_p=0.2
    #耗时测试
    test_speed(batchsize, seqs,cols,dropout_p,repeat_num=100,use_half=False)
    #性能分析
    test_profile(batchsize, seqs,cols,dropout_p,repeat_num=10,use_half=False)
