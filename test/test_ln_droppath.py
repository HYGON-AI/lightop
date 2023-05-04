import torch
import time
import torch.nn.functional as F
from torch.autograd import Variable
from lightop.fuselndroppath import LnDropPath as FuseLnDropPath
from lightop.droppath import DropPath
from lightop.layernorm import LayerNorm
from lightop import op


def drop_path(x, drop_prob, training, scale_by_keep):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class LnDropPath(torch.nn.Module):
    def __init__(self, cols:int, drop_prob: float = 0., scale_by_keep: bool = True, device=None, dtype=None):
        super(LnDropPath, self).__init__()
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
    
class LnDropPath2(torch.nn.Module):
    def __init__(self, cols:int, drop_prob: float = 0., scale_by_keep: bool = True, device=None, dtype=None):
        super(LnDropPath2, self).__init__()
        self.norm = LayerNorm(cols, dtype=dtype).to(device)
        self.weight = self.norm.weight
        self.bias = self.norm.bias
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        res = self.norm(x)
        if (not self.training) or self.drop_prob==1.:
            return res
        return op.droppath_forward_autograd(res, self.drop_prob, self.training, self.scale_by_keep)

def compare(name, origin, fuse):
    delta = torch.abs(origin - fuse)
    print("[{!s:^11}] min delta: {:.25f}, max delta: {:.25f}".format(name, torch.min(delta), torch.max(delta)))

def test_lndrop(cols, seqs, batch, drop_prob, scale_by_keep):
    device = torch.device('cuda')
    torch.manual_seed(0)
    dtype=torch.float32
    x0=torch.rand([batch, seqs, cols], dtype=dtype).cuda()
    grad=torch.rand([batch, seqs, cols],dtype=dtype).cuda()

    fuselndrop = FuseLnDropPath(cols, drop_prob=drop_prob, scale_by_keep=scale_by_keep, device=device)
    lndrop = LnDropPath(cols, drop_prob=drop_prob, scale_by_keep=scale_by_keep, device=device)
    lndrop2 = LnDropPath2(cols, drop_prob=drop_prob, scale_by_keep=scale_by_keep, device=device)
    x1=x0.clone().detach().requires_grad_()
    x2=x0.clone().detach().requires_grad_()
    x3=x0.clone().detach().requires_grad_()

    for i in range(10):
        torch.manual_seed(0)
        fuseout = fuselndrop(x1)
        fuseout.backward(grad)
        torch.cuda.synchronize()
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof1:
        start_time = time.time()
        for i in range(100):
            torch.manual_seed(0)
            fuseout = fuselndrop(x1)
            fuseout.backward(grad)
            torch.cuda.synchronize()
        end_time = time.time()
        print("fuse ln_droppath "+str(end_time-start_time))
    prof1.export_chrome_trace('./profile_fuse_ln_droppath.json')

    for i in range(10):
        torch.manual_seed(0)
        out = lndrop(x2)
        out.backward(grad)
        torch.cuda.synchronize()

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof2:
        start_time = time.time()
        for i in range(100):
            torch.manual_seed(0)
            out = lndrop(x2)
            out.backward(grad)
            torch.cuda.synchronize()
        end_time = time.time()
        print("torch ln_droppath "+str(end_time-start_time))
    prof2.export_chrome_trace('./profile_torch_ln_droppath.json')

    for i in range(10):
        torch.manual_seed(0)
        out2 = lndrop2(x3)
        out2.backward(grad)
        torch.cuda.synchronize()
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                      profile_memory=False) as prof3:
        start_time = time.time()
        for i in range(100):
            torch.manual_seed(0)
            out2 = lndrop2(x3)
            out2.backward(grad)
            torch.cuda.synchronize()
        end_time = time.time()
        print("lightop ln_droppath "+str(end_time-start_time))
    prof3.export_chrome_trace('./profile_lightop_ln_droppath.json')

    res_out = out.equal(fuseout)
    res_grad = x2.grad.equal(x1.grad)
    res_weight = lndrop.weight.grad.equal(fuselndrop.weight.grad)
    # print(lndrop.weight.grad-fuselndrop.weight.grad)
    res_bias = lndrop.bias.grad.equal(fuselndrop.bias.grad)
    res = (res_out, res_grad, res_weight, res_bias)

    tensors = {
        "y" : (out, fuseout),
        "grad" : (x1.grad, x2.grad),
        "weight" : (lndrop.weight, fuselndrop.weight),
        "bias" : (lndrop.bias, fuselndrop.bias),
        "weight_grad" : (lndrop.weight.grad, fuselndrop.weight.grad),
        "bias_grad" : (lndrop.bias.grad, fuselndrop.bias.grad),
    }

    # for k,v in tensors.items():
    #     compare(k, *v)
    
    return res, tensors["y"]

def main(epoch):
    with open("test_lndrop.txt", "w+") as f:
        import sys
        # sys.stdout = f
        for _ in range(epoch):
            import random
            # cols = random.randint(1, 5)*512
            # seqs = random.randint(1, 5)*16
            # batch = random.randint(1, 10)*64
            # drop_prob = random.random()
            cols=192
            seqs=784
            batch=128
            drop_prob = 0.5
            scale_by_keep = bool(random.randint(1,1))
            args = (cols, seqs, batch, drop_prob, scale_by_keep)
            # if batch * seqs < 512:
            res, (out, fuseout) = test_lndrop(cols, seqs, batch, drop_prob, scale_by_keep)
            s2 = "{!s:^5} | cols={:4d}; seqs={:<2d}; batch={:<3d}; drop_prob={:.17f}; scale_by_keep={:<1d}".format(str(res), *args)
            print(s2)

main(1)

# cols=192
# seqs=300
# batch=128
# drop_prob = 0.5
# scale_by_keep=True
# device = torch.device('cuda')
# torch.manual_seed(0)
# dtype=torch.float32
# x0=torch.rand([batch, seqs, cols], dtype=dtype).cuda()
# grad=torch.rand([batch, seqs, cols],dtype=dtype).cuda()

# fuselndrop = FuseLnDropPath(cols, drop_prob=drop_prob, scale_by_keep=scale_by_keep, device=device)
# lndrop = LnDropPath(cols, drop_prob=drop_prob, scale_by_keep=scale_by_keep, device=device)
# x1=x0.clone().detach().requires_grad_()
# x2=x0.clone().detach().requires_grad_()

# torch.manual_seed(0)
# out=fuselndrop(x1)
# out.backward(grad)
# torch.manual_seed(0)
# out2=lndrop(x2)
# out2.backward(grad)

# print(out-out2)

# print(out.equal(out2))
# print(x1.grad.equal(x2.grad))
# print(fuselndrop.weight.grad.equal(lndrop.weight.grad))
# print(fuselndrop.bias.grad.equal(lndrop.bias.grad))
