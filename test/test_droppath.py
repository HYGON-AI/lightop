import torch
import time
import torch.nn.functional as F
from torch.autograd import Variable
from lightop.droppath import DropPath


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath2(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath2, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


x=torch.rand([128, 3136, 96]).cuda()
grad=torch.rand([128, 3136, 96]).cuda()

x1=x.clone().detach().requires_grad_()
x2=x.clone().detach().requires_grad_()


test1 = DropPath(0.1)
torch.manual_seed(0)
for i in range(10):
    out1=test1(x1)
    out1.backward(grad)

torch.cuda.synchronize()
start_time = time.time()
with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof2:
    for i in range(1000):
        out1=test1(x1)
        out1.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()
    print("fused time is ", end_time - start_time)
prof2.export_chrome_trace('./profile2.json')

test2 = DropPath2(0.1)
torch.manual_seed(0)
for i in range(10):
    out2=test2(x2)
    out2.backward(grad)

torch.cuda.synchronize()
start_time = time.time()
with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof1:
    for i in range(1000):
        out2=test2(x2)
        out2.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()
    print("origin time is ", end_time - start_time)
prof1.export_chrome_trace('./profile1.json')

print(out1.equal(out2))
print(x1.grad.equal(x2.grad))
