import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from lightop.layernorm import LayerNorm


cols=132
seqs=100
batchsize=128
device = torch.device('cuda')
torch.manual_seed(0)
x0=torch.rand([batchsize, seqs, cols]).cuda()
grad=torch.rand([batchsize, seqs, cols]).cuda()


fuseln = LayerNorm(cols).to(device)
ln=torch.nn.LayerNorm(cols).to(device)

x1=x0.clone().detach().requires_grad_()


for i in range(10):
    out1=ln(x1)
    out1.backward(grad)
    torch.cuda.synchronize()
with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof1:
    start_time = time.time()
    for i in range(1000):
        out1=ln(x1)
        out1.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()
    print("torch time is ", end_time - start_time)
prof1.export_chrome_trace('./profile1.json')

x2=x0.clone().detach().requires_grad_()

for i in range(10):
    out2=fuseln(x2)
    out2.backward(grad)
    torch.cuda.synchronize()

with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof2:
    start_time = time.time()
    for i in range(1000):
        out2=fuseln(x2)
        out2.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()
    print("fuse time is ", end_time - start_time)
prof2.export_chrome_trace('./profile2.json')


print(out1.equal(out2))
print(x1.grad.equal(x2.grad))
print(ln.weight.grad.equal(fuseln.weight.grad))
print(ln.bias.grad.equal(fuseln.bias.grad))

