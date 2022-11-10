import torch
import time
import torch.nn.functional as F
from lightop.layernorm import LayerNorm

cols=768
seqs=6
batchsize=128
device = torch.device('cuda')
torch.manual_seed(0)
x0=torch.rand([batchsize, seqs, cols]).cuda()
grad=torch.rand([batchsize, seqs, cols]).cuda()

fuseln = LayerNorm(cols).to(device)
ln=torch.nn.LayerNorm(cols).to(device)

with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof2:
    for _ in range(10):
        x2=x0.requires_grad_()
        out = ln(x2)
        out.backward(grad)
        time.sleep(0.01)
prof2.export_chrome_trace('./resnet_profile2.json')

with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof:
    for _ in range(10):
        x1=x0.requires_grad_()
        fuseout = fuseln(x1)
        fuseout.backward(grad)
        time.sleep(0.01)
prof.export_chrome_trace('./resnet_profile.json')


fuseln = LayerNorm(cols).to(device)
ln=torch.nn.LayerNorm(cols).to(device)

x1=x0.requires_grad_()
torch.manual_seed(0)
fuseout = fuseln(x1)
fuseout.backward(grad)

x2=x0.requires_grad_()
torch.manual_seed(0)
out = ln(x2)
out.backward(grad)

print(out.equal(fuseout))
print(x1.grad.equal(x2.grad))
print(ln.weight.grad.equal(fuseln.weight.grad))
print(ln.bias.grad.equal(fuseln.bias.grad))

