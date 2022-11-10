import torch
import time
import torch.nn.functional as F
from lightop.fuselnadddropout import LnAddDropout

cols=768
seqs=6
batchsize=128
device = torch.device('cuda')
torch.manual_seed(0)
x0=torch.rand([batchsize, seqs, cols]).cuda()
y0=torch.rand([batchsize, seqs, cols]).cuda()
grad=torch.rand([batchsize, seqs, cols]).cuda()

fuselndropout = LnAddDropout(cols).to(device)
ln=torch.nn.LayerNorm(cols).to(device)

with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof2:
    for _ in range(10):
        x2=x0.requires_grad_()
        y2=y0.requires_grad_()
        out = ln(F.dropout(x2)+y2)
        out.backward(grad)
        time.sleep(0.01)
prof2.export_chrome_trace('./resnet_profile2.json')

with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof:
    for _ in range(10):
        x1=x0.requires_grad_()
        y1=y0.requires_grad_()
        fuseout = fuselndropout(x1,y1)
        fuseout.backward(grad)
        time.sleep(0.01)
prof.export_chrome_trace('./resnet_profile.json')


fuselndropout = LnAddDropout(cols).to(device)
ln=torch.nn.LayerNorm(cols).to(device)

x1=x0.requires_grad_()
y1=y0.requires_grad_()
torch.manual_seed(0)
fuseout = fuselndropout(x1,y1)
fuseout.backward(grad)

x2=x0.requires_grad_()
y2=y0.requires_grad_()
torch.manual_seed(0)
out = ln(F.dropout(x2)+y2)
out.backward(grad)

#print(fuseout)
#print(out)
print(out.equal(fuseout))
print(x1.grad.equal(x2.grad) and y1.grad.equal(y2.grad))
print(ln.weight.grad.equal(fuselndropout.weight.grad))
print(ln.bias.grad.equal(fuselndropout.bias.grad))

