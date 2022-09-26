import torch
import time
import torch.nn.functional as F
from torch.autograd import Variable
from fuse.fuseadddropout import AddDropout

device = torch.device('cuda')
x0=torch.rand([128, 50, 768]).cuda()
y0=torch.rand([128, 50, 768]).cuda()
torch.manual_seed(0)

test = AddDropout().to(device)

with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof:
    for _ in range(10):
        x = Variable(x0, requires_grad=True)
        y = Variable(y0, requires_grad=True)
        torch.manual_seed(0)
        z = test(x, y)
        z.sum().backward()
        time.sleep(0.1)
prof.export_chrome_trace('./resnet_profile.json')

with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                     profile_memory=False) as prof2:
    for _ in range(10):
        x2 = Variable(x0, requires_grad=True)
        y2 = Variable(y0, requires_grad=True)
        torch.manual_seed(0)
        z2 = F.dropout(x2)+y2
        z2.sum().backward()
        time.sleep(0.1)
prof2.export_chrome_trace('./resnet_profile2.json')


print(x.grad)
print(x2.grad)