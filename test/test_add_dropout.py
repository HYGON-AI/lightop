import torch
import torch.nn.functional as F
from torch.autograd import Variable
from fuse.fuseadddropout import AddDropout
from fuse.fuseadddropout import TestAdd

x0=torch.rand([2, 10])
y0=torch.rand([2, 10])
torch.manual_seed(0)

x = Variable(x0.cuda(), requires_grad=True)
y = Variable(y0.cuda(), requires_grad=True)
test = AddDropout()
z = test(x, y)
z.sum().backward()
print('x: ', x)
print('y: ', y)
print('z: ', z)
print('x.grad: ', x.grad)
print('y.grad: ', y.grad)
print("*****************************************************")
torch.manual_seed(0)
x2 = Variable(x0.cuda(), requires_grad=True)
y2 = Variable(y0.cuda(), requires_grad=True)
add = TestAdd()
z2 = add(x2, y2)
z2 = F.dropout(z2)
z2.sum().backward()
print('x2: ', x2)
print('y2: ', y2)
print('z2: ', z2)
print('x2.grad: ', x2.grad)
print('y2.grad: ', y2.grad)
