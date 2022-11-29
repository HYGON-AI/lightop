#  -*-  coding: utf-8   -*-
import torch
import time
from lightop.fusesoftmax  import FuseSoftmax,FuseSoftmax2d
from torch.autograd import Variable

def test_1d():
  torch.backends.cudnn.enabled=False
 
  x0 = torch.rand([64, 5000], dtype=torch.float32).cuda()
  grad = torch.rand([64, 5000], dtype=torch.float32).cuda()
  soft = torch.nn.Softmax(dim=1).cuda()
  fuse_soft = FuseSoftmax(dim=1).cuda()
  for i in range(1000):
    x1=x0.requires_grad_()
    x2=x0.requires_grad_()
    out1=soft(x1)
    out2=fuse_soft(x2)
    out1.backward(grad)
    out2.backward(grad)
    if i == 999:
      print(out1.equal(out2))
      print(x1.grad.equal(x2.grad))
    

  

def test_2d():
  torch.backends.cudnn.enabled=False
  x =torch.rand([2, 2000, 12, 13]).cuda()
  soft = torch.nn.Softmax().cuda()
  out = soft(x)
  
  fuse_soft2 = FuseSoftmax2d().cuda()
  fuse_out2=fuse_soft2(x) 

  if (out.equal(fuse_out2)):    
    print("test_2d:TRUE")
  else:
    # print("test_2:", out.equal(fuse_out), x.grad.equal(fuse_x.grad), soft.weight.grad.equal(fuse_soft.weight.grad),  soft.bias.grad.equal(fuse_soft.bias.grad))
    print("test_2d: ", out.equal(fuse_out2))
    print(out)
    print(fuse_out2)
 



if __name__ == "__main__":    
  test_1d()
  #test_2d()

