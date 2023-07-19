import torch
import os
from lightop.fusebndropout import FuseBatchNormDropout2d
torch.manual_seed(1)
def test_2d_1():
  torch.manual_seed(1)
  torch.backends.cudnn.enabled=False
  x = torch.rand(2,3,8,8).cuda().requires_grad_()
  # x = torch.rand(128,64,100,112).cuda().requires_grad_()
  bn = torch.nn.BatchNorm2d(x.size(1)).cuda()
  out = bn(x)
  torch.manual_seed(1)
  dropout = torch.nn.Dropout(p=0.3).cuda()
  out = dropout(out)
  out += 1
  out.backward(out)

  torch.manual_seed(1)
  fuse_x = torch.rand(2,3,8,8).cuda().requires_grad_()
  # fuse_x = torch.rand(128,64,100,112).cuda().requires_grad_()
  fuse_dropout = FuseBatchNormDropout2d(fuse_x.size(1), p = 0.3).cuda()
  fuse_out = fuse_dropout(fuse_x)
  fuse_out += 1
  fuse_out.backward(fuse_out)

  if (torch.allclose(out,fuse_out, rtol=0, atol=10e-6) and torch.allclose(x.grad,fuse_x.grad, rtol=0, atol=10e-6) and torch.allclose(bn.weight.grad, fuse_dropout.weight.grad, rtol=0, atol=10e-5) and torch.allclose(bn.bias.grad, fuse_dropout.bias.grad, rtol=0, atol=10e-5)):
    print("test_2d_1:TRUE")
  else:
    print("test_2d_1:", torch.allclose(out,fuse_out, rtol=0, atol=10e-6), torch.allclose(x.grad,fuse_x.grad, rtol=0, atol=10e-5),torch.allclose(bn.weight.grad, fuse_dropout.weight.grad, rtol=0, atol=10e-5),torch.allclose(bn.bias.grad, fuse_dropout.bias.grad, rtol=0, atol=10e-5))

def test_2d_2():
  torch.manual_seed(1)
  torch.backends.cudnn.enabled=False
  x = torch.rand(2,3,8,8).cuda().requires_grad_()
  # x = torch.rand(128,64,112,112).cuda().requires_grad_()
  bn = torch.nn.BatchNorm2d(x.size(1)).cuda()
  out = bn(x)
  torch.manual_seed(1)
  dropout = torch.nn.Dropout().cuda()
  out = dropout(out)
  out += 1
  out.backward(out)

  torch.manual_seed(1)
  fuse_x = torch.rand(2,3,8,8).cuda().requires_grad_()
  # fuse_x = torch.rand(128,64,112,112).cuda().requires_grad_()
  fuse_dropout = FuseBatchNormDropout2d(fuse_x.size(1)).cuda()
  fuse_out = fuse_dropout(fuse_x)
  fuse_out += 1
  fuse_out.backward(fuse_out)

  if (out.equal(fuse_out) and torch.allclose(x.grad,fuse_x.grad, rtol=0, atol=10e-6) and torch.allclose(bn.weight.grad, fuse_dropout.weight.grad, rtol=0, atol=10e-4) and torch.allclose(bn.bias.grad, fuse_dropout.bias.grad, rtol=0, atol=10e-6)):
    print("test_2d_2:TRUE")
  else:
    print("test_2d_2:", out.equal(fuse_out), torch.allclose(x.grad,fuse_x.grad, rtol=0, atol=10e-6),torch.allclose(bn.weight.grad, fuse_dropout.weight.grad, rtol=0, atol=10e-6),torch.allclose(bn.bias.grad, fuse_dropout.bias.grad, rtol=0, atol=10e-6))

def test_2d_3():
  torch.manual_seed(1)
  torch.backends.cudnn.enabled=False
  x = torch.rand(2,3,8,8).cuda().requires_grad_()
  x = torch.rand(128,64,112,112).cuda().requires_grad_()
  bn = torch.nn.BatchNorm2d(x.size(1)).cuda()
  out = bn(x)
  torch.manual_seed(1)
  dropout = torch.nn.Dropout(p=1).cuda()
  out = dropout(out)
  out += 1
  out.backward(out)

  torch.manual_seed(1)
  fuse_x = torch.rand(2,3,8,8).cuda().requires_grad_()
  fuse_x = torch.rand(128,64,112,112).cuda().requires_grad_()
  fuse_dropout = FuseBatchNormDropout2d(fuse_x.size(1), p = 1).cuda()
  fuse_out = fuse_dropout(fuse_x)
  fuse_out += 1
  fuse_out.backward(fuse_out)

  if (out.equal(fuse_out) and torch.allclose(x.grad,fuse_x.grad, rtol=0, atol=10e-6) and torch.allclose(bn.weight.grad, fuse_dropout.weight.grad, rtol=0, atol=10e-4) and torch.allclose(bn.bias.grad, fuse_dropout.bias.grad, rtol=0, atol=10e-6)):
    print("test_2d_3:TRUE")
  else:
    print("test_2d_3:", out.equal(fuse_out), torch.allclose(x.grad,fuse_x.grad, rtol=0, atol=10e-6),torch.allclose(bn.weight.grad, fuse_dropout.weight.grad, rtol=0, atol=10e-6),torch.allclose(bn.bias.grad, fuse_dropout.bias.grad, rtol=0, atol=10e-6))

def test_2d_4():
  torch.manual_seed(1)
  torch.backends.cudnn.enabled=False
  x = torch.rand(2,3,8,8).cuda().requires_grad_()
  # x = torch.rand(128,64,112,112).cuda().requires_grad_()
  bn = torch.nn.BatchNorm2d(x.size(1)).cuda()
  out = bn(x)
  torch.manual_seed(1)
  dropout = torch.nn.Dropout(p=0).cuda()
  out = dropout(out)
  out += 1
  out.backward(out)

  torch.manual_seed(1)
  fuse_x = torch.rand(2,3,8,8).cuda().requires_grad_()
  # fuse_x = torch.rand(128,64,112,112).cuda().requires_grad_()
  fuse_dropout = FuseBatchNormDropout2d(fuse_x.size(1), p = 0).cuda()
  fuse_out = fuse_dropout(fuse_x)
  fuse_out += 1
  fuse_out.backward(fuse_out)

  if (out.equal(fuse_out) and torch.allclose(x.grad,fuse_x.grad, rtol=0, atol=10e-6) and torch.allclose(bn.weight.grad, fuse_dropout.weight.grad, rtol=0, atol=10e-5) and torch.allclose(bn.bias.grad, fuse_dropout.bias.grad, rtol=0, atol=10e-5)):
    print("test_2d_4:TRUE")
  else:
    print("test_2d_4:", out.equal(fuse_out), torch.allclose(x.grad,fuse_x.grad, rtol=0, atol=10e-6),torch.allclose(bn.weight.grad, fuse_dropout.weight.grad, rtol=0, atol=10e-5),torch.allclose(bn.bias.grad, fuse_dropout.bias.grad, rtol=0, atol=10e-5))
  

if __name__ == "__main__":
  test_2d_1()
  test_2d_2()
  test_2d_3()
  test_2d_4()