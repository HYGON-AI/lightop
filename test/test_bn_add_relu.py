import torch

from lightop.fusebnaddrelu import FuseBatchNormAddRelu1d
from lightop.fusebnaddrelu import FuseBatchNormAddRelu2d
from lightop.fusebnaddrelu import FuseBatchNormAddRelu3d

def test_1d():
  torch.backends.cudnn.enabled=False
  x=torch.arange(48, dtype=torch.float32).reshape((2,24)).cuda().requires_grad_()
  y=torch.arange(48, dtype=torch.float32).reshape((2,24)).cuda().requires_grad_()
  bn = torch.nn.BatchNorm1d(x.size(1)).cuda()
  out = bn(x)
  out = torch.add(out, y)
  relu = torch.nn.ReLU().cuda()
  out = relu(out)
  out = torch.add(out, 10)
  out.backward(out)
  #print(out)
  #print(x.grad)
  #print(y.grad)
  #print(bn.weight.grad)
  #print(bn.bias.grad)

  fuse_x=torch.arange(48, dtype=torch.float32).reshape((2,24)).cuda().requires_grad_()
  fuse_y=torch.arange(48, dtype=torch.float32).reshape((2,24)).cuda().requires_grad_()
  fuse_bn = FuseBatchNormAddRelu1d(fuse_x.size(1)).cuda()
  fuse_out=fuse_bn(fuse_x, fuse_y)
  fuse_out = torch.add(fuse_out, 10)
  fuse_out.backward(fuse_out)
  #print(fuse_out)
  #print(fuse_x.grad)
  #print(fuse_y.grad)
  #print(fuse_bn.weight.grad)
  #print(fuse_bn.bias.grad)

  if (out.equal(fuse_out) and x.grad.equal(fuse_x.grad) and bn.weight.grad.equal(fuse_bn.weight.grad) and bn.bias.grad.equal(fuse_bn.bias.grad)): # and y.grad.equal(fuse_y.grad):
    print("test_1d:TRUE")
  else:
    print("test_1d:", out.equal(fuse_out), x.grad.equal(fuse_x.grad), bn.weight.grad.equal(fuse_bn.weight.grad),  bn.bias.grad.equal(fuse_bn.bias.grad))#, y.grad.equal(fuse_y.grad))


def test_2d():
  torch.backends.cudnn.enabled=False
  x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  y=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  bn = torch.nn.BatchNorm2d(x.size(1)).cuda()
  out = bn(x)
  out = torch.add(out, y)
  relu = torch.nn.ReLU().cuda()
  out = relu(out)
  out = torch.add(out, 10)
  out.backward(out)
  #print(out)
  #print(x.grad)
  #print(y.grad)
  #print(bn.weight.grad)
  #print(bn.bias.grad)
  
  fuse_x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  fuse_y=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  fuse_bn = FuseBatchNormAddRelu2d(fuse_x.size(1)).cuda()
  fuse_out=fuse_bn(fuse_x, fuse_y)
  fuse_out = torch.add(fuse_out, 10)
  fuse_out.backward(fuse_out) 
  #print(fuse_out)
  #print(fuse_x.grad)
  #print(fuse_y.grad)
  #print(fuse_bn.weight.grad)
  #print(fuse_bn.bias.grad)
  
  if (out.equal(fuse_out) and x.grad.equal(fuse_x.grad) and bn.weight.grad.equal(fuse_bn.weight.grad) and bn.bias.grad.equal(fuse_bn.bias.grad)):# and y.grad.equal(fuse_y.grad):
    print("test_2d:TRUE")
  else:
    print("test_2d:", out.equal(fuse_out), x.grad.equal(fuse_x.grad), bn.weight.grad.equal(fuse_bn.weight.grad),  bn.bias.grad.equal(fuse_bn.bias.grad))#, y.grad.equal(fuse_y.grad))
  
def test_3d():
  torch.backends.cudnn.enabled=False
  x=torch.arange(1152, dtype=torch.float32).reshape((2,3,3,8,8)).cuda().requires_grad_()
  y=torch.arange(1152, dtype=torch.float32).reshape((2,3,3,8,8)).cuda().requires_grad_()
  bn = torch.nn.BatchNorm3d(x.size(1)).cuda()
  out = bn(x)
  out = torch.add(out, y)
  relu = torch.nn.ReLU().cuda()
  out = relu(out)
  out = torch.add(out, 10)
  out.backward(out)
  #print(out)
  #print(x.grad)
  #print(y.grad)
  #print(bn.weight.grad)
  #print(bn.bias.grad)

  fuse_x=torch.arange(1152, dtype=torch.float32).reshape((2,3,3,8,8)).cuda().requires_grad_()
  fuse_y=torch.arange(1152, dtype=torch.float32).reshape((2,3,3,8,8)).cuda().requires_grad_()
  fuse_bn = FuseBatchNormAddRelu3d(fuse_x.size(1)).cuda()
  fuse_out=fuse_bn(fuse_x, fuse_y)
  fuse_out = torch.add(fuse_out, 10)
  fuse_out.backward(fuse_out)
  #print(fuse_out)
  #print(fuse_x.grad)
  #print(fuse_y.grad)
  #print(fuse_bn.weight.grad)
  #print(fuse_bn.bias.grad)

  if (out.equal(fuse_out) and x.grad.equal(fuse_x.grad) and bn.weight.grad.equal(fuse_bn.weight.grad) and bn.bias.grad.equal(fuse_bn.bias.grad)):# and y.grad.equal(fuse_y.grad):
    print("test_3d:TRUE")
  else:
    print("test_3d:", out.equal(fuse_out), x.grad.equal(fuse_x.grad), bn.weight.grad.equal(fuse_bn.weight.grad),  bn.bias.grad.equal(fuse_bn.bias.grad))#, y.grad.equal(fuse_y.grad))

if __name__ == "__main__":
  test_1d()
  test_2d()
  test_3d()
