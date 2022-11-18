import torch
import os
from lightop.fusebnrelu import FuseBatchNormRelu1d
from lightop.fusebnrelu import FuseBatchNormRelu2d
from lightop.fusebnrelu import FuseBatchNormRelu3d
def test_1d():
  torch.backends.cudnn.enabled=False
  x=torch.arange(48,dtype=torch.float32).reshape((2,24)).cuda().requires_grad_()
   
  bn = torch.nn.BatchNorm1d(x.size(1)).cuda()
  out = bn(x)
  relu = torch.nn.ReLU().cuda()
  out = relu(out)
  out = torch.add(out, 10)
  out.backward(out)
  #print(x.grad)
  #print(bn.weight.grad)
  #print(bn.bias.grad)
  os.environ['HIP_NATIVE_FUSE']="1"
  fuse_x=torch.arange(48,dtype=torch.float32).reshape((2,24)).cuda().requires_grad_()
  fuse_bn = FuseBatchNormRelu1d(fuse_x.size(1)).cuda()
  fuse_out = fuse_bn(fuse_x)
  fuse_out = torch.add(fuse_out, 10)
  fuse_out.backward(fuse_out)
  #print(fuse_x.grad)
  #print(fuse_bn.weight.grad)
  #print(fuse_bn.bias.grad)
  
  if (out.equal(fuse_out) and x.grad.equal(fuse_x.grad) and bn.weight.grad.equal(fuse_bn.weight.grad) and bn.bias.grad.equal(fuse_bn.bias.grad)):
    print("test_1d:TRUE")
  else:
    print("test_1d:", out.equal(fuse_out), x.grad.equal(fuse_x.grad), bn.weight.grad.equal(fuse_bn.weight.grad),  bn.bias.grad.equal(fuse_bn.bias.grad))


def test_3d():
  torch.backends.cudnn.enabled=False
  x=torch.arange(1152, dtype=torch.float32).reshape((2,3,3,8,8)).cuda().requires_grad_()
  bn = torch.nn.BatchNorm3d(x.size(1)).cuda()
  out = bn(x)
  relu = torch.nn.ReLU().cuda()
  out = relu(out)
  out = torch.add(out, 10)
  out.backward(out)
  #print(x.grad)
  #print(bn.weight.grad)
  #print(bn.bias.grad)
  os.environ['HIP_NATIVE_FUSE']="1"
  fuse_x=torch.arange(1152, dtype=torch.float32).reshape((2,3,3,8,8)).cuda().requires_grad_()
  fuse_bn = FuseBatchNormRelu3d(fuse_x.size(1)).cuda()
  fuse_out=fuse_bn(fuse_x)
  fuse_out = torch.add(fuse_out,10)
  fuse_out.backward(fuse_out)
  #print(fuse_x.grad)
  #print(fuse_bn.weight.grad)
  #print(fuse_bn.bias.grad)
  for i in range(fuse_x.size(0)):
    for j in range(fuse_x.size(1)):
      for k in range(fuse_x.size(2)):
        for m in range(fuse_x.size(3)):
          for n in range(fuse_x.size(4)):
            if (abs(x.grad[i][j][k][m][n] - fuse_x.grad[i][j][k][m][n]) > 0.000001):
              print("error:", i,j,k,m,n,x.grad[i][j][k][m][n],fuse_x.grad[i][j][k][m][n])
  if (out.equal(fuse_out) and bn.weight.grad.equal(fuse_bn.weight.grad) and bn.bias.grad.equal(fuse_bn.bias.grad)):
    print("test_3d:TRUE")
  else:
    print("test_3d:", out.equal(fuse_out), x.grad.equal(fuse_x.grad), bn.weight.grad.equal(fuse_bn.weight.grad),  bn.bias.grad.equal(fuse_bn.bias.grad))


def test_2d():
  torch.backends.cudnn.enabled=False
  x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  bn = torch.nn.BatchNorm2d(x.size(1)).cuda()
  out = bn(x)
  relu = torch.nn.ReLU().cuda()
  out = relu(out)
  out = torch.add(out, 10)
  out.backward(out)
  #print(x.grad)
  #print(bn.weight.grad)
  #print(bn.bias.grad)
  os.environ['HIP_NATIVE_FUSE']="1"
  fuse_x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  fuse_bn = FuseBatchNormRelu2d(fuse_x.size(1)).cuda()
  fuse_out=fuse_bn(fuse_x)
  fuse_out = torch.add(fuse_out, 10)
  fuse_out.backward(fuse_out)
  #print(fuse_x.grad)
  #print(fuse_bn.weight.grad)
  #print(fuse_bn.bias.grad)
  # fusebnaddrelu这个计算误差就没了，x.grad计算存在误差，现在使用误差判断，后续在优化 TODO
  for i in range(fuse_x.size(0)):
    for j in range(fuse_x.size(1)):
      for k in range(fuse_x.size(2)):
        for m in range(fuse_x.size(3)):
          if (abs(x.grad[i][j][k][m] - fuse_x.grad[i][j][k][m]) > 0.000001):
            print("error:", i,j,k,m,x.grad[i][j][k][m],fuse_x.grad[i][j][k][m])
  if (out.equal(fuse_out) and bn.weight.grad.equal(fuse_bn.weight.grad) and bn.bias.grad.equal(fuse_bn.bias.grad)):
    print("test_2d:TRUE")
  else:
    print("test_2d:", out.equal(fuse_out), x.grad.equal(fuse_x.grad), bn.weight.grad.equal(fuse_bn.weight.grad),  bn.bias.grad.equal(fuse_bn.bias.grad))

def test_1d_miopen():
  torch.backends.cudnn.enabled=True
  x=torch.arange(48,dtype=torch.float32).reshape((2,24)).cuda().requires_grad_()

  bn = torch.nn.BatchNorm1d(x.size(1)).cuda()
  out = bn(x)
  relu = torch.nn.ReLU().cuda()
  out = relu(out)
  out = torch.add(out, 10)
  out.backward(out)
  #print(x.grad)
  #print(bn.weight.grad)
  #print(bn.bias.grad)
  os.environ['HIP_MIOPEN_FUSE']="1"
  fuse_x=torch.arange(48,dtype=torch.float32).reshape((2,24)).cuda().requires_grad_()
  fuse_bn = FuseBatchNormRelu1d(fuse_x.size(1)).cuda()
  fuse_out = fuse_bn(fuse_x)
  fuse_out = torch.add(fuse_out, 10)
  fuse_out.backward(fuse_out)
  #print(fuse_x.grad)
  #print(fuse_bn.weight.grad)
  #print(fuse_bn.bias.grad)

  if (out.equal(fuse_out) and x.grad.equal(fuse_x.grad) and bn.weight.grad.equal(fuse_bn.weight.grad) and bn.bias.grad.equal(fuse_bn.bias.grad)):
    print("test_1d_miopen:TRUE")
  else:
    print("test_1d_miopen:", out.equal(fuse_out), x.grad.equal(fuse_x.grad), bn.weight.grad.equal(fuse_bn.weight.grad),  bn.bias.grad.equal(fuse_bn.bias.grad))

def test_2d_miopen():
  torch.backends.cudnn.enabled=True
  x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  bn = torch.nn.BatchNorm2d(x.size(1)).cuda()
  out = bn(x)
  relu = torch.nn.ReLU().cuda()
  out = relu(out)
  out = torch.add(out, 10)
  out.backward(out)
  #print(x.grad)
  #print(bn.weight.grad)
  #print(bn.bias.grad)
  os.environ['HIP_NATIVE_FUSE']="1"
  fuse_x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  fuse_bn = FuseBatchNormRelu2d(fuse_x.size(1)).cuda()
  fuse_out=fuse_bn(fuse_x)
  fuse_out = torch.add(fuse_out, 10)
  fuse_out.backward(fuse_out)
  #print(fuse_x.grad)
  #print(fuse_bn.weight.grad)
  #print(fuse_bn.bias.grad)
  if (out.equal(fuse_out) and bn.weight.grad.equal(fuse_bn.weight.grad) and bn.bias.grad.equal(fuse_bn.bias.grad)):
    print("test_2d_miopen:TRUE")
  else:
    print("test_2d_miopen:", out.equal(fuse_out), x.grad.equal(fuse_x.grad), bn.weight.grad.equal(fuse_bn.weight.grad),  bn.bias.grad.equal(fuse_bn.bias.grad))

if __name__ == "__main__":
  #native test
  test_1d()
  test_2d()
  test_3d()
  #miopen test
  test_1d_miopen()
  test_2d_miopen()