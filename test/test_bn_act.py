import torch
import numpy
import os
from lightop import FuseBatchNormAct2d,fuseactmode

#use_half：rtol=1e-02, atol=1e-02,use_float32:rtol=1e-05, atol=1e-05
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def test_2d_relu():
  torch.backends.cudnn.enabled=False
  x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  bn = torch.nn.BatchNorm2d(x.size(1)).cuda()
  out = bn(x)
  act = torch.nn.ReLU().cuda()
  out = act(out)
  out = torch.add(out, 1)
  out.backward(out)

  os.environ['HIP_NATIVE_FUSE']="1" #强制走native
  fuse_x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  fuse_bn = FuseBatchNormAct2d(fuse_x.size(1), actmode=fuseactmode.relu.value).cuda()
  fuse_out=fuse_bn(fuse_x)
  fuse_out = torch.add(fuse_out, 1)
  fuse_out.backward(fuse_out)

  if (allclose(out,fuse_out,rtol=1e-05, atol=1e-05) and allclose(x.grad,fuse_x.grad,rtol=1e-05, atol=1e-05) 
      and allclose(bn.weight.grad,fuse_bn.weight.grad,rtol=1e-05, atol=1e-05) and allclose(bn.bias.grad,fuse_bn.bias.grad,rtol=1e-05, atol=1e-05)):
    print("test_2d_relu:TRUE")
  else:
    print("test_2d_relu:", allclose(out,fuse_out,rtol=1e-05, atol=1e-05) and allclose(x.grad,fuse_x.grad,rtol=1e-05, atol=1e-05) 
      and allclose(bn.weight.grad,fuse_bn.weight.grad,rtol=1e-05, atol=1e-05) and allclose(bn.bias.grad,fuse_bn.bias.grad,rtol=1e-05, atol=1e-05))

def test_2d_relu_miopen():
  torch.backends.cudnn.enabled=True
  x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  bn = torch.nn.BatchNorm2d(x.size(1)).cuda()
  out = bn(x)
  act = torch.nn.ReLU().cuda()
  out = act(out)
  out = torch.add(out, 1)
  out.backward(out)

  os.environ['HIP_NATIVE_FUSE']="0"
  fuse_x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  fuse_bn = FuseBatchNormAct2d(fuse_x.size(1), actmode=fuseactmode.relu.value).cuda()
  fuse_out=fuse_bn(fuse_x)
  fuse_out = torch.add(fuse_out, 1)
  fuse_out.backward(fuse_out)
  if (allclose(out,fuse_out,rtol=1e-05, atol=1e-05) and allclose(x.grad,fuse_x.grad,rtol=1e-05, atol=1e-05) 
      and allclose(bn.weight.grad,fuse_bn.weight.grad,rtol=1e-05, atol=1e-05) and allclose(bn.bias.grad,fuse_bn.bias.grad,rtol=1e-05, atol=1e-05)):
    print("test_2d_relu_miopen:TRUE")
  else:
    print("test_2d_relu_miopen:", allclose(out,fuse_out,rtol=1e-05, atol=1e-05) and allclose(x.grad,fuse_x.grad,rtol=1e-05, atol=1e-05) 
      and allclose(bn.weight.grad,fuse_bn.weight.grad,rtol=1e-05, atol=1e-05) and allclose(bn.bias.grad,fuse_bn.bias.grad,rtol=1e-05, atol=1e-05))


def test_2d_silu():
  torch.backends.cudnn.enabled=False
  x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  bn = torch.nn.BatchNorm2d(x.size(1)).cuda()
  out = bn(x)
  act = torch.nn.SiLU().cuda()
  out = act(out)
  out = torch.add(out, 1)
  out.backward(out)

  fuse_x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  fuse_bn = FuseBatchNormAct2d(fuse_x.size(1), actmode=fuseactmode.native_silu.value).cuda()
  fuse_out=fuse_bn(fuse_x)
  fuse_out = torch.add(fuse_out, 1)
  fuse_out.backward(fuse_out)

  if (allclose(out,fuse_out,rtol=1e-05, atol=1e-05) and allclose(x.grad,fuse_x.grad,rtol=1e-05, atol=1e-05) 
      and allclose(bn.weight.grad,fuse_bn.weight.grad,rtol=1e-05, atol=1e-05) and allclose(bn.bias.grad,fuse_bn.bias.grad,rtol=1e-05, atol=1e-05)):
    print("test_2d_silu:TRUE")
  else:
    print("test_2d_silu:", allclose(out,fuse_out,rtol=1e-05, atol=1e-05) and allclose(x.grad,fuse_x.grad,rtol=1e-05, atol=1e-05) 
      and allclose(bn.weight.grad,fuse_bn.weight.grad,rtol=1e-05, atol=1e-05) and allclose(bn.bias.grad,fuse_bn.bias.grad,rtol=1e-05, atol=1e-05))

def test_2d_sigmoid_miopen():
  torch.backends.cudnn.enabled=True
  x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  bn = torch.nn.BatchNorm2d(x.size(1)).cuda()
  out = bn(x)
  act = torch.nn.Sigmoid()
  out = act(out)
  out = torch.add(out, 1)
  out.backward(out)

  os.environ['HIP_NATIVE_FUSE']="0"
  fuse_x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  fuse_bn = FuseBatchNormAct2d(fuse_x.size(1), actmode=fuseactmode.miopen_sigmoid.value).cuda()
  fuse_out=fuse_bn(fuse_x)
  fuse_out = torch.add(fuse_out, 1)
  fuse_out.backward(fuse_out)

  if (allclose(out,fuse_out,rtol=1e-05, atol=1e-05) and allclose(x.grad,fuse_x.grad,rtol=1e-05, atol=1e-05) 
      and allclose(bn.weight.grad,fuse_bn.weight.grad,rtol=1e-05, atol=1e-05) and allclose(bn.bias.grad,fuse_bn.bias.grad,rtol=1e-05, atol=1e-05)):
    print("test_2d_sigmoid_miopen:TRUE")
  else:
    print("test_2d_sigmoid_miopen:", allclose(out,fuse_out,rtol=1e-05, atol=1e-05) and allclose(x.grad,fuse_x.grad,rtol=1e-05, atol=1e-05) 
      and allclose(bn.weight.grad,fuse_bn.weight.grad,rtol=1e-05, atol=1e-05) and allclose(bn.bias.grad,fuse_bn.bias.grad,rtol=1e-05, atol=1e-05))

def test_2d_abs_miopen():
  torch.backends.cudnn.enabled=True
  x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  bn = torch.nn.BatchNorm2d(x.size(1)).cuda()
  out = bn(x)
  out = torch.abs(out)
  out = torch.add(out, 1)
  out.backward(out)

  os.environ['HIP_NATIVE_FUSE']="0"
  fuse_x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  fuse_bn = FuseBatchNormAct2d(fuse_x.size(1), actmode=fuseactmode.miopen_abs.value).cuda()
  fuse_out=fuse_bn(fuse_x)
  fuse_out = torch.add(fuse_out, 1)
  fuse_out.backward(fuse_out)

  if (allclose(out,fuse_out,rtol=1e-05, atol=1e-05) and allclose(x.grad,fuse_x.grad,rtol=1e-05, atol=1e-05) 
      and allclose(bn.weight.grad,fuse_bn.weight.grad,rtol=1e-05, atol=1e-05) and allclose(bn.bias.grad,fuse_bn.bias.grad,rtol=1e-05, atol=1e-05)):
    print("test_2d_abs_miopen:TRUE")
  else:
    print("test_2d_abs_miopen:", allclose(out,fuse_out,rtol=1e-05, atol=1e-05) and allclose(x.grad,fuse_x.grad,rtol=1e-05, atol=1e-05) 
      and allclose(bn.weight.grad,fuse_bn.weight.grad,rtol=1e-05, atol=1e-05) and allclose(bn.bias.grad,fuse_bn.bias.grad,rtol=1e-05, atol=1e-05))


def test_2d_leakyrelu_miopen():
  torch.backends.cudnn.enabled=True
  x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  bn = torch.nn.BatchNorm2d(x.size(1)).cuda()
  out = bn(x)
  act = torch.nn.LeakyReLU(0.1).cuda()
  out = act(out)
  out = torch.add(out, 1)
  out.backward(out)

  os.environ['HIP_NATIVE_FUSE']="0"
  fuse_x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  fuse_bn = FuseBatchNormAct2d(fuse_x.size(1), act_alpha = 0.1, actmode=fuseactmode.miopen_leaky_relu.value).cuda()
  fuse_out=fuse_bn(fuse_x)
  fuse_out = torch.add(fuse_out, 1)
  fuse_out.backward(fuse_out)
  if (allclose(out,fuse_out,rtol=1e-05, atol=1e-05) and allclose(x.grad,fuse_x.grad,rtol=1e-05, atol=1e-05) 
    and allclose(bn.weight.grad,fuse_bn.weight.grad,rtol=1e-05, atol=1e-05) and allclose(bn.bias.grad,fuse_bn.bias.grad,rtol=1e-05, atol=1e-05)):
    print("test_2d_leakyrelu_miopen:TRUE")
  else:
    print("test_2d_leakyrelu_miopen:", allclose(out,fuse_out,rtol=1e-05, atol=1e-05) and allclose(x.grad,fuse_x.grad,rtol=1e-05, atol=1e-05) 
      and allclose(bn.weight.grad,fuse_bn.weight.grad,rtol=1e-05, atol=1e-05) and allclose(bn.bias.grad,fuse_bn.bias.grad,rtol=1e-05, atol=1e-05))

def test_2d_elu_miopen():
  torch.backends.cudnn.enabled=True
  x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  bn = torch.nn.BatchNorm2d(x.size(1)).cuda()
  out = bn(x)
  act = torch.nn.ELU(1).cuda()
  out = act(out)
  out = torch.add(out, 1)
  out.backward(out)

  os.environ['HIP_NATIVE_FUSE']="0"
  fuse_x=torch.arange(384, dtype=torch.float32).reshape((2,3,8,8)).cuda().requires_grad_()
  fuse_bn = FuseBatchNormAct2d(fuse_x.size(1), act_alpha = 1, actmode=fuseactmode.miopen_elu.value).cuda()
  fuse_out=fuse_bn(fuse_x)
  fuse_out = torch.add(fuse_out, 1)
  fuse_out.backward(fuse_out)
  if (allclose(out,fuse_out,rtol=1e-05, atol=1e-05) and allclose(x.grad,fuse_x.grad,rtol=1e-05, atol=1e-05) 
    and allclose(bn.weight.grad,fuse_bn.weight.grad,rtol=1e-05, atol=1e-05) and allclose(bn.bias.grad,fuse_bn.bias.grad,rtol=1e-05, atol=1e-05)):
    print("test_2d_elu_miopen:TRUE")
  else:
    print("test_2d_elu_miopen:", allclose(out,fuse_out,rtol=1e-05, atol=1e-05) and allclose(x.grad,fuse_x.grad,rtol=1e-05, atol=1e-05) 
      and allclose(bn.weight.grad,fuse_bn.weight.grad,rtol=1e-05, atol=1e-05) and allclose(bn.bias.grad,fuse_bn.bias.grad,rtol=1e-05, atol=1e-05))
 
if __name__ == "__main__":
  test_2d_silu()
  test_2d_relu()
  test_2d_relu_miopen()
  test_2d_sigmoid_miopen()
  test_2d_abs_miopen()
  test_2d_leakyrelu_miopen()
  test_2d_elu_miopen()
