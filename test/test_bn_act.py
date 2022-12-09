import torch
import numpy
import os
from lightop.fusebnact import FuseBatchNormAct2d
from lightop.fuseactmode import fuseactmode

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

  if (out.equal(fuse_out) and x.grad.equal(fuse_x.grad) and bn.weight.grad.equal(fuse_bn.weight.grad) and bn.bias.grad.equal(fuse_bn.bias.grad)):
    print("test_2d_relu:TRUE")
  else:
    print("test_2d_relu:", out.equal(fuse_out), x.grad.equal(fuse_x.grad), bn.weight.grad.equal(fuse_bn.weight.grad),  bn.bias.grad.equal(fuse_bn.bias.grad))

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

  if (out.equal(fuse_out) and x.grad.equal(fuse_x.grad) and bn.weight.grad.equal(fuse_bn.weight.grad) and bn.bias.grad.equal(fuse_bn.bias.grad)):
    print("test_2d_relu_miopen:TRUE")
  else:
    print("test_2d_relu_miopen:", out.equal(fuse_out), x.grad.equal(fuse_x.grad), bn.weight.grad.equal(fuse_bn.weight.grad),  bn.bias.grad.equal(fuse_bn.bias.grad))


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

  if (out.equal(fuse_out) and x.grad.equal(fuse_x.grad) and bn.weight.grad.equal(fuse_bn.weight.grad) and bn.bias.grad.equal(fuse_bn.bias.grad)):
    print("test_2d_silu:TRUE")
  else:
    print("test_2d_silu:", out.equal(fuse_out), x.grad.equal(fuse_x.grad), bn.weight.grad.equal(fuse_bn.weight.grad),  bn.bias.grad.equal(fuse_bn.bias.grad))

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
  if (out.equal(fuse_out) and torch.allclose(x.grad,fuse_x.grad, rtol=0, atol=10e-6) and torch.allclose(bn.weight.grad, fuse_bn.weight.grad, rtol=0, atol=10e-6) and torch.allclose(bn.bias.grad, fuse_bn.bias.grad, rtol=0, atol=10e-6)):
    print("test_2d_sigmoid_miopen:TRUE")
  else:
    print("test_2d_sigmoid_miopen:", out.equal(fuse_out), torch.allclose(x.grad,fuse_x.grad, rtol=0, atol=10e-6),torch.allclose(bn.weight.grad, fuse_bn.weight.grad, rtol=0, atol=10e-6),torch.allclose(bn.bias.grad, fuse_bn.bias.grad, rtol=0, atol=10e-6))

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

  if (out.equal(fuse_out) and x.grad.equal(fuse_x.grad) and bn.weight.grad.equal(fuse_bn.weight.grad) and bn.bias.grad.equal(fuse_bn.bias.grad)):
    print("test_2d_abs_miopen:TRUE")
  else:
    print("test_2d_abs_miopen:", out.equal(fuse_out), x.grad.equal(fuse_x.grad), bn.weight.grad.equal(fuse_bn.weight.grad),  bn.bias.grad.equal(fuse_bn.bias.grad))

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

  if (out.equal(fuse_out) and x.grad.equal(fuse_x.grad) and bn.weight.grad.equal(fuse_bn.weight.grad) and bn.bias.grad.equal(fuse_bn.bias.grad)):
    print("test_2d_leakyrelu_miopen:TRUE")
  else:
    print("test_2d_leakyrelu_miopen:", out.equal(fuse_out), x.grad.equal(fuse_x.grad), bn.weight.grad.equal(fuse_bn.weight.grad),  bn.bias.grad.equal(fuse_bn.bias.grad))

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
  if (out.equal(fuse_out) and torch.allclose(x.grad, fuse_x.grad, rtol=0, atol=10e-6) and bn.weight.grad.equal(fuse_bn.weight.grad) and bn.bias.grad.equal(fuse_bn.bias.grad)):
    print("test_2d_elu_miopen:TRUE")
  else:
    print("test_2d_elu_miopen:", out.equal(fuse_out), x.grad.equal(fuse_x.grad), bn.weight.grad.equal(fuse_bn.weight.grad),  bn.bias.grad.equal(fuse_bn.bias.grad))

if __name__ == "__main__":
  test_2d_silu()
  test_2d_relu()
  test_2d_relu_miopen()
  test_2d_sigmoid_miopen()
  test_2d_abs_miopen()
  test_2d_leakyrelu_miopen()
  test_2d_elu_miopen()
