#  -*-  coding: utf-8   -*-
import torch
import time
import copy
from lightop.fusesoftmax  import FuseSoftmax,FuseLogSoftmax
from torch.autograd import Variable
import numpy as np
device = torch.device('cuda')

def check_torch_fuse_Softmax():
  torch.backends.cudnn.enabled=False
 
  x0 = torch.rand([64, 4000], dtype=torch.float32).cuda()
  soft = torch.nn.Softmax(dim=1).cuda()
  x = Variable(x0, requires_grad=True)
  out = soft(x)
  out.sum().backward()

  x0_fuse = copy.deepcopy(x).cuda()
  x_fuse = Variable(x0_fuse, requires_grad=True)

  fuse_soft = FuseSoftmax(dim=1).cuda()
  fuse_out=fuse_soft(x_fuse) 
  fuse_out.sum().backward()
  
  print("########forward#########")
  if np.allclose(out.detach().cpu().numpy(), fuse_out.detach().cpu().numpy(), rtol=1e-03, atol=1e-04):
    print("Passed")
  else:
    print("Failed")

   
  print("########backward#########")  
  if np.allclose(x.grad.detach().cpu().numpy(), x_fuse.grad.detach().cpu().numpy(), rtol=1e-03, atol=1e-04):
    print("Passed")
  else:
    print("Failed")


def check_torch_fuse_LogSoftmax():
  torch.backends.cudnn.enabled=False
 
  x0 = torch.rand([64, 4000], dtype=torch.float32).cuda()
  soft = torch.nn.LogSoftmax(dim=1).cuda()
  x = Variable(x0, requires_grad=True)
  out = soft(x)
  out.sum().backward()

  x0_fuse = copy.deepcopy(x).cuda()
  x_fuse = Variable(x0_fuse, requires_grad=True)

  fuse_soft = FuseLogSoftmax(dim=1).cuda()
  fuse_out=fuse_soft(x_fuse) 
  fuse_out.sum().backward()
  
  print("########forward#########")
  #|a - b|<=(atol + rtol * |b|)
  if np.allclose(out.detach().cpu().numpy(), fuse_out.detach().cpu().numpy(), rtol=1e-03, atol=1e-04):
    print("Passed")
  else:
    print("Failed")
    
   
  print("########backward#########")  
  if np.allclose(x.grad.detach().cpu().numpy(), x_fuse.grad.detach().cpu().numpy(), rtol=1e-03, atol=1e-04):
    print("Passed")
  else:
    print("Failed")


def check_log_softmax_cpu_result(device):
    x_cuda = torch.randn(64, 4000, device=device, dtype=torch.float, requires_grad=True)
    softmax = torch.nn.LogSoftmax(dim=1).to(device)
    y_cuda = softmax(x_cuda)

    logsoftmax_cpu = torch.nn.LogSoftmax(dim=1).to('cpu')
    x_cpu = x_cuda.detach().clone().cpu().requires_grad_().to('cpu')
    y_cpu = logsoftmax_cpu(x_cpu)

    y_cuda.sum().backward()
    y_cpu.sum().backward()
    diff_y = (((y_cpu - y_cuda.cpu()) ** 2).sum()).sqrt()
    diff_bw = (((x_cpu.grad - x_cuda.grad.cpu()) ** 2).sum()).sqrt()
    print("diff_y is", format(diff_y, '.10f'))
    print("diff_bw is", format(diff_bw, '.10f'))

def check_softmax_cpu_result(device):
    x_cuda = torch.randn(64, 4000, device=device, dtype=torch.float, requires_grad=True)
    softmax = torch.nn.Softmax(dim=1).to(device)
    y_cuda = softmax(x_cuda)
    # fuse_soft = FuseSoftmax(dim=1).cuda()
    # y_cuda = fuse_soft(x_cuda)

    softmax_cpu = torch.nn.Softmax(dim=1).to('cpu')
    x_cpu = x_cuda.detach().clone().cpu().requires_grad_().to('cpu')
    y_cpu = softmax_cpu(x_cpu)

    y_cuda.sum().backward()
    y_cpu.sum().backward()
    diff_y = (((y_cpu - y_cuda.cpu()) ** 2).sum()).sqrt()
    diff_bw = (((x_cpu.grad - x_cuda.grad.cpu()) ** 2).sum()).sqrt()
    print("diff_y is", format(diff_y, '.10f'))
    print("diff_bw is", format(diff_bw, '.10f'))    

#def test_fuse_softmax_perf():
#  x0 = torch.rand([1024, 4000], dtype=torch.float32).cuda()
#  x = Variable(x0, requires_grad=True)
#  fuse_soft = FuseSoftmax(dim=1).cuda()
#  fuse_out=fuse_soft(x)
#  fuse_out.sum().backward()
#
#for i in range(10100):
#    if i == 100:
#        print("max:", i)
#        torch.cuda.synchronize()
#        gpu_start_time2 = time.time()
#    torch.manual_seed(0)
#    test_fuse_softmax_perf()
#torch.cuda.synchronize()
#gpu_end_time2 = time.time()
#print("fuse Total time cost: ", gpu_end_time2- gpu_start_time2)


if __name__ == "__main__":   
  print("softmax_cpu_result: ") 
  check_softmax_cpu_result('cuda')
  
  print("softmax_torch_fuse_result: ") 
  check_torch_fuse_Softmax()

  print("Logsoftmax_cpu_result: ") 
  check_log_softmax_cpu_result('cuda')

  print("Logsoftmax_torch_fuse_result: ") 
  check_torch_fuse_LogSoftmax()
