#  -*-  coding: utf-8   -*-
import torch
import time
import copy
from lightop  import FuseSoftmax,FuseLogSoftmax
from torch.autograd import Variable
import numpy as np
device = torch.device('cuda')

def check_torch_fuse_Softmax():
  torch.backends.cudnn.enabled=False
 
  x0 = torch.rand([1024, 1024], dtype=torch.float32).cuda()
  soft = torch.nn.Softmax(dim=1).cuda()
  x = Variable(x0, requires_grad=True)

  start_time = time.time()
  for i in range(1000):
    out = soft(x)
    out.sum().backward()
    torch.cuda.synchronize()
  end_time = time.time()
  fc1_espl = end_time - start_time  
  print('torch', 'averange time is', fc1_espl *1000/ 1000, 'ms')

  x0_fuse = copy.deepcopy(x).cuda()
  x_fuse = Variable(x0_fuse, requires_grad=True)
  fuse_soft = FuseSoftmax(dim=1).cuda()

  start_time = time.time()
  for i in range(1000):
      fuse_out=fuse_soft(x_fuse) 
      fuse_out.sum().backward()
      torch.cuda.synchronize()
      end_time = time.time()    
  fc2_espl = end_time - start_time
  print('fuse', 'averange time is', fc2_espl *1000/ 1000, 'ms')
  
  print("########forward#########")
  if np.allclose(out.detach().cpu().numpy(), fuse_out.detach().cpu().numpy(), rtol=1e-05, atol=1e-05):
    print("Passed")
  else:
    print("Failed")

   
  print("########backward#########")  
  if np.allclose(x.grad.detach().cpu().numpy(), x_fuse.grad.detach().cpu().numpy(), rtol=1e-04, atol=1e-04):
    print("Passed")
  else:
    print("Failed")


def check_torch_fuse_LogSoftmax():
  torch.backends.cudnn.enabled=False
 
  x0 = torch.rand([1024, 1024], dtype=torch.float32).cuda()
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





if __name__ == "__main__":   
  print("softmax_torch_fuse_result: ") 
  check_torch_fuse_Softmax()

  print("Logsoftmax_torch_fuse_result: ") 
  check_torch_fuse_LogSoftmax()
