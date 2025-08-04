import lightop
import torch

a = torch.randn(2, 640, 640).cuda()
b = torch.randn(640, 4096).cuda()
c = torch.randn(2, 640, 4096).cuda()

out = lightop.miopen_gemm_add(a, b, c)
torch.cuda.synchronize()
print(out)
