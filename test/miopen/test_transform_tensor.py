import lightop
import torch
from lightop import miopen_transform_tensor

input_ = torch.randn(16, 64, 8, 8).cuda().half()
input = input_.contiguous(memory_format=torch.channels_last)  # NHWC 格式
alpha = 1
beta  = 0

out = lightop.miopen_transform_tensor(input, alpha, beta)

print("out is :\n", out)
print("Transform---------------\n")



