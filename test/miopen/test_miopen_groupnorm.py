import lightop
import torch
from lightop import miopenGroupNorm

input = torch.randn(3, 512, 14, 14).cuda() # NCHW 格式
weight = torch.randn(1, 512, 1, 1).cuda()
bias = torch.randn(1, 512, 1, 1).cuda()
add_tensor = torch.randn(3, 512, 14, 14).cuda()
output = lightop.groupnorm(
        x=input,
        weight=weight,
        bias=bias,
        num_groups=2,
        epsilon=0.00001,
        mode= 0)
torch.cuda.synchronize()
print(output)


print("groupnorm-----------------------\n")

output = lightop.add_groupnorm(
        x1=input,
        x2=add_tensor,
        weight=weight,
        bias=bias,
        num_groups=2,
        epsilon=0.00001,
        mode= 0)
torch.cuda.synchronize()
print(output)

print("addgroupnorm-----------------------\n")


input = torch.randn(3, 512, 128, 128).cuda().half()
group_norm = miopenGroupNorm(2, 512, 0).cuda().half()
group_norm(input)

print("class GroupNorm-------------------\n")