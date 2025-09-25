import lightop
import torch
from lightop import miopenGroupNorm

input = torch.randn(3, 512, 14, 14).cuda()  # NCHW 格式
weight = torch.randn(1, 512, 1, 1).cuda()
bias = torch.randn(1, 512, 1, 1).cuda()
num_groups = 2
add_tensor = torch.randn(3, 512, 14, 14).cuda()
output = lightop.miopen_groupnorm(x=input, weight=weight, bias=bias, num_groups=2, epsilon=0.00001, mode=0)
torch.cuda.synchronize()
print(output)


print("groupnorm-----------------------\n")

output = lightop.miopen_add_groupnorm(
    x1=input, x2=add_tensor, weight=weight, bias=bias, num_groups=2, epsilon=0.00001, mode=0
)
torch.cuda.synchronize()
print(output)

print("addgroupnorm-----------------------\n")

dy = torch.randn(3, 512, 14, 14).cuda()
mean = torch.randn(6, 1, 1, 1).cuda()
rstd = torch.randn(3*num_groups, 1, 1, 1).cuda()
output = lightop.miopen_groupnorm_backward(input, dy, weight, mean, rstd, num_groups, epsilon=0.00001, mode=1)
torch.cuda.synchronize()
print(output)


output_cu = lightop.groupnorm_backward(input, dy, weight, mean, rstd)
print("output_cu", output_cu)

print("miopen_backward----------------------\n")
