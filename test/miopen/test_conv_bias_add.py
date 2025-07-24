import lightop
import torch
from lightop import miopenConvBiasAdd

input = torch.randn(1, 1024, 14, 14).cuda()  # NCHW 格式
weight = torch.randn(256, 1024, 1, 1).cuda()
bias = torch.randn(1, 256, 1, 1).cuda()
add_tensor = torch.randn(1, 256, 16, 16).cuda()
output = lightop.conv_bias_add(
    input=input, weight=weight, bias=bias, add=add_tensor, padding=[1, 1], stride=[1, 1], dilation=[1, 1]
)
torch.cuda.synchronize()
print(output)


print("---------------conv_bias_add function call-------------\n")

input = torch.randn(1, 1024, 14, 14).cuda()
extra = torch.randn(1, 256, 16, 16).cuda()
convbiasadd = miopenConvBiasAdd(1024, 256, 1, padding=1).cuda()
result = convbiasadd(input, extra)
torch.cuda.synchronize()
print(result)


print("---------------conv_bias_add class call---------------\n")
