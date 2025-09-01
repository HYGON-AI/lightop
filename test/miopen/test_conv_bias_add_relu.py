import lightop
import torch
from lightop import miopenConvBiasAddRelu
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

input = torch.randn(1, 1024, 14, 14).cuda()
weight = torch.randn(256, 1024, 1, 1).cuda()
bias = torch.randn(1, 256, 1, 1).cuda()
add_tensor = torch.randn(1, 256, 16, 16).cuda()
for i in range(10):
    output = lightop.miopen_conv_bias_add_relu(
    input=input, weight=weight, bias=bias, add=add_tensor, padding=[1, 1], stride=[1, 1], dilation=[1, 1]
)
torch.cuda.synchronize()
print("===================================================================================================================")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    output = lightop.miopen_conv_bias_add_relu(
    input=input, weight=weight, bias=bias, add=add_tensor, padding=[1, 1], stride=[1, 1], dilation=[1, 1]
)
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total"))
torch.cuda.synchronize()
print(output)
print("---------------conv_bias_add_relu function call end-------------\n")

input = torch.randn(1, 1024, 14, 14).cuda()
extra = torch.randn(1, 256, 16, 16).cuda()
convbiasadd = miopenConvBiasAddRelu(1024, 256, 1, padding=1).cuda()
result = convbiasadd(input, extra)
torch.cuda.synchronize()
print(result)


print("---------------conv_bias_add_relu class call---------------\n")
