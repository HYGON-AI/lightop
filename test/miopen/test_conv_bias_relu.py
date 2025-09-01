import lightop
import torch
from lightop import miopenConvBiasRelu
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

input_nchw = torch.randn(1, 1024, 14, 14).cuda()  # NCHW 格式
# input_nhwc = input_nchw.contiguous(memory_format=torch.channels_last)
weight_nchw = torch.randn(256, 1024, 1, 1).cuda()
# weight_nhwc = weight_nchw.contiguous(memory_format=torch.channels_last)
bias_nchw = torch.randn(1, 256, 1, 1).cuda()
# bias_nhwc = bias_nchw.contiguous(memory_format=torch.channels_last)


for i in range(10):
    output = lightop.miopen_conv_bias_relu(
    input=input_nchw, weight=weight_nchw, bias=bias_nchw, padding=[1, 1], stride=[1, 1], dilation=[1, 1]
)
torch.cuda.synchronize()
print("===================================================================================================================")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    output = lightop.miopen_conv_bias_relu(
    input=input_nchw, weight=weight_nchw, bias=bias_nchw, padding=[1, 1], stride=[1, 1], dilation=[1, 1]
)
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total"))
torch.cuda.synchronize()
print(output)
print("---------------conv_bias_add_relu function call end-------------\n")

convbiasrelu = miopenConvBiasRelu(1024, 256, 1).cuda()
result = convbiasrelu(input_nchw).cuda()
torch.cuda.synchronize()
print(result)

print("---------------conv_bias_add_relu class function call end-------------\n")