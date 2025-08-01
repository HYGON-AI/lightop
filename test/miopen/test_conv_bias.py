import lightop
import torch
from lightop import miopenConvBias

input = torch.randn(3, 512, 14, 14).cuda()  # NCHW 格式
input_nhwc = input.contiguous(memory_format=torch.channels_last)
weight = torch.randn(256, 512, 1, 1).cuda()
weight_nhwc = weight.contiguous(memory_format=torch.channels_last)
bias = torch.randn(1, 256, 1, 1).cuda()
bias_nhwc = bias.contiguous(memory_format=torch.channels_last)

output = lightop.miopen_conv_bias(
    x=input_nhwc, weight=weight_nhwc, bias=bias_nhwc, padding=[1, 1], stride=[1, 1], dilation=[1, 1]
)
torch.cuda.synchronize()
print(output)


print("---------------conv_bias function call-------------\n")

convbias = miopenConvBias(512, 256, 1).cuda()
result = convbias(input_nhwc).cuda()
torch.cuda.synchronize()
print(result)


print("---------------conv_bias class call---------------\n")
