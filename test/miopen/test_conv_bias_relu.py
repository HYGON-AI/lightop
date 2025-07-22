import lightop
import torch

input_nchw = torch.randn(1, 1024, 14, 14).cuda() # NCHW 格式
#input_nhwc = input_nchw.contiguous(memory_format=torch.channels_last)
weight_nchw = torch.randn(256, 1024, 1, 1).cuda()
#weight_nhwc = weight_nchw.contiguous(memory_format=torch.channels_last)
bias_nchw = torch.randn(1,256,1,1).cuda()
#bias_nhwc = bias_nchw.contiguous(memory_format=torch.channels_last)

output = lightop.conv_bias_relu(
        input=input_nchw,
        weight=weight_nchw,
        bias=bias_nchw,
        padding=[1,1],
        stride=[1,1],
        dilation=[1,1])
torch.cuda.synchronize()
print(output)
