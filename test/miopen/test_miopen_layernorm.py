import lightop
import torch
from lightop import miopenGroupNorm, miopenLayerNorm

input = torch.randn(1, 1024, 14, 14).cuda() # NCHW 格式
weight = torch.randn(1).cuda()
bias = torch.randn(1).cuda()
add_tensor = torch.randn(1, 1024, 14, 14).cuda()
output = lightop.layernorm(
        x=input,
        weight=weight,
        bias=bias,
        epsilon=0.00001,
        normalized_dim=4,
        mode= 0 )
torch.cuda.synchronize()
print(output)


print("layernorm-----------------------\n")

output = lightop.add_layernorm(
        x1=input,
        x2=add_tensor,
        weight=weight,
        bias=bias,
        epsilon=0.00001,
        normalized_dim=4,
        mode= 0)
torch.cuda.synchronize()
print(output)


print("addlayernor-----------------------\n")

output = lightop.skip_layernorm(
        x1=input,
        x2=add_tensor,
        weight=weight,
        bias=bias,
        epsilon=0.00001,
        normalized_dim=4,
        mode= 0)
torch.cuda.synchronize()
print(output)



print("skiplayernorm-----------------------\n")


input = torch.randn(3, 512, 128).cuda().half()
layernorm = miopenLayerNorm([512, 128], 1).cuda().half()
out_y, out_mean, out_rstd = layernorm(input)
print(out_y)
print(out_mean)
print(out_rstd)

print("class LayerNorm-------------------\n")
