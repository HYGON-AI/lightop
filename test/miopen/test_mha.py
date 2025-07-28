import lightop
import torch


Q = torch.randn(1, 4, 128, 128).cuda()
K = torch.randn(1, 4, 128, 128).cuda()
V = torch.randn(1, 4, 128, 128).cuda()
out = lightop.miopen_scaled_dot_product_attention(Q, K, V, 0.5, "BHSD")
torch.cuda.synchronize()
print(out)
