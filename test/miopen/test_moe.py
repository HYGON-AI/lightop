import lightop
import torch

# 定义张量维度
num_tokens = 256
hidden_size = 16
topk = 2
num_shared_experts_per_rank = 1
expert_start_idx = 16
expert_end_idx = 32
num_experts_per_rank = expert_end_idx - expert_start_idx
num_tokens_offset = 16
num_tokens_per_rank = 64

max_sactter_tokens = num_shared_experts_per_rank * num_tokens_per_rank + num_tokens * topk

# 生成随机输入数据 (bfloat16)
hidden = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16).cuda()
selected_experts = torch.randint(0, 2, (num_tokens, topk), dtype=torch.int32).cuda()
smooth = torch.randn(num_shared_experts_per_rank + num_experts_per_rank, hidden_size, dtype=torch.float32).cuda()
moe_weights = torch.ones(num_tokens, topk, dtype=torch.float32, device="cuda")
out_scatter_tokens = torch.full((max_sactter_tokens, hidden_size), 0, dtype=torch.int8, device='cuda')
out_scatter_per_token_scale = torch.full((max_sactter_tokens, ), 0, dtype=torch.float32, device='cuda')
out_scatter_tokens_offset = torch.full((max_sactter_tokens, ), 0, dtype=torch.int32, device='cuda')
out_experts_token_count = torch.full((num_shared_experts_per_rank + num_experts_per_rank, ), 0, dtype=torch.int32, device='cuda')
out_experts_token_start = torch.full((num_shared_experts_per_rank + num_experts_per_rank, ), 0, dtype=torch.int32, device='cuda')

# 调用被测函数
scatter_tokens, scatter_per_token_scale, scatter_token_offset, experts_token_count, experts_token_start = (
    lightop.miopen_moe_scatter_dynamic_quant(
        hidden,
        selected_experts,
        moe_weights,
        smooth,
        out_scatter_tokens,
        out_scatter_per_token_scale,
        out_scatter_tokens_offset,
        out_experts_token_count,
        out_experts_token_start,
        expert_start_idx,
        expert_end_idx,
        num_shared_experts_per_rank,
        num_tokens_offset,
        num_tokens_per_rank,
        num_tokens,
    )
)

torch.cuda.synchronize()
print("miopen_moe_scatter_dynamic_quant scatter_tokens is:\n", scatter_tokens)


print("----miopen_moe_scatter_dynamic_quant end -------------\n")


hidden_size = 64
token_nums = 4
total_expert_num = 2

scatter_tokens = torch.randn(token_nums, hidden_size * 2, dtype=torch.bfloat16).cuda()
smooth = torch.randn(total_expert_num, hidden_size, dtype=torch.float32).cuda()
experts_tokens_count = torch.randint(0, 2, (total_expert_num,), dtype=torch.int32).cuda()
exports_tokens_start = torch.randint(0, 2, (total_expert_num,), dtype=torch.int32).cuda()

y, per_token_scale = lightop.miopen_moe_swiglu_dynamic_quant(
    scatter_tokens,
    smooth,
    experts_tokens_count,
    exports_tokens_start,
    1.0,
)
print("miopen_moe_swiglu_dynamic_quant y is:\n", y)

print("----moe_swiglu_dynamic_quant end-------------\n")


world_size = 4
num_tokens_per_rank = 64
topk = 2
current_rank = 1
num_tokens_offset = num_tokens_per_rank * current_rank
allocated_tokens = num_tokens_per_rank * topk
hidden_size = 64
num_shared_experts_per_rank = 2
num_tokens = world_size * num_tokens_per_rank
real_scatter_tokens = num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens

scatterDesc = torch.randn(real_scatter_tokens, hidden_size, dtype=torch.bfloat16).cuda()
tokenoffsetDesc = torch.randint(0, 2, (real_scatter_tokens,), dtype=torch.int32).cuda()

y = lightop.miopen_moe_gather(scatterDesc, tokenoffsetDesc, real_scatter_tokens, num_tokens, hidden_size)

print("miopen_moe_gather y is:\n", y)

print("---------moe_gather end-------------\n")
