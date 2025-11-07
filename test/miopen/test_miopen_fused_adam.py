import lightop
import torch
import torch.nn.functional as F
import torch.optim._functional as Fi
from torch.profiler import profile, ProfilerActivity

# 定义张量维度
lr = 0.001
beta1 = 0.9
beta2 = 0.999
weight_decay = 0
eps = 1e-8
amsgrad = False
maximize = False

# 生成随机输入数据 (float)
params = torch.randn(64, 32, 128, 3, dtype=torch.float).cuda()
grads = torch.randn(64, 32, 128, 3, dtype=torch.float).cuda()
exp_avgs = torch.zeros(64, 32, 128, 3, dtype=torch.float).cuda()
exp_avg_sqs = torch.zeros(64, 32, 128, 3, dtype=torch.float).cuda()
max_exp_avg_sqs = torch.zeros(64, 32, 128, 3, dtype=torch.float).cuda()


params0 = torch.randn(64, 32, 128, 3, dtype=torch.float).cuda()
grads0 = torch.randn(64, 32, 128, 3, dtype=torch.float).cuda()
exp_avgs0 = torch.zeros(64, 32, 128, 3, dtype=torch.float).cuda()
exp_avg_sqs0 = torch.zeros(64, 32, 128, 3, dtype=torch.float).cuda()
max_exp_avg_sqs0 = torch.zeros(64, 32, 128, 3, dtype=torch.float).cuda()


state_steps = torch.tensor(10, dtype=torch.float)
state_steps0 = torch.tensor(10, dtype=torch.float)

for i in range(2):
    lightop.miopen_adamw(
        [params, params0],
        [grads, grads0],
        [exp_avgs, exp_avgs0],
        [exp_avg_sqs, exp_avg_sqs0],
        [max_exp_avg_sqs, max_exp_avg_sqs0],
        [state_steps, state_steps0],
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
    )


print(
    "==================================================================================================================="
)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    lightop.miopen_adamw(
        [params, params0],
        [grads, grads0],
        [exp_avgs, exp_avgs0],
        [exp_avg_sqs, exp_avg_sqs0],
        [max_exp_avg_sqs, max_exp_avg_sqs0],
        [state_steps, state_steps0],
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
    )
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total"))
torch.cuda.synchronize()
print("state_steps in miopen:===========", state_steps)
print("state_steps0 in miopen:===========", state_steps0)

state_steps1 = torch.tensor(10, dtype=torch.float)
state_steps2 = torch.tensor(10, dtype=torch.float)


for i in range(2):
    Fi.adamw(
        [params, params0],
        [grads, grads0],
        [exp_avgs, exp_avgs0],
        [exp_avg_sqs, exp_avg_sqs0],
        [max_exp_avg_sqs, max_exp_avg_sqs0],
        [state_steps1, state_steps2],
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
    )


print(
    "==================================================================================================================="
)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    Fi.adamw(
        [params, params0],
        [grads, grads0],
        [exp_avgs, exp_avgs0],
        [exp_avg_sqs, exp_avg_sqs0],
        [max_exp_avg_sqs, max_exp_avg_sqs0],
        [state_steps1, state_steps2],
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
    )
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total"))
torch.cuda.synchronize()
print("state_step1 in torch:===========", state_steps1)
print("state_steps2 in torch:===========", state_steps2)
