

import torch
from lightop import op
from triton.testing import do_bench 
from lightop import fuse_fill_rms_x2_concat


# inputs_embeds[positions == 0] = 0
# inputs_embeds = self.enorm(inputs_embeds)
# previous_hidden_states = self.hnorm(previous_hidden_states)

# hidden_states = self.eh_proj(
# torch.cat([inputs_embeds, previous_hidden_states], dim=-1))

def no_fuse_opt(positions, inputs_embeds, previous_hidden_states, weight_inputs_embeds, weight_previous_hidden_states, epsilon):
    inputs_embeds[positions == 0] = 0
    # pdb.set_trace()
    op.rms_norm_opt(inputs_embeds, inputs_embeds, weight_inputs_embeds, epsilon)
    op.rms_norm_opt(previous_hidden_states, previous_hidden_states, weight_previous_hidden_states, epsilon)
    hidden_states = torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
    return hidden_states

def fuse_opt(hidden_states_fuse, positions, inputs_embeds, previous_hidden_states, weight_inputs_embeds, weight_previous_hidden_states, epsilon):
    fuse_fill_rms_x2_concat(hidden_states_fuse, positions, inputs_embeds, previous_hidden_states, weight_inputs_embeds, weight_previous_hidden_states, epsilon)

def test_fuse_fill_rms_x2_concat(batchsize, hidden_size,test_speed):
    device = "cuda"
    dtype = torch.bfloat16
    positions = torch.randint(0, batchsize, (batchsize,), dtype=torch.int64, device=device)
    inputs_embeds = torch.rand((batchsize, hidden_size), dtype=dtype, device=device)
    inputs_embeds_fuse = inputs_embeds.clone()
    previous_hidden_states = torch.rand((batchsize, hidden_size), dtype=dtype, device=device)
    previous_hidden_states_fuse = previous_hidden_states.clone()
    weight_inputs_embeds = torch.rand(hidden_size, dtype=dtype, device=device)
    weight_previous_hidden_states = torch.rand(hidden_size, dtype=dtype, device=device)
    hidden_states_fuse = torch.empty((batchsize, hidden_size * 2), dtype=dtype, device=device)
    epsilon = 1e-6

        # pdb.set_trace()
    

    if test_speed:
        warmup_runs = 25
        test_runs = 100
        avg_time_no_fuse_opt = do_bench(lambda: no_fuse_opt(positions, inputs_embeds, previous_hidden_states, weight_inputs_embeds, weight_previous_hidden_states, epsilon), 
                                 warmup=warmup_runs, rep=test_runs) 
        avg_time_fuse_opt = do_bench(lambda: fuse_opt(hidden_states_fuse, positions, inputs_embeds_fuse, previous_hidden_states_fuse, weight_inputs_embeds, weight_previous_hidden_states, epsilon), 
                                 warmup=warmup_runs, rep=test_runs) 
        print(f"bs: {batchsize}, hidden_size: {hidden_size}, opt non-fused time: {avg_time_no_fuse_opt * 1000:.2f} us, opt fused time: {avg_time_fuse_opt * 1000:.2f} us")
    else:
        hidden_states = no_fuse_opt(positions, inputs_embeds, previous_hidden_states, weight_inputs_embeds, weight_previous_hidden_states, epsilon)
        fuse_opt(hidden_states_fuse, positions, inputs_embeds_fuse, previous_hidden_states_fuse, weight_inputs_embeds, weight_previous_hidden_states, epsilon)
        inputs_embeds_fuse_out = hidden_states_fuse[:, :hidden_size]
        previous_hidden_states_fuse_out = hidden_states_fuse[:, hidden_size:]
        diff_inputs_embeds_max = torch.max(torch.abs(inputs_embeds_fuse_out - inputs_embeds))
        diff_previous_hidden_states_max = torch.max(torch.abs(previous_hidden_states_fuse_out - previous_hidden_states))
        print(f"batchsize: {batchsize}, hidden_size: {hidden_size}, diff_inputs_embeds_max: {diff_inputs_embeds_max}, diff_previous_hidden_states_max: {diff_previous_hidden_states_max}")
        torch.allclose(inputs_embeds_fuse_out, inputs_embeds, atol=1e-4)
        torch.allclose(previous_hidden_states_fuse_out, previous_hidden_states, atol=1e-4)
        # pdb.set_trace()



if __name__ == "__main__":
    # test_fuse_fill_rms_x2_concat(batchsize=1024, hidden_size=7168, test_speed=True)
    # for batchsize in [1, 4, 8,32,64, 512, 1024, 2048, 4096, 8192]:
        # test_fuse_fill_rms_x2_concat(batchsize=batchsize, hidden_size=1024, test_speed=True)
    for batchsize in [1, 4, 8,32,64, 512, 1024, 2048, 4096, 8192]:
        test_fuse_fill_rms_x2_concat(batchsize=batchsize, hidden_size=7168, test_speed=False)