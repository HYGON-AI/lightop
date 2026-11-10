# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.profiler import ProfilerActivity
from lightop import op 
import time
import pdb

def demo_box_cost_i(i, box_pred, box_target, instance_reg_weights, reg_weights, box_weight):
    return torch.sum(torch.abs(box_pred[i, :, None] - box_target[i][None])
        * instance_reg_weights[i][None]
        * box_pred.new_tensor(reg_weights), 
        dim=-1,) * box_weight 

def lightop_box_cost(box_pred, box_target_list, inst_weights_list, reg_weights_list, box_weight):
    device = box_pred.device 
    reg_weights_tensor = torch.tensor(reg_weights_list, device=device, dtype=torch.float32)
    lightop_cost_list = []
    bs = box_pred.shape[0]
    
    for i in range(bs):
        cost_i = op.box_cost(
            box_pred[i],             
            box_target_list[i],      
            inst_weights_list[i],   
            reg_weights_tensor,   
            box_weight
        )
        lightop_cost_list.append(cost_i)
    return lightop_cost_list

if __name__ == '__main__':
    test_speed = True
    
    box_target_len = 20  
    box_pred_shape0 = 20 
    N_dim = 900         
    D_dim = 10        

    m_dims = [
        40, 12, 25, 111, 45, 84, 19, 14, 52, 13, 
        20, 19, 28, 28, 6, 62, 26, 56, 13, 33
    ]
    
    assert len(m_dims) == box_target_len, "m_dims list length must match batch size (box_target_len)"
    box_pred = torch.randn(box_target_len, N_dim, D_dim, device='cuda')
    
    box_target = [torch.rand(m, D_dim, dtype=torch.float32, device='cuda') for m in m_dims]
    instance_reg_weights = [torch.rand(m, D_dim, dtype=torch.float32, device='cuda') for m in m_dims]
    
    # reg_weights: [D]
    reg_weights = [2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0] 
    box_weight = 0.25
    
    bs = box_pred.shape[0]
    
    print("Running Python reference...")
    cost = []
    for i in range(bs):
        cost.append(demo_box_cost_i(i, box_pred, box_target, instance_reg_weights, reg_weights, box_weight))

    print(f"Python ref's list len: {len(cost)}")
    print(f"Shape of first item (Py): {cost[0].shape}") 


    print("\nRunning Lightop implementation...")
    try:
        lightop_cost_list = lightop_box_cost(box_pred, box_target, instance_reg_weights, reg_weights, box_weight)
        print(f"Lightop's list len: {len(lightop_cost_list)}")
        print(f"Shape of first item (Op): {lightop_cost_list[0].shape}") 
        print("\nValidating results...")
        all_close = True
        max_diff = 0.0
        mean_diff = 0.0
        total_items = 0

        for i in range(bs):
            py_cost_i = cost[i]
            op_cost_i = lightop_cost_list[i]
            # pdb.set_trace()
            if not torch.allclose(py_cost_i, op_cost_i, atol=1e-5):
                all_close = False
                print(f"❌ Mismatch at batch item {i}!")
            # pdb.set_trace()
            diff_i = torch.abs(py_cost_i - op_cost_i)
            max_diff = max(max_diff, diff_i.max().item())
            mean_diff += diff_i.mean().item()
            total_items += 1

        if all_close:
            print("✅ 验证通过！ (op.box_cost)")
        else:
            print("❌ 验证失败！ (op.box_cost)")

        print(f"   最大差异: {max_diff}")
        print(f"   平均差异: {mean_diff / total_items}")

    except (ImportError, RuntimeError) as e:
        print(f"Could not run or validate lightop: {e}")
        print("Skipping lightop execution and validation.")
        test_speed = False

   
    if test_speed:
        num_warmup = 10
        num_runs = 100
        
     
        print("\n--- Performance Test: Lightop ---")
        try:
            for _ in range(num_warmup):
                _ = lightop_box_cost(box_pred, box_target, instance_reg_weights, reg_weights, box_weight)
            torch.cuda.synchronize()
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record() 
            
            for _ in range(num_runs):
                _ = lightop_box_cost(box_pred, box_target, instance_reg_weights, reg_weights, box_weight)

            end_event.record() 
            torch.cuda.synchronize()

            elapsed_time_ms = start_event.elapsed_time(end_event)
            avg_time_ms = elapsed_time_ms / num_runs
            print(f"lightop (op.box_cost in loop) 测试结果:")
            print(f"   耗时 : {avg_time_ms:.4f} 毫秒")

        except (ImportError, RuntimeError) as e:
            print(f"Could not run lightop performance test: {e}")

        # --- Python ref 测试 ---
        print("\n--- Performance Test: Python Ref ---")
        for _ in range(num_warmup):
            cost_warmup = []
            for i in range(bs):
                cost_warmup.append(demo_box_cost_i(i, box_pred, box_target, instance_reg_weights, reg_weights, box_weight))
        torch.cuda.synchronize()
        
        start_event_py = torch.cuda.Event(enable_timing=True)
        end_event_py = torch.cuda.Event(enable_timing=True)
        
        start_event_py.record()
        for _ in range(num_runs):
            cost_run = []
            for i in range(bs):
                cost_run.append(demo_box_cost_i(i, box_pred, box_target, instance_reg_weights, reg_weights, box_weight))
        end_event_py.record()
        torch.cuda.synchronize()

        elapsed_time_ms_py = start_event_py.elapsed_time(end_event_py)
        avg_time_ms_py = elapsed_time_ms_py / num_runs
        print(f"Python (demo_box_cost_i in loop) 测试结果:")
        print(f"   耗时: {avg_time_ms_py:.4f} 毫秒")
        
    print("\nTest end.")