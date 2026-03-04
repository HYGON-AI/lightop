from lightop import op

import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_ITER = 10
TEST_ITER = 100
SEED = 33



def pack8_uint32_to_float16(data):
    data_flatten = data.flatten()
    num = data_flatten.shape[0]
    data_fp16 = np.zeros(num * 8, dtype=np.float16)
    for i in range(0, num * 8, 8):
        data_fp16[i:i+8] = np.array([
            data_flatten[i // 8] & 0xF,
            (data_flatten[i // 8] >> 16) & 0xF,
            
            (data_flatten[i // 8] >> 4) & 0xF,
            (data_flatten[i // 8] >> 20) & 0xF,
            
            (data_flatten[i // 8] >> 8) & 0xF,
            (data_flatten[i // 8] >> 24) & 0xF,
            
            (data_flatten[i // 8] >> 12) & 0xF,
            (data_flatten[i // 8] >> 28) & 0xF            
        ], dtype=np.float16)

    return data_fp16

def pack8_float16_to_uint32(data):
    data_flatten = data.flatten()
    num = data_flatten.shape[0]
    data_uint32 = np.zeros(num // 8, dtype=np.uint32)
    for i in range(0, num, 8):
        data_uint32[i // 8] |= (data_flatten[i].astype(np.uint32) << 28)
        data_uint32[i // 8] |= (data_flatten[i+1].astype(np.uint32) << 24)
        data_uint32[i // 8] |= (data_flatten[i+2].astype(np.uint32) << 20)
        data_uint32[i // 8] |= (data_flatten[i+3].astype(np.uint32) << 16)
        data_uint32[i // 8] |= (data_flatten[i+4].astype(np.uint32) << 12)
        data_uint32[i // 8] |= (data_flatten[i+5].astype(np.uint32) << 8)
        data_uint32[i // 8] |= (data_flatten[i+6].astype(np.uint32) << 4)
        data_uint32[i // 8] |= (data_flatten[i+7].astype(np.uint32))
    return data_uint32

def get_kernel_time_ms(start_event, end_event):
    end_event.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms

def warmup():
    print("\n=== 预热 (awq_w4a16_dequant_trans) ===")
    data = np.random.randint(0, 2**32, (128, 4608), dtype=np.uint32)
    scale = torch.ones(4, 4608 * 4, dtype=torch.float16, device="cuda")
    zero = torch.zeros(4, 4608 * 4, dtype=torch.float16, device="cuda")
    for _ in range(WARMUP_ITER):
        _ = op.awq_w4a16_dequant_trans(torch.from_numpy(data).to("cuda"), scale, zero, 128, 4608 * 4, 64)
        torch.cuda.synchronize()




# ===================== 一致性验证 =====================
def valid():
    print("\n=== 一致性验证 (awq_w4a16_dequant_trans) ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)  

    test_shapes = [
        (32, 1000), 
        (32, 4608),
        (64, 8190),
    ]
    for i, shape in enumerate(test_shapes):
        data = np.random.randint(0, 2**32, shape, dtype=np.uint32)

        data_flatten = data.flatten()
        data_fp16 = pack8_uint32_to_float16(data_flatten)

        IC = shape[0] * 2
        OC = shape[1] * 4
        group_size = 64

        weight = torch.from_numpy(data).to("cuda")

        scale = torch.ones(IC // group_size, OC, dtype=torch.float16, device="cuda")
        zero = torch.zeros(IC // group_size, OC, dtype=torch.float16, device="cuda")

        a = op.awq_w4a16_dequant_trans(weight, scale, zero, IC, OC, group_size)
        torch.cuda.synchronize() 
        diff = np.max(a.t().contiguous().flatten().cpu().numpy() - data_fp16)
        print(f"用例 {i+1}/{len(test_shapes)}: max diff (shape={diff:.6f})")


# ===================== 性能测试 =====================
def perf(test_numpy=False):
    print("\n=== 性能测试 (awq_w4a16_dequant_trans) ===")
    test_cases = [
            (32, 512), 
            (128, 1000), 
            (32, 3072),
            (2048, 3072),
        ]

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i, shape in enumerate(test_cases):
            data = np.random.randint(0, 2**32, shape, dtype=np.uint32)

            data_flatten = data.flatten()
            data_fp16 = pack8_uint32_to_float16(data_flatten)

            IC = shape[0] * 2
            OC = shape[1] * 4
            group_size = 64

            weight = torch.from_numpy(data).to("cuda")
            scale = torch.ones(IC // group_size, OC, dtype=torch.float16, device="cuda")
            zero = torch.zeros(IC // group_size, OC, dtype=torch.float16, device="cuda")

            if test_numpy:
                start_event.record()
                for _ in range(TEST_ITER):
                    data_fp16 = pack8_uint32_to_float16(data_flatten)
                end_event.record()
                pt_sat_time_ms = get_kernel_time_ms(start_event, end_event)
                pt_sat_avg_us = (pt_sat_time_ms / TEST_ITER) * 1000
                print(f"numpy shape: {shape}, avg_us: {pt_sat_avg_us:.4f}")


            start_event.record()
            for _ in range(TEST_ITER):
                a = op.awq_w4a16_dequant_trans(weight, scale, zero, IC, OC, group_size) 
            end_event.record()
            pt_sat_time_ms = get_kernel_time_ms(start_event, end_event)
            pt_sat_avg_us = (pt_sat_time_ms / TEST_ITER) * 1000
            print(f"shape: {shape}, avg_us: {pt_sat_avg_us:.4f}")
    
    return  



if __name__ == '__main__':
    warmup()
    valid()
    perf()
    


