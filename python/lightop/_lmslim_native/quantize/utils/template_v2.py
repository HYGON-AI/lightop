import torch
import math
import json
import os
device_sinfo = torch.cuda.get_device_properties(torch.cuda.current_device())
def get_config_v2(M:int,N:int,K:int):
    if "BW" in device_sinfo.name:
        json_file_name = "bw_gemm_templates_prof_best_map_v2.json"
    elif device_sinfo.multi_processor_count == 120:
        json_file_name = "k100ai_120_gemm_templates_prof_best_map_v2.json"
    else:
        return None

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "awq_configs", json_file_name)

    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            config_name_v2 = json.load(f)
    else:
        return None

    if M>16 and M<32:
        M_normal="32"
    else:
        M_normal = str(1 << round(math.log2(M)))

    if f"N{N}K{K}" in config_name_v2:
        return config_name_v2[f"N{N}K{K}"].get(M_normal)
    else:
        return None


#test
# config = get_config_v2(17,5120,6400)
# print(type(config))  # 通常是 dict
# print(config)