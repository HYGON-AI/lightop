# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import json

import torch
from lightop.utils import resource_path

    
class W8a8GetCacheJSON:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(W8a8GetCacheJSON, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # from vllm.platforms import current_platform
        json_folder_path = resource_path("configs", "w8a8")
    
        self.triton_json_dir=(os.getenv('TRITON_JSON_DIR', json_folder_path))
        self.triton_json_dict={}
        self.triton_moejson_dict={}
        self.triton_json_list=[]
        self.weight_shapes=[]
        self.moe_weight_shapes=[]
        arch_name = torch.cuda.get_device_properties("cuda").gcnArchName.split(':')[0]
        arch_cu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.cache_json_data = {}
        device_name =arch_name+'_'+str(arch_cu)+'cu'
        self.device_name=device_name
        self.topk=1
        self.quant_method=None

    #析构函数，最后会生成model.json的配置文件
    def gen_model_json(self, E: int | None =0, block_size: list | None = None):
        json_dir = os.getenv('LMSLIM_TUNING_JSON', "None")
        if json_dir != "None" and os.path.exists(json_dir):
            #生成模型配置文件
            # logger.info("model_tuning.json is at LMSLIM_TUNING_JSON:%s", json_dir)
            config = {
                "layers": {
                    "linear": {
                        "shapes": [],
                        "m_range":"None",
                    },
                    "moe": {
                        "shapes": [],
                        "m_range": "None",
                        "topk": self.topk
                    }
                },
                "quantization_config": {
                    "quant_method": self.quant_method,
                    "weight_block_size": "None"
                }
            }
            
            # 处理 MoE shapes
            for shape in self.moe_weight_shapes:
                if len(shape) == 4:  # 假设 MoE shape 是 [N1, N2,K] 格式
                    moe_config = {
                        "E": shape[0],
                        "N1": shape[1],
                        "N2": shape[2],
                        "K": shape[3],      # 默认值
                    }
                    config["layers"]["moe"]["shapes"].append(moe_config)
                    
            for shape in self.weight_shapes:
                config["layers"]["linear"]["shapes"].append(shape)
            
            if block_size is not None:
                config["quantization_config"]["weight_block_size"]=block_size
                                    
            with open(json_dir+"/model.json", 'w') as f:
                json.dump(config, f, indent=4)
        # else:
        #     logger.info("LMSLIM_TUNING_JSON is not set")
                   
    def getspec_config(self,configs_dict,M,N,K):
        if f"{M}_{N}_{K}" in configs_dict:
            return configs_dict[f"{M}_{N}_{K}"]
        else:
            return None  
        
    def get_triton_cache(self,file_path,n,k):
        #在非tuning的时候使用，当文件不存在则直接返回none
        cache_json_file=file_path
        
        if os.path.exists(file_path):
        #try:
            with open(cache_json_file, 'r') as file:
                cachedata = json.load(file)
        else:
            return None 
                    
        #把所有的cache解析成key:config的形式：[M_N_K]:[config]
        configs_dict={}
        for key, value in cachedata.items():
            for sub_key, sub_value in value.items():
                configs_key= f"{sub_key}_{key}"
                configs_dict[configs_key]=sub_value
        return configs_dict
  
    def get_w8a8json_name(self,n,k):
        return self.triton_json_dir+f"/W8A8_{n}_{k}_{self.device_name}.json"
    
    def get_blockint8_triton_cache(self,file_path,n,k,block_n,block_k):
        cache_json_file=file_path
        
        if os.path.exists(file_path):
        #try:
            with open(cache_json_file, 'r') as file:
                cachedata = json.load(file)
        else:
            return None  
                    
        #把所有的cache解析成key:config的形式：[M_N_K]:[config]
        configs_dict={}
        for key, value in cachedata.items():
            for sub_key, sub_value in value.items():
                configs_key= f"{sub_key}_{key}"
                configs_dict[configs_key]=sub_value
        return configs_dict

    def get_blockint8json_name(self,n,k,block_n,block_k):
        return self.triton_json_dir+f"/linear_{n}_{k}_block[{block_n},{block_k}]_{self.device_name}.json"

    def get_moeint8json_name(self, E, N1, N2, K, TOPK,
                             block_size: list | None = None, use_int4_w4a8: bool | None = False,
                             use_int8_w8a8: bool | None = False):
        if use_int4_w4a8:
            if block_size is not None:
                return self.triton_json_dir + f"/MOE_W4A8INT8[{block_size[0]},{block_size[1]}]_E={E}_N1={N1}_N2={N2}_K={K}_TOPK{TOPK}_{self.device_name}.json"
            else:
                return self.triton_json_dir + f"/MOE_W4A8INT8_E={E}_N1={N1}_N2={N2}_K={K}_TOPK{TOPK}_{self.device_name}.json"
        elif use_int8_w8a8:
            if block_size is not None:
                return self.triton_json_dir + f"/MOE_BLOCKINT8[{block_size[0]},{block_size[1]}]_E={E}_N1={N1}_N2={N2}_K={K}_TOPK{TOPK}_{self.device_name}.json"
            else:
                return self.triton_json_dir + f"/MOE_W8A8INT8_E={E}_N1={N1}_N2={N2}_K={K}_TOPK{TOPK}_{self.device_name}.json"
        else:
            if block_size is not None:
                return self.triton_json_dir + f"/MOE_BLOCKFP8[{block_size[0]},{block_size[1]}]_E={E}_N1={N1}_N2={N2}_K={K}_TOPK{TOPK}_{self.device_name}.json"
            else:
                return self.triton_json_dir + f"/MOE_W8A8FP8_E={E}_N1={N1}_N2={N2}_K={K}_TOPK{TOPK}_{self.device_name}.json"


    def get_moeint8_triton_cache(self,file_path,E,N1,N2,K,TOPK):
        if file_path in self.cache_json_data:
            # 直接返回缓存数据，避免重复读取
            return self.cache_json_data[file_path]
        cache_json_file=file_path
        
        if os.path.exists(file_path):
        #try:
            with open(cache_json_file, 'r') as file:
                cachedata = json.load(file)
        else:
            return None  
                    
        #把所有的cache解析成key:config的形式：[M_N_K]:[config1,config2]
        configs_dict={}
        for key, value in cachedata.items():
            for sub_key, sub_value in value.items():
                configs_key= f"{sub_key}_{key}"   
                configs_dict[configs_key]=sub_value
        self.cache_json_data[file_path] = configs_dict
        return configs_dict
   
