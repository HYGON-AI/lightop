# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

#from lightop import op
from . import op

def blockwise_scaled_cast_transpose(input,output,scale_inv,output_t,scale_inv_t,noop,
                                    block_len,block_scaling_dim,rowwise,columnwise,
                                    epsilon,pow2_scale,all_gather_usage,out_dtype):
    
    return op.blockwise_scaled_cast_transpose(input,output,scale_inv,output_t,scale_inv_t,noop,
                                              block_len,block_scaling_dim,rowwise,columnwise,
                                              epsilon,pow2_scale,all_gather_usage,out_dtype)
