from lightop import op

def blockwise_scaled_1d_cast_transpose(input,output,scale_inv,output_t,scale_inv_t,noop,
                                       epsilon,pow2_scale,all_gather_usage,out_dtype):
    
    return op.blockwise_scaled_1d_cast_transpose(input,output,scale_inv,output_t,scale_inv_t,noop,
                                                 epsilon,pow2_scale,all_gather_usage,out_dtype)
