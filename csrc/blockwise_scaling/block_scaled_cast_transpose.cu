// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "block_scaled_1d_cast_transpose.h"

namespace at { 
namespace native {

void blockwise_scaled_cast_transpose(const Tensor& input,std::optional<Tensor> output,std::optional<Tensor> scale_inv,
                                    std::optional<Tensor> output_tanspose,std::optional<Tensor> scale_inv_tanspose,std::optional<Tensor> noop,
                                    const int block_len,const int block_scaling_dim,const bool rowwise,const bool columnwise,
                                    const float epsilon,const bool pow2_scale,const bool all_gather_usage, at::ScalarType out_dtype)
    {

      if(block_scaling_dim==1)
      {
  
          blockwise_scaled_1d_cast_transpose(input,output,scale_inv,output_tanspose,scale_inv_tanspose,noop,block_len,rowwise,
                                             columnwise,epsilon,pow2_scale,all_gather_usage,out_dtype); 
      }

      else if (block_scaling_dim==2)
      {
          TORCH_ERROR("BLOCK_SCALING_2D is not support.");

      }


    }

  }

}
