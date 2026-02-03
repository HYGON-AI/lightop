//delete when compile whl
#include "moe_w8a8_utils.h"
#include "moe_w8a8_config.h"

namespace at {
namespace native {


#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define ATOMIC_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      torch::Tensor output_tmp = output.alias();\
      constexpr static bool CONST_NAME = true;  \
      const int BLOCK_SIZE_K_ = 128;            \
      return __VA_ARGS__();                     \
    } else {                                    \
      torch::Tensor output_tmp = output.alias();\
      constexpr static bool CONST_NAME = false; \
      const int BLOCK_SIZE_K_ = 128;            \
      return __VA_ARGS__();                     \
    }                                           \
  }()
  

template<
  typename Element,
  typename OutputType,
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K, 
  uint16_t GROUP_N,
  uint16_t GROUP_K,
  int STAGES,
  bool mul_topk_weight> 
__global__ void __launch_bounds__(1024) MOE_W8A8_FP8_BLOCKWISE_MARLIN_HIP_NT_DECODE_UP(
  //const Element* __restrict__ input,
  const Element* input,
  const Element* __restrict__ qweight,
  bhalf_t* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const float* __restrict__ topk_weights,
  const int32_t* __restrict__ sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_post_pad,
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k, 
  uint32_t stride_asm,
  uint32_t stride_ask,
  uint32_t stride_bse,
  uint32_t stride_bsn, 
  uint32_t stride_bsk,
  uint32_t sorted_token_lens,
  uint32_t top_k,
  uint32_t delta 
) {
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k 

  // int bidx = blockIdx.x; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  // int bidy = blockIdx.y; // pid_n
  // int bidz = blockIdx.z; // pid_k 
  // GetBLockIdx(bidx + bidy * gridDim.x, gridDim.x, gridDim.y, 0, 4, bidx, bidy);

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m*top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  //constexpr int STAGES = 4; 
  const uint32_t input_offset = bidz * BLOCK_SIZE_K; //输入k方向分块的位置
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移
  //const uint64_t qweight_offset = expert_offset + bidy * size_k * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块
  const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * 64; // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N; // 计算之后是mxn,这应该是计算输出n方向的位置
  
  auto g_input = input;
  auto g_qweight = qweight + qweight_offset;
  bhalf_t* g_output =  output + output_offset;

  if(expert_id == -1){ //EP算法处理 epxert_id为-1 写回0
    return;
  }  

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;
  constexpr int group_size = 128;

  int warp_id_vec = threadIdx.x / 64; //warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63; // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[]; // 声明lds信息
  Element* input_lds = (Element*)&(smem); // decode这里没用
  Element* qweight_lds = input_lds; // decode这里没用
  //bhalf_t* output_lds = (bhalf_t*)&(smem); // 重复使用lds,留给output_lds
  uint16_t* output_lds = (uint16_t*)&(smem); // 重复使用lds,留给output_lds
  const int *sorted_token_ids_offset = sorted_token_ids;

  // union_vec<Element, 8> A_reg[WARP_M / mfma_m * STAGES][WARP_K/mfma_k]; 
  // union_vec<Element, 8> B_reg[WARP_N / mfma_n][WARP_K/mfma_k];
  union_vec<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES]; 
  union_vec<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][STAGES];
  floatx4 C_reg[2][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0}; // [2]一个作为最后的输出 一个作为中间结果

  const uint32_t weight_scale_offset = stride_bse * expert_id + (bidy * BLOCK_SIZE_N / group_size) * stride_bsn; // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t input_scale_offset = warp_k_id * WARP_K / group_size; // 计算k方向的warp偏移
  auto g_input_scale = input_scale + input_scale_offset; // 配置全局显存信息  k方向缺少col_id的偏移
  auto g_weight_scale = weight_scale + weight_scale_offset + (warp_n_id*WARP_N / group_size) * stride_bsn + input_scale_offset; // 缺少row_id和col_id的偏移
  float* a_scale_ptr; // warpM大于16需要单独处理
  #pragma unroll
  for(int idx = 0; idx < WARP_M / mfma_m; idx++){
      int sorted_token_offset = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
      a_scale_ptr = (g_input_scale + std::min(sorted_token_offset / top_k,  size_m - 1) * stride_asm); //添加m方向的偏移
  }


  { 
    // gemm_nt_marlin_fp8_decode<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>
    // (g_input, g_qweight, input_lds, qweight_lds, a_scale_ptr, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, 
    //   stride_asm, stride_ask, size_n, stride_bsn, stride_bsk, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);

    //gemm_nt_marlin_fp8_decode  gemm_nt_marlin_blockwise_decode
    gemm_nt_marlin_blockwise_decode<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>
    (g_input, g_qweight, input_lds, qweight_lds, a_scale_ptr, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, 
      stride_asm, stride_ask, size_n, stride_bsn, stride_bsk, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
  
  }

  // save output to lds 
#if 1
  if(warp_k_id == 0 || warp_k_num == 1)
  {
    for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
      for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
        #pragma unroll
        for(int reg_id = 0; reg_id < 4; reg_id++){
          float value = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n][reg_id];
          int index = min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15 ) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + (lane_id % 16) )/2 * 2/*padding*/;
          output_lds[index] = f32_to_bf16(value);
          // #if defined(__gfx938__)
          // output_lds[index] = __builtin_hcu_cvt_bf16_f32(value, false, false); //fp32->bf16
          // #endif
        }
      }
    }
  }

  __syncthreads();

  
  {
    // 最大化利用dwordx4 通用
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8;  //N方向需要的线程数 使用dwordx4即8个bf16
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
    int m_idx = threadIdx.x / N_thread;  
    int n_idx = threadIdx.x % N_thread;  
    for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
      const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];

      if(token_index < size_m * top_k) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[token_index * size_n + n_idx * 8]) = 
        *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
      }
    } 
  }      
#else
  using vec_bf16_4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) unsigned short;
  if(warp_k_id == 0 || warp_k_num == 1)
  {
    for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
      float *a_scale = a_scale_ptr_arr[min_tile_m];
      int m_idx = bidx * BLOCK_SIZE_M + min_tile_m * mfma_m + row_id;
      int token_index = sorted_token_ids[std::min(m_idx, int(sorted_token_lens - 1))];

      for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n  && token_index < size_m * top_k; min_tile_n++){
        vec_bf16_4 bf16_vec4;
        int n_idx = min_tile_n * mfma_n + warp_n_id * WARP_N + col_id*4;

        #pragma unroll
        for(int reg_id = 0; reg_id < 4; reg_id++){
          float *b_scale = b_scale_ptr + min_tile_n * mfma_n + col_id + reg_id * 4;
          float value = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n][reg_id] * a_scale[0] * b_scale[0];
          int index = min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15 ) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16;
          bf16_vec4[reg_id] = f32_to_bf16(value);
        }
        *reinterpret_cast<vec_bf16_4*>(&g_output[token_index * size_n + n_idx]) = bf16_vec4;
      }
    }
  }
#endif  

}


//prefill
template<
  typename Element, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K, 
  uint16_t GROUP_N,
  uint16_t GROUP_K,
  bool mul_topk_weight> 
__global__ void __launch_bounds__(1024) MOE_W8A8_FP8_BLOCKWISE_MARLIN_HIP_NT_PREFILL_UP(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  bhalf_t* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const float* __restrict__ topk_weights,
  const int32_t* __restrict__ sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_post_pad,
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k, 
  uint32_t stride_asm,
  uint32_t stride_ask,
  uint32_t stride_bse,
  uint32_t stride_bsn, 
  uint32_t stride_bsk,
  uint32_t sorted_token_lens,
  uint32_t top_k,
  uint32_t delta  
) {

  int bidx = blockIdx.x; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  int bidy = blockIdx.y; // pid_n
  int bidz = blockIdx.z; // pid_k 
  int marlin_size_k = size_n * WARP_K;
  // GetBLockIdx(bidx + bidy * gridDim.x, gridDim.x, gridDim.y, 0, 8, bidx, bidy);

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m*top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  
  constexpr int STAGES = BLOCK_SIZE_K/WARP_K; 
  constexpr int group_size = 128;
  const uint32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引
  const uint32_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移
  const uint32_t qweight_offset = expert_offset + bidy * WARP_K * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N; // 计算之后是mxn,这应该是计算输出n方向的位置
  
  auto g_input = input;
  auto g_qweight = qweight + qweight_offset;
  bhalf_t* g_output = output + output_offset;

  if(expert_id == -1){ //EP算法处理 epxert_id为-1 写回0
    return;
  } 

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;                                  //warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);           // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                                      // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  int warp_n_id = warp_id % warp_n_num;
  int warp_m_id = warp_id / warp_n_num;
  extern __shared__ Element smem[];                                    // 声明lds信息
  Element* input_lds = (Element*)&(smem);                              // 输入lds首地址,首地址开始分配给input
  Element* qweight_lds = input_lds + STAGES * (BLOCK_SIZE_M * WARP_K); // 按照最简单的方式分块
  bhalf_t* output_lds = (bhalf_t*)&(smem);                             // 重复使用lds,留给output_lds
  const int *sorted_token_ids_offset = sorted_token_ids;

  union_vec<Element, 8> A_reg[WARP_M / mfma_m * STAGES][WARP_K/mfma_k]; 
  union_vec<Element, 8> B_reg[WARP_N / mfma_n][WARP_K/mfma_k];
  floatx4 C_reg[2][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0};   //[2]是为了最后计算输出和中间结果

  const uint32_t weight_scale_offset = stride_bse * expert_id + (bidy * BLOCK_SIZE_N / group_size) * stride_bsn; // 具体偏移到对应专家的weight的某一个小的分块
  auto g_input_scale = input_scale; // 配置全局显存信息  k方向缺少col_id的偏移
  auto g_weight_scale = weight_scale + weight_scale_offset + (warp_n_id*WARP_N / group_size) * stride_bsn; // 缺少row_id和col_id的偏移
  float* a_scale_ptr[WARP_M / mfma_m]; // warpM大于16需要单独处理
  #pragma unroll
  for(int idx = 0; idx < WARP_M / mfma_m; idx++){
      int sorted_token_offset = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + warp_m_id * WARP_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
      a_scale_ptr[idx] = (g_input_scale + std::min(sorted_token_offset / top_k,  size_m - 1) * stride_asm); //添加m方向的偏移
  }

  { 
    gemm_nt_fp8_marlin_blockwise_prefill<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>
    (g_input, g_qweight, input_lds, qweight_lds, a_scale_ptr, g_weight_scale, size_m, A_reg, B_reg, C_reg, 
      warp_id, size_k, marlin_size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);

    // gemm_nt_fp8_marlin_prefill<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>
    // (g_input, g_qweight, input_lds, qweight_lds, a_scale_ptr, g_weight_scale, size_m, A_reg, B_reg, C_reg, 
    //   warp_id, size_k, marlin_size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, 
    //   sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);

    // gemm_nt_fp8_marlin_prefill<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>
    // (g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, 
    //   warp_id, size_k, marlin_size_k, stride_asm, 
    //   stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
  }
  //save output to lds
  for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
    for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
      #pragma unroll
      for(int reg_id = 0; reg_id < 4; reg_id++){
        float value = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n][reg_id];
        int index = warp_m_id * WARP_M * BLOCK_SIZE_N + min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15 ) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + warp_m_id * WARP_M+ (lane_id % 16) )/2 * 2/*padding*/;
        output_lds[index] = f32_to_bf16(value);
      }
    }
  }

  __syncthreads();

  {
    // 最大化利用dwordx4 通用
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8;  //N方向需要的线程数 使用dwordx4即8个bf16
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
    int m_idx = threadIdx.x / N_thread;  
    int n_idx = threadIdx.x % N_thread;  
    for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
      const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];

      if(token_index < size_m * top_k) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[token_index * size_n + n_idx * 8]) = 
        //*reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8]);
        *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
      }
    } 
  }      
}

template<
  typename Element,
  typename OutputType,
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K, 
  uint16_t GROUP_N,
  uint16_t GROUP_K,
  int STAGES,
  bool mul_topk_weight> 
__global__ void __launch_bounds__(1024) MOE_W8A8_FP8_BLOCKWISE_MARLIN_HIP_NT_DECODE_DOWN(
  //const Element* __restrict__ input,
  const Element* input,
  const Element* __restrict__ qweight,
  bhalf_t* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const float* __restrict__ topk_weights,
  const int32_t* __restrict__ sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_post_pad,
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k, 
  uint32_t stride_asm,
  uint32_t stride_ask,
  uint32_t stride_bse,
  uint32_t stride_bsn, 
  uint32_t stride_bsk,
  uint32_t sorted_token_lens,
  uint32_t top_k,
  uint32_t delta 
) {
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k 


  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m*top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  //constexpr int STAGES = 4; 
  const uint32_t input_offset = bidz * BLOCK_SIZE_K; //输入k方向分块的位置
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移
  //const uint64_t qweight_offset = expert_offset + bidy * size_k * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块
  const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * 64; // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N; // 计算之后是mxn,这应该是计算输出n方向的位置
  
  auto g_input = input;
  auto g_qweight = qweight + qweight_offset;
  bhalf_t* g_output =  output + output_offset;

  if(expert_id == -1){ //EP算法处理 epxert_id为-1 写回0
    return;
  }  

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;
  constexpr int group_size = 128;

  int warp_id_vec = threadIdx.x / 64; //warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63; // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem[]; // 声明lds信息
  Element* input_lds = (Element*)&(smem); // decode这里没用
  Element* qweight_lds = input_lds; // decode这里没用
  //bhalf_t* output_lds = (bhalf_t*)&(smem); // 重复使用lds,留给output_lds
  uint16_t* output_lds = (uint16_t*)&(smem); // 重复使用lds,留给output_lds
  const int *sorted_token_ids_offset = sorted_token_ids;

  // union_vec<Element, 8> A_reg[WARP_M / mfma_m * STAGES][WARP_K/mfma_k]; 
  // union_vec<Element, 8> B_reg[WARP_N / mfma_n][WARP_K/mfma_k];
  union_vec<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES]; 
  union_vec<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][STAGES];
  floatx4 C_reg[2][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0}; // [2]一个作为最后的输出 一个作为中间结果

  const uint32_t weight_scale_offset = stride_bse * expert_id + (bidy * BLOCK_SIZE_N / group_size) * stride_bsn; // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t input_scale_offset = warp_k_id * WARP_K / group_size; // 计算k方向的warp偏移
  auto g_input_scale = input_scale + input_scale_offset; // 配置全局显存信息  k方向缺少col_id的偏移
  auto g_weight_scale = weight_scale + weight_scale_offset + (warp_n_id*WARP_N / group_size) * stride_bsn + input_scale_offset; // 缺少row_id和col_id的偏移
  float* a_scale_ptr; // warpM大于16需要单独处理
  float topk_weights_value[WARP_M / mfma_m];

  #pragma unroll
  for(int idx = 0; idx < WARP_M / mfma_m; idx++){
      int sorted_token_offset = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
      int token_index_safe = std::min(sorted_token_offset, size_m * top_k-1);
      a_scale_ptr = (g_input_scale + token_index_safe * stride_asm); //添加m方向的偏移
      topk_weights_value[idx] = topk_weights[token_index_safe];
  }


  { 
    // gemm_nt_marlin_fp8_decode<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>
    // (g_input, g_qweight, input_lds, qweight_lds, a_scale_ptr, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, 
    //   stride_asm, stride_ask, size_n, stride_bsn, stride_bsk, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);

    //gemm_nt_marlin_fp8_decode  gemm_nt_marlin_blockwise_decode
    gemm_nt_marlin_blockwise_decode<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>
    (g_input, g_qweight, input_lds, qweight_lds, a_scale_ptr, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, 
      stride_asm, stride_ask, size_n, stride_bsn, stride_bsk, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
  
  }

  // save output to lds 
#if 1
  if(warp_k_id == 0 || warp_k_num == 1)
  {
    for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
      for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
        #pragma unroll
        for(int reg_id = 0; reg_id < 4; reg_id++){
          float value = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n][reg_id] * topk_weights_value[min_tile_m];
          int index = min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15 ) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + (lane_id % 16) )/2 * 2/*padding*/;
          output_lds[index] = f32_to_bf16(value);
          // #if defined(__gfx938__)
          // output_lds[index] = __builtin_hcu_cvt_bf16_f32(value, false, false); //fp32->bf16
          // #endif
        }
      }
    }
  }

  __syncthreads();

  
  {
    // 最大化利用dwordx4 通用
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8;  //N方向需要的线程数 使用dwordx4即8个bf16
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
    int m_idx = threadIdx.x / N_thread;  
    int n_idx = threadIdx.x % N_thread;  
    for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
      const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];

      if(token_index < size_m * top_k) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[token_index * size_n + n_idx * 8]) = 
        *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
      }
    } 
  }      
#else
  using vec_bf16_4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) unsigned short;
  if(warp_k_id == 0 || warp_k_num == 1)
  {
    for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
      float *a_scale = a_scale_ptr_arr[min_tile_m];
      int m_idx = bidx * BLOCK_SIZE_M + min_tile_m * mfma_m + row_id;
      int token_index = sorted_token_ids[std::min(m_idx, int(sorted_token_lens - 1))];

      for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n  && token_index < size_m * top_k; min_tile_n++){
        vec_bf16_4 bf16_vec4;
        int n_idx = min_tile_n * mfma_n + warp_n_id * WARP_N + col_id*4;

        #pragma unroll
        for(int reg_id = 0; reg_id < 4; reg_id++){
          float *b_scale = b_scale_ptr + min_tile_n * mfma_n + col_id + reg_id * 4;
          float value = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n][reg_id] * a_scale[0] * b_scale[0];
          int index = min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15 ) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16;
          bf16_vec4[reg_id] = f32_to_bf16(value);
        }
        *reinterpret_cast<vec_bf16_4*>(&g_output[token_index * size_n + n_idx]) = bf16_vec4;
      }
    }
  }
#endif  

}

//prefill
template<
  typename Element, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K, 
  uint16_t GROUP_N,
  uint16_t GROUP_K,
  bool mul_topk_weight> 
__global__ void __launch_bounds__(1024) MOE_W8A8_FP8_BLOCKWISE_MARLIN_HIP_NT_PREFILL_DOWN(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  bhalf_t* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const float* __restrict__ topk_weights,
  const int32_t* __restrict__ sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_post_pad,
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k, 
  uint32_t stride_asm,
  uint32_t stride_ask,
  uint32_t stride_bse,
  uint32_t stride_bsn, 
  uint32_t stride_bsk,
  uint32_t sorted_token_lens,
  uint32_t top_k,
  uint32_t delta  
) {

  int bidx = blockIdx.x; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  int bidy = blockIdx.y; // pid_n
  int bidz = blockIdx.z; // pid_k 
  int marlin_size_k = size_n * WARP_K;
  // GetBLockIdx(bidx + bidy * gridDim.x, gridDim.x, gridDim.y, 0, 8, bidx, bidy);

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m*top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  
  constexpr int STAGES = BLOCK_SIZE_K/WARP_K; 
  constexpr int group_size = 128;
  const uint32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引
  const uint32_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移
  const uint32_t qweight_offset = expert_offset + bidy * WARP_K * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N; // 计算之后是mxn,这应该是计算输出n方向的位置
  
  auto g_input = input;
  auto g_qweight = qweight + qweight_offset;
  bhalf_t* g_output = output + output_offset;

  if(expert_id == -1){ //EP算法处理 epxert_id为-1 写回0
    return;
  } 

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;                                  //warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);           // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                                      // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  int warp_n_id = warp_id % warp_n_num;
  int warp_m_id = warp_id / warp_n_num;
  extern __shared__ Element smem[];                                    // 声明lds信息
  Element* input_lds = (Element*)&(smem);                              // 输入lds首地址,首地址开始分配给input
  Element* qweight_lds = input_lds + STAGES * (BLOCK_SIZE_M * WARP_K); // 按照最简单的方式分块
  bhalf_t* output_lds = (bhalf_t*)&(smem);                             // 重复使用lds,留给output_lds
  const int *sorted_token_ids_offset = sorted_token_ids;

  union_vec<Element, 8> A_reg[WARP_M / mfma_m * STAGES][WARP_K/mfma_k]; 
  union_vec<Element, 8> B_reg[WARP_N / mfma_n][WARP_K/mfma_k];
  floatx4 C_reg[2][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0};   //[2]是为了最后计算输出和中间结果

  const uint32_t weight_scale_offset = stride_bse * expert_id + (bidy * BLOCK_SIZE_N / group_size) * stride_bsn; // 具体偏移到对应专家的weight的某一个小的分块
  auto g_input_scale = input_scale; // 配置全局显存信息  k方向缺少col_id的偏移
  auto g_weight_scale = weight_scale + weight_scale_offset + (warp_n_id*WARP_N / group_size) * stride_bsn; // 缺少row_id和col_id的偏移
  float* a_scale_ptr[WARP_M / mfma_m]; // warpM大于16需要单独处理
  float topk_weights_value[WARP_M / mfma_m];
  #pragma unroll
  for(int idx = 0; idx < WARP_M / mfma_m; idx++){
      int sorted_token_offset = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + warp_m_id * WARP_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
      int token_index_safe = std::min(sorted_token_offset, size_m - 1);
      a_scale_ptr[idx] = (g_input_scale + token_index_safe * stride_asm); //添加m方向的偏移
      topk_weights_value[idx] = topk_weights[token_index_safe];
  }

  { 
    gemm_nt_fp8_marlin_blockwise_prefill<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>
    (g_input, g_qweight, input_lds, qweight_lds, a_scale_ptr, g_weight_scale, size_m, A_reg, B_reg, C_reg, 
      warp_id, size_k, marlin_size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
  }
  //save output to lds
  for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
    for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
      #pragma unroll
      for(int reg_id = 0; reg_id < 4; reg_id++){
        float value = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n][reg_id] * topk_weights_value[min_tile_m];
        int index = warp_m_id * WARP_M * BLOCK_SIZE_N + min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15 ) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + warp_m_id * WARP_M+ (lane_id % 16) )/2 * 2/*padding*/;
        output_lds[index] = f32_to_bf16(value);
      }
    }
  }

  __syncthreads();

  {
    // 最大化利用dwordx4 通用
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8;  //N方向需要的线程数 使用dwordx4即8个bf16
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
    int m_idx = threadIdx.x / N_thread;  
    int n_idx = threadIdx.x % N_thread;  
    for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
      const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];

      if(token_index < size_m * top_k) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[token_index * size_n + n_idx * 8]) = 
        //*reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8]);
        *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
      }
    } 
  }      
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// tmp src perchannel
template<
  typename Element, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K, 
  uint16_t GROUP_N,
  uint16_t GROUP_K,
  bool mul_topk_weight> 
__global__ void __launch_bounds__(1024) MOE_W8A8_FP8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP_T(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  bhalf_t* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const float* __restrict__ topk_weights,
  const int32_t* __restrict__ sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_post_pad,
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k, 
  uint32_t stride_asm,
  uint32_t stride_ask,
  uint32_t stride_bse,
  uint32_t stride_bsn, 
  uint32_t stride_bsk,
  uint32_t sorted_token_lens,
  uint32_t top_k,
  uint32_t delta  
) {

  int bidx = blockIdx.x; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  int bidy = blockIdx.y; // pid_n
  int bidz = blockIdx.z; // pid_k 
  int marlin_size_k = size_n * WARP_K;
  // GetBLockIdx(bidx + bidy * gridDim.x, gridDim.x, gridDim.y, 0, 8, bidx, bidy);

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m*top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  
  constexpr int STAGES = BLOCK_SIZE_K/WARP_K; 
  const uint32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移
  const uint64_t qweight_offset = expert_offset + bidy * WARP_K * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N; // 计算之后是mxn,这应该是计算输出n方向的位置

  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块
  
  auto g_input = input;
  auto g_input_scale = input_scale; // 配置全局显存信息
  bhalf_t* g_output;

  g_output = output + output_offset;

  if(expert_id == -1){ //EP算法处理 epxert_id为-1 写回0
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8;  //N方向需要的线程数 使用dwordx4即8个bf16
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
    vec_bf16_8 zero_bf16_8;

    #pragma unroll
    for(int i = 0; i < 8; ++i)
      zero_bf16_8[i] = 0;

    int m_idx = threadIdx.x / N_thread;  
    int n_idx = threadIdx.x % N_thread;  
    for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
      const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];

      if(token_index < size_m * top_k) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[token_index * size_n + n_idx * 8]) = zero_bf16_8;
      }
    } 
    return;
  } 

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

  int warp_id_vec = threadIdx.x / 64;                                  //warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);           // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                                      // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  int warp_n_id = warp_id % warp_n_num;
  int warp_m_id = warp_id / warp_n_num;
  extern __shared__ Element smem[];                                    // 声明lds信息
  Element* input_lds = (Element*)&(smem);                              // 输入lds首地址,首地址开始分配给input
  Element* qweight_lds = input_lds + STAGES * (BLOCK_SIZE_M * WARP_K); // 按照最简单的方式分块
  bhalf_t* output_lds = (bhalf_t*)&(smem);                             // 重复使用lds,留给output_lds
  const int *sorted_token_ids_offset = sorted_token_ids;

  union_vec<Element, 8> A_reg[WARP_M / mfma_m * STAGES][WARP_K/mfma_k]; 
  union_vec<Element, 8> B_reg[WARP_N / mfma_n][WARP_K/mfma_k];

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;            // 配置weight的scale信息 todo 这里n_loop=0没有计算偏移
  floatx4 C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0};                 // [4][2]  每个warp在n方向重复两次 tileN = 16*2
  float *a_scale_ptr_arr[WARP_M / mfma_m];
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id*WARP_N;

  { 
    gemm_nt_fp8_marlin_prefill<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>
    (g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, marlin_size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
  }
  //save output to lds
  for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
    for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
      #pragma unroll
      for(int reg_id = 0; reg_id < 4; reg_id++){
        float value = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n][reg_id];
        int index = warp_m_id * WARP_M * BLOCK_SIZE_N + min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15 ) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + warp_m_id * WARP_M+ (lane_id % 16) )/2 * 2/*padding*/;
        output_lds[index] = f32_to_bf16(value);
      }
    }
  }

  __syncthreads();

  {
    // 最大化利用dwordx4 通用
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8;  //N方向需要的线程数 使用dwordx4即8个bf16
    using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
    int m_idx = threadIdx.x / N_thread;  
    int n_idx = threadIdx.x % N_thread;  
    for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
      const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];

      if(token_index < size_m * top_k) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[token_index * size_n + n_idx * 8]) = 
        //*reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8]);
        *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
      }
    } 
  }      
}


// launch_gemm1
// 在GemmParams结构体后添加
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_moe_w8a8_blockwise_first_stage_prefill(const GemmParams<T>& params) {
    const int STAGES = BLOCK_SIZE_K/WARP_K;
    const int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N);
    const bool mul_topk_weight = false;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;

    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;


    gridDim.x = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
    gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
    gridDim.z = 1; // k方向

    const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + BLOCK_SIZE_N * WARP_K), BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
    const int shared_mem_size = lds_size;
    const hipStream_t stream = at::cuda::getCurrentHIPStream();

    { //marlin版本    
      // MOE_W8A8_FP8_BLOCKWISE_MARLIN_HIP_NT_PREFILL_UP  MOE_W8A8_FP8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP
      MOE_W8A8_FP8_BLOCKWISE_MARLIN_HIP_NT_PREFILL_UP<uint8_t, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
      GROUP_N, GROUP_K, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
      params.ptr_A, 
      params.ptr_B0, 
      params.ptr_C, 
      params.ptr_A_scale, 
      params.ptr_B_scale,  
      params.topk_weights,
      params.sorted_token_ids,
      params.expert_ids, 
      params.num_tokens_post_pad_ptr,
      params.size_m,
      params.size_n,
      params.size_k,
      params.stride_asm,
      params.stride_ask,
      params.stride_bse,
      params.stride_bsn,
      params.stride_bsk,
      params.sorted_token_lens,
      params.top_k,
      params.delta);
    }

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
    }
    
}


// launch_gemm1
// 在GemmParams结构体后添加
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_moe_w8a8_blockwise_second_stage_prefill(const GemmParams<T>& params) {
  const int STAGES = BLOCK_SIZE_K/WARP_K;
  const int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N);
  const bool mul_topk_weight = false;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;

  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM*64;
  blockDim.y = 1;
  blockDim.z = 1;


  gridDim.x = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
  gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
  gridDim.z = 1; // k方向

  const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + BLOCK_SIZE_N * WARP_K), BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
  const int shared_mem_size = lds_size;
  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  { //marlin版本    
    // MOE_W8A8_FP8_BLOCKWISE_MARLIN_HIP_NT_PREFILL_UP  MOE_W8A8_FP8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP
    MOE_W8A8_FP8_BLOCKWISE_MARLIN_HIP_NT_PREFILL_DOWN<uint8_t, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
    GROUP_N, GROUP_K, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
    params.ptr_A, 
    params.ptr_B0, 
    params.ptr_C, 
    params.ptr_A_scale, 
    params.ptr_B_scale,  
    params.topk_weights,
    params.sorted_token_ids,
    params.expert_ids, 
    params.num_tokens_post_pad_ptr,
    params.size_m,
    params.size_n,
    params.size_k,
    params.stride_asm,
    params.stride_ask,
    params.stride_bse,
    params.stride_bsn,
    params.stride_bsk,
    params.sorted_token_lens,
    params.top_k,
    params.delta);
  }

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
  }
    

}

// launch_gemm1_decode
// warp在n和k方向排列
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T>
void launch_moe_w8a8_blockwise_first_stage_decode(const GemmParams<T>& params) {
    //constexpr int STAGES = 2;
    constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
    const bool mul_topk_weight = false;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;

    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;

    const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 4;//max(BLOCK_SIZE_M * BLOCK_SIZE_N * 2, BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
    const int shared_mem_size = lds_size;
    const hipStream_t stream = at::cuda::getCurrentHIPStream();

    
    { //marlin版本    
      gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
      gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
      gridDim.x = 1; // k方向

      MOE_W8A8_FP8_BLOCKWISE_MARLIN_HIP_NT_DECODE_UP<uint8_t, bhalf_t, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
      GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
      params.ptr_A, 
      params.ptr_B0, 
      params.ptr_C, 
      params.ptr_A_scale, 
      params.ptr_B_scale,  
      params.topk_weights,
      params.sorted_token_ids,
      params.expert_ids, 
      params.num_tokens_post_pad_ptr,
      params.size_m,
      params.size_n,
      params.size_k,
      params.stride_asm,
      params.stride_ask,
      params.stride_bse,
      params.stride_bsn,
      params.stride_bsk,
      params.sorted_token_lens,
      params.top_k,
      params.delta);
    }

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
    }
}

// launch_gemm2_decode
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T>
void launch_moe_w8a8_blockwise_second_stage_decode(const GemmParams<T>& params) {
    //const int STAGES = 2;
    const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
    const bool mul_topk_weight = true;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;

    const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 4;//max(BLOCK_SIZE_M * BLOCK_SIZE_N * 2, BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
    const int shared_mem_size = lds_size;
    const hipStream_t stream = at::cuda::getCurrentHIPStream();

    
    { //marlin版本    
      gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
      gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
      gridDim.x = 1; // k方向

      MOE_W8A8_FP8_BLOCKWISE_MARLIN_HIP_NT_DECODE_DOWN<uint8_t, bhalf_t, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
      GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
      params.ptr_A, 
      params.ptr_B0, 
      params.ptr_C, 
      params.ptr_A_scale, 
      params.ptr_B_scale,  
      params.topk_weights,
      params.sorted_token_ids,
      params.expert_ids, 
      params.num_tokens_post_pad_ptr,
      params.size_m,
      params.size_n,
      params.size_k,
      params.stride_asm,
      params.stride_ask,
      params.stride_bse,
      params.stride_bsn,
      params.stride_bsk,
      params.sorted_token_lens,
      params.top_k,
      params.delta);
    }

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
    }

}

// 模板抽象
// BLOCK_MNK: [16, 128, 128]
// WARP_MNK: [16, 32, 64]
// MMA_MNK: [16, 16, 32]
// BLOCK_MNK / WARP_MNK = [1, 4, 2]  代表warp在MN方向的排布 block_k/warp_k代表stage
// WARP_MNK / MMA_MNK = [1, 2, 2]    代表warp在MNK方向重复计算的次数 会分配额外的寄存器
torch::Tensor moe_gemm_marlin_w8a8_blockwise_fp8(torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  std::optional<torch::Tensor> topk_weights,
  torch::Tensor sorted_token_ids, 
  torch::Tensor expert_ids,
  torch::Tensor num_tokens_post_pad, 
  int64_t top_k, // gemm1为topk  gemm2为1  因为gemm1输入为[m, k]  gemm2输入为[m*topk, k]
  int mode,
  int delta
  ) {
    const int size_m = input.size(0); 
    const int EXPERTS = b_qweight.size(0);
    const int size_k = input.size(1);
    const int size_n = b_qweight.size(2)*b_qweight.size(1)/size_k;
    const int topk_size = output.size(1); // 输出为[m, topk_size, n]
    const int stride_asm = a_scale.stride(0);
    const int stride_ask = a_scale.stride(1);
    const int stride_bse = b_scale.stride(0);
    const int stride_bsn = b_scale.stride(1); 
    const int stride_bsk = b_scale.stride(2);
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    bool is_marlin = true; // weight为[E, N, K]时 代表不进行重排
    bool first_stage = true;
    torch::Tensor output_alias = output.alias();

    //printf("size_n: %d stride_bse: %d stride_bsn: %d stride_bsk: %d\n", size_n, stride_bse, stride_bsn, stride_bsk);

    const float* topk_weights_ptr; // 第一阶段这里为null
    if (topk_weights.has_value()){
      topk_weights_ptr = (const float*)topk_weights.value().data_ptr();
      first_stage = false;
    }

    int num_pad = 0;

    int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id
    GemmParams<uint8_t> params_in(
      (const uint8_t*)input.data_ptr(), 
      (const uint8_t*)b_qweight.data_ptr(), 
      (bhalf_t*)output_alias.data_ptr(),
      (float*)a_scale.data_ptr(), 
      (float*)b_scale.data_ptr(),  
      topk_weights_ptr,
      sorted_token_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(), 
      num_pad, //num_tokens_post_pad[0].item<int>(), //这里获取值 会造成device->host的拷贝和一部分空泡
      num_tokens_post_pad.data_ptr<int32_t>(),
      size_m,
      size_n,
      size_k,
      stride_asm,
      stride_ask,
      stride_bse,
      stride_bsn,
      stride_bsk,
      EM,
      top_k,
      delta,
      is_marlin
    );

    
    if(first_stage){
      launch_moe_w8a8_blockwise_first_stage_prefill<16, 32, 128, 16, 32, 64>(params_in); // warpk=128有问题
      //launch_moe_w8a8_blockwise_first_stage_decode<16, 16, 128, 16, 16, 128, 2>(params_in);
    }else{
      launch_moe_w8a8_blockwise_second_stage_prefill<16, 32, 128, 16, 32, 64>(params_in); // warpk=128有问题
      //launch_moe_w8a8_blockwise_second_stage_decode<16, 16, 128, 16, 16, 128, 2>(params_in);
    }
#if 0

    if (input.scalar_type() == at::ScalarType::uint8_t){
      // 根据output类型分发到不同的模板实例
      if (output.scalar_type() == at::ScalarType::BFloat16) {
        // BFloat16 类型
        if(first_stage){
       
          int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id
          GemmParams<uint8_t, bhalf_t> params_in(
            (const uint8_t*)input.data_ptr(), 
            (const uint8_t*)b_qweight.data_ptr(), 
            (bhalf_t*)output_alias.data_ptr(),
            (float*)a_scale.data_ptr(), 
            (float*)b_scale.data_ptr(),  
            topk_weights_ptr,
            sorted_token_ids.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(), 
            num_pad, //num_tokens_post_pad[0].item<int>(), //这里获取值 会造成device->host的拷贝和一部分空泡
            num_tokens_post_pad.data_ptr<int32_t>(),
            size_m,
            size_n,
            size_k,
            stride_asm,
            stride_ask,
            stride_bse,
            stride_bsn,
            stride_bsk,
            EM,
            top_k,
            delta,
            is_marlin
          );

          if(mode >= 500){
            // 使用 kernel_maps
            auto kernel_maps = get_kernel_maps_marlin_gemm1_prefill<bhalf_t>();
            auto it = kernel_maps.find(mode);
            if (it != kernel_maps.end()) {
                it->second(params_in);
            } else {
                printf("No matching kernel configuration found for mode %d, using default settings \n", mode);
                launch_moe_w8a8_blockwise_first_stage_prefill<16, 64, 128, 16, 16, 64, uint8_t, bhalf_t>(params_in);
            }
          }else{ //decode 
            auto kernel_maps = get_kernel_maps_marlin_gemm1_decode<bhalf_t>();
            auto it = kernel_maps.find(mode);
            if (it != kernel_maps.end()) {
                it->second(params_in);
            } else {
                printf("No matching kernel configuration found for mode %d, using default settings \n", mode);
                launch_moe_w8a8_blockwise_first_stage_decode<16, 32, 128, 16, 16, 64, 2, uint8_t, bhalf_t>(params_in);
            }
          }
        }else{ //gemm2
          int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id
            
          // 使用int8类型处理
          GemmParams<uint8_t, bhalf_t> params_in(
            (const uint8_t*)input.data_ptr(), 
            (const uint8_t*)b_qweight.data_ptr(), 
            (bhalf_t*)output_alias.data_ptr(),
            (float*)a_scale.data_ptr(), 
            (float*)b_scale.data_ptr(),  
            topk_weights_ptr,
            sorted_token_ids.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(), 
            num_pad, //num_tokens_post_pad[0].item<int>(),
            num_tokens_post_pad.data_ptr<int32_t>(),
            size_m,
            size_n,
            size_k,
            stride_asm,
            stride_ask,
            stride_bse,
            stride_bsn,
            stride_bsk,
            EM,
            top_k,
            delta,
            is_marlin
          );

          if(mode >= 500){
            auto kernel_maps = get_kernel_maps_marlin_gemm2_prefill<bhalf_t>();
            auto it = kernel_maps.find(mode);
            if (it != kernel_maps.end()) {
                it->second(params_in);
            } else {
                printf("No matching kernel configuration found for mode %d, using default settings \n", mode);
                launch_moe_w8a8_blockwise_second_stage_prefill<16, 64, 128, 16, 16, 64, uint8_t, bhalf_t>(params_in);
            }
          }else{
            auto kernel_maps = get_kernel_maps_marlin_gemm2_decode<bhalf_t>();
            auto it = kernel_maps.find(mode);
            if (it != kernel_maps.end()) {
                it->second(params_in);
            } else {
                printf("No matching kernel configuration found for mode %d, using default settings \n", mode);
                launch_moe_w8a8_blockwise_second_stage_decode<16, 32, 128, 16, 16, 64, 2, uint8_t, bhalf_t>(params_in);
            }
          }
        }
      } else if (output.scalar_type() == at::ScalarType::Half) {
        // Float16 类型
        if(first_stage){
       
          int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id
          GemmParams<uint8_t, half> params_in(
            (const uint8_t*)input.data_ptr(), 
            (const uint8_t*)b_qweight.data_ptr(), 
            (half*)output_alias.data_ptr(),
            (float*)a_scale.data_ptr(), 
            (float*)b_scale.data_ptr(),  
            topk_weights_ptr,
            sorted_token_ids.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(), 
            num_pad,
            num_tokens_post_pad.data_ptr<int32_t>(),
            size_m,
            size_n,
            size_k,
            stride_asm,
            stride_ask,
            stride_bse,
            stride_bsn,
            stride_bsk,
            EM,
            top_k,
            delta,
            is_marlin
          );

          if(mode >= 500){
            auto kernel_maps = get_kernel_maps_marlin_gemm1_prefill<half>();
            auto it = kernel_maps.find(mode);
            if (it != kernel_maps.end()) {
                it->second(params_in);
            } else {
                printf("No matching kernel configuration found for mode %d, using default settings \n", mode);
                launch_moe_w8a8_blockwise_first_stage_prefill<16, 64, 128, 16, 16, 64, uint8_t, half>(params_in);
            }
          }else{ //decode 
            auto kernel_maps = get_kernel_maps_marlin_gemm1_decode<half>();
            auto it = kernel_maps.find(mode);
            if (it != kernel_maps.end()) {
                it->second(params_in);
            } else {
                printf("No matching kernel configuration found for mode %d, using default settings \n", mode);
                launch_moe_w8a8_blockwise_first_stage_decode<16, 32, 128, 16, 16, 64, 2, uint8_t, half>(params_in);
            }
          }
        }else{ //gemm2
          int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id
          // 使用int8类型处理
          GemmParams<uint8_t, half> params_in(
            (const uint8_t*)input.data_ptr(), 
            (const uint8_t*)b_qweight.data_ptr(), 
            (half*)output_alias.data_ptr(),
            (float*)a_scale.data_ptr(), 
            (float*)b_scale.data_ptr(),  
            topk_weights_ptr,
            sorted_token_ids.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(), 
            num_pad,
            num_tokens_post_pad.data_ptr<int32_t>(),
            size_m,
            size_n,
            size_k,
            stride_asm,
            stride_ask,
            stride_bse,
            stride_bsn,
            stride_bsk,
            EM,
            top_k,
            delta,
            is_marlin
          );

          if(mode >= 500){
            auto kernel_maps = get_kernel_maps_marlin_gemm2_prefill<half>();
            auto it = kernel_maps.find(mode);
            if (it != kernel_maps.end()) {
                it->second(params_in);
            } else {
                printf("No matching kernel configuration found for mode %d, using default settings \n", mode);
                launch_moe_w8a8_blockwise_second_stage_prefill<16, 64, 128, 16, 16, 64, uint8_t, half>(params_in);
            }
          }else{
            auto kernel_maps = get_kernel_maps_marlin_gemm2_decode<half>();
            auto it = kernel_maps.find(mode);
            if (it != kernel_maps.end()) {
                it->second(params_in);
            } else {
                printf("No matching kernel configuration found for mode %d, using default settings \n", mode);
                launch_moe_w8a8_blockwise_second_stage_decode<16, 32, 128, 16, 16, 64, 2, uint8_t, half>(params_in);
            }
          }
        }
      } else {
        TORCH_CHECK(false, "moe_w8a8_gemm output只支持bfloat16或float16类型");
      }
    } else {
      TORCH_CHECK(false, "moe_w8a8_gemm only supports int8");
    }
#endif
    return output;
}

}
}