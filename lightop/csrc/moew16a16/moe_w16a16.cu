
#include "moe_w16a16_utils.h"
#include "moe_w16a16_config.h"
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

/****************************************kernel*************************************************/
template<
  typename Element, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K, 
  bool mul_topk_weight> 
__global__ void __launch_bounds__(1024) MOE_W16A16_HIP_NT_DOWN_decode(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  Element* __restrict__ output,
  const float*  topk_weights,
  const int32_t* __restrict__ sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_post_pad,
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k,
  uint32_t size_kb,
  uint32_t sorted_token_lens,
  uint32_t top_k,
  uint32_t delta,
  uint32_t expert_num
) {
  
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k

  if (bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回
  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m*top_k) return; // 对于无效的block,直接返回
  
  constexpr int STAGES = 1; 
  const uint32_t delta_bidx = bidx / delta;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引
  if(expert_id >= expert_num || expert_id <0) return;
 
  const uint64_t expert_offset = ((uint64_t)size_n) * size_kb  * expert_id; // 这个是对应专家的weight偏移
  
  const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * size_kb; // 具体偏移到对应专家的weight的某一个小的分
  auto g_qweight = qweight + qweight_offset;
  const uint32_t output_offset = bidy *  BLOCK_SIZE_N; // 计算之后是mxn,这应该是计算输出n方向的位置
  auto g_input = input;
  bhalf_t* g_output;
  g_output = output + output_offset;
  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 16;
  int tid = threadIdx.x;
  int warp_id_vec = tid / 64; //warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = tid & 63; // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;

  const int warp_n_num = BLOCK_SIZE_N / WARP_N ;
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
   
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  const int *sorted_token_ids_offset = sorted_token_ids;
  //存储数据的寄存器
  bf16_vec<WARP_K/4> A_reg[WARP_M/mfma_m][2]; //bf16_vec<4>
  bf16_vec<WARP_K/4> B_reg[BLOCK_SIZE_N/WARP_N][WARP_N/mfma_n][2]; //bf16_vec<4>

  floatx4 C_reg[BLOCK_SIZE_N/WARP_N][(WARP_M/16)*(WARP_N/16)];
  for(int init1=0; init1< (BLOCK_SIZE_N/WARP_N); init1++){
    for(int init2=0; init2< ((WARP_M/16)*(WARP_N/16)); init2++){
      C_reg[init1][init2] = {0.0,0.0,0.0,0.0};
    }
  }
  //kpack==2
  if(WARP_K == 32){
    if(BLOCK_SIZE_N > 16){
      MOE_W16A16_HIP_NT_DOWN_decode_kpack2_kernel<false,WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
      (g_input, g_qweight, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_kb, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, bidy,size_n,g_output,topk_weights);   
    }
    else{
      bf16_vec<WARP_K/4> B_reg1[WARP_N/mfma_n][2]; //bf16_vec<4>
      MOE_W16A16_HIP_NT_DOWN_decode_kpack2_kernel2<false,WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
      (g_input, g_qweight, size_m, A_reg, B_reg1, C_reg, warp_id, size_k, size_kb, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, bidy,size_n,g_output,topk_weights);
    }
  }
  else{
    MOE_W16A16_HIP_NT_DOWN_decode_kernel<false,WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
    (g_input, g_qweight, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_kb, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, bidy,size_n,g_output,topk_weights);  
  }
}


template<
  typename Element, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K, 
  bool mul_topk_weight> 
__global__ void __launch_bounds__(1024) MOE_W16A16_MARLIN_HIP_NN_UP_decode(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  bhalf_t* __restrict__ output,
  const float* __restrict__ topk_weights,
  const int32_t* __restrict__ sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_post_pad,
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k,
  uint32_t size_kb,
  uint32_t sorted_token_lens,
  uint32_t top_k,
  uint32_t delta,
  uint32_t expert_num
) {
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k

  if (bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回
  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m*top_k) return; // 对于无效的block,直接返回
  constexpr int STAGES = 2; 
  const uint32_t delta_bidx = bidx / delta;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引
  if(expert_id >=expert_num || expert_id <0) return;
 
  const uint64_t expert_offset = ((uint64_t)size_n) * size_kb  * expert_id; // 这个是对应专家的weight偏移
  
  const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * 16; // 具体偏移到对应专家的weight的某一个小的分块

  const uint32_t output_offset = bidy *  BLOCK_SIZE_N; 

  auto g_qweight = qweight + qweight_offset;
  auto g_input = input ;
  bhalf_t* g_output;
  g_output = output + output_offset;
  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 16;

  int warp_id_vec = threadIdx.x / 64; //warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); 
  int lane_id = threadIdx.x & 63; // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;

  const int warp_n_num = BLOCK_SIZE_N / (WARP_N * 2);
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
   
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem3[]; // 声明lds信息
  float* output_lds = (float*)&(smem3); 

  const int *sorted_token_ids_offset = sorted_token_ids;
  //存储数据的寄存器
  bf16_vec<WARP_K/4> A_reg[WARP_M/mfma_m][2];
  bf16_vec<WARP_K/2> B_reg[WARP_N/mfma_n][2];

  floatx4 C_reg[1][2 * (WARP_M/16)*(WARP_N/16)];
  for(int init=0; init< (2 * (WARP_M/16)*(WARP_N/16)); init++){
    C_reg[0][init] = {0.0,0.0,0.0,0.0};
  }
  if(WARP_K == 32 ){
      MOE_W16A16_MARLIN_HIP_NN_UP_decode_kpack2_kernel<false,WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
    (g_input, g_qweight, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx,bidy);      
  }
  else{
      MOE_W16A16_MARLIN_HIP_NN_UP_decode_kernel<false,WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
    (g_input, g_qweight, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx,bidy);  
  }
  if(warp_k_id == 0){
    using vec_bf16_2 = __attribute__((__vector_size__(2 * sizeof(uint16_t)))) unsigned short;
    for(int m_tile = 0; m_tile < WARP_M / mfma_m; m_tile++){
      int token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + m_tile * mfma_m + row_id, int(sorted_token_lens - 1))];
      for (int i = 0; i < 4; i++) {
          if(token_index < size_m * top_k){
                vec_bf16_2 D;
                for (int j = 0; j < 2; j++) {
                  ((bhalf_t*)&D)[j] = f32_to_bf16((C_reg[0][m_tile*2+j])[i]);
                }
                //写出是16*8 * 768的
                *reinterpret_cast<vec_bf16_2*>(&g_output[token_index * (size_n>>4) + warp_n_id * WARP_N *4 + col_id * 2 + i * 8]) = D;
          }
      }
    }
  }
} 



template<
  typename Element, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K, 
  bool mul_topk_weight> 
__global__ void __launch_bounds__(1024) MOE_W16A16_MARLIN_HIP_NN_DOWN_decode(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  bhalf_t* __restrict__ output,
  const float* __restrict__ topk_weights,
  const int32_t* __restrict__ sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_post_pad,
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k,
  uint32_t size_kb,
  uint32_t sorted_token_lens,
  uint32_t top_k,
  uint32_t delta,
  uint32_t expert_num
) {
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k

  if (bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回
  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m*top_k) return; // 对于无效的block,直接返回
  constexpr int STAGES = 2; 
  const uint32_t delta_bidx = bidx / delta;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引
  if(expert_id >=expert_num || expert_id <0) return;
 
  const uint64_t expert_offset = ((uint64_t)size_n) * size_kb  * expert_id; // 这个是对应专家的weight偏移
  
  const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * 16; // 具体偏移到对应专家的weight的某一个小的分块

  const uint32_t output_offset = bidy *  BLOCK_SIZE_N; 

  auto g_qweight = qweight + qweight_offset;
  auto g_input = input ;
  bhalf_t* g_output;
  g_output = output + output_offset;
  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 16;

  int warp_id_vec = threadIdx.x / 64; //warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63; // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;

  const int warp_n_num = BLOCK_SIZE_N / (WARP_N * 2);
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
   
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ Element smem3[]; // 声明lds信息
  float* output_lds = (float*)&(smem3); 

  const int *sorted_token_ids_offset = sorted_token_ids;
  //存储数据的寄存器
  bf16_vec<WARP_K/4> A_reg[WARP_M/mfma_m][2];
  bf16_vec<WARP_K/2> B_reg[WARP_N/mfma_n][2];

  floatx4 C_reg[1][2 * (WARP_M/16)*(WARP_N/16)];
  for(int init=0; init< (2 * (WARP_M/16)*(WARP_N/16)); init++){
    C_reg[0][init] = {0.0,0.0,0.0,0.0};
  }
  if(WARP_K == 32 ){
      MOE_W16A16_MARLIN_HIP_NN_DOWN_decode_kpack2_kernel<false,WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
    (g_input, g_qweight, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx,bidy,topk_weights);      
  }
  else{
      MOE_W16A16_MARLIN_HIP_NN_DOWN_decode_kernel<false,WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
    (g_input, g_qweight, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx,bidy,topk_weights);  
  }
  if(warp_k_id == 0){
    using vec_bf16_2 = __attribute__((__vector_size__(2 * sizeof(uint16_t)))) unsigned short;
    for(int m_tile = 0; m_tile < WARP_M / mfma_m; m_tile++){
      int token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + m_tile * mfma_m + row_id, int(sorted_token_lens - 1))];
      float topk_weight_value  = (topk_weights + token_index)[0];
      for (int i = 0; i < 4; i++) {
          if(token_index < size_m * top_k){
                vec_bf16_2 D;
                for (int j = 0; j < 2; j++) {
                  ((bhalf_t*)&D)[j] = f32_to_bf16((C_reg[0][m_tile*2+j])[i] * topk_weight_value );
                }
                //写出是16*8 * 768的
                *reinterpret_cast<vec_bf16_2*>(&g_output[token_index * (size_n>>4) + warp_n_id * WARP_N *2 + col_id * 2 + i * 8]) = D;
          }
      }
    }
  }
} 


template<
  typename Element, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K, 
  bool mul_topk_weight> 
__global__ void __launch_bounds__(1024) MOE_W16A16_HIP_NT_UP_prefill(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  Element* __restrict__ output,
  const float* __restrict__ topk_weights,
  const int32_t* __restrict__ sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_post_pad,
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k,
  uint32_t size_kb,
  uint32_t sorted_token_lens,
  uint32_t top_k,
  uint32_t delta,
  uint32_t expert_num
) {
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; 
  const int bidz = blockIdx.x; 

  if (bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回
  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m*top_k) return; // 对于无效的block,直接返回

  constexpr int STAGES = BLOCK_SIZE_K/WARP_K; //1
 
  const uint32_t delta_bidx = bidx / delta;

  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引

  if(expert_id >=expert_num || expert_id <0) return;
 
  const uint64_t expert_offset = ((uint64_t)size_n) * size_kb  * expert_id; // 这个是对应专家的weight偏移
  
  const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * size_kb ; // 具体偏移到对应专家的weight的某一个小的分块

  const uint32_t output_offset = bidy *  BLOCK_SIZE_N; 

  auto g_qweight = qweight + qweight_offset;

  auto g_input = input ;
  
  bhalf_t* g_output;
  
  extern __shared__ Element smem[]; // 声明lds信息

  Element* input_lds = (Element*)&(smem); // 输入lds首地址,首地址开始分配给input
  
  Element* qweight_lds = input_lds + STAGES * (BLOCK_SIZE_M * WARP_K); // 按照最简单的方式分块
  
  bhalf_t* output_lds = (bhalf_t*)&(smem); // 重复使用lds,留给output_lds  
  
  g_output = output + output_offset;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 16;

  int warp_id_vec = threadIdx.x / 64;
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); 
  int lane_id = threadIdx.x & 63; 
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;

  const int warp_n_num = BLOCK_SIZE_N / WARP_N; 
   
  int warp_n_id = warp_id % warp_n_num; //0
  int warp_m_id = warp_id / warp_n_num; //0


  const int *sorted_token_ids_offset = sorted_token_ids;
  
  union_vec<Element,4> A_reg[WARP_M/mfma_m* STAGES][WARP_K/mfma_k]; //bf16_vec<4> A[2][4]
  union_vec<Element,4> B_reg[WARP_N/mfma_n][WARP_K/mfma_k]; //bf16_vec<4> B[1][4]

  floatx4 C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0}; 
  { 
    MOE_W16A16_HIP_NT_prefill_kernel<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
    (g_input, g_qweight, input_lds, qweight_lds, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
  }

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
        *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
      }
    } 
  }      
}

template<
  typename Element, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K, 
  bool mul_topk_weight> 
__global__ void __launch_bounds__(1024) MOE_W16A16_MARLIN_HIP_NT_UP_prefill(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  Element* __restrict__ output,
  const float* __restrict__ topk_weights,
  const int32_t* __restrict__ sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_post_pad,
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k,
  uint32_t size_kb,
  uint32_t sorted_token_lens,
  uint32_t top_k,
  uint32_t delta,
  uint32_t expert_num
) {
  const int bidx = blockIdx.z; // m方向，分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // n方向
  const int bidz = blockIdx.x;
  const int bytes_per_element = 2; 
  const int marlin_warp_k = 16;// marlin weight tile: 16x32
  const int marlin_warp_n = 32;
  const int marlin_warp_size = marlin_warp_k * marlin_warp_n;
  const int marlin_stride_b = size_n * bytes_per_element;
  const int real_size_n = (size_n * size_kb) / size_k;

  if (bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回
  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m*top_k) return; // 对于无效的block,直接返回

  constexpr int STAGES = BLOCK_SIZE_K/WARP_K; //1
  const uint32_t delta_bidx = bidx / delta;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引

  if(expert_id >=expert_num || expert_id <0) return;
 
  const uint64_t expert_offset = ((uint64_t)size_n) * size_kb * expert_id; // 这个是对应专家的weight偏移
  const uint64_t qweight_offset = expert_offset + bidy * (BLOCK_SIZE_N/marlin_warp_n) * marlin_warp_size ; // 具体偏移到对应专家的weight的某一个小的分块

  const uint32_t output_offset = bidy *  BLOCK_SIZE_N; 
  auto g_qweight = qweight + qweight_offset;
  auto g_input = input;
  
  bhalf_t* g_output;
  extern __shared__ Element smem[]; // 声明lds信息
  Element* input_lds = (Element*)&(smem); // 输入lds首地址,首地址开始分配给input
  Element* qweight_lds = input_lds + STAGES * (BLOCK_SIZE_M * WARP_K); // 按照最简单的方式分块
  bhalf_t* output_lds = (bhalf_t*)&(smem); // 重复使用lds,留给output_lds  
  
  g_output = output + output_offset;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 16;

  int warp_id_vec = threadIdx.x / 64;
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); 
  int lane_id = threadIdx.x & 63; 
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;

  const int warp_n_num = BLOCK_SIZE_N / WARP_N; 
   
  int warp_n_id = warp_id % warp_n_num; //0
  int warp_m_id = warp_id / warp_n_num; //0

  const int *sorted_token_ids_offset = sorted_token_ids;
  
  union_vec<Element,4> A_reg[WARP_M/mfma_m * STAGES][WARP_K/mfma_k]; //bf16_vec<4>
  union_vec<Element,8> B_reg[WARP_N/marlin_warp_n][WARP_K/marlin_warp_k]; //bf16_vec<4>

  floatx4 C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0}; 

  MOE_W16A16_MARLIN_HIP_NT_prefill_kernel<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
    (g_input, g_qweight, input_lds, qweight_lds, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);


  //save output to lds
  #pragma unroll
  for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
    #pragma unroll
    for(int min_tile_n = 0; min_tile_n < WARP_N / marlin_warp_n; min_tile_n++){
      #pragma unroll
      for(int n_tile = 0; n_tile < marlin_warp_n / mfma_n; n_tile++){
        #pragma unroll
        for(int reg_id = 0; reg_id < 4; reg_id++){
          float value = C_reg[0][min_tile_m * ((WARP_N/marlin_warp_n) * (marlin_warp_n / mfma_n)) + min_tile_n * (marlin_warp_n / mfma_n) + n_tile][reg_id];
          int index = warp_m_id * WARP_M * BLOCK_SIZE_N + min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * marlin_warp_n + warp_n_id * WARP_N + (lane_id & 15 ) * BLOCK_SIZE_N + 
                    reg_id * (marlin_warp_n / mfma_n) * 4 + (lane_id / 16) * (marlin_warp_n / mfma_n) + n_tile + (min_tile_m * mfma_m + warp_m_id * WARP_M + (lane_id % 16))/2 * 2/*padding*/;// + (lane_id & 15 )/2 * 2/*padding*/

          output_lds[index] = f32_to_bf16(value);
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
        *reinterpret_cast<vec_bf16_8*>(&g_output[token_index * real_size_n + n_idx * 8]) = 
        
        *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
      }
    } 
  }      
}

template<
  typename Element, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K, 
  bool mul_topk_weight> // true
__global__ void __launch_bounds__(1024) MOE_W16A16_HIP_NT_DOWN_prefill(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  bhalf_t* __restrict__ output,
  const float* __restrict__ topk_weights,
  const int32_t* __restrict__ sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_post_pad,
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k,
  uint32_t size_kb,
  uint32_t sorted_token_lens,
  uint32_t top_k,
  uint32_t delta 
) {
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k 

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m*top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  
  const int STAGES = BLOCK_SIZE_K/WARP_K; // 采用两stage
  const uint32_t delta_bidx = bidx / delta;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移
  const uint64_t qweight_offset = expert_offset + bidy * size_k * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N; // 计算之后是mxn,这应该是计算输出n方向的位置

  auto g_input = input;
  bhalf_t* g_output;
  g_output = output + output_offset; 

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 16;

  int warp_id_vec = threadIdx.x / 64; //warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63; // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  int warp_n_id = warp_id % warp_n_num;
  int warp_m_id = warp_id / warp_n_num;

  extern __shared__ Element smem[]; // 声明lds信息
  Element* input_lds = (Element*)&(smem); // 输入lds首地址,首地址开始分配给input
  Element* qweight_lds = input_lds + STAGES * (BLOCK_SIZE_M * WARP_K); // 按照最简单的方式分块
  bhalf_t* output_lds = (bhalf_t*)&(smem); // 重复使用lds,留给output_lds

  const int *sorted_token_ids_offset = sorted_token_ids;

  union_vec<Element,4> A_reg[WARP_M / mfma_m * STAGES][WARP_K/mfma_k]; //
  union_vec<Element,4> B_reg[WARP_N / mfma_n][WARP_K/mfma_k];

  auto g_qweight = qweight + qweight_offset;

  floatx4 C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0}; 

  float weight_value_arr[WARP_M / mfma_m];

  #pragma unroll
  for(int idx = 0; idx < WARP_M / mfma_m; idx++){
      int token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + warp_m_id * WARP_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
      int token_index_safe = std::min(token_index, size_m * top_k-1);
      weight_value_arr[idx] = topk_weights[token_index_safe] ;
  }

  MOE_W16A16_HIP_NT_prefill_kernel<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
    (g_input, g_qweight, input_lds, qweight_lds, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
  
  //save output to lds
  #pragma unroll
  for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
    float weight_value = weight_value_arr[min_tile_m];
    #pragma unroll
    for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
      #pragma unroll
      for(int reg_id = 0; reg_id < 4; reg_id++){
        float value = C_reg[0][min_tile_m * (WARP_N/mfma_n) + min_tile_n][reg_id] * weight_value;
        int index = warp_m_id * WARP_M * BLOCK_SIZE_N + min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15 ) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + warp_m_id * WARP_M + (lane_id % 16))/2 * 2/*padding*/;// + (lane_id & 15 )/2 * 2/*padding*/
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
    int m_idx = threadIdx.x / N_thread;  // tid / 4 = 0-32
    int n_idx = threadIdx.x % N_thread;  // tid % 4 = 0-3
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
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K, 
  bool mul_topk_weight> // true
__global__ void __launch_bounds__(1024) MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  bhalf_t* __restrict__ output,
  const float* __restrict__ topk_weights,
  const int32_t* __restrict__ sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_post_pad,
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k, 
  uint32_t size_kb,
  uint32_t sorted_token_lens,
  uint32_t top_k,
  uint32_t delta 
) {
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k 
  const int bytes_per_element = 2; 
  const int marlin_warp_k = 16;// marlin weight tile: 16x32
  const int marlin_warp_n = 32;
  const int marlin_warp_size = marlin_warp_k * marlin_warp_n;
  const int marlin_stride_b = size_n * bytes_per_element;
  const int real_size_n = (size_n * size_kb) / size_k;

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m*top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  
  const int STAGES = BLOCK_SIZE_K/WARP_K; // 采用两stage
  const uint32_t delta_bidx = bidx / delta;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引
  const uint64_t expert_offset = ((uint64_t)size_n) * size_kb * expert_id; // 这个是对应专家的weight偏移
  const uint64_t qweight_offset = expert_offset + bidy * (BLOCK_SIZE_N/marlin_warp_n) * marlin_warp_size; // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N; // 计算之后是mxn,这应该是计算输出n方向的位置

  auto g_input = input;
  bhalf_t* g_output;
  g_output = output + output_offset; 

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 16;

  int warp_id_vec = threadIdx.x / 64; //warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63; // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  int warp_n_id = warp_id % warp_n_num;
  int warp_m_id = warp_id / warp_n_num;

  extern __shared__ Element smem[]; // 声明lds信息
  Element* input_lds = (Element*)&(smem); // 输入lds首地址,首地址开始分配给input
  Element* qweight_lds = input_lds + STAGES * (BLOCK_SIZE_M * WARP_K); // 按照最简单的方式分块
  bhalf_t* output_lds = (bhalf_t*)&(smem); // 重复使用lds,留给output_lds

  const int *sorted_token_ids_offset = sorted_token_ids;

  union_vec<Element,4> A_reg[WARP_M / mfma_m * STAGES][WARP_K/mfma_k]; //
  union_vec<Element,8> B_reg[WARP_N / marlin_warp_n][WARP_K/marlin_warp_k];

  auto g_qweight = qweight + qweight_offset;

  floatx4 C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0}; 

  float weight_value_arr[WARP_M / mfma_m];

  #pragma unroll
  for(int idx = 0; idx < WARP_M / mfma_m; idx++){
      int token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + warp_m_id * WARP_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
      int token_index_safe = std::min(token_index, size_m * top_k-1);
      weight_value_arr[idx] = topk_weights[token_index_safe] ;
  }

  
  MOE_W16A16_MARLIN_HIP_NT_prefill_kernel<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
    (g_input, g_qweight, input_lds, qweight_lds, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
  
  //save output to lds
  #pragma unroll
  for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
    float weight_value = weight_value_arr[min_tile_m];
    #pragma unroll
    for(int min_tile_n = 0; min_tile_n < WARP_N / marlin_warp_n; min_tile_n++){
      #pragma unroll
      for(int n_tile = 0; n_tile < marlin_warp_n / mfma_n; n_tile++){
        #pragma unroll
        for(int reg_id = 0; reg_id < 4; reg_id++){
          float value = C_reg[0][min_tile_m * ((WARP_N/marlin_warp_n) * (marlin_warp_n / mfma_n)) + min_tile_n * (marlin_warp_n / mfma_n) + n_tile][reg_id] * weight_value;
          int index = warp_m_id * WARP_M * BLOCK_SIZE_N + min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * marlin_warp_n + warp_n_id * WARP_N + (lane_id & 15 ) * BLOCK_SIZE_N + 
                        reg_id * (marlin_warp_n / mfma_n) * 4 + (lane_id / 16) * (marlin_warp_n / mfma_n) + n_tile + (min_tile_m * mfma_m + warp_m_id * WARP_M + (lane_id % 16))/2 * 2/*padding*/;// + (lane_id & 15 )/2 * 2/*padding*/
          output_lds[index] = f32_to_bf16(value);
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
    int m_idx = threadIdx.x / N_thread;  // tid / 4 = 0-32
    int n_idx = threadIdx.x % N_thread;  // tid % 4 = 0-3
    for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
      const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];

      if(token_index < size_m * top_k) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[token_index * real_size_n + n_idx * 8]) = 
        *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
      }
    } 
  }
       

}



template<
  typename Element, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K, 
  bool mul_topk_weight> 
__global__ void __launch_bounds__(1024) MOE_W16A16_HIP_NT_UP_decode(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  Element* __restrict__ output,
  const float* __restrict__ topk_weights,
  const int32_t* __restrict__ sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_post_pad,
  uint32_t size_m,
  uint32_t size_n, 
  uint32_t size_k,
  uint32_t size_kb,
  uint32_t sorted_token_lens,
  uint32_t top_k,
  uint32_t delta,
  uint32_t expert_num
) {
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // 0~47
  const int bidz = blockIdx.x; 

  if (bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回
  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m*top_k) return; // 对于无效的block,直接返回
  constexpr int STAGES = 2; 
  const uint32_t delta_bidx = bidx / delta;
  const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引
  // if(bidx ==11 && bidy==0){
  //   printf("对应的专家号：expert_id%d\n",expert_id);
  // }
  if(expert_id >=expert_num || expert_id <0) return;
 
  const uint64_t expert_offset = ((uint64_t)size_n) * size_kb  * expert_id; // 这个是对应专家的weight偏移
  
  const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * size_kb ; // 具体偏移到对应专家的weight的某一个小的分块

  const uint32_t output_offset = bidy *  BLOCK_SIZE_N; //输出是NN的layout

  auto g_qweight = qweight + qweight_offset;

  auto g_input = input ;
  
  bhalf_t* g_output;
  
  g_output = output + output_offset;
  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  // constexpr int mfma_k = 16;

  int warp_id_vec = threadIdx.x / 64; //warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  // int lane_id = threadIdx.x & 63; // thread_id
  // int row_id = lane_id % 16;
  // int col_id = lane_id / 16;

  // const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  // const int warp_k_num = BLOCK_SIZE_K / WARP_K;
   
  // int warp_k_id = warp_id % warp_k_num;
  // int warp_n_id = warp_id / warp_k_num;

  const int *sorted_token_ids_offset = sorted_token_ids;
  //存储数据的寄存器
  bf16_vec<WARP_K/4> A_reg[WARP_M/mfma_m][2]; //bf16_vec<4>
  bf16_vec<WARP_K/4> B_reg[BLOCK_SIZE_N/WARP_N][WARP_N/mfma_n][2]; //bf16_vec<4>

  floatx4 C_reg[BLOCK_SIZE_N/WARP_N][(WARP_M/16)*(WARP_N/16)];
  for(int init1=0; init1< (BLOCK_SIZE_N/WARP_N); init1++){
    for(int init2=0; init2< ((WARP_M/16)*(WARP_N/16)); init2++){
      C_reg[init1][init2] = {0.0,0.0,0.0,0.0};
    }
  }
  //kpack==2
  if(WARP_K == 32){
    if(BLOCK_SIZE_N > 16){
      MOE_W16A16_HIP_NT_UP_decode_kpack2_kernel<false,WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
      (g_input, g_qweight, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_kb, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, bidy,size_n,g_output);   
    }
    else{
      bf16_vec<WARP_K/4> B_reg1[WARP_N/mfma_n][2]; //bf16_vec<4>
      MOE_W16A16_HIP_NT_UP_decode_kpack2_kernel2<false,WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
      (g_input, g_qweight, size_m, A_reg, B_reg1, C_reg, warp_id, size_k, size_kb, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, bidy,size_n,g_output);
    }
  }
  else{
    MOE_W16A16_HIP_NT_UP_decode_kernel<false,WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
    (g_input, g_qweight, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_kb, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, bidy,size_n,g_output);  
  }

} 

/****************************************marlin layout launch*************************************************/
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode(const GemmParams3<T>& params) {

    constexpr int kwarp=BLOCK_SIZE_K / WARP_K ;
    constexpr int nwarp=BLOCK_SIZE_N / (WARP_N * 2);
    constexpr int WARP_NUM = kwarp*nwarp ;
    const bool mul_topk_weight =  true;
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;
    
    //规定BLOCKSIZE_N固定位256
    gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
    gridDim.y = DIVIDE(params.size_n , 16 * BLOCK_SIZE_N); // n方向
    gridDim.x = 1; // k方向

    int lds_size =  (kwarp-1)* nwarp * 33 * 16 * 4;
    if(kwarp==0) lds_size==0;
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    
    MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
       mul_topk_weight><<<gridDim, blockDim, lds_size, stream>>>(
      params.ptr_A, 
      params.ptr_B0, 
      params.ptr_C,  
      params.topk_weights,
      params.sorted_token_ids,
      params.expert_ids, 
      params.num_tokens_post_pad_ptr,
      params.size_m,
      params.size_n,
      params.size_k,
      params.size_kofb,
      params.sorted_token_lens,
      params.top_k,
      params.delta,
      params.expert_num
      );

      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
          printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
      }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int splitk, typename T>
void launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode(const GemmParams3<T>& params) {
    constexpr int kwarp=BLOCK_SIZE_K / WARP_K ; // 1
    constexpr int nwarp=BLOCK_SIZE_N / (WARP_N * 2); // 1
    constexpr int WARP_NUM = kwarp*nwarp ; // 1
    const bool mul_topk_weight = false;

    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;
    
    //规定BLOCKSIZE_N固定位256
    gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
    gridDim.y = DIVIDE(params.size_n , 16 * BLOCK_SIZE_N); // n方向
    gridDim.x = splitk; // k方向

    constexpr int lds_size = (kwarp-1)* nwarp * 33 * 16 * 4;
    // constexpr int lds_size = 0;
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    MOE_W16A16_MARLIN_HIP_NN_UP_decode<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
      mul_topk_weight><<<gridDim, blockDim, lds_size, stream>>>(
      params.ptr_A, 
      params.ptr_B0, 
      params.ptr_C,   
      params.topk_weights,
      params.sorted_token_ids,
      params.expert_ids, 
      params.num_tokens_post_pad_ptr,
      params.size_m,
      params.size_n,
      params.size_k,
      params.size_kofb,
      params.sorted_token_lens,
      params.top_k,
      params.delta,
      params.expert_num
      );
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
          printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
      }
}  
/****************************************nt layout launch*************************************************/
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int splitk, typename T>
void launch_MOE_W16A16_HIP_NT_UP_decode(const GemmParams3<T>& params) {
    
    constexpr int kwarp=BLOCK_SIZE_K / WARP_K ; //1
    constexpr int nwarp=BLOCK_SIZE_N / WARP_N  ; //2
    constexpr int WARP_NUM = kwarp*nwarp ; //2
    const bool mul_topk_weight = false;

    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;
    
    //规定BLOCKSIZE_N固定位256
    gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向 16大小分块
    gridDim.y = DIVIDE(params.size_n , BLOCK_SIZE_N); // n方向 16大小分块 数量为48
    gridDim.x = 1; // k方向
    
    int lds_size = kwarp * nwarp * 64 * 4 * 4;
    // if(WARP_K==32) lds_size = kwarp * nwarp * 64 * 8 * 4;
    if(kwarp==1) lds_size = (nwarp * 64 * 4 + 14 )* 2 ;
    // out_smemf[(row_id) * warp_k_num * BLOCK_N  + n_tile * 16 * warp_k_num * BLOCK_N + col_id * 4 + warp_n_id * WARP_N + k_tile * BLOCK_N + i];
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    MOE_W16A16_HIP_NT_UP_decode<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
      mul_topk_weight><<<gridDim, blockDim, lds_size, stream>>>(
      params.ptr_A, 
      params.ptr_B0, 
      params.ptr_C,   
      params.topk_weights,
      params.sorted_token_ids,
      params.expert_ids, 
      params.num_tokens_post_pad_ptr,
      params.size_m,
      params.size_n,
      params.size_k,
      params.size_kofb,
      params.sorted_token_lens,
      params.top_k,
      params.delta,
      params.expert_num
      );
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
          printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
      }
} 


template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_MOE_W16A16_HIP_NT_DOWN_decode(const GemmParams3<T>& params) {

    constexpr int kwarp=BLOCK_SIZE_K / WARP_K ; //1
    constexpr int nwarp=BLOCK_SIZE_N / WARP_N; //1
    constexpr int WARP_NUM = kwarp*nwarp ; //1
    const bool mul_topk_weight =  true;
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;
    

    gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); //2048 /16  
    gridDim.y = DIVIDE(params.size_n , BLOCK_SIZE_N); // 2048 /16 
    gridDim.x = 1; 

    int lds_size = kwarp * nwarp * 64 * 4 * 4;
    // if(WARP_K==32) lds_size = kwarp * nwarp * 64 * 8 * 4;
    if(kwarp==1) lds_size = (nwarp * 64 * 4 + 14 )* 2;
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    
    MOE_W16A16_HIP_NT_DOWN_decode<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
      mul_topk_weight><<<gridDim, blockDim, lds_size, stream>>>(
      params.ptr_A, 
      params.ptr_B0, 
      params.ptr_C, 
      params.topk_weights,
      params.sorted_token_ids,
      params.expert_ids, 
      params.num_tokens_post_pad_ptr,
      params.size_m,
      params.size_n,
      params.size_k,
      params.size_kofb,
      params.sorted_token_lens,
      params.top_k,
      params.delta, 
      params.expert_num
      );
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
          printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
      }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_MOE_W16A16_HIP_NT_UP_prefill(const GemmParams3<T>& params) {
    
    const int STAGES = BLOCK_SIZE_K/WARP_K; //1
    constexpr int mwarp=BLOCK_SIZE_M / WARP_M ; //1
    constexpr int nwarp=BLOCK_SIZE_N / WARP_N  ; //1
    constexpr int WARP_NUM = mwarp*nwarp ; //1
    const bool mul_topk_weight = false;

    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;
    constexpr int bytes_per_element = 2;
    //规定BLOCKSIZE_N固定位256
    gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向 16大小分块
    gridDim.y = DIVIDE(params.size_n , BLOCK_SIZE_N); // n方向 16大小分块 数量为48
    gridDim.x = 1; // k方向
    
    // int lds_size = std::max(kwarp * nwarp * 64 * 4 * 4,BLOCK_SIZE_M*BLOCK_SIZE_K + BLOCK_SIZE_N*BLOCK_SIZE_K);
    // if(kwarp==1) lds_size = std::max(BLOCK_SIZE_M*BLOCK_SIZE_K + BLOCK_SIZE_N*BLOCK_SIZE_K,(nwarp * 64 * 4 + 14 )* 2) ;
    const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + BLOCK_SIZE_N * WARP_K)  * bytes_per_element, BLOCK_SIZE_M * BLOCK_SIZE_N * 2 * bytes_per_element);
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    MOE_W16A16_HIP_NT_UP_prefill<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
      mul_topk_weight><<<gridDim, blockDim, lds_size, stream>>>(
      params.ptr_A, 
      params.ptr_B0, 
      params.ptr_C,   
      params.topk_weights,
      params.sorted_token_ids,
      params.expert_ids, 
      params.num_tokens_post_pad_ptr,
      params.size_m,
      params.size_n,
      params.size_k,
      params.size_kofb,
      params.sorted_token_lens,
      params.top_k,
      params.delta,
      params.expert_num
      );
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
          printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
      }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill(const GemmParams3<T>& params) {
    
    constexpr int STAGES = BLOCK_SIZE_K/WARP_K; //1
    constexpr int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N);
    const bool mul_topk_weight = false;

    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;
    constexpr int bytes_per_element = 2;
    const int real_size_n = (params.size_n * params.size_kofb) / params.size_k;  
    //规定BLOCKSIZE_N固定位256
    gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向 16大小分块
    gridDim.y = DIVIDE(real_size_n , BLOCK_SIZE_N); // n方向 16大小分块 数量为48
    gridDim.x = 1; // k方向

    const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + BLOCK_SIZE_N * WARP_K) * bytes_per_element, BLOCK_SIZE_M * BLOCK_SIZE_N * 2 * bytes_per_element);
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    MOE_W16A16_MARLIN_HIP_NT_UP_prefill<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
      mul_topk_weight><<<gridDim, blockDim, lds_size, stream>>>(
      params.ptr_A, 
      params.ptr_B0, 
      params.ptr_C,   
      params.topk_weights,
      params.sorted_token_ids,
      params.expert_ids, 
      params.num_tokens_post_pad_ptr,
      params.size_m,
      params.size_n,
      params.size_k,
      params.size_kofb,
      params.sorted_token_lens,
      params.top_k,
      params.delta,
      params.expert_num
      );
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
          printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
      }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_MOE_W16A16_HIP_NT_DOWN_prefill(const GemmParams3<T>& params) {
    
    const int STAGES = BLOCK_SIZE_K/WARP_K;
    constexpr int mwarp=BLOCK_SIZE_M / WARP_M ; //1
    constexpr int nwarp=BLOCK_SIZE_N / WARP_N  ; //4
    constexpr int WARP_NUM = mwarp*nwarp ; //4
    const bool mul_topk_weight = true;

    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;
    constexpr int bytes_per_element = 2;

    gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向 16大小分块
    gridDim.y = DIVIDE(params.size_n , BLOCK_SIZE_N); // n方向 16大小分块 数量为48
    gridDim.x = 1; // k方向
    
    const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + BLOCK_SIZE_N * WARP_K) * bytes_per_element, BLOCK_SIZE_M * BLOCK_SIZE_N * 2 * bytes_per_element);
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    MOE_W16A16_HIP_NT_DOWN_prefill<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
      mul_topk_weight><<<gridDim, blockDim, lds_size, stream>>>(
      params.ptr_A, 
      params.ptr_B0, 
      params.ptr_C,   
      params.topk_weights,
      params.sorted_token_ids,
      params.expert_ids, 
      params.num_tokens_post_pad_ptr,
      params.size_m,
      params.size_n,
      params.size_k,
      params.size_kofb,
      params.sorted_token_lens,
      params.top_k,
      params.delta
      );
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
          printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
      }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill(const GemmParams3<T>& params) {
    
    constexpr int STAGES = BLOCK_SIZE_K/WARP_K;
    constexpr int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N);
    const bool mul_topk_weight = true;

    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;
    constexpr int bytes_per_element = 2;
    const int real_size_n = (params.size_n * params.size_kofb) / params.size_k;  

    gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向 16大小分块
    gridDim.y = DIVIDE(real_size_n , BLOCK_SIZE_N); // n方向 16大小分块 数量为48
    gridDim.x = 1; // k方向
    
    const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + BLOCK_SIZE_N * WARP_K) * bytes_per_element, BLOCK_SIZE_M * BLOCK_SIZE_N * 2 * bytes_per_element);
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
      mul_topk_weight><<<gridDim, blockDim, lds_size, stream>>>(
      params.ptr_A, 
      params.ptr_B0, 
      params.ptr_C,   
      params.topk_weights,
      params.sorted_token_ids,
      params.expert_ids, 
      params.num_tokens_post_pad_ptr,
      params.size_m,
      params.size_n,
      params.size_k,
      params.size_kofb,
      params.sorted_token_lens,
      params.top_k,
      params.delta
      );
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
          printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
      }
}

/****************************************pybind func entry*************************************************/
torch::Tensor moe_gemm_w16a16(torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  std::optional<torch::Tensor> topk_weights,
  torch::Tensor sorted_token_ids, 
  torch::Tensor expert_ids,
  torch::Tensor num_tokens_post_pad, 
  int64_t top_k,
  int mode,
  int delta
  ) {
    const int size_m = input.size(0); //16
    const int EXPERTS = b_qweight.size(0);//128
    const int size_k = input.size(1);//2048
    // marlin nn layout
    int size_n = b_qweight.size(2);  //12288
    int size_kb = b_qweight.size(1); //128

    bool first_stage = true;
    torch::Tensor output_alias = output.alias();
    const float* topk_weights_ptr; 
    if (topk_weights.has_value()){
      topk_weights_ptr = (const float*)topk_weights.value().data_ptr();
      first_stage = false;
    }
    int num_pad = 0;
    if (input.scalar_type() == at::ScalarType::BFloat16){
      if(mode<400){      
        if(first_stage){
          int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id        
          GemmParams3<bhalf_t> params_in3(
          (const bhalf_t*)input.data_ptr(), 
          (const bhalf_t*)b_qweight.data_ptr(), 
          (bhalf_t*)output_alias.data_ptr(),
          topk_weights_ptr,
          sorted_token_ids.data_ptr<int32_t>(),
          expert_ids.data_ptr<int32_t>(), 
          num_pad, //num_tokens_post_pad[0].item<int>(), //这里获取值 会造成device->host的拷贝和一部分空泡
          num_tokens_post_pad.data_ptr<int32_t>(),
          size_m,
          size_n,
          size_k,
          size_kb,
          EM,
          top_k,
          delta,
          EXPERTS
        );
        
        if(mode < 300){
          auto it = kernel_maps_gemm1_prefill_marlin_w16a16.find(mode);
            if (it != kernel_maps_gemm1_prefill_marlin_w16a16.end()) {
              it->second(params_in3);
            } else {
              printf("kernel_maps_gemm1_prefill_marlin_w16a16: No matching kernel configuration found, using default settings \n");
              launch_MOE_W16A16_MARLIN_HIP_NT_UP_prefill<16, 64, 128, 16, 16, 64, bhalf_t>(params_in3);
            }
        }else{ //decode
          auto it = kernel_maps_gemm1_decode_marlin_w16a16.find(mode);
            if (it != kernel_maps_gemm1_decode_marlin_w16a16.end()) {
              it->second(params_in3);
            } else {
              printf("No matching kernel configuration found, using default settings \n");
              launch_MOE_W16A16_MARLIN_HIP_NN_UP_decode<16, 64, 512, 16, 16, 64,1, bhalf_t>(params_in3);
            }
        }      
      
      }
      else{ //gemm2 
        int64_t EM = sorted_token_ids.size(0); 
        GemmParams3<bhalf_t> params_in3(
          (const bhalf_t*)input.data_ptr(), 
          (const bhalf_t*)b_qweight.data_ptr(), 
          (bhalf_t*)output_alias.data_ptr(), 
          topk_weights_ptr,
          sorted_token_ids.data_ptr<int32_t>(),
          expert_ids.data_ptr<int32_t>(), 
          num_pad, //num_tokens_post_pad[0].item<int>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          size_m,
          size_n,
          size_k,
          size_kb,
          EM,
          top_k,
          delta,
          EXPERTS
        );
        
        if(mode < 300){
          auto it = kernel_maps_gemm2_prefill_marlin_w16a16.find(mode);
            if (it != kernel_maps_gemm2_prefill_marlin_w16a16.end()) {
              it->second(params_in3);
            } else {
              printf("kernel_maps_gemm2_prefill_marlin_w16a16: No matching kernel configuration found, using default settings \n");
              launch_MOE_W16A16_MARLIN_HIP_NT_DOWN_prefill<16, 64, 128, 16, 16, 64, bhalf_t>(params_in3);
            }
        }else{
          auto it = kernel_maps_gemm2_decode_marlin_w16a16.find(mode);
          if (it != kernel_maps_gemm2_decode_marlin_w16a16.end()) {
              it->second(params_in3);
          } else {
              printf("No matching kernel configuration found, using default settings \n");
              launch_MOE_W16A16_MARLIN_HIP_NN_DOWN_decode<16, 256, 32, 16, 16, 32, bhalf_t>(params_in3);
          }
        }
      }
    }
    else if(399<mode && mode<500){
      //NT layout decode
      int size_n = b_qweight.size(1);  //2048
      int size_kb = b_qweight.size(2); //384
      if(first_stage){
        int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id        
        GemmParams3<bhalf_t> params_in3(
          (const bhalf_t*)input.data_ptr(), 
          (const bhalf_t*)b_qweight.data_ptr(), 
          (bhalf_t*)output_alias.data_ptr(),
          topk_weights_ptr,
          sorted_token_ids.data_ptr<int32_t>(),
          expert_ids.data_ptr<int32_t>(), 
          num_pad, 
          num_tokens_post_pad.data_ptr<int32_t>(),
          size_m,
          size_n,
          size_k,
          size_kb,
          EM,
          top_k,
          delta,
          EXPERTS
        );

        if(size_m > 16384){
          printf("developing \n");
          return output;
        }else{ //decode
          auto it = kernel_maps_gemm1_decode_w16a16.find(mode);
            if (it != kernel_maps_gemm1_decode_w16a16.end()) {
              it->second(params_in3);
            } else {
              printf("No matching kernel configuration found, using default settings \n");
              launch_MOE_W16A16_HIP_NT_UP_decode<16, 16, 128, 16, 16, 16,1, bhalf_t>(params_in3);
            }
        }      
      }
      else{ //gemm2 
        int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id
        GemmParams3<bhalf_t> params_in3(
          (const bhalf_t*)input.data_ptr(), 
          (const bhalf_t*)b_qweight.data_ptr(), 
          (bhalf_t*)output_alias.data_ptr(), 
          topk_weights_ptr,
          sorted_token_ids.data_ptr<int32_t>(),
          expert_ids.data_ptr<int32_t>(), 
          num_pad, //num_tokens_post_pad[0].item<int>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          size_m,
          size_n,
          size_k,
          size_kb,
          EM,
          top_k,
          delta,
          EXPERTS
        );
        if(size_m / 8 > 16384){
          printf("developing \n");
        }
        else{
          auto it = kernel_maps_gemm2_decode_w16a16.find(mode);
          if (it != kernel_maps_gemm2_decode_w16a16.end()) {
              it->second(params_in3);
          } else {
              printf("No matching kernel configuration found, using default settings \n");
              launch_MOE_W16A16_HIP_NT_DOWN_decode<16, 16, 16, 16, 16, 16, bhalf_t>(params_in3);
          }
        }
      }      
    }
    else if(499<mode && mode<600){
      //NT layout prefill
      int size_n = b_qweight.size(1);  //768
      int size_kb = b_qweight.size(2); //2048
      if(first_stage){
        int64_t EM = sorted_token_ids.size(0);  
        GemmParams3<bhalf_t> params_in3(
          (const bhalf_t*)input.data_ptr(), 
          (const bhalf_t*)b_qweight.data_ptr(), 
          (bhalf_t*)output_alias.data_ptr(),
          topk_weights_ptr,
          sorted_token_ids.data_ptr<int32_t>(),
          expert_ids.data_ptr<int32_t>(), 
          num_pad, 
          num_tokens_post_pad.data_ptr<int32_t>(),
          size_m,
          size_n,
          size_k,
          size_kb,
          EM,
          top_k,
          delta,
          EXPERTS
        );

        if(size_m > 16384){
          printf("developing \n");
          return output;
        }else{ //prefill
          
          auto it = kernel_maps_gemm1_prefill_w16a16.find(mode);
            if (it != kernel_maps_gemm1_prefill_w16a16.end()) {
              it->second(params_in3);
            } else {
              printf("kernel_maps_gemm1_prefill_w16a16: No matching kernel configuration found, using default settings \n");
              launch_MOE_W16A16_HIP_NT_UP_prefill<16, 64, 128, 16, 16, 64, bhalf_t>(params_in3);
            }
        }      
      }
      else{ //gemm2 
        int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id
        GemmParams3<bhalf_t> params_in3(
          (const bhalf_t*)input.data_ptr(), 
          (const bhalf_t*)b_qweight.data_ptr(), 
          (bhalf_t*)output_alias.data_ptr(), 
          topk_weights_ptr,
          sorted_token_ids.data_ptr<int32_t>(),
          expert_ids.data_ptr<int32_t>(), 
          num_pad, //num_tokens_post_pad[0].item<int>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          size_m,
          size_n,
          size_k,
          size_kb,
          EM,
          top_k,
          delta,
          EXPERTS
        );
        if(size_m / 8 > 16384){
          printf("developing \n");
        }
        else{
          auto it = kernel_maps_gemm2_prefill_w16a16.find(mode);
          if (it != kernel_maps_gemm2_prefill_w16a16.end()) {
              it->second(params_in3);
          } else {
              printf("kernel_maps_gemm2_prefill_w16a16: No matching kernel configuration found, using default settings \n");
              launch_MOE_W16A16_HIP_NT_DOWN_prefill<16, 64, 128, 16, 16, 64, bhalf_t>(params_in3);
          }
        }
      }    
    }
    }
    else {
      TORCH_CHECK(false, "moe_gemm_w16a16 only supports bf16 now");
    }
    return output;
}
}
}