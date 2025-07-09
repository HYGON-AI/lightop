//delete when compile whl
#include "moe_w8a8_utils.h"

namespace at {
namespace native {

// GemmParams结构体定义
template <typename T>
struct GemmParams {
    GemmParams(const T* ptr_A,
               const T* ptr_B0,
               bhalf_t* ptr_C,
               float* ptr_A_scale,
               float* ptr_B_scale,
               const float* topk_weights,
               const int32_t* sorted_token_ids,
               const int32_t* expert_ids,
               const int32_t num_tokens_post_pad,
               const int32_t* num_tokens_post_pad_ptr,
               uint32_t size_m,
               uint32_t size_n, 
               uint32_t size_k, 
               uint32_t stride_asm,
               uint32_t stride_ask,
               uint32_t stride_bse,
               uint32_t stride_bsn, 
               uint32_t stride_bsk,
               uint32_t sorted_token_lens,
               uint32_t top_k)
        : ptr_A(ptr_A),
          ptr_B0(ptr_B0),
          ptr_C(ptr_C),
          ptr_A_scale(ptr_A_scale),
          ptr_B_scale(ptr_B_scale),
          topk_weights(topk_weights),
          sorted_token_ids(sorted_token_ids),
          expert_ids(expert_ids),
          num_tokens_post_pad(num_tokens_post_pad),
          num_tokens_post_pad_ptr(num_tokens_post_pad_ptr),
          size_m(size_m),
          size_n(size_n),
          size_k(size_k),
          stride_asm(stride_asm),
          stride_ask(stride_ask),
          stride_bse(stride_bse),
          stride_bsn(stride_bsn),
          stride_bsk(stride_bsk),
          sorted_token_lens(sorted_token_lens),
          top_k(top_k)
         {}

    const T* ptr_A;                    // input
    const T* ptr_B0;                   // weight
    bhalf_t* ptr_C;                   // output
    float* ptr_A_scale;                // input scale
    float* ptr_B_scale;                // weight scale
    const float* topk_weights;         // topk weights
    const int32_t* sorted_token_ids;   // sorted token ids
    const int32_t* expert_ids;         // expert ids
    const int32_t num_tokens_post_pad;// num tokens after padding
    const int32_t* num_tokens_post_pad_ptr;// num tokens after padding
    uint32_t size_m;                   // input size m
    uint32_t size_n;                   // output size n
    uint32_t size_k;                   // input size k
    uint32_t stride_asm;               // input scale stride m
    uint32_t stride_ask;               // input scale stride k
    uint32_t stride_bse;               // weight scale stride expert
    uint32_t stride_bsn;               // weight scale stride n
    uint32_t stride_bsk;               // weight scale stride k
    uint32_t sorted_token_lens;        // sorted token length
    uint32_t top_k;                    // top k
};

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
__global__ void __launch_bounds__(1024) moe_w8a8_gemm_first_stage_kernel(
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
  uint32_t top_k 
) {
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k 

  // int bidx = blockIdx.x; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  // int bidy = blockIdx.y; // pid_n
  // int bidz = blockIdx.z; // pid_k 
  // GetBLockIdx(bidx + bidy * gridDim.x, gridDim.x, gridDim.y, 0, 4, bidx, bidy);

  if (bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  constexpr int STAGES = BLOCK_SIZE_K/WARP_K; 
  const uint32_t input_offset = bidz * BLOCK_SIZE_K; //输入k方向分块的位置
  const int32_t expert_id = expert_ids[bidx]; // 专家的索引
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移
  const uint64_t qweight_offset = expert_offset + bidy * size_k * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N; // 计算之后是mxn,这应该是计算输出n方向的位置

  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块
  
  auto g_input = input;
  auto g_input_scale = input_scale; // 配置全局显存信息
  bhalf_t* g_output;

  g_output = output + output_offset;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

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

  union_vec<Element, 8> A_reg[WARP_M / mfma_m * STAGES][WARP_K/mfma_k]; 
  union_vec<Element, 8> B_reg[WARP_N / mfma_n][WARP_K/mfma_k];

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset; // 配置weight的scale信息 todo 这里n_loop=0没有计算偏移
  intx4 C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
  float *a_scale_ptr_arr[WARP_M / mfma_m];
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id*WARP_N;

  #pragma unroll
  for(int idx = 0; idx < WARP_M / mfma_m; idx++){
      int sorted_token_offset = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + warp_m_id * WARP_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
      a_scale_ptr_arr[idx] = input_scale + std::min(sorted_token_offset / top_k,  size_m - 1) * stride_asm; //计算M方向的偏移
  }

  { 
    gemm_nt_first_stage<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>
    (g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
  }
  //save output to lds
  for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
    float *a_scale = a_scale_ptr_arr[min_tile_m];
    for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
      #pragma unroll
      for(int reg_id = 0; reg_id < 4; reg_id++){
        float *b_scale = b_scale_ptr + min_tile_n * mfma_n + col_id + reg_id * 4;
        float value = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n][reg_id] * a_scale[0] * b_scale[0];
        int index = warp_m_id * WARP_M * BLOCK_SIZE_N + min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15 ) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + warp_m_id * WARP_M+ (lane_id % 16) )/2 * 2/*padding*/;
        output_lds[index] = __float2bfloat16(value);
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
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K, 
  uint16_t GROUP_N,
  uint16_t GROUP_K,
  bool mul_topk_weight> // true
__global__ void __launch_bounds__(1024) moe_w8a8_gemm_second_stage_kernel(
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
  uint32_t top_k 
) {
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.y; // pid_n
  const int bidz = blockIdx.x; // pid_k 

  // int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  // int bidy = blockIdx.y; // pid_n
  // int bidz = blockIdx.x; // pid_k 
  // GetBLockIdx(bidx + bidy * gridDim.x, gridDim.x, gridDim.y, 0, 8, bidx, bidy);

  if (bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  const int STAGES = BLOCK_SIZE_K/WARP_K; // 采用两stage
  const uint32_t input_offset = bidz * BLOCK_SIZE_K; //输入k方向分块的位置
  const int32_t expert_id = expert_ids[bidx]; // 专家的索引
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id; // 这个是对应专家的weight偏移
  const uint64_t qweight_offset = expert_offset + bidy * size_k * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N; // 计算之后是mxn,这应该是计算输出n方向的位置

  const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块
  
  auto g_input = input;
  auto g_input_scale = input_scale; // 配置全局显存信息
  bhalf_t* g_output;

  g_output = output + output_offset;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;

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

  union_vec<Element, 8> A_reg[WARP_M / mfma_m * STAGES][WARP_K/mfma_k]; //
  union_vec<Element, 8> B_reg[WARP_N / mfma_n][WARP_K/mfma_k];

  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset; // 配置weight的scale信息 todo 这里n_loop=0没有计算偏移
  intx4 C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
  float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id*WARP_N;
  
  gemm_nt_first_stage<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>
  (g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
  //save output to lds
  #pragma unroll
  for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
    const int32_t token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + warp_m_id * WARP_M + min_tile_m * mfma_m + (lane_id % 16), int(sorted_token_lens - 1))];
    const int32_t token_index_safe = std::min(token_index, size_m * top_k-1);
    float *a_scale = input_scale + token_index_safe * stride_asm; //计算M方向的偏移
    float weight_value = topk_weights[token_index_safe];
    #pragma unroll
    for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
      #pragma unroll
      for(int reg_id = 0; reg_id < 4; reg_id++){
        float *b_scale = b_scale_ptr + min_tile_n * mfma_n + col_id + reg_id * 4;
        float value = C_reg[0][min_tile_m * (WARP_N/mfma_n) + min_tile_n][reg_id] * weight_value * a_scale[0] * b_scale[0];
        int index = warp_m_id * WARP_M * BLOCK_SIZE_N + min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15 ) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + warp_m_id * WARP_M + (lane_id % 16))/2 * 2/*padding*/;// + (lane_id & 15 )/2 * 2/*padding*/
        output_lds[index] = __float2bfloat16(value);
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

// launch_gemm1
// 在GemmParams结构体后添加
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_moe_gemm_first_stage(const GemmParams<T>& params) {
    const int STAGES = BLOCK_SIZE_K/WARP_K;
    const int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N);
    const bool mul_topk_weight = false;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;

    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;
    
    if(params.size_m > 128) //减少小shape的 device->host拷贝
      gridDim.z = std::min(DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M), DIVIDE(params.num_tokens_post_pad, BLOCK_SIZE_M)); // m方向
    else
      gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向

    gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
    gridDim.x = 1; // k方向

    // block swizzle 效果提升不是很明显 需要tuning验证
    // gridDim.x = min(DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M), DIVIDE(params.num_tokens_post_pad, BLOCK_SIZE_M)); // m方向
    // //gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
    // gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
    // gridDim.z = 1; // k方向

    const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + BLOCK_SIZE_N * WARP_K), BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
    const int shared_mem_size = lds_size;
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    
    moe_w8a8_gemm_first_stage_kernel<char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
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
      params.top_k);

      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
          printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
      }
}


// launch_gemm1
// 在GemmParams结构体后添加
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_moe_gemm_second_stage(const GemmParams<T>& params) {
    const int STAGES = BLOCK_SIZE_K/WARP_K;
    const int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N);
    const bool mul_topk_weight = true;
    // constexpr int GROUP_N = 128;
    // constexpr int GROUP_K = 128;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;

    if(params.size_m > 128) //减少小shape的 device->host拷贝
      gridDim.z = std::min(DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M), DIVIDE(params.num_tokens_post_pad, BLOCK_SIZE_M)); // m方向
    else
      gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
    gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
    gridDim.x = 1; // k方向

    // block swizzle 效果提升不是很明显 需要tuning验证
    // gridDim.x = min(DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M), DIVIDE(params.num_tokens_post_pad, BLOCK_SIZE_M)); // m方向
    // //gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
    // gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
    // gridDim.z = 1; // k方向

    const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + BLOCK_SIZE_N * WARP_K), BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
    const int shared_mem_size = lds_size;// + BLOCK_SIZE_M * 4 * 2 + 32; // 额外分配sort_token_ids的空间
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    
    moe_w8a8_gemm_second_stage_kernel<char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
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
      params.top_k);

      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
          printf("CUDA error in gemm2: %s\n", cudaGetErrorString(err));
      }
}


// 模板抽象
// BLOCK_MNK: [16, 128, 128]
// WARP_MNK: [16, 32, 64]
// MMA_MNK: [16, 16, 32]
// BLOCK_MNK / WARP_MNK = [1, 4, 2]  代表warp在MN方向的排布 block_k/warp_k代表stage
// WARP_MNK / MMA_MNK = [1, 2, 2]    代表warp在MNK方向重复计算的次数 会分配额外的寄存器
torch::Tensor moe_gemm_w8a8(torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  std::optional<torch::Tensor> topk_weights,
  torch::Tensor sorted_token_ids, 
  torch::Tensor expert_ids,
  torch::Tensor num_tokens_post_pad, 
  int64_t top_k,
  int mode
  ) {
    const int size_m = input.size(0); 
    const int size_n = b_qweight.size(1); 
    const int size_k = input.size(1);
    const int stride_asm = a_scale.stride(0);
    const int stride_ask = a_scale.stride(1);
    const int stride_bse = b_scale.stride(0);
    const int stride_bsn = b_scale.stride(1); 
    const int stride_bsk = b_scale.stride(2);
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    bool first_stage = true;
    torch::Tensor output_alias = output.alias();

    const float* topk_weights_ptr; // 第一阶段这里为null
    if (topk_weights.has_value()){
      topk_weights_ptr = (const float*)topk_weights.value().data_ptr();
      first_stage = false;
    }

    int num_pad = 0;
    if (size_m > 128)
      num_pad = num_tokens_post_pad[0].item<int>();

    if (input.scalar_type() == at::ScalarType::Char){
      if(first_stage){
     
        int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id
        
        // 使用int8类型处理
        GemmParams<char> params_in(
          (const char*)input.data_ptr<int8_t>(), 
          (const char*)b_qweight.data_ptr<int8_t>(), 
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
          top_k
        );

        // 建立映射
        // 创建kernel映射
        using KernelFunc = std::function<void(const GemmParams<char>&)>;
        
        static std::unordered_map<int, KernelFunc> kernel_maps = {
            ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
            // 4warp BLOCKN 64 
            {0, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<16, 64, 128, 16, 16, 64, char>(p); }},
            {1, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 64, 128, 32, 16, 64, char>(p); }},
            {2, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 64, 128, 64, 16, 64, char>(p); }},
            // 4warp BLOCKN 128 
            {3, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<16, 128, 128, 16, 32, 64, char>(p); }},
            {4, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 128, 128, 32, 32, 64, char>(p); }},
            //{5, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 128, 128, 64, 32, 64, char>(p); }}, // 寄存器溢出 32
            // 4warp BLOCKN 256 
            {6, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<16, 256, 128, 16, 64, 64, char>(p); }},
            //{7, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 256, 128, 32, 64, 64, char>(p); }}, // 寄存器溢出 60
            //{8, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 256, 128, 64, 64, 64, char>(p); }}, //寄存器溢出 392
            // 8warp BLOCKN 128 
            {9, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 128, 128, 32, 16, 64, char>(p); }},
            {10, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 128, 128, 64, 16, 64, char>(p); }},
            // 8warp BLOCKN 256 
            {11, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 256, 128, 32, 32, 64, char>(p); }},
            //{12, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 256, 128, 64, 32, 64, char>(p); }}, //寄存器溢出28
            // 16warp
            {13, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 256, 128, 64, 16, 64, char>(p); }},
            ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
            // 4 warp:  2,2,1
            // thread: 2,2,1    1,2,1    2,1,1  1,1,1
            {14, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 32, 128, 16, 16, 64, char>(p); }},
            {15, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 64, 128, 16, 32, 64, char>(p); }},
            {16, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 32, 128, 32, 16, 64, char>(p); }},
            {17, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 64, 128, 32, 32, 64, char>(p); }},
            // 8 warp:  2,4,1
            // thread: 2,2,1    1,2,1    2,1,1  1,1,1
            {18, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 64, 128, 16, 16, 64, char>(p); }},
            {19, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 64, 128, 32, 16, 64, char>(p); }},
            {20, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 128, 128, 16, 32, 64, char>(p); }},
            {21, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 128, 128, 32, 32, 64, char>(p); }},
            // 8 warp:  4,2,1
            // thread: 2,2,1    1,2,1    2,1,1  1,1,1
            {22, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 32, 128, 16, 16, 64, char>(p); }},
            {23, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 64, 128, 16, 32, 64, char>(p); }},
            {24, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<128, 32, 128, 32, 16, 64, char>(p); }},
            {25, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<128, 64, 128, 32, 32, 64, char>(p); }},
            /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
            // 1 warp
            {26, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<16, 16, 128, 16, 16, 64, char>(p); }},
            {27, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 16, 128, 16, 16, 64, char>(p); }},
            {28, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<16, 32, 128, 16, 32, 64, char>(p); }},
            {29, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<16, 64, 128, 16, 64, 64, char>(p); }},
            {30, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<16, 64, 128, 16, 64, 64, char>(p); }},
            // 2 warp
            {31, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 16, 128, 16, 16, 64, char>(p); }},
            {32, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 32, 128, 16, 32, 64, char>(p); }},
            {33, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 64, 128, 16, 64, 64, char>(p); }},
            //{34, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 128, 128, 16, 128, 64, char>(p); }}, //寄存器溢出 156
            // 4 warp
            {35, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 16, 128, 16, 16, 64, char>(p); }},
            {36, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 32, 128, 16, 32, 64, char>(p); }},
            // {37, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 64, 128, 16, 64, 64, char>(p); }}, // 寄存器溢出 120
            // {38, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 128, 128, 16, 128, 64, char>(p); }}, // 寄存器溢出 120
            
            ////////////////////////////////////////////////////N_warp-stages=4//////////////////////////////////////////////
            // 4warp-4stages-warpk
            {39, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 64, 128, 32, 16, 32, char>(p); }},
            {40, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 64, 128, 64, 16, 32, char>(p); }},

            {41, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 128, 128, 32, 32, 32, char>(p); }},
            {42, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 128, 128, 64, 32, 32, char>(p); }},

            {43, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 256, 128, 32, 64, 32, char>(p); }},
            {44, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 256, 128, 64, 64, 32, char>(p); }},
            
            // 8warp-4stages-warpk
            {45, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 128, 128, 64, 16, 32, char>(p); }},
            // 8warp
            {46, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 256, 128, 64, 32, 32, char>(p); }},

            ////////////////////////////////////////////////////MN_WARP-stages=4/////////////////////////////////////////////
            // 4 warp:  2,2,1
            // thread: 2,2,1    1,2,1    2,1,1  1,1,1
            {47, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 64, 128, 32, 32, 32, char>(p); }},
            // 8 warp:  2,4,1
            // thread: 2,2,1    1,2,1    2,1,1  1,1,1
            {48, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 128, 128, 32, 32, 32, char>(p); }},
            // 8 warp:  4,2,1
            // thread: 2,2,1    1,2,1    2,1,1  1,1,1
            {49, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<128, 64, 128, 32, 32, 32, char>(p); }},

            ///////////////////////////////////////////////////////N_warp-256////////////////////////////////////////////////
            // 4warp BLOCKN 64 
            {50, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<16, 64, 256, 16, 16, 64, char>(p); }},
            {51, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 64, 256, 32, 16, 64, char>(p); }},
            {52, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 64, 256, 64, 16, 64, char>(p); }},
            // 4warp BLOCKN 128 
            {53, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<16, 128, 256, 16, 32, 64, char>(p); }},
            {54, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 128, 256, 32, 32, 64, char>(p); }},
            //{55, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 128, 128, 64, 32, 64, char>(p); }}, // 寄存器溢出 32
            // 4warp BLOCKN 256 
            {56, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<16, 256, 256, 16, 64, 64, char>(p); }},
            //{57, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 256, 128, 32, 64, 64, char>(p); }}, // 寄存器溢出 60
            //{58, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 256, 128, 64, 64, 64, char>(p); }}, //寄存器溢出 392
            // 8warp BLOCKN 128 
            {59, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 128, 256, 32, 16, 64, char>(p); }},
            {60, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 128, 256, 64, 16, 64, char>(p); }},
            // 8warp BLOCKN 256 
            {61, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<32, 256, 256, 32, 32, 64, char>(p); }},
            //{62, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 256, 128, 64, 32, 64, char>(p); }}, //寄存器溢出28
            // 16warp
            {63, [](const GemmParams<char>& p) { launch_moe_gemm_first_stage<64, 256, 256, 64, 16, 64, char>(p); }},
        };

        auto it = kernel_maps.find(mode);
        if (it != kernel_maps.end()) {
            it->second(params_in);
        } else {
            printf("No matching kernel configuration found, using default settings \n");
            launch_moe_gemm_first_stage<16, 64, 128, 16, 16, 64>(params_in);
        }
        
      }else{ //gemm2
        int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id
        
        // 使用int8类型处理
        GemmParams<char> params_in(
          (const char*)input.data_ptr<int8_t>(), 
          (const char*)b_qweight.data_ptr<int8_t>(), 
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
          top_k
        );

        // 建立映射
        // 创建kernel映射
        using KernelFunc = std::function<void(const GemmParams<char>&)>;
        
        static std::unordered_map<int, KernelFunc> kernel_maps = {
            ///////////////////////////////////////////////////////N_warp////////////////////////////////////////////////
            // 4warp BLOCKN 64 
            {0, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<16, 64, 128, 16, 16, 64, char>(p); }},
            {1, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 64, 128, 32, 16, 64, char>(p); }},
            {2, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 64, 128, 64, 16, 64, char>(p); }},
            // 4warp BLOCKN 128 
            {3, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<16, 128, 128, 16, 32, 64, char>(p); }},
            {4, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 128, 128, 32, 32, 64, char>(p); }},
            //{5, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 128, 128, 64, 32, 64, char>(p); }}, // 寄存器溢出 32
            // 4warp BLOCKN 256 
            {6, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<16, 256, 128, 16, 64, 64, char>(p); }},
            //{7, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 256, 128, 32, 64, 64, char>(p); }}, // 寄存器溢出 60
            //{8, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 256, 128, 64, 64, 64, char>(p); }}, //寄存器溢出 392
            // 8warp BLOCKN 128 
            {9, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 128, 128, 32, 16, 64, char>(p); }},
            {10, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 128, 128, 64, 16, 64, char>(p); }},
            // 8warp BLOCKN 256 
            {11, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 256, 128, 32, 32, 64, char>(p); }},
            //{12, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 256, 128, 64, 32, 64, char>(p); }}, //寄存器溢出28
            // 16warp
            {13, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 256, 128, 64, 16, 64, char>(p); }},
            ///////////////////////////////////////////////////////MN_WARP////////////////////////////////////////////////
            // 4 warp:  2,2,1
            // thread: 2,2,1    1,2,1    2,1,1  1,1,1
            {14, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 32, 128, 16, 16, 64, char>(p); }},
            {15, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 64, 128, 16, 32, 64, char>(p); }},
            {16, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 32, 128, 32, 16, 64, char>(p); }},
            {17, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 64, 128, 32, 32, 64, char>(p); }},
            // 8 warp:  2,4,1
            // thread: 2,2,1    1,2,1    2,1,1  1,1,1
            {18, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 64, 128, 16, 16, 64, char>(p); }},
            {19, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 64, 128, 32, 16, 64, char>(p); }},
            {20, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 128, 128, 16, 32, 64, char>(p); }},
            {21, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 128, 128, 32, 32, 64, char>(p); }},
            // 8 warp:  4,2,1
            // thread: 2,2,1    1,2,1    2,1,1  1,1,1
            {22, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 32, 128, 16, 16, 64, char>(p); }},
            {23, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 64, 128, 16, 32, 64, char>(p); }},
            {24, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<128, 32, 128, 32, 16, 64, char>(p); }},
            {25, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<128, 64, 128, 32, 32, 64, char>(p); }},
            /////////////////////////////////////////////////////////M_WARP////////////////////////////////////////////////////
            // 1 warp
            {26, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<16, 16, 128, 16, 16, 64, char>(p); }},
            {27, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 16, 128, 16, 16, 64, char>(p); }},
            {28, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<16, 32, 128, 16, 32, 64, char>(p); }},
            {29, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<16, 64, 128, 16, 64, 64, char>(p); }},
            {30, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<16, 64, 128, 16, 64, 64, char>(p); }},
            // 2 warp
            {31, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 16, 128, 16, 16, 64, char>(p); }},
            {32, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 32, 128, 16, 32, 64, char>(p); }},
            {33, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 64, 128, 16, 64, 64, char>(p); }},
            //{34, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 128, 128, 16, 128, 64, char>(p); }}, //寄存器溢出 156
            // 4 warp
            {35, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 16, 128, 16, 16, 64, char>(p); }},
            {36, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 32, 128, 16, 32, 64, char>(p); }},
            // {37, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 64, 128, 16, 64, 64, char>(p); }}, // 寄存器溢出 120
            // {38, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 128, 128, 16, 128, 64, char>(p); }}, // 寄存器溢出 120
            
            ////////////////////////////////////////////////////N_warp-stages=4//////////////////////////////////////////////
            // 4warp-4stages-warpk
            {39, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 64, 128, 32, 16, 32, char>(p); }},
            {40, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 64, 128, 64, 16, 32, char>(p); }},

            {41, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 128, 128, 32, 32, 32, char>(p); }},
            {42, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 128, 128, 64, 32, 32, char>(p); }},

            {43, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 256, 128, 32, 64, 32, char>(p); }},
            {44, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 256, 128, 64, 64, 32, char>(p); }},
            
            // 8warp-4stages-warpk
            {45, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 128, 128, 64, 16, 32, char>(p); }},
            // 8warp
            {46, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 256, 128, 64, 32, 32, char>(p); }},
            
            ////////////////////////////////////////////////////MN_WARP-stages=4/////////////////////////////////////////////
            // 4 warp:  2,2,1
            // thread: 2,2,1    1,2,1    2,1,1  1,1,1
            {47, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 64, 128, 32, 32, 32, char>(p); }},
            // 8 warp:  2,4,1
            // thread: 2,2,1    1,2,1    2,1,1  1,1,1
            {48, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 128, 128, 32, 32, 32, char>(p); }},
            // 8 warp:  4,2,1
            // thread: 2,2,1    1,2,1    2,1,1  1,1,1
            {49, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<128, 64, 128, 32, 32, 32, char>(p); }},

            ///////////////////////////////////////////////////////N_warp-256////////////////////////////////////////////////
            // 4warp BLOCKN 64 
            {50, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<16, 64, 128, 16, 16, 64, char>(p); }},
            {51, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 64, 128, 32, 16, 64, char>(p); }},
            {52, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 64, 128, 64, 16, 64, char>(p); }},
            // 4warp BLOCKN 128 
            {53, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<16, 128, 128, 16, 32, 64, char>(p); }},
            {54, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 128, 128, 32, 32, 64, char>(p); }},
            //{55, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 128, 128, 64, 32, 64, char>(p); }}, // 寄存器溢出 32
            // 4warp BLOCKN 256 
            {56, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<16, 128, 128, 16, 64, 64, char>(p); }},
            //{57, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 128, 128, 32, 64, 64, char>(p); }}, // 寄存器溢出 60
            //{58, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 128, 128, 64, 64, 64, char>(p); }}, //寄存器溢出 392
            // 8warp BLOCKN 128 
            {59, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 128, 128, 32, 16, 64, char>(p); }},
            {60, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 128, 128, 64, 16, 64, char>(p); }},
            // 8warp BLOCKN 256 
            {61, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<32, 128, 128, 32, 32, 64, char>(p); }},
            //{62, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 128, 128, 64, 32, 64, char>(p); }}, //寄存器溢出28
            // 16warp
            {63, [](const GemmParams<char>& p) { launch_moe_gemm_second_stage<64, 128, 128, 64, 16, 64, char>(p); }},
        };

        auto it = kernel_maps.find(mode);
        if (it != kernel_maps.end()) {
            it->second(params_in);
        } else {
            printf("No matching kernel configuration found, using default settings \n");
            launch_moe_gemm_second_stage<16, 64, 128, 16, 16, 64>(params_in);
        }
      }
    } else {
      TORCH_CHECK(false, "moe_w8a8_gemm only supports int8");
    }

    return output;
}

}
}

