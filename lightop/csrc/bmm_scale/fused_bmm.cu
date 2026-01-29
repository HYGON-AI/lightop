//delete when compile whl
#include "fused_bmm_utils.h"
#include "fused_bmm_config.h"

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
  typename ElementAccum, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K,
  bool LDS_BYPASS> 
__global__ void __launch_bounds__(1024) FUSED_BMM_B8_B8_PERCHANNEL_HIP_NT_DECODE(
  const Element* __restrict__ q_nope, // [batch, n, p]
  const Element* __restrict__ w_uv,   // [n, p, l]
  unsigned char* __restrict__ output, // [batch, n, l]
  float* __restrict__ a_scale,        // [n, batch, 1]
  float* __restrict__ b_scale,        // [n, l, 1]
  bhalf_t* __restrict__ bias,         // [batch, l]
  uint32_t batch_size,
  uint32_t n_size, 
  uint32_t p_size,
  uint32_t l_size,
  uint32_t stride_ab,
  uint32_t stride_an,
  uint32_t stride_ap,
  uint32_t stride_bn,
  uint32_t stride_bl,
  uint32_t stride_bp,
  uint32_t stride_ob,
  uint32_t stride_on,
  uint32_t stride_ol,
  uint32_t stride_asb,
  uint32_t stride_asn,
  uint32_t stride_ask,
  uint32_t stride_bsn,
  uint32_t stride_bsl,
  uint32_t stride_bsk,
  bool has_bias
) {

  int bidx = blockIdx.x; // pid_m
  int bidy = blockIdx.y; // pid_n
  int bidz = blockIdx.z; // pid_batch: n_size

  constexpr int STAGES = BLOCK_SIZE_K/WARP_K; 
  const int w_uv_offset = bidz * stride_bl * l_size + bidy * BLOCK_SIZE_N * stride_bl;
  const uint32_t output_offset = bidz * stride_on + bidy * BLOCK_SIZE_N;
  const int size_k = p_size;
  const int size_m = batch_size;
  const int size_n = l_size;

  auto g_input = q_nope;
  unsigned char* g_output;

  g_output = output + output_offset;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;
  constexpr int mfma_kpack = 32*2;

  int warp_id_vec = threadIdx.x / 64;                                  //warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);           // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                                      // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  int warp_n_id = warp_id % warp_n_num;
  int warp_m_id = warp_id / warp_n_num;

  constexpr int dwordx4_num_elements = 16;
  constexpr int N_thread = BLOCK_SIZE_N / dwordx4_num_elements;  //N方向需要的线程数 使用dwordx4即16个fp8
  // const int m_num_threads = std::max(64 / N_thread, 1);
  const int m_num_threads = 64 / N_thread; // 要求BLOCK_SIZE_N <= 512
  const int n_num_threads = 64 / m_num_threads;
  const int m_thread_idx = lane_id / n_num_threads;
  const int n_thread_idx = lane_id & (n_num_threads-1);

  extern __shared__ unsigned char smem[];                              // 声明lds信息（统一用 u8，避免 char/uchar 重定义）
  Element* input_lds = reinterpret_cast<Element*>(smem);               // 输入lds首地址,首地址开始分配给input
  Element* w_uv_lds = input_lds + STAGES * (BLOCK_SIZE_M * WARP_K);    // 按照最简单的方式分块
  uint8_t* output_lds = reinterpret_cast<uint8_t*>(smem);              // 重复使用lds,留给output_lds

  union_vec<Element, 16> A_reg[WARP_M / mfma_m * STAGES][WARP_K/mfma_kpack]; 
  union_vec<Element, 16> B_reg[WARP_N / mfma_n * (1+LDS_BYPASS)][WARP_K/mfma_kpack]; // only support STAGES==1/2

  auto g_w_uv = w_uv + w_uv_offset;
  union_vec<ElementAccum, 4> C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0};
  float a_scale_arr[WARP_M / mfma_m];
  float b_scale_arr[(WARP_N / mfma_n) * 4];
  float *a_scale_ptr = (a_scale + bidz * stride_asn + bidx * BLOCK_SIZE_M * stride_asb + warp_m_id * WARP_M);
  float *b_scale_ptr = (b_scale + bidz * stride_bsn + bidy * BLOCK_SIZE_N * stride_bsl + warp_n_id * WARP_N);
  // bhalf_t *bias_ptr = bias + bidz * l_size + bidy * BLOCK_SIZE_N + warp_n_id * WARP_N;

  #pragma unroll
  for(int idx = 0; idx < WARP_M / mfma_m; idx++){
      uint32_t a_scale_offset = warp_m_id * WARP_M + idx * mfma_m + row_id;
      a_scale_arr[idx] = *(a_scale_ptr + std::min(a_scale_offset,  batch_size - 1) * stride_asb); //计算M方向的偏移
  }

  #pragma unroll
  for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
    #pragma unroll
    for(int reg_id = 0; reg_id < 4; reg_id++){
      b_scale_arr[min_tile_n * 4 + reg_id] = *(b_scale_ptr + min_tile_n * mfma_n + col_id + reg_id * 4);
    }
  }

  if constexpr(LDS_BYPASS){
    if constexpr(STAGES == 1){
      batch_gemm_stage_one_lds_bypass_impl<0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element, ElementAccum>
        (g_input, g_w_uv, input_lds, w_uv_lds, A_reg, B_reg, C_reg, warp_id, batch_size, n_size, p_size, stride_ab, stride_an, stride_bl, bidx, bidz);
    }else{
      batch_gemm_lds_bypass_impl<0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element, ElementAccum>
        (g_input, g_w_uv, input_lds, w_uv_lds, A_reg, B_reg, C_reg, warp_id, batch_size, n_size, p_size, stride_ab, stride_an, stride_bl, bidx, bidz);
    }
  }else{
    if constexpr(STAGES == 1){
      batch_gemm_stage_one_impl<0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element, ElementAccum>
        (g_input, g_w_uv, input_lds, w_uv_lds, A_reg, B_reg, C_reg, warp_id, batch_size, n_size, p_size, stride_ab, stride_an, stride_bl, bidx, bidz);
    }else{
      batch_gemm_impl<0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element, ElementAccum>
        (g_input, g_w_uv, input_lds, w_uv_lds, A_reg, B_reg, C_reg, warp_id, batch_size, n_size, p_size, stride_ab, stride_an, stride_bl, bidx, bidz);
    }
  }

  floatx4 scale_valuex4 = {0.0f, 0.0f, 0.0f, 0.0f};
  uint32_t float8x4_value;
  //save output to lds
  #pragma unroll
  for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
    float a_scale = a_scale_arr[min_tile_m];
    #pragma unroll
    for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
      #pragma unroll
      for(int reg_id = 0; reg_id < 4; reg_id++){
        float b_scale = b_scale_arr[min_tile_n * 4 + reg_id];
        scale_valuex4[reg_id] = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n].scalar_array[reg_id] * a_scale * b_scale;
      }
      float8x4_value = float_to_fp8e4m3_x4(scale_valuex4[0], scale_valuex4[1], scale_valuex4[2], scale_valuex4[3]);
      #pragma unroll
      for(int reg_id = 0; reg_id < 4; reg_id++){
        int index = (warp_m_id * WARP_M + min_tile_m * mfma_m + (lane_id & 15 )) * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N 
            + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + warp_m_id * WARP_M + (lane_id % 16) )/2 * 4/*padding*/;
        // bhalf_t acc_scale_value = f32_to_bf16(value);
        // if (has_bias){
        //   bhalf_t bias_value = bias_ptr[min_tile_n * mfma_n + col_id + reg_id * 4];
        //   acc_scale_value = acc_scale_value + bias_value;
        // }
        output_lds[index] = reinterpret_cast<uint8_t*>(&float8x4_value)[reg_id];
      }
    }
  }

  __syncthreads();

  // {
  //   // 最大化利用dwordx4 通用
  //   const int tid = threadIdx.x;
  //   constexpr int N_thread = BLOCK_SIZE_N / 16;  //N方向需要的线程数 使用dwordx4即8个bf16
  //   using vec16_fp8 = __attribute__((__vector_size__(16 * sizeof(char)))) char;
  //   int m_idx = threadIdx.x / N_thread;  
  //   int n_idx = threadIdx.x % N_thread;  
  //   for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
  //     const int32_t batch_index = (bidx * BLOCK_SIZE_M + m_idx);

  //     if(batch_index < batch_size) {
  //       *reinterpret_cast<vec16_fp8*>(&g_output[batch_index * stride_ob + n_idx * 16]) = 
  //       *reinterpret_cast<vec16_fp8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 16 + m_idx/2 * 4/*padding*/]);
  //     }
  //   } 
  // }  
  
  {
    const int m_pos_in_block = m_thread_idx + warp_id * m_num_threads;
    using vec16_fp8 = __attribute__((__vector_size__(16 * sizeof(char)))) char;
    #pragma unroll
    for(int m_idx = 0; m_idx < BLOCK_SIZE_M; m_idx += m_num_threads*WARP_NUM) {
      const int m_lds_idx = m_idx + m_pos_in_block;
      if(m_lds_idx < BLOCK_SIZE_M) {
        const int32_t batch_index = (bidx * BLOCK_SIZE_M + m_pos_in_block);
        if(batch_index < batch_size) {
          #pragma unroll
          for(int n_idx = 0; n_idx < BLOCK_SIZE_N; n_idx += n_num_threads*dwordx4_num_elements) {
            *reinterpret_cast<vec16_fp8*>(&g_output[batch_index * stride_ob + n_idx + n_thread_idx * dwordx4_num_elements]) = 
            *reinterpret_cast<vec16_fp8*>(&output_lds[(m_idx + m_thread_idx + warp_id * m_num_threads) * BLOCK_SIZE_N 
              + n_idx + n_thread_idx*dwordx4_num_elements + (m_idx + m_thread_idx + warp_id * m_num_threads)/2 * 4/*padding*/]);
          }
        }
      }
    }
  }

}


template<
  typename Element,
  typename ElementAccum, 
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K,
  bool LDS_BYPASS> 
__global__ void __launch_bounds__(1024) FUSED_BMM_B8_B16_PERCHANNEL_HIP_NT_DECODE(
  const Element* __restrict__ q_nope, // [batch, n, p]
  const Element* __restrict__ w_uv,   // [n, p, l]
  bhalf_t* __restrict__ output,       // [batch, n, l]
  float* __restrict__ a_scale,        // [n, batch, 1]
  float* __restrict__ b_scale,        // [n, l, 1]
  bhalf_t* __restrict__ bias,         // [batch, l]
  uint32_t batch_size,
  uint32_t n_size, 
  uint32_t p_size,
  uint32_t l_size,
  uint32_t stride_ab,
  uint32_t stride_an,
  uint32_t stride_ap,
  uint32_t stride_bn,
  uint32_t stride_bl,
  uint32_t stride_bp,
  uint32_t stride_ob,
  uint32_t stride_on,
  uint32_t stride_ol,
  uint32_t stride_asb,
  uint32_t stride_asn,
  uint32_t stride_ask,
  uint32_t stride_bsn,
  uint32_t stride_bsl,
  uint32_t stride_bsk,
  bool has_bias
) {

  int bidx = blockIdx.x; // pid_m
  int bidy = blockIdx.y; // pid_n
  int bidz = blockIdx.z; // pid_batch: n_size

  constexpr int STAGES = BLOCK_SIZE_K/WARP_K; 
  const int w_uv_offset = bidz * stride_bl * l_size + bidy * BLOCK_SIZE_N * stride_bl;
  const uint32_t output_offset = bidz * stride_on + bidy * BLOCK_SIZE_N;
  const int size_k = p_size;
  const int size_m = batch_size;
  const int size_n = l_size;

  auto g_input = q_nope;
  bhalf_t* g_output;

  g_output = output + output_offset;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int mfma_k = 32;
  constexpr int mfma_kpack = 32*2;

  int warp_id_vec = threadIdx.x / 64;                                  //warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);           // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
  int lane_id = threadIdx.x & 63;                                      // thread_id
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;
  const int warp_n_num = BLOCK_SIZE_N / WARP_N;
  int warp_n_id = warp_id % warp_n_num;
  int warp_m_id = warp_id / warp_n_num;

  constexpr int dwordx4_num_elements = 8;
  constexpr int N_thread = BLOCK_SIZE_N / dwordx4_num_elements;  //N方向需要的线程数 使用dwordx4即8个bf16
  // const int m_num_threads = std::max(64 / N_thread, 1);
  const int m_num_threads = 64 / N_thread; // 要求BLOCK_SIZE_N <= 512
  const int n_num_threads = 64 / m_num_threads;
  const int m_thread_idx = lane_id / n_num_threads;
  const int n_thread_idx = lane_id & (n_num_threads-1);

  extern __shared__ unsigned char smem[];                              // 声明lds信息（统一用 u8，避免 char/uchar 重定义）
  Element* input_lds = reinterpret_cast<Element*>(smem);               // 输入lds首地址,首地址开始分配给input
  Element* w_uv_lds = input_lds + STAGES * (BLOCK_SIZE_M * WARP_K);    // 按照最简单的方式分块
  bhalf_t* output_lds = reinterpret_cast<bhalf_t*>(smem);              // 重复使用lds,留给output_lds

  union_vec<Element, 16> A_reg[WARP_M / mfma_m * STAGES][WARP_K/mfma_kpack]; 
  union_vec<Element, 16> B_reg[WARP_N / mfma_n * (1+LDS_BYPASS)][WARP_K/mfma_kpack]; // only support STAGES==1/2

  auto g_w_uv = w_uv + w_uv_offset;
  union_vec<ElementAccum, 4> C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0};
  float a_scale_arr[WARP_M / mfma_m];
  float b_scale_arr[(WARP_N / mfma_n) * 4];
  float *a_scale_ptr = (a_scale + bidz * stride_asn + bidx * BLOCK_SIZE_M * stride_asb + warp_m_id * WARP_M);
  float *b_scale_ptr = (b_scale + bidz * stride_bsn + bidy * BLOCK_SIZE_N * stride_bsl + warp_n_id * WARP_N);
  // bhalf_t *bias_ptr = bias + bidz * l_size + bidy * BLOCK_SIZE_N + warp_n_id * WARP_N;

  #pragma unroll
  for(int idx = 0; idx < WARP_M / mfma_m; idx++){
      uint32_t a_scale_offset = warp_m_id * WARP_M + idx * mfma_m + row_id;
      a_scale_arr[idx] = *(a_scale_ptr + std::min(a_scale_offset,  batch_size - 1) * stride_asb); //计算M方向的偏移
  }

  #pragma unroll
  for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
    #pragma unroll
    for(int reg_id = 0; reg_id < 4; reg_id++){
      b_scale_arr[min_tile_n * 4 + reg_id] = *(b_scale_ptr + min_tile_n * mfma_n + col_id + reg_id * 4);
    }
  }

  if constexpr(LDS_BYPASS){
    if constexpr(STAGES == 1){
      batch_gemm_stage_one_lds_bypass_impl<0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element, ElementAccum>
        (g_input, g_w_uv, input_lds, w_uv_lds, A_reg, B_reg, C_reg, warp_id, batch_size, n_size, p_size, stride_ab, stride_an, stride_bl, bidx, bidz);
    }else{
      batch_gemm_lds_bypass_impl<0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element, ElementAccum>
        (g_input, g_w_uv, input_lds, w_uv_lds, A_reg, B_reg, C_reg, warp_id, batch_size, n_size, p_size, stride_ab, stride_an, stride_bl, bidx, bidz);
    }
  }else{
    if constexpr(STAGES == 1){
      batch_gemm_stage_one_impl<0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element, ElementAccum>
        (g_input, g_w_uv, input_lds, w_uv_lds, A_reg, B_reg, C_reg, warp_id, batch_size, n_size, p_size, stride_ab, stride_an, stride_bl, bidx, bidz);
    }else{
      batch_gemm_impl<0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element, ElementAccum>
        (g_input, g_w_uv, input_lds, w_uv_lds, A_reg, B_reg, C_reg, warp_id, batch_size, n_size, p_size, stride_ab, stride_an, stride_bl, bidx, bidz);
    }
  }

  // floatx4 scale_valuex4 = {0.0f, 0.0f, 0.0f, 0.0f};
  // uint32_t float8x4_value;
  //save output to lds
  #pragma unroll
  for(int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++){
    float a_scale = a_scale_arr[min_tile_m];
    #pragma unroll
    for(int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++){
      // #pragma unroll
      // for(int reg_id = 0; reg_id < 4; reg_id++){
      //   float b_scale = b_scale_arr[min_tile_n * 4 + reg_id];
      //   scale_valuex4[reg_id] = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n].scalar_array[reg_id] * a_scale * b_scale;
      // }
      // float8x4_value = float_to_fp8e4m3_x4(scale_valuex4[0], scale_valuex4[1], scale_valuex4[2], scale_valuex4[3]);
      #pragma unroll
      for(int reg_id = 0; reg_id < 4; reg_id++){
        float b_scale = b_scale_arr[min_tile_n * 4 + reg_id];
        float value = C_reg[0][min_tile_m * WARP_N/mfma_n + min_tile_n].scalar_array[reg_id] * a_scale * b_scale;
        int index = (warp_m_id * WARP_M + min_tile_m * mfma_m + (lane_id & 15 )) * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N 
            + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + warp_m_id * WARP_M + (lane_id % 16) )/2 * 2/*padding*/;
        bhalf_t acc_scale_value = f32_to_bf16(value);
        // if (has_bias){
        //   bhalf_t bias_value = bias_ptr[min_tile_n * mfma_n + col_id + reg_id * 4];
        //   acc_scale_value = acc_scale_value + bias_value;
        // }
        output_lds[index] = acc_scale_value;
      }
    }
  }

  __syncthreads();

  // {
  //   // 最大化利用dwordx4 通用
  //   const int tid = threadIdx.x;
  //   constexpr int N_thread = BLOCK_SIZE_N / 16;  //N方向需要的线程数 使用dwordx4即8个bf16
  //   using vec16_fp8 = __attribute__((__vector_size__(16 * sizeof(char)))) char;
  //   int m_idx = threadIdx.x / N_thread;  
  //   int n_idx = threadIdx.x % N_thread;  
  //   for(; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM*64) / N_thread) {
  //     const int32_t batch_index = (bidx * BLOCK_SIZE_M + m_idx);

  //     if(batch_index < batch_size) {
  //       *reinterpret_cast<vec16_fp8*>(&g_output[batch_index * stride_ob + n_idx * 16]) = 
  //       *reinterpret_cast<vec16_fp8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 16 + m_idx/2 * 4/*padding*/]);
  //     }
  //   } 
  // }  
  
  {
    const int m_pos_in_block = m_thread_idx + warp_id * m_num_threads;
    using vec8_bf16 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
    #pragma unroll
    for(int m_idx = 0; m_idx < BLOCK_SIZE_M; m_idx += m_num_threads*WARP_NUM) {
      const int m_lds_idx = m_idx + m_pos_in_block;
      if(m_lds_idx < BLOCK_SIZE_M) {
        const int32_t batch_index = (bidx * BLOCK_SIZE_M + m_pos_in_block);
        if(batch_index < batch_size) {
          #pragma unroll
          for(int n_idx = 0; n_idx < BLOCK_SIZE_N; n_idx += n_num_threads*dwordx4_num_elements) {
            *reinterpret_cast<vec8_bf16*>(&g_output[batch_index * stride_ob + n_idx + n_thread_idx * dwordx4_num_elements]) = 
            *reinterpret_cast<vec8_bf16*>(&output_lds[(m_idx + m_thread_idx + warp_id * m_num_threads) * BLOCK_SIZE_N 
              + n_idx + n_thread_idx*dwordx4_num_elements + (m_idx + m_thread_idx + warp_id * m_num_threads)/2 * 2/*padding*/]);
          }
        }
      }
    }
  }

}

// launch_gemm1
// 在GemmParams结构体后添加
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T, typename U, typename V, bool LDS_BYPASS>
void launch_fused_bmm_decode(const GemmParams<T, V>& params) {
    const int STAGES = BLOCK_SIZE_K/WARP_K;
    const int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N);
    const bool BIAS = false;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;

    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;
    // printf("BLOCK_SIZE_M: %d, BLOCK_SIZE_N: %d, BLOCK_SIZE_K: %d, WARP_M: %d, WARP_N: %d, WARP_K: %d\n", BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K);
    gridDim.x = DIVIDE(params.batch_size, BLOCK_SIZE_M); // m方向
    gridDim.y = DIVIDE(params.l_size, BLOCK_SIZE_N); // n方向
    gridDim.z = params.n_size; // batch方向
    const int bank_bytes_width = 4; // 4bytes=32bits
    constexpr size_t bytes_per_out_elem = sizeof(V);
    const int padding_bytes = bank_bytes_width / bytes_per_out_elem;
    // printf("gridDim.x: %d, gridDim.y: %d, gridDim.z: %d\n", gridDim.x, gridDim.y, gridDim.z);
    const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + (1-LDS_BYPASS)*BLOCK_SIZE_N * WARP_K), BLOCK_SIZE_M * BLOCK_SIZE_N * bytes_per_out_elem + BLOCK_SIZE_M * BLOCK_SIZE_M/2 * padding_bytes/*padding*/);
    // const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + BLOCK_SIZE_N * WARP_K), BLOCK_SIZE_M * BLOCK_SIZE_N);
    const int shared_mem_size = lds_size;
    // printf("shared_mem_size: %d\n", shared_mem_size);
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    
    if constexpr (std::is_same_v<V, unsigned char>){
      FUSED_BMM_B8_B8_PERCHANNEL_HIP_NT_DECODE<T, U, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, LDS_BYPASS>
      <<<gridDim, blockDim, shared_mem_size, stream>>>(
          params.ptr_q_nope, 
          params.ptr_w_uv, 
          params.ptr_output, 
          params.ptr_a_scale, 
          params.ptr_b_scale,  
          params.ptr_bias,
          params.batch_size,
          params.n_size,
          params.p_size,
          params.l_size,
          params.stride_ab,
          params.stride_an,
          params.stride_ap,
          params.stride_bn,
          params.stride_bl,
          params.stride_bp,
          params.stride_ob,
          params.stride_on,
          params.stride_ol,
          params.stride_asb,
          params.stride_asn,
          params.stride_ask,
          params.stride_bsn,
          params.stride_bsl,
          params.stride_bsk,
          params.has_bias
        );
    }else if constexpr (std::is_same_v<V, bhalf_t>){

      FUSED_BMM_B8_B16_PERCHANNEL_HIP_NT_DECODE<T, U, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, LDS_BYPASS>
      <<<gridDim, blockDim, shared_mem_size, stream>>>(
          params.ptr_q_nope, 
          params.ptr_w_uv, 
          params.ptr_output, 
          params.ptr_a_scale, 
          params.ptr_b_scale,  
          params.ptr_bias,
          params.batch_size,
          params.n_size,
          params.p_size,
          params.l_size,
          params.stride_ab,
          params.stride_an,
          params.stride_ap,
          params.stride_bn,
          params.stride_bl,
          params.stride_bp,
          params.stride_ob,
          params.stride_on,
          params.stride_ol,
          params.stride_asb,
          params.stride_asn,
          params.stride_ask,
          params.stride_bsn,
          params.stride_bsl,
          params.stride_bsk,
          params.has_bias
        );
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
torch::Tensor fused_bmm(torch::Tensor q_nope,                    // [b, n, p]
                        torch::Tensor w_uv,                      // [n, l, p]
                        torch::Tensor output,                    // [b, n, l]
                        torch::Tensor a_scale,                   // [1, b, 1]
                        torch::Tensor b_scale,                   // [1, l, 1]
                        std::optional<torch::Tensor> bias,       // [b, l]
                        at::ScalarType dtype,                   // fp8_e4m3fn or fp8_e5m2
                        bool transpose_bm,
                        bool transpose_bm_in,
                        int mode
  ) {
    int batch_size = q_nope.size(0); 
    int n_size = q_nope.size(1);
    const int p_size = w_uv.size(2);
    const int l_size = w_uv.size(1);
    int stride_ab = q_nope.stride(0);
    int stride_an = q_nope.stride(1);
    const int stride_ap = q_nope.stride(2);
    const int stride_bn = w_uv.stride(0);
    const int stride_bl = w_uv.stride(1);
    const int stride_bp = w_uv.stride(2);
    int stride_ob = output.stride(0);
    int stride_on = output.stride(1);
    const int stride_ol = output.stride(2);
    int stride_asb = a_scale.stride(0);
    int stride_asn = a_scale.stride(1);
    const int stride_ask = a_scale.stride(2);
    const int stride_bsn = b_scale.stride(0);
    const int stride_bsl = b_scale.stride(1);
    const int stride_bsk = b_scale.stride(2);

    if (transpose_bm_in){
      batch_size = q_nope.size(1);
      n_size = q_nope.size(0);
      stride_ab = q_nope.stride(1);
      stride_an = q_nope.stride(0);
      stride_asb = a_scale.stride(1);
      stride_asn = a_scale.stride(0);
    }

    if (transpose_bm){
      stride_ob = output.stride(1);
      stride_on = output.stride(0);
    }

    torch::Tensor output_alias = output.alias();

    bhalf_t* bias_ptr = nullptr; // 第一阶段这里为null
    bool has_bias = false;
    if (bias.has_value()){
      bias_ptr = (bhalf_t*)bias.value().data_ptr();
      has_bias = true;
    }

    if(dtype == at::ScalarType::Float8_e4m3fn){
      if (q_nope.scalar_type() == at::ScalarType::Char){

        GemmParams<char, unsigned char> params_in(
          (const char*)q_nope.data_ptr<int8_t>(), 
          (const char*)w_uv.data_ptr<int8_t>(), 
          (unsigned char*)output_alias.data_ptr(),
          (float*)a_scale.data_ptr(), 
          (float*)b_scale.data_ptr(),  
          bias_ptr,
          batch_size,
          n_size,
          p_size,
          l_size,
          stride_ab,
          stride_an,
          stride_ap,
          stride_bn,
          stride_bl,
          stride_bp,
          stride_ob,
          stride_on,
          stride_ol,
          stride_asb,
          stride_asn,
          stride_ask,
          stride_bsn,
          stride_bsl,
          stride_bsk,
          has_bias
        );
      
        auto it = kernel_maps_batched_gemm_int8_fp8.find(mode);
        if (it != kernel_maps_batched_gemm_int8_fp8.end()) {
            it->second(params_in);
        } else {
            printf("No matching kernel configuration found, using default settings \n");
            launch_fused_bmm_decode<16, 64, 128, 16, 16, 64, char, int, unsigned char, false>(params_in);
        }

      } else if (q_nope.scalar_type() == at::kFloat8_e4m3fn) {

        GemmParams<fp8_t> params_in(
          reinterpret_cast<const fp8_t*>(q_nope.data_ptr()),
          reinterpret_cast<const fp8_t*>(w_uv.data_ptr()),
          (unsigned char*)output_alias.data_ptr(),
          (float*)a_scale.data_ptr(), 
          (float*)b_scale.data_ptr(),  
          bias_ptr,
          batch_size,
          n_size,
          p_size,
          l_size,
          stride_ab,
          stride_an,
          stride_ap,
          stride_bn,
          stride_bl,
          stride_bp,
          stride_ob,
          stride_on,
          stride_ol,
          stride_asb,
          stride_asn,
          stride_ask,
          stride_bsn,
          stride_bsl,
          stride_bsk,
          has_bias
        );
      
        auto it = kernel_maps_batched_gemm_fp8_fp8.find(mode);
        if (it != kernel_maps_batched_gemm_fp8_fp8.end()) {
            it->second(params_in);
        } else {
            printf("No matching kernel configuration found, using default settings \n");
            launch_fused_bmm_decode<16, 64, 128, 16, 16, 64, fp8_t, float, unsigned char, false>(params_in);
        }
      }
    }else if(dtype == at::ScalarType::BFloat16){
      if (q_nope.scalar_type() == at::ScalarType::Char){

        GemmParams<char, bhalf_t> params_in(
          (const char*)q_nope.data_ptr<int8_t>(), 
          (const char*)w_uv.data_ptr<int8_t>(), 
          (bhalf_t*)output_alias.data_ptr(),
          (float*)a_scale.data_ptr(), 
          (float*)b_scale.data_ptr(),  
          bias_ptr,
          batch_size,
          n_size,
          p_size,
          l_size,
          stride_ab,
          stride_an,
          stride_ap,
          stride_bn,
          stride_bl,
          stride_bp,
          stride_ob,
          stride_on,
          stride_ol,
          stride_asb,
          stride_asn,
          stride_ask,
          stride_bsn,
          stride_bsl,
          stride_bsk,
          has_bias
        );
      
        auto it = kernel_maps_batched_gemm_int8_bf16.find(mode);
        if (it != kernel_maps_batched_gemm_int8_bf16.end()) {
            it->second(params_in);
        } else {
            printf("No matching kernel configuration found, using default settings \n");
            launch_fused_bmm_decode<16, 64, 128, 16, 16, 64, char, int, bhalf_t, false>(params_in);
        }

      } else if (q_nope.scalar_type() == at::kFloat8_e4m3fn) {

        GemmParams<fp8_t, bhalf_t> params_in(
          reinterpret_cast<const fp8_t*>(q_nope.data_ptr()),
          reinterpret_cast<const fp8_t*>(w_uv.data_ptr()),
          (bhalf_t*)output_alias.data_ptr(),
          (float*)a_scale.data_ptr(), 
          (float*)b_scale.data_ptr(),  
          bias_ptr,
          batch_size,
          n_size,
          p_size,
          l_size,
          stride_ab,
          stride_an,
          stride_ap,
          stride_bn,
          stride_bl,
          stride_bp,
          stride_ob,
          stride_on,
          stride_ol,
          stride_asb,
          stride_asn,
          stride_ask,
          stride_bsn,
          stride_bsl,
          stride_bsk,
          has_bias
        );
      
        auto it = kernel_maps_batched_gemm_fp8_bf16.find(mode);
        if (it != kernel_maps_batched_gemm_fp8_bf16.end()) {
            it->second(params_in);
        } else {
            printf("No matching kernel configuration found, using default settings \n");
            launch_fused_bmm_decode<16, 64, 128, 16, 16, 64, fp8_t, float, bhalf_t, false>(params_in);
        }
      }
    }else {
      TORCH_CHECK(false, "fused_bmm only supports input:int8 or fp8 and output:fp8 or bf16");
    }

    return output;
}

}
}

