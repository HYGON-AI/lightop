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
  uint16_t WARP_NUM,
  uint16_t BLOCK_SIZE_M, 
  uint16_t BLOCK_SIZE_N, 
  uint16_t BLOCK_SIZE_K, 
  uint16_t WARP_M, 
  uint16_t WARP_N, 
  uint16_t WARP_K, 
  bool BIAS> 
__global__ void __launch_bounds__(1024) FUSED_BMM_I8_DECODE(
  const Element* __restrict__ q_nope, // [batch, n, p]
  const Element* __restrict__ w_uv,   // [n, p, l]
  bhalf_t* __restrict__ output,       // [batch, n, l]
  float* __restrict__ a_scale,        // [n, batch, 1]
  float* __restrict__ b_scale,        // [n, l, 1]
  const bhalf_t* __restrict__ bias,   // [batch, l]
  uint32_t batch_size,
  uint32_t n_size, 
  uint32_t p_size,
  uint32_t l_size,
  uint32_t stride_asn,
  uint32_t stride_asb,
  uint32_t stride_ask,
  uint32_t stride_bsn,
  uint32_t stride_bsl,
  uint32_t stride_bsk
) {

  int bidx = blockIdx.x; // pid_m
  int bidy = blockIdx.y; // pid_n
  int bidz = blockIdx.z; // pid_batch: n_size

  constexpr int STAGES = BLOCK_SIZE_K/WARP_K; 
  const int w_uv_offset = bidz * p_size * l_size + bidy * BLOCK_SIZE_N;
  const uint32_t output_offset = bidz * l_size + bidy * BLOCK_SIZE_N;
  const int size_k = p_size;
  const int size_m = batch_size;
  const int size_n = l_size;
  const int nl_size = n_size * l_size;

  auto g_input = q_nope;
  bhalf_t* g_output;

  g_output = output + output_offset;

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
  Element* w_uv_lds = input_lds + STAGES * (BLOCK_SIZE_M * WARP_K);    // 按照最简单的方式分块
  bhalf_t* output_lds = (bhalf_t*)&(smem);                             // 重复使用lds,留给output_lds

  union_vec<Element, 8> A_reg[WARP_M / mfma_m * STAGES][WARP_K/mfma_k]; 
  union_vec<Element, 8> B_reg[WARP_N / mfma_n][WARP_K/mfma_k];

  auto g_w_uv = w_uv + w_uv_offset;
  intx4 C_reg[1][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0};                 // [4][2]  每个warp在n方向重复两次 tileN = 16*2
  float *a_scale_ptr_arr[WARP_M / mfma_m];
  float *a_scale_ptr = a_scale + bidz * stride_asn + bidx * BLOCK_SIZE_M * stride_asb;
  float *b_scale_ptr = b_scale + bidz * stride_bsn + bidy * BLOCK_SIZE_N * stride_bsl;

  #pragma unroll
  for(int idx = 0; idx < WARP_M / mfma_m; idx++){
      uint32_t a_scale_offset = warp_m_id * WARP_M + idx * mfma_m + row_id;
      a_scale_ptr_arr[idx] = a_scale_ptr + std::min(a_scale_offset,  batch_size - 1) * stride_asb; //计算M方向的偏移
  }

  { 
    batch_gemm_impl<0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>
      (g_input, g_w_uv, input_lds, w_uv_lds, A_reg, B_reg, C_reg, warp_id, batch_size, n_size, p_size, p_size, bidx, bidz);
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
      const int32_t batch_index = (bidx * BLOCK_SIZE_M + m_idx);

      if(batch_index < batch_size) {
        *reinterpret_cast<vec_bf16_8*>(&g_output[batch_index * nl_size + n_idx * 8]) = 
        //*reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8]);
        *reinterpret_cast<vec_bf16_8*>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx/2 * 2/*padding*/]);
      }
    } 
  }      
}


// launch_gemm1
// 在GemmParams结构体后添加
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
void launch_fused_bmm_decode(const GemmParams<T>& params) {
    const int STAGES = BLOCK_SIZE_K/WARP_K;
    const int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N);
    const bool BIAS = false;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;

    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64;
    blockDim.y = 1;
    blockDim.z = 1;

    // printf("launch_moe_w8a8_first_stage_prefill: %d, %d, %d, %d, %d, %d\n", BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K);
    // printf("moe_w8a8_gemm_second_stage_kernel_prefill: %d, %d, %d, %d, %d, %d, %d, %d\n", params.size_m, params.size_n, params.size_k, params.stride_asm, params.stride_ask, params.stride_bse, params.stride_bsn, params.stride_bsk);
    // block swizzle 效果提升不是很明显 需要tuning验证
    gridDim.x = DIVIDE(params.batch_size, BLOCK_SIZE_M); // m方向
    gridDim.y = DIVIDE(params.l_size, BLOCK_SIZE_N); // n方向
    gridDim.z = params.n_size; // batch方向

    const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + BLOCK_SIZE_N * WARP_K), BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
    const int shared_mem_size = lds_size;
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    
    FUSED_BMM_I8_DECODE<char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, BIAS>
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
          params.stride_asn,
          params.stride_asb,
          params.stride_ask,
          params.stride_bsn,
          params.stride_bsl,
          params.stride_bsk
          );
    
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
                        torch::Tensor a_scale,                   // [n, b, 1]
                        torch::Tensor b_scale,                   // [n, l, 1]
                        std::optional<torch::Tensor> bias,       // [b, l]
                        int mode
  ) {
    const int batch_size = q_nope.size(0); 
    const int n_size = q_nope.size(1);
    const int p_size = q_nope.size(2);
    // const int mode = 500;
    const int l_size = w_uv.size(1);
    const int stride_asn = a_scale.stride(0);
    const int stride_asb = a_scale.stride(1);
    const int stride_ask = a_scale.stride(2);;
    const int stride_bsn = b_scale.stride(0);
    const int stride_bsl = b_scale.stride(1);
    const int stride_bsk = b_scale.stride(2);
    torch::Tensor output_alias = output.alias();

    if (q_nope.scalar_type() == at::ScalarType::Char){
     
        GemmParams<char> params_in(
          (const char*)q_nope.data_ptr<int8_t>(), 
          (const char*)w_uv.data_ptr<int8_t>(), 
          (bhalf_t*)output_alias.data_ptr(),
          (float*)a_scale.data_ptr(), 
          (float*)b_scale.data_ptr(),  
          // (const bhalf_t*)bias.data_ptr<bhalf_t>(),
          nullptr,
          batch_size,
          n_size,
          p_size,
          l_size,
          stride_asn,
          stride_asb,
          stride_ask,
          stride_bsn,
          stride_bsl,
          stride_bsk
        );

        launch_fused_bmm_decode<16, 64, 128, 16, 16, 64>(params_in);
        // {
        //   auto it = kernel_maps_fused_bmm.find(mode);
        //   if (it != kernel_maps_fused_bmm.end()) {
        //       it->second(params_in);
        //   } else {
        //       printf("No matching kernel configuration found, using default settings \n");
        //       launch_moe_w8a8_second_stage_decode<16, 32, 128, 16, 16, 64, 2>(params_in);
        //   }

        // }

    } else {
      TORCH_CHECK(false, "moe_w8a8_gemm only supports int8");
    }

    return output;
}

}
}

