#include "marlin.h"
#include "marlin_dtypes.h"


using half_t  = _Float16;
using half2_t = __attribute__((__vector_size__(2 * sizeof(half_t)))) half_t;
using half4_t = __attribute__((__vector_size__(4 * sizeof(half_t)))) half_t;
using float4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using bhalf_t = uint16_t;  // bf16 type using uint16_t

namespace at {
namespace native {
namespace marlin_moe_w4a16{

  // 设备侧函数：将4位整数反量化为FP16并写入FragB
  __device__ void dequantize_int4_to_fp16(int q, typename ScalarType<__half>::FragB& output) {
    // 位掩码和魔法数定义
    int LO_MASK = 0x000f000f;    // 提取低4位组
    int HI_MASK = 0x00f000f0;    // 提取高4位组ffetch
    int EXP_MAGIC = 0x64006400;  // FP16指数部分魔法数
    int SUB_MAGIC = 0x64086408;  // 减去零点偏置 (8.0)
    int MUL_MAGIC = 0x2c002c00;  // 缩放因子 (0.0625)
    int ADD_MAGIC = 0xd480d480;  // 额外偏移 (-72.0)

    // 提取4位整数并构造中间FP16表示
    int lo = ((q & LO_MASK) | EXP_MAGIC);
    int hi = ((q & HI_MASK) | EXP_MAGIC);

    // 执行反量化计算
    __half2 lo_val = *reinterpret_cast<__half2*>(&lo);
    __half2 hi_val = *reinterpret_cast<__half2*>(&hi);
    __half2* sub_ptr = reinterpret_cast<__half2*>(&SUB_MAGIC);
    __half2* mul_ptr = reinterpret_cast<__half2*>(&MUL_MAGIC);
    __half2* add_ptr = reinterpret_cast<__half2*>(&ADD_MAGIC);

    // 写入FragB
    output[0] = __hsub2(lo_val, *sub_ptr);        // lo - 8.0
    output[1] = __hfma2(hi_val, *mul_ptr, *add_ptr);  // (hi * 0.0625) - 72.0
  }

  // 设备侧函数：将4位整数反量化为BF16并写入FragB
  __device__ void dequantize_int4_to_bf16(int q, typename ScalarType<uint16_t>::FragB& output) {
    // 复用fp16的反量化逻辑，然后转换为bf16
    typename ScalarType<__half>::FragB fp16_output;
    dequantize_int4_to_fp16(q, fp16_output);
    
    // 将fp16转换为bf16
    __half* fp16_ptr = reinterpret_cast<__half*>(&fp16_output);
    
    // 转换4个fp16值到bf16
    for (int i = 0; i < 4; i++) {
      float f32_val = __half2float(fp16_ptr[i]);
      output[i] = ScalarType<uint16_t>::float2num(f32_val);
    }
  }

  // 通用反量化函数，根据scalar_t类型选择正确的反量化函数
  template <typename scalar_t>
  __device__ inline void dequantize_int4(int q, typename ScalarType<scalar_t>::FragB& output) {
    if constexpr (std::is_same<scalar_t, __half>::value) {
      dequantize_int4_to_fp16(q, output);
    } else {
      dequantize_int4_to_fp16(q, *reinterpret_cast<typename ScalarType<__half>::FragB*>(&output));
    }
  }

  // FragB = Vec<__half2, 2>;
  // scalar_t = __half;
  // FragS = Vec<__half2, 1>;
  // using scalar_t2 = __half2;
  template <typename scalar_t>
  __device__ inline void scale_and_sub(
      typename ScalarType<scalar_t>::FragB& frag_b, __half s, __half zp) {
    using scalar_t2 = typename ScalarType<__half>::scalar_t2;
    scalar_t2 s2 = __half2half2(s);
    scalar_t2 zp2 = __half2half2(zp);
    if constexpr (std::is_same<scalar_t, __half>::value) {
      frag_b[0] = __hfma2(frag_b[0], s2, __hneg2(zp2));
      frag_b[1] = __hfma2(frag_b[1], s2, __hneg2(zp2));
    } 
    else {
      FragUnion& frag_union = *reinterpret_cast<FragUnion*>(&frag_b);
      using Dtype = ScalarType<scalar_t>;
      frag_union.frag_fp16[0] = __hfma2(frag_union.frag_fp16[0], s2, __hneg2(zp2));
      frag_union.frag_fp16[1] = __hfma2(frag_union.frag_fp16[1], s2, __hneg2(zp2));

      // half -> float -> half  : 750 -> 754
      // frag_union.frag_fp16[0].x = __float2half(__half2float(frag_union.frag_fp16[0].x));
      // frag_union.frag_fp16[0].y = __float2half(__half2float(frag_union.frag_fp16[0].y));
      // frag_union.frag_fp16[1].x = __float2half(__half2float(frag_union.frag_fp16[1].x));
      // frag_union.frag_fp16[1].y = __float2half(__half2float(frag_union.frag_fp16[1].y));

        

      // half -> flaot -> bf16 : 750 -> 1200 
      frag_union.frag_bf16[0] = Dtype::float2num_fast(__half2float(frag_union.frag_fp16[0].x));        
      frag_union.frag_bf16[1] = Dtype::float2num_fast(__half2float(frag_union.frag_fp16[0].y));
      frag_union.frag_bf16[2] = Dtype::float2num_fast(__half2float(frag_union.frag_fp16[1].x));
      frag_union.frag_bf16[3] = Dtype::float2num_fast(__half2float(frag_union.frag_fp16[1].y));
    }
    
  }

  template <typename scalar_t>
  __device__ inline void mmac_volatile(const typename ScalarType<scalar_t>::FragA& frag_a,
                            const typename ScalarType<scalar_t>::FragB& frag_b,
                            typename ScalarType<scalar_t>::FragC& frag_c) {
    // #if defined(__gfx936__) || defined(__gfx936__) || defined(__gfx928__)
    //   *(reinterpret_cast<float4_t*>(&frag_c)) = __builtin_amdgcn_mmac_f32_16x16x16f16(*(reinterpret_cast<const half4_t*>(&frag_a)), *(reinterpret_cast<const half4_t*>(&frag_b)), *(reinterpret_cast<float4_t*>(&frag_c)));
    // #endif

    if constexpr (std::is_same<scalar_t, __half>::value) {
      // fp16 computation
      *(reinterpret_cast<float4_t*>(&frag_c)) = __builtin_amdgcn_mmac_f32_16x16x16f16(*(reinterpret_cast<const half4_t*>(&frag_a)), *(reinterpret_cast<const half4_t*>(&frag_b)), *(reinterpret_cast<float4_t*>(&frag_c)));
    } else {
      // bf16 computation - need to convert frag_b from uint16_t to v4bh format
      using v4bh = __attribute__((__vector_size__(4 * sizeof(short)))) short;
      
      // Directly reinterpret frag_b as v4bh type
      const v4bh& bf16_vec = *(reinterpret_cast<const v4bh*>(&frag_b));
      

      *(reinterpret_cast<float4_t*>(&frag_c)) = __builtin_amdgcn_mmac_f32_16x16x16bf16(*(reinterpret_cast<const v4bh*>(&frag_a)), bf16_vec, *(reinterpret_cast<float4_t*>(&frag_c)));
  
      // if(blockIdx.x==0 && threadIdx.x==0)
      // {
      //   auto temp_a = *(reinterpret_cast<const half4_t*>(&frag_a));
      //   float frag_a_f[4] = {
      //     ScalarType<uint16_t>::num2float(reinterpret_cast<const uint16_t*>(&temp_a)[0]),
      //     ScalarType<uint16_t>::num2float(reinterpret_cast<const uint16_t*>(&temp_a)[1]),
      //     ScalarType<uint16_t>::num2float(reinterpret_cast<const uint16_t*>(&temp_a)[2]),
      //     ScalarType<uint16_t>::num2float(reinterpret_cast<const uint16_t*>(&temp_a)[3])
      //   };
      //   float frag_b_f[4] = {
      //     ScalarType<uint16_t>::num2float(frag_b.elems[0]),
      //     ScalarType<uint16_t>::num2float(frag_b.elems[1]),
      //     ScalarType<uint16_t>::num2float(frag_b.elems[2]),
      //     ScalarType<uint16_t>::num2float(frag_b.elems[3])
      //   };
      //   printf("bf16 frag_a: %f %f %f %f | bf16 frag_b: %f %f %f %f | bf16 frag_c before: %f %f %f %f\n",
      //     frag_a_f[0], frag_a_f[1], frag_a_f[2], frag_a_f[3],
      //     frag_b_f[0], frag_b_f[1], frag_b_f[2], frag_b_f[3],
      //     frag_c[0], frag_c[1], frag_c[2], frag_c[3]
      //   );
      }

    }

  template <typename scalar_t,  // compute dtype, half or __hip_float16
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const int stages,  // number of stages for the async global->shared
                              // fetch pipeline
          const int group_blocks   // number of consecutive 16x16 blocks
          >
  __global__ void Marlin(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    const int4* __restrict__ scales_zeros_ptr,  // fp16 quantization scales of shape
    const int32_t* __restrict__ sorted_token_ids_ptr,        // moe sorted_ids
    const int32_t* __restrict__ expert_ids_ptr,              // moe expert ids
    const int32_t* __restrict__ num_tokens_past_padded_ptr,  // moe num tokens
    const float* __restrict__ topk_weights_ptr,              // moe top weights
    int top_k,              // num of experts per token
    bool mul_topk_weights,  // mul topk weights or not
    int num_groups,         // number of scale groups per output channel
    int prob_m,             // batch dimension m
    int prob_n,             // output dimension n
    int prob_k,             // reduction dimension k
    int* locks             // extra global storage for barrier synchronization
  ) {
    using Dtype = ScalarType<scalar_t>;
    using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
    using FragA = typename ScalarType<scalar_t>::FragA;
    using FragB = typename ScalarType<scalar_t>::FragB;
    using FragC = typename ScalarType<scalar_t>::FragC;
    // For both fp16 and bf16, scale and zero point should be fp16 to reuse existing logic
    using FragS = Vec<__half2, 1>;
    using FragZP = Vec<__half2, 4>;
    # ifdef __gfx936__
      constexpr int gridx = 80*3;
    # else
      constexpr int gridx = 120*3;
    # endif


    const int blcok_idx_x = blockIdx.x;
    const int warp_idx = threadIdx.x % 64;
    
    extern __shared__ int4 sh[];
  
    constexpr int pack_factor = 32 / 4; //  8
    constexpr int moe_block_size = 16 * thread_m_blocks; // block_m
    int group_size = // 128
        (group_blocks == -1) ? prob_k : prob_k / num_groups;
    int scales_expert_stride = prob_n * prob_k / group_size / 8; // scales 一个专家的 int4 数据个数
    int zp_expert_stride = prob_n * prob_k / group_size / (pack_factor * 4); // zeros 一个专家的 int4 数据个数
    int scales_zp_expert_stride = scales_expert_stride + zp_expert_stride;

    // parallel: num valid moe blocks // 167
    int num_tokens_past_padded = num_tokens_past_padded_ptr[0]; // 一共几个token选择
    int parallel = num_tokens_past_padded / moe_block_size; // 
  
    int k_tiles = prob_k / 16 / thread_k_blocks; // k 方向 有几个 [128, 128]、[]的数据块 56  有几个 [64, 256] 的数据块 112
    int n_tiles = prob_n / 16 / thread_n_blocks; // n 方向 有几个 [128, 128]的数据块 4  有几个 [64, 256] 的数据块  2
    int iters = div_ceil(k_tiles * n_tiles * parallel, gridx); // 每个block 需要处理 [128, 128]数据块的数量 468  需处理 [64, 256] 的总数据块
    if (group_blocks >= thread_k_blocks) {
      iters = (group_blocks / thread_k_blocks) *
              div_ceil(iters, (group_blocks / thread_k_blocks));
    }
  
    int slice_row = (iters * blcok_idx_x) % k_tiles; // k方向 数据块的索引 
    int slice_col_par = (iters * blcok_idx_x) / k_tiles; // N方向 包含 parallel 的数据块的索引
    int slice_col = slice_col_par; // N方向 不包含 parallel 的数据块的索引
    int slice_iters;  // 当前条带中，当前block需要处理的 [128, 128]数据块的个数 
    int slice_count = 0;  // 当前条带中, 一共需要几个block执行计算
    int slice_idx;  // 当前条带中, 所有执行计算任务blcok中，该block的索引，索引从下 往上编号
    int slice_scale_row = 0; // 用来标记 当前的scale行索引，以确定是否 load scale和zero的值

    int par_id = 0;
    int block_id = -1;
    int64_t expert_id = 0;  // use int64 to avoid computation result overflow
    int old_expert_id = 0;
    int64_t B_expert_off = 0;
  
    int4* sh_block_sorted_ids_int4 = sh;
    int32_t* sh_block_sorted_ids =
        reinterpret_cast<int*>(sh_block_sorted_ids_int4);
    int4* sh_block_topk_weights_int4 =
        sh_block_sorted_ids_int4 + moe_block_size / 4;
    scalar_t2* sh_block_topk_weights =
        reinterpret_cast<scalar_t2*>(sh_block_topk_weights_int4);
    int4* sh_new = sh_block_topk_weights_int4 + moe_block_size / 4;
  
    int32_t block_num_valid_tokens = 0;
    int32_t locks_off = 0;

    // 修正 slice_col 和 par_id
    if (slice_col_par >= n_tiles) {
      slice_col = slice_col_par % n_tiles;
      par_id = slice_col_par / n_tiles;
    }
    
    if (parallel * n_tiles >= gridx) {
      // when parallel * n_tiles >= sms
      // then there are at most $sms$ conflict tile blocks
      locks_off = blockIdx.x;
    } else {
      locks_off = (iters * blockIdx.x) / k_tiles - 1;
    }
  
    // sh_block_sorted_ids_int4 sh_block_topk_weights 赋值
    auto read_moe_block_data = [&](int block_id) {
      block_num_valid_tokens = moe_block_size;
      for (int i = 0; i < moe_block_size / 4; i++) {
        int4 sorted_token_ids_int4 = reinterpret_cast<const int4*>(
            sorted_token_ids_ptr)[block_id * moe_block_size / 4 + i];
        int* sorted_token_ids = reinterpret_cast<int*>(&sorted_token_ids_int4);
        for (int j = 0; j < 4; j++) {
          if (sorted_token_ids[j] >= prob_m * top_k) {
            block_num_valid_tokens = i * 4 + j;
            break;
          }
        }
        if (block_num_valid_tokens != moe_block_size) break;
      }
  
      __syncthreads();
      int tid4 = threadIdx.x / 4;
      if (threadIdx.x % 4 == 0 && threadIdx.x < block_num_valid_tokens) {
        sh_block_sorted_ids_int4[tid4] = reinterpret_cast<const int4*>(
            sorted_token_ids_ptr)[block_id * moe_block_size / 4 + tid4];
        // 先将 float 转为 __half 然后 将其复制两次 组成 __half2
        if (mul_topk_weights) {
          for (int i = 0; (tid4 * 4 + i) < block_num_valid_tokens; i++) {
            sh_block_topk_weights[tid4 * 4 + i] =
                Dtype::num2num2g(Dtype::float2num(
                    topk_weights_ptr[sh_block_sorted_ids[tid4 * 4 + i]]));
          }
        }
      }
      __syncthreads();
    };
  
    // int scales_offset_int4 = 0;
    // int zeros_offset_int4 = 0;
    int scales_zeros_offset_int4 = 0;
    // 计算 B_expert_off ： 当前专家权重的全局偏移,指向 该专家权重的 （0, 0）处
    // scales_ptr 和 zp_ptr 指针指向当前的起始位置
    // 更新 sh_block_sorted_ids_int4、sh_block_topk_weights 中的值
    auto update_next_moe_block_data = [&]() {
      if (par_id >= parallel) return;
  
      old_expert_id = expert_id;

      block_id = par_id;
      expert_id = expert_ids_ptr[block_id];
  
      B_expert_off = expert_id * prob_n * prob_k / (pack_factor * 4);
      scales_zeros_offset_int4 += (expert_id - old_expert_id) * scales_expert_stride;
      scales_zeros_offset_int4 += (expert_id - old_expert_id) * zp_expert_stride;
      
      read_moe_block_data(block_id);
    };

    // A sizes/strides
    int a_gl_stride = prob_k / 8; // int4 7168(fp16) --> 896(int4)
    constexpr int a_sh_stride = 16 * thread_k_blocks / 8; // int4 16  8
    constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8; // 全局 load时, A矩阵 横向的偏移量
    int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o); // 全局load时， A矩阵 竖向的偏移量
  
    constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o); // A矩阵写入lds时, 一个数据块内两次load的偏移量
  
    constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks); // A矩阵一个tile(16,128)在lds中的总大小 [16, 128](fp16) -->  [16, 16](int4)
    constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta); // A 矩阵一个tile(16, 128) 写入lds时需要几次store
    // A矩阵两次[16,16]load时的偏移量   64bit
    //    (thread_n_blocks / 8)代表B矩阵[128, 128]的一行需要几个warp处理
    //    ((threads / 64) / (thread_n_blocks / 8)) 代表 一次处理 [128, 128]的 几个[16, 128]的数据块
    //    4 * 是因为一个 [16, 128] 的 B数据块， 对应A中 [x, 16]的数据块，16个fp16转成 64bit的话是 4个64bit
    // constexpr int a_sh_rd_delta_o = 4 * ((threads / 64) / (thread_n_blocks / 8)); // int2 
    constexpr int a_sh_rd_delta_o = 4 * ((threads / 64) / (thread_n_blocks / 8)); // int2  变成横向偏移多少 int2 个数据
    // [16,16]内两次load的偏移量 64bit
    // constexpr int a_sh_rd_delta_i = (a_sh_stride * 2) * 16; // int2  (a_sh_stride * 2) 代表一行有几个 64bit的数据, 
    constexpr int a_sh_rd_delta_i = 16; // int2  变成 竖向偏移多少行数据
    
    // B sizes/strides
    int b_gl_stride = 16 * prob_n / (pack_factor * 4); // 256 int4 
    constexpr int b_sh_stride = ((thread_n_blocks * 16) * 16 / pack_factor) / 4;  // 64  128

    constexpr int b_sh_stride_threads = b_sh_stride; // 64  128
  
    int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks; // 256 * 8   b_gl_stride * 4
    int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride_threads); //  b_gl_stride * 2
    constexpr int b_sh_wr_delta = threads; // 256 256
    constexpr int b_sh_rd_delta = threads; // 256 256
    constexpr int b_sh_stage = b_sh_stride * thread_k_blocks; // 64 * 8 = 512  128 * 4 = 512
    constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta; // 2  2

    // 合并后的 scale zero矩阵
    int s_zp_gl_stride = prob_n / 8 + (prob_n / pack_factor) / 4;
    int s_zp_sh_stride = 16 * thread_n_blocks / 8 +  ((16 * thread_n_blocks) / pack_factor) / 4;
    int s_zp_tb_groups = group_blocks != -1 && group_blocks < thread_k_blocks
                          ? thread_k_blocks / group_blocks
                          : 1;
    int s_zp_sh_stage = s_zp_tb_groups * s_zp_sh_stride;
    int s_zp_gl_rd_delta = s_zp_gl_stride;

    // Global A read index of current thread. 64bit
    int row = threadIdx.x / a_gl_rd_delta_o;
    int col = threadIdx.x % a_gl_rd_delta_o;
    // 执行异或列坐标的变换 ,每8行为一个整体
    col = col ^ (row % 8);

    int a_gl_rd_row = row;
    int a_gl_rd_col = col;
    a_gl_rd_col = a_gl_rd_col + a_gl_rd_delta_o * slice_row;
  
    int a_sh_rd_row = (warp_idx) % 16;
    int a_sh_rd_col = warp_idx / 16 + 4 * ((threadIdx.x / 64) / (thread_n_blocks / 8));

    // 256 * 3 + 63 = 831
    int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride_threads) +
                  (threadIdx.x % b_sh_stride_threads);
    b_gl_rd += b_sh_stride * slice_col;
    b_gl_rd += b_gl_rd_delta_o * slice_row;
    int b_sh_rd = threadIdx.x;
    
    int s_zp_gl_rd;
    if(group_blocks == -1) {
      s_zp_gl_rd = s_zp_sh_stride * slice_col + threadIdx.x;
    } else {  //走这里
      s_zp_gl_rd = s_zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                s_zp_sh_stride * slice_col + threadIdx.x;
    }

    // 每个线程 load 128bit 的数据
    int s_zp_sh_wr = threadIdx.x;
    bool s_zp_sh_wr_pred = threadIdx.x < s_zp_sh_stride; // scale 矩阵仅有 s_sh_stride 个线程执行 load 指令就可以了

    // 由于需要处理 [16, 128]duiyin
    // 预处理中已将 8个fp16 warp_idx: threadIdx.x % 64;
    int s_sh_rd = 20 * ((threadIdx.x / 64) % (thread_n_blocks / 8)) +
                (warp_idx) % 16;

    // 
    int zp_sh_rd = 80 * ((threadIdx.x / 64) % (thread_n_blocks / 8)) + 64 +
                  (warp_idx % 16);
    
    int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
    for (int i = 0; i < b_sh_wr_iters; i++) {
      for (int j = 0; j < thread_m_blocks; j++){

        int row = a_sh_rd_row + a_sh_rd_delta_i * j;
        int col = a_sh_rd_col + a_sh_rd_delta_o * i;
        // 异或变换
          // (col / 2) ^ row : 以int4为单个数据进行 异或变换，获取变换后的int4 列索引
          // * 2 ： 转为 int2为单个数据的列索引 
          // + col % 2： 加上在int4数据中的偏移
        int trans_col = ((col / 2) ^ (row % 8)) * 2 + col % 2;
        // int trans_col = col;
        a_sh_rd_trans[i][j] = row * (a_sh_stride * 2 ) + trans_col;
      }
    }
  
    // 从B矩阵中 load数据时的偏移 int4
    int B_idx_int[b_sh_wr_iters];
    for (int i = 0; i < b_sh_wr_iters; i++)
      B_idx_int[i] = b_gl_rd_delta_i * i + b_gl_rd;
    
    // Shared memory storage for global fetch pipelines.
    int4* sh_a = sh_new;
    // 交叉放置 scale 和 zero, 先放置 128个 scale（fp16），在放置 128个zero（uint4）
    int4* sh_s_zp = sh_a + (stages * a_sh_stage);

    // 用来进行 block内的规约
    int4* sh_red = sh_a;
  
    // Register storage for double buffer of shared memory reads.
    // 2 代表双缓冲， thread_m_blocks 代表m维度迭代几次就需要 load 几次左矩阵的数据到寄存器中
    // [16, 16] 大小的A矩阵分到64个线程中，每个线程只持有4个fp16数据
    FragA frag_a[2][thread_m_blocks]; // Vec<__half2, 2>

    // 直接load到寄存器中
    I4 frag_b_quant[stages][b_sh_wr_iters]; // I4 = Vec<int, 4>;
    
    FragC frag_c[thread_m_blocks][4][2]; // FragC: Vec<float, 4>
                                         // [4] 代表 [16, 128] 中第几个 [16, 32] 
                                         // [2] 代表 [16, 32] 中两次 mma的 哪一次 mma
    FragS frag_s[2][4];  // FragS = Vec<__half2, 1>
                          //  [2] 代表双缓冲的含义
                         // [4] 代表 [16, 128]中第几个 [16, 32]的计算， 一个[16, 32] 中每个线程需要2个fp16的 scale数据(只有col 坐标不同时，才是不同的 scale值)
    int frag_qzp[2];  // Zero-points
    FragZP frag_zp;        // Vec<__half2, 4>
                          //   Zero-points in fp16   8个 half2数据

    // Zero accumulators.
    auto zero_accums = [&]() {
      for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
        reinterpret_cast<float*>(frag_c)[i] = 0;
    };

    // 根据 slice_col_par 、slice_row
    //      更新 slice_iters
    //      更新 slice_count、slice_idx
    //      slice_col、par_id
    //      B_expert_off
    //      scales_ptr 和 zp_ptr 指针指向当前的起始位置
    //      sh_block_sorted_ids_int4、sh_block_topk_weights
    //      新增 获取当前条带中 每个线程 发射 buffer_load_dwordx4的指令条数
    
    auto init_slice = [&](bool first_init = false) {
      slice_iters = // 当前 block还需 处理的总 [128, 128] 数据块数   468 - 56 = 412
          iters * (blcok_idx_x + 1) - (k_tiles * slice_col_par + slice_row);
      // 不符合要求的 slice_iters 置为零
      if (slice_iters < 0 || slice_col_par >= n_tiles * parallel) slice_iters = 0;
      if (slice_iters == 0) return;
      // 将 slice_iters 修正为 当前条带中需要处理 [128, 128] 的数据块数
      if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;   //修正为56
      slice_count = 1;
      slice_idx = 0;
  
      // 分为几种情况：
          // 第一个条带：col_first 代表之前的blcok 已经计算的数据块,  k_tiles * (slice_col_par + 1) 代表做完这个条带后已经计算的数据块
                  // 此时需要更新 slice_count、 slice_idx， 因为当前条带可能由两个及以上的block同时计算
          // 中间的条带：col_first代表该blcok全部执行完后 已经计算的数据块, k_tiles * (slice_col_par + 1) 仍代表做完这个条带后已经计算的数据块
                  // 此时 slice_count=1 slice_idx=0
          // 最后一个条带：col_first代表该blcok全部执行完后 已经计算的数据块, k_tiles * (slice_col_par + 1) 仍代表做完这个条带后已经计算的数据块
                  // 此时需要更新 slice_count、 slice_idx， 因为当前条带可能由两个及以上的block同时计算
      int col_first = iters * div_ceil(k_tiles * slice_col_par, iters);   // 468 * (56*1， 468) = 468
      if (col_first <= k_tiles * (slice_col_par + 1)) { // 第一个条带 和 最后一个条带会修正 slice_count 和 slice_idx的值  56*2=112
        int col_off = col_first - k_tiles * slice_col_par; // 列方向 向下偏移多少 数据块
        slice_count = div_ceil(k_tiles - col_off, iters);
        if (col_off > 0) slice_count++;
        // 第一个条带（delta_first == 0）：
              // col_off == 0 说明该条带只由一个block处理， 则 slice_idx = slice_count - 1; slice_idx =0
              // col_off ！= 0 多个block同时处理， 
                    // slice_idx = slice_count - 1 - delta_first / iters;
                    // slice_idx--;
        int delta_first = iters * blcok_idx_x - col_first;
        if (delta_first < 0 || (col_off == 0 && delta_first == 0)) // delta_first == 0 代表第一个条带，因此
          slice_idx = slice_count - 1;
        else {
          slice_idx = slice_count - 1 - delta_first / iters;
          if (col_off > 0) slice_idx--;
        }
      }

      if (parallel * n_tiles >= gridx) {
        if (slice_count > 1 && slice_idx == slice_count - 1) {
          locks_off++;
        }
      } else {
        locks_off++;
      }

      if (first_init && slice_count > 1 && slice_idx == 0) {
        constexpr int threads_per_m = 16 * thread_n_blocks / 8;
        int m_per_thread =
            div_ceil(block_num_valid_tokens, threads / threads_per_m);
        for (int i = 0; i < m_per_thread; i++) {
          int row = threads / threads_per_m * i + threadIdx.x / threads_per_m;
          if (row < block_num_valid_tokens) {
            int64_t sorted_row = sh_block_sorted_ids[row];
            int col = slice_col * 16 * thread_n_blocks / 8 +
                      threadIdx.x % threads_per_m;
            C[sorted_row * prob_n / 8 + col] = {0, 0, 0, 0};
          }
        }
        // After write zero to output, write a negative value to lock.
        // Every SM that processes the same slice would wait for
        // the negative value, and then atomicAdd 1 to it.
        // After all SMs are processed, the lock value would back to 0 again.
        __syncthreads();
        if (threadIdx.x == 0) locks[locks_off] = 1 - slice_count;
      }

      if (slice_col == n_tiles) {
        slice_col = 0;
        par_id++;
        update_next_moe_block_data();
      }
    };

    update_next_moe_block_data();
    init_slice(true);

    // 增加是否load a 的标志位，用于 缓存区长度大于等于K维度长度时，后续的slice 可以复用缓存区中的A矩阵数据，无需再次 load 
    bool should_load_a = true;
    // load a:[16, 128] b:[128, 128] 到lds中
    int warp_a_idx = a_sh_stride * ((threadIdx.x / 64) * (64 / a_gl_rd_delta_o));
    int warp_b_idx = (threadIdx.x / 64) * 64;
    auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true, bool scale_first_load=false) {
      if (pred){
        // load [16, 128](fp16) 
        int4* sh_a_stage = sh_a + a_sh_stage * pipe; 
        for (int i = 0; i < a_sh_wr_iters; i++) {
          int row = a_gl_rd_row + (threads / a_gl_rd_delta_o) * i;
          int col = a_gl_rd_col + a_gl_rd_delta_o * a_off;

          int sorted_row = sh_block_sorted_ids[row] / top_k; // 213 / 8 --> 26
          // 获取A矩阵的实际偏移 128bit  
          int true_idx = 16 * (sorted_row * a_gl_stride + col);
          sh_a_stage = sh_a_stage + threads * i; // 行索引已经提前

          // 通过 read_a 判断是否进行 load
          bool read_a = (row < block_num_valid_tokens);
          cp_async4_pred(&sh_a_stage[warp_a_idx], &A[0], true_idx, read_a);

          // cp_async4_pred_a(&sh_a_stage[warp_a_idx], prob_m, prob_k, &A[0], true_idx);
        }

        // load [128, 128](fp16)-》 [8, 64](int4)
        for (int i = 0; i < b_sh_wr_iters; i++) {
          // 移动到0号线程的起始地址
          int b_global_idx = 16 * (B_idx_int[i] + B_expert_off);

          // buffer_load_dwordx4 嵌入汇编
          cp_async4_vgpr(&frag_b_quant[pipe][i], &B[0], b_global_idx);
          // 直接load到寄存器上
          // frag_b_quant[pipe][i] = reinterpret_cast<const I4*>(B)[b_global_idx / 16];

          // 移动到下一个数据块上
          B_idx_int[i] += b_gl_rd_delta_o;
        }

        int4* sh_s_zp_stage = sh_s_zp + s_zp_sh_stage * pipe;
        if constexpr (group_blocks >= thread_k_blocks){
          if (scale_first_load || ((slice_scale_row % (group_blocks / thread_k_blocks))==0)){ // 每次都 load scale 和 zero
            cp_async4_pred(&sh_s_zp_stage[0], &scales_zeros_ptr[0], (s_zp_gl_rd + scales_zeros_offset_int4) * 16, s_zp_sh_wr_pred);
            s_zp_gl_rd += s_zp_gl_rd_delta; // row 方向向下平移一行，所以隐含 1*s_gl_rd_delta
          }else{
            // 如果是复用，为了适配新的流水，也要重新load scale和zero，这样做的目的是因为新的流水下，scale和zero的数据会被覆盖
            // 这里记得需要减去 上一个load过程中加上的 s_zp_gl_rd_delta
            cp_async4_pred(&sh_s_zp_stage[0], &scales_zeros_ptr[0], (s_zp_gl_rd - s_zp_gl_rd_delta + scales_zeros_offset_int4) * 16, s_zp_sh_wr_pred);
          }
          slice_scale_row ++;
        } else {
          for (int i = 0; i < s_zp_tb_groups; i++) { // 迭代需要 load 多少行 scale\zero 矩阵的数据到lds中
            cp_async4_pred(&sh_s_zp_stage[i * s_zp_sh_stride], &scales_zeros_ptr[0], (s_zp_gl_rd + scales_zeros_offset_int4) * 16, s_zp_sh_wr_pred);
            s_zp_gl_rd += s_zp_gl_rd_delta; // row 方向向下平移一行，所以隐含 1*s_gl_rd_delta
          }
        }
      }
    };
    
    auto fetch_to_registers = [&](int k, int pipe) {
      // 先以 int4的步长偏移到 [16, 128](fp16)--》[16, 16](int4) 大小的位置处
      // 再强转成指向 FragA: Vec<__half2, 2>
      int4* sh_a_stage = sh_a + a_sh_stage*pipe;
      // __syncthreads(); //加在此处可以解决
      FragA* fraga_ptr = reinterpret_cast<FragA*>(sh_a_stage);
      // 一个warp load [16, 16]大小的数据到每个线程寄存器中，每个线程拥有4个fp16数据，每两个fp16数据存放在一个half2的数据结构中
      // __syncthreads(); //加在此处可以解决
      for (int i = 0; i < thread_m_blocks; i++){
        // 读取预先计算的 偏移 然后去读取
        frag_a[k % 2][i] = fraga_ptr[a_sh_rd_trans[k % b_sh_wr_iters][i]];
      }
    };

    constexpr int tb_k = 16 * thread_k_blocks;
    constexpr int k_iter_size = tb_k / b_sh_wr_iters; // warp在B矩阵上的一次迭代 跨越的K维度行数 64
    auto fetch_scales_zp_to_registers = [&](int k, // 当前是数据块中线程第几次load数据
                                        int full_pipe // 当前处理的是第几个数据块
                                      ) {
      // 首选确定 确定现在要取的是 LDS 中的哪一块数据
      int pipe = full_pipe % stages;
      if constexpr (group_blocks >= thread_k_blocks) {
        // 由于只有两块 数据存储区，因此每次都需要从自己的数据区中取 scale zero
        int4* sh_s_zp_stage = sh_s_zp + s_zp_sh_stage * pipe;
        // load scale 矩阵
        reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_zp_stage[s_sh_rd];
        // load zero 矩阵
        frag_qzp[k % 2] = (reinterpret_cast<int*>(sh_s_zp_stage))[zp_sh_rd];
      } else { // LDS中，有多行scale和zero 数据
        int warp_id = threadIdx.x / 64; // 确定当前线程在哪个 warp 中   0， 1, 2, 3        
        int n_warps = thread_n_blocks / 8;  // 确定 N维度放置了几个 warp    1

        int warp_row = warp_id / n_warps; // 当前 warp 位于warp layout的哪一行 0，1，2，3
 
        int cur_k = warp_row * 16; // 第一次迭代时 K维度的索引  0， 16， 32， 48
        cur_k += k_iter_size * (k % b_sh_wr_iters); //  加上数据块中第几次计算带来的 K维度偏移

        int cur_group_id =  cur_k / 16 / group_blocks; // 获取当前warp处理的数据位于第几行scalezero数据中

        int4* sh_s_zp_stage = sh_s_zp + s_zp_sh_stage * pipe + cur_group_id * s_zp_sh_stride;
        // load scale
        reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_zp_stage[s_sh_rd];
        // load zero
        frag_qzp[k % 2] = (reinterpret_cast<int*>(sh_s_zp_stage))[zp_sh_rd];
      }
    };

    auto get_frag_zp = [&](int k) {
      bool pred = (k == 0) || (group_blocks < thread_k_blocks);
      if (pred){
        int k2 = k % 2;
        // 反量化 zero矩阵 - 对于bf16版本，也使用fp16进行反量化以复用现有逻辑
        // if (is_new_zp) { // 仅在 k==0 时执行一次， 因为 两次[16, 128]计算用的同一个 scale和zero数据
        int zp_quant_0; // int32 
        int zp_quant_1; // int32

        zp_quant_0 = frag_qzp[k2];
        zp_quant_1 = zp_quant_0 >> 8;

        // 直接反量化到frag_zp中，复用同一个寄存器
        typename ScalarType<__half>::FragB* frag_zp_ptr = reinterpret_cast<typename ScalarType<__half>::FragB*>(&frag_zp[0]);
        dequantize_int4<__half>(zp_quant_0, frag_zp_ptr[0]);
        dequantize_int4<__half>(zp_quant_1, frag_zp_ptr[1]);
        
        // zero*scale - 都使用fp16运算，因为scale和zero都是fp16
        for (int j = 0; j < 4; j++) { 
          using scalar_t2 = typename ScalarType<__half>::scalar_t2;
          frag_zp[j] = __hmul2(frag_zp[j], *reinterpret_cast<scalar_t2*>(&frag_s[k2][j]));
        }
      }
    };

    // 左矩阵：[16, 16]（fp16）   右矩阵：[16, 128]（fp16）
    auto matmul = [&](int k, int b_com_pipe) { // 0, 1 代表当前在第几个 [16, 128]块上执行计算
      int k2 = k % 2; // 0, 1
      // zero矩阵反量化后 frag_zp: Vec<__half2, 4>
      for (int j = 0; j < 4; j++) { // 在 [16, 128]的矩阵块上执行4次 [16*32]大小数据块的计算
        FragB frag_b0; // 4个fp16 Vec<__half2, 2> 第一次mac zero 矩阵数据
        FragB frag_b1; // 4个fp16 Vec<__half2, 2> 第一次mac zero 矩阵数据
        int b_quant_0, b_quant_1;
        
        // 取 第 j 次 [16, 32]计算中8个uint4 B矩阵 pack成int32的数据
        b_quant_0 = frag_b_quant[b_com_pipe][k2][j];
        b_quant_1 = b_quant_0 >> 8;

        // 将8个uint4的 B 矩阵数据 反量化成 8个fp16
        // frag_b0: 存储前4个fp16数据，用于第一次mac Vec<half2, 2>
        // frag_b1: 存储后4个fp16数据，用于第二次mac Vec<half2, 2>
        // dequant<scalar_t, w_type.size_bits()>(b_quant_0, frag_b0);
        // dequant<scalar_t, w_type.size_bits()>(b_quant_1, frag_b1);
        dequantize_int4<scalar_t>(b_quant_0, frag_b0);
        dequantize_int4<scalar_t>(b_quant_1, frag_b1);
        // b *scale - zero*scale
        // 将b 还原成 fp16                  
        scale_and_sub<scalar_t>(frag_b0, frag_s[k2][j][0].x, frag_zp[j].x);
        scale_and_sub<scalar_t>(frag_b1, frag_s[k2][j][0].y, frag_zp[j].y);

        for (int i = 0; i < thread_m_blocks; i++) {
          mmac_volatile<scalar_t>(frag_a[k2][i], frag_b0, frag_c[i][j][0]);
          mmac_volatile<scalar_t>(frag_a[k2][i], frag_b1, frag_c[i][j][1]);
        }
      }
    };

    auto thread_block_reduce = [&]() {
      constexpr int red_off = threads / b_sh_stride_threads / 2; // k维度一共有几个并行执行的warp 2
      if (red_off >= 1) { // 如果K方向只有一个 warp，则其结果就是写回的最终结果，否则的话 需要进行多个warp间结果规约
        // 当前线程 在  256 个线程 layout的第几行
        int red_idx = threadIdx.x / b_sh_stride_threads;  // 当前warp是规约的 第几行  0, 1
        // 以 int4类型为基准, [16, 32]中有两个int4类型数据， [16, 128] 大小的数据块中有 8个int4类型数据
        constexpr int red_sh_stride = b_sh_stride_threads * 4 * 2; // 每一行有多少个 float4的数据需要规约  128 * 4 *2
        constexpr int red_sh_delta = b_sh_stride_threads; // 128
        int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) +
                        (threadIdx.x % b_sh_stride_threads);
        
        for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
          // 先执行此循环
          for (int i = red_off; i > 0; i /= 2) { // 1 执行一次
            if (i <= red_idx && red_idx < 2 * i) { // warp 2 3 执行操作
              // 逐个处理 8个[16, 32]的结果
              for (int j = 0; j < 4 * 2; j += 1) {
                int red_sh_wr =
                    red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
                if (i < red_off) {
                  // 将 int4 类型强制转换成 float 类型，
                  float* c_rd = reinterpret_cast<float*>(
                      &sh_red[red_sh_delta * j + red_sh_rd]);
                  // 将 int4 类型强制转换成 float 类型，
                  float* c_wr = reinterpret_cast<float*>(&sh_red[red_sh_wr]);
                  for (int k = 0; k < 4; k++)
                    reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] +=
                        c_rd[k] + c_wr[k];
                }
                // // 根据当前所处的32bank的行，进行对应的数据变换 
                // FragC frag_c_temp; // FragC: Vec<float, 4>
                // int bank_row = red_sh_wr / 8 % 4;
                // int bank_col = red_sh_wr % 8;

                // frag_c_temp[(bank_row) % 4] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j].x;
                // frag_c_temp[(bank_row + 1) % 4] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j].y;
                // frag_c_temp[(bank_row + 2) % 4] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j].z;
                // frag_c_temp[(bank_row + 3) % 4] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j].w;

                //warp 2, 3 将对应的 fragc 中的结果数据写入 warp0 和 warp1 的lds空间上
                sh_red[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
              }
            }
            __syncthreads();
          }
          // warp 0 warp1 先读取 red_sh_rd 位置的数据，然后加到 自己持有的frag_c 寄存器中
          if (red_idx == 0) {
            for (int i = 0; i < 4 * 2; i += 1) { 
              // 将 int4 类型强制转换成 float 类型
              float* c_rd =
                  reinterpret_cast<float*>(&sh_red[red_sh_delta * i + red_sh_rd]);
              for (int j = 0; j < 4; j++)
                reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
            }
          }
          __syncthreads();
        }
      }
    };

    // 以正确的布局写出还原的最终结果：
    //      1、warp0 中64个线程将各自 fragc 中的数据，写到 LDS 的对应位置（按照结果矩阵的实际数据位置进行放置）
    //      2、lds-->global:
    //          (1). 如果该条带中只有一个block执行运算：warp 0-3 共256个线程，每个线程写回 128bit的数据到全局内存中的结果矩阵。
    //         （2). 多个block共同完成一个slice 的计算，则使用 atomic_add 进行结果回写。
    auto write_result = [&]() {
      // 每 8个 fp16的数据以一个int4的类型进行store和load, 因此 “/8”
      int c_gl_stride = prob_n / 8; // 64
      // int4类型的偏移, 一个thread_n_blocks 说明横向有16个fp16的数据, 也意味着有 2个int4
      // 实际上 thread_n_blocks*16 /8 -->  2 * thread_n_blocks
      constexpr int c_sh_stride = 2 * thread_n_blocks + 1; // 32个int4
      // 一个thread_n_blocks 里面有 16个fp16的数据，需要2个线程写入一个才能写入完成
      int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks)); // c_gl_stride * （256 / 32） = c_gl_stride * 8
      constexpr int c_sh_rd_delta =
          c_sh_stride * (threads / (2 * thread_n_blocks));  // c_sh_stride * 8

      auto write = [&](int idx, float c_float) { // 在 sh_red 的 idx 索引处插入 float转成的 fp16/bf16
        // // 进行异或变换, 一个 swizzle的组包含8行数据
        // int row = idx / (8 * c_sh_stride);
        // int col = idx % (8 * c_sh_stride);

        // col = ((row % 8) ^ (col / 8)) * 8 + col % 8;

        // idx = row * (8 * c_sh_stride) + col;
        ((scalar_t*)sh_red)[idx] = Dtype::float2num(c_float);
      };
      
      // 遍历 多个thread_m_blocks, 128个线程将自己 frag_c寄存器中的数据写入到lds中
      for (int b_blocks_idx=0;b_blocks_idx<thread_m_blocks;b_blocks_idx++){
        // 初始化写回的参数
        int c_gl_wr = c_gl_stride * (b_blocks_idx * 16 + (threadIdx.x / (2 * thread_n_blocks))) +
                    (threadIdx.x % (2 * thread_n_blocks));
        c_gl_wr += (2 * thread_n_blocks) * slice_col;
        
        int c_sh_wr;
        // c_=sh_stride 是以 int4 为步长进行计算的 
        // c_sh_wr 是以 fp16 为基础单位，因此 8 * c_sh_stride
        c_sh_wr = // 64个线程第一次写入 lds的偏移量（fp16）
            (8 * c_sh_stride) * (warp_idx % 16) + warp_idx / 16;
        c_sh_wr += 128 * (threadIdx.x / 64);
  
        int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                            (threadIdx.x % (2 * thread_n_blocks));

        // 处理 B矩阵第一行数据的warp将 结果寄存器中的数据写入 lds中
        if (threadIdx.x / 64 < thread_n_blocks / 8) {
          // 将 frag_c中的结果写入到 [16, 128]的 LDS中
          for (int j = 0; j < 4; j++) { // 写入 [16, 128] 中 第 j个 [16, 32]的结果矩阵中
            // 以 float 为偏移单位
            int wr = c_sh_wr + 32 * j;

            write(wr + 0, frag_c[b_blocks_idx][j][0][0]);
            write(wr + 4, frag_c[b_blocks_idx][j][0][1]);

            write(wr + 8, frag_c[b_blocks_idx][j][0][2]);
            write(wr + 12, frag_c[b_blocks_idx][j][0][3]);

            write(wr + 16, frag_c[b_blocks_idx][j][1][0]);
            write(wr + 20, frag_c[b_blocks_idx][j][1][1]);

            write(wr + 24, frag_c[b_blocks_idx][j][1][2]);
            write(wr + 28, frag_c[b_blocks_idx][j][1][3]);
          }
        }
        __syncthreads();
        // 256个线程将lds中的数据进行写回
        for (int i = 0; // div_ceil(32, 8) 
              i < div_ceil(16, threads / (2 * thread_n_blocks));
              i++) { 
          int row = c_gl_wr / c_gl_stride;
          if (row < block_num_valid_tokens) { // 仅有效的数据才进行写入
            // int sw_row = row % 8;
            // int col = c_gl_wr % c_gl_stride;
            // col = sw_row ^ col;

            int col = c_gl_wr % c_gl_stride;
            int64_t sorted_row = sh_block_sorted_ids[row];
            int64_t true_idx = sorted_row * c_gl_stride + col; //int4
            // 获取 topk_weight 中的值
            scalar_t2 topk_weight_score;
            if (mul_topk_weights) topk_weight_score = sh_block_topk_weights[row];

            if (slice_count > 1 || mul_topk_weights) {
              scalar_t2* C_half2 = reinterpret_cast<scalar_t2*>(&C[true_idx]); // 指向 __half2
              scalar_t2* sh_red_half2 =
                  reinterpret_cast<scalar_t2*>(&sh_red[c_sh_rd]);
              for (int a = 0; a < 4; a++) { // 逐次原子加 4 次, 每次 __half2 即32bit
                scalar_t2 res = sh_red_half2[a];
                if (mul_topk_weights) {
                  res = mul2(res, topk_weight_score);
                }
                if (slice_count > 1) {
                  atomic_add_generic<scalar_t2>(&C_half2[a], res);
                } else {
                  C_half2[a] = res;
                };
              }
            } else { // 直接以 int4(128bit) 进行写回
              C[true_idx] = sh_red[c_sh_rd];
            }
            c_gl_wr += c_gl_wr_delta;
            c_sh_rd += c_sh_rd_delta;
          }
        }
        __syncthreads();
      }
    }; // 完成结果回写

    // 启动 全局预取 和 寄存器流水
    auto start_pipes = [&]() {
      // 发射三组global --》 lds的指令
      for (int i = 0; i < stages - 1; i++) { // 0, 1, 2
        fetch_to_shared(i, i, i < slice_iters, slice_scale_row==slice_row);
      }
      zero_accums(); // 结果寄存器 重置为 0 
      // cp_async_wait_0(); // 由于只有 两块缓冲区，因此等待第一块数据缓冲区中的数据全部load完成才能进入循环计算
      // wait_for_stage(); // 等待 一组 global--> lds 的load指令执行完成

      a_gl_rd_col += a_gl_rd_delta_o * (stages - 1);
    
    };

    slice_scale_row = slice_row;
    // __syncthreads();
    if (slice_iters) {
      start_pipes();
    }

    while (slice_iters) { // 处理第一个条带的数据

      for (int pipe = 0; pipe < stages;pipe++) { // 遍历 4 段流水
        cp_async_wait_0(); 
        __syncthreads();
        for (int k = 0; k < b_sh_wr_iters; k++) { // b: [128, 128](fp16) 数据块上，每个warp执行 b_sh_wr_iters（2）次[16, 128]大小数据块的计算
          // lds --> vgpr 


          // 发射下一个数据块的全局 load指令
          if (k == b_sh_wr_iters - 2) {
            fetch_to_registers(0, pipe);
            fetch_scales_zp_to_registers(0, pipe);
            fetch_to_registers(1, pipe);
            fetch_scales_zp_to_registers(1, pipe);
            // 加载下一个数据块的
            fetch_to_shared((pipe + stages - 1) % stages, pipe,
                            slice_iters >= stages, slice_scale_row==slice_row);
          }
          // A: 始终和 [16, 16] 大小的数据块进行计算
          // B: 在[16, 128]大小的数据块上遍历4次, 每次计算 [16, 32]大小的数据块，包含两次 mmac指令
          get_frag_zp(k);
          matmul(k, pipe);
        }
        slice_iters--;
        if (slice_iters == 0) {
          break;
        }
      }
      a_gl_rd_col += a_gl_rd_delta_o * stages;

      if (slice_iters == 0) {
        cp_async_wait_0();
        __syncthreads();

        thread_block_reduce();
        
        if (slice_count > 1 && slice_idx != 0)
          wait_negative_and_add(&locks[locks_off]);

        write_result();

        int old_slice_row = slice_row;
        // 转到后一个条带中执行计算
        slice_row = 0;
        slice_col_par++;
        slice_col++;

        init_slice(); // 初始化 slice_row、slice_col
      
        if (slice_iters) {
          
          // 更新a 矩阵新的索引地址
          // a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
          //           (threadIdx.x % a_gl_rd_delta_o);
          a_gl_rd_row = row;
          a_gl_rd_col = col;

          // 更新B 矩阵新的索引地址 先向右偏移 b_sh_stride 个 int4步长, 再向上偏移 整个条带的步长
          for (int i = 0; i < b_sh_wr_iters; i++)
            B_idx_int[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
          if (slice_col == 0) { // 如果转到下一个专家进行处理, 则整体左移 b_gl_stride 步长，相当于回到专家矩阵左上角第一个数据块
            for (int i = 0; i < b_sh_wr_iters; i++) B_idx_int[i] -= b_gl_stride;
          }

          // 更新 scale、zero 的全局访问索引 slice_row =0
          s_zp_gl_rd = s_zp_sh_stride * slice_col + threadIdx.x;
       
          // 启动下一阶段的流水
          slice_scale_row = slice_row;
          start_pipes(); 
        }
      }
    }
  }

}
}
}
