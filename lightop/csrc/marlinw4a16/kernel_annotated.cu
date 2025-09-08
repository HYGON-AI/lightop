/*
 * Marlin W4A16 MoE GEMM Kernel - 详细注释版本
 * 
 * 这是一个高度优化的混合专家(MoE)模型的矩阵乘法内核，专门用于：
 * - 权重量化：4位整数(W4) 
 * - 激活精度：16位半精度(A16)
 * - 支持混合专家架构
 * 
 * 核心优化技术：
 * 1. Tensor Core加速
 * 2. 流水线预取(Pipeline Prefetch)
 * 3. 共享内存优化
 * 4. 块级并行处理
 * 5. 专家权重动态加载
 */

#include "marlin.h"       // Marlin核心定义和常量
#include "marlin_dtypes.h" // 数据类型特化定义

// 定义向量化数据类型，利用GPU的向量指令
using half_t  = _Float16;  // 单个半精度浮点数
using half2_t = __attribute__((__vector_size__(2 * sizeof(half_t)))) half_t;  // 2个半精度向量
using half4_t = __attribute__((__vector_size__(4 * sizeof(half_t)))) half_t;  // 4个半精度向量  
using float4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;   // 4个单精度向量

namespace at {
namespace native {
namespace marlin_moe_w4a16{

  /*
   * 4位整数到FP16的反量化函数
   * 
   * 输入：q - 打包的4位整数数据
   * 输出：output - 反量化后的FP16数据片段
   * 
   * 算法原理：
   * 1. 使用位操作提取4位数据
   * 2. 通过FP16魔法数快速转换为浮点格式
   * 3. 应用线性变换完成反量化
   */
  __device__ void dequantize_int4_to_fp16(int q, typename ScalarType<__half>::FragB& output) {
    // === 位掩码和魔法数定义 ===
    int LO_MASK = 0x000f000f;    // 低位掩码：提取第0和第16位的4位数据 (位模式: 0000 1111 0000 0000 1111)
    int HI_MASK = 0x00f000f0;    // 高位掩码：提取第4和第20位的4位数据 (位模式: 1111 0000 0000 1111 0000)
    int EXP_MAGIC = 0x64006400;  // FP16指数魔法数：快速构造FP16格式的指数部分
    int SUB_MAGIC = 0x64086408;  // 减法魔法数：减去零点偏置值 (等价于减去8.0)
    int MUL_MAGIC = 0x2c002c00;  // 乘法魔法数：缩放因子 (等价于乘以0.0625 = 1/16)
    int ADD_MAGIC = 0xd480d480;  // 加法魔法数：额外偏移量 (等价于减去72.0)

    // === 提取并构造中间FP16表示 ===
    int lo = ((q & LO_MASK) | EXP_MAGIC);  // 提取低4位，组合指数部分
    int hi = ((q & HI_MASK) | EXP_MAGIC);  // 提取高4位，组合指数部分

    // === 执行反量化数学运算 ===
    __half2 lo_val = *reinterpret_cast<__half2*>(&lo);      // 重新解释为half2类型
    __half2 hi_val = *reinterpret_cast<__half2*>(&hi);      // 重新解释为half2类型
    __half2* sub_ptr = reinterpret_cast<__half2*>(&SUB_MAGIC); // 减法操作数指针
    __half2* mul_ptr = reinterpret_cast<__half2*>(&MUL_MAGIC); // 乘法操作数指针
    __half2* add_ptr = reinterpret_cast<__half2*>(&ADD_MAGIC); // 加法操作数指针

    // === 写入输出片段 ===
    output[0] = __hsub2(lo_val, *sub_ptr);                   // 低位数据: lo - 8.0
    output[1] = __hfma2(hi_val, *mul_ptr, *add_ptr);         // 高位数据: (hi * 0.0625) - 72.0
  }

  /*
   * 缩放和零点减法操作
   * 
   * 对反量化后的权重应用缩放因子和零点校正：
   * frag_b = frag_b * scale - zero_point
   * 
   * 这是量化感知训练中的标准反量化公式
   */
  template <typename scalar_t>
  __device__ inline void scale_and_sub(
      typename ScalarType<scalar_t>::FragB& frag_b, scalar_t s, scalar_t zp) {
    
    using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;  // 获取向量化类型(half2)
    
    // 将标量扩展为向量化格式
    scalar_t2 s2 = ScalarType<scalar_t>::num2num2(s);   // scale扩展为half2
    scalar_t2 zp2 = ScalarType<scalar_t>::num2num2(zp); // zero_point扩展为half2
    
    // 执行向量化的缩放和零点减法: frag_b = frag_b * s - zp
    frag_b[0] = __hfma2(frag_b[0], s2, __hneg2(zp2));  // 第一个half2: b[0] * s - zp
    frag_b[1] = __hfma2(frag_b[1], s2, __hneg2(zp2));  // 第二个half2: b[1] * s - zp
  }

  /*
   * 混合精度矩阵乘累加(MMAC)操作
   * 
   * 使用AMD GPU的内置矩阵乘法指令执行：
   * frag_c = frag_a * frag_b + frag_c
   * 
   * 矩阵规模: 16x16x16 (M x N x K)
   */
  template <typename scalar_t>
  __device__ inline void mmac_volatile(const typename ScalarType<scalar_t>::FragA& frag_a,
                            const typename ScalarType<scalar_t>::FragB& frag_b,
                            typename ScalarType<scalar_t>::FragC& frag_c) {
    
    // 使用AMD GPU的内置矩阵乘法指令
    // __builtin_amdgcn_mmac_f32_16x16x16f16: 16x16的half输入，16x16的float输出
    *(reinterpret_cast<float4_t*>(&frag_c)) = __builtin_amdgcn_mmac_f32_16x16x16f16(
        *(reinterpret_cast<const half4_t*>(&frag_a)),  // A矩阵片段(half)
        *(reinterpret_cast<const half4_t*>(&frag_b)),  // B矩阵片段(half) 
        *(reinterpret_cast<float4_t*>(&frag_c))        // C矩阵片段(float)
    );
  }

  /*
   * Marlin MoE W4A16 主要内核函数
   * 
   * 模板参数：
   * - scalar_t: 计算数据类型 (half或__hip_float16)
   * - threads: 线程块大小
   * - thread_m_blocks: M维度的16x16块数量
   * - thread_n_blocks: N维度的16x16块数量  
   * - thread_k_blocks: K维度的16x16块数量
   * - stages: 异步全局->共享内存预取的流水线级数
   * - group_blocks: 连续16x16块的数量(用于分组量化)
   */
  template <typename scalar_t,         // 计算数据类型
          const int threads,           // 线程块中的线程数量
          const int thread_m_blocks,   // M维度(批量大小)的16x16块数量
          const int thread_n_blocks,   // N维度(输出)的16x16块数量
          const int thread_k_blocks,   // K维度(归约)的16x16块数量
          const int stages,            // 异步全局->共享内存预取流水线级数
          const int group_blocks       // 连续16x16块数量，用于分组量化 (通常为4)
          >
  __global__ void Marlin(
    const int4* __restrict__ A,                          // FP16输入矩阵，形状为mxk
    const int4* __restrict__ B,                          // 4位量化权重矩阵，形状为kxn
    int4* __restrict__ C,                                // FP16输出缓冲区，形状为mxn
    const int4* __restrict__ scales_zeros_ptr,           // FP16量化缩放因子和零点
    const int32_t* __restrict__ sorted_token_ids_ptr,    // MoE排序后的token ID
    const int32_t* __restrict__ expert_ids_ptr,          // MoE专家ID数组
    const int32_t* __restrict__ num_tokens_past_padded_ptr, // MoE token数量(填充后)
    const float* __restrict__ topk_weights_ptr,          // MoE top-k权重
    int top_k,                                           // 每个token的专家数量
    bool mul_topk_weights,                               // 是否乘以top-k权重
    int num_groups,                                      // 每个输出通道的缩放组数 (通常为112)
    int prob_m,                                          // 批量维度m
    int prob_n,                                          // 输出维度n  
    int prob_k,                                          // 归约维度k
    int* locks                                           // 用于屏障同步的额外全局存储
  ) {
    // === 类型别名定义 ===
    using Dtype = ScalarType<scalar_t>;                  // 数据类型特化
    using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2; // 向量化类型(half2)
    using FragA = typename ScalarType<scalar_t>::FragA;  // A矩阵片段类型
    using FragB = typename ScalarType<scalar_t>::FragB;  // B矩阵片段类型  
    using FragC = typename ScalarType<scalar_t>::FragC;  // C矩阵片段类型
    using FragS = typename ScalarType<scalar_t>::FragS;  // 缩放因子片段类型
    using FragZP = typename ScalarType<scalar_t>::FragZP; // 零点片段类型

    // === 线程和块索引 ===
    const int blcok_idx_x = blockIdx.x;                 // 当前块的X坐标
    const int warp_idx = threadIdx.x % 64;              // 当前线程在warp中的索引(AMD GPU每个warp 64线程)
    
    extern __shared__ int4 sh[];                        // 共享内存声明
  
    // === 常量定义 ===
    constexpr int pack_factor = 32 / 4;                 // 打包因子: 32位可存储8个4位数据
    constexpr int moe_block_size = 16 * thread_m_blocks; // MoE块大小 = block_m
    
    // === 计算分组大小和专家数据步长 ===
    int group_size = (group_blocks == -1) ? prob_k : prob_k / num_groups; // 分组大小 (128)
    int scales_expert_stride = prob_n * prob_k / group_size / 8;          // 每个专家的缩放因子数据量(int4单位)
    int zp_expert_stride = prob_n * prob_k / group_size / (pack_factor * 4); // 每个专家的零点数据量(int4单位)
    int scales_zp_expert_stride = scales_expert_stride + zp_expert_stride;   // 合并后的步长

    // === MoE并行计算参数 ===
    int num_tokens_past_padded = num_tokens_past_padded_ptr[0]; // 总共选择的token数量
    int parallel = num_tokens_past_padded / moe_block_size;     // 实际参与计算的专家数

    // === 计算数据块划分 ===
    int k_tiles = prob_k / 16 / thread_k_blocks;        // K方向的[128,128]或[64,256]数据块数量
    int n_tiles = prob_n / 16 / thread_n_blocks;        // N方向的[128,128]或[64,256]数据块数量
    int iters = div_ceil(k_tiles * n_tiles * parallel, gridDim.x); // 每个block需要处理的数据块数量
    
    // === 调整迭代次数以适应分组模式 ===
    if (group_blocks >= thread_k_blocks) {               // 如果分组块数大于等于线程K块数
      iters = (group_blocks / thread_k_blocks) *        // 调整迭代次数
              div_ceil(iters, (group_blocks / thread_k_blocks));
    }
  
    // === 数据块索引计算 ===
    int slice_row = (iters * blcok_idx_x) % k_tiles;    // K方向数据块索引
    int slice_col_par = (iters * blcok_idx_x) / k_tiles; // N方向包含parallel的数据块索引
    int slice_col = slice_col_par;                       // N方向不包含parallel的数据块索引
    int slice_iters;                                     // 当前条带中当前block需要处理的数据块个数
    int slice_count = 0;                                 // 当前条带中一共需要几个block执行计算
    int slice_idx;                                       // 当前条带中该block的索引
    int slice_scale_row = 0;                             // 当前scale行索引，用于确定是否加载scale和zero

    // === MoE专家管理变量 ===
    int par_id = 0;                                      // 并行ID
    int block_id = -1;                                   // 块ID
    int64_t expert_id = 0;                               // 专家ID (使用int64避免计算溢出)
    int old_expert_id = 0;                               // 上一个专家ID
    int64_t B_expert_off = 0;                            // B矩阵专家偏移量
  
    // === 共享内存布局 ===
    int4* sh_block_sorted_ids_int4 = sh;                 // 存储排序后的token ID
    int32_t* sh_block_sorted_ids = reinterpret_cast<int*>(sh_block_sorted_ids_int4);
    int4* sh_block_topk_weights_int4 = sh_block_sorted_ids_int4 + moe_block_size / 4;
    scalar_t2* sh_block_topk_weights = reinterpret_cast<scalar_t2*>(sh_block_topk_weights_int4);
    int4* sh_new = sh_block_topk_weights_int4 + moe_block_size / 4;
  
    int32_t block_num_valid_tokens = 0;                  // 块中有效token数量
    int32_t locks_off = 0;                               // 锁偏移量

    // === 修正slice_col和par_id ===
    if (slice_col_par >= n_tiles) {                     // 如果超出N维度边界
      slice_col = slice_col_par % n_tiles;              // 取模得到实际列索引  
      par_id = slice_col_par / n_tiles;                 // 计算并行ID
    }
    
    // === 计算锁偏移量 ===
    if (parallel * n_tiles >= gridDim.x) {              // 当并行数*N瓦片数大于等于SM数时
      locks_off = blockIdx.x;                           // 最多有gridDim.x个冲突瓦片块
    } else {
      locks_off = (iters * blockIdx.x) / k_tiles - 1;   // 否则使用计算得到的偏移
    }
  
    /*
     * MoE块数据读取Lambda函数
     * 
     * 功能：
     * 1. 计算块中有效token数量
     * 2. 加载排序后的token ID到共享内存
     * 3. 加载top-k权重到共享内存(如果需要)
     */
    auto read_moe_block_data = [&](int block_id) {
      block_num_valid_tokens = moe_block_size;           // 初始化为最大值
      
      // === 计算有效token数量 ===
      for (int i = 0; i < moe_block_size / 4; i++) {    // 每次处理4个int32
        int4 sorted_token_ids_int4 = reinterpret_cast<const int4*>(sorted_token_ids_ptr)[block_id * moe_block_size / 4 + i];
        int* sorted_token_ids = reinterpret_cast<int*>(&sorted_token_ids_int4);
        
        for (int j = 0; j < 4; j++) {                    // 检查每个token ID
          if (sorted_token_ids[j] >= prob_m * top_k) {   // 如果超出有效范围
            block_num_valid_tokens = i * 4 + j;          // 记录有效token数量
            break;
          }
        }
        if (block_num_valid_tokens != moe_block_size) break; // 找到边界则退出
      }
  
      __syncthreads();                                   // 同步确保所有线程都计算完成
      // 加载sorted_ids和topk_weight
      int tid4 = threadIdx.x / 4;                        // 4线程为一组
      if (threadIdx.x % 4 == 0 && threadIdx.x < block_num_valid_tokens) { // 每组的第一个线程负责加载
        // === 加载排序后的token ID ===
        sh_block_sorted_ids_int4[tid4] = reinterpret_cast<const int4*>(
            sorted_token_ids_ptr)[block_id * moe_block_size / 4 + tid4];
        
        // === 加载top-k权重(如果启用) ===
        if (mul_topk_weights) {
          for (int i = 0; (tid4 * 4 + i) < block_num_valid_tokens; i++) {
            sh_block_topk_weights[tid4 * 4 + i] =       // 转换float到half2格式
                Dtype::num2num2(Dtype::float2num(
                    topk_weights_ptr[sh_block_sorted_ids[tid4 * 4 + i]]));
          }
        }
      }
      __syncthreads();                                   // 同步确保数据加载完成
    }; // read_moe_block_data定义结束
  
    int scales_zeros_offset_int4 = 0;                    // 缩放因子和零点的偏移量
    
    /*
     * 更新下一个MoE块数据Lambda函数
     * 
     * 功能：
     * 1. 计算当前专家的权重偏移量
     * 2. 更新缩放因子和零点指针
     * 3. 加载新的MoE块数据
     */
    auto update_next_moe_block_data = [&]() {
      if (par_id >= parallel) return;                   // 如果超出并行范围则返回
  
      old_expert_id = expert_id;                         // 保存旧的专家ID

      block_id = par_id;                                 // 设置块ID
      expert_id = expert_ids_ptr[block_id];              // 获取当前专家ID
  
      // === 计算专家权重偏移量 ===
      B_expert_off = expert_id * prob_n * prob_k / (pack_factor * 4); // 专家权重在B矩阵中的偏移
      
      // === 更新缩放因子和零点偏移量 ===
      scales_zeros_offset_int4 += (expert_id - old_expert_id) * scales_expert_stride;  // 缩放因子偏移
      scales_zeros_offset_int4 += (expert_id - old_expert_id) * zp_expert_stride;      // 零点偏移
      
      read_moe_block_data(block_id);                     // 读取MoE块数据
    };

    // === A矩阵访问参数 ===
    int a_gl_stride = prob_k / 8;                        // A矩阵全局步长 (int4单位): 7168(fp16) -> 896(int4)
    constexpr int a_sh_stride = 16 * thread_k_blocks / 8; // A矩阵共享内存步长 (int4单位) block内部的偏移 64/8需要8个线程 8*fp16 = 128
    constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8; // 全局加载时A矩阵横向偏移量 线程偏移量
    int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o); // 全局加载时A矩阵纵向偏移量
  
    // threads = warp*64 = 256
    constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o); // A矩阵写入LDS时的偏移量 ??
    constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);         // A矩阵一个tile在LDS中的总大小
    constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta);       // A矩阵一个tile写入LDS需要的迭代次数
    
    // A矩阵读取偏移计算
    // thread_n_blocks/8 表示B矩阵[128,128]的一行需要几个warp处理 n_blocks*16
    // (threads/64)/(thread_n_blocks/8) 表示一次处理几个[16,128]的数据块
    // 4* 是因为一个[16,128]的B数据块对应A中[x,16]的数据块，16个fp16转为64bit是4个64bit
    constexpr int a_sh_rd_delta_o = 4 * ((threads / 64) / (thread_n_blocks / 8)); // 横向偏移(int2单位)
    constexpr int a_sh_rd_delta_i = 16;                  // 纵向偏移(行数)
    
    // === B矩阵访问参数 ===
    int b_gl_stride = 16 * prob_n / (pack_factor * 4);  // B矩阵全局步长 (int4单位)
    constexpr int b_sh_stride = ((thread_n_blocks * 16) * 16 / pack_factor) / 4; // B矩阵共享内存步长
    constexpr int b_sh_stride_threads = b_sh_stride;     // B矩阵共享内存线程步长
  
    int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks; // B矩阵全局读取横向偏移
    int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride_threads); // B矩阵全局读取纵向偏移
    constexpr int b_sh_wr_delta = threads;               // B矩阵共享内存写入偏移
    constexpr int b_sh_rd_delta = threads;               // B矩阵共享内存读取偏移
    constexpr int b_sh_stage = b_sh_stride * thread_k_blocks; // B矩阵一个stage的大小
    constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta; // B矩阵写入迭代次数

    // === 合并的缩放因子和零点矩阵访问参数 ===
    int s_zp_gl_stride = prob_n / 8 + (prob_n / pack_factor) / 4;               // 全局步长
    int s_zp_sh_stride = 16 * thread_n_blocks / 8 + ((16 * thread_n_blocks) / pack_factor) / 4; // 共享内存步长
    int s_zp_tb_groups = group_blocks != -1 && group_blocks < thread_k_blocks   // 线程块组数
                          ? thread_k_blocks / group_blocks : 1;
    int s_zp_sh_stage = s_zp_tb_groups * s_zp_sh_stride;                        // 一个stage的大小
    int s_zp_gl_rd_delta = s_zp_gl_stride;                                       // 全局读取偏移

    // === 全局A矩阵读取索引(64bit单位) ===
    int row = threadIdx.x / a_gl_rd_delta_o;             // 计算当前线程的行索引
    int col = threadIdx.x % a_gl_rd_delta_o;             // 计算当前线程的列索引
    col = col ^ (row % 8);                               // 执行异或列坐标变换，每8行为一个整体

    int a_gl_rd_row = row;                               // A矩阵全局读取行索引
    int a_gl_rd_col = col;                               // A矩阵全局读取列索引
    a_gl_rd_col = a_gl_rd_col + a_gl_rd_delta_o * slice_row; // 加上slice偏移
  
    // === 共享内存A矩阵读取索引 ===
    int a_sh_rd_row = (warp_idx) % 16;                   // 共享内存读取行索引
    int a_sh_rd_col = warp_idx / 16 + 4 * ((threadIdx.x / 64) / (thread_n_blocks / 8)); // 共享内存读取列索引

    // === B矩阵全局读取索引 ===
    int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride_threads) +
                  (threadIdx.x % b_sh_stride_threads);   // 基础偏移
    b_gl_rd += b_sh_stride * slice_col;                  // 加上列slice偏移
    b_gl_rd += b_gl_rd_delta_o * slice_row;              // 加上行slice偏移
    int b_sh_rd = threadIdx.x;                           // B矩阵共享内存读取索引
    
    // === 缩放因子和零点全局读取索引 ===
    int s_zp_gl_rd;
    if(group_blocks == -1) {                             // 如果没有分组
      s_zp_gl_rd = s_zp_sh_stride * slice_col + threadIdx.x;
    } else {                                             // 如果有分组(通常走这里)
      s_zp_gl_rd = s_zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                s_zp_sh_stride * slice_col + threadIdx.x;
    }

    // === 共享内存写入和读取索引 ===
    int s_zp_sh_wr = threadIdx.x;                        // 每个线程加载128bit数据
    bool s_zp_sh_wr_pred = threadIdx.x < s_zp_sh_stride; // 只有部分线程需要执行load指令

    // === 缩放因子共享内存读取索引 ===
    // 由于需要处理[16,128]对应的数据，预处理中已将8个fp16转换
    int s_sh_rd = 20 * ((threadIdx.x / 64) % (thread_n_blocks / 8)) +
                (warp_idx) % 16;                         // 缩放因子读取索引

    // === 零点共享内存读取索引 ===
    int zp_sh_rd = 80 * ((threadIdx.x / 64) % (thread_n_blocks / 8)) + 64 +
                  (warp_idx % 16);                       // 零点读取索引
    
    // === 预计算A矩阵共享内存读取变换索引 ===
    int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];  // 存储变换后的索引
    for (int i = 0; i < b_sh_wr_iters; i++) {           // 遍历B矩阵写入迭代
      for (int j = 0; j < thread_m_blocks; j++){        // 遍历M维度块

        int row = a_sh_rd_row + a_sh_rd_delta_i * j;     // 计算实际行索引
        int col = a_sh_rd_col + a_sh_rd_delta_o * i;     // 计算实际列索引
        
        // === 异或变换以优化内存访问模式 ===
        // (col/2) ^ row: 以int4为单位进行异或变换，获取变换后的int4列索引
        // *2: 转换为int2为单位的列索引
        // +col%2: 加上在int4数据中的偏移
        int trans_col = ((col / 2) ^ (row % 8)) * 2 + col % 2;
        a_sh_rd_trans[i][j] = row * (a_sh_stride * 2 ) + trans_col; // 存储最终索引
      }
    }
  
    // === 预计算B矩阵全局读取偏移 ===
    int B_idx_int[b_sh_wr_iters];                        // B矩阵读取索引数组
    for (int i = 0; i < b_sh_wr_iters; i++)
      B_idx_int[i] = b_gl_rd_delta_i * i + b_gl_rd;     // 计算每次迭代的偏移

    // === 共享内存分配 ===
    int4* sh_a = sh_new;                                 // A矩阵共享内存
    int4* sh_s_zp = sh_a + (stages * a_sh_stage);       // 缩放因子和零点共享内存(交叉放置)
    int4* sh_red = sh_a;                                 // 用于块内归约的共享内存

    // === 寄存器存储定义 ===
    // 双缓冲共享内存读取的寄存器存储
    FragA frag_a[2][thread_m_blocks];                    // A矩阵片段: Vec<__half2, 2>
    I4 frag_b_quant[stages][b_sh_wr_iters];             // 量化B矩阵片段: Vec<int, 4>
    FragC frag_c[thread_m_blocks][4][2];                // C矩阵片段: Vec<float, 4>
                                                         // [4]表示[16,128]中第几个[16,32]
                                                         // [2]表示[16,32]中两次mma的哪一次
    FragS frag_s[2][4];                                  // 缩放因子片段: Vec<__half2, 1>
                                                         // [2]表示双缓冲
                                                         // [4]表示[16,128]中第几个[16,32]
    int frag_qzp[2];                                     // 量化零点
    FragZP frag_zp;                                      // 零点片段: Vec<__half2, 4>

    /*
     * 累加器清零Lambda函数
     */
    auto zero_accums = [&]() {
      for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
        reinterpret_cast<float*>(frag_c)[i] = 0;         // 将所有累加器设为0
    };

    /*
     * 初始化slice的Lambda函数
     * 
     * 功能：
     * 1. 计算当前slice需要处理的迭代次数
     * 2. 更新slice_count和slice_idx
     * 3. 处理多块协作的情况
     * 4. 初始化输出为零(如果需要)
     */
    auto init_slice = [&](bool first_init = false) {
      // === 计算当前block还需处理的总数据块数 ===
      slice_iters = iters * (blcok_idx_x + 1) - (k_tiles * slice_col_par + slice_row);
      
      // === 边界检查 ===
      if (slice_iters < 0 || slice_col_par >= n_tiles * parallel) slice_iters = 0;
      if (slice_iters == 0) return;
      
      // === 修正为当前条带中需要处理的数据块数 ===
      if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
      slice_count = 1;                                   // 默认只有一个块处理
      slice_idx = 0;                                     // 默认块索引为0
  
      // === 处理多块协作的情况 ===
      // 分为三种情况：第一个条带、中间条带、最后一个条带
      int col_first = iters * div_ceil(k_tiles * slice_col_par, iters);
      if (col_first <= k_tiles * (slice_col_par + 1)) { // 第一个和最后一个条带需要修正
        int col_off = col_first - k_tiles * slice_col_par; // 列方向偏移
        slice_count = div_ceil(k_tiles - col_off, iters); // 计算需要的块数
        if (col_off > 0) slice_count++;                  // 如果有偏移则增加一个块
        
        int delta_first = iters * blcok_idx_x - col_first;
        if (delta_first < 0 || (col_off == 0 && delta_first == 0))
          slice_idx = slice_count - 1;                   // 设置为最后一个块
        else {
          slice_idx = slice_count - 1 - delta_first / iters;
          if (col_off > 0) slice_idx--;                  // 调整索引
        }
      }

      // === 更新锁偏移量 ===
      if (parallel * n_tiles >= gridDim.x) {
        if (slice_count > 1 && slice_idx == slice_count - 1) {
          locks_off++;
        }
      } else {
        locks_off++;
      }

      // === 首次初始化时将输出清零 ===
      if (first_init && slice_count > 1 && slice_idx == 0) {
        constexpr int threads_per_m = 16 * thread_n_blocks / 8;
        int m_per_thread = div_ceil(block_num_valid_tokens, threads / threads_per_m);
        
        for (int i = 0; i < m_per_thread; i++) {
          int row = threads / threads_per_m * i + threadIdx.x / threads_per_m;
          if (row < block_num_valid_tokens) {
            int64_t sorted_row = sh_block_sorted_ids[row];
            int col = slice_col * 16 * thread_n_blocks / 8 + threadIdx.x % threads_per_m;
            C[sorted_row * prob_n / 8 + col] = {0, 0, 0, 0}; // 清零输出
          }
        }
        
        // 写入负值到锁中，其他SM会等待这个负值
        __syncthreads();
        if (threadIdx.x == 0) locks[locks_off] = 1 - slice_count;
      }

      // === 处理列边界和专家切换 ===
      if (slice_col == n_tiles) {
        slice_col = 0;                                   // 重置列索引
        par_id++;                                        // 移动到下一个并行组
        update_next_moe_block_data();                    // 更新MoE数据
      }
    };

    // === 初始化 ===
    update_next_moe_block_data();                        // 更新MoE块数据
    init_slice(true);                                    // 初始化slice

    bool should_load_a = true;                           // A矩阵加载标志(用于缓存优化)
    int warp_a_idx = a_sh_stride * ((threadIdx.x / 64) * (64 / a_gl_rd_delta_o)); // A矩阵warp索引
    int warp_b_idx = (threadIdx.x / 64) * 64;            // B矩阵warp索引
    
    /*
     * 数据预取到共享内存的Lambda函数
     * 
     * 功能：
     * 1. 异步加载A矩阵[16,128]到LDS
     * 2. 异步加载B矩阵[128,128]到VGPR
     * 3. 异步加载缩放因子和零点到LDS
     */
    auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true, bool scale_first_load=false) {
      if (pred){
        // === 加载A矩阵[16,128](fp16) ===
        int4* sh_a_stage = sh_a + a_sh_stage * pipe;     // 获取当前流水级的共享内存位置
        for (int i = 0; i < a_sh_wr_iters; i++) {       // 迭代写入
          int row = a_gl_rd_row + (threads / a_gl_rd_delta_o) * i; // 计算实际行
          int col = a_gl_rd_col + a_gl_rd_delta_o * a_off;         // 计算实际列

          int sorted_row = sh_block_sorted_ids[row] / top_k;       // 获取排序后的行索引
          int true_idx = 16 * (sorted_row * a_gl_stride + col);   // 计算真实索引(128bit单位)
          sh_a_stage = sh_a_stage + threads * i;                  // 更新共享内存指针

          bool read_a = (row < block_num_valid_tokens);            // 判断是否需要读取
          cp_async4_pred(&sh_a_stage[warp_a_idx], &A[0], true_idx, read_a); // 异步复制
        }

        // === 加载B矩阵[128,128](fp16) -> [8,64](int4) ===
        for (int i = 0; i < b_sh_wr_iters; i++) {
          int b_global_idx = 16 * (B_idx_int[i] + B_expert_off); // 计算全局索引

          // 直接加载到寄存器(使用buffer_load_dwordx4内嵌汇编)
          cp_async4_vgpr(&frag_b_quant[pipe][i], &B[0], b_global_idx);
          
          B_idx_int[i] += b_gl_rd_delta_o;               // 移动到下一个数据块
        }

        // === 加载缩放因子和零点 ===
        int4* sh_s_zp_stage = sh_s_zp + s_zp_sh_stage * pipe;
        if constexpr (group_blocks >= thread_k_blocks){
          // 每次都加载scale和zero(或复用以适配新流水)
          if (scale_first_load || ((slice_scale_row % (group_blocks / thread_k_blocks))==0)){ 
            cp_async4_pred(&sh_s_zp_stage[0], &scales_zeros_ptr[0], 
                          (s_zp_gl_rd + scales_zeros_offset_int4) * 16, s_zp_sh_wr_pred);
            s_zp_gl_rd += s_zp_gl_rd_delta;             // 向下移动一行
          }else{
            // 复用时重新加载以适配新流水
            cp_async4_pred(&sh_s_zp_stage[0], &scales_zeros_ptr[0], 
                          (s_zp_gl_rd - s_zp_gl_rd_delta + scales_zeros_offset_int4) * 16, s_zp_sh_wr_pred);
          }
          slice_scale_row ++;
        } else {
          // 迭代加载多行scale/zero矩阵数据到LDS
          for (int i = 0; i < s_zp_tb_groups; i++) {
            cp_async4_pred(&sh_s_zp_stage[i * s_zp_sh_stride], &scales_zeros_ptr[0], 
                          (s_zp_gl_rd + scales_zeros_offset_int4) * 16, s_zp_sh_wr_pred);
            s_zp_gl_rd += s_zp_gl_rd_delta;             // 向下移动一行
          }
        }
      }
    };
    
    /*
     * 数据加载到寄存器的Lambda函数
     * 
     * 从共享内存加载A矩阵数据到寄存器
     */
    auto fetch_to_registers = [&](int k, int pipe) {
      int4* sh_a_stage = sh_a + a_sh_stage*pipe;         // 获取共享内存阶段
      FragA* fraga_ptr = reinterpret_cast<FragA*>(sh_a_stage); // 转换为FragA指针
      
      // 一个warp加载[16,16]大小数据到寄存器，每个线程4个fp16数据
      for (int i = 0; i < thread_m_blocks; i++){
        frag_a[k % 2][i] = fraga_ptr[a_sh_rd_trans[k % b_sh_wr_iters][i]]; // 使用预计算的变换索引
      }
    };

    constexpr int tb_k = 16 * thread_k_blocks;          // 线程块K维度大小
    constexpr int k_iter_size = tb_k / b_sh_wr_iters;   // warp在B矩阵上一次迭代跨越的K维度行数
    
    /*
     * 缩放因子和零点加载到寄存器的Lambda函数
     */
    auto fetch_scales_zp_to_registers = [&](int k, int full_pipe) {
      int pipe = full_pipe % stages;                     // 确定当前流水级
      
      if constexpr (group_blocks >= thread_k_blocks) {
        // 只有两块数据存储区，每次从自己的数据区取scale/zero
        int4* sh_s_zp_stage = sh_s_zp + s_zp_sh_stage * pipe;
        reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_zp_stage[s_sh_rd];  // 加载scale
        frag_qzp[k % 2] = (reinterpret_cast<int*>(sh_s_zp_stage))[zp_sh_rd]; // 加载zero
      } else { 
        // LDS中有多行scale和zero数据
        int warp_id = threadIdx.x / 64;                  // 当前warp ID
        int n_warps = thread_n_blocks / 8;               // N维度warp数量
        int warp_row = warp_id / n_warps;                // 当前warp在layout中的行

        int cur_k = warp_row * 16;                       // 第一次迭代时K维度索引
        cur_k += k_iter_size * (k % b_sh_wr_iters);     // 加上迭代带来的K维度偏移

        int cur_group_id = cur_k / 16 / group_blocks;    // 获取当前处理数据的组ID

        int4* sh_s_zp_stage = sh_s_zp + s_zp_sh_stage * pipe + cur_group_id * s_zp_sh_stride;
        reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_zp_stage[s_sh_rd];  // 加载scale
        frag_qzp[k % 2] = (reinterpret_cast<int*>(sh_s_zp_stage))[zp_sh_rd]; // 加载zero
      }
    };

    /*
     * 零点片段获取Lambda函数
     * 
     * 反量化零点矩阵并应用缩放
     */
    auto get_frag_zp = [&](int k) {
      bool pred = (k == 0) || (group_blocks < thread_k_blocks); // 判断是否需要更新
      if (pred){
        int k2 = k % 2;
        
        // === 反量化零点矩阵 ===
        FragB frag_zp_0, frag_zp_1;                     // 零点片段
        int zp_quant_0, zp_quant_1;                     // 量化零点

        zp_quant_0 = frag_qzp[k2];                      // 获取量化零点
        zp_quant_1 = zp_quant_0 >> 8;                   // 右移8位获取高位

        // 反量化零点数据
        dequantize_int4_to_fp16(zp_quant_0, frag_zp_0);
        dequantize_int4_to_fp16(zp_quant_1, frag_zp_1);

        // 组装零点片段
        frag_zp[0] = frag_zp_0[0];
        frag_zp[1] = frag_zp_0[1];
        frag_zp[2] = frag_zp_1[0];
        frag_zp[3] = frag_zp_1[1];
        
        // === 零点乘以缩放因子: zero*scale ===
        for (int j = 0; j < 4; j++) { 
          frag_zp[j] = __hmul2(frag_zp[j], *reinterpret_cast<scalar_t2*>(&frag_s[k2][j]));
        }
      }
    };

    /*
     * 矩阵乘法Lambda函数
     * 
     * 执行左矩阵[16,16](fp16) × 右矩阵[16,128](fp16)的计算
     */
    auto matmul = [&](int k, int b_com_pipe) {
      int k2 = k % 2;                                    // 双缓冲索引
      
      // 在[16,128]矩阵块上执行4次[16,32]数据块计算
      for (int j = 0; j < 4; j++) { 
        FragB frag_b0, frag_b1;                         // B矩阵片段
        int b_quant_0, b_quant_1;                       // 量化B矩阵数据
        
        // === 获取量化B矩阵数据 ===
        b_quant_0 = frag_b_quant[b_com_pipe][k2][j];    // 第j次[16,32]计算中的8个uint4数据
        b_quant_1 = b_quant_0 >> 8;                     // 右移8位获取高位数据

        // === 反量化B矩阵 ===
        dequantize_int4_to_fp16(b_quant_0, frag_b0);    // 前4个fp16数据
        dequantize_int4_to_fp16(b_quant_1, frag_b1);    // 后4个fp16数据
        
        // === 应用缩放和零点校正: b*scale - zero*scale ===
        scale_and_sub<scalar_t>(frag_b0, frag_s[k2][j][0].x, frag_zp[j].x);
        scale_and_sub<scalar_t>(frag_b1, frag_s[k2][j][0].y, frag_zp[j].y);

        // === 执行矩阵乘累加操作 ===
        for (int i = 0; i < thread_m_blocks; i++) {
          mmac_volatile<scalar_t>(frag_a[k2][i], frag_b0, frag_c[i][j][0]); // 第一次MMAC
          mmac_volatile<scalar_t>(frag_a[k2][i], frag_b1, frag_c[i][j][1]); // 第二次MMAC
        }
      }
    };

    /*
     * 线程块归约Lambda函数
     * 
     * 如果K方向有多个warp并行，需要进行warp间结果归约
     */
    auto thread_block_reduce = [&]() {
      constexpr int red_off = threads / b_sh_stride_threads / 2; // K维度并行warp数的一半
      
      if (red_off >= 1) { // 如果K方向有多个warp
        int red_idx = threadIdx.x / b_sh_stride_threads;  // 当前warp在归约中的索引
        constexpr int red_sh_stride = b_sh_stride_threads * 4 * 2; // 每行需要归约的float4数据数量
        constexpr int red_sh_delta = b_sh_stride_threads;           // 归约步长
        int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) +
                        (threadIdx.x % b_sh_stride_threads);       // 归约读取索引
        
        for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
          // === 多级归约 ===
          for (int i = red_off; i > 0; i /= 2) {
            if (i <= red_idx && red_idx < 2 * i) {      // 确定参与归约的warp
              // 逐个处理8个[16,32]的结果
              for (int j = 0; j < 4 * 2; j += 1) {
                int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
                
                if (i < red_off) {
                  // 从共享内存读取数据并累加到寄存器
                  float* c_rd = reinterpret_cast<float*>(&sh_red[red_sh_delta * j + red_sh_rd]);
                  float* c_wr = reinterpret_cast<float*>(&sh_red[red_sh_wr]);
                  for (int k = 0; k < 4; k++)
                    reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];
                }
                
                // 将frag_c结果写入共享内存
                sh_red[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
              }
            }
            __syncthreads();                             // 同步确保数据写入完成
          }
          
          // === 最终归约：warp 0和warp 1读取共享内存数据并累加 ===
          if (red_idx == 0) {
            for (int i = 0; i < 4 * 2; i += 1) { 
              float* c_rd = reinterpret_cast<float*>(&sh_red[red_sh_delta * i + red_sh_rd]);
              for (int j = 0; j < 4; j++)
                reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
            }
          }
          __syncthreads();                               // 同步确保归约完成
        }
      }
    };

    /*
     * 结果写回Lambda函数
     * 
     * 将计算结果以正确布局写回全局内存：
     * 1. warp0中64个线程将frag_c数据写到LDS对应位置
     * 2. LDS->Global: 根据是否多块协作选择直接写入或原子加
     */
    auto write_result = [&]() {
      int c_gl_stride = prob_n / 8;                      // C矩阵全局步长(int4单位)
      constexpr int c_sh_stride = 2 * thread_n_blocks + 1; // C矩阵共享内存步长
      int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks)); // 全局写入偏移
      constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks)); // 共享内存读取偏移

      /*
       * 写入函数：在共享内存指定索引处插入float转换的fp16
       */
      auto write = [&](int idx, float c_float) {
        ((__half*)sh_red)[idx] = __float2half(c_float); // 转换并写入
      };
      
      // === 遍历多个thread_m_blocks ===
      for (int b_blocks_idx=0; b_blocks_idx<thread_m_blocks; b_blocks_idx++){
        // === 初始化写回参数 ===
        int c_gl_wr = c_gl_stride * (b_blocks_idx * 16 + (threadIdx.x / (2 * thread_n_blocks))) +
                    (threadIdx.x % (2 * thread_n_blocks)); // 全局写入索引
        c_gl_wr += (2 * thread_n_blocks) * slice_col;      // 加上slice偏移
        
        int c_sh_wr;
        c_sh_wr = (8 * c_sh_stride) * (warp_idx % 16) + warp_idx / 16; // 共享内存写入索引
        c_sh_wr += 128 * (threadIdx.x / 64);               // 加上warp偏移

        int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                            (threadIdx.x % (2 * thread_n_blocks)); // 共享内存读取索引

        // === 处理B矩阵第一行数据的warp将结果写入LDS ===
        if (threadIdx.x / 64 < thread_n_blocks / 8) {
          // 将frag_c中的结果写入[16,128]的LDS中
          for (int j = 0; j < 4; j++) { // 写入第j个[16,32]的结果矩阵
            int wr = c_sh_wr + 32 * j;                     // 计算写入偏移

            // 写入8个float值(转换为half)
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
        __syncthreads();                                 // 同步确保LDS写入完成
        
        // === 256个线程将LDS数据写回全局内存 ===
        for (int i = 0; i < div_ceil(16, threads / (2 * thread_n_blocks)); i++) { 
          int row = c_gl_wr / c_gl_stride;               // 计算行索引
          if (row < block_num_valid_tokens) {            // 只有有效数据才写入
            int col = c_gl_wr % c_gl_stride;             // 计算列索引
            int64_t sorted_row = sh_block_sorted_ids[row]; // 获取排序后的行索引
            int64_t true_idx = sorted_row * c_gl_stride + col; // 计算真实索引
            
            // === 获取top-k权重 ===
            scalar_t2 topk_weight_score;
            if (mul_topk_weights) topk_weight_score = sh_block_topk_weights[row];

            // === 根据情况选择写入方式 ===
            if (slice_count > 1 || mul_topk_weights) {
              // 需要原子操作或权重乘法
              scalar_t2* C_half2 = reinterpret_cast<scalar_t2*>(&C[true_idx]); // 指向half2
              scalar_t2* sh_red_half2 = reinterpret_cast<scalar_t2*>(&sh_red[c_sh_rd]);
              
              for (int a = 0; a < 4; a++) {              // 逐次原子加4次，每次half2(32bit)
                scalar_t2 res = sh_red_half2[a];
                if (mul_topk_weights) {
                  res = __hmul2(res, topk_weight_score);  // 乘以top-k权重
                }
                if (slice_count > 1) {
                  atomicAdd(&C_half2[a], res);            // 原子加
                } else {
                  C_half2[a] = res;                       // 直接赋值
                };
              }
            } else { 
              // 直接以int4(128bit)写回
              C[true_idx] = sh_red[c_sh_rd];
            }
            c_gl_wr += c_gl_wr_delta;                    // 更新全局写入索引
            c_sh_rd += c_sh_rd_delta;                    // 更新共享内存读取索引
          }
        }
        __syncthreads();                                 // 同步确保写回完成
      }
    };

    /*
     * 启动流水线Lambda函数
     * 
     * 初始化异步预取和寄存器流水线
     */
    auto start_pipes = [&]() {
      // === 发射前几组global->LDS的异步指令 ===
      for (int i = 0; i < stages - 1; i++) {
        fetch_to_shared(i, i, i < slice_iters, slice_scale_row==slice_row);
      }
      zero_accums();                                     // 重置结果寄存器为0
      
      a_gl_rd_col += a_gl_rd_delta_o * (stages - 1);    // 更新A矩阵读取列索引
    };

    // === 主要计算循环 ===
    slice_scale_row = slice_row;                         // 初始化scale行索引
    if (slice_iters) {
      start_pipes();                                     // 启动流水线
    }

    while (slice_iters) {                                // 处理当前条带的数据
      for (int pipe = 0; pipe < stages; pipe++) {       // 遍历流水线级数
        cp_async_wait_0();                               // 等待异步加载完成
        __syncthreads();                                 // 同步所有线程
        
        for (int k = 0; k < b_sh_wr_iters; k++) {       // B矩阵每个warp执行b_sh_wr_iters次[16,128]计算
          // === 发射下一个数据块的全局load指令 ===
          if (k == b_sh_wr_iters - 2) {
            fetch_to_registers(0, pipe);                 // 加载A矩阵到寄存器
            fetch_scales_zp_to_registers(0, pipe);       // 加载scale/zero到寄存器
            fetch_to_registers(1, pipe);                 // 加载下一批A矩阵
            fetch_scales_zp_to_registers(1, pipe);       // 加载下一批scale/zero
            
            // 加载下一个数据块
            fetch_to_shared((pipe + stages - 1) % stages, pipe,
                            slice_iters >= stages, slice_scale_row==slice_row);
          }
          
          // === 执行矩阵乘法 ===
          get_frag_zp(k);                                // 获取零点片段
          matmul(k, pipe);                               // 执行矩阵乘法
        }
        slice_iters--;                                   // 减少剩余迭代次数
        if (slice_iters == 0) {
          break;                                         // 如果完成则退出
        }
      }
      a_gl_rd_col += a_gl_rd_delta_o * stages;          // 更新A矩阵读取列索引

      if (slice_iters == 0) {
        cp_async_wait_0();                               // 等待最后的异步操作完成
        __syncthreads();                                 // 同步所有线程

        thread_block_reduce();                           // 执行线程块归约
        
        // === 多块协作时等待其他块完成 ===
        if (slice_count > 1 && slice_idx != 0)
          wait_negative_and_add(&locks[locks_off]);      // 等待负值锁并原子加1

        write_result();                                  // 写回结果

        // === 转到下一个条带 ===
        int old_slice_row = slice_row;
        slice_row = 0;                                   // 重置行索引
        slice_col_par++;                                 // 增加列索引
        slice_col++;

        init_slice();                                    // 初始化新slice
      
        if (slice_iters) {
          // === 更新A矩阵新的索引地址 ===
          a_gl_rd_row = row;
          a_gl_rd_col = col;

          // === 更新B矩阵新的索引地址 ===
          for (int i = 0; i < b_sh_wr_iters; i++)
            B_idx_int[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles; // 先右移再上移
          if (slice_col == 0) {                          // 如果转到下一个专家
            for (int i = 0; i < b_sh_wr_iters; i++) 
              B_idx_int[i] -= b_gl_stride;               // 回到专家矩阵左上角
          }

          // === 更新scale、zero的全局访问索引 ===
          s_zp_gl_rd = s_zp_sh_stride * slice_col + threadIdx.x;
       
          // === 启动下一阶段的流水线 ===
          slice_scale_row = slice_row;
          start_pipes(); 
        }
      }
    }
  }

} // namespace marlin_moe_w4a16
} // namespace native  
} // namespace at