#include "marlin.h"

namespace at {
namespace native {
namespace marlin_moe_w4a8{
  /*
    采用顺序分割 条带方式
  */

  template <
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const int stages 
          >
  __global__ void Marlin_512_64(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const float* __restrict__ A_scales,  // 
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    const float* __restrict__ B_scales, 
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    const int32_t* __restrict__ sorted_token_ids_ptr,        // moe sorted_ids
    const int32_t* __restrict__ expert_ids_ptr,              // moe expert ids
    const int32_t* __restrict__ num_tokens_past_padded_ptr,  // moe num tokens
    const float* __restrict__ topk_weights_ptr,              // moe top weights
    int top_k,              // num of experts per token
    bool mul_topk_weights,  // mul topk weights or not
    int prob_m,             // batch dimension m
    int prob_n,             // output dimension n
    int prob_k,             // reduction dimension k
    int* locks,             // 用于 全局归约的信号控制
    int4* C_tmp             // 用于 全局归约的 缓冲区
  ) {

    extern __shared__ int4 sh[];
  
    constexpr int pack_factor = 8; //  一个int32中有几个 数据被pack
    constexpr int mfma_m = 16; //  mmac指令 m 方向的大小
    constexpr int mfma_n = 16; //  mmac指令 n 方向的大小
    constexpr int mfma_k = 32; //  mmac指令 k 方向的大小
    
    constexpr int moe_block_size = 16 * thread_m_blocks; // block_m

    // parallel: 需要计算的专家数 // 167
    int num_tokens_past_padded = num_tokens_past_padded_ptr[0]; // 一共几个token选择
    int parallel = num_tokens_past_padded / moe_block_size; 
  
    int n_tiles = prob_n / mfma_n / thread_n_blocks; // n 方向 有几个 [128, 128]的数据块 4 
    int k_tiles = prob_k / mfma_k / thread_k_blocks; // k 方向 有几个 [128, 128] 的数据块 56

    int iters = div_ceil(n_tiles * parallel, gridDim.x); // 每个方向
    int slice_row = 0; // k方向 数据块的索引 现在必定从 0 开始
    int slice_col_par = iters * blockIdx.x; // N方向 包含 parallel 的数据块的索引
    int slice_col_par_max = iters * blockIdx.x + iters; // 记录最大值是啥
    int slice_col = slice_col_par; // N方向 不包含 parallel 的数据块的索引
    int slice_iters = k_tiles;  // 每个条带都有一个block单独执行，因此是全部的K方向的数据块
    int slice_count = 1;  // 当前条带中, 一共需要几个block执行计算 当前由于都有一个block执行计算，因此恒为0
    int slice_idx=0;  // 当前条带中, 所有执行计算任务blcok中，该block的索引，索引从下 往上编号

    int par_id = 0;
    int block_id = -1;
    int64_t expert_id = 0;  // use int64 to avoid computation result overflow
    int old_expert_id = 0;
    int64_t B_expert_off = 0;
    int64_t B_scale_off = 0;
  
    int4* sh_block_sorted_ids_int4 = sh;
    int32_t* sh_block_sorted_ids =
        reinterpret_cast<int*>(sh_block_sorted_ids_int4);
    int4* sh_block_topk_weights_int4 =
        sh_block_sorted_ids_int4 + moe_block_size / 4;
    float* sh_block_topk_weights =
        reinterpret_cast<float*>(sh_block_topk_weights_int4);

    int4* sh_new = sh_block_topk_weights_int4 + moe_block_size / 4;

    int32_t block_num_valid_tokens = 0;

    // 修正 slice_col 和 par_id
    if (slice_col_par >= n_tiles) {
      slice_col = slice_col_par % n_tiles;
      par_id = slice_col_par / n_tiles;
    }

    // A sizes/strides
    int a_gl_stride = prob_k / 16; // int4 7168(uint8) --> 448(int4) 16个int8数据类型组成一个 int4 数据类型
    constexpr int a_sh_stride = mfma_k * thread_k_blocks / 16; // int4 8
    constexpr int a_gl_rd_delta_o = mfma_k * thread_k_blocks / 16; // 全局 load时, A矩阵 横向的偏移量 // 8
    int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o); // 全局load时， A矩阵 竖向的偏移量  a_gl_stride * 8
  
    constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_sh_stride); // A矩阵写入lds时, 一个数据块内两次写入的偏移量 a_sh_stride * 8

    constexpr int a_sh_stage = a_sh_stride * (mfma_m * thread_m_blocks); // A矩阵一个tile(16,128)在lds中的总大小 [16, 128](fp16) -->  [16, 8](int4)
    constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta); // A 矩阵一个tile(16, 128) 写入lds时需要几次  1次，并且一次中只有两个 warp 进行写入

    // lds 中 读取 A矩阵数据时， 两次[16, 32]load时的偏移量   64bit
    //    (thread_n_blocks / 4)代表B矩阵[128, 128]的一行需要几个warp处理
    //    ((threads / 64) / (thread_n_blocks / 4)) 代表当前warp需要的是 第几个 32大小的K维度数据块
    //    4 * 是因为 32个int8 转为 int2 是 4 个int2
    // 横向偏移增量
    constexpr int a_sh_rd_delta_o = 4 * ((threads / 64) / (thread_n_blocks / 4)); // int2  变成横向偏移多少 int2 个数据
    // 竖向偏移增量
    constexpr int a_sh_rd_delta_i = 16; // int2  变成 竖向偏移多少行数据
   
    // B sizes/strides
    int b_gl_stride = 32 * prob_n / pack_factor / 4; // 512 int4 全局B矩阵 一行的步长
    constexpr int b_sh_stride = (thread_n_blocks * 16) * 32 / pack_factor / 4;  // 256 int4 lds中B矩阵一行的步长

    constexpr int b_sh_stride_threads = b_sh_stride; // 256 lds中 B矩阵一行需要多少线程读取

    int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks; // b_gl_stride * 4   B矩阵跨数据块的偏移
    int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride_threads); //  b_gl_stride * 2  B矩阵内部两次load的偏移

    constexpr int b_sh_wr_delta = threads; // 512
    constexpr int b_sh_rd_delta = threads; // 512

    constexpr int b_sh_stage = b_sh_stride * thread_k_blocks; // b_sh_stride * 4   lds中B矩阵的大小 int4 256 * 4
    constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta; // 2 B矩阵需要 512 个线程load 两次才能load完成

    // Global A read index of current thread. 64bit
    int row = threadIdx.x / a_gl_rd_delta_o;
    int col = threadIdx.x % a_gl_rd_delta_o;
    // swizzle 坐标变化
    //     每几行数据做一次swizzle
    int bank_rows = 8 / a_sh_stride;
    // swizzle 
    col = col ^ (row / bank_rows % (8 / bank_rows));

    int a_gl_rd_row = row;
    int a_gl_rd_col = col;
    a_gl_rd_col = a_gl_rd_col + a_gl_rd_delta_o * slice_row;
  
    int a_sh_rd_row = (threadIdx.x % 64) % 16; // 0
    int a_sh_rd_col = threadIdx.x % 64 / 16 + 4 * ((threadIdx.x / 64) / (thread_n_blocks / 4)); // 0

    int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride_threads) +
                  (threadIdx.x % b_sh_stride_threads);
    b_gl_rd += b_sh_stride * slice_col;
    b_gl_rd += b_gl_rd_delta_o * slice_row;
    int b_sh_rd = threadIdx.x;
    
    int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks]; // a_sh_rd_trans[2][1]

    for (int i = 0; i < b_sh_wr_iters; i++) {
      for (int j = 0; j < thread_m_blocks; j++){

        int row = a_sh_rd_row + a_sh_rd_delta_i * j; // 竖向偏移16
        int col = a_sh_rd_col + a_sh_rd_delta_o * i; // 
        // swizzle
          // (col / 2) ^ row : 以int4为单个数据进行 异或变换，获取变换后的int4 列索引
          // * 2 ： 转为 int2为单个数据的列索引 
          // + col % 2： 加上在int4数据中的偏移
        int trans_col = ((col / 2) ^ (row /  bank_rows % (8 / bank_rows))) * 2 + col % 2;
        // int trans_col = col;
        int idx = row * (a_sh_stride * 2 ) + trans_col;
        a_sh_rd_trans[i][j] = idx;
      }
    }

    // 从B矩阵中 load数据时的偏移 int4
    int B_idx_int[b_sh_wr_iters];
    for (int i = 0; i < b_sh_wr_iters; i++)
      B_idx_int[i] = b_gl_rd_delta_i * i + b_gl_rd;

    // Shared memory storage for global fetch pipelines.
    int4* sh_a = sh_new;
    // 用来进行 block内的规约
    int4* sh_red = sh_a;
    
    // 将每个线程load A 矩阵的行索引存储在寄存器中，避免每次执行任务块前都在 LDS 中load
    int32_t load_a_row_idx[a_sh_wr_iters];

    FragA frag_a[2][thread_m_blocks]; // Vec<int8_t, 8>;   frag_a[2][1]
    FragB frag_b_mmac[4];
    I4 frag_b_quant[stages][b_sh_wr_iters]; // I4 = Vec<int, 4>;   frag_b_quant[2][2]
    FragC frag_c[thread_m_blocks][4]; // FragC = Vec<int32_t, 4>; 存储每个线程 mmac 计算之后的结果

    // Zero accumulators.
    auto zero_accums = [&]() {
      for (int i = 0; i < thread_m_blocks * 4 * 4; i++)
        reinterpret_cast<int*>(frag_c)[i] = 0;
    };

    constexpr int sorted_token_ids_int4_len = moe_block_size / 4;
    // 重构更新逻辑
    auto read_moe_block_data = [&](int block_id) {
      // 第一步：128bit 带宽加载 sort_token_ids 的数据到 LDS 中
      if (threadIdx.x < sorted_token_ids_int4_len) {
        sh_block_sorted_ids_int4[threadIdx.x] = reinterpret_cast<const int4*>(sorted_token_ids_ptr)[block_id * sorted_token_ids_int4_len + threadIdx.x];
      }
      // 等待 sorted_token_ids 全部加载到LDS中
      S_BARRIER;
      // 第二步：遍历获取 有效的 索引数
      block_num_valid_tokens = moe_block_size;
      for (int j = 0; j < moe_block_size; j++) {
        // 遍历到第一个 不合法的索引，记录到 block_num_valid_tokens 中后返回
        if (sh_block_sorted_ids[j] >= prob_m * top_k) {
          block_num_valid_tokens = j;
          break;
        }
      }
      // 将每个线程需要 全局内存中 load 的行索引提前load到寄存器中，并只在 跳转专家矩阵时更新
      for(int ii=0;ii < a_sh_wr_iters;ii++){
        int block_idx = std::min((block_num_valid_tokens - 1), (threadIdx.x / a_gl_rd_delta_o  + (threads / a_gl_rd_delta_o) * ii));
        load_a_row_idx[ii] = (sh_block_sorted_ids[block_idx] / top_k) * a_gl_stride * 16;
      }
      // 将 topk_weight 加载到 LDS中, block_num_valid_tokens 个线程把 topk_weight 数据加载到 LDS 中
      if (mul_topk_weights && (threadIdx.x < block_num_valid_tokens)) {
        sh_block_topk_weights[threadIdx.x] = topk_weights_ptr[sh_block_sorted_ids[threadIdx.x]]; 
      }
    };
  
    // 更新 B_expert_off ： 当前专家权重的全局偏移,指向 该专家权重的 （0, 0）处
    // 更新 B_scale_off
    // read_moe_block_data
    //        更新 sh_block_sorted_ids_int4 中的值
    //        更新 sh_block_topk_weights 中的值
    auto update_next_moe_block_data = [&]() {
      if (par_id >= parallel) return;
  
      old_expert_id = expert_id;

      block_id = par_id;
      expert_id = expert_ids_ptr[block_id];
  
      B_expert_off = expert_id * prob_n * prob_k / (pack_factor * 4);

      // 增加 a_scale 和  b_scale的偏移
      B_scale_off = expert_id * prob_n;
      
      read_moe_block_data(block_id);
    };

    // 更新 slice_col，并选择性更新 slice_col、par_id、B_expert_off、B_scale_offsh_block_sorted_ids_int4、sh_block_topk_weights
    auto init_slice = [&]() {
      slice_col_par++;
      // 超过block处理的 最大值 后返回 
      if (slice_col_par >= slice_col_par_max){
        return;
      }

      slice_row = 0;
      slice_col++;

      // 待计算的 任务块数目 重置为 k_tiles
      slice_iters = k_tiles;
      // 如果 列索引越过了当前专家的 列索引边界，则
      if (slice_col == n_tiles) {
        slice_col = 0;
        par_id++;
        update_next_moe_block_data();
      }
    };

    // 更新第一个条带的 B_expert_off、B_scale_off、sh_block_sorted_ids_int4、sh_block_topk_weights
    update_next_moe_block_data();

    int warp_a_idx = a_sh_stride * ((threadIdx.x / 64) * (64 / a_gl_rd_delta_o));
    auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
      if (pred){
        // // // load [16, 128](uint8) ==> [16， 8]（int4） 
        // int4* sh_a_stage = sh_a + a_sh_stage * pipe; 
        // for (int i = 0; i < a_sh_wr_iters; i++) {
        //   int row = a_gl_rd_row + (threads / a_gl_rd_delta_o) * i;
        //   int col = a_gl_rd_col + a_gl_rd_delta_o * a_off;

        //   int sorted_row = sh_block_sorted_ids[row] / top_k; // 213 / 8 --> 26
        //   // 获取A矩阵的实际偏移 128bit  
        //   int true_idx = 16 * (sorted_row * a_gl_stride + col);
        //   sh_a_stage = sh_a_stage + threads * i; // 行索引已经提前
        //   // 通过 read_a 判断是否进行 load
        //   bool read_a = (row < block_num_valid_tokens);
        //   cp_async4_pred(&sh_a_stage[warp_a_idx], &A[0], true_idx, read_a);
        // }
        // 避免从 LDS从加载 sorted_ids 
        int4* sh_a_stage = sh_a + a_sh_stage * pipe;
        for (int i = 0; i < a_sh_wr_iters; i++) {
          int row = a_gl_rd_row + (threads / a_gl_rd_delta_o) * i;
          int col = a_gl_rd_col + a_gl_rd_delta_o * a_off;
          int cur_row_offset_bytes = load_a_row_idx[i];
          // 获取A矩阵的实际偏移 128bit  
          int true_idx = cur_row_offset_bytes + 16 * col;
          sh_a_stage = sh_a_stage + threads * i; // 行索引已经提前
          // 通过 read_a 判断是否进行 load
          bool read_a = (row < block_num_valid_tokens);
          cp_async4_pred(&sh_a_stage[warp_a_idx], &A[0], true_idx, read_a);
        }

        // load B矩阵
        for (int i = 0; i < b_sh_wr_iters; i++) {
          // // 移动到0号线程的起始地址
          int b_global_idx = 16 * (B_idx_int[i] + B_expert_off);
          cp_async4_vgpr(&frag_b_quant[pipe][i], &B[0], b_global_idx);
          B_idx_int[i] += b_gl_rd_delta_o;
        }
      }
    };

    // 从 [16, 128]的 lds A中拷贝 每个线程需要的 A矩阵 数据 
    auto fetch_a_to_registers = [&](int k, int pipe) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
      // 强转成 FragA类型指针  FragA: Vec<int8_t, 8>
      FragA* sh_a_stage_fraga = reinterpret_cast<FragA*>(sh_a_stage);
      for (int i = 0; i < thread_m_blocks; i++){
        // 读取预先计算的 偏移 然后去读取
        frag_a[k][i] = sh_a_stage_fraga[a_sh_rd_trans[k][i]];
      }
    };

    // 执行拆解
    auto prepare_frag_b = [&](int k, int b_com_pipe) { // 0, 1 代表当前在第几个 [16, 128]块上执行计算
      // 在 [32, 64]的四个 [32, 16] 数据块上逐个计算
      for (int j = 0; j < 4; j++) { 
        int b_quant_0, b_quant_1;
        // 取 第 j 次 [32, 16]计算中8个uint4 B矩阵 pack成int32的数据
        b_quant_0 = frag_b_quant[b_com_pipe][k][j];
        b_quant_1 = b_quant_0 << 4;
        // 0 1 2 3 位置的 int8 数据
        int first_val = b_quant_0 & 0xf0f0f0f0; // 保留高 4位
        // 4 5 6 7 位置的 int8 数据
        int second_val = b_quant_1 & 0xf0f0f0f0; // 保留高四位

        // 强制类型转换为 int32_t 指针（8 字节 → 2 × int32_t）
        int32_t* frag_b_int32_ptr = reinterpret_cast<int32_t*>(&frag_b_mmac[j]);
        frag_b_int32_ptr[0] = first_val;
        frag_b_int32_ptr[1] = second_val;
      }
    };
    // 只执行计算
    auto matmul = [&](int k, int b_com_pipe) { // 0, 1 代表当前在第几个 [16, 128]块上执行计算
      // 在 [32, 64]的四个 [32, 16] 数据块上逐个计算
      for (int j = 0; j < 4; j++) { 
        // A矩阵数据 和 拆解出来的B矩阵数据 进行mmac运算
        for (int i = 0; i < thread_m_blocks; i++) {
          mmac_bw_int8(frag_a[k][i], frag_b_mmac[j], frag_c[i][j]);
        }

      }
    };
    
    // block内归约函数
    auto thread_block_reduce = [&]() {
      //  2  需要进行几次 规约操作
      constexpr int red_off = threads / b_sh_stride_threads / 2; 
      if (red_off >= 1) { // 如果K方向只有一行 warp，则其结果就是写回的最终结果，否则的话 需要进行多个warp间结果规约
        //  0 1 2 3  当前线程 在  K 维度的索引
        int red_idx = threadIdx.x / b_sh_stride_threads;
        
        // 64*4 = 256 每一行有多少个 intx4 数据需要规约
        constexpr int red_sh_stride = b_sh_stride_threads * 4; // 每一行有多少个 float4的数据需要规约 64*4 = 256
        constexpr int red_sh_delta = b_sh_stride_threads; // 64

        // 每个线程 laod数据的索引
        int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) +
                        (threadIdx.x % b_sh_stride_threads);
        
        for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
          // red_off = 2  red_idx = 0 1 2 3 执行两次归约操作
          for (int i = red_off; i > 0; i /= 2) { 
            // 第一次执行 要求 2<= red_idx < 4,因此 2 3 号warp执行  // 第二次执行 要求 1<= red_idx < 2 warp 1 激活
            if (i <= red_idx && red_idx < 2 * i) {
              // 逐个处理 4个[32, 64]的结果 ？？
              for (int j = 0; j < 4; j += 1) {
                int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);

                if (i < red_off) { // 第一次循环中，跳过 || 第二次循环 进入
                  // 读取 warp3 写入到自己 LDS块中的 int4 数据
                  int4 warp_3_data = sh_red[red_sh_delta * j + red_sh_rd];
                  // 读取 warp2 写入到 warp0 所属 LDS块中的数据
                  int4 warp_2_data = sh_red[red_sh_wr];
                  // 将上述数据 累加到 自己的寄存器上
                  frag_c[m_block][j][0] += warp_2_data.x + warp_3_data.x;
                  frag_c[m_block][j][1] += warp_2_data.y + warp_3_data.y;
                  frag_c[m_block][j][2] += warp_2_data.z + warp_3_data.z;
                  frag_c[m_block][j][3] += warp_2_data.w + warp_3_data.w;
                }
                // 第一次循环: warp 2, 3 将对应的 fragc 中的结果数据写入 warp0 和 warp1 的lds空间上
                // 第二次循环：warp1 将 warp2 和warp3 以及自己的数据 相加后写入到 warp0 所属 LDS 空间上
                sh_red[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * m_block + j];
              }
            }
            S_BARRIER;
          }
          // warp 0 先读取 red_sh_rd 位置的数据，然后加到 自己持有的frag_c 寄存器中
          if (red_idx == 0) {
            for (int i = 0; i < 4; i += 1) { 
              int4 add_data = sh_red[red_sh_delta * i + red_sh_rd];
              frag_c[m_block][i][0] += add_data.x;
              frag_c[m_block][i][1] += add_data.y;
              frag_c[m_block][i][2] += add_data.z;
              frag_c[m_block][i][3] += add_data.w;
            }
          }
          S_BARRIER;
        }
      }
    };

    // // 以正确的布局写出还原的最终结果：
    // //      1、warp0 中64个线程将各自 fragc 中的数据，写到 LDS 的对应位置（按照结果矩阵的实际数据位置进行放置）
    // //      2、lds-->global:
    // //          (1). 如果该条带中只有一个block执行运算：warp 0-3 共256个线程，每个线程写回 128bit的数据到全局内存中的结果矩阵。
    // //         （2). 多个block共同完成一个slice 的计算，则使用 atomic_add 进行结果回写。
    // auto write_result = [&]() {
    //   // 每 8个 bf16 的数据以一个int4的类型进行store和load, 因此 “/8”
    //   int c_gl_stride = prob_n / 8; // 64
    //   // int4类型的偏移, 一个thread_n_blocks 说明横向有16个fp16的数据, 也意味着有 2个int4
    //   // 实际上 thread_n_blocks*16 /8 -->  2 * thread_n_blocks
    //   constexpr int c_sh_stride = 2 * thread_n_blocks + 1; 
    //   // 往 全局内存 写数据时，两次写回 的偏移
    //   int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks)); 
    //   // 从 LDS 中读取数据时，两次 读取 的偏移
    //   constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));

    //   // 写回之前，先乘 a_scale 和 b_scale
    //   auto write = [&](int idx, int c_int, float a_scale, int b_scale_idx) { // 在 sh_red 的 idx 索引处插入 float转成的 fp16
    //     // 先 load B_scale 到 vgpr 中
    //     int cur_b_scale_idx =  B_scale_off + slice_col * (thread_n_blocks * 16) + b_scale_idx;
    //     float b_scale = B_scales[cur_b_scale_idx];
    //     // 反量化成 fp32
    //     float c_float =  c_int * a_scale * b_scale;
    //     // float c_float =  c_int * 1.0;
    //     // 转成 bf16 然后写入 LDS 中
    //     // ((bhalf_t*)sh_red)[idx] = __float2bfloat16(c_float);
    //     ((bhalf_t*)sh_red)[idx] = f32_to_bf16(c_float);
    //   };
      
    //   // 遍历 多个thread_m_blocks, 128个线程将自己 frag_c寄存器中的数据写入到lds中
    //   for (int b_blocks_idx = 0;b_blocks_idx < thread_m_blocks;b_blocks_idx++){
    //     // 初始化写回的参数
    //     int c_gl_wr = c_gl_stride * (b_blocks_idx * 16 + (threadIdx.x / (2 * thread_n_blocks))) +
    //                 (threadIdx.x % (2 * thread_n_blocks));
    //     c_gl_wr += (2 * thread_n_blocks) * slice_col;
        
    //     int c_sh_wr;
    //     // c_sh_stride 是以 int4 为步长进行计算的 
    //     // c_sh_wr 是以 bf16 为基础单位，因此 8 * c_sh_stride
    //     // 64个线程第一次写入 lds的偏移量（bf16）
    //     c_sh_wr = (8 * c_sh_stride) * (threadIdx.x % 64 % 16) + (threadIdx.x % 64 / 16) * 4;
    //     c_sh_wr += 64 * (threadIdx.x / 64);
  
    //     int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
    //                         (threadIdx.x % (2 * thread_n_blocks));

    //     int b_scale_idx_base = 64 * (threadIdx.x / 64) + (threadIdx.x % 64) / 16 * 4; 
    //     // 处理 B矩阵第一行数据的warp将 结果寄存器中的数据写入 lds中
    //     if (threadIdx.x / 64 < thread_n_blocks / 4) {
    //       // a_scale的话 同个线程中使用同一个a_scale，所以提前 load 到寄存器中进行复用
    //       float cur_a_scale_val = 0.0;
    //       int cur_a_scale_idx = b_blocks_idx * 16 + threadIdx.x % 16;
    //       if(cur_a_scale_idx < block_num_valid_tokens){
    //         int cur_a_scale_idx_0 = sh_block_sorted_ids[cur_a_scale_idx] / top_k;
    //         cur_a_scale_val = A_scales[cur_a_scale_idx_0];
    //         if(mul_topk_weights){
    //           float cur_topk_weight = sh_block_topk_weights[cur_a_scale_idx];
    //           cur_a_scale_val = cur_a_scale_val * cur_topk_weight;
    //         }
    //       }
    //       // // 避免线程分化
    //       // int cur_a_scale_idx = min((block_num_valid_tokens - 1), (m_block_idx * 16 + threadIdx.x % 16));
    //       // // 转成实际索引
    //       // cur_a_scale_idx = sh_block_sorted_ids[cur_a_scale_idx] / top_k;
    //       // // 获取a_scale
    //       // float cur_a_scale_val = A_scales[cur_a_scale_idx];
    //       // if(mul_topk_weights){
    //       //   float cur_topk_weight = sh_block_topk_weights[cur_a_scale_idx];
    //       //   cur_a_scale_val = cur_a_scale_val * cur_topk_weight;
    //       // }

    //       // 将 frag_c 中的结果写入到 [16, 128]的 LDS中
    //       for (int j = 0; j < 4; j++) { // 写入 [16, 64] 结果矩阵中 第 j个 [16, 16]的结果矩阵中 
    //         int wr = c_sh_wr + j;
    //         int cur_b_scale_idx = b_scale_idx_base + j;
          
    //         write(wr +  0, frag_c[b_blocks_idx][j][0], cur_a_scale_val, cur_b_scale_idx +  0);
    //         write(wr + 16, frag_c[b_blocks_idx][j][1], cur_a_scale_val, cur_b_scale_idx + 16);
    //         write(wr + 32, frag_c[b_blocks_idx][j][2], cur_a_scale_val, cur_b_scale_idx + 32);
    //         write(wr + 48, frag_c[b_blocks_idx][j][3], cur_a_scale_val, cur_b_scale_idx + 48); 
    //       }
    //     }
    //     S_BARRIER;
    //     // 将 [16, 128] LDS中已经重排好的数据 写回到全局内存
    //     for (int i = 0; i < div_ceil(16, threads / (2 * thread_n_blocks)); i++) { 
    //       int row = c_gl_wr / c_gl_stride;
    //       if (row < block_num_valid_tokens) { // 仅有效的数据才进行写入
    //         int col = c_gl_wr % c_gl_stride;
    //         int64_t sorted_row = sh_block_sorted_ids[row];
    //         int64_t true_idx = sorted_row * c_gl_stride + col; //int4
    //         C[true_idx] = sh_red[c_sh_rd];
    //         c_gl_wr += c_gl_wr_delta;
    //         c_sh_rd += c_sh_rd_delta;
    //       }
    //     }
    //     S_BARRIER;
    //   }
    // }; // 完成结果回写

    // 以正确的布局写出还原的最终结果：
    //      1、warp0 中64个线程将各自 fragc 中的数据，写到 LDS 的对应位置（按照结果矩阵的实际数据位置进行放置）
    //      2、lds-->global:
    //          (1). 如果该条带中只有一个block执行运算：warp 0-3 共256个线程，每个线程写回 128bit的数据到全局内存中的结果矩阵。
    //         （2). 多个block共同完成一个slice 的计算，则使用 atomic_add 进行结果回写。
    auto write_result = [&]() {
      // 每 8个 bf16 的数据以一个int4的类型进行store和load, 因此 “/8”
      int c_gl_stride = prob_n / 8; // 64
      // int4类型的偏移, 一个thread_n_blocks 说明横向有16个fp16的数据, 也意味着有 2个int4
      // 实际上 thread_n_blocks*16 /8 -->  2 * thread_n_blocks
      constexpr int c_sh_stride = 2 * thread_n_blocks + 1; 
      // 往 全局内存 写数据时，两次写回 的偏移
      int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks)); 
      // 从 LDS 中读取数据时，两次 读取 的偏移
      constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));

      // 写回之前，先乘 a_scale 和 b_scale
      auto write = [&](int idx, int c_int32, float top_weight_val, float a_scale, int b_scale_idx) { // 在 sh_red 的 idx 索引处插入 float转成的 fp16
        // 先 load B_scale 到 vgpr 中
        int cur_b_scale_idx =  B_scale_off + slice_col * (thread_n_blocks * 16) + b_scale_idx;
        float b_scale = B_scales[cur_b_scale_idx];
        // 反量化
        float c_float;
        if(mul_topk_weights){
          c_float =  c_int32 * top_weight_val * a_scale * b_scale;
        }else{
          c_float =  c_int32 * a_scale * b_scale;
        }
        // 转成 bf16
        ((bhalf_t*)sh_red)[idx] = f32_to_bf16(c_float);
      };
      
      // 遍历 多个thread_m_blocks, 128个线程将自己 frag_c寄存器中的数据写入到lds中
      for (int b_blocks_idx = 0;b_blocks_idx < thread_m_blocks;b_blocks_idx++){
        // 初始化写回的参数
        int c_gl_wr = c_gl_stride * (b_blocks_idx * 16 + (threadIdx.x / (2 * thread_n_blocks))) +
                    (threadIdx.x % (2 * thread_n_blocks));
        c_gl_wr += (2 * thread_n_blocks) * slice_col;
        
        int c_sh_wr;
        // c_sh_stride 是以 int4 为步长进行计算的 
        // c_sh_wr 是以 bf16 为基础单位，因此 8 * c_sh_stride
        // 64个线程第一次写入 lds的偏移量（bf16）
        c_sh_wr = (8 * c_sh_stride) * (threadIdx.x % 64 % 16) + (threadIdx.x % 64 / 16) * 4;
        c_sh_wr += 64 * (threadIdx.x / 64);
  
        int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                            (threadIdx.x % (2 * thread_n_blocks));

        int b_scale_idx_base = 64 * (threadIdx.x / 64) + (threadIdx.x % 64) / 16 * 4; 
        // 处理 B矩阵第一行数据的warp将 结果寄存器中的数据写入 lds中
        if (threadIdx.x / 64 < thread_n_blocks / 4) {
          // a_scale的话 同个线程中使用同一个a_scale，所以提前 load 到寄存器中进行复用
          float topk_weight_val = 1.0;
          float cur_a_scale_val = 0.0;
          int cur_a_scale_idx = b_blocks_idx * 16 + threadIdx.x % 16;
          if(cur_a_scale_idx < block_num_valid_tokens){
            int cur_a_scale_idx_0 = sh_block_sorted_ids[cur_a_scale_idx] / top_k;
            cur_a_scale_val = A_scales[cur_a_scale_idx_0];
            // gemm2 先乘 topk_weight 
            
            if(mul_topk_weights){
              topk_weight_val = sh_block_topk_weights[cur_a_scale_idx];
            }
          }

          // 如果gemm2的话，在这里增加一个 乘 topk_weight的操作

          // 将 frag_c 中的结果写入到 [16, 128]的 LDS中
          for (int j = 0; j < 4; j++) { // 写入 [16, 64] 结果矩阵中 第 j个 [16, 16]的结果矩阵中 
            int wr = c_sh_wr + j;
            int cur_b_scale_idx = b_scale_idx_base + j;
          
            write(wr +  0, frag_c[b_blocks_idx][j][0], topk_weight_val, cur_a_scale_val, cur_b_scale_idx +  0);
            write(wr + 16, frag_c[b_blocks_idx][j][1], topk_weight_val, cur_a_scale_val, cur_b_scale_idx + 16);
            write(wr + 32, frag_c[b_blocks_idx][j][2], topk_weight_val, cur_a_scale_val, cur_b_scale_idx + 32);
            write(wr + 48, frag_c[b_blocks_idx][j][3], topk_weight_val, cur_a_scale_val, cur_b_scale_idx + 48); 
          }
        }
        __syncthreads();

        // 将 [16, 128] LDS中已经重排好的数据 写回到全局内存
        for (int i = 0; i < div_ceil(16, threads / (2 * thread_n_blocks)); i++) { 
          int row = c_gl_wr / c_gl_stride;
          if (row < block_num_valid_tokens) { // 仅有效的数据才进行写入
            int col = c_gl_wr % c_gl_stride;
            int64_t sorted_row = sh_block_sorted_ids[row];
            int64_t true_idx = sorted_row * c_gl_stride + col; //int4
            C[true_idx] = sh_red[c_sh_rd];
            c_gl_wr += c_gl_wr_delta;
            c_sh_rd += c_sh_rd_delta;
          }
        }
        __syncthreads();
      }
    }; // 完成结果回写

    // // =============================== 流水计算的整体代码 ========================
    // 启动 全局预取 和 寄存器流水
    auto start_pipes = [&]() {
      // 发射 一组global ==> lds的指令
      for (int i = 0; i < stages - 1; i++) { // 0, 1, 2
        fetch_to_shared(i, i, i < slice_iters);
      }
      zero_accums(); // 结果寄存器 重置为 0 
      a_gl_rd_col += a_gl_rd_delta_o * (stages - 1);    
    };

    // 启动流水
    if (slice_iters) {
      start_pipes();
    }

    while(slice_iters){
      // 遍历 两级流水
      for (int pipe = 0; pipe < stages;pipe++) { // 遍历 2 段流水
        // 等待当前执行数据块的数据均 加载到 LDS 或 VGPR中
        // cp_async_wait_0(); 
        // 等待 除最后一条加载 B矩阵的 buffer_load_dwordx4指令外的全部 buffer_load_dwordx4指令
        cp_async_wait_0();

        fetch_a_to_registers(0, pipe);
        fetch_a_to_registers(1, pipe);
        // 发射下一个数据块的 全局 load 指令
        fetch_to_shared((pipe + stages - 1) % stages, pipe,
                        slice_iters >= stages);

        // for (int k = 0; k < b_sh_wr_iters; k++) {
        //   prepare_frag_b(k, pipe);
        //   // 执行 a: [16, 32] b:[32, 128] 数据块上的mmac计算
        //   matmul(k, pipe);
        // }
        // 拆成两次计算
        prepare_frag_b(0, pipe);
        // 执行 a: [16, 32] b:[32, 128] 数据块上的mmac计算
        matmul(0, pipe);
        
        prepare_frag_b(1, pipe);
        // 执行 a: [16, 32] b:[32, 128] 数据块上的mmac计算
        matmul(1, pipe);

        slice_iters--;
        if (slice_iters == 0) {
          break;
        }
      }
      a_gl_rd_col += a_gl_rd_delta_o * stages;

      // 

      if (slice_iters == 0) {
        // block 内进行归约 
        thread_block_reduce();
        
        write_result();

        init_slice(); // 初始化 slice_row、slice_col

        // 更新a 矩阵新的索引地址
        a_gl_rd_row = row;
        a_gl_rd_col = col;
      
        // 更新B 矩阵新的索引地址 先向右偏移 b_sh_stride 个 int4步长, 再向上偏移 整个条带的步长
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_idx_int[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) { // 如果转到下一个专家进行处理, 则整体左移 b_gl_stride 步长，相当于回到专家矩阵左上角第一个数据块
          for (int i = 0; i < b_sh_wr_iters; i++) B_idx_int[i] -= b_gl_stride;
        }
        // 启动下一阶段的流水 
        start_pipes();
      } 
    } // 结束 while 循环

  }//  结束 Marlin 算子循环

}// namespace MARLIN_NAMESPACE_NAME
}
}
