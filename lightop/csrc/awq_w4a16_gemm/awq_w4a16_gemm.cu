#include "awq_w4a16_common.h"

namespace at {
namespace native {




  template <typename scalar_t, int NUM_K_WARPS,int NPerWarp,int KPerWarp, int NUM_N_WARPS, bool UseMarlin>
  __global__ void __launch_bounds__(NUM_K_WARPS*NUM_N_WARPS*C10_WARP_SIZE, 2)  gemm_awq_w4a16_kernel(scalar_t* d, scalar_t* a, int32_t* b, int32_t* scale_zeros, int M, int N, int K, int GroupSize)
  {
  
    auto scale_n = reinterpret_cast<const half2_t*>(scale_zeros);
    constexpr int KSizePerWarp = KPerWarp;            // 每个warp计算的k的大小
    constexpr int NSizePerWarp = 16;                // 每个warp计算的n的大小
    constexpr int tc_number    = KSizePerWarp / 16; // 每个warp调用tensor core的个数  
    constexpr int AWQQuantSize = 8;
    constexpr int NLoop = NPerWarp / NSizePerWarp;
    constexpr int k_per_thread = tc_number * 4;
    constexpr int warpSize = 64;
    constexpr int kNWarps = NUM_K_WARPS;
    constexpr int KSizePerWarpQuant = KPerWarp / AWQQuantSize; // 我们的warpk=128， 量化后压缩成16
    constexpr int BlockM = 16;
    
    const int bidm = blockIdx.x;
    a += bidm * BlockM * K;
    d += bidm * BlockM * N;
    M = MIN(BlockM, M - bidm * BlockM);

    const int num_element_per_n_warp = NUM_K_WARPS > 1 ? NUM_K_WARPS/2*NPerWarp*M : NPerWarp*M;
    const int StrideA  = K;
    const int StrideB  = K;
    const int StrideC  = N;
    
    
    

    const int warp_n_idx = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int bwid       = __builtin_amdgcn_readfirstlane(blockIdx.y*NUM_N_WARPS+warp_n_idx);
    const int tid       = threadIdx.x;
    const int warp_idx  = __builtin_amdgcn_readfirstlane(tid / warpSize);
    const int lane      = tid % warpSize;
    const int row_id    = lane % NSizePerWarp;
    const int col_id    = lane / NSizePerWarp;  
    const bool row_id_valid = row_id < M;
    const int N_lenth = __builtin_amdgcn_readfirstlane(N - NPerWarp*bwid);
    if(N_lenth <=0) return;
  
    const int block_n = NPerWarp * bwid;
    d += block_n;
    // b += block_n * StrideB / AWQQuantSize;
    if constexpr(UseMarlin){
      // b += block_n * StrideB / AWQQuantSize;
      b +=  block_n * KSizePerWarpQuant;
    } else {
      b += block_n * StrideB / AWQQuantSize;
    }
    scale_n += block_n * StrideB / GroupSize;
    
    using A_VEC_TYPE = __attribute__( (__vector_size__(k_per_thread * sizeof(_Float16)) )) _Float16;
    using B_VEC_TYPE = __attribute__( (__vector_size__((k_per_thread / AWQQuantSize) * sizeof(int32_t)) )) int32_t;
  
    A_VEC_TYPE a_vec;
    float4_t d_vec[NLoop]; 
    #pragma unroll
    for(int i = 0; i < NLoop; i++) {
        d_vec[i] = {0, 0, 0, 0};
    }
    for(int i = warp_idx * KSizePerWarp; i < K; i += KSizePerWarp * kNWarps) {
      const int k_idx = i + col_id * k_per_thread;
      if(row_id < M) {
        a_vec = *(A_VEC_TYPE*)(a + row_id * StrideA + k_idx);
      }
      for(int nn = 0; nn < NLoop; nn++) {
        const int n_idx = NSizePerWarp * nn + row_id;
        // int weight_offset = (n_idx * StrideB + k_idx) / AWQQuantSize;
        int weight_offset;
        const int line_offset = n_idx * StrideB + k_idx;
        if constexpr(UseMarlin){
          weight_offset = n_idx * KSizePerWarpQuant + (col_id*k_per_thread)/AWQQuantSize + (k_idx / KPerWarp)*N*KSizePerWarpQuant;
        } else {
          weight_offset = line_offset / AWQQuantSize;
        }
        auto tb         = *(B_VEC_TYPE*)(b + weight_offset);
        half2_t t_scale = scale_n[line_offset / GroupSize];
        half8_t b_vec;
        #pragma unroll
        for(int ii = 0; ii < tc_number/2; ii++) {
            dequantize_s4_to_fp16x2(tb[ii], (uint32_t*)&b_vec);
            auto b_vec_half = reinterpret_cast<half_t*>(&b_vec);
            #pragma unroll
            for(int kk = 0; kk < 8; kk++) {
                b_vec_half[kk]  = (b_vec_half[kk] - t_scale[0]) * t_scale[1];
            }
            builtin_amdgcn_mmac(*((half4_t*)&a_vec + ii * 2), *((half4_t*)&b_vec), d_vec[nn]);
            builtin_amdgcn_mmac(*((half4_t*)&a_vec + ii * 2 + 1), *((half4_t*)&b_vec + 1), d_vec[nn]);
        } 
      }    
    }
    extern __shared__ float start_smem[];
    float4_t* out_smem = reinterpret_cast<float4_t*>(start_smem + num_element_per_n_warp*warp_n_idx);
    half_t* out_smem_output_ptr = reinterpret_cast<half_t*>(start_smem) + num_element_per_n_warp*warp_n_idx;
  
    if constexpr(NUM_K_WARPS>1){
        int lds_size_n_offset=NUM_K_WARPS/2*16*M/4;
          #pragma unroll
          for (int i = NUM_K_WARPS; i > 1; i /= 2) {
              int mid = i / 2;
              if(row_id_valid)
              {
                if (warp_idx >= mid && warp_idx < i) {
                    for(int nn=0;nn<NLoop;nn++){
                        out_smem[(warp_idx - mid) * 4 * M+row_id*4+col_id+lds_size_n_offset*nn]=d_vec[nn];
                    }
                }
              }
              __syncthreads();
              if (warp_idx < mid) {
                  for(int nn=0;nn<NLoop;nn++){
                    if(row_id_valid)
                    {
                      float4_t tmp=out_smem[warp_idx* 4 * M+row_id*4+col_id+lds_size_n_offset*nn];
                      #pragma unroll
                      for (int i = 0; i < 4; i++) {
                          d_vec[nn][i] += tmp[i];
                      }
                    }
                  }
              }
              __syncthreads();
          }
    }
    using out_vec_dtype = half4_t;
    out_vec_dtype out_vec_16bit;
    if(warp_idx == 0) {
      if (row_id_valid) 
      { 
        #pragma unroll
        for(int nn = 0; nn < NLoop; nn++) {
            int out_smem_offset = nn*NSizePerWarp*M;
            #pragma unroll
            for(int i = 0; i < 4; i++) {
              out_smem_output_ptr[out_smem_offset + row_id * NSizePerWarp + i*4 + col_id] = static_cast<half_t>(d_vec[nn][i]);
            }
        }
      }
    }
    __syncthreads();   
    {
      if(row_id_valid)
      {
        for(int nn = warp_idx; nn < NLoop; nn+=NUM_K_WARPS) {
          int out_smem_offset = nn*NSizePerWarp*M;
          const bool n_idx_valid = nn*NSizePerWarp+col_id * 4 < N_lenth;
          if(n_idx_valid)
          {
            out_vec_16bit = *reinterpret_cast<out_vec_dtype*>(out_smem_output_ptr + out_smem_offset + row_id * NSizePerWarp + col_id * 4);
            *reinterpret_cast<out_vec_dtype*>(d + row_id * N + nn * NSizePerWarp + col_id * 4) = out_vec_16bit;
          }
        }
      }
  
    }
  
  }


  template <typename scalar_t, int NUM_K_WARPS,int NPerWarp,int KPerWarp, int NUM_N_WARPS, bool UseMarlin>
  __global__ void gemm_awq_w4a16_kernel_with_splitk(scalar_t* d, scalar_t* a, int32_t* b, int32_t* scale_zeros, float* splitk_work_space, int* semaphores, int M, int N, int K, int GroupSize)
  {
  
    auto scale_n      = reinterpret_cast<const half2_t*>(scale_zeros);
  
    constexpr int KSizePerWarp = KPerWarp;            // 每个warp计算的k的大小
    constexpr int NSizePerWarp = 16;                // 每个warp计算的n的大小
    constexpr int tc_number    = KSizePerWarp / 16; // 每个warp调用tensor core的个数
    constexpr int AWQQuantSize = 8;
    constexpr int NLoop = NPerWarp / NSizePerWarp;
    constexpr int k_per_thread = tc_number * 4;
    constexpr int kNWarps = NUM_K_WARPS;
    constexpr int KSizePerWarpQuant = KPerWarp / AWQQuantSize; // 我们的warpk=128， 量化后压缩成16
    constexpr int BlockM = 16;


    const int bidm = __builtin_amdgcn_readfirstlane(blockIdx.x);
    a += bidm * BlockM * K;
    d += bidm * BlockM * N;
    splitk_work_space += bidm * BlockM * N;
    // semaphores += bidm*1024;
    semaphores = semaphores + bidm*(gridDim.y*NUM_N_WARPS);
    M = MIN(BlockM, M - bidm * BlockM);

    const int bidy      = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int bidx      = __builtin_amdgcn_readfirstlane(blockIdx.z);  // splitk

    const int tid       = threadIdx.x;
    const int bwid = __builtin_amdgcn_readfirstlane(bidy*NUM_N_WARPS + threadIdx.y);
    const int warp_idx  = __builtin_amdgcn_readfirstlane(tid / warpSize);
    const int lane      = tid % warpSize;
    const int row_id    = lane % NSizePerWarp;
    const int col_id    = lane / NSizePerWarp;
    const int SplitKBatch = __builtin_amdgcn_readfirstlane(gridDim.z);
    const int part_k    = ((K / KSizePerWarp - 1) / SplitKBatch + 1) * KSizePerWarp;
    const int N_lenth = N - NPerWarp*bwid;
    if(N_lenth<=0) return;
    const int StrideA              = K;
    const int StrideB              = K;
    const int StrideC              = N;
    const int warp_n_idx = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int  num_element_per_n_warp= NUM_K_WARPS > 1 ? NUM_K_WARPS/2*NPerWarp*M : 0;    
  
    float4_t d_vec[NLoop];
    #pragma unroll
    for(int i = 0; i < NLoop; i++) {
        d_vec[i] = {0, 0, 0, 0};
    }
    const int block_n = NPerWarp * bwid;



    a += bidx * part_k;
    d += block_n;
    // b += (block_n * StrideB + part_k * bidx) / AWQQuantSize;
    if constexpr(UseMarlin){
      b +=  block_n * KSizePerWarpQuant  + (part_k * bidx) / KSizePerWarp * (N * KSizePerWarpQuant);
    } else {
      b += (block_n * StrideB + part_k * bidx) / AWQQuantSize;
    }


    scale_n += (block_n * StrideB + part_k * bidx) / GroupSize;
    splitk_work_space += block_n;
    const int length_per_block = (bidx == SplitKBatch - 1 ? K - part_k * bidx : part_k);
    using A_VEC_TYPE = __attribute__( (__vector_size__(k_per_thread * sizeof(_Float16)) )) _Float16;
    using B_VEC_TYPE = __attribute__( (__vector_size__((k_per_thread / AWQQuantSize) * sizeof(int32_t)) )) int32_t;
    A_VEC_TYPE a_vec;
    #pragma unroll
    // for(int i = 0; i < k_per_thread; i++) {
        // *((half_t*)&a_vec + i) = 0;
    // } 
      
    for(int i = warp_idx * KSizePerWarp; i < length_per_block; i += KSizePerWarp * kNWarps) {
      if(row_id < M) {
          a_vec = *(A_VEC_TYPE*)(a + row_id * StrideA + i + col_id * k_per_thread);
      }
      const int k_idx = i + col_id * k_per_thread;
      #pragma unroll
      for(int nn = 0; nn < NLoop; nn++) {
        const int n_idx = NSizePerWarp * nn + row_id;
        // int line_offset = n_idx * StrideB + k_idx;
        int weight_offset;
        const int line_offset = n_idx * StrideB + k_idx;
        if constexpr(UseMarlin){
          weight_offset = n_idx * KSizePerWarpQuant + (col_id*k_per_thread)/AWQQuantSize + (k_idx / KPerWarp)*N*KSizePerWarpQuant;
        } else {
          weight_offset = line_offset / AWQQuantSize;
        }
        auto tb         = *(B_VEC_TYPE*)(b + weight_offset);  
        half2_t t_scale = scale_n[line_offset / GroupSize];
        half8_t b_vec;
        #pragma unroll
        for(int ii = 0; ii < tc_number/2; ii++) {
            dequantize_s4_to_fp16x2(tb[ii], (uint32_t*)&b_vec);
            auto b_vec_half = reinterpret_cast<half_t*>(&b_vec);
            #pragma unroll
            for(int kk = 0; kk < 8; kk++) {
                b_vec_half[kk]  = (b_vec_half[kk] - t_scale[0]) * t_scale[1];
            }
            builtin_amdgcn_mmac(*((half4_t*)&a_vec + ii * 2), *((half4_t*)&b_vec), d_vec[nn]);
            builtin_amdgcn_mmac(*((half4_t*)&a_vec + ii * 2 + 1), *((half4_t*)&b_vec + 1), d_vec[nn]);
        } 

      }
    }
  
    extern __shared__ float start_smem[];
    float4_t* out_smem = reinterpret_cast<float4_t*>(start_smem + num_element_per_n_warp*warp_n_idx);
    if constexpr(NUM_K_WARPS>1){
      const int lds_size_n_offset=NUM_K_WARPS/2*16*M/4;
      if(row_id<M){
          #pragma unroll
          for (int i = NUM_K_WARPS; i > 1; i /= 2) {
              int mid = i / 2;
              if (warp_idx >= mid && warp_idx < i) {
                  for(int nn=0;nn<NLoop;nn++){
                      out_smem[(warp_idx - mid) * 4 * M+row_id*4+col_id+lds_size_n_offset*nn]=d_vec[nn];
                  }
              }
              __syncthreads();
              if (warp_idx < mid) {
                  for(int nn=0;nn<NLoop;nn++){
                      float4_t tmp=out_smem[warp_idx* 4 * M+row_id*4+col_id+lds_size_n_offset*nn];
                      #pragma unroll
                      for (int i = 0; i < 4; i++) {
                          d_vec[nn][i] += tmp[i];
                      }
                  }
              }
              __syncthreads();
          }
      }
    }
  
    using out_vec_dtype = half4_t;
    if(warp_idx == 0) {
      if(row_id < M) {
        for(int nn=0;nn<NLoop;nn++){
            for(int i=0;i<4;i++){
              if(nn*NSizePerWarp+col_id + i*4 < N_lenth){
                atomicAdd(splitk_work_space+nn*NSizePerWarp+i*4+row_id*N+col_id,d_vec[nn][i]);
              }
            }
        }
      }  
      int order=-1;  
      __syncthreads();
      if(tid==0)order = atomicAdd(semaphores+bwid,1); 
      order=__shfl(order,0);
      if(order==SplitKBatch-1){
        if(tid==0)semaphores[bwid]=0;
        for(int i=tid*4;i<M*NPerWarp;i+=256){
          // const bool n_idx_valid  = __builtin_amdgcn_readfirstlane(i%NPerWarp<N_lenth);
          const bool n_idx_valid = i%NPerWarp<N_lenth;        
          if(n_idx_valid)
          {
            const int offset=i/NPerWarp*N+i%NPerWarp; // 输出元素的索引
            float4_t out_vec_float = *(float4_t*)(splitk_work_space+offset);
            *(float4_t*)(splitk_work_space+offset) = {0.0f,0.0f,0.0f,0.0f};
            out_vec_dtype out_vec_16bit;
            #pragma unroll
            for(int ii=0;ii<4;ii++){
              out_vec_16bit[ii] = static_cast<half_t>(out_vec_float[ii]);
            }
            *(out_vec_dtype*)(d + offset) = out_vec_16bit;
          }
        }
      }
    }
  }


struct GPTQ_GEMM_W4A16_WorkSpace {
    float* splitk_work_space;
    int* semaphores;
    int gpu_id;
    int process_id;
    pid_t thread_id;
    
    GPTQ_GEMM_W4A16_WorkSpace() : splitk_work_space(nullptr), semaphores(nullptr) {
        hipGetDevice(&gpu_id);
        process_id = getpid();
        thread_id = syscall(SYS_gettid);
        // printf("gemm_w8_a8_smooth_colmajor Current GPU ID: %d, Process ID: %d, Thread ID: %d, hipmalloc\n", gpu_id, process_id, thread_id);
        
        hipMalloc(&splitk_work_space, 100000000); // 100m
        hipMemset(splitk_work_space, 0, 100000000);
        hipMalloc(&semaphores, 1000000); // 10m
        hipMemset(semaphores, 0, 1000000); // 10m
    }
    
    ~GPTQ_GEMM_W4A16_WorkSpace() {
        if (splitk_work_space) {
            hipFree(splitk_work_space);
            // printf("gemm_w8_a8_smooth_colmajor GPU ID: %d, Process ID: %d, Thread ID: %d, hipfree splitk_work_space\n", gpu_id, process_id, thread_id);
        }
        if (semaphores) {
            hipFree(semaphores);
            // printf("gemm_w8_a8_smooth_colmajor GPU ID: %d, Process ID: %d, Thread ID: %d, hipfree semaphores\n", gpu_id, process_id, thread_id);
        }
    }
};
  
void gemm_awq_w4a16(Tensor& ret,const Tensor& x,const Tensor& weight,const Tensor& scales_zeros, 
  int num_n_warps, int num_k_warps,int n_per_warp,int k_per_warp, int splitk, bool use_marlin)
{
  int temp1;  // 用于让setup.py感知到修改
  const int M = x.sizes()[0];
  const int K = x.sizes()[1];
  const int GroupSize = K / scales_zeros.sizes()[1];
  auto stream = at::cuda::getCurrentCUDAStream();
  auto out_dtype = x.scalar_type();

  // printf("M=%d, N=%d, K=%d, G=%d, use_marlin=%d\n", M, N, K, G, use_marlin);
  
  #define UseStaticWorkSpace 1
  #if UseStaticWorkSpace
      static GPTQ_GEMM_W4A16_WorkSpace* workspace = nullptr;
      if(workspace == nullptr){
          workspace = new GPTQ_GEMM_W4A16_WorkSpace();
      }
      float* splitk_work_space = workspace->splitk_work_space;
      int* semaphores = workspace->semaphores;
  #else
      float* splitk_work_space = (float*)work_space.data_ptr();  // 100M个float
      int* semaphores = (int*)splitk_work_space + 100*1024*1024;  // 10M个int
  #endif
  UseMarlin_SWITCH(use_marlin,[&] {
    NPerWarp_SWITCH(n_per_warp,[&] {
      NUM_N_WARPS_SWITCH(num_n_warps,[&] {
        NUM_K_WARPS_SWITCH(num_k_warps,[&] {
          KPerWarp_SWITCH(k_per_warp,[&] {
            Output_Type_SWITCH(out_dtype,[&] {
              constexpr int BlockM = 16;
              const int N = use_marlin ? weight.sizes()[1]/(KPerWarp/8) : weight.sizes()[0];
              if(splitk>1){
                const int  lds_size= NUM_K_WARPS > 1 ? NUM_K_WARPS/2*NPerWarp*(MIN(BlockM,M))*sizeof(float) : 0;
                const int all_lds_size = lds_size*NUM_N_WARPS;
                const int blocks = (N-1)/(NPerWarp*NUM_N_WARPS)+1;
                const int m_blocks = (M-1)/(BlockM)+1;

                gemm_awq_w4a16_kernel_with_splitk<scalar_t,NUM_K_WARPS,NPerWarp,KPerWarp,NUM_N_WARPS,UseMarlin><<<dim3(m_blocks,blocks,splitk),dim3(NUM_K_WARPS*C10_WARP_SIZE, NUM_N_WARPS),all_lds_size,stream>>>
                ((scalar_t*)ret.data_ptr(),(scalar_t*)x.data_ptr(),(int32_t*)weight.data_ptr(),(int32_t*)scales_zeros.data_ptr(),splitk_work_space,semaphores,M,N,K, GroupSize);
              }
              else{
                const int blocks = (N-1)/(NPerWarp*NUM_N_WARPS)+1;
                const int m_blocks = (M-1)/(BlockM)+1;
                const int  lds_size= NUM_K_WARPS > 1 ? NUM_K_WARPS/2*NPerWarp*MIN(BlockM,M)*sizeof(float) : NPerWarp*(MIN(BlockM,M))*sizeof(half_t);
                const int all_lds_size = lds_size*NUM_N_WARPS;
                gemm_awq_w4a16_kernel<scalar_t,NUM_K_WARPS,NPerWarp,KPerWarp,NUM_N_WARPS,UseMarlin><<<dim3(m_blocks,blocks),dim3(NUM_K_WARPS*C10_WARP_SIZE, NUM_N_WARPS),all_lds_size,stream>>>
                ((scalar_t*)ret.data_ptr(),(scalar_t*)x.data_ptr(),(int32_t*)weight.data_ptr(),(int32_t*)scales_zeros.data_ptr(),M,N,K, GroupSize); 
              }

            });
          });
        });
      });
    });
  });


}
 

}
}
