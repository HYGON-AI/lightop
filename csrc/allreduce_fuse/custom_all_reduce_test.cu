/**
 * This is a standalone test for custom allreduce.
 * To compile, make sure you have MPI and NCCL installed in your system.
 * export MPI_HOME=XXX
 * nvcc -O2 -arch=native -std=c++17 custom_all_reduce_test.cu -o
 * custom_all_reduce_test -lnccl -I${MPI_HOME}/include -lmpi
 *
 * Warning: this C++ test is not designed to be very readable and was used
 * during the rapid prototyping process.
 *
 * To run:
 * mpirun --allow-run-as-root -np 8 ./custom_all_reduce_test
 */
#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#include <limits>
#include <vector>
#include <random>

#include "cuda_profiler_api.h"
#include "custom_all_reduce.cuh"
#include "mpi.h"
#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
typedef __hip_bfloat16 nv_bfloat16;
  #include "rccl/rccl.h"
  #include "custom_all_reduce_hip.cuh"
#else
  #include "nccl.h"
  #include "custom_all_reduce.cuh"
#endif

#define MPICHECK(cmd)                                                  \
  do {                                                                 \
    int e = cmd;                                                       \
    if (e != MPI_SUCCESS) {                                            \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

#define NCCLCHECK(cmd)                                              \
  do {                                                              \
    ncclResult_t r = cmd;                                           \
    if (r != ncclSuccess) {                                         \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
             ncclGetErrorString(r));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

#ifdef USE_ROCM
__global__ void dummy_kernel() {
  for (int i = 0; i < 100; i++) {
    uint64_t start = wall_clock64();
    uint64_t cycles_elapsed;
    do {
      cycles_elapsed = wall_clock64() - start;
    } while (cycles_elapsed < 100);
  }
  for (int i = 0; i < 100; i++) __nanosleep(1000000);  // 100ms
}
#else
__global__ void dummy_kernel() {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  for (int i = 0; i < 100; i++) __nanosleep(1000000);  // 100ms
  #else
  for (int i = 0; i < 100; i++) {
    long long int start = clock64();
    while (clock64() - start < 150000000);  // approximately 98.4ms on P40
  }
  #endif
}
#endif

template <typename T>
__global__ void set_data(T* data, int size, int myRank) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    data[idx] = myRank * 0.11f;
  }
}

template <typename T>
__global__ void convert_data(const T* data1, const T* data2, double* fdata1,
                             double* fdata2, int size) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    fdata1[idx] = data1[idx];
    fdata2[idx] = data2[idx];
  }
}

__global__ void init_rand(curandState_t* state, int size, int nRanks) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    for (int i = 0; i < nRanks; i++) {
      curand_init(i + 1, idx, 0, &state[idx * nRanks + i]);
    }
  }
}

template <typename T>
__global__ void gen_data(curandState_t* state, T* data, double* ground_truth,
                         int myRank, int nRanks, int size) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    double sum = 0.0;
    for (int i = 0; i < nRanks; i++) {
      double val = curand_uniform_double(&state[idx * nRanks + i]) * 4;
      T hval = val;  // downcast first
      sum += static_cast<double>(hval);
      if (i == myRank) data[idx] = hval;
    }
    ground_truth[idx] = sum;
  }
}
/*************************************************/
template <typename T,int reducesize=64>
__inline__ __device__ T WarpReduceSum_NEW(T val) {
#pragma unroll
  for (int offset = reducesize/2; offset > 0; offset >>= 1) {
    val += __shfl_down(val, offset);
  }
  return val;
}

template <typename T,int block_size=512>
__inline__ __device__ T BlockReduceSum_NEW(T val, T* shared) {
  constexpr int share_size=block_size/64;
  val = WarpReduceSum_NEW<T>(val);
  if constexpr(block_size==64)
  {
    return val;
  }
  else{
    const int lid = threadIdx.x % 64;
    const int wid = threadIdx.x / 64;
    if (lid == 0&&wid<share_size) {
      shared[wid] = val;
    }
    __syncthreads();
    if (wid == 0&&lid<share_size) {
      val = WarpReduceSum_NEW<T,share_size>(shared[lid]);
    }
    return val;
  }
}

template <typename scalar_t,typename T_ACC,int Vec=4,int block_size=512>
__global__ void fused_add_rms_kernel_opt(scalar_t* input,scalar_t* residual,scalar_t* gamma,int cols,T_ACC eps)
{
  constexpr int share_size=block_size/64;
  __shared__ T_ACC val_shared[share_size];
  __shared__ T_ACC s_rstd;
  T_ACC val=0;
  int i=blockIdx.x;
  int j=threadIdx.x;
  int tcol=cols/Vec;
  using LoadT = typename vllm::packed_t<scalar_t>::P;
  scalar_t intput_vec[Vec];
  scalar_t residual_vec[Vec];
  T_ACC trstd;
  int64_t idx = i * tcol + j;
  idx*=Vec;
  if (j < tcol) {
    *(LoadT*)intput_vec = *(LoadT*)(input+idx);
    *(LoadT*)residual_vec = *(LoadT*)(residual+idx);
    #pragma unroll
    for (int ii = 0; ii < Vec; ii++) {
      residual_vec[ii]+=intput_vec[ii];
      val += static_cast<T_ACC>(residual_vec[ii])*static_cast<T_ACC>(residual_vec[ii]);
    }
  }
  val = BlockReduceSum_NEW<T_ACC,block_size>(val,val_shared);
  if (j == 0) s_rstd=rsqrtf(val/cols + eps);
  __syncthreads();
  trstd=s_rstd;
  if (j < tcol) {
    #pragma unroll
    for(int ii=0;ii<Vec;ii++){
      int jj=j*Vec+ii;
      intput_vec[ii] = static_cast<T_ACC>(residual_vec[ii]) *trstd* static_cast<T_ACC>(gamma[jj]);
    }
    *(LoadT*)(residual+idx)=*(LoadT*)residual_vec;
    *(LoadT*)(input+idx)=*(LoadT*)intput_vec;
  }
}
template <typename scalar_t>
void fused_add_rms_norm_choose(cudaStream_t stream, scalar_t* self_data, scalar_t* other_data, 
                              scalar_t*weight_data, double eps, int hidden_size, int num_tokens) {
  if (hidden_size<=1024){
      fused_add_rms_kernel_opt<scalar_t,float,8,128><<<num_tokens,  128, 0, stream>>>(self_data,other_data,weight_data,hidden_size,eps);
  }
  else if(hidden_size<=2048){
      fused_add_rms_kernel_opt<scalar_t,float,8,256><<<num_tokens,  256, 0, stream>>>(self_data,other_data,weight_data,hidden_size,eps);
  }
  else if(hidden_size<=4096){
      if(num_tokens>1200){
        fused_add_rms_kernel_opt<scalar_t,float,8,512><<<num_tokens,  512, 0, stream>>>(self_data,other_data,weight_data,hidden_size,eps);
      }
      else{
        fused_add_rms_kernel_opt<scalar_t,float,4,1024><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,eps);
      }
  }
  else if(hidden_size<=8192){
      fused_add_rms_kernel_opt<scalar_t,float,8,1024><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,eps);
  }
  else{
      fused_add_rms_kernel_opt<scalar_t,float,16,1024><<<num_tokens,  1024, 0, stream>>>(self_data,other_data,weight_data,hidden_size,eps);
  } 
}
/*****************************************************************/

template <typename T>
void run(int myRank, int nRanks, ncclComm_t& comm, int threads, int block_limit,
         int data_size, bool performance_test, int hidden_dim) {
  T* result_ori, *result_fuse;
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDACHECK(cudaMalloc(&result_ori, data_size * sizeof(T)));
  CUDACHECK(cudaMemset(result_ori, 0, data_size * sizeof(T)));
  CUDACHECK(cudaMalloc(&result_fuse, data_size * sizeof(T)));
  CUDACHECK(cudaMemset(result_fuse, 0, data_size * sizeof(T)));
  cudaIpcMemHandle_t self_data_handle;
  cudaIpcMemHandle_t data_handles[8];
  vllm::Signal* buffer;
  T* self_data_copy;
  /**
   * Allocate IPC buffer
   *
   * The first section is a temporary buffer for storing intermediate allreduce
   * results, if a particular algorithm requires it. The second section is for
   * the input to the allreduce. The actual API takes the input pointer as an
   * argument (that is, they can and usually should be allocated separately).
   * But since the input pointers and the temporary buffer all require IPC
   * registration, they are allocated and registered together in the test for
   * convenience.
   */
#ifdef USE_ROCM
  CUDACHECK(hipExtMallocWithFlags(
      (void**)&buffer, 2 * data_size * sizeof(T) + sizeof(vllm::Signal),
      hipDeviceMallocUncached));
#else
  CUDACHECK(
      cudaMalloc(&buffer, 2 * data_size * sizeof(T) + sizeof(vllm::Signal)));
#endif
  CUDACHECK(
      cudaMemset(buffer, 0, 2 * data_size * sizeof(T) + sizeof(vllm::Signal)));
  CUDACHECK(cudaMalloc(&self_data_copy, data_size * sizeof(T)));
  CUDACHECK(cudaIpcGetMemHandle(&self_data_handle, buffer));

  MPICHECK(MPI_Allgather(&self_data_handle, sizeof(cudaIpcMemHandle_t),
                         MPI_BYTE, data_handles, sizeof(cudaIpcMemHandle_t),
                         MPI_BYTE, MPI_COMM_WORLD));

  void* rank_data;
  size_t rank_data_sz = 16 * 1024 * 1024;
  CUDACHECK(cudaMalloc(&rank_data, rank_data_sz));
  vllm::Signal* ipc_ptrs[8];
  for (int i = 0; i < nRanks; i++) {
    if (i == myRank)
      ipc_ptrs[i] = buffer;
    else
      CUDACHECK(cudaIpcOpenMemHandle((void**)&ipc_ptrs[i], data_handles[i],
                                     cudaIpcMemLazyEnablePeerAccess));
  }
  vllm::CustomAllreduce fa(ipc_ptrs, rank_data, rank_data_sz, myRank, nRanks);
  auto* self_data =
      reinterpret_cast<T*>(reinterpret_cast<char*>(buffer) +
                           sizeof(vllm::Signal) + data_size * sizeof(T));
  // hack buffer registration
  {
    void* data[8]; //gpu数据部分
    for (int i = 0; i < nRanks; i++) {
      data[i] =
          ((char*)ipc_ptrs[i]) + sizeof(vllm::Signal) + data_size * sizeof(T);
    }
    fa.register_buffer(data);
  }

  double* ground_truth;
  CUDACHECK(cudaMallocHost(&ground_truth, data_size * sizeof(double)));
  curandState_t* states;
  CUDACHECK(cudaMalloc(&states, sizeof(curandState_t) * nRanks * data_size));
  init_rand<<<108, 1024, 0, stream>>>(states, data_size, nRanks);
  gen_data<T><<<108, 1024, 0, stream>>>(states, self_data, ground_truth, myRank,
                                        nRanks, data_size);
  CUDACHECK(cudaMemcpyAsync(self_data_copy, self_data, data_size * sizeof(T),
                            cudaMemcpyDeviceToDevice, stream));
  cudaEvent_t start, stop;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  /*******************************/
  int token_num = data_size / hidden_dim;
  T* residual_h, *residual_d, *weight_h, *weight_d;
  residual_h = (T*)malloc(data_size * sizeof(T));
	std::random_device rd;  // 用于获取随机数种子
	std::mt19937 gen(7);
	std::uniform_real_distribution<float> dis(-3.0f, 3.0f);
  for (int i = 0; i < data_size; ++i)
    residual_h[i] = static_cast<T>(dis(gen));
  for (int i = 0; i < hidden_dim; ++i)
    weight_h[i] = static_cast<T>(dis(gen));
  
  cudaMalloc((void**)&residual_d, sizeof(T)*data_size);
  cudaMalloc((void**)&weight_d, sizeof(T)*hidden_dim);

  cudaMemcpyAsync(residual_d, residual_h, sizeof(T)*data_size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(weight_d, weight_h, sizeof(T)*hidden_dim, cudaMemcpyHostToDevice, stream);

  float eps = 1.0f;
  /*******************************/
  ncclDataType_t ncclDtype;
  if (std::is_same<T, half>::value) {
    ncclDtype = ncclFloat16;
  } else if (std::is_same<T, nv_bfloat16>::value) {
    ncclDtype = ncclBfloat16;
  } else {
    ncclDtype = ncclFloat;
  }
  double *nccl_result, *my_result;
  CUDACHECK(cudaMallocHost(&nccl_result, data_size * sizeof(double)));
  CUDACHECK(cudaMallocHost(&my_result, data_size * sizeof(double)));
  if (performance_test) {
    dummy_kernel<<<1, 1, 0, stream>>>();
    constexpr int warmup_iters = 5;
    constexpr int num_iters = 10;
    // warmup
    for (int i = 0; i < warmup_iters; i++) {
      // NCCLCHECK(ncclAllReduce(result, result, data_size, ncclDtype, ncclSum,
      //                         comm, stream));
      fa.allreduce<T>(stream, self_data, result_ori, data_size, threads, block_limit);
      fused_add_rms_norm_choose<T>(stream, result_ori, residual_d, weight_d, 1.0, hidden_dim, token_num);
    }
    CUDACHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < num_iters; i++) {
      // NCCLCHECK(ncclAllReduce(result, result, data_size, ncclDtype, ncclSum,
      //                         comm, stream));
      fa.allreduce<T>(stream, self_data, result_ori, data_size, threads, block_limit);
      fused_add_rms_norm_choose<T>(stream, result_ori, residual_d, weight_d, 1.0, hidden_dim, token_num);
    }
    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    float allreduce_ms = 0;
    cudaEventElapsedTime(&allreduce_ms, start, stop);

    dummy_kernel<<<1, 1, 0, stream>>>();
    // warm up
    for (int i = 0; i < warmup_iters; i++) {
      fa.allreduce_fuse_norm<T>(stream, self_data, result_fuse, data_size, token_num,
                            hidden_dim, residual_d, weight_d, eps,
                            threads, block_limit);
    }
    CUDACHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < num_iters; i++) {
      fa.allreduce_fuse_norm<T>(stream, self_data, result_fuse, data_size, token_num,
                            hidden_dim, residual_d, weight_d, eps,
                            threads, block_limit);
    }
    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    float duration_ms = 0;
    cudaEventElapsedTime(&duration_ms, start, stop);
    if (myRank == 0)
      printf(
          "Rank %d done, nGPUs:%d, sz (kb): %d, %d, %d, allreduse_fuse_norm time:%.2fus, allreduce+norm "
          "time:%.2fus\n",
          myRank, nRanks, data_size * sizeof(T) / 1024, threads, block_limit,
          duration_ms * 1e3 / num_iters, allreduce_ms * 1e3 / num_iters);

    // And wait for all the queued up work to complete
    CUDACHECK(cudaStreamSynchronize(stream));

    NCCLCHECK(ncclAllReduce(self_data_copy, self_data, data_size, ncclDtype,
                            ncclSum, comm, stream));
    fused_add_rms_norm_choose<T>(stream, self_data, residual_d, weight_d, 1.0, hidden_dim, token_num);

    convert_data<T><<<108, 1024, 0, stream>>>(result_ori, result_fuse, nccl_result,
                                              my_result, data_size);
    CUDACHECK(cudaStreamSynchronize(stream));

    for (unsigned long j = 0; j < data_size; j++) {
      auto diff = abs(nccl_result[j] - my_result[j]);
      if (diff >= 4e-2) {
        printf("Rank %d: Verification mismatch at %lld: %f != (my) %f, gt=%f\n",
               myRank, j, nccl_result[j], my_result[j], ground_truth[j]);
        break;
      }
    }
    long double nccl_diffs = 0.0;
    long double my_diffs = 0.0;
    for (int j = 0; j < data_size; j++) {
      nccl_diffs += abs(nccl_result[j] - ground_truth[j]);
      my_diffs += abs(my_result[j] - ground_truth[j]);
    }
    if (myRank == 0)
      std::cout << "average abs diffs: nccl: " << nccl_diffs / data_size
                << " me: " << my_diffs / data_size << std::endl;
  } else {
    for (int i = 0; i < 100; i++) {
      fa.allreduce<T>(stream, self_data, result_ori, data_size, threads,
                      block_limit);
      CUDACHECK(cudaStreamSynchronize(stream));
      NCCLCHECK(ncclAllReduce(self_data, self_data_copy, data_size, ncclDtype,
                              ncclSum, comm, stream));
      convert_data<T><<<108, 1024, 0, stream>>>(
          self_data_copy, result_ori, nccl_result, my_result, data_size);
      CUDACHECK(cudaStreamSynchronize(stream));

      for (unsigned long j = 0; j < data_size; j++) {
        auto diff = abs(nccl_result[j] - my_result[j]);
        if (diff >= 4e-2) {
          printf(
              "Rank %d: Verification mismatch at %lld: %f != (my) %f, gt=%f\n",
              myRank, j, nccl_result[j], my_result[j], ground_truth[j]);
          break;
        }
      }
    }
    if (myRank == 0)
      printf("Test passed: nGPUs:%d, sz (kb): %d, %d, %d\n", nRanks,
             data_size * sizeof(T) / 1024, threads, block_limit);
    // long double nccl_diffs = 0.0;
    // long double my_diffs = 0.0;
    // for (int j = 0; j < data_size; j++) {
    //   nccl_diffs += abs(nccl_result[j] - ground_truth[j]);
    //   my_diffs += abs(my_result[j] - ground_truth[j]);
    // }
    // if (myRank == 0)
    //   std::cout << "average abs diffs: nccl: " << nccl_diffs / data_size
    //             << " me: " << my_diffs / data_size << std::endl;
  }

  CUDACHECK(cudaFree(result_ori));
  CUDACHECK(cudaFree(result_fuse));
  CUDACHECK(cudaFree(self_data_copy));
  CUDACHECK(cudaFree(rank_data));
  CUDACHECK(cudaFree(buffer));
  CUDACHECK(cudaFree(states));
  CUDACHECK(cudaFreeHost(ground_truth));
  CUDACHECK(cudaFreeHost(nccl_result));
  CUDACHECK(cudaFreeHost(my_result));
  CUDACHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
  int nRanks, myRank;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
  CUDACHECK(cudaSetDevice(myRank));
  ncclUniqueId id;
  ncclComm_t comm;
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast(static_cast<void*>(&id), sizeof(id), MPI_BYTE, 0,
                     MPI_COMM_WORLD));
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  bool performance_test = true;
  cudaProfilerStart();
// Uncomment to scan through different block size configs.
// for (int threads : {256, 512, 1024}) {
//   for (int block_limit = 16; block_limit < 112; block_limit += 4) {
//     run<half>(myRank, nRanks, comm, threads, block_limit, 1024 * 1024,
//     performance_test);
//   }
// }
#ifdef USE_ROCM
  const int block_limit = 16;
#else
  const int block_limit = 36;
#endif
  // Scan through different sizes to test performance.
  // for (int sz = 512; sz <= (8 << 20); sz *= 2) {
    run<half>(myRank, nRanks, comm, 512, 36, 7168 * 80, performance_test, 7168);
  // }

  cudaProfilerStop();
  MPICHECK(MPI_Finalize());
  return EXIT_SUCCESS;
}