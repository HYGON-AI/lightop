// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "lmslimquant_native/quant_kernels/awq/kernels/format.cuh"
__device__ uint32_t read_u4(const uint32_t* address, uint32_t index)
{
    return (*address >> (index * 4u)) & 0xfu;
}

__device__ void atomic_assign_u4(uint32_t* address, uint32_t index, uint32_t value)
{
    uint32_t old = *address;
    uint32_t assumed;
    do {
        assumed      = old;
        uint32_t tmp = (assumed & ~(0xfu << (index * 4u))) | (value << (index * 4u));
        old          = atomicCAS(address, assumed, tmp);
    } while (assumed != old);
}

template<int... Ds>
__global__ void permute_u4(uint* dst, const uint* src, Array<int, sizeof...(Ds)> dims)
{
    constexpr int N = sizeof...(Ds);

    size_t count = 1;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        count *= dims[i];
    }

    constexpr int order[] = {Ds...};

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {

        int indices[N]{};

        #pragma unroll
        for (int j = N - 1, ii = i; j >= 0; --j) {
            indices[j] = ii % dims[j];
            ii /= dims[j];
        }

        auto data = read_u4(src + i / 8, i % 8);

        int index = 0;

        #pragma unroll
        for (int j = N - 1, stride = 1; j >= 0; --j) {
            index += indices[order[j]] * stride;
            stride *= dims[order[j]];
        }

        atomic_assign_u4(dst + index / 8, index % 8, data);
    }
}

__inline__ __device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source)
{
    uint4 result;

    uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
    uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut                = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
    static constexpr uint32_t TOP_MASK              = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    // Note that the entire sequence only requires 1 shift instruction. This is
    // thanks to the register packing format and the fact that we force our
    // integers to be unsigned, and account for this in the fp16 subtractions. In
    // addition, I exploit the fact that sub and fma have the same throughput in
    // order to convert elt_23 and elt_67 to fp16 without having to shift them to
    // the bottom bits before hand.

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW
    // dependency if we issue immediately before required.
    const uint32_t top_i4s = i4s >> 8;
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    // asm("lop3.b32 %0, %1, %2, %3, %4;\n"
    //     : "=r"(h[0])
    //     : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    h[0]=(i4s & BOTTOM_MASK)|I4s_TO_F16s_MAGIC_NUM;
    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    // asm("lop3.b32 %0, %1, %2, %3, %4;\n"
    //     : "=r"(h[1])
    //     : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    h[1]=(i4s & TOP_MASK)|I4s_TO_F16s_MAGIC_NUM;
    // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    // asm("lop3.b32 %0, %1, %2, %3, %4;\n"
    //     : "=r"(h[2])
    //     : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    h[2]=(top_i4s & BOTTOM_MASK)|I4s_TO_F16s_MAGIC_NUM;
    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    // asm("lop3.b32 %0, %1, %2, %3, %4;\n"
    //     : "=r"(h[3])
    //     : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    h[3]=(top_i4s & TOP_MASK)|I4s_TO_F16s_MAGIC_NUM;

    // I use inline PTX below because I am not sure if the compiler will emit
    // float2half instructions if I use the half2 ctor. In this case, I chose
    // performance reliability over code readability.

    // This is the half2 {1032, 1032} represented as an integer.
    // static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
    // Haotian: subtract {1024, 1024} instead, we do not need to map to [-8, 7]
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
    // This is the half2 {1 / 16, 1 / 16} represented as an integer.
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    // This is the half2 {-72, -72} represented as an integer.
    // static constexpr uint32_t NEG_72 = 0xd480d480;
    // Haotian: Let's use {-64, -64}.
    static constexpr uint32_t NEG_64 = 0xd400d400;

    // Finally, we construct the output numbers.
    // Convert elt_01
    //asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    h[0]=h[0]-FP16_TOP_MAGIC_NUM;
    // Convert elt_23
    //asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
    h[1]=h[1]*ONE_SIXTEENTH+NEG_64;
    // Convert elt_45
    //asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    h[2]=h[2]-FP16_TOP_MAGIC_NUM;
    // Convert elt_67
    //asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
    h[3]=h[3]*ONE_SIXTEENTH+NEG_64;
    return result;
}

__inline__ __device__ uint4 dequantize_s4_to_fp16x2_v2(uint32_t const& source)
{
    uint4 result;

    uint32_t*       h   = reinterpret_cast<uint32_t*>(&result);
    uint32_t const& i4s = reinterpret_cast<uint32_t const&>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut      = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOT_MASK    = 0x000f000f;
    static constexpr uint32_t TOP_MASK    = 0x00f000f0;
    static constexpr uint32_t MAGIC_NUM_0 = 0x64006400;        // `1024`
    static constexpr uint32_t MAGIC_NUM_1 = 0x54005400;        // `64`
    static constexpr uint32_t MAGIC_NUM_2 = MAGIC_NUM_1 >> 4;  // `64` >> 4

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW
    // dependency if we issue immediately before required.
    const uint32_t top_i4s = i4s >> 8;

 //  64 only, trade 4 hfma2 with 2 shifts
    h[0] =(i4s & BOT_MASK) |MAGIC_NUM_2;
    h[1] =(i4s & TOP_MASK) |MAGIC_NUM_1;
    h[2] =(top_i4s & BOT_MASK) |MAGIC_NUM_2;
    h[3] =(top_i4s & TOP_MASK) |MAGIC_NUM_1;
    h[0] <<= 4;
    h[2] <<= 4;
    // we don't need to subtract the magic nums because zeros will go through the same dequant function
    // and carry the same magic constant, the magic num will be canceled out after subtracting zeros


    return result;
}
__global__ void dequantize_s4_offset_64(uint4* dst, const uint32_t* src, size_t count)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {
        dst[i] = dequantize_s4_to_fp16x2_v2(src[i]);
    }
}

__global__ void merge_Q(half2* Q, const half* scales, const half* zeros, int count)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {
        Q[i] = __halves2half2(zeros[i], scales[i]);
    }
}

void reformat_s4_k_m8_tarnsw4(uint32_t* dst, const uint32_t* src, int m, int k, hipStream_t st)
{
    Array<int, 10> shape{1, k / 8, 2, 2, 2, 1, m / 8, 2, 2, 2};
    // 0123456-->4,6,7,5,0,3,1,2
    //permute_u4<4, 6, 7, 5, 0, 3, 1, 2><<<512, 512, 0, st>>>(dst, src, shape);
    permute_u4<5, 6, 8, 9, 7, 0, 1, 4, 2, 3><<<1024, 256, 0, st>>>(dst, src, shape);
}

__global__ void dequant_kernel(int num_kernels,half* weight ,const half2* zeros_and_scales,int k,int n,int group_size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= num_kernels) return;
    int j=id%n;
    int i=id/n;
    half x=zeros_and_scales[i/group_size*n+j].data[0];
    half y= zeros_and_scales[i/group_size*n+j].data[1];
    float tmp=(weight[id]-x)*y;
    weight[id]=__float2half(tmp);
}

__global__ void dequant_kernel_colmajor(int64_t num_kernels,half* weight ,const half2* zeros_and_scales,int64_t k,int64_t n,int64_t group_size)
{
    int64_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= num_kernels) return;

    int64_t j=id/group_size;
    half x=zeros_and_scales[j].data[0];
    half y= zeros_and_scales[j].data[1];
    float tmp=(weight[id]-x)*y;
    weight[id]=__float2half(tmp);
}

void dequant_w4_gemm(hipStream_t stream, half* output,const uint32_t* weight,const half2* zeros_and_scales,int k, int n, int group_size)
{
    dequantize_s4_offset_64<<<256, 256, 0, stream>>>((uint4*)output, weight, k  * n / 8);
    int num_kernels=k*n;
    dequant_kernel<<<(num_kernels+BLOCKSIZE-1)/BLOCKSIZE,BLOCKSIZE,0,stream>>>(num_kernels,output,zeros_and_scales,k,n,group_size);
}


void dequant_w4_gemm_colmajor(hipStream_t stream, half* output,const uint32_t* weight,const half2* zeros_and_scales,int64_t k, int64_t n, int64_t group_size)
{
    dequantize_s4_offset_64<<<256, 256, 0, stream>>>((uint4*)output, weight, k  * n / 8);
    int64_t num_kernels=k*n;

    dequant_kernel_colmajor<<<(num_kernels+BLOCKSIZE-1)/BLOCKSIZE,BLOCKSIZE,0,stream>>>(num_kernels,output,zeros_and_scales,k,n,group_size);
}


__global__ void FusedSiluActivation_kernel(int num_kernels,half* output ,const uint32_t* src,int m,int n)
{

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= num_kernels) return;

    auto data = ((half2*)src)[id];
    float x= __half2float(data.data[0]);
    float y= __half2float(data.data[1]);

    float silu=x / (1.f + __expf(-x))*y;

    output[id]=__float2half(silu);

}

__global__ void assign_kernel(int num_kernels,half* output ,const half* src,int m,int n)
{

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= num_kernels) return;
    output[id]=src[id];
}

void addFusedSiluActivation(hipStream_t stream,half* output, const half* src,int m,int n,int type)
{
    int num_kernels=m*n;
    switch (type) {
        case 0:
            assign_kernel<<<(num_kernels+BLOCKSIZE-1)/BLOCKSIZE,BLOCKSIZE,0,stream>>>(num_kernels,output,src,m,n);
            break;
        case 1:
            FusedSiluActivation_kernel<<<(num_kernels+BLOCKSIZE-1)/BLOCKSIZE,BLOCKSIZE,0,stream>>>(int(num_kernels/2),output,(const uint32_t*)src,m,n);
            break;
        default:
            return;
    }
}

template <typename T>
__global__ void input_padding_kernel(int num_kernels,T* output,const T* input,int m,int k,int group_size,int count)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= num_kernels) return;

    size_t j=id%(k+count*group_size);
    size_t i=id/(k+count*group_size);

    size_t id_o=i*(k+count*group_size)+j;
    size_t id_i=i*(k)+j;

    if(j<k)
    {
        output[id_o]=input[id_i];
    }
    else
    {
        output[id_o]=0.f;
    }
}


template <typename T>
void input_padding(hipStream_t stream, T* output,const T* input,int m,int k,int group_size,int pad_groupcount)
{

    int num_kernels=m*(k+pad_groupcount*group_size);

    input_padding_kernel<<<(num_kernels+BLOCKSIZE-1)/BLOCKSIZE,BLOCKSIZE,0,stream>>>(num_kernels, output,input,m,k,group_size,pad_groupcount);
}


#define INSTANTIATEINPUTPADING(T)  \
template void input_padding(hipStream_t stream, T* output,const T* input,int m,int k,int group_size,int pad_groupcount);

INSTANTIATEINPUTPADING(__half)


template <typename T>
void PrintData(hipStream_t stream, const T* input,int size)
{
    int input_size=size;
    T* h_data;
    h_data=new T[input_size];
    hipMemcpy(h_data,input, input_size * sizeof(T), hipMemcpyDeviceToHost);

    if constexpr (std::is_same<T, half>::value)
    {
        for(int i=0;i<input_size;i++)
        {
            printf("%f ",__half2float(h_data[i]));
        }
    }
    else if constexpr(std::is_same<T, half2>::value)
    {
        for(int i=0;i<input_size;i++)
        {
            printf("x:%f  y:%f ",__half2float(h_data[i].data[0]),__half2float(h_data[i].data[1]));
        }
    }
    else if constexpr(std::is_same<T, uint32_t>::value)
    {
        for(int i=0;i<input_size;i++)
        {
            printf(" %u ",h_data[i]);
        }
    }
    printf("\n");
    delete[] h_data;
}

// PrintData is a debugging template, not an operator entry point. Do not
// explicitly instantiate it in the production extension: FP16 host printing
// introduces compiler-runtime dependencies even when no caller uses it.



void dequant_gemm_rocblas_gpu(hipStream_t stream,
                                    half* output,
                                    const half* input,
                                    const half2* zeros_and_scales,
                                    const uint32_t* weight,
                                    const int M,
                                    const int N,
                                    const int K,
                                    half* dequant_weight,
                                    int group_size)
{
    //参数说明
    //dequant_tempworkspace 主要用于反量化时候的临时空间存储

    dequant_w4_gemm_colmajor(stream,reinterpret_cast<half*>(dequant_weight),(const uint32_t*)weight,(const half2*)zeros_and_scales,K,N,group_size);

    // 矩阵乘法，将结果存储在矩阵C中
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 创建hipBLAS句柄
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    rocblasGemmEx(handle,
                    rocblas_operation_transpose,
                    //rocblas_operation_none,
                    rocblas_operation_none,
                    N,
                    M,
                    K,
                    alpha,
                    dequant_weight,
                    K,       //lda
                    //N,
                    input,
                    K,       //ldb
                    beta,
                    output,
                    N        //ldc
                    );
    hipDeviceSynchronize();
    rocblas_destroy_handle(handle); // 不再需要时销毁句柄
}

void launch_dequant_gemm_rocblas_gpu(hipStream_t stream,
                                void* output,
                                const void* input,
                                const void* zeros_and_scales,
                                const void* weight,
                                const int M,
                                const int N,
                                const int K,
                                void* dequant_weight,
                                int group_size)
    {
        return dequant_gemm_rocblas_gpu(stream,
                                            (half*) output,
                                            (const half*) input,
                                            (const half2*) zeros_and_scales,
                                            (const uint32_t*) weight,
                                            M,
                                            N,
                                            K,
                                            (half*) dequant_weight,
                                            group_size);
    }

void convert_s4_k_m8_(uint32_t*       A_dst,
                     half2*          Q_dst,
                     half*           workspace,
                     const uint32_t* A_src,
                     const half*     scales,
                     const uint32_t* qzeros,
                     int              m,
                     int              k,
                     int              group_size,
                     hipStream_t      stream
                     )
{
    dequantize_s4_offset_64<<<256, 256, 0, stream>>>((uint4*)workspace, qzeros, k / group_size * m / 8);
    merge_Q<<<256, 256, 0, stream>>>(Q_dst, scales, workspace, k / group_size * m);
    reformat_s4_k_m8_tarnsw4(A_dst, A_src, m, k, stream);
}


//转置矩阵
template <typename T>
__global__ void permute_trans_kernel(int num_kernels,T* output ,const T* src,int64_t dim0,int64_t n)
{

    int64_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= num_kernels) return;
    int64_t j = id%dim0;
    int64_t i = id/dim0;
    output[i*dim0+j]=src[j*n+i];

}


template <typename T>
void permute_trans(T*   _sz,
                    T*   sz,
                    int64_t dim0,
                    int64_t n,
                    hipStream_t  stream
                    )
    {
        int64_t num_kernels=dim0*n;
        permute_trans_kernel<<<(num_kernels+BLOCKSIZE-1)/BLOCKSIZE,BLOCKSIZE,0,stream>>>(num_kernels,_sz,sz,dim0,n);
    }

#define INSTANTIATEPERMUTE_TRANS(T)  \
template void permute_trans(T* _sz,T* sz,int64_t dim0,int64_t n,hipStream_t stream);

INSTANTIATEPERMUTE_TRANS(__half)
INSTANTIATEPERMUTE_TRANS(float)
INSTANTIATEPERMUTE_TRANS(uint32_t)

void sgemm_rocblas_fp16_gpu(hipStream_t stream,
                            rocblas_handle handle,
                                    half* output,
                                    const half* input,
                                    const half* weight,
                                    const int M,
                                    const int N,
                                    const int K,
                                    bool layout_tn)
{
    //参数说明
    // 矩阵乘法，将结果存储在矩阵C中
    const float alpha = 1.0f;
    const float beta = 0.0f;

    //需要注意处理非32的倍数时候的操作，需要进行pad
    //  tn
    if(layout_tn){
        rocblasGemmEx(handle,
                        rocblas_operation_transpose,
                        rocblas_operation_none,
                        N,
                        M,
                        K,
                        alpha,
                        weight,
                        K,       //lda
                        input,
                        K,       //ldb
                        beta,
                        output,
                        N        //ldc
                        );
    }
    //nn
    else {
        rocblasGemmEx(handle,
                        rocblas_operation_none,
                        rocblas_operation_none,
                        N,
                        M,
                        K,
                        alpha,
                        weight,
                        N,       //lda
                        input,
                        K,       //ldb
                        beta,
                        output,
                        N        //ldc
                        );
        }
    }
