#include "base.cuh"


template <typename T>
__global__ void assign_kernel(int n,T* A,const T* B)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= n) return;

    A[id]=B[id];
}


template <typename T>
void assign_fun(hipStream_t stream, T* A,const  T* B,int size)
{
    int num_kernels=size;
    assign_kernel<<<(num_kernels+BLOCKSIZE-1)/BLOCKSIZE,BLOCKSIZE,0,stream>>>(num_kernels,A,B);
}

#define INSTANTIATEASSIGN(T)  \
template void assign_fun(hipStream_t stream, T* A,const T* B,int size);

INSTANTIATEASSIGN(__half)
INSTANTIATEASSIGN(float)
INSTANTIATEASSIGN(half2)
INSTANTIATEASSIGN(uint)

template <typename T>
void PrintScale(hipStream_t stream,const T* scales_and_zeros,int size){
    printf("start printf ****\n");
    int input_size=size;
    T* h_scales_and_zeros;
    h_scales_and_zeros=new T[input_size];

    T* d_scales_and_zeros;
    hipMalloc((void**)&d_scales_and_zeros, input_size * sizeof(T));

    assign_fun<T>(stream,d_scales_and_zeros,scales_and_zeros,input_size);
    hipStreamSynchronize(stream);

    hipMemcpy(h_scales_and_zeros,d_scales_and_zeros, input_size * sizeof(T), hipMemcpyDeviceToHost);


    if constexpr (std::is_same_v<T, half>)
    {
        for(int i=0;i<input_size;i++)
        {
            printf("%f ",__half2float(h_scales_and_zeros[i]));
        }
    }
    else if constexpr(std::is_same_v<T, half2>)
    {
        for(int i=0;i<input_size;i++)
        {
            printf("%f  %f ",__half2float(h_scales_and_zeros[i].x),__half2float(h_scales_and_zeros[i].y));
        }
    }
    else if constexpr(std::is_same_v<T, uint>)
    {

    }
    printf("\n");
    delete[] h_scales_and_zeros;
    hipFree(d_scales_and_zeros);
    return ;
}


#define INSTANTIATEPRINT(T)  \
template void PrintScale(hipStream_t stream,const T* scales_and_zeros,int size);

INSTANTIATEPRINT(__half)
INSTANTIATEPRINT(float)
INSTANTIATEPRINT(half2)
INSTANTIATEPRINT(uint)
