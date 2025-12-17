#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_hcc.h>
#include <vector>

#include "aiter_hip_common.h"

using half_t  = __half;
using bhalf_t = __hip_bfloat16;

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

template <typename T>
struct __attribute__((packed)) GemmInfo {
    uint32_t m;
    uint32_t n;
    uint32_t batch;
    uint32_t k;
    T* d;
    T* c;
    int8_t* a;
    int8_t* b;
    uint32_t strideD1;
    uint32_t strideD2;
    uint32_t strideC1;
    uint32_t strideC2;
    uint32_t strideA1;
    uint32_t strideA2;
    uint32_t strideB1;
    uint32_t strideB2;
    int8_t  alpha[16];
    int8_t   beta[16];
    float* scaleA=nullptr;
    float* scaleB=nullptr;
};

namespace at {
namespace native {

void _m_grouped_w8a8_gemm_nt_contig_no_pad_asm_impl(
    torch::Tensor input, 
    torch::Tensor b_qweight, 
    torch::Tensor output,
    torch::Tensor a_scale, 
    torch::Tensor b_scale, 
    torch::Tensor m_indices,
    torch::Tensor token_per_expert, 
    uint32_t experts_num) {
    
    std::vector<GemmInfo<bhalf_t>> d_prob_vec;

    const int token_num = input.size(0);
    printf("token_num : %d\n", token_num);
    const int size_n = b_qweight.size(1) * 16;
    printf("size_n : %d\n", size_n);
    const int size_k = input.size(1);
    printf("size_k : %d\n", size_k);
    int32_t lda = size_k;
    int32_t ldb = size_k;
    int32_t ldc = size_n;
    int32_t ldd = size_n;

    float d_alpha = 1.0;
    float d_beta = 0.0;

    int8_t* inp = input.data_ptr<int8_t>();
    // std::cout << std::hex << reinterpret_cast<void*>(inp) << std::endl;
    int8_t* weight = b_qweight.data_ptr<int8_t>();
    //std::cout << std::hex << reinterpret_cast<void*>(weight) << std::endl;

    int32_t* m_indices_ptr = m_indices.data_ptr<int32_t>();
    bhalf_t* output_ptr = reinterpret_cast<bhalf_t *>(output.data_ptr());
    //std::cout << std::hex << output.data_ptr() << std::endl;
    float* scaleA = a_scale.data_ptr<float>();
    float* scaleB = b_scale.data_ptr<float>();

    size_t 	size_nk = size_k * size_n;
    size_t stride_b = size_n * size_k;

    int32_t* token_per_expert_ptr = token_per_expert.data_ptr<int32_t>();
    int32_t start_token = 0;
    for (int i = 0, j = 0; i < token_num; ++j) {
        GemmInfo<bhalf_t> d_prob;
        //printf("start_token : %d\n", start_token);
        int32_t expertId = m_indices_ptr[i];
        //printf("expertId : %d\n", expertId);
        int32_t actual_size_m = token_per_expert_ptr[expertId];
        //printf("size_m : %d\n", actual_size_m);

        size_t stride_a = actual_size_m * size_k;
        size_t stride_c = actual_size_m * size_n;
        size_t stride_d = actual_size_m * size_n;
        d_prob.m = size_n;
        d_prob.n = actual_size_m;
        d_prob.batch = 1;
        d_prob.k = size_k;
        d_prob.strideD1 = ldd;
        d_prob.strideD2 = stride_d;
        d_prob.strideC1 = ldc;
        d_prob.strideC2 = stride_c;
        d_prob.strideA1 = ldb;
        d_prob.strideA2 = stride_b;
        d_prob.strideB1 = lda;
        d_prob.strideB2 = stride_a;

        memcpy(d_prob.alpha, &d_alpha, sizeof(float));
        memcpy(d_prob.beta,  &d_beta, sizeof(float));

        d_prob.a = weight + expertId * stride_b;
        //std::cout << std::hex << reinterpret_cast<void*>(d_prob.a) << std::endl;
        d_prob.b = inp + start_token * lda;
        // std::cout << std::hex << reinterpret_cast<void*>(d_prob.b) << std::endl;
        d_prob.c = output_ptr + start_token * ldc;
        d_prob.d = output_ptr + start_token * ldd;
        d_prob.scaleA = scaleB + expertId * ldd; 
        d_prob.scaleB = scaleA + start_token;
        
        start_token += actual_size_m;
        i += actual_size_m;
        d_prob_vec.push_back(d_prob);
    }
    //printf("d_pro_vec size : %d\n", d_prob_vec.size());
    size_t localWorkSize[3] = {768, 1, 1};
    size_t globalWorkSize[3] = {1, 1, 1};

    size_t wgTile = 0;
    for (int i = 0; i < d_prob_vec.size(); ++i) {
        // printf("MNK %d,%d,%d,%d,GM=%d,", d_prob_vec[i].m, d_prob_vec[i].n, 1, d_prob_vec[i].k, i);
        // printf(" wgRight%d\n",wgTile);
        size_t size_m = d_prob_vec[i].m;
        size_t size_n = d_prob_vec[i].n;
        int wg_M = DIVIDE(size_m, 256);
        int wg_N = DIVIDE(size_n, 256);
        wgTile += wg_M * wg_N;
    }
    

	struct __attribute__((packed)){
        uint32_t gemm_count;              //4
        void const* DeviceUserArguments;  //8
        void const* argsPtr;              //8
        uint32_t wgm;                     //4
        unsigned int gsu;                 //4
        int32_t* m_indics;               //8
        int32_t* debug_d;                                
    }hipFunctionArgs;

    uint8_t* d_argsPtr;
    size_t arg_size =  d_prob_vec.size() * sizeof(GemmInfo<bhalf_t>);
    hipMalloc((void**)&d_argsPtr, arg_size);
    hipMemcpy(d_argsPtr, d_prob_vec.data(), arg_size, hipMemcpyHostToDevice);

    hipFunctionArgs.gemm_count = d_prob_vec.size();
    hipFunctionArgs.DeviceUserArguments = d_argsPtr;
    hipFunctionArgs.argsPtr = nullptr;
    hipFunctionArgs.wgm = 1;
    hipFunctionArgs.gsu = 1;
    //hipFunctionArgs.m_indics = m_indices.data_ptr<int32_t>();

    size_t size = sizeof(hipFunctionArgs);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    char func_name_fp[1024], fp16_co_file_tn[1024], bf16_co_file_tn[1024], func_name_bf[1024];
    memset(func_name_fp, 0x0, sizeof(func_name_fp));
    memset(fp16_co_file_tn, 0x0, sizeof(fp16_co_file_tn));
    memset(func_name_bf, 0x0, sizeof(func_name_bf));
    memset(bf16_co_file_tn, 0x0, sizeof(bf16_co_file_tn));
    
    sprintf(func_name_bf, "DeepGemm_W8A8_I8_PERCHANNEL_ASM_TN_MT%dX%dX%d_BF16", 256, 256, 128);
    sprintf(bf16_co_file_tn, "deepgemm_channel/w8a8/DeepGemm_W8A8_I8_MARLIN_NOPAD_PERCHANNEL_ASM_TN_MT%dX%dX%d_BF16.co", 256, 256, 128);	static AiterAsmKernel deepgemm_w8a8_bf16_tn(func_name_bf, bf16_co_file_tn);

    AiterAsmKernel *impl_ptr = nullptr;
    impl_ptr = &deepgemm_w8a8_bf16_tn;
    impl_ptr->launch_kernel_ext({
        &hipFunctionArgs,
        &size,
        wgTile, 1, 1,
        localWorkSize[0],
        localWorkSize[1],
        localWorkSize[2],
        stream
    });
    return;
}




template <int BLOCKM, int BLOCKN, int BLOCKK, typename OutputType, int BLOCKDIM = 768>
void _m_grouped_w8a8_gemm_nt_contig_asm_impl(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,                   // (M)
    torch::Tensor b_scale,                   // (E, N)
    torch::Tensor m_indices,
    uint32_t experts_num) {
    
    //GemmInfo<OutputType> d_prob;
    GemmInfo<OutputType> *d_prob;
    d_prob = (GemmInfo<OutputType> *)malloc(sizeof(GemmInfo<OutputType>));

    const int size_m = input.size(0);        //(M, K)
    const int size_n = b_qweight.size(1) * 16;    //(E, N // 16, K * 16)
    const int size_k = input.size(1);

    uint32_t lda = size_k;
    uint32_t ldb = size_k;
    uint32_t ldc = size_n;
    uint32_t ldd = size_n;

    float d_alpha = 1.0;
    float d_beta = 0.0;

    size_t stride_a = size_m * size_k;
    size_t stride_b = size_n * size_k;
    size_t stride_c = size_m * size_n;
    size_t stride_d = size_m * size_n;
    {
    d_prob->m = size_n;
    d_prob->n = size_m;
    d_prob->batch = 1;
    d_prob->k = size_k;
    d_prob->strideD1 = ldd;
    d_prob->strideD2 = stride_d;
    d_prob->strideC1 = ldc;
    d_prob->strideC2 = stride_c;
    d_prob->strideA1 = ldb;
    d_prob->strideA2 = stride_b;
    d_prob->strideB1 = lda;
    d_prob->strideB2 = stride_a;
    memcpy(d_prob->alpha, &d_alpha, sizeof(float));
    memcpy(d_prob->beta,  &d_beta, sizeof(float));
    d_prob->a = b_qweight.data_ptr<int8_t>();
    d_prob->b = input.data_ptr<int8_t>();
    d_prob->d = static_cast<OutputType*>(output.data_ptr());
    d_prob->scaleA = b_scale.data_ptr<float>();
    d_prob->scaleB = a_scale.data_ptr<float>();
    }
    
    size_t localWorkSize[3] = {BLOCKDIM, 1, 1};
    size_t globalWorkSize[3] = {1, 1, 1};

    size_t wgTile = 0;
    int wg_M = DIVIDE(size_m, BLOCKM);
    int wg_N = DIVIDE(size_n, BLOCKN);
    wgTile += wg_M * wg_N;

	struct __attribute__((packed)){
        uint32_t gemm_count;              //4
        void const* DeviceUserArguments;  //8
        void const* argsPtr;              //8
        uint32_t wgm;                     //4
        unsigned int gsu;                 //4
        int32_t* m_indics;               //8
        int32_t* debug_d;                                
    }hipFunctionArgs;

    // uint8_t* d_argsPtr;
    // size_t arg_size = sizeof(d_prob);
    // hipMalloc((void**)&d_argsPtr, arg_size);
    hipHostRegister(d_prob, sizeof(GemmInfo<OutputType>), hipHostRegisterMapped);
    uint8_t*  d_argsPtr;
    hipHostGetDevicePointer((void**)&d_argsPtr, d_prob, 0);
    // hipMemcpyAsync(d_argsPtr, &d_prob, arg_size, hipMemcpyHostToDevice, 0);

    hipFunctionArgs.gemm_count = 1;
    hipFunctionArgs.DeviceUserArguments = d_argsPtr;
    hipFunctionArgs.argsPtr = nullptr;
    hipFunctionArgs.wgm = 1;
    hipFunctionArgs.gsu = 1;
    hipFunctionArgs.m_indics = m_indices.data_ptr<int32_t>();

    size_t size = sizeof(hipFunctionArgs);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // hipMemcpyAsync(d_argsPtr, &d_prob, arg_size, hipMemcpyHostToDevice, 0);
    char func_name_fp[1024], fp16_co_file_tn[1024], bf16_co_file_tn[1024], func_name_bf[1024];
    memset(func_name_fp, 0x0, sizeof(func_name_fp));
    memset(fp16_co_file_tn, 0x0, sizeof(fp16_co_file_tn));
    memset(func_name_bf, 0x0, sizeof(func_name_bf));
    memset(bf16_co_file_tn, 0x0, sizeof(bf16_co_file_tn));
    
    sprintf(func_name_fp, "DeepGemm_W8A8_I8_PERCHANNEL_ASM_TN_MT%dX%dX%d_FP16", BLOCKM, BLOCKN, BLOCKK);
    sprintf(func_name_bf, "DeepGemm_W8A8_I8_PERCHANNEL_ASM_TN_MT%dX%dX%d_BF16", BLOCKM, BLOCKN, BLOCKK);
    sprintf(fp16_co_file_tn, "deepgemm_channel/w8a8/DeepGemm_W8A8_I8_MARLIN_PERCHANNEL_ASM_TN_MT%dX%dX%d_FP16.co", BLOCKM, BLOCKN, BLOCKK);
    sprintf(bf16_co_file_tn, "deepgemm_channel/w8a8/DeepGemm_W8A8_I8_MARLIN_PERCHANNEL_ASM_TN_MT%dX%dX%d_BF16.co", BLOCKM, BLOCKN, BLOCKK);
    // string fp16_co_file_tn = "deepgemm_channel/w8a8/DeepGemm_W8A8_I8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_FP16.co";
    // string bf16_co_file_tn = "deepgemm_channel/w8a8/DeepGemm_W8A8_I8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16.co";

	static AiterAsmKernel deepgemm_w8a8_fp16_tn(func_name_fp, fp16_co_file_tn);
	static AiterAsmKernel deepgemm_w8a8_bf16_tn(func_name_bf, bf16_co_file_tn);

    AiterAsmKernel *impl_ptr = nullptr;
    if (output.scalar_type() == at::ScalarType::BFloat16)
        impl_ptr = &deepgemm_w8a8_bf16_tn;
    else 
        impl_ptr = &deepgemm_w8a8_fp16_tn;
    impl_ptr->launch_kernel_ext({
        &hipFunctionArgs,
        &size,
        wgTile, 1, 1,
        localWorkSize[0],
        localWorkSize[1],
        localWorkSize[2],
        stream
    });
    return;
}


torch::Tensor m_grouped_w8a8_gemm_nt_contig_asm(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor m_indices,
  int mode) {
  
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    uint32_t experts_num = b_qweight.sizes()[0];
    if (mode == 1000) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(), "_m_grouped_w8a8_gemm_nt_contig_asm_impl",[&] {
            _m_grouped_w8a8_gemm_nt_contig_asm_impl<256, 256, 128, scalar_t>(
            input, b_qweight, output, a_scale, b_scale, m_indices, experts_num);
        }
        );
    } else if (mode == 100) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(), "_m_grouped_w8a8_gemm_nt_contig_asm_impl",[&] {
            _m_grouped_w8a8_gemm_nt_contig_asm_impl<256, 128, 128, scalar_t, 512>(
            input, b_qweight, output, a_scale, b_scale, m_indices, experts_num);
        }
        );
    } else {
        TORCH_CHECK(false, mode, " mode error.");
    }
  return output;
}


torch::Tensor m_grouped_w8a8_gemm_nt_nopad_contig_asm(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor m_indices,
  torch::Tensor token_per_expert) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    uint32_t experts_num = b_qweight.sizes()[0];
    _m_grouped_w8a8_gemm_nt_contig_no_pad_asm_impl(
        input, b_qweight, output, a_scale, b_scale, m_indices, token_per_expert, experts_num
    );
    return output;
  }
}
}