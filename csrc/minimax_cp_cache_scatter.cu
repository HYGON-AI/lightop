#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace at::native {
namespace {
// One wave writes one logical token directly from rank-major CP transport.
// K/V are exact E4M3FN bytes; index K remains exact BF16 bytes.
__global__ void minimax_cp_reorder_cache_scatter_kernel(
    const uint8_t* __restrict__ raw,const int64_t* __restrict__ reorder,
    const int64_t* __restrict__ slots,uint8_t* __restrict__ kc,
    uint8_t* __restrict__ vc,uint8_t* __restrict__ ikc,
    int64_t rows,int64_t raw_rows,int64_t capacity,
    int64_t ks,int64_t vs,int64_t iks) {
  const int64_t row=blockIdx.x;
  if(row>=rows)return;
  const int64_t src=reorder[row],slot=slots[row];
  if(src<0||src>=raw_rows||slot<0||slot>=capacity)return;
  const int lane=threadIdx.x;
  // The native main-KV writer reserves slot zero; the BF16 index writer
  // accepts zero as an ordinary tensor-index location.
  if(slot>0){
    const uint4 value=*reinterpret_cast<const uint4*>(raw+src*1280+lane*16);
    if(lane<32)*reinterpret_cast<uint4*>(kc+slot*ks+lane*16)=value;
    else *reinterpret_cast<uint4*>(vc+slot*vs+(lane-32)*16)=value;
  }
  if(lane<16){
    const uint4 value=*reinterpret_cast<const uint4*>(raw+src*1280+1024+lane*16);
    *reinterpret_cast<uint4*>(ikc+slot*iks+lane*16)=value;
  }
}
void check_cache(const torch::Tensor& t,int64_t width,const torch::Tensor& raw){
  TORCH_CHECK(t.is_cuda()&&t.device()==raw.device()&&t.scalar_type()==at::kByte,"cache must be same-device uint8 view");
  TORCH_CHECK(t.dim()==2&&t.size(1)==width&&t.stride(1)==1&&t.stride(0)>=width&&t.stride(0)%16==0,"unsupported cache layout");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%16==0,"cache requires 16-byte alignment");
}
}
void minimax_cp_cache_scatter(torch::Tensor raw,torch::Tensor reorder,
    torch::Tensor slots,torch::Tensor kc,torch::Tensor vc,torch::Tensor ikc){
  TORCH_CHECK(raw.is_cuda()&&raw.scalar_type()==at::kByte&&raw.dim()==2&&raw.size(1)==1280&&raw.is_contiguous(),"raw CP packet must be contiguous uint8 [rows,1280]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(raw.data_ptr())%16==0,"raw packet alignment");
  for(const auto& t:{reorder,slots})TORCH_CHECK(t.is_cuda()&&t.device()==raw.device()&&t.scalar_type()==at::kLong&&t.dim()==1&&t.is_contiguous(),"reorder/slots must be contiguous same-device int64 vectors");
  TORCH_CHECK(reorder.numel()==slots.numel(),"reorder/slot row mismatch");
  check_cache(kc,512,raw);check_cache(vc,512,raw);check_cache(ikc,256,raw);
  TORCH_CHECK(kc.size(0)==vc.size(0)&&kc.size(0)==ikc.size(0),"cache capacities differ");
  TORCH_CHECK(raw.data_ptr()!=kc.data_ptr()&&raw.data_ptr()!=vc.data_ptr()&&raw.data_ptr()!=ikc.data_ptr(),"input/cache alias unsupported");
  if(slots.numel()==0)return;
  const c10::cuda::CUDAGuard device_guard(raw.device());
  hipLaunchKernelGGL(minimax_cp_reorder_cache_scatter_kernel,dim3(slots.numel()),dim3(64),0,at::cuda::getCurrentCUDAStream(),raw.data_ptr<uint8_t>(),reorder.data_ptr<int64_t>(),slots.data_ptr<int64_t>(),kc.data_ptr<uint8_t>(),vc.data_ptr<uint8_t>(),ikc.data_ptr<uint8_t>(),slots.numel(),raw.size(0),kc.size(0),kc.stride(0),vc.stride(0),ikc.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
