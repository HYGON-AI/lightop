#pragma once
#include <unordered_map>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device_wrapper/device_gemm_multiple_b_col_int4_splitK_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device_wrapper/device_gemm_multiple_b_col_int4_splitK_atomicAdd_xdl_cshuffle.hpp"
#include "gemm_utils.h"


namespace instance {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ALayout = Row;
using BLayout = Col;
using ELayout = Row;
using SpkLayout = Row;

using AElementOp   = PassThrough;
using BElementOp   = weight_only_tools::Dequation;
using CDEElementOp = PassThrough;

using SpkDataType       = F16;
using EDataType         = F16;
using ADataType         = ck::half_t;
using BDataType0        = uint32_t;
using BDataType1        = uint32_t;
using AccDataType       = float;
using CShuffleDataType  = ck::half_t;
using COutDataType      = ck::half_t;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace splitK_atomicAdd {

using GemmMultipleB_col_SplitK_AtomicAdd_V1 = ck::tensor_operation::device_wrapper::DeviceGemmMultipleB_col_SplitK_AtomicAdd_Xdl_CShuffle<
    ck::Tuple<ALayout>, ck::Tuple<BLayout, BLayout>, SpkLayout, ELayout,
    ck::Tuple<ADataType>, ck::Tuple<BDataType0, BDataType1>, AccDataType, COutDataType, SpkDataType, EDataType,
    AElementOp, BElementOp, CDEElementOp,
//    | Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|     CShuffle|    CShuffle|
//    | Size | Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|  MXdlPerWave| NXdlPerWave|
//    |      |      |      |      |    |    |     |     | Wave| Wave|   PerShuffle|  PerShuffle|
        512,    64,   64,    64,   16,   16,   16,   16,    1,    2,        1,           2>;

using GemmMultipleB_col_SplitK_AtomicAdd_V2 = ck::tensor_operation::device_wrapper::DeviceGemmMultipleB_col_SplitK_AtomicAdd_Xdl_CShuffle<
    ck::Tuple<ALayout>, ck::Tuple<BLayout, BLayout>, SpkLayout, ELayout,
    ck::Tuple<ADataType>, ck::Tuple<BDataType0, BDataType1>, AccDataType, COutDataType, SpkDataType, EDataType,
    AElementOp, BElementOp, CDEElementOp,
//    | Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|     CShuffle|    CShuffle|
//    | Size | Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|  MXdlPerWave| NXdlPerWave|
//    |      |      |      |      |    |    |     |     | Wave| Wave|   PerShuffle|  PerShuffle|
        512,    64,   128,    64,   16,  16,  16,   16,    1,    4,         1,           2>;

using GemmMultipleB_col_SplitK_AtomicAdd_V3 = ck::tensor_operation::device_wrapper::DeviceGemmMultipleB_col_SplitK_AtomicAdd_Xdl_CShuffle<
    ck::Tuple<ALayout>, ck::Tuple<BLayout, BLayout>, SpkLayout, ELayout,
    ck::Tuple<ADataType>, ck::Tuple<BDataType0, BDataType1>, AccDataType, COutDataType, SpkDataType, EDataType,
    AElementOp, BElementOp, CDEElementOp,
//    | Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|     CShuffle|    CShuffle|
//    | Size | Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|  MXdlPerWave| NXdlPerWave|
//    |      |      |      |      |    |    |     |     | Wave| Wave|   PerShuffle|  PerShuffle|
        512,    16,   128,    64,   16,  16,  16,   16,    1,    1,        1,           1>;
}   // namespace splitK_atomicAdd

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace splitK_reduce {

using GemmMultipleB_col_SplitK_reduce_V1 = ck::tensor_operation::device_wrapper::DeviceGemmMultipleB_col_SplitK_Xdl_CShuffle<
    ck::Tuple<ALayout>, ck::Tuple<BLayout, BLayout>, SpkLayout, ELayout,
    ck::Tuple<ADataType>, ck::Tuple<BDataType0, BDataType1>, AccDataType, COutDataType, SpkDataType, EDataType,
    AElementOp, BElementOp, CDEElementOp,
//    | Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|     CShuffle|    CShuffle|
//    | Size | Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|  MXdlPerWave| NXdlPerWave|
//    |      |      |      |      |    |    |     |     | Wave| Wave|   PerShuffle|  PerShuffle|
        512,    16,   128,    64,   16,  16,  16,   16,    1,    1,        1,           1>;

using GemmMultipleB_col_SplitK_reduce_V2 = ck::tensor_operation::device_wrapper::DeviceGemmMultipleB_col_SplitK_Xdl_CShuffle<
    ck::Tuple<ALayout>, ck::Tuple<BLayout, BLayout>, SpkLayout, ELayout,
    ck::Tuple<ADataType>, ck::Tuple<BDataType0, BDataType1>, AccDataType, COutDataType, SpkDataType, EDataType,
    AElementOp, BElementOp, CDEElementOp,
//    | Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|     CShuffle|    CShuffle|
//    | Size | Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|  MXdlPerWave| NXdlPerWave|
//    |      |      |      |      |    |    |     |     | Wave| Wave|   PerShuffle|  PerShuffle|
        512,    64,   128,    64,   16,  16,  16,   16,    1,    4,         1,           2>;

using GemmMultipleB_col_SplitK_reduce_V3 = ck::tensor_operation::device_wrapper::DeviceGemmMultipleB_col_SplitK_Xdl_CShuffle<
    ck::Tuple<ALayout>, ck::Tuple<BLayout, BLayout>, SpkLayout, ELayout,
    ck::Tuple<ADataType>, ck::Tuple<BDataType0, BDataType1>, AccDataType, COutDataType, SpkDataType, EDataType,
    AElementOp, BElementOp, CDEElementOp,
//    | Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|     CShuffle|    CShuffle|
//    | Size | Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|  MXdlPerWave| NXdlPerWave|
//    |      |      |      |      |    |    |     |     | Wave| Wave|   PerShuffle|  PerShuffle|
        512,    64,   64,    64,   16,   16,   16,   16,    1,    2,        1,           2>;

}   // namespace splitK_reduce


template <typename DeviceOpStu>
__host__ void profiler_gemm_splitK_xdl_int4(const void* A,
                                            const void* B0,
                                            const void* B1,
                                            void* C,
                                            int M,
                                            int N,
                                            int K,
                                            int StrideA,
                                            int StrideB,
                                            int StrideC,
                                            int SplitK_Batch,
                                            int GroupSize,
                                            void* splitK_workspace, // 用于SplitK添加pad的显存空间
                                            hipStream_t stream_id) {
    //// 1.设置模板，运行kernel
    auto device_op = DeviceOpStu{};

    auto invoker = device_op.MakeInvoker();

    auto argument = device_op.MakeArgument(std::array<const void*, 1>{A},
                                           std::array<const void*, 2>{B0, B1},
                                           splitK_workspace,
                                           C,
                                           M,
                                           N,
                                           K,
                                           std::array<ck::index_t, 1>{StrideA},
                                           std::array<ck::index_t, 2>{StrideB, StrideB},
                                           StrideC,
                                           StrideC,
                                           AElementOp{},
                                           BElementOp{},
                                           CDEElementOp{},
                                           SplitK_Batch,
                                           GroupSize);
    if(!device_op.IsSupportedArgument(argument)) {
        throw std::runtime_error("wrong! device_gemm with the specified compilation parameters does "
                                 "not support this GEMM problem");
    }
    invoker.Run(argument, StreamConfig{stream_id, false});
}

}   // namespace instance
