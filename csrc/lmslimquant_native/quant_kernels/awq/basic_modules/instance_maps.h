#pragma once

#include <unordered_map>

#include "ck/ck.hpp"
#include "instance_weight_only.h"
#include "instance_weight_only_splitK.h"
#include "gemm_utils.h"

namespace instance {

// // // 后面要是做M，N，K模糊匹配可以使用tuple的方式
// std::unordered_map<std::string, profiler_InstanceType> profiler_int4Gemm_instance_map = {
//     {"GemmMultipleB_col_V1", profiler_gemm_xdl_int4<weightonly::GemmMultipleB_col_V1>},
//     {"GemmMultipleB_col_V2", profiler_gemm_xdl_int4<weightonly::GemmMultipleB_col_V2>},
//     {"GemmMultipleB_col_V3", profiler_gemm_xdl_int4<weightonly::GemmMultipleB_col_V3>},
// };

// std::unordered_map<std::string, profiler_InstanceType> profiler_int4Gemm_splitk_reduce_instance_map = {
//     {"GemmMultipleB_col_SplitK_reduce_V1", profiler_gemm_splitK_xdl_int4<splitK_reduce::GemmMultipleB_col_SplitK_reduce_V1>},
//     {"GemmMultipleB_col_SplitK_reduce_V2", profiler_gemm_splitK_xdl_int4<splitK_reduce::GemmMultipleB_col_SplitK_reduce_V2>},
//     {"GemmMultipleB_col_SplitK_reduce_V3", profiler_gemm_splitK_xdl_int4<splitK_reduce::GemmMultipleB_col_SplitK_reduce_V3>},
// };

// std::unordered_map<std::string, profiler_InstanceType> profiler_int4Gemm_splitk_atomicAdd_instance_map = {
//     {"GemmMultipleB_col_SplitK_AtomicAdd_V1", profiler_gemm_splitK_xdl_int4<splitK_atomicAdd::GemmMultipleB_col_SplitK_AtomicAdd_V1>},
//     {"GemmMultipleB_col_SplitK_AtomicAdd_V2", profiler_gemm_splitK_xdl_int4<splitK_atomicAdd::GemmMultipleB_col_SplitK_AtomicAdd_V2>},
//     {"GemmMultipleB_col_SplitK_AtomicAdd_V3", profiler_gemm_splitK_xdl_int4<splitK_atomicAdd::GemmMultipleB_col_SplitK_AtomicAdd_V3>},
// };
///////////////////////////////// 准备好可用的gemm函数 /////////////////////////////////
// 总结可用的gemm函数
std::unordered_map<std::string, profiler_InstanceType> int4Gemm_available_map = {
    {"GemmMultipleB_col_gemv", profiler_gemv_int4},
    {"GemmMultipleB_col_V1", profiler_gemm_xdl_int4<weightonly::GemmMultipleB_col_V1>},
    {"GemmMultipleB_col_V2", profiler_gemm_xdl_int4<weightonly::GemmMultipleB_col_V2>},
    {"GemmMultipleB_col_V3", profiler_gemm_xdl_int4<weightonly::GemmMultipleB_col_V3>},
    {"GemmMultipleB_col_SplitK_reduce_V1", profiler_gemm_splitK_xdl_int4<splitK_reduce::GemmMultipleB_col_SplitK_reduce_V1>},
    {"GemmMultipleB_col_SplitK_reduce_V2", profiler_gemm_splitK_xdl_int4<splitK_reduce::GemmMultipleB_col_SplitK_reduce_V2>},
    {"GemmMultipleB_col_SplitK_reduce_V3", profiler_gemm_splitK_xdl_int4<splitK_reduce::GemmMultipleB_col_SplitK_reduce_V3>},
    {"GemmMultipleB_col_SplitK_AtomicAdd_V1", profiler_gemm_splitK_xdl_int4<splitK_atomicAdd::GemmMultipleB_col_SplitK_AtomicAdd_V1>},
    {"GemmMultipleB_col_SplitK_AtomicAdd_V2", profiler_gemm_splitK_xdl_int4<splitK_atomicAdd::GemmMultipleB_col_SplitK_AtomicAdd_V2>},
    {"GemmMultipleB_col_SplitK_AtomicAdd_V3", profiler_gemm_splitK_xdl_int4<splitK_atomicAdd::GemmMultipleB_col_SplitK_AtomicAdd_V3>},
};

} // namespace instance
