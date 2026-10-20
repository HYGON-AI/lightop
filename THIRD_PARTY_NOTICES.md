# Third-Party Notices

Lightop 的根许可证是 Apache License 2.0，完整文本见 [LICENSE](LICENSE)。本文件登记仓库中已确认的第三方源码、版权和许可证义务。

## 记录方法

- “本地引入提交”是当前仓库 Git 历史中首次引入该路径的提交，用于固定本仓库中的证据；它不是第三方项目的上游 Commit。
- 本次已基于工作区内的 AITER、vLLM 和 Transformer Engine Git 克隆补录上游路径、基线选择器和 40 位 Commit。Transformer Engine 的固定参考基线与实际导入版本尚需区分。表中的“上游 Commit”是第三方仓库的提交；“本地引入提交”仍只表示 Lightop 自身历史中的引入点，二者不能互换。
- 文件若为抽取或改编，只登记实际核对到的上游源文件和基线 Commit，并明确“改编/抽取”；这不表示 Lightop 文件与上游 Blob 逐字一致。没有在本地 vLLM 历史中找到匹配文件的路径继续保留为待补证，不凭项目名称推断来源。
- 已确认来源文件保留其原始版权和 SPDX/许可证声明；`bias_swiglu.cu` 曾删除原声明，本轮恢复记录见第4节。对 HYGON 的适配或修改以 Git 历史为准；未确认来源的内部组件不机械添加 HYGON Copyright 或新的 SPDX 声明。

当前记录分为“已确认”和“待补证”两种状态。待补证项已经登记了本地路径、首次引入 Commit、当前文件哈希或二进制 SHA-256；补齐官方上游 Commit 或内部授权材料后，必须把状态改为“已确认”，再进行正式审计复扫。

## 1. ROCm AITER

| 项目 | 内容 |
| --- | --- |
| 项目/仓库 | ROCm AITER；<https://github.com/ROCm/aiter> |
| 许可证 | MIT |
| 版权 | Copyright (C) 2024-2025, Advanced Micro Devices, Inc. |
| 上游 LICENSE | `LICENSE`；Blob `6311bdec0f330b61adb89cd32f719a43717504ed` |
| 上游仓库 HEAD | 本地克隆 `b1cdc19ebf3a52b101221fee5b24a3ef7753c244`；远程 `git@github.com:ROCm/aiter.git` |

逐文件核验结果（上游文件 Blob 用于复核路径和 Commit 是否对应）：

| 本地路径 | AITER 上游路径 | 基线选择器 | 上游 40 位 Commit | 上游文件 Blob | 本地当前 Blob | 关系/本地引入提交 |
| --- | --- | --- | --- | --- | --- | --- |
| `csrc/aiter_hip_common.h` | `csrc/include/aiter_hip_common.h` | `main` | `b25a045a55257f1f9ae830ac72f6b8b49a3b8159` | `5de294a60cf6c5456458deb716d0b69e6ebc3823` | `26c0fcaeb48c487adc6ec57389fc212d9976a7e9` | 高相似 AITER 集成；本地引入 `bd98e062c1aaa7515f6bcc917500561dfd3e8183` |
| `python/lightop/utils.py` | `aiter/jit/utils/chip_info.py` | `main` | `df789ccd9a2839d8a6f32ace641e95cbc995f70f` | `a45af28c40282a0792d25932aae9e5c358882fdb` | `680bbf58b4aa1b53e2b431a14baed08f8acce94a` | 改编而非逐字复制；本地引入 `e00781a470c084d0742f430a8bfcdf8a90ee8702` |

`b25a045a...` 和 `df789ccd...` 均可由本地 AITER `main` 到达。AITER 另有内容相同的重复提交 `51351fc...`，但它不是 `main` 的祖先；这里选用 `main` 可达的 `df789ccd...` 作为可复核基线。第二个文件增加了 Lightop 的 `adb_shell`、环境变量和资源路径逻辑，发布说明应把它标成 AITER 派生改编并保留本地差异。

MIT 许可文本：

```text
MIT License

Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. vLLM 来源文件

这些文件保留 vLLM 的 Apache-2.0 SPDX 和 `Copyright contributors to the vLLM project` 声明。公开来源项目为 vLLM：<https://github.com/vllm-project/vllm>。上游 LICENSE Blob 为 `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`。下表同时保留本地引入提交和已经核对的上游基线；本地引入提交不能替代上游 Commit。

关系说明：`完整/近完整快照` 表示文件主体可以在该上游 Blob 中复核；`抽取/改编` 表示仅复用函数或 kernel 结构；`参考基线` 表示本地实现明确以该上游文件为技术起点，但不是逐字副本。

### vLLM 版权声明文件

| 本地路径 | vLLM 上游路径 | 基线选择器 | 上游 40 位 Commit | 关系 | 本地引入提交 |
| --- | --- | --- | --- | --- | --- |
| `csrc/allreduce_fuse/_custom_ops.py` | `vllm/_custom_ops.py` | `main` | `3d184b95b83bd1c755811238b013b51c49da3951` | 抽取/改编；增加 Lightop/LMSLIM 注册和分页注意力入口 | `5b05e4ed06d26c4eaac830b5d4d6e86138fdc1a6` |
| `csrc/allreduce_fuse/custom_all_reduce.py` | `vllm/distributed/device_communicators/custom_all_reduce.py` | `main` | `02f0c7b220422792f5e53de2a7d51d2d3ff2df28` | 抽取/改编；增加融合 Norm/Quant 和 HYGON 多卡策略 | `5b05e4ed06d26c4eaac830b5d4d6e86138fdc1a6` |
| `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` | `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` | `main` | `59d0236193a1cb5e88bee9cd4a3692a5587c9558` | 参考基线；本地改为 ROCm/ATen 兼容实现，不是逐字 Blob | `07e7033161c35da365e32d6e7efaa44e36a3e96c` |
| `csrc/fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp_kernel.cu` | `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` | `main` | `59d0236193a1cb5e88bee9cd4a3692a5587c9558` | 参考基线；PCP 不同 Q/KV token 数是 Lightop 新增变体，vLLM 历史无同名文件 | `086435003c8a7c26391c60425b7facd4f44ad3f7` |
| `python/lightop/_lmslim_native/vllm_compat/moe_align_block_size.py` | `vllm/model_executor/layers/fused_moe/moe_align_block_size.py` | `main` | `8fcaaf6a165e661f63fc51be906bc05b0767332f` | 抽取/改编；增加 Lightop dispatch 和参数处理 | `b4b65b751fc6b46a4115289dab5b858ee4f97fd7` |
| `python/lightop/_lmslim_native/vllm_compat/w8a8_get_cache_json.py` | 未在本地 vLLM 历史找到对应路径或同名实现 | — | — | 待补证；不能仅因目录名含 `vllm_compat` 认定为 vLLM 来源 | `b4b65b751fc6b46a4115289dab5b858ee4f97fd7` |
| `test/common/fused_moe_kernel_gptq_awq.py` | `vllm/model_executor/layers/fused_moe/fused_moe.py` | `main` | `22a97e66134a26c74b9dae73d9446c4e32718269` | 抽取；`fused_moe_kernel_gptq_awq` 主体与该 Commit 高度一致 | `32e4aad617608934ea2fea9283627f0c60527b2b` |
| `test/common/quant_utils.py` | `vllm/model_executor/layers/quantization/utils/quant_utils.py` | `main` | `d6953beb91da4e9c99be4c0a1304a2d24189535c` | 抽取；`quantize_weights`/`pack_cols`/`awq_pack` 函数可逐段复核 | `32e4aad617608934ea2fea9283627f0c60527b2b` |
| `test/common/scalar_type.py` | `vllm/scalar_type.py` | `main` | `8fcaaf6a165e661f63fc51be906bc05b0767332f` | 完整文件快照；上游与本地 Blob 均为 `05760f3f829988d4640955b6d19ff2549b1d28cd` | `32e4aad617608934ea2fea9283627f0c60527b2b` |
| `test/moe_marlin/moe_w16a16/benchmark_moe_Triton.py` | `benchmarks/kernels/benchmark_moe.py` | `main` | `cede942b87b5d8baa0b95447f3e87e3c600ff5f5` | 改编快照；增加 HCU/NN-MoE 参数和设备逻辑 | `30b88ebc31c2d8fc4c57f6f6f1b9e0586752e856` |
| `test/moe_w16a16/benchmark_moe_Triton.py` | `benchmarks/kernels/benchmark_moe.py` | `main` | `cede942b87b5d8baa0b95447f3e87e3c600ff5f5` | 改编快照；增加 HCU/NN-MoE 参数和设备逻辑 | `30b88ebc31c2d8fc4c57f6f6f1b9e0586752e856` |

**仍需补证动作：**`w8a8_get_cache_json.py` 在本地 vLLM 历史中没有匹配文件，必须由组件负责人提供真实来源或内部授权记录。两个 DeepSeek 文件只能使用上述 vLLM Commit 作为技术参考基线；若发布审批要求“逐字上游文件”，还需提供独立 `vllm-deepseek_v4` 仓库的 URL、分支/Tag 和 40 位 Commit。其余表项的 Commit 已能在本地 `main` 上复核，正式审计时应把这些元数据随提交一并提交。

### vLLM 对 NVIDIA 来源的引用方式（格式依据）

本节只记录 vLLM 的引用格式，不把 vLLM 中的示例自动认定为 Lightop 文件的来源。核验基线为本地 vLLM 克隆的 `main`，固定 Commit 为 `87d9996abe7ae0173fa8c1b6ccc2411041bcac81`，根 `LICENSE` Blob 为 `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`。

| vLLM 文件 | vLLM 实际写法 | Lightop 应采用的规则 |
| --- | --- | --- |
| `csrc/libtorch_stable/dsv3_fused_a_gemm.cu` | 先写 `Adapted from` SGLang，再写 `which was adapted from` NVIDIA TensorRT-LLM 的固定 40 位 Commit `619709fc33bd5dc268f19d6a741fe7ed51c0f8f5`，并保留 NVIDIA 与 NAVER 的版权声明 | 有多级派生时逐级列出仓库、上游文件路径和 40 位 Commit；保留全部上游版权与许可证文本 |
| `csrc/libtorch_stable/moe/topk_softmax_kernels.cu` | `Adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/...`，并列出 NVIDIA 与 vLLM 的 SPDX/版权 | Tag 可以作为可读选择器，但发布登记还要记录该 Tag 解析出的 40 位 Commit |
| `csrc/libtorch_stable/moe/moeTopKFuncs.cuh` | 同时列出 NVIDIA TensorRT-LLM 和 FlashInfer 两个来源，并保留 `SPDX-FileCopyrightText` | 多来源抽取分别登记，不要合并成一个模糊的“NVIDIA 来源” |
| `vllm/third_party/pynvml.py` | `copied from https://pypi.org/project/nvidia-ml-py`，写明版本 `12.570.86`，保留 NVIDIA 的 BSD 条款 | 完整复制要写来源项目、发布版本和原始许可证；不能只写项目名称 |
| `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu` | 注释引用 TensorRT-LLM 的 `main` 分支 | `main` 只能作为检索线索；Lightop 的最终记录必须替换成不可变 40 位 Commit |

Lightop 的 NVIDIA 文件尚未找到与上述 vLLM 文件的逐字 Blob 或唯一上游路径。因此目前只能采用这些格式规则，不能把 `vllm`、`cuda-samples` 或文件主题相似性写成真实来源。

#### Lightop NVIDIA 文件头模板

当组件负责人确认真实公开来源后，文件头按实际关系填写。`<UPSTREAM_COMMIT>` 必须是 40 位十六进制 Commit；若存在中间派生仓库，应按 vLLM 的链式写法逐级列出。

```cpp
/*
 * Adapted from
 * https://github.com/<owner>/<repo>/blob/<UPSTREAM_COMMIT>/<upstream/path>
 * Copyright (c) <original years>, <original holder>. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright (c) <original years>, <original holder>
 *
 * LightOp changes: <ROCm/HCU/HYGON changes, or “none” for an unchanged copy>.
 */
```

若文件是完整复制而不是改编，把第一行改为 `Copied from`，并保留来源版本及原许可证原文。若最终确认是内部 NVIDIA 适配代码，不能虚构 GitHub URL；应在本表登记内部源码快照、权利人、许可证或授权编号、固定版本/制品号、LightOp 修改范围和发布范围，源码继续保留 NVIDIA 原始版权与 Apache-2.0 条款。

如果核验确认文件包含 HYGON 的实质修改，再依照开源审计模板追加 HYGON 的 Copyright 和 `SPDX-FileCopyrightText`；不得用 HYGON 声明替换 NVIDIA 原始声明，也不得在来源尚未确认时借模板推断权利人。

## 3. NVIDIA 版权声明文件

以下文件保留 NVIDIA 的 Apache-2.0 原始声明。`common.h` 已基于本地 Transformer Engine 克隆完成多文件逐段核对；`block_scaled_cast_transpose.cu` 按本次用户确认归入 Transformer Engine 接口适配，恢复 NVIDIA 原始版权并补充固定参考链接、SPDX 和 HYGON 修改说明。其余 NVIDIA 文件仍需逐文件来源证据或内部授权记录。

### Transformer Engine `common.h` 逐文件来源证据

本节登记 `csrc/blockwise_scaling/common.h` 中已经在本地 Transformer Engine 克隆内逐段核对到的来源。该文件不是 Transformer Engine 单一文件的逐字副本，而是多个 NVIDIA Transformer Engine 头文件的抽取、合并和 HIP/HCU 适配版本。

| 项目 | 内容 |
| --- | --- |
| 上游项目/仓库 | NVIDIA Transformer Engine；<https://github.com/NVIDIA/TransformerEngine.git> |
| 上游分支 | `main` |
| 可复核基线 Commit | `077e26c319d0fffbff75e56124b26b8b04aac0e5` |
| 基线提交主题 | `Use userbuffers for MXFP8 wgrad all-gather overlap (#1982)` |
| 上游 LICENSE | `LICENSE`；Blob `e97bf01c50dfb20764ba2ef1b5b786fe3d547dd2`；SHA-256 `bea1b4266f1499fa4bbb16d6e7577e6b5b53d1ef010835a9d460876022cc8722` |
| LightOp 初始引入 | `e4bd6962fa68c4557df2f17fcaea0599d41304e9` |
| LightOp 后续修改 | `1f0e6d421abc38beaa67b03c32af51b4a97a9191`、`8cdf607e5c562385f2fb73cab1c3c722ad0054d0` |
| 当前 LightOp Blob | `64c1eccda927781931b9dcbf41abd3a1864fbdc5` |

`077e26c...` 是 LightOp 初始导入时间（2025-08-11）之前最近的主线提交；相关 Transformer Engine 文件在该时间窗口内未发生变化。由于 LightOp 历史未记录实际导入时使用的上游远程和 Commit，发布前仍应由组件负责人确认该 Commit 是否为实际导入基线。

逐段映射如下。表中的上游 Blob 是固定基线下的文件 Blob；LightOp 当前 Blob 不同，表示这是改编/合并关系，而不是完整复制。

| LightOp 路径 | Transformer Engine 上游路径 | 上游文件 Blob | 代码范围与关系 |
| --- | --- | --- | --- |
| `csrc/blockwise_scaling/common.h` | `transformer_engine/common/common.h` | `aa47f2c3d9ec8f7d1518ef107f267e14e7353020` | `DIVUP`、类型别名、`TRANSFORMER_ENGINE_TYPE_NAME`、`TypeExtrema`、`TypeInfo`、条件分派宏；LightOp 改为 HIP 类型并置于 `at::native` |
| `csrc/blockwise_scaling/common.h` | `transformer_engine/common/include/transformer_engine/transformer_engine.h` | `dab4fcfe754e9c2440e59ce7b37140048653bdaa` | `DType` 枚举结构；LightOp 增加 `kInt8` |
| `csrc/blockwise_scaling/common.h` | `transformer_engine/common/transpose/cast_transpose.h` | `a737239260cfa8f6affefa5e61a6962f3cba2ebc` | `FP8BlockwiseRowwiseOption`、`FP8BlockwiseColumnwiseOption` 枚举和注释 |
| `csrc/blockwise_scaling/common.h` | `transformer_engine/common/util/string.h` | `0064144102d2d121f1d658f8155b771b51162b23` | `to_string_like`、`concat_strings` 实现被合并进本地头文件 |
| `csrc/blockwise_scaling/common.h` | `transformer_engine/common/utils.cuh` | `3f5bcc975dbd6322eefe30a7eae894d299098ded` | `uint16`、`uint8`、`BytesToType`、`Vec`；LightOp 增加赋值和 HIP 相关适配 |
| `csrc/blockwise_scaling/common.h` | `transformer_engine/common/recipe/recipe_common.cuh` | `11f9bc12990550a8ed112706342c4166346c5995` | `compute_scale_from_amax`、`compute_scale_from_types` 被移动到本地头文件 |

当前文件仍有以下来源未闭环，不能仅凭名称认定为 Transformer Engine 代码：

- `TORCH_ERROR`、`TORCH_CHECK_CUDA`；
- `TRANSFORMER_ENGINE_TYPE_SWITCH_8BIT`；
- `DType::kInt8`、`Vec::operator=`、`__HOST_DEVICE__`；
- `at::native` 和 `at::k*`/`torch::kInt8` 分派逻辑。

组件负责人需要为这些部分提供公开仓库、文件路径、固定 40 位 Commit 和许可证，或提供内部源码快照、权利人和授权编号。当前 LightOp 文件还缺少 `Adapted from`、`SPDX-License-Identifier: Apache-2.0`、`SPDX-FileCopyrightText` 和 HYGON 修改声明；本节来源登记不等同于源码文件头整改完成。

### Transformer Engine `block_scaled_cast_transpose.cu` 接口适配条目

本次用户确认将该文件按 Transformer Engine 接口适配登记，撤销此前的 HYGON 原创归类。以下固定路径用于复核块量化接口和分发逻辑；本地基线未找到同名包装函数或完整文件副本，因此不将其登记为逐字复制，也不将参考 Commit 认定为实际导入 Commit。

| 项目 | 内容 |
| --- | --- |
| 本地路径 | `csrc/blockwise_scaling/block_scaled_cast_transpose.cu` |
| 上游项目/仓库 | NVIDIA Transformer Engine；<https://github.com/NVIDIA/TransformerEngine.git> |
| 上游分支 | `main` |
| 固定参考基线 Commit | `077e26c319d0fffbff75e56124b26b8b04aac0e5`；可由本地 `main` 到达 |
| 许可证/版权 | Apache-2.0；Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved. |
| 上游 LICENSE | 同上方 Transformer Engine 条目的 `LICENSE` Blob 和 SHA-256 |
| 本地引入提交 | `1f0e6d421abc38beaa67b03c32af51b4a97a9191` |
| LightOp 适配范围 | 改用 ATen `Tensor`/`std::optional<Tensor>` 参数；以 `block_scaling_dim` 分发至本地 1D 实现，2D 模式报告不支持 |
| 文件头状态 | 已恢复 NVIDIA 版权并补充 `SPDX-License-Identifier: Apache-2.0`、NVIDIA `SPDX-FileCopyrightText`、Transformer Engine 固定参考链接及 HYGON 修改说明；已移除 HYGON 原创 Copyright |

| Transformer Engine 参考路径（固定于上述 Commit） | 上游文件 Blob | 对应范围 |
| --- | --- | --- |
| [transformer_engine/common/util/cast_kernels.cuh](https://github.com/NVIDIA/TransformerEngine/blob/077e26c319d0fffbff75e56124b26b8b04aac0e5/transformer_engine/common/util/cast_kernels.cuh) | `5590cee1056aa68b32c33315d2caad2c39e1cfc6` | `quantize_helper` 的 `NVTE_BLOCK_SCALING_1D`/`NVTE_BLOCK_SCALING_2D` 分发与 `epsilon`、`force_pow_2_scales` 参数；LightOp 仅保留本地 1D 入口 |
| [transformer_engine/common/transpose/quantize_transpose_vector_blockwise.cu](https://github.com/NVIDIA/TransformerEngine/blob/077e26c319d0fffbff75e56124b26b8b04aac0e5/transformer_engine/common/transpose/quantize_transpose_vector_blockwise.cu) | `6f5c0f3a6c3df65df5f305e207d16d21b6fe715e` | `quantize_transpose_vector_blockwise` 的输入、行列输出、逆缩放、`epsilon` 和 `pow2_scale` 接口；用于说明本地 1D 调用的参考关系 |

实际导入 Commit 和从上述上游接口到本地包装函数的导入/改编记录仍需组件负责人补齐；本条不扩展到被调用的 `block_scaled_1d_cast_transpose.h` 的完整来源审核。

| 本地路径 | 许可证/版权 | 本地引入提交 |
| --- | --- | --- |
| `csrc/blockwise_scaling/block_scaled_cast_transpose.cu` | Apache-2.0；NVIDIA 原始版权、Transformer Engine 接口适配与 HYGON 修改说明 | `1f0e6d421abc38beaa67b03c32af51b4a97a9191` |
| `csrc/blockwise_scaling/common.h` | Apache-2.0；Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES | `e4bd6962fa68c4557df2f17fcaea0599d41304e9` |
| `csrc/rotary_positional_embedding.cu` | Apache-2.0；Copyright (c) 2020, NVIDIA CORPORATION | `41746aaef6d611ad210a1cf76948222ddba973f1` |

**待补证动作：**Transformer Engine `common.h` 的复合来源和 `block_scaled_cast_transpose.cu` 的接口参考基线已登记；仍需确认实际导入 Commit、补齐 `common.h` 未闭环宏的来源或内部授权及源码文件头，并保存包装层的导入/改编记录。其余 NVIDIA 文件仍需逐文件路径和固定 Commit，或内部授权记录。

### NVIDIA `cuda-samples` 本地仓库核验

本次使用本地完整克隆核验 NVIDIA CUDA Samples：<https://github.com/NVIDIA/cuda-samples>。

| 项目 | 核验结果 |
| --- | --- |
| 仓库 | `https://github.com/NVIDIA/cuda-samples.git` |
| 分支 | `master` |
| 固定 Tag | `v13.4` |
| Tag/分支 Commit | `5443602d89ed99aede2e4b7bf329daddeadb320e` |
| 上游 LICENSE 路径 | `LICENSE` |
| 上游 LICENSE Blob | `1e58bd5df0dd619135c58af0428c0f73c62921d7` |
| 上游 LICENSE SHA-256 | `b3e40c5bfed1fca5c62d2c1f2208bf51f8d2c910219f94c443f657ace9001be3` |
| 许可证说明 | NVIDIA CUDA Samples 许可证文本；包含 BSD-3-Clause 风格条款和 CUDA EULA 说明，不能直接替代 Lightop 文件的许可证判定 |

逐文件比对结果（比对范围为 `cuda-samples` 的当前 `master` 及全部可达 Git 历史）：

| Lightop 文件 | `cuda-samples` 候选 | 候选固定 Commit/Blob | 比对结论 |
| --- | --- | --- | --- |
| `csrc/blockwise_scaling/block_scaled_cast_transpose.cu` | `cpp/9_CUDA_Tile/tileTranspose/tileTranspose.cu`（同主题示例） | `5443602d89ed99aede2e4b7bf329daddeadb320e` / `221ea1523dfa0d1c1c78d3c31091a09d142ccb3f` | 文件路径、完整内容和 Blob 均不匹配；当前按 Transformer Engine 接口适配登记，cuda-samples 不作为来源 |
| `csrc/blockwise_scaling/common.h` | 无对应路径或同名文件 | — | 在 `cuda-samples` 全部可达历史中未找到逐文件或同名匹配 |
| `csrc/rotary_positional_embedding.cu` | `cpp/9_CUDA_Tile/tileRope/tileRope.cu`（同主题示例） | `5443602d89ed99aede2e4b7bf329daddeadb320e` / `8356d849fda874ab0e12349033ffc3b944f1e226` | 文件路径、完整内容和 Blob 均不匹配；仅能作为 RoPE 主题参考，不能作为来源 Commit |

此前 cuda-samples 核验使用的三个 Lightop 文件 Git Blob 分别为 `f49ce0d9173bbd7d73d94699bbba9ed96aea0453`、`64c1eccda927781931b9dcbf41abd3a1864fbdc5` 和 `8a003fa6cf7c84d4d550f106f97715fa238c10f3`，其中第一个是包装层恢复来源标注之前的历史版本；在 `cuda-samples` 全部可达对象中均没有对应 Blob。该核验排除了 cuda-samples 的完整文件来源关系。`common.h` 和包装层的 Transformer Engine 证据见上方条目；`rotary_positional_embedding.cu` 仍需其他 NVIDIA 官方仓库或中间派生仓库的逐文件来源和固定 Commit，或内部授权记录。

## 4. SUGON CORPORATION 历史版权声明文件（来源授权待补证）

| 项目 | 内容 |
| --- | --- |
| 历史原声明 | `Copyright (c) 2023, SUGON CORPORATION. All rights reserved.`；不是 HYGON CORPORATION |
| 许可证 | 初次引入文件包含完整 Apache-2.0 声明，本轮已逐字恢复 |
| 本地路径 | `csrc/bias_swiglu.cu` |
| 本地引入提交 | `2d416883f006555c1c1d0f8a054824ef943ec1b0` 的 `lightop/csrc/bias_swiglu.cu` |
| 声明变更证据 | `cfc95f717cb6ba71679890cfb375e554bda2f899` 替换原权利人；`c42aabb90b33d6208876a1f38390d0fc1ac5312c` 恢复；`652fa227e2f5399169d4b371e7ad615af7fe13e1` 删除原头 |
| 本轮处理 | 恢复历史原版权及许可全文；逐段比对确认相对初次引入仅删除两个过时 include，属于非实质修改，按 H3 保留修改说明并移除缺乏实质贡献依据的 HYGON Copyright；未改变算子实现 |
| 待补材料 | 实际来源或内部授权、权利转移（如有）的有效证明，以及原声明保留与品牌规则冲突的法务/治理结论 |

历史版权声明证明本仓库需要保留的原始声明，不单独证明完整权属链或公开分发批准。本轮不依据品牌规则删改原版权，不扩大 HYGON 的权利范围。

## 5. half 精度库（公开来源已核验）

| 项目 | 内容 |
| --- | --- |
| 项目/主页 | half - IEEE 754-based half-precision floating-point library；<https://sourceforge.net/projects/half/> |
| 主页项目说明 | SourceForge 将 `half` 描述为基于 IEEE 754 的半精度浮点库；该说明与本目录的 `half.hpp` 文件头和 MIT 许可证一致，但未提及 `hytsf`、`squeue` 或 Tensile 制品 |
| 版本 | 2.1.0（源码注释明确标注） |
| 许可证 | MIT |
| 版权 | Copyright (c) 2012-2019 Christian Rau <rauy@users.sourceforge.net> |
| 本地路径 | `third_party/librarygemmbiasf/half.hpp`; `third_party/librarygemmbiasf/LICENSE-half-2.1.0` |
| 本地引入提交 | `57872c41339ab8e906b67c7215a30be93435ecb1` |
| SourceForge 发布目录 | <https://sourceforge.net/projects/half/files/half/2.1.0/> |
| 核验日期 | 2026-10-09 |
| 发布归档 | `half-2.1.0.zip`；页面显示发布日期 `2019-08-05`、大小 `47.0 kB`；下载文件实测 `47,043` 字节 |
| SourceForge 页面 SHA-256 | `ad1788afe0300fa2b02b0d1df128d857f021f92ccf7c8bddd07812685fa07a25`（下载页显示值；本次下载后本地复算一致） |
| 归档内容 | `include/half.hpp`、`LICENSE.txt`、`README.txt`、`ChangeLog.txt`；归档内 `include/half.hpp` 与当前工作树文件逐字相同 |
| 归档/历史 CRLF `LICENSE-half-2.1.0` SHA-256 | `98df52a2aa12febe52f2a022f07e2a76cc4dc75839f02bb94868e8666396954d` |
| 历史 CRLF 工作树 `half.hpp` SHA-256 | `bb8592da2f97bab5e34ae3477358dd4c19aa3f909a3fde0864dfee2aa7c4ac1a`；与归档内 `include/half.hpp` 相同 |
| Lightop 处理 | 保留完整 MIT 许可文本、版权声明和版本注释；未添加 HYGON 文件头。本次已下载归档并复算 SHA-256；正式发布证据应保留归档或受控存档记录。 |

## 6. `librarygemmbiasf` HYGON 原创源码

以下五个源码文件已确认为 HYGON 原创实现，统一采用仓库根目录 Apache-2.0 许可，并在文件头标注 Hygon Information Technology Co., Ltd. 版权和 SPDX。它们不属于 SourceForge `half` 2.1.0 归档；`half.hpp` 仍按其原始 MIT 许可和版权声明单独登记。

| 本地路径 | 本地引入提交 | 许可证/版权 | 当前结论 |
| --- | --- | --- | --- |
| `third_party/librarygemmbiasf/compile.sh` | `57872c41339ab8e906b67c7215a30be93435ecb1` | Apache-2.0；Hygon Information Technology Co., Ltd. | HYGON 原创源码，已补 H1 文件头 |
| `third_party/librarygemmbiasf/hytsf.h` | `57872c41339ab8e906b67c7215a30be93435ecb1` | Apache-2.0；Hygon Information Technology Co., Ltd. | HYGON 原创源码，已补 H1 文件头 |
| `third_party/librarygemmbiasf/squeue.h` | `57872c41339ab8e906b67c7215a30be93435ecb1` | Apache-2.0；Hygon Information Technology Co., Ltd. | HYGON 原创源码，已补 H1 文件头 |
| `third_party/librarygemmbiasf/test-hytsf.cpp` | `57872c41339ab8e906b67c7215a30be93435ecb1` | Apache-2.0；Hygon Information Technology Co., Ltd. | HYGON 原创源码，已补 H1 文件头 |
| `third_party/librarygemmbiasf/test-hytsf.sh` | `57872c41339ab8e906b67c7215a30be93435ecb1` | Apache-2.0；Hygon Information Technology Co., Ltd. | HYGON 原创源码，已补 H1 文件头 |

SourceForge 的 [half 2.1.0 发布目录](https://sourceforge.net/projects/half/files/half/2.1.0/) 只覆盖 `half-2.1.0.zip` 中的 `include/half.hpp`、`LICENSE.txt`、`README.txt` 和 `ChangeLog.txt`，不构成上述五个文件的来源或授权证据。

### 历史构建线索（不能替代许可证或再分发授权）

初次引入提交 `57872c41339ab8e906b67c7215a30be93435ecb1` 同时提交了 `readme.txt` 和 `TensileManifest.txt`；随后目录整理提交将两份说明文件移除，并把目录从 `lightop/librarygemmbiasf` 重命名为 `third_party/librarygemmbiasf`。历史文件提供了以下可复核线索：

| 证据 | 内容 | 证明范围 |
| --- | --- | --- |
| `TensileManifest.txt`（引入提交） | `/home/guowei/gemm-bias/rocblas-dtk-22.04.2/build/release/Tensile/library/` 下的 `TensileLibrary.dat`、`Kernels.so-000-gfx900.hsaco`、`Kernels.so-000-gfx906.hsaco`、`TensileLibrary_gfx906.co` | 表明这些代码对象来自一次 ROCm/DTK 22.04.2 构建路径；不能证明该构建树的公开许可证或分发权 |
| `readme.txt`（引入提交） | 说明将上述 Tensile 制品部署到 `/opt/rocm/rocblas/lib/librarygemmbiasf`，并通过 `hytsf.h` 调用 `gemmbias_forward_*` API | 说明运行时集成方式；不能证明 `hytsf`、`squeue` 的权利人或许可证 |
| `compile.sh`（当前路径） | `hipcc -o test-hytsf test-hytsf.cpp -lhytsf -lsqueue -L.` | 证明测试可执行文件链接 `libhytsf.so` 和 `libsqueue.so`；不能证明这些动态库的生成源码或再分发授权 |

因此，SourceForge 的 half 2.1.0 证据仅覆盖 `half.hpp`；上述五个源码文件已按 HYGON 原创和根目录 Apache-2.0 处理。Tensile 制品、`libhytsf.so`、`libsqueue.so` 和 `test-hytsf` 仍须由组件负责人补充生成源码或来源仓库、固定版本/制品编号、权利人、许可证或内部再分发授权，并将授权范围绑定到本节列出的 SHA-256。

### 已纳入发布包的二进制证据

`setup.py` 和 `setup_torch29.py` 都会打包 `third_party/librarygemmbiasf/*`，`.github/scripts/manylinux_repair.sh` 还会把该目录加入动态库搜索路径。因此以下二进制不能按“只用于测试”处理，必须纳入授权核对：

| 本地文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `third_party/librarygemmbiasf/TensileLibrary.dat` | 2,589,651 | `6a483a3af3619ae63b00fbabc0f1f0aa63756abba43e345f6669c29b5992ec52` |
| `third_party/librarygemmbiasf/Kernels.so-000-gfx900.hsaco` | 478,064 | `3e7f1b571b5d98592ab349249b0e001076975e7f7e23f2772799fcd76cb88a91` |
| `third_party/librarygemmbiasf/Kernels.so-000-gfx906.hsaco` | 482,160 | `6e6b75181aac647b019cd8a1f887c62005b95fe2336a99882bf3374e43c6e82e` |
| `third_party/librarygemmbiasf/TensileLibrary_gfx906.co` | 6,128,584 | `1732b40140cd751f0691298e8d39d322b1f6cbb68890a25e40904418563e2ff6` |
| `third_party/librarygemmbiasf/libhytsf.so` | 3,563,816 | `2d182a001933f8fedb44e3364ac5ace61f1700cd448b774babc0747696374ecd` |
| `third_party/librarygemmbiasf/libsqueue.so` | 66,240 | `96216b42fd3b913e3db4574509e6c766c0370caeef33682961b69a11d4f205a9` |
| `third_party/librarygemmbiasf/test-hytsf` | 76,344 | `ed025d60dfc535d147d29c990df0ce14cdbf39a6c9be99f50aa26df6a5d26d86` |

### SourceForge 网址无法补充的二进制字段

SourceForge 的 `half` 项目不提供上述七个制品的构建或再分发证据。每个制品仍需补齐：

- 生成源码的 40 位 Commit 或内部制品版本，并与表中的 SHA-256 建立一一对应关系；
- 生成脚本、完整构建参数、编译器以及 DTK/ROCm 版本；
- 依赖库清单、`hytsf`/`squeue` 及其他依赖的许可证；
- 权利人、内部再分发批准和源码、wheel、二进制的公开分发范围；
- 从源码到制品的可复核构建关系，包括生成日期、目标架构和制品版本。

只有头文件授权而没有动态库、代码对象和测试可执行文件的授权时，不能认定 LightOp 发布包合规。

### `librarygemmbiasf` 授权记录模板

```yaml
component: librarygemmbiasf
source_type: internal-or-public
source_repository: <官方或内部仓库地址>
baseline_commit_or_artifact: <40位Commit或制品版本>
copyright_holder: <权利人>
license_or_distribution_authorization: <许可证或法务授权编号>
source_files:
  - third_party/librarygemmbiasf/compile.sh
  - third_party/librarygemmbiasf/hytsf.h
  - third_party/librarygemmbiasf/squeue.h
  - third_party/librarygemmbiasf/test-hytsf.cpp
  - third_party/librarygemmbiasf/test-hytsf.sh
artifact_files:
  - third_party/librarygemmbiasf/TensileLibrary.dat
  - third_party/librarygemmbiasf/Kernels.so-000-gfx900.hsaco
  - third_party/librarygemmbiasf/Kernels.so-000-gfx906.hsaco
  - third_party/librarygemmbiasf/TensileLibrary_gfx906.co
  - third_party/librarygemmbiasf/libhytsf.so
  - third_party/librarygemmbiasf/libsqueue.so
  - third_party/librarygemmbiasf/test-hytsf
source_sha256: <与本文件源码哈希表一致>
artifact_sha256: <与本文件制品哈希表一致>
source_diff: <相对固定来源快照的差异记录>
build:
  source_commit_or_artifact: <生成源码40位Commit或内部制品版本>
  script: <生成脚本和完整构建参数>
  compiler: <编译器及版本>
  dtk_rocm: <DTK/ROCm版本>
  dependencies_and_licenses: <依赖库清单及许可证>
public_distribution_scope: <源码、wheel、二进制或限制条件>
approver: <组件负责人和法务审批人>
approval_date: <YYYY-MM-DD>
```

## 7. 当前补证状态

| 组件 | 已确认 | 尚缺材料 | 处理结论 |
| --- | --- | --- | --- |
| AITER | 官方仓库、MIT、AMD 版权、逐文件路径、基线选择器、40 位 Commit、LICENSE Blob | `python/lightop/utils.py` 是改编版本，需保留差异说明 | 已完成公开来源固定；按改编关系审查 |
| vLLM | 官方仓库、Apache-2.0、逐文件路径、可达 `main`、40 位 Commit、LICENSE Blob | `w8a8_get_cache_json.py` 无匹配；DeepSeek V4 仍只有参考基线 | 已完成可匹配文件固定；两类未匹配项仍阻断 |
| Transformer Engine | NVIDIA Transformer Engine 仓库、`main`、40 位基线 `077e26c319d0fffbff75e56124b26b8b04aac0e5`、六个上游文件 Blob、LICENSE Blob 和 SHA-256、`common.h` 逐段映射 | 组件负责人确认实际导入基线；`TORCH_ERROR`、`TORCH_CHECK_CUDA`、`TRANSFORMER_ENGINE_TYPE_SWITCH_8BIT`、`DType::kInt8`、`Vec::operator=`、`__HOST_DEVICE__` 及 ATen 适配的来源/授权；源码文件头仍需按映射补充 `Adapted from`、SPDX 和 HYGON 修改声明 | 公开来源部分固定；未闭环片段和文件头整改完成前，`common.h` 仍保留来源审查项 |
| Transformer Engine 接口适配 | `csrc/blockwise_scaling/block_scaled_cast_transpose.cu`；用户确认来源项目；`main` 可达参考 Commit `077e26c319d0fffbff75e56124b26b8b04aac0e5`、两个参考文件及 Blob；NVIDIA 版权、Apache-2.0 SPDX 和 HYGON 修改说明 | 实际导入 Commit 与接口改编记录；本地基线无同名包装函数或完整文件匹配 | 已撤销原创归类并完成来源文件头标注；实际导入来源审查仍待闭环 |
| 其他 NVIDIA 文件 | NVIDIA 版权、Apache-2.0、本地路径和本地引入提交 | 公开仓库、逐文件路径和固定 Commit，或内部再分发授权 | 暂不关闭来源阻断 |
| SUGON 历史声明文件 | 初次引入的原版权、Apache-2.0、本地引入提交；已恢复源码原声明 | 实际来源或内部授权及品牌冲突治理结论 | 声明与清单矛盾已修复；授权及治理仍待闭环 |
| `librarygemmbiasf` | 五个源码文件的 HYGON 原创归类、文件头和根目录 Apache-2.0 | Tensile 制品、动态库和测试可执行文件的生成来源及公开分发授权 | 源码已确认；制品授权仍需补证 |
| half 2.1.0 | SourceForge 发布目录、`half-2.1.0.zip` 文件名、发布日期/大小、MIT、完整原文许可、页面显示 SHA-256、本地复算值和逐字文件比对 | 将下载归档或受控存档编号随发布证据保存 | `half.hpp` 公开来源已确认 |

## 8. 发布前核对

1. 为所有“来源待补证”文件补充可审计的上游 URL、固定版本或 Commit、版权和许可证，并将对应许可证文本或 NOTICE 纳入发布包。
2. 保留本文件已登记的 AITER/vLLM/Transformer Engine 上游 Commit；为 Transformer Engine `common.h` 的未闭环片段和文件头、`block_scaled_cast_transpose.cu` 的实际导入/改编记录、`w8a8_get_cache_json.py`、DeepSeek V4 独立来源、其他 NVIDIA 文件、SUGON 历史声明文件和 `librarygemmbiasf` 二进制制品补充可复现的公开 Commit 或内部授权记录。本地引入提交只能作为本仓库的追溯证据。
3. 保留第三方原始声明；HYGON 的原创或实质修改文件使用根目录 Apache-2.0，并按文件头规则补充 HYGON Copyright 和 SPDX。

## 9. 最新 master 新增文件（2026-10-12）

最新仓库 `master`（固定 Commit `5762c4c32535f4fb9532009b6744fc4c8d6c8504`）相对旧仓库合规基线 `6b8553cd227776f0f6c8886a036cf5eb7665ea20` 新增 40 个路径。新增文件按当前确认分为 HYGON 原创和作者/授权复核两类；本节列出的 9 个文件已按用户确认补充 HYGON Apache-2.0 原创文件头。

| 本地路径 | 当前状态 | 关闭条件 |
| --- | --- | --- |
| `csrc/paged_mqa_logits/dsv41_candidate_block_table.cu` | HYGON 原创；已补 `Copyright (c) 2026 Hygon Information Technology Co., Ltd.` 和 Apache-2.0 SPDX | 由组件负责人在正式审计中复核原创记录 |
| `csrc/paged_mqa_logits/paged_mqa_logits_persisten.cu` | HYGON 原创；已补 HYGON Apache-2.0 文件头 | 由组件负责人在正式审计中复核原创记录 |
| `csrc/paged_mqa_logits/paged_mqa_logits_persisten.h` | HYGON 原创；已补 HYGON Apache-2.0 文件头 | 由组件负责人在正式审计中复核原创记录 |
| `csrc/sparse_page_mqa/paged_sparse_mqa_logits_metadata.cu` | HYGON 原创；已补 HYGON Apache-2.0 文件头；DeepGEMM 兼容性仅描述接口布局 | 由组件负责人在正式审计中复核原创记录 |
| `csrc/sparse_page_mqa/paged_sparse_mqa_logits_metadata.h` | HYGON 原创；已补 HYGON Apache-2.0 文件头 | 由组件负责人在正式审计中复核原创记录 |
| `test/ci_tests/test_dsv41_candidate_block_table.py` | HYGON 原创测试；已补 HYGON Apache-2.0 文件头 | 由组件负责人在正式审计中复核原创记录 |
| `test/ci_tests/test_paged_mqa_logits_fp4_metadata.py` | HYGON 原创测试；已补 HYGON Apache-2.0 文件头 | 由组件负责人在正式审计中复核原创记录 |
| `test/ci_tests/test_paged_sparse_mqa_logits_metadata.py` | HYGON 原创测试；已补 HYGON Apache-2.0 文件头；DeepGEMM 语义为接口参考 | 由组件负责人在正式审计中复核原创记录 |
| `test/test_topk_transform_ragged_v2.py` | HYGON 原创测试；已补 HYGON Apache-2.0 文件头 | 由组件负责人在正式审计中复核原创记录 |

`csrc/mqa_logits/beta/` 下 24 个新增文件已有 HYGON Apache-2.0 版权/许可声明，本次保持其内容。Git 作者邮箱不能单独证明代码权属；这些文件仍保留作者和授权复核项。上述 9 个文件已完成用户确认的原创许可补充，不再列为来源待核对文件。

## 9. 2026-10-20 整改核验补充

核验基线为 `bfb9c85e8d357e9fba3a210d34e0e3cd207cce03`。逐文件核验记录和详细证据随本轮外部交付资料保存；固定来源及未闭环事项仍在本清单登记。历史登记中的“本次下载/核验”仍指该条原登记日期，不表示所有外部来源已在本轮重新验证。

### MIT 改编贡献（仓库级登记）

以下 HYGON 原创修改的版权为 `Copyright (c) 2026 Hygon Information Technology Co., Ltd.`，仅覆盖明确列出的本地贡献；相关文件继续遵循 MIT，保留 AMD 原文件头，不在文件头增加 HYGON 版权。

| 本地路径 | 已核验的本地修改 |
| --- | --- |
| `csrc/aiter_hip_common.h` | 移除 CK include、资源环境变量及日志适配、网格维度改为 `size_t`，增加 `launch_kernel_ext` |
| `python/lightop/utils.py` | shell 辅助函数、资源路径解析、LightOp 环境变量及设备查询适配 |

vLLM 的五个实质改编文件补充 HYGON 修改声明及仅覆盖原创修改的版权；两个测试辅助文件的函数定义与固定上游 AST 一致，但抽取、导入及模块说明有变化，仅补非实质修改声明；`scalar_type.py` 的完整 Blob 一致，保持原文件。各固定来源不因本轮声明修改而变更。

half 2.1.0 的当前 Git 文件为 LF 换行：`half.hpp` SHA-256 为 `7873fc452caf2e6808a49096dd01e22a3af3caa909b169d2c2ba600d1305f004`，独立许可证为 `d098b70e0b195ce538efa38b73c02fccbae89d169126efea7546892c7a3d5d9d`。转换为 CRLF 后分别匹配第5节历史归档哈希；换行差异不构成新来源。本轮已重新取得并保存官方归档 `half-2.1.0.zip`（47,043 字节，SHA-256 `ad1788afe0300fa2b02b0d1df128d857f021f92ccf7c8bddd07812685fa07a25`），逐字核对归档成员，排除换行差异后与当前文件相同。归档原件及核验记录随本轮外部交付证据保存。

第6节七个二进制的尺寸和 SHA-256 已在本轮逐个重算且与原登记一致；这只固定制品身份，不代表许可证或公开再分发权已获确认。两个打包入口在 wheel 许可证元数据及包内 `lightop/licenses/` 显式携带根许可证、第三方清单、AITER MIT 原文和 half 许可证（同时覆盖 legacy egg）；随包保存声明不替代待补授权。
