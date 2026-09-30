# Third-Party Notices

Lightop 的根许可证是 Apache License 2.0，完整文本见 [LICENSE](LICENSE)。本文件登记仓库中已确认的第三方源码、版权和许可证义务。

## 记录方法

- “本地引入提交”是当前仓库 Git 历史中首次引入该路径的提交，用于固定本仓库中的证据；它不是第三方项目的上游 Commit。
- 当前仓库没有为部分复制或适配源码保存上游 Commit、发布包或 vendoring manifest。此类项目使用可核对的官方仓库或项目主页，并明确标注“上游 Commit 未随仓库保留”，不得把本地引入提交误写成上游版本。
- 源码文件中的原始版权和 SPDX/许可证声明已保留。对 HYGON 的适配或修改以 Git 历史为准；未确认来源的内部组件不机械添加 HYGON Copyright 或新的 SPDX 声明。

当前记录分为“已确认”和“待补证”两种状态。待补证项已经登记了本地路径、首次引入 Commit、当前文件哈希或二进制 SHA-256；补齐官方上游 Commit 或内部授权材料后，必须把状态改为“已确认”，再进行正式审计复扫。

## 1. ROCm AITER

| 项目 | 内容 |
| --- | --- |
| 项目/仓库 | ROCm AITER；<https://github.com/ROCm/aiter> |
| 许可证 | MIT |
| 版权 | Copyright (C) 2024-2025, Advanced Micro Devices, Inc. |
| 上游版本 | 上游 Commit 未随当前仓库保留；以源码中的 MIT SPDX 与版权声明为准 |
| 本地路径 | `csrc/aiter_hip_common.h`；`python/lightop/utils.py` |
| 本地引入提交 | `bd98e062c1aaa7515f6bcc917500561dfd3e8183`（C++ 头文件）；`e00781a470c084d0742f430a8bfcdf8a90ee8702`（Python 工具） |
| 当前文件 Blob | `26c0fcaeb48c487adc6ec57389fc212d9976a7e9`；`680bbf58b4aa1b53e2b431a14baed08f8acce94a` |
| Lightop 处理 | 保留 AMD 版权和 MIT 标识；代码在 Lightop 中被集成或适配，差异由本地 Git 历史记录。 |

**待补证动作：**从 AITER 官方仓库或内部镜像取得与上述两个文件逐字匹配的上游路径和 40 位 Commit，并登记上游 LICENSE 文件的路径和哈希。若无法匹配，应改走内部授权记录，不得继续把 AITER 作为确定来源。

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

这些文件保留 vLLM 的 Apache-2.0 SPDX 和 `Copyright contributors to the vLLM project` 声明。公开来源项目为 vLLM：<https://github.com/vllm-project/vllm>。当前仓库未保存对应的上游 Commit；下表的 Commit 均为本地引入提交。

当前已确认项目和许可证，尚未确认每个文件的上游固定 Commit。尤其是两个 DeepSeek V4 kernel，文件注释同时提到 `vllm-deepseek_v4`，需要核对 vLLM 主仓库、对应分支或独立公开仓库，不能仅凭文件名归入 vLLM。

### vLLM 版权声明文件

| 本地路径 | 许可证/版权 | 本地引入提交 |
| --- | --- | --- |
| `csrc/allreduce_fuse/_custom_ops.py` | Apache-2.0；vLLM project contributors | `5b05e4ed06d26c4eaac830b5d4d6e86138fdc1a6` |
| `csrc/allreduce_fuse/custom_all_reduce.py` | Apache-2.0；vLLM project contributors | `5b05e4ed06d26c4eaac830b5d4d6e86138fdc1a6` |
| `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` | Apache-2.0；vLLM project contributors | `07e7033161c35da365e32d6e7efaa44e36a3e96c` |
| `csrc/fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp_kernel.cu` | Apache-2.0；vLLM project contributors | `086435003c8a7c26391c60425b7facd4f44ad3f7` |
| `python/lightop/_lmslim_native/vllm_compat/moe_align_block_size.py` | Apache-2.0；vLLM project contributors | `b4b65b751fc6b46a4115289dab5b858ee4f97fd7` |
| `python/lightop/_lmslim_native/vllm_compat/w8a8_get_cache_json.py` | Apache-2.0；vLLM project contributors | `b4b65b751fc6b46a4115289dab5b858ee4f97fd7` |
| `test/common/fused_moe_kernel_gptq_awq.py` | Apache-2.0；vLLM project contributors | `32e4aad617608934ea2fea9283627f0c60527b2b` |
| `test/common/quant_utils.py` | Apache-2.0；vLLM project contributors | `32e4aad617608934ea2fea9283627f0c60527b2b` |
| `test/common/scalar_type.py` | Apache-2.0；vLLM project contributors | `32e4aad617608934ea2fea9283627f0c60527b2b` |
| `test/moe_marlin/moe_w16a16/benchmark_moe_Triton.py` | Apache-2.0；vLLM project contributors | `30b88ebc31c2d8fc4c57f6f6f1b9e0586752e856` |
| `test/moe_w16a16/benchmark_moe_Triton.py` | Apache-2.0；vLLM project contributors | `30b88ebc31c2d8fc4c57f6f6f1b9e0586752e856` |

**待补证动作：**对每个本地路径记录 vLLM（或 DeepSeek V4 独立仓库）的原始路径、上游 40 位 Commit、上游 LICENSE/NOTICE 路径和本地差异。当前已保存的本地引入提交只能作为 LightOp 追溯证据。

## 3. NVIDIA 版权声明文件

以下文件明确保留 NVIDIA 的 Apache-2.0 原始声明。源码布局与 vLLM 相关组件存在对应关系，vLLM 仓库（<https://github.com/vllm-project/vllm>）是可复核的公开线索，但当前 Git 历史没有保存能唯一确认的上游路径和 Commit，因此不能把该线索当作固定上游版本。发布前应补充逐文件来源证据；在此之前保留原始版权和许可证声明。

| 本地路径 | 许可证/版权 | 本地引入提交 |
| --- | --- | --- |
| `csrc/blockwise_scaling/block_scaled_cast_transpose.cu` | Apache-2.0；Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES | `1f0e6d421abc38beaa67b03c32af51b4a97a9191` |
| `csrc/blockwise_scaling/common.h` | Apache-2.0；Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES | `e4bd6962fa68c4557df2f17fcaea0599d41304e9` |
| `csrc/rotary_positional_embedding.cu` | Apache-2.0；Copyright (c) 2020, NVIDIA CORPORATION | `41746aaef6d611ad210a1cf76948222ddba973f1` |

**待补证动作：**以函数名、注释和完整文件内容分别检索 vLLM 与 NVIDIA 公开仓库。只有找到逐文件匹配并能锁定 40 位 Commit 后，才能填写公开来源；否则应由组件负责人提供内部授权记录，不得把 vLLM 线索写成事实。

## 4. SUGON CORPORATION 来源文件

| 项目 | 内容 |
| --- | --- |
| 来源 | `SUGON CORPORATION` 版权声明；当前 Git 历史未提供可核对的公开仓库 URL 或上游 Commit |
| 许可证 | Apache-2.0（文件内完整声明） |
| 本地路径 | `csrc/bias_swiglu.cu` |
| 本地引入提交 | `2d416883f006555c1c1d0f8a054824ef943ec1b0` |
| 处理要求 | 保留原始版权和 Apache-2.0 条款，不将其改写为 HYGON 版权。法务或组件负责人需在发布前补充来源授权记录；该文件中的 SUGON 身份属于第三方原始声明。 |

## 5. half 精度库

| 项目 | 内容 |
| --- | --- |
| 项目/主页 | half - IEEE 754-based half-precision floating-point library；<https://sourceforge.net/projects/half/> |
| 版本 | 2.1.0（源码注释明确标注） |
| 许可证 | MIT |
| 版权 | Copyright (c) 2012-2019 Christian Rau <rauy@users.sourceforge.net> |
| 本地路径 | `third_party/librarygemmbiasf/half.hpp` |
| 本地引入提交 | `57872c41339ab8e906b67c7215a30be93435ecb1` |
| Lightop 处理 | 保留完整 MIT 许可文本、版权声明和版本注释；未添加 HYGON 文件头。 |

## 6. `librarygemmbiasf` 内部组件（来源待补证）

以下文件与 `half.hpp` 一同进入仓库，但当前 Git 历史只显示本地引入，没有公开仓库 URL、固定版本或许可证文本。它们可能属于内部 `hytsf`/`squeue` 组件，不能根据目录名推断许可证。发布前必须由组件负责人补齐来源、版权、许可证和授权记录；在补证前不得把它们声明为 Apache-2.0 或 MIT，也不得机械添加 HYGON 文件头。

| 本地路径 | 本地引入提交 | 当前结论 |
| --- | --- | --- |
| `third_party/librarygemmbiasf/compile.sh` | `57872c41339ab8e906b67c7215a30be93435ecb1` | 构建脚本；来源和许可证待补证 |
| `third_party/librarygemmbiasf/hytsf.h` | `57872c41339ab8e906b67c7215a30be93435ecb1` | `hytsf` API 头文件；来源和许可证待补证 |
| `third_party/librarygemmbiasf/squeue.h` | `57872c41339ab8e906b67c7215a30be93435ecb1` | `squeue` API 头文件；来源和许可证待补证 |
| `third_party/librarygemmbiasf/test-hytsf.cpp` | `57872c41339ab8e906b67c7215a30be93435ecb1` | 测试程序；来源和许可证待补证 |
| `third_party/librarygemmbiasf/test-hytsf.sh` | `57872c41339ab8e906b67c7215a30be93435ecb1` | 测试脚本；来源和许可证待补证 |

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

**内部授权记录必须覆盖：**上述二进制的生成源码、编译工具链、权利人、可公开分发范围、依赖的 `hytsf`/`squeue` 许可证，以及这些精确 SHA-256 对应的制品版本。只有头文件授权而没有动态库授权时，不能认定 LightOp 发布包合规。

### `librarygemmbiasf` 授权记录模板

```yaml
component: librarygemmbiasf
source_type: internal-or-public
source_repository: <官方或内部仓库地址>
baseline_commit_or_artifact: <40位Commit或制品版本>
copyright_holder: <权利人>
license_or_distribution_authorization: <许可证或法务授权编号>
files:
  - third_party/librarygemmbiasf/hytsf.h
  - third_party/librarygemmbiasf/squeue.h
  - third_party/librarygemmbiasf/libhytsf.so
  - third_party/librarygemmbiasf/libsqueue.so
artifact_sha256: <与本文件表格一致>
public_distribution_scope: <源码、wheel、二进制或限制条件>
approver: <组件负责人和法务审批人>
approval_date: <YYYY-MM-DD>
```

## 7. 当前补证状态

| 组件 | 已确认 | 尚缺材料 | 处理结论 |
| --- | --- | --- | --- |
| AITER | 官方仓库、MIT、AMD 版权、本地路径和本地引入提交 | 上游逐文件 40 位 Commit、LICENSE 哈希 | 暂不关闭来源阻断 |
| vLLM | 官方仓库、Apache-2.0、vLLM 版权、本地路径和本地引入提交 | 每个文件的上游 Commit；DeepSeek V4 的实际上游仓库 | 暂不关闭来源阻断 |
| NVIDIA | NVIDIA 版权、Apache-2.0、本地路径和本地引入提交 | 公开仓库、逐文件路径和固定 Commit | 暂不关闭来源阻断 |
| SUGON | 原始版权、Apache-2.0、本地引入提交 | 来源 URL 或内部再分发授权 | 暂不关闭授权阻断 |
| `librarygemmbiasf` | 本地 Commit、源码路径、打包配置和二进制 SHA-256 | 源码/制品来源、许可证、公开分发授权 | 暂不允许发布 |
| half 2.1.0 | SourceForge 主页、版本、MIT、完整原文许可 | 可复现的上游归档校验值 | 可保留，建议补归档哈希 |

## 8. 发布前核对

1. 为所有“来源待补证”文件补充可审计的上游 URL、固定版本或 Commit、版权和许可证，并将对应许可证文本或 NOTICE 纳入发布包。
2. 为 vLLM、AITER 和 NVIDIA 来源补充可复现的上游 Commit 或内部 vendoring 记录；本地引入提交只能作为本仓库的追溯证据。
3. 保留第三方原始声明；HYGON 的原创或实质修改文件使用根目录 Apache-2.0，并按文件头规则补充 HYGON Copyright 和 SPDX。
