# 2026-10-20 LightOp 合规整改记录

## 基线与结论边界

- 核验及整改起点：`bfb9c85e8d357e9fba3a210d34e0e3cd207cce03`；原报告基线：`176d8723fe41c2aa5010c10b14003d9ffb9125e4`。
- 两者相差11个提交、32个变更文件，本轮检查了这些增量文件。
- 本轮完成有证据支持的代码及声明整改；来源、授权和治理待办不以编译测试结果销项。本记录不是中心化规则 `hygon-open-source-v1.7` 的正式复扫报告。
- 原有 Git 历史保持不变；`7fc81eb1255f` 身份问题留作发布前治理事项。

## 已修复

### 原报告四项及增量文件头

按原创全树规则核对本地引入记录和文件内容：以下文件未发现第三方声明或生成器证据，补充 H1；`.inc` 为手写实现包含文件，并非仅因扩展名就作为生成文件处理。已有 Apache SPDX 的测试文件仅补版权。

- `csrc/dsv41_pair_partners.cu`
- `test/ci_tests/test_dsv41_select_candidate_blocks.py`
- `test/ci_tests/test_hc_mix_sinkhorn.py`
- `test/test_situ_ep_packed_schedule.py`
- `csrc/mla/indexer_k_prepare.cu`
- `csrc/mla/indexer_q_prepare.cu`
- `csrc/mla/mla_q_absorb_concat.cu`
- `csrc/sparse_page_mqa/paged_sparse_mqa_logits.cu`
- `csrc/sparse_page_mqa/paged_sparse_mqa_logits.h`
- `csrc/sparse_page_mqa/paged_sparse_mqa_logits_gfx936.inc`
- `csrc/sparse_page_mqa/paged_sparse_mqa_logits_gfx938.inc`
- `csrc/sparse_page_mqa/prefill_sparse_mqa_bindings.cpp`
- `csrc/sparse_page_mqa/prefill_sparse_sort.h`
- `test/test_fp8_fp4_paged_sparse_mqa_logits.py`
- `test/test_indexer_k_prepare.py`
- `test/test_indexer_q_prepare.py`
- `test/test_mla_q_absorb_concat.py`
- `tests/test_dsv4_topk_workspace.py`

### 历史声明与第三方改编

- `bias_swiglu.cu`：逐字恢复初次引入的 SUGON 原版权和完整 Apache 条款；修改历史见第三方清单第4节。逐段比对初次引入版本，内核主体未变，仅删除两个过时 include（`ATen/CUDAGeneratorImpl.h`、`THC/THCGeneral.h`）。因此采用 H3，只保留修改说明，移除缺乏实质贡献证据的 HYGON Copyright；未改变实现。
- 五个 vLLM 实质改编文件采用 H2；两个函数抽取辅助文件采用 H3，仅标注抽取/导入修改，不声明函数主体为 HYGON 原创；`scalar_type.py` 保持逐字副本。
- AITER 两个 MIT 改编文件保留 AMD 头，在第三方清单第9节登记 HYGON 本地贡献。
- `hytsf.h` 从 `100755` 改为普通源码 `100644`。
- README 已包含项目名、根许可证和第三方清单入口，无需重复修改。
- 两个 setup 入口显式携带许可证及第三方声明，保留现有二进制打包行为。

## 已核实的来源关系

十个 AITER/vLLM 本地路径对应九份固定上游文件，本轮重新取得内容并验证 Git Blob；完整关系和函数核验结果见 `20261020-source-evidence.json`。源码差异与取得的原始内容存入本轮外部证据目录。

| 文件/组 | 核验结果与处置 |
| --- | --- |
| AITER `aiter_hip_common.h`、`utils.py` | 上游 Blob 与原清单一致；确认扩展 launch、资源路径及环境适配，MIT 原头不变 |
| vLLM `_custom_ops.py`、`custom_all_reduce.py`、`moe_align_block_size.py` | 明确存在本地接口/算子调度修改，保留原声明并补 H2 |
| 两个 `benchmark_moe_Triton.py` | 分别核验；两者本地 Blob 相同，均为 HCU/NN-MoE 改编，补 H2 |
| `fused_moe_kernel_gptq_awq.py` | `write_zeros_to_output`、`fused_moe_kernel_gptq_awq` 的 AST 与上游相同；抽取及导入改变采用 H3 |
| `quant_utils.py` | `get_pack_factor`、`pack_cols`、`quantize_weights`、`awq_pack` 的 AST 与上游相同；抽取及导入改变采用 H3 |
| `scalar_type.py` | 完整 Blob `05760f3f829988d4640955b6d19ff2549b1d28cd` 相同，保持原文件 |
| half 2.1.0 | 已重新取得官方47,043字节归档；归档及成员哈希匹配原登记，LF/CRLF 归一后与本地逐字相同，原件随交付证据保存 |

half 归档的固定下载地址、SHA-256 与成员比对见 `20261020-half-archive.json`；归档原件在本轮交付证据中以 `half-2.1.0.zip` 保存。

AITER 和 vLLM 的所记录许可证 Blob 已重新取得并核对。在本次核验的 AITER `b25a045a...` 和 vLLM `8fcaaf6a...` 根路径未找到独立 `NOTICE`；此检查不扩展为所有组件或所有历史版本不存在 NOTICE 的结论。

## 未闭环事项

| 项目 | 本轮可核实证据 | 仍需材料/结论 |
| --- | --- | --- |
| Transformer Engine `common.h` | 本地首次引入 `e4bd6962...`，后续 `1f0e6d42...`、`8cdf607e...`；既有六文件片段映射保留 | 实际导入基线、未闭环宏/ATen 片段来源及授权；闭环后完成相应文件头 |
| `block_scaled_cast_transpose.cu` | 本地引入 `1f0e6d42...`；现有 NVIDIA 声明和修改说明保留 | 实际导入版本及接口改编链；参考 Commit 不冒充导入 Commit |
| DeepSeek V4 原始融合内核 | 引入 `07e70331...`，后续 `61d8c82d...` | 真实派生仓库、固定 Commit 及 ROCm/ATen 改编链 |
| DeepSeek V4 PCP 变体 | 引入 `08643500...` | 原始实现到 PCP 变体的派生关系与修改归属 |
| `w8a8_get_cache_json.py` | 迁移提交 `b4b65b75...` | 真实来源或内部权利依据；现有 vLLM 头不等于来源证明 |
| `rotary_positional_embedding.cu` | 引入 `41746aae...`，后续融合/优化历史可追溯 | NVIDIA 官方或中间派生仓库的固定来源，或内部授权 |
| `bias_swiglu.cu` | 初始原声明已恢复，清单错误已纠正 | 来源/授权、权利转移（如有）、版权保留与品牌规则冲突的治理结论 |
| 七个发布二进制 | 见下方逐个重算的尺寸及哈希 | 各制品生成源码/固定版本、构建链、依赖许可证、权利人、公开分发授权范围 |
| 历史 commit `7fc81eb1255f` | 保留原报告身份问题记录 | 发布前处理作者/提交者身份；本轮仅新增 commit，不重写历史 |
| 正式复扫 | 本轮为工程整改及证据核验 | 在最终固定提交上执行中心化扫描，保存正式报告 |

七个制品均位于 `third_party/librarygemmbiasf/`，当前默认打包入口会复制到安装包的 `lightop/librarygemmbiasf/`。下列哈希与原登记一致，不能视为授权批准：

| 制品 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `TensileLibrary.dat` | 2589651 | `6a483a3af3619ae63b00fbabc0f1f0aa63756abba43e345f6669c29b5992ec52` |
| `Kernels.so-000-gfx900.hsaco` | 478064 | `3e7f1b571b5d98592ab349249b0e001076975e7f7e23f2772799fcd76cb88a91` |
| `Kernels.so-000-gfx906.hsaco` | 482160 | `6e6b75181aac647b019cd8a1f887c62005b95fe2336a99882bf3374e43c6e82e` |
| `TensileLibrary_gfx906.co` | 6128584 | `1732b40140cd751f0691298e8d39d322b1f6cbb68890a25e40904418563e2ff6` |
| `libhytsf.so` | 3563816 | `2d182a001933f8fedb44e3364ac5ace61f1700cd448b774babc0747696374ecd` |
| `libsqueue.so` | 66240 | `96216b42fd3b913e3db4574509e6c766c0370caeef33682961b69a11d4f205a9` |
| `test-hytsf` | 76344 | `ed025d60dfc535d147d29c990df0ce14cdbf39a6c9be99f50aa26df6a5d26d86` |

## 工程验证

本轮对所有修改的 Python 源码检查 AST 保持不变（setup 的许可证元数据除外），对 C/C++ 文件检查原实现完整保留；执行 `git diff --check`。
编译安装、全量 `test/ci_tests`、SiTU 三组环境组合的结果以最终提交对应的独立构建日志、JUnit 和交付记录为准。在结果产生前，本记录不宣称测试已通过。
