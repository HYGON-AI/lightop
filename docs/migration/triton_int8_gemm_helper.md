# triton_int8_gemm_helper 迁移记录

- 分类：`lightop.gemm_ops`
- 来源：LMSlim `lmslim/quantize/quant_tools.py`
- 实现策略：直接导出源保真的 GEMM 暖机/配置辅助函数。
- 接口语义：按给定维度和量化粒度生成 INT8 输入，调用 `triton_scaled_mm` 指定次数并同步 GPU；保持原始 `None` 返回值。
- 测试：检查直接绑定和完整签名，并在 ROCm GPU 上完成一次含 bias 的逐 token/逐通道暖机调用。

测试结果记录在 `test-results/triton_int8_gemm_helper.txt`。
