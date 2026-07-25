# hipblaslt_w8a8_gemm 迁移记录

- 分类：`lightop.gemm_ops`
- 来源：LMSlim `lmslim/quantize/quant_ops.py`
- 实现策略：复用已按原始目录语义迁入的私有实现，并从 `lightop.gemm_ops` 直接导出，不复制计算逻辑。
- 接口：保留输入、缩放因子、`NT` 转置标志、输出类型和可选 bias 的原始签名。
- 测试：检查公开接口为原实现的直接绑定，并分别以 FP16、BF16 输出对比 PyTorch 参考计算。

测试结果记录在 `test-results/hipblaslt_w8a8_gemm.txt`。
