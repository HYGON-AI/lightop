# rocblas_scaled_mm_nn 迁移记录

- 分类：`lightop.gemm_ops`
- 来源：LMSlim `lmslim/quantize/quant_ops.py`
- 实现策略：从 `lightop.gemm_ops` 直接导出私有源保真实现。
- 计算语义：INT8 `NN` 矩阵乘，乘以逐行/逐输出通道 FP32 scale，并支持可选 bias 与 FP16/BF16 输出。
- 测试：检查直接绑定和完整签名；FP16、BF16 分别与 PyTorch 参考结果对比。

测试结果记录在 `test-results/rocblas_scaled_mm_nn.txt`。
