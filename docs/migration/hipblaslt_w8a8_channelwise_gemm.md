# hipblaslt_w8a8_channelwise_gemm 迁移记录

- 分类：`lightop.gemm_ops`
- 来源：LMSlim `lmslim/quantize/quant_ops.py`
- 实现策略：直接导出源保真的 hipBLASLt channelwise GEMM 实现。
- 计算语义：批量 INT8 `NT` GEMM，激活逐行缩放、权重逐输出通道缩放，支持 bias 和 FP16/BF16 输出。
- 测试：检查直接绑定与完整签名；FP16、BF16 分别和 PyTorch 参考计算对比。

测试结果记录在 `test-results/hipblaslt_w8a8_channelwise_gemm.txt`。
