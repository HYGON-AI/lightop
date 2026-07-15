# triton_scaled_mm 迁移记录

- 分类：`lightop.gemm_ops`
- 来源：LMSlim `lmslim/quantize/quant_ops.py`
- 实现策略：直接导出源保真实现，保留“LightOp 优先、Triton 回退”的原始调度逻辑和 `best_config` 参数。
- 计算语义：INT8 GEMM，逐 token/逐输出通道缩放，可选 bias，FP16/BF16 输出。
- 测试：检查直接绑定与完整签名，并对两种输出类型进行 PyTorch 数值对比。

测试结果记录在 `test-results/triton_scaled_mm.txt`。
