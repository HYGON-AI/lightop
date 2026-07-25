# lm_faster_rmsquant 迁移记录

- 分类：`lightop.norm`
- 来源：LMSlim `lmslim/quantize/quant_ops.py`
- 实现：从 `_lmslim_native.quantize.quant_ops` 按需直接导出，不复制或重写函数体。
- 覆盖：公开对象/签名一致；512、1024、4096 hidden size 的 INT8 RMSQuant 数值；scale；`update_input=True`；residual 累加与原地更新。
- 参考：PyTorch FP32 RMSNorm + per-token dynamic INT8 quant，量化误差允许 1 LSB。
