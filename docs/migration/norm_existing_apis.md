# lightop.norm 既有接口归类记录

- 按设计文档将 22 个现有 LightOp normalization/residual-fusion API 直接重导出到 `lightop.norm`。
- 顶层 `lightop.<name>` 保持不变，分类入口与顶层入口为同一 Python 对象。
- LMSlim 原生入口采用按需加载，导入既有接口时不会注册 LMSlim custom op。
- `test/test_norm_existing_api.py` 检查全部 22 个对象身份和 23 个公开符号清单。
