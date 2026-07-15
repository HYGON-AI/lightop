# lightop.activation 既有接口归类记录

- 按设计文档将 19 个现有 LightOp activation/gating/dropout/drop-path API 直接重导出到 `lightop.activation`。
- 顶层兼容 API 保持不变，分类入口与顶层入口为同一 Python 对象。
- LMSlim 原生入口采用按需加载，导入既有接口时不会注册 LMSlim custom op。
- `test/test_activation_existing_api.py` 检查全部 19 个对象身份和 20 个公开符号清单。
