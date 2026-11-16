# 索引、证据与缓存

本包携带可读的 `assets/index.jsonl`，保留3,672 页来源索引、别名和仓库映射。查询脚本需要 Python 3.10+ 和 PyYAML 6+；未安装 PyYAML 时可在开发环境执行 `python3 -m pip install 'PyYAML>=6'`，无需改变 LightOp 运行时依赖。

查询直接逐行加载仓库中的 JSONL，不解压、不生成索引缓存。默认离线，不下载 PR/diff。索引记录上游 URL、提交及 `artifact_dir`，不包含完整源码；历史约 1.1 GB 的完整证据树不随包分发。

从本 knowledge 目录使用：

```bash
python3 scripts/query.py "rmsnorm vectorized" --compact --limit 10
python3 scripts/get_page.py <page-id>
python3 scripts/fetch_evidence.py <page-id>
python3 scripts/fetch_evidence.py <page-id> --fetch
```

不加 `--fetch` 仅读取完整库或已缓存的 diff；没有证据则明确报错，不访问网络。`--fetch` 在缓存缺失时才访问 GitHub API 获取单条 PR diff，支持现有 `GITHUB_TOKEN`/`GH_TOKEN`；不会在参数、日志或 manifest 中保存凭据。已缓存内容校验 SHA-256 后复用，不自动重试或更新。脚本不会运行取得的源码。

自动获取目前支持 GitHub PR。索引若有 `head_ref_oid`，要求上游 HEAD 一致；没有记录 HEAD 时，manifest 明确标注历史版本匹配未知。获取前后核对上游 base/HEAD 稳定，并记录实际 base/HEAD、索引版本和 diff 内容哈希；这证明所取得快照的身份，不等于恢复了旧证据库或验证了该方案。索引过期、认证失败、网络不可达均显式失败。SourceFind/GitLab 等来源通过索引 URL 和有权限的工具取得相应版本，不虚构完整库。

证据缓存位于 `${XDG_CACHE_HOME:-~/.cache}/lightop-kernel-agent/evidence/`。无外网时可从联网机器拷贝相同索引对应的证据缓存，离线校验后读取；也可把 `KERNEL_KNOWLEDGE_ROOT` 指向预置完整知识目录，其中应有 `sources/` 和需要的 `evidence/pull-bundles/`；同样支持包含 `assets/index.jsonl` 的知识目录。脚本对证据包路径做目录边界检查，不将缺库误报为无匹配。

用户目录缓存不属于 Git 工作区。需要仓库内缓存时将 `XDG_CACHE_HOME` 设为仓库内 `.cache/`；`.cache/lightop-kernel-agent/` 与 `.lightop-kernel-agent/` 已由根 `.gitignore` 排除，不提交缓存、profile 或优化状态。原始 trace 按用户约定保存；本包不强制复制到其他位置。

LightOp 仅维护使用端。批量抓取、生成页面、重建/审计索引及仓库管理工具继续留在 hcu-infer-pilot。维护者从源知识库导出 JSONL 并通过代码审查更新：第一行为 `kind=catalog`、`schema_version=1`、条目数量和别名；后续每行是包含 `path/meta/body` 的 `kind=page`，或保留配套资料原文的 `kind=file`。非标准 YAML 格式通过可选 `frontmatter` 保留；`$date/$datetime` 恢复原元数据类型。查询和 import 不自动更新资料。

HCU 公开来源使用 HYGON-AI 的已核验仓库及固定提交。无法对应公开提交的旧私有 MR 和受保护入口不进入本包；相关原始记录保存在维护端，不能将旧 MR 编号直接套用为 GitHub PR 编号。
