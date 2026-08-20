# LightOp CI 说明

本文档记录 `.github/workflows/ci.yml` 的策略取舍，方便下一个接手 CI 的人不用重新推理。

## 触发矩阵

| 事件 | validate | test | publish |
|---|:---:|:---:|:---:|
| PR → `master` | ✓ | ✓ | ✗ |
| PR → `master`（来自 fork） | ✓ | ✗（不在内网 runner 上执行外部代码） | ✗ |
| `push` 到 `master`（PR merge 之后） | ✓ | ✓ | ✓ |
| Actions 面板 `workflow_dispatch`，勾选 `publish_packages` | ✓ | ✓ | ✓ |
| Actions 面板 `workflow_dispatch`，不勾 `publish_packages` | ✓ | ✓ | ✗ |

## 关键策略：**测例失败不阻塞 merge / publish**

`Run functional tests` 步骤按 pytest 退出码分类处理：

| pytest exit | 含义 | CI 行为 |
|---|---|---|
| `0` | 所有测例通过 | 正常绿 |
| `1` | 有测例失败 | **依然绿**，只在日志和 Step Summary 打 `::warning::` |
| 其他（2/3/4/5） | pytest 本身崩了（收集错误、用法错误、被中断、无测例） | **红**，job fail |

**为什么这么设计**：算子库在版本迭代过程中经常有已知的、允许暂时挂的测例（性能回退、数值容差、上游 torch 变化等）。硬门槛会把这些噪音变成阻塞。管理员应通过 **JUnit 报告**判断这些失败是否可接受，再决定是否 merge。CI 只做"跑完并如实报告"，不做"通过/不通过"的判断。

如果未来需要**强制某些测试必须通过**，两种做法：

1. 用 `@pytest.mark.critical` 标记，另加一步 `pytest -m critical`，这一步不做上面的容错，退出非 0 就 fail 整个 job；
2. GitHub 分支保护规则里勾 "Require status checks"，但因为 `test` job 恒绿，只能挡"pytest 崩了/编译失败"。

## 管理员如何决定是否 merge

1. 打开 PR，等 CI 跑完（validate + test × 每个 torch 版本）
2. 在 GitHub Actions run 页顶部看 **Step Summary**：一个 `Total/Passed/Failed/Errors/Skipped` 表格 + 折叠的失败测例列表（每条第一行错误消息）
3. 如果失败细节不够，下载 artifact `lightop-test-report-<pr>-<attempt>-torch-<ver>`（保留 30 天），里面是完整的 `lightop_test.xml`，本地 `pytest --junit-report` 视图工具或 CI 报告插件都能吃
4. 需要 wheel 也能下载：artifact `lightop-ci-<pr>-<attempt>-torch-<ver>`（14 天），仅在 test job 成功（含"仅测例失败"）时上传
5. 确认可接受 → merge → `push` 事件触发 publish，两个 torch lane 各上传一个 `*manylinux*.whl` 到 devpi

## 应急：不合并直接发包

Actions → LightOp CI → **Run workflow** → 选目标分支 → 勾 `publish_packages: true` → Run。

跳过 PR 流程，仍然要求 validate + test job 拉起来（test 恒绿，除非 pytest 崩）。用于修复紧急问题而不想走 PR review 的场景。

## GPU 分配

test job 编译前调用 `CIsmi need 1 --timeout ${CISMI_TIMEOUT}` 动态申卡，写入 `GITHUB_ENV` 后由 `docker run -e HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES` 传给编译容器。多 PR / 多 torch lane 并发时不会互相抢固定卡号。

超时可通过仓库变量 `LIGHTOP_CISMI_TIMEOUT` 覆盖（默认 3600 秒）。

## 可覆盖的仓库变量 / secrets

`vars.*`（Settings → Secrets and variables → Actions → Variables）：

- `LIGHTOP_TORCH_VERSIONS` — JSON 数组，默认 `["2.10.0", "2.11.0"]`
- `LIGHTOP_CI_BASE_IMAGE_REPOSITORY` — 编译 base image
- `LIGHTOP_DTK_PKG_PATH` — DTK 包下载地址
- `LIGHTOP_PYTHON_INDEX_URL` / `LIGHTOP_PYTHON_TRUSTED_HOST` — 内部 pip 索引
- `LIGHTOP_FALLBACK_INDEX_URL` / `LIGHTOP_FALLBACK_INDEX_HOST` — 内部索引不可达时的兜底
- `LIGHTOP_DEVPI_URL` — devpi 发布地址
- `LIGHTOP_CISMI_TIMEOUT` — CIsmi 申卡超时（秒）

`secrets.*`：

- `LIGHTOP_DEVPI_USER` / `LIGHTOP_DEVPI_PASSWORD` — devpi 发布凭据（缺失时 publish 会 fail-fast 并 mask 打印）
