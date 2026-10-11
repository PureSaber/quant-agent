# 建模、稳健性与研究治理扩展

`quant_agent.research_workflow` 是 quant-agent 内独立的本地研究业务层。它读取数据、
Lab 记录与已有检索结果，输出可复核产物；不写 Lab 的登记/冻结记录、Factors 核心、
账户、执行授权、调度或 Studio。所有状态与报告都包含 `execution_authorized: false`。

本次支持范围是**结构化回归研究**，内置训练均值、岭回归（含 OLS）和 KNN。
支持多资产行、缺失特征的训练内中位数填补、训练内标准化、多个时间验证窗口、
参数比较、一次性最终留出集、JSON 模型及预测归档。分类、神经网络、在线更新、
超大规模训练和真实订单不在这个接口版本内。

## 安装和入口

沿用仓库 `requirements.lock`，没有改变依赖版本或共享发布锁：

```bash
python -m pip install --no-deps -r requirements.lock
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check
quant-research-workflow --help
# 等价入口
python -m quant_agent.research_workflow.cli --help
```

DSR、CSCV/PBO 无需额外安装。Bootstrap 与 SPA/MCS 使用 quant-lab 已声明的
`statistics` 可选依赖 `arch>=7.2,<9`；未安装时报告逐项 `available: false` 和原因，
不会把未计算当成通过。测试保留这种降级分支。仓库主锁不包含可选统计栈，不能把
可选安装称作主锁环境复现；实际验证环境的完整 freeze 应与报告一起保存。

CLI 提供 `ml`、`verify`、`robustness`、`lifecycle`、`graph`。返回码 2 表示输入错误、
失败或未完成的 ML 家族。写入命令使用新文件/目录，不覆盖已发布产物。

## M1：只读数据与模型闭环

输入是 `quant.ml-dataset/v1` JSON 清单和同目录内 CSV，具体可由示例生成。

| 字段 | 含义与约束 |
| --- | --- |
| `data_file`, `data_sha256` | 相对路径与文件 SHA-256；路径不得逃离清单目录 |
| `features` | 唯一特征列名；每项都有 `<feature>__known_at` |
| `sample_kind` | `synthetic`、`retrospective` 或 `historical_pit`；必须附 `provenance` |
| `target_kind` | 当前仅 `regression`；预测误差不被转换为策略收益 |
| `row_id`, `asset` | 唯一行 ID；同资产同决策时刻不得重复 |
| `decision_at` | 带明确时区的决策时点；同一时刻的资产行一起划分 |
| `<feature>__known_at` | 每项特征的可得时刻，必须不晚于该行决策 |
| `target_start`, `target_end`, `target_known_at` | `decision <= start < end <= known` |
| `target` | 有限的实际回归目标；缺失目标不得填造 |

仅特征 NaN 可由训练集拟合的中位数填补；某特征在训练中全缺失会使该尝试失败。
无穷值、无时区时间、重复实体时点、未知 schema、文件摘要错误均拒绝。

`historical_pit` 是**数据生产者声明**，时间字段只能检查内部一致性，不能证明历史
修订版本或真实发布时间。批量重新拟合的程序拒绝 `independent_forward` 输入标记，
避免把一次回放写成独立前向；结果始终标明 batch chronological replay。

`quant.ml-study/v1` 定义 `study_id`、`family_id`、假设、完整模型/参数列表、
`train_start`、不重叠的验证 `folds`、最终 `holdout`、`evaluation_as_of`，以及可选
`embargo_seconds` 和已有格式的 `lab_references`（会调用 `verify_citations`）。
当前选择规则固定为全部验证样本的加权 MSE，完全相同时按候选 ID 排序。

每个窗口在起点减 embargo 时拟合一次。只使用决策和标签可得时点**严格早于**拟合
时点的训练行。验证标签还必须在最终留出集起点前已知；边界未闭合标签和未可得标签
明确记录在 `purged_training_rows` / `unscored_evaluation_rows`，没有静默跨日期填充。
验证窗口结束后才公开的标签，可以在最终选择时使用，但不会进入其未知时点之前的训练。

全部候选、全部验证窗口都记 `started.json` 与 `terminal.json`。任何失败都会使家族
不完整并禁止打开最终留出集。候选选定后先持久化 `holdout-consumed.json`，再进行
最终训练/预测；最终训练允许使用留出集前已经公开的验证期标签。留出集仅评估选定模型。

```bash
quant-research-workflow ml --plan plan.json --dataset dataset.json --output-root runs
quant-research-workflow verify runs/study-id
```

产物包括 `plan.json`、`quant.ml-dataset-reference/v1` 数据引用、所有尝试记录、
训练行/评估行/拟合时点、每次模型 JSON、解释、完整验证预测与选定模型的最终预测、
`result.json` 和逐文件摘要的 `archive.json`。保存了运行版本与实现文件摘要。
数据引用不是输入数据的拷贝；应保留原始清单和 CSV。模型为普通 JSON，加载不执行 pickle。
岭回归解释含标准化/原单位系数；KNN 解释含训练空间与近邻均值机制；均值基线解释含训练均值。
这些是描述性解释，没有因果含义。

不可变目录是**本地治理域**。同根下不能重用 study ID；崩溃留下的目录也阻止自动重试，
要保留其 running/失败事实。删除目录、改 ID 或改输出根能绕过本地限制，因此协调者须保留
完整家族和研究历史。摘要是完整性检查，不能替代外部签名或证明真正的事前登记。

## M2：统一稳健性

`quant.robustness-plan/v1` 与收益 CSV 的哈希封装为 bundle。输入必须是已产生的、
**相同观测日期的净简单收益**，并提供明确基准、币种、成本政策、年化频率、样本类型。
这不是预测值变成收益的隐式回测器；ML 输出不会被乘以实际目标来伪造可投资收益。

计划记录完整 `planned`、`statuses`、`parameters`、不重叠且覆盖全部输入的 `intervals`、
`holdout_start`、CSCV 区块数、bootstrap 区块长度列表、重复次数、种子和 alpha。
只接受 `partition: validation`，任何到达最终留出集起点的收益行均拒绝。
失败、未尝试、重试、缺失日期和未登记候选不得从家族中消失。

复用接口：

- `quant_lab.selection.audit_family` 与 `quant_lab.family_evidence.evaluate_family`：
  DSR、CSCV/PBO、共同日期区块 bootstrap、SPA/MCS。CSCV 样本不能整除区块时标记不可用，
  不丢弃尾部。没有基准时不假定无风险利率为零。
- `quant_factors.validation.benjamini_hochberg`：对所有候选 × 区间 × 区块长度的 bootstrap
  检验一起调整，同时列出 Bonferroni 上界。BH 的独立/正依赖条件未获证明，需看保守结果；
  任一子家族不能计算时，不对剩余部分宣称完整家族推断。
- `registered_family(database, family_id)`：以 SQLite `mode=ro`/`query_only` 复用现有
  `TrialRegistry` 读取与 `collect_family`，检查登记摘要、结果及产物摘要；发现并发变动即失败。
- `registered_robustness(database, family_id, plan, benchmark)`：强制完整登记清单和参数精确一致；
  不能遗漏失败重试或重新描述其参数。数据不完整时返回完整审计与不可用原因。

新增参数敏感性只比较恰好一项参数不同的候选，输出配对均值差与逐期改善占比；这是差异描述，
不是“稳健通过”的认证。跨区间报告列出全部统计方法和复利/回撤/Sharpe，绝不自动选策略。

```bash
quant-research-workflow robustness robustness-bundle.json --output robustness.json
```

独立 CSV bundle 的登记时序是外部责任；报告明确说“frozen input plan”，不伪称有受信时间戳。
DSR 不是未来盈利的后验概率，CSCV 不是按时间顺序的实盘检验，SPA 不拒绝也不证明所有策略无效。

## M3：研究生命周期

`ResearchLifecycle` 只创建自己的 `quant.research-lifecycle/v1` SQLite 文件；遇到已有 Lab、
账户或其他 schema 的数据库直接拒绝。读操作使用只读连接。写入使用事务、序号、前序摘要和
`expected_head` 乐观并发检查，保存 actor、原因、递增时点、证据与 `execution_authorized=false`。

| 当前状态 | 可进入状态 |
| --- | --- |
| 未创建 | candidate（候选） |
| candidate | review、paused、retired |
| review（复核） | candidate、observe、paused、retired |
| observe（研究观察） | review、paused、retired |
| paused（研究暂停） | review、retired |
| retired（淘汰） | 终态 |

每次转换都要求有效的已有 JSON/SQLite 引用；观察晋级还要求历史引用仍有效。
先前来源变更时仍可用新的有效证据暂停，避免失效证据阻止研究停用。
actor 是调用者声明，未实现身份认证、多人审批或账户权限映射。

```bash
quant-research-workflow lifecycle init --database lifecycle.sqlite --strategy sample
quant-research-workflow lifecycle transition --database lifecycle.sqlite --strategy sample --event event.json
quant-research-workflow lifecycle state --database lifecycle.sqlite --strategy sample
```

`event.json` 包含 `to_state, actor, reason, evidence, recorded_at, expected_head`。
`evidence` 沿用 `research_history` 引用；`cite_json(path, pointer, citation_id, claim)` 可构造
同格式引用。研究 observe 不激活真实策略，研究 paused 不改变任何现有前向账户。

## M4：假设—证据—结论

复用已有 `research_history` 搜索和 `verify_citations`，不另建搜索/Notebook 建议功能。
`evidence_graph(history, assertions)` 或 `search_evidence_graph(query, assertions, ...)`
生成假设、已完成证据、技术失败、明确结论及可追溯边。

每项 assertion 指定 `study_id`、`attempt_id`、稳定 `claim_key`、理由、自己的引用 ID、
`supports/rejects/inconclusive/execution_failed`，以及数据摘要、区间、总体、目标、成本和
样本类型组成的完整 `scope`。已完成结论另需 `artifact_root`，每次建图重读产物字节并验证摘要。
技术失败只连 `failed_to_test`，不能用于支持或否定投资假设。负面研究结论使用 `rejects`，
与代码/数据失败分开保存。

相同 claim 与完全相同 scope 的相反、来源仍有效结论会形成待复核冲突；scope 不同列为
incomparable。引用失效的结论仍保留但 `verified=false`。引用真实性只能证明来源未变，
不能证明人的推理成立；搜索结果上限和缺失证据仍在输出中，图不是全研究库证明。

```bash
quant-research-workflow graph "lagged" --assertions assertions.json --study-path study.json --output graph.json
```

## 可重复运行的例子与独立对照

```bash
python examples/research_workflow.py --output ../evidence/synthetic-new
python examples/verify_research_workflow.py ../evidence/synthetic-new
# 本地已有 statsmodels 可选统计栈时，可使用其随包真实宏观历史样本
python examples/research_workflow.py --output ../evidence/macro-new --macrodata-csv PATH_TO_MACRODATA_CSV
python examples/verify_research_workflow.py ../evidence/macro-new
```

示例构造完整输入、ML 归档、研究历史兼容视图、带技术失败的证据图、研究状态流转、
稳健性 bundle/report 和 summary。示例失败节点明确标为演示，不冒充真实研究失败。
两种示例里的**金融收益稳健性矩阵均为 synthetic**。宏观示例的 ML 部分使用实际历史
失业率和短期利率，目标是下一季度失业率变化，绝不当作投资收益。

宏观来源是 [statsmodels macrodata](https://www.statsmodels.org/stable/datasets/generated/macrodata.html)
保存的 FRED/BLS 历史截面。输入 200 行，1959Q3–2009Q2，前两行用于构造滞后特征、最后一行
缺少下一期标签而不进入数据集。特征滞后两季、季度末 +100 天可得时间是**明确的回放假设**，
原始发布时间与修订 vintage 未核验。它验证真实数据路径，不能验证 historical PIT 或独立前向。

独立脚本不调用生产训练/预测/切分函数，使用标量中位数与矩、岭回归正规方程和逐距离近邻
排序重建每个归档预测。测试另对 DSR/PSR 的偏度/峰度公式、PBO 组合及排序、BH/Bonferroni、
复利和回撤进行独立对照；反例覆盖 holdout 扰动、未来特征、标签越界、失败家族、遗漏重试、
来源篡改、错误状态流转和非同口径结论。

```bash
ruff check src tests examples/research_workflow.py examples/verify_research_workflow.py
ruff format --check src tests examples/research_workflow.py examples/verify_research_workflow.py
coverage run --branch --source=quant_agent -m pytest -q
coverage report --show-missing --fail-under=80
```

方法参考：[scikit-learn 的训练内预处理与泄漏说明](https://scikit-learn.org/stable/common_pitfalls.html)、
[arch SPA 官方定义](https://bashtage.github.io/arch/multiple-comparison/generated/arch.bootstrap.SPA.html)。
实现使用既有依赖和可核验公式，不依赖外部 LLM 或服务。
