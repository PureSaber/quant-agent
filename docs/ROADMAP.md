# quant-agent ROADMAP

状态核对日期：2026-10-07。平台路线见[quant-research-notes](https://github.com/PureSaber/quant-research-notes/blob/main/roadmap.md)，研究助手的使用边界见[research-assistant.md](research-assistant.md)。

## v0.1 MVP (done)

- MultifactorAdapter, offline rules, LangGraph pipeline, `quant-review` CLI

## P1 (implemented)

1. `check_backtest_stats()` — sharpe, max_drawdown thresholds
2. `check_ic_decay()` — horizon decay / overfitting signal
3. `compare_with_previous()` — diff vs prior timestamp run
4. multifactor CI + quant-agent CI
5. Push to `PureSaber/quant-agent`

## P2 (implemented)

- Pitfalls injection into LLM prompts
- Structured findings in `review_manifest.json`

## P3 (partly implemented)

- 已实现`FuturesSpreadAdapter`，消费期货研究产物。
- 已实现`standard/v2`原生校验及研究历史索引、可复核引用和失败记录查询；详见[README](../README.md)。这是只读消费，不代表Agent可以自行运行研究或重写索引事实。
- `SklearnAdapter`和额外的QDK独立validate hook仍暂缓；现有QLab产物校验不可绕过。

上述完成项以当前源码、对应规则/适配器测试和CI为依据，不代表真实市场数据GA或策略盈利认证。历史TaskSolver推进状态不再作为当前开发清单。

## Explicitly deferred

- CrewAI / multi-agent debate
- Fetch or backtest inside agent
- Human-in-the-loop interrupts
- Embedding inside multifactor
