# quant-agent ROADMAP

See also [quant-research-notes/roadmap/quant-agent.md](../../quant-research-notes/roadmap/quant-agent.md).

## v0.1 MVP (done)

- MultifactorAdapter, offline rules, LangGraph pipeline, `quant-review` CLI

## P1 (in progress via TaskSolver)

1. `check_backtest_stats()` — sharpe, max_drawdown thresholds
2. `check_ic_decay()` — horizon decay / overfitting signal
3. `compare_with_previous()` — diff vs prior timestamp run
4. multifactor CI + quant-agent CI
5. Push to `PureSaber/quant-agent`

## P2 (optional tail)

- Pitfalls injection into LLM prompts
- Structured findings in `review_manifest.json`

## P3 (deferred)

- SklearnAdapter, spread adapter, quant-lab indexing, quant-data-kit validate hook

## Explicitly deferred

- CrewAI / multi-agent debate
- Fetch or backtest inside agent
- Human-in-the-loop interrupts
- Embedding inside multifactor
