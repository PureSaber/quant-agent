# quant-agent

LangGraph review layer for the PureSaber quant stack. It reads completed run outputs from sibling repos (starting with `a-share-multifactor`), applies deterministic QA rules, optionally calls an LLM for interpretation, and writes experiment notes plus a `review_manifest.json` back into the run directory.

## What it does (and does not do)

| Does | Does not |
|------|----------|
| Post-backtest review of IC / stats / IC decay CSVs | Fetch market data |
| Validate the immutable standard run contract before review | Repair mutated run artifacts |
| Rule-based QA (NaN factors, weak IC, backtest stats, IC decay, run diff) | Run backtests or replace `quant-data-kit` |
| Optional LLM explain + skeptic nodes | Embed inside multifactor |
| Write markdown to `experiment-log/` | Require API keys in CI (`--offline`) |
| Emit `review_manifest.json` for quant-lab scanning | |

## Install

```bash
cd quant-agent
pip install -e ".[dev]"
# optional LLM:
pip install -e ".[llm]"
```

## CLI

```bash
# offline review (default, no API key)
quant-review run --project multifactor --run-dir ../a-share-multifactor/outputs/four_factors/latest

# auto-detect project from run dir
quant-review detect ../a-share-multifactor/outputs/four_factors/latest

# enable LLM (needs OPENAI_API_KEY)
quant-review run --project multifactor --run-dir PATH --llm
```

Exit code `2` when deterministic rules report an `error` severity finding.

## Graph

```
load → rules → explain → skeptic → write_report
```

- **load**: adapter validates `standard/run_manifest.json`, then reads project-specific QA files
- **rules**: Python checks (IC quality, alt-data NaN, PIT config)
- **explain / skeptic**: LLM when `--llm`, else offline templates
- **write**: `experiment-log/review_*.md` + `review_manifest.json`

## Config

`configs/agent_review.yaml` — thresholds, rule toggles, LLM model.

## Adding adapters

Implement `RunAdapter` in `src/quant_agent/adapters/` and register in `get_adapter()`. Planned: `sklearn-stock-trend`, futures spread.

## Tests

```bash
pytest
ruff check src tests
```
