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
python -m pip install -r requirements.lock
python -m pip check
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check
```

`requirements.lock` is the audited Python3.10-3.12 environment. It includes the base runtime,
the optional LLM runtime, development tools, and editable-build requirements. Do not install extras
on top of this environment because that would resolve a second, unaudited dependency graph.

## CLI

```bash
# offline review (default, no API key)
quant-review run --project multifactor --run-dir <validated-run-dir>

# auto-detect project from run dir
quant-review detect <validated-run-dir>

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

## M6 artifact and dependency governance

This repository declares the `orchestration` layer and consumes `standard/v2@2.0.0` through
`[tool.quant-workspace]`. The declared external lock is `requirements.lock`; a release audit records
its SHA-256. Internal runtime dependencies use published annotated tags in both project metadata
and the lock; their verified peeled commits are:

- `quant-lab v0.3.1` at commit `27489d270e132adbec1bced93eb2ae84ad5e1a9b`;
- `quant-workspace v0.2.0` at commit `1a9134ac329704060a3ae96cc81e31db481a938f`.

For a run containing `standard/v2`, the agent first calls the version-independent `quant-lab`
validator. Only after all manifest, hash, schema, lineage, profile, and file-set checks succeed does
it read the manifest-listed `config.json` and `metrics.json`. Root-level project CSV files and an
external `--config` cannot override a v2 run. If v2 exists but is invalid, review fails; it never
downgrades to v1. When v2 is absent, the immutable v1 reader and historical project layouts remain
supported permanently.

V2 producers may put agent inputs in the validated `metrics.json` object as `ic_summary`,
`backtest_stats`, and `ic_decay` arrays of objects. Recognized scalar performance metrics are also
projected from the top level. Factor or strategy names come from validated config first and the
manifest strategy IDs otherwise.

## Rebuild the dependency lock

Review changes to `pyproject.toml` and `requirements-constraints.txt` first. Run the rebuild under
Python3.10, the lowest supported interpreter, so its conditional `tomli` runtime remains explicit;
then regenerate with the pinned resolver used for this release:

```bash
python -m pip install "pip-tools==7.6.1"
pip-compile --extra dev --extra llm --build-deps-for editable --allow-unsafe --strip-extras \
  --resolver backtracking --index-url https://pypi.org/simple \
  --constraint requirements-constraints.txt \
  --output-file requirements.lock pyproject.toml
```

Install the result in clean Python3.10,3.11, and3.12 environments using the commands in
`Install`. CI runs `pip check` both before and after the no-dependency/no-build-isolation editable
install, then runs Ruff check/format, the complete test suite with at least80% branch coverage, and
the artifact-consumption core with at least90% pure branch coverage. Update the declaration,
constraints, and lock as one review unit; never hand-edit an isolated transitive pin.

## V1 migration and rollback

Migration is producer-owned: publish a new immutable `standard/v2` directory beside v1, including
its complete manifest, manifest checksum, config, metrics, Parquet artifacts, and lineage. Do not
modify v1 and do not copy unvalidated private CSVs into the v2 review path. Validate the new run with
`quant-lab v0.3.1` before invoking `quant-review`; a failed migration remains a failed v2 run and
must be repaired by publishing a new run rather than by forcing v1 fallback.

Rollback this governance change with a Git revert that restores `pyproject.toml`, the constraints,
and `requirements.lock` together. Existing runs and v1 data require no rewrite. Never move, delete,
or recreate historical tags; if a dependency resolution must change, publish a new version and a
new lock hash.

## Config

`configs/agent_review.yaml` — thresholds, rule toggles, LLM model.

## Adding adapters

Implement `RunAdapter` in `src/quant_agent/adapters/` and register in `get_adapter()`. Planned: `sklearn-stock-trend`, futures spread.

## Tests

```bash
ruff check src tests
ruff format --check src tests
coverage run --branch --source=quant_agent -m pytest -q
coverage report --show-missing --fail-under=80
```
