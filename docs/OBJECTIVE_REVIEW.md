# Read-only ex-ante objective review (20)

```sh
quant-objective-review --registry research-family.db --study-id momentum-a
```

The command opens SQLite read-only, checks the registered definition hash, resolves candidate
results from their registered artifact roots, verifies result/artifact hashes and recomputes
acceptance against the original objective. Citations include registry/study/attempt identifiers
and definition/result hashes. It does not need an LLM or API key.

Return, risk, cost, execution, data and stop-condition failures remain separate. Failed/running
attempts are retained as insufficient evidence, not removed from research history. A study
without an objective is `not_preregistered`; the agent will not retrofit a goal after seeing
returns. It never changes objectives, relaxes risk controls or authorizes investment/trading.

The objective schema and units are documented in quant-lab `docs/RESEARCH_INTEGRITY_11_20.md`.
Hashes protect local evidence consistency, not authenticity of untrusted economic observations.
