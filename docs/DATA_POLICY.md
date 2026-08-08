# LLM data policy

When `enable_llm: true` and `--llm` is passed:

1. Set `QUANT_AGENT_LLM_OK=1` in the environment to confirm external API use.
2. By default `llm.send_paths: false` — run directory paths are redacted from prompts.
3. Set `llm.send_paths: true` in `agent_review.yaml` only for trusted local debugging.

Never enable LLM in CI without mocking.
