# Research federation and explicit review

`quant-research-workflow federate --manifest input.json` reads a
`quant.research-federation-input/v1` inventory. It does not execute research, merge
databases, perform fuzzy identity matching or grant trading authority. Studio supplies
its immutable project ID, exact historical recipe versions, Notebook attempts and
original evidence hashes. Failed, pending and incomplete attempts remain visible.

The input contains `research_id`, `question`, explicit `context`, `current_sources`
(absolute local paths and expected SHA256), and `attempts`. Each attempt declares
`id`, `pathway` (`notebook`, `native`, `lab`, `agent_ml`), `status`, `files`, optional
`comparison_scope` and diagnostics. These are producer assertions, not certification.
`--manifest -` accepts the same object on stdin for process-isolated consumers.

Optional `lab_bindings` entries contain a database and an exact `study_ids` list.
The existing read-only TrialRegistry adapter verifies registered definitions. The
result includes all planned candidates, all event rows, and each attempt's latest
state, including failed retries. It does not truncate by lexical search relevance.
Terminal result and artifact hashes are checked where the registry supplies them.
Missing or incompatible bindings are explicit diagnostics, not empty success.

## Reviewing a snapshot

Add `--review-directory PATH` to display the existing independent lifecycle journal.
Reading an absent journal does not create it. `--review review.json` records an
explicit interpretation using the existing ResearchLifecycle transition and CAS
head. For stdin consumers, `--review -` reads `manifest.review_request`.

The review request requires `expected_snapshot`, `expected_head`, `to_state`,
`verdict`, `attempt_ids`, `actor`, `reason`, and timezone-aware `recorded_at`.
Allowed verdicts: `supports`, `rejects`, `inconclusive`, `execution_failed`.
Hypothesis verdicts require completed, byte-verified attempts; a technical failure
cannot reject a hypothesis. Multiple attempts require identical explicit comparison
scopes; the service never infers financial comparability from matching field names.

The transition chain remains candidate → review → observe, with the existing pause,
retirement and reconsideration transitions. `observe` is a research state only.
Actor text is caller-declared, not authentication or multi-person approval.

Reviews store an immutable copy of the cited snapshot and are authoritative only
when referenced by a committed lifecycle event. A competing stale writer can leave
an unreferenced content-addressed review file, which is not displayed as a decision.
Original claims stay readable when source bytes change; they are marked for review.
New attempts, project revisions, source/dataset identities or registered events alter
the snapshot identity. This does not automatically retire a strategy or pause jobs.

## Boundaries and acceptance

Hash matching proves byte identity, not PIT correctness, profitability, statistical
significance, or completeness of unregistered external inputs. Local administrators
control paths; this API is not an arbitrary-file service for untrusted web clients.
External files are checked during collection and before writing a review; they are
not locked across the lifetime of all source systems. Later mutation remains visible
when rereading the review. A project save is exploratory unless a real Lab registration
is explicitly linked.

`record_review` rereads all referenced files, SQLite citations and linked Lab study
inventories. For other producers, a static manifest cannot discover newly created
unlisted files: regenerate the producer inventory before every review. Studio does
this on every GET/POST and checks the submitted snapshot identity. The generic CLI's
scope remains the explicitly supplied manifest, never the whole filesystem.

`tests/test_research_federation.py` covers actual Lab SQLite reads without writes,
unattempted candidates and failed retries, two CLI processes competing on one head,
changed evidence, duplicate identities, incompatible scopes and failure semantics.
