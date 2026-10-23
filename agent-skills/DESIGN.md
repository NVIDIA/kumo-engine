# Design notes: `kumo-data-science`

Rationale for the skill's rules. This file is not packaged: the agent reads only the rules
themselves, written plainly in `skills/`. Traceability to the specification lives here, so the
skill never cites a document the agent cannot open.

## Behavioral referee

The intake, gates and artifact rules implement the **GA Spec** for the Kumo Coding Agent. Each
rule in the skill maps to a section of it:

| Rule in the skill | Where | GA Spec |
|---|---|---|
| The agent never chooses the problem, target, horizon, graph or tuning; refuses and names the minimum inputs | router *Does NOT*, Node 1b, `bad_request.md` §5 | §5 |
| No joins, cleaning, aggregation, label creation or other data transformation | router *Does NOT*, `preflight.md` §6, §8 | §6 *graph* b4 |
| No searching, ranking or curating tables by business relevance; enumerate only inside `source_scope` | router *Does NOT*, `preflight.md` §1, §8, adapter preflight SQL | §6 *graph* b1 |
| No retrying, resuming or cross-session state; the artifact calls the blocking `fit` once and prints its metrics; Step 3 is the only execution | router *Does NOT*, Step 2, Step 3, adapter open questions | §2 artifact item 6, §6 *artifact* b3 |
| RFM and zero-shot workflows are out of scope | router *Does NOT* | §1 |
| Supported tasks: temporal binary classification and link prediction | router *Task scope* | §1 |
| Ask only for missing, ambiguous or contradictory inputs; restate the resolved contract; never request secrets | router Step 1, `intake_contract.md` | §6 *request contract* |
| Required fields: `source_scope`, `prediction_entity`, `prediction_target`, `time_window` | `intake_contract.md` | §2, §3, §5, §6 *request contract* |
| Infer structure, never intent; no outcome definition is invented; a schema never fills a required field | router Node 1a, `intake_contract.md`, `preflight.md` §6 | §2 acceptance, §5, §6 *graph and task confirmation* |
| Inferred values are shown in a complete proposal and confirmed; user-supplied values win | router Node 1c, `intake_contract.md`, `preflight.md` §7 | §2 *Clarification and confirmation*, §6 *graph* b2 |
| Preflight is read-only SQL against the user's source | `preflight.md` | §1, §6 *graph and task confirmation* |
| Seeds that cannot be connected without transforming data → REJECT, generate nothing | router Node 1c, `preflight.md` §6 | §3 story acceptance, §6 *graph* b4 |
| No code before the task is confirmed | router Node 1d | §2 "waits for explicit PQL confirmation", §6 *graph* b3 |
| One session: one operation, one task, one graph, one artifact | router Step 2, `intake_contract.md`, `bad_request.md` §6 | §6 *request contract*, §6 *artifact* b1–b2 |
| Artifact contents: connector, tables, metadata, links, graph and PQL validation, SDK-default training table and plan, no tuning | router Step 2 | §2 artifact items 1–5 |
| Every confirmed input is preserved verbatim; no invented link, horizon or table | router Step 2 | §2, §3 story acceptance |
| Batch inference reuses the trained task unchanged and never reopens it | router Node 1a/1d, `intake_contract.md`, `bad_request.md` §4 | §4 |
| Default is generate-only | router Step 3 | §1, §6 *artifact* b3 |
| A warning or error from Kumo about the data leads the run's report and goes to the user; data, types and the query are never changed to silence it | router Step 3, `troubleshooting_signals.md` *Who applies a fix*, §1a | §6 *graph* b4 |
| Every key the proposal uses is probed for repeats and nulls, not only rule-5 candidates | `preflight.md` §3 | §6 *graph* b2 |

## Deliberate deviations from the GA Spec

Decided 2026-09-22 (the same-notebook predict on 2026-09-23). The skill states these as plain
rules; this is why they exist.

| | GA Spec | Skill | Why |
|---|---|---|---|
| Task scope | temporal binary classification + link prediction | also regression, ranking, static link prediction, embeddings (router *Task scope*) | product decision; the eval set covers them |
| Execution | the agent never runs the artifact | runs it on explicit request only (router Step 3) | keeps the CoCo P1 path available |
| Predict in notebook | one submission call, no execution | the predict call is part of the artifact (router Step 2) | user decision 2026-09-23 |
| Snowflake ML Jobs | no scheduling | the artifact may run as an ML Job on request; a recurring run is proposed and created only on explicit confirmation (`hosts/snowflake-coco/ml_job.md`) | spec feature 7 (2026-10-13). How Kumo reaches the job container is pending; the skill names it only as host values (`adapter.md` §10) |

Accommodation, not deviation: the spec has the user select the table mode. Nobody does, so it
defaults to `all-connected` and is confirmed at Node 1c.
