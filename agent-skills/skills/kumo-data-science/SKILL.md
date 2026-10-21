---
name: kumo-data-science
description: "Use for any work with KumoAI predictive models on relational data in a warehouse, an object store or local files: train a model, score data with a trained one, or answer questions about an existing model or run — explain its predictions, diagnose its training, publish, reload or retain it, train or score on custom labels or rows, or tune it. Covers churn, propensity, fraud, recommendation, forecasting and link prediction; feature attribution and column cohorts; metrics, trials and learning curves; registry publishing and artifact retention; caller-supplied labels, weights and splits; encoders, run mode and experiments. For a new model it checks that the request names the data (a schema, dataset, folder of files, connector, or semantic view), the outcome and the time window, confirms the inferred tables and the task, then generates one notebook or script; it runs the job only when asked, in the notebook or as a host job — a Snowflake ML Job on a compute pool, a Databricks job on a GPU cluster. Triggers: kumo, predict, churn, recommend, forecast, fraud, train a model, run inference, explain a prediction, feature importance, model metrics, publish or load a model, tune a model, semantic view, metric view, ML Job, compute pool, Databricks job."
---

# Data Science Agent — Router Skill

## Build marker

**`SKILL-BUILD: source`**

Quote this string verbatim whenever asked which build of the skill is loaded. If you cannot read
it from this file, you have not loaded the skill — say so rather than guessing.

Targets Kumo bundle `20261007+2e1eb12` (import `kumo.engine`; kumo-ai/kumo#31382).
One SDK, no variants.

## Routing

Some requests are not new-training intake. Route these to their own skill, which has its own
build and artifact checks:

| Request | Skill |
|---|---|
| Attribution or column cohorts for an existing run | `kumo-model-explanations/SKILL.md` |
| Metrics, trial comparisons, learning curves, recorded progress | `kumo-training-diagnostics/SKILL.md` |
| Publish, reload, retain or retrieve a model or its artifacts | `kumo-model-persistence/SKILL.md` |
| Caller-supplied labels, weights, splits, or exact scoring rows | `kumo-custom-populations/SKILL.md` |
| Encoders, run mode, bounded experiments, model-plan options | `kumo-model-tuning/SKILL.md` |

None of these require new-training intake, and none authorizes another fit. Preserve intent the
user has already given — task, target, run, model version, source, destination — and ask only
for what the selected workflow is missing. If the host has not provided the referenced skill,
report that limitation rather than inventing its workflow. **New training requests still follow
the gates below**, unchanged.

A request to run the artifact as a **job** — a Snowflake ML Job on a compute pool, a Databricks
job on a cluster, unattended, or on a schedule — or a question about a job already submitted (status, logs, cost) also reads the
host's job file, when the host adapter names one. It adds settings to the gates below and changes
where Step 3 runs; it never replaces the gates. If the adapter names no job file, say that this
host does not run jobs and offer the notebook. A request that does not mention a job stays in the
notebook.

## Purpose
Take a plain-language ML request, get it to a **confirmed** task and graph through a multi-turn
intake, and generate **one** runnable KumoAI artifact — a notebook for training, a Python script
for batch inference. The agent asks only for what is missing, infers structure rather than asking
for it, confirms before writing code, and does not run jobs unless told to.

## What this skill does
1. **Accept** — a subDAG of checks; each node emits ACCEPT / ASK BACK / REJECT
2. **Preflight & graph gate** — infer tables, keys, time columns, links; user confirms
3. **Task gate** — restate entity / outcome / window / task type; user confirms
4. **Generate** — one notebook that **trains and then predicts** in the same process, with
   exactly one blocking `fit` call; the predict call is part of the same artifact
5. **Run** — *only on explicit request*

**The P1 deliverable is train → predict in one notebook.** Batch inference from a *saved* model
(`batch_inference`) is also supported: `Trainer.load` and `Trainer.load_from_connector` restore a
trainer, and `predict` runs against an explicitly rebuilt graph snapshot (see Step 2).

## What this skill does NOT do
- Choose the predictive problem, target, horizon, graph, or tuning for the user — refuse and name
  the minimum inputs needed
- Join, clean, deduplicate, aggregate, create labels, or transform data **as model input**.
  Diagnostic scalar aggregates during preflight — reported as proposal evidence, never written and
  never fed to training — are not this; they exist so a candidate key can be disqualified before
  it silently discards rows
- Search for, rank, or curate tables by business relevance
- Retry or resume jobs, or keep state across sessions. The one execution the agent performs is
  Step 3, and only when the user asks.
- Schedule anything on its own initiative (a recurring run is proposed and created only on explicit
  confirmation, and only on a host whose job file covers it)
- RFM / zero-shot workflows
- Audit the user's schema for leakage at intake beyond the mandatory availability guards in
  Node 1c. The codegen guard in `kumoai-reference/SKILL.md` (the target never appears as a
  feature) still applies
- Turn a vague question into a complete request beyond the ask-back, or repair runtime job failures

## Task scope
**Temporal binary classification**, **link prediction**, regression, ranking, static link
prediction, and embeddings as an output. Treat them all with the same intake; `time_window` is `not_critical` for
static shapes.

---

## Step 1 — Accept subDAG

Accept is a subDAG of check nodes run in a multi-turn chat. Each node produces one of:

- **ACCEPT** — proceed to the next node.
- **ASK BACK** — ask one targeted question for exactly what is missing; re-run on the next turn.
- **REJECT** — cannot be fixed by clarification. Stop with a clear explanation.

Ask **only** for missing, ambiguous, or contradictory inputs; never request secrets.
One ask-back turn lists every missing required field at once.

### Node 1a — Operation and required fields
Reference: `accept/intake_contract.md`

First decide the operation: `train` or `batch_inference`.

**`train` — required (ask if missing):**
- `source_scope` — where the tables live, in a form the host accepts (`accept/intake_contract.md`)
- `prediction_entity` — what gets one prediction
- `prediction_target` — the outcome or interaction, with the positive-label condition
- `time_window` — `specified` / `not_critical` (static task) / `missing` → ask only when `missing`.
  Static vs. temporal is decided by *what is predicted*, never by a date column in the data: a
  value the entity row carries ("final price of each listing") or embeddings / observed links are
  static; anything that happens or accumulates after a point in time is temporal
  (`accept/intake_contract.md`, *static or temporal*).

**`train` — never ask for:** column names, types, keys, joins, table mode (inferred at 1c);
output table (optional; preserve if given); metrics; embeddings (add only if asked).

**`batch_inference`:** its required and resolved fields, and what changes at Nodes 1c, 1d and
Step 2, are in `accept/batch_inference.md`. Read it when the operation is `batch_inference`; a
training request never needs it.

**Never fill a required field from the schema.** A preflight result (or a schema listing handed
to you) can tell you which tables exist, not what the user wants predicted. If the user has not
named the outcome, the positive-label condition, the item, the interaction, or the window, it is
missing — ASK BACK, even when the tables make a guess look obvious.

**Quote test:** a required field is present only if you can quote the user's words for it; a
table or column name is not a quote. **Conversely, never ask the user which column or value
implements a field they already named** — infer the mapping and show it `[inferred]` at the gate
(`accept/intake_contract.md`).

| Condition | Transition |
|---|---|
| Any required field missing | ASK BACK — list exactly the missing fields |
| `source_scope` given but does not resolve (location or semantic view not found) | ASK BACK — ask for a valid source; do not guess a similar name |
| All required present | proceed to Node 1b |

### Node 1b — Atomic question check
Reference: `validate/atomic_question_check.md`

| Condition | Transition |
|---|---|
| Goal is vague but potentially clarifiable | ASK BACK — ask the user to define the target concretely |
| Multiple goals bundled that cannot be split | REJECT — explain and ask for separate requests |
| "Figure out what to predict / pick the best target / tune and deploy" | REJECT — the agent does not choose the problem; list the minimum inputs for one supported request |
| Single, well-defined target | proceed to Node 1c |

### Node 1c — Preflight & graph gate
Reference: `accept/preflight.md`

First establish that you can read `source_scope` (`accept/preflight.md` §0). Then run read-only
preflight inside it.

Present the graph proposal — seeds, tables,
primary keys, time columns, links, `table_mode: all-connected [default]` — each line tagged
`[supplied]` / `[declared]` / `[inferred]` / `[inferred, unverified]` / `[default]`. A semantic
view is resolved by `accept/semantic_view.md`, read only for a semantic view. If the user asked for embeddings, say they are included. **Stop and wait for confirmation.**
User-supplied values take precedence over inferred ones.

**Mandatory column-availability pass.** Before presenting the graph, decide the selected columns
for every included table and add an `Excluded columns` section to the proposal.

- When a table's `time_column` marks the beginning of a row's validity or an interval, exclude any
  candidate feature whose value is determined only after that time: the interval's eventual
  endpoint, its realized duration, or a summary of the completed interval. At the row timestamp
  those values are future information. Decide by semantic role, never by matching column names.
- Separate the target column from columns derived from it; both rules apply at once.
  - The target column itself **always stays** in the positional column list. The query references
    it by name, so dropping it fails with a missing column.
  - **Exclude** every other column whose value is a function of the target — a margin, ratio,
    rate, share, rank or flag computed from it. Such a column reproduces the answer under a
    different name, and the dependency is often visible only in the naming, so treat a plausible
    derivation as a derivation and show it as an exclusion the user can overturn.
- Apply the active host adapter's runtime column constraints (`hosts/<host>/adapter.md`). A host
  constraint may exclude a column; it must never invent a replacement semantic type. Exclude only
  on evidence the adapter accepts as establishing the constrained property — a physical catalog
  type generally does not establish a semantic one — or on a prior runtime-guard result. Where the
  first pass cannot establish it, say so rather than guessing, and if the host's runtime guard
  later resolves it, return to this node with a revised proposal rather than proceeding.

Tag each exclusion `[inferred]`, give the reason, and let the user correct it at this same gate.
Carry the confirmed exclusion set into code generation: a table with any exclusion must be built
with an explicit positional column list, because omitting it silently restores every source
column.

**Assert what preflight could not find.** The proposal must state these explicitly rather than
leaving them unmentioned, because each one changes whether the task is expressible:

- a table with no **confirmed** primary key — say so. Until the user confirms one it is treated
  as keyless: it may be an edge source, but not an edge destination and not the `FOR EACH`
  entity. An `[inferred, unverified]` candidate does not lift that restriction; confirming it
  does. Never promote a merely unique-looking column to primary key, and do not
  conclude from a failed heuristic that the table must be an event or fact table.
- a time-ranged target whose referenced table has no usable time column — say so, and say what
  would make the task expressible. A timestamp on another table does not repair the target.

| Condition | Transition |
|---|---|
| You cannot read `source_scope` | ASK BACK — say what you checked; offer access or the structure in the user's own words (`accept/preflight.md` §0) |
| Seeds cannot be connected without transforming data | REJECT — explain; generate nothing |
| A semantic view fails a check in `accept/semantic_view.md` | ASK BACK — as that file says; do not pick a schema, drop tables, or substitute a link |
| User corrects a line | re-present the proposal with the correction `[supplied]` |
| Confirmed proposal still lacks metadata the task requires (entity key, target time column) | remain at Node 1c — explain the correction needed; generate nothing |
| User confirms | proceed to Node 1d |

### Node 1d — Task interpretation gate
Restate, in this shape, then **stop and wait for explicit confirmation** — no code is
generated before the task is confirmed:

```
- Entity: one prediction per <table>.<key>
- Outcome: <positive-label condition> / <interaction>
- Window: <0 to N days after the prediction anchor> | static
- Task type: <temporal binary classification | link prediction | ...>
- Output: predictions written to <destination, verbatim> in the same notebook | notebook only, no predictions written
- Scoring population (static tasks): rows whose <target> is NULL | all <entity> rows, staged at <DB.SCHEMA.TABLE>
- Embeddings: <included, if explicitly requested>
```

**Static tasks: confirm the scoring population.** A static query scores only the rows whose
target is NULL (`kumoai-reference/references/kumo_finetuning_sdk.md` §7.1). If the user wants rows that
already have a target scored (for example "every order"), the notebook stages an explicit
population table. That is a write, so name it here: next to the confirmed destination by default
(`<DESTINATION>_POPULATION`), or ask for a schema when there is no destination. Never stage it in
the source schema.

| Condition | Transition |
|---|---|
| User confirms | ACCEPT — proceed to Step 2 |
| User changes intent | back to Node 1a with the new facts |

**Do not emit any Kumo SDK code before this ACCEPT.** In the eval harness and in production alike,
a reply that reaches Node 1d ends with the restatement and the confirmation question.

### Default metrics by task type
The generated notebook does not set metrics; the SDK's default model plan supplies them. Use
this table to tell the user what the model will be evaluated and tuned on.

| Task type | Reported metrics | Tuned on |
|---|---|---|
| Binary classification | `auroc`, `ap`, `auprc` | `auroc` |
| Multiclass classification | `acc` | `acc` |
| Multilabel classification | `ap`, `auroc`, `auprc`, each `_macro` / `_micro` / `_per_label` | `ap_macro` |
| Regression, forecasting | `mae`, `mse`, `rmse` | `mae` |
| Ranking, link prediction | `map`, `precision`, `recall`, `f1`, each `@1` / `@10` / `@100` | `map@10` |
| Link prediction with a training-table weight column | `weighted_ndcg`, `weighted_recall`, `precision`, each `@1` / `@10` / `@100` | `weighted_ndcg@10` |

Source: shipped package (`kumo_core.model_plan`)

---

## Step 2 — Generate via KumoAI

Read the host's `runtime.md` (`hosts/<host>/runtime.md`) before generating.

After ACCEPT, load `kumoai-reference/SKILL.md` and hand off the confirmed payload. Produce
**one** artifact.

### `train` → one notebook that fits, then predicts

1. Connector from environment credentials; confirmed tables, metadata, links; graph and PQL
   validation; SDK-default training table and model plan, no tuning.
2. **Exactly one** training call — `trainer.fit(...)`. It blocks until training finishes and
   returns a `TrainingResult`; print its `metrics`.
3. **Predict in the same notebook** — same `Trainer`, same process, immediately after
   `fit`:
   - `output_location` confirmed at Node 1d → the notebook includes the `predict` call with a
     `PredictionOutputConfig` pointing at that destination, written **verbatim**.
   - no destination → predictions stay local; the notebook ends with a short markdown cell
     describing the one line to add a destination later. That example is markdown text, never a
     code cell: a notebook with no confirmed destination contains no code that writes. Inside
     the markdown, show the code as text (a `text` fence), never as a `python` code block —
     one that would run is a code cell that writes.
   The predict call is part of the artifact, not a separate job the agent manages. The agent
   still executes nothing (Step 3 is opt-in).

The user may ask for a script instead of a notebook.

### `batch_inference`

See `accept/batch_inference.md`, *Step 2*.

The artifact must preserve every confirmed input verbatim — entity, outcome, window, table mode,
metadata, links, destination. No invented link, horizon, or table.

---

## Step 3 — Run **(only on explicit user request)**

Default is generate-only. Run only when the user says to run it,
in the same or a later turn. Then, per `kumoai-reference/references/kumo_finetuning_sdk.md`:
- Run the notebook. `fit` blocks until training finishes; there is no Kumo job to poll. When the
  user asked for a job, running means submitting it and waiting for its status as the host's job
  file describes.
- Report the outcome: the training metrics, or the exception and the cell that raised it
- Let the notebook's own predict step run; predictions land at the confirmed destination — **the
  name is written verbatim**, no suffix, no case change; report it in the destination's own
  form (`DB.SCHEMA.TABLE`, or a path)
- Include embeddings if they were confirmed at Node 1d

---

## Host adapter
This package carries exactly one host: the single directory under `hosts/`. Everything specific to
where you run and where the data lives is there — never assume another host's rules.

- `hosts/<host>/adapter.md` — read at intake: how to check access to the source, the source and
  destination forms the host accepts, how to run preflight, and runtime column constraints
- `hosts/<host>/runtime.md` — read at Step 2: how the generated code connects, authenticates,
  writes and verifies

---

## Extension points
- **Host expansion:** add a directory under `hosts/` with its own `adapter.md`. The portable core
  (this file + `accept/` + `validate/` + `kumoai-reference/`) does not change.
