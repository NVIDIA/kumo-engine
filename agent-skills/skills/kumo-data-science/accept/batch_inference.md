# Batch inference — scoring with a saved model

Read this file only when the operation is `batch_inference`: the user names a trained model and
wants it to score data. A training request never needs it. It adds to `SKILL.md` and
`intake_contract.md`; everything they say still applies unless this file overrides it.

## Node 1a — fields

**`batch_inference` — required (ask if missing):**
- `model_address` — the model the user names: a name or ID, a registry location with a version, or
  a retained local bundle directory
- `output_location` — a destination the active host can write (`hosts/<host>/adapter.md`,
  *Destinations*); a connector plus a table name is complete

**`batch_inference` — resolved, never asked** (`accept/intake_contract.md`):
- `source_scope`: a source the user named, else the connector named for this request, else the
  source recorded with the model. The bundle restores trainer state, **not** the graph, so the
  graph is rebuilt from source at 1c.
- `scoring_population`: defaults to every entity of the trained entity table, anchored at the
  latest timestamp in the rebuilt snapshot, tagged `[default]`. A subset or date the user names
  replaces it.
- The target, positive-label condition, window and task type are fixed by the trained model and
  must not be reopened.

A request with a model and a destination goes to the gates. It does not get an ask-back.

### `batch_inference` — required fields (ask if missing)

Two fields are the user's to give. Everything else is reused from the model, resolved from what
the user already said, or defaulted — and shown at the gates for confirmation.

| Field | What it must include |
|---|---|
| **`model_address`** | The model the user names, in their words: a model name or ID ("trainingjob-benchmark-001", "the benchmark model"), a registry location with a version, or a retained local bundle directory |
| **`output_location`** | Where scores are written, in a form the active host can write (`hosts/<host>/adapter.md`, *Destinations*). A connector plus a table name ("connector `benchmark_sf`, table `MODEL_SCORES`") is complete, because the connector supplies the database and schema. A fully qualified table (`DB.SCHEMA.TABLE`), or a path on a host that writes files, is also complete. |

Both pass the quote test through earlier turns too. "Run the benchmark model again and put the
preds in the scores table", after a turn that named the model and the table, supplies both.

### `batch_inference` — resolved, defaulted, or inferred (never ask)

| Field | How it is obtained | Shown as |
|---|---|---|
| **task** — entity, outcome, window, task type | Reused from the model unchanged. Never reopened. | restated at 1d |
| **`source_scope`** | In order: a source the user named → the connector the user named for this request (one connector reads the source and writes the scores unless the user separates them) → the source scope recorded with the model at publication. Ask only if none of the three exists. | `[supplied]` / `[inferred]` at 1c |
| **graph** | Rebuilt from the model's recorded graph definition inside `source_scope`, then confirmed | `[inferred]` at 1c |
| **`scoring_population`** | Default: every entity of the trained entity table, anchored at the latest timestamp in the rebuilt snapshot ("current predictions"). A static task has no anchor. If the user names a subset or an anchor date, use theirs. | `[default]` / `[supplied]` at 1d |
| **model version** | The version the user named. If the address resolves to more than one version and none was named, ask which one. Never pick the latest yourself. | `[supplied]` at 1d |
| **embeddings** | Only when the user explicitly asks ("export the entity and target-item embeddings") | restated at 1d |

**The saved bundle does not carry the graph.** It restores trainer state; `predict` takes an
explicit `snapshot`, so the source data must be available and the graph rebuilt and confirmed at
Node 1c exactly as for `train`. "Reuse the model" needs reading precisely here: the *task* is
reused unchanged and must not be reopened. The *graph* is rebuilt, but from the recorded definition, not re-asked.

**When the request has a model and a destination, it goes to the gates. It does not get an
ask-back.** Do not ask for the source schema, the scoring population, the anchor date, or a fully
qualified output path when the rules above resolve them. Put each resolved value in the proposal,
tagged, so the user can correct it at confirmation.

### `batch_inference` — invalid values (ask back, do not substitute)

| Condition | Response |
|---|---|
| `model_address` does not resolve in the registry or as a bundle directory, or the user calls it rejected/invalid | Name it as unusable and ask for a corrected model identifier. Do not guess a similarly named model. |
| `output_location` is a kind the host cannot write (the adapter's *Destinations* lists what it can) | Name it as unusable here, say what the host writes instead, and ask for such a destination. Do not silently swap in another one. |

When one field is invalid and the other is valid, ask only about the invalid one. When one is
missing and the other is present, ask only for the missing one. Do not also list the resolved
fields as questions.

| Condition | Transition |
|---|---|
| `batch_inference`: `model_address` does not resolve, or the user calls it rejected | ASK BACK — name it as unusable, ask for a corrected identifier; do not guess |
| `batch_inference`: `output_location` is a kind the host cannot write | ASK BACK — name it as unusable, ask for a destination the host writes; do not substitute |

## Node 1c — rebuilt graph

For `batch_inference`, do not re-derive this set. The selected columns are the ones the model
was trained on: take them from the graph-building code and column metadata recorded with the
model (`kumo-model-persistence/references/reuse.md`), and show each table's list at this gate.
Re-inferring them can build the table with a different column list than the model saw, so
`predict` either fails on the mismatch or scores on different features. If no recorded column
selection is available, say so and remain at this node rather than substituting a fresh one.

This node runs for `batch_inference` too. A saved bundle does not carry graph storage or source
data, and `predict` requires an explicit `snapshot`, so the graph is rebuilt and confirmed the
same way. Confirm that the rebuilt structure matches what the model was trained on; a bundle
from an older SDK, or source tables whose shape has drifted, fails at load or predict rather
than scoring silently.

| Condition | Transition |
|---|---|
| `batch_inference` and no column selection is recorded with the model | remain at Node 1c — say the training-time selection is unavailable; generate nothing |

## Node 1d — restatement

For `batch_inference`, restate in this shape and stop for confirmation:

```
- Model: <model address, verbatim> — trained task reused unchanged
- Source: <source_scope> [supplied | inferred]
- Population: <all <entity> as of latest snapshot timestamp [default] | the user's subset/date [supplied]>
- Output: predictions written to <destination, verbatim>
- Embeddings: <included, if explicitly requested>
```

Without tools, where 1c cannot run, fold the source and graph lines into this one restatement.
Do not turn them into questions.

## Step 2

Follow `kumo-model-persistence/SKILL.md` for the load and predict code. Restore the trainer with
`Trainer.load` (retained bundle) or `Trainer.load_from_connector` (registry), materialize the
snapshot for the graph confirmed at 1c, build the prediction table for the confirmed scoring
population — `kumo-custom-populations/SKILL.md` when the user supplied exact rows — and call
`predict(prediction_table, snapshot=snapshot, output_config=...)`.

Do not refit. If the bundle cannot be loaded, or the rebuilt graph is incompatible with it,
report the failing stage and stop; do not substitute a fresh training run without being asked.

## Examples

### Model and destination given → gate

> "Load trained model `trainingjob-benchmark-001` and write current batch predictions through
> connector `benchmark_sf` to table `MODEL_SCORES`."

Node 1a: `model_address` ✓ and `output_location` ✓. A connector plus a table name is a complete
destination. Nothing is asked. The rest is resolved and goes straight to the gate:

```
- Model: trainingjob-benchmark-001 — trained task reused unchanged
- Source: benchmark_sf [inferred — the connector named for this request]
- Population: all entities of the trained entity table, as of the latest snapshot timestamp [default]
- Output: predictions written to MODEL_SCORES via benchmark_sf
Shall I generate the batch-inference notebook with this setup?
```

With "…export the entity and target-item embeddings" added, the gate also carries
`Embeddings: included`. It is still one confirmation, not an ask-back.

### Destination missing → ASK BACK

> "Use trained model `trainingjob-abc123` to score current accounts."

- `model_address` ✓ · `output_location` ✗
- Not asked: the source (recorded with the model), the population ("current accounts" means all
  accounts at the latest snapshot timestamp, the default), and the task. The task is never
  reopened.

> "Where should the scores be written? For example a connector plus a table name, or
> `DB.SCHEMA.TABLE`."

### Destination unusable → ASK BACK

On a host whose *Destinations* are Snowflake tables only:

> "Score with `trainingjob-abc123` and write predictions to `s3://tmp/scores`."

- `model_address` ✓ · `output_location` invalid: not a Snowflake table

> "`s3://tmp/scores` isn't a destination I can write to here, because scores go to a Snowflake
> table. Which Snowflake table should receive them?"

Only the invalid field is raised. Do not add questions about the source or the population.
