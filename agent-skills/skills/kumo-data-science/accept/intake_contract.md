# Input Contract — What a request must contain

The contract is keyed by **operation**. Decide the operation first, then apply that operation's
table. The tables below are the complete contract: a field that is not in them is not required.

Three kinds of field exist, and the distinction is the whole point of this file:

| Kind | Who supplies it | If absent |
|---|---|---|
| **Required** | the user | ASK BACK — ask for exactly these, nothing else |
| **Inferred, then confirmed** | the agent, by read-only preflight (`accept/preflight.md`) | never asked; shown as `[inferred]` and confirmed at the graph gate — except a key that preflight could only fail to disqualify, which is shown `[inferred, unverified]` and cannot be used until confirmed |
| **Optional** | the user, if they want | never asked; defaults apply; if supplied it must be preserved verbatim |

**Infer structure, never infer intent.** Column names, keys, time columns and links are structure
— measure them. Entity, outcome, positive-label condition, interaction, item, window and
destination are intent — the user's, and only theirs. No outcome definition is ever invented.

A preflight result or schema listing **never fills a required field.** Seeing a `sales_events`
table does not tell you the positive interaction is "a sale"; seeing a `target_label` column does
not tell you the positive-label condition. If the user did not state it, it is missing — ask,
even when the schema makes the answer look obvious. Preflight may resolve *which table holds*
what the user named; it may not supply what they did not name.

**The quote test — is a required field present?** A required field is present only if you can
quote the user's own words for it, from any turn, in any vocabulary: "each customer", "a
cancelled booking", "at least one fraud event", "next 30 days". A table name, a column name, or "the only
entity in the schema" is not a quote — the field is missing, and the reply is an ask-back, not a
graph proposal.

**Named intent, inferred mapping — never ask the user to map their words onto columns.** Once the
user has named a field, turning their words into a physical table, column, or value is structure.
Preflight infers it — "a cancelled booking" → `bookings` where `status` marks a cancellation —
and it is shown as `[inferred]` in the proposal for the user to confirm or correct. Do
not ask which column or value implements something the user already named, and do not ask them to
re-define it more precisely. Inferred values are fine as long as the complete proposal is shown
and confirmed; questions are only for missing, ambiguous, or contradictory inputs.

The two rules are one rule seen from both sides: the **user's words** decide *whether* a field is
present; the **data** decides *how* a present field is implemented.

---

## Operation

`train` or `batch_inference`. Read it from the request: "train / build a model / notebook that
predicts" is `train`; "use my trained model / score / run inference with job X" is
`batch_inference`. One session covers one operation, one task, one graph, one artifact.
Ask only if the request genuinely reads both ways.

---

## `train` — required fields (ask if missing)

| Field | What it must include |
|---|---|
| **`source_scope`** | A connector name, **or** a Snowflake database + schema (warehouse optional). This is where tables are enumerated. Credentials are never requested in chat. |
| **`prediction_entity`** | What receives one prediction — "each account", "each customer". A physical table/key is welcome but not required; preflight resolves it. |
| **`prediction_target`** | The observable outcome (classification/regression) or the positive interaction (link prediction), **stated by the user**. For classification this includes the positive-label condition — "at least one fraud event"; "predict fraud" alone leaves it missing. For link prediction and ranking it includes the **item** being recommended and the **interaction** that counts as positive — "spare-part families the customer ordered"; "recommend to customers" alone leaves both missing. |
| **`time_window`** | Tri-state. `specified` ("next 30 days") → pass. `not_critical` → the task is static (no temporal component: static link prediction, static regression/classification) → pass, do not ask. `missing` → the task is temporal and no window was given → ASK BACK. Decide static vs. temporal with the test below. |

**Static or temporal — decide from what is being predicted, never from the data.** Almost every
table has a date column (`booked_date`, `event_at`); its presence says nothing about the task.

- **Static** (`not_critical`, do not ask):
  - the target is a value the entity row itself carries — "the final price of each listing",
    "the category of each article". The entity is one record; nothing has to happen after it.
  - the user asks for **embeddings**, or for links **learned from observed activity** without a
    forecasting framing — "learn user and product embeddings from past clicks",
    "suggest tags for each article from the article-tag links we already have".
- **Temporal** (window required): the target is something that happens, or is aggregated, over a
  period after a point in time — "will churn", "will have a fraud event", "total spend",
  "what each customer will reorder", and recommenders, which by default predict future
  interactions ("a recommender for each member").

If the task is static, restate `Window: static` at the task gate and do not ask. A time column
may still be used as the graph's time column — that is structure, not a window.

A request that names the dataset ("the e-commerce fraud dataset", a schema name) has supplied
`source_scope` if that name resolves to a database and schema. Do not ask the user to restate a
location they have already given in an earlier turn.

## `train` — inferred, then confirmed (never ask)

| Field | How it is obtained |
|---|---|
| **seed tables** | The physical tables behind the entity and the target, found by preflight inside `source_scope` |
| **primary keys, time columns** | Preflight; each shown as `[inferred]` unless the user stated it (`[supplied]`). A key that preflight could only fail to disqualify is shown `[inferred, unverified]` and must be confirmed before the table can be an entity or an edge destination |
| **links** | Preflight; shown as `[inferred]` |
| **`table_mode`** | Default `all-connected` — every table reachable from the seeds. Shown as `[default]` in the graph proposal; the user may switch to `named-table` there. |

User-supplied values always win over inferred ones.
The complete proposal — every table, key, time column, link, and the table mode — is shown and
confirmed **before** any PQL is written.

## `train` — optional (never ask; preserve if given)

| Field | Default |
|---|---|
| **`output_location`** | none — training yields a notebook; when given, the same notebook also predicts into it (SKILL.md Step 2) |
| **artifact format** | one notebook (user may ask for a script) |
| **embeddings** | not included; add only when the user explicitly asks (see below) |
| **metrics** | by task type, see `SKILL.md` |

If the user supplies an `output_location`, carry it verbatim into the task interpretation gate
and the generated artifact — the notebook's predict call writes there. If none is supplied,
predictions stay local and the notebook shows how to add a destination. **The table is written with exactly the name passed** — no suffix,
no case change (`hosts/snowflake-coco/adapter.md` §7, `kumoai-reference/references/kumo_finetuning_sdk.md`
§9.1). Older SDK builds appended `_predictions`; the current one does not.

---

## `batch_inference` — required fields (ask if missing)

Two fields are the user's to give. Everything else is reused from the model, resolved from what
the user already said, or defaulted — and shown at the gates for confirmation.

| Field | What it must include |
|---|---|
| **`model_address`** | The model the user names, in their words: a model name or ID ("trainingjob-benchmark-001", "the benchmark model"), a registry location with a version, or a retained local bundle directory |
| **`output_location`** | Where scores are written: a Snowflake table. A connector plus a table name ("connector `benchmark_sf`, table `MODEL_SCORES`") is complete, because the connector supplies the database and schema. A fully qualified `DB.SCHEMA.TABLE` is also complete. |

Both pass the quote test through earlier turns too. "Run the benchmark model again and put the
preds in the scores table", after a turn that named the model and the table, supplies both.

## `batch_inference` — resolved, defaulted, or inferred (never ask)

| Field | How it is obtained | Shown as |
|---|---|---|
| **task** — entity, outcome, window, task type | Reused from the model unchanged. Never reopened. | restated at 1d |
| **`source_scope`** | In order: a source the user named → the connector the user named for this request (one Snowflake connector reads the source and writes the scores unless the user separates them) → the source scope recorded with the model at publication. Ask only if none of the three exists. | `[supplied]` / `[inferred]` at 1c |
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

## `batch_inference` — invalid values (ask back, do not substitute)

| Condition | Response |
|---|---|
| `model_address` does not resolve in the registry or as a bundle directory, or the user calls it rejected/invalid | Name it as unusable and ask for a corrected model identifier. Do not guess a similarly named model. |
| `output_location` is not a Snowflake table (an `s3://`, `gs://` or file path) | Name it as unusable here, since scores are written to Snowflake tables, and ask for a Snowflake destination. Do not silently swap in another one. |

When one field is invalid and the other is valid, ask only about the invalid one. When one is
missing and the other is present, ask only for the missing one. Do not also list the resolved
fields as questions.

---

## Embeddings

Default to predictions alone. Add embeddings **only when the user explicitly asks for them** —
never ask back for this, and never infer it from the task being recommendation-shaped.

Recognise: "we need shopper and item embeddings", "learn customer and product embeddings from
completed orders", "export the entity and target-item embeddings from that model".

When embeddings are requested, say so explicitly in the task interpretation gate — the user must
see that embeddings are part of the output. Carry it into codegen as
`PredictionOutputConfig(artifact_type=...)`; see `kumoai-reference/references/kumo_finetuning_sdk.md` §9.

---

## Canonical good example (`train`)

> "Create a notebook that trains a Kumo model for our fraud use case. Data is in
> `USER_DATA.E_COMMERCE`. Entity: each customer. Outcome: at least one fraud event. Window: next 30 days."

What makes it good:
- `source_scope`: `USER_DATA.E_COMMERCE` ✓
- `prediction_entity`: each customer ✓
- `prediction_target` with positive-label condition: at least one fraud event ✓
- `time_window`: `specified`, 30 days ✓
- No columns, keys or joins described — **correct**; preflight infers them and the user confirms.

## Canonical bad example (`train`)

> "Use my sales data to figure out which customers are risky."

What is missing:
- `source_scope`: "sales data" does not resolve to a database and schema ✗
- `prediction_target`: "risky" is not an observable outcome ✗
- `time_window`: `missing` for what reads as a temporal question ✗

Not missing, and not to be asked for: schema, join keys, output table.

---

## What to do when a required field is missing

Ask for **exactly** the missing required fields, once, in one turn. Do not ask for structure,
table mode, or an output table on a training request. On a `batch_inference` request, do not ask
for the source schema, population, or anchor date (see the *resolved, defaulted, or inferred*
table). Do not invent values. Ask in the user's
vocabulary ("what should be recommended to each customer?"), and do not attach questions
about columns or values for the fields that *are* present — those are confirmed later, as
`[inferred]`. Example:

> "To proceed I need:
> 1. **Where the data lives** — a Snowflake database and schema (or a connector name).
> 2. **The prediction window** — e.g. 'in the next 30 days'."

Ask only for missing, ambiguous, or contradictory inputs, restate the resolved contract, and never
request secrets in chat.
