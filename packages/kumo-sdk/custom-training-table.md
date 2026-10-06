# Custom training tables

Modifying a generated training table — adding a weight column, features, or a
candidate list — and training on the result. Today this works only on the
cloud backend, and only by round-tripping the table through a registered
connector. This document proposes making it work on both backends, without
the round trip.

## Principle

**A training table is a value you derive, not a job you re-point.** Generation
produces rows; a modification produces *new* rows; training consumes whichever
you hand it. The backend decides how those rows reach the trainer — a staged
parquet in scratch on local, a staged object in the project's managed area on
cloud — and that decision never appears in user code.

The one concept the local backend is missing is a **source table that names a
local file**. That single gap is why custom training tables are unreachable
locally *and* why the only registered local reader is Snowflake. Closing it
serves both.

______________________________________________________________________

## What exists today

**Cloud, before this work.** `TrainingTable.export()` wrote the table to a
connector you own; you modified it in your own compute;
`TrainingTable.update(source_table, spec)` recorded a `CustomTrainingTable` on
the *Python object*; `Trainer.fit` forwarded it as
`TrainingJobRequest.custom_train_table` (`trainer/trainer.py:291,309`).

Two consequences were worth naming. The state was in-memory only, so a fresh
`TrainingTable(job_id)` in another process silently lost the modification. And
the modification had to be expressed as a connector URI even when the caller
was holding the DataFrame.

`update()` is deleted by this work; `TrainingTable.from_source_table( source_table, spec, pq=...)` replaces it and registers the table on the cloud
rather than on the object, which is what removes the first consequence.

**Local.** Generation, export, and custom training tables are wired; see
**Implementation, in order** for what each phase landed.

**`kumo_core`.** The capability is already there, and it is already
local-shaped:

```python
# kumo_core/training/table.py:136
@dataclass(frozen=True)
class PreparedCustomTrainingTable:
    parquet_path: str                 # a local path, not a connector
    weight_column: str | None = None
    time_column: str = "TIMESTAMP"
```

`TrainingWorkflowRequest.from_spec()` accepts `custom_training_table=`, and
`materialize_training_task` branches to `generate_custom_train_table`
(`workflow.py:467`), which calls
`diskgraph.build_train_table_artifacts_for_existing_table`. That builder
recomputes the manifest, split spec, and column stats *from the supplied
parquet* — it does not merely trust it.

So the engine's contract for a custom training table is a parquet file. The
SDK's contract is a registered connector in the right project. The round trip
is imposed by the SDK, not by the engine.

______________________________________________________________________

## Public API

### New wire type (kumo-api)

```python
LocalFileSourceTableType = Literal["LOCAL_FILE"]


@dataclass(frozen=True)
class LocalFileSourceTable:
    r"""A source table that is a file on the filesystem of the process."""
    path: str
    file_type: FileType = FileType.PARQUET
    data_source_type: LocalFileSourceTableType = LOCAL_FILE_SOURCE_TABLE_TYPE


SourceTableType = Union[..., LocalFileSourceTable]
```

One additive member. It is what lets a local staged parquet travel inside
the existing `CustomTrainingTable`, so **`Trainer.fit` needs no change on
either backend** — the same dataclass crosses, and only the union member
differs.

It is also the type a local file reader would return from `prepare_ingest`,
which is the follow-on that lets a local graph read parquet off disk. Adding
it here is not a favour to that work; it is the same missing concept.

The alternative — the local train-table area keeping a private
`{job_id: staged_path}` registry and `_LocalTrainingBackend` resolving it by
`train_table_job_id` — needs no kumo-api change and matches the local
backend's "resolve by ID from the area that owns it" idiom. It is rejected
because the modification then cannot cross a process boundary or be recorded
on a job, and because it makes the two backends' `fit` paths diverge for no
gain.

### Changed: `TrainingTable`

```python
class TrainingTable:

    def attach(
        self,
        rows: Union[pd.DataFrame, str, Path, SourceTable],
        spec: TrainingTableSpec,
        *,
        validate: bool = True,
        extensive_validation: bool = False,
    ) -> "TrainingTable": ...

    def transform(
        self,
        fn: Callable[[pd.DataFrame], pd.DataFrame],
        spec: TrainingTableSpec,
        *,
        validate: bool = True,
    ) -> "TrainingTable": ...

    def transform_sql(
        self,
        sql: str,                      # `{train_table}` names the input
        spec: TrainingTableSpec,
        *,
        validate: bool = True,
    ) -> "TrainingTable": ...
```

`attach` dispatches on the argument type. A `DataFrame` is staged to parquet;
a path is validated in place and never copied; a `SourceTable` is staged
through `source.prepare_ingest`. All three converge on one staged parquet and
one `CustomTrainingTable`.

All three return a **new** `TrainingTable` sharing the same `job_id` — the
plan, split, and snapshot come from the generation job and are not being
changed — with `_custom` set. Derivation stops being a mutation, which is what
makes `weighted = train_table.transform(...)` read correctly and what lets a
caller train both the weighted and unweighted table from one generation.

`transform` and `transform_sql` are `attach` plus the read: `transform` reads
the generated parquet, applies `fn`, and attaches the result; `transform_sql`
runs the query — DuckDB over the parquet locally, pushed to the warehouse the
table was exported to on cloud — and attaches its output.

`update()` remains, unchanged in signature and in its mutate-and-return-`Self`
behaviour, as a deprecated alias for `attach` restricted to `SourceTable`.
Every existing caller keeps working, including
`examples/datasets/weighted_train_table.py`.

### Changed: `PredictiveQuery`

```python
class PredictiveQuery:

    def custom_training_table(
        self,
        rows: Union[pd.DataFrame, str, Path, SourceTable],
        spec: TrainingTableSpec,
        plan: Optional[TrainingTableGenerationPlan] = None,
        *,
        validate: bool = True,
    ) -> TrainingTable: ...
```

The second entry point, for a caller who is **supplying** the rows rather than
modifying generated ones. It runs no generation job, and the `TrainingTable`
it returns has `job_id is None` and carries `_pquery_id` instead.

`TrainingTable.job_id` therefore becomes `Optional[str]`, and `Trainer.fit`
resolves the query from whichever is present:

```python
if train_table.job_id is not None:
    pq_id = train_table_api.get(train_table.job_id).config.pquery_id
else:
    pq_id = train_table.pquery_id
```

`fit`'s existing `train_table` XOR `pquery` rule is untouched — a custom table
built this way is still passed as `train_table`.

### When generation is required, and when it is waste

The predictive query is **always** required, and not as bookkeeping. Every
field the native builder validates a supplied parquet against comes from it:
`split_config` and `split_family`, `entity_node_type` and
`entity_primary_key` (used to check the parquet's `ENTITY1` values against the
index), `target_stype`, `task_type`, `rhs_node_type`, `rhs_is_temporal`,
`input_timestamp_unit`. `build_training_task_plan` derives all of it from
query + graph + model plan + snapshot. The query is the schema contract.

A **generated training table** is required only when the modification is a
function of the generated rows — weighting, which is the motivating case. When
the caller brings their own labels it is pure waste, and the engine already
says so: `generate_custom_train_table` reads `split_config`, `tt_id`,
`entity_node_type`, `entity_primary_key`, `target_stype`, `index_path`,
`input_timestamp_unit`, `rhs_node_type`, `task_type`, and `rhs_is_temporal`
off the plan — and never touches `request.output_path`, the generated table.
`materialize_training_task` correspondingly skips `generate_train_table`
entirely when a custom table is set (`workflow.py:467`).

The wire already permits the un-anchored form: `TrainingJobRequest.pquery_id`
is required, while `train_table_job_id` and `custom_train_table` are
independent optionals (`kumoapi/jobs.py:311-332`). Only the SDK forbids it,
because `TrainingTable.__init__` demands a `DONE` generation job and is the
sole carrier of `_custom_train_table`. There is even dead prior art for the
un-anchored shape: `TrainingJobRequest.train_table_override`, deprecated and
referenced nowhere in the SDK.

### Usage

```python
train_table = pquery.generate_training_table(plan)

# 1. In-process — the common case on a local backend.
weighted = train_table.transform(
    lambda df: df.assign(weight=np.where(df.TARGET > df.TARGET.mean(), 2, 1)),
    spec=TrainingTableSpec(weight_col="weight"),
)

# 2. Already on disk, built by something else.
weighted = train_table.attach("/data/weighted.parquet",
                              TrainingTableSpec(weight_col="weight"))

# 3. Built in a warehouse — today's flow, unchanged.
weighted = train_table.attach(SourceTable("weighted_tt", connector),
                              TrainingTableSpec(weight_col="weight"))

trainer.fit(graph=graph, train_table=weighted)
```

Bringing your own labels, with no generation job at all:

```python
train_table = pquery.custom_training_table(
    my_labels_df, TrainingTableSpec(weight_col="weight"))

trainer.fit(graph=graph, train_table=train_table)
```

Identical on both backends. Nothing in this block names a bucket, a project,
or a staging path.

### Deliberately not in the API

**No write-back into the generated table.** It is addressed by a job ID, other
records reference it, and reproducibility depends on it not moving.
Immutability is correct; derivation is the fix.

**No `Weight` vocabulary, yet.** `TrainingTableGenerationPlan.weight_col`
already exists and already covers the case where the weight is a column on the
target table — no modification needed at all. A declarative
`Weight.balanced()` / `Weight.recency(...)` is the natural extension of that
field rather than of this API, and belongs in a separate document; it is noted
in **Phase 4** only to fix its place in the ladder.

**No `LocalConnector`.** `attach` takes a bare path because a custom training
table is one file. Naming a directory of local parquet as a graph table wants
a connector with a root and a reader, which is the follow-on this document
unblocks and does not attempt.

______________________________________________________________________

## Backend

### `TrainTableBackend` — one new method

```python
class TrainTableBackend(JobBackend, Protocol):

    def stage_custom_table(
        self,
        rows: Any,                     # DataFrame | path | SourceTableType
        spec: TrainingTableSpec,
        *,
        id: Optional[str] = None,      # the generation job, when there is one
        pquery_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> CustomTrainingTable:
        ...
```

Exactly one of `id` and `pquery_id` is set: `id` when deriving from generated
rows, `pquery_id` when supplying them.

Cloud stages into the project's managed area — the destination
`ArtifactExportRequest` already reaches — and returns a `CustomTrainingTable`
over the resulting `SourceTableType`, running the existing
`validate_custom_train_table` check when asked. Its `update()` path is
unchanged: a `SourceTable` argument is passed straight through, with no
staging.

Local writes `<scratch>/train-tables/<job_id>/custom.parquet` and returns a
`CustomTrainingTable` over a `LocalFileSourceTable`.

### New: the local train-table area

`_LocalTrainTableBackend`, replacing `_Unwired`. Generation needs a snapshot
and a plan, both resolvable from what the request already carries:
`GenerateTrainTableRequest.pquery_id` names a `PQueryResource`, which holds
the `GraphDefinition`, which the graph area snapshots. `graph_snapshot_id` is
`None` from `generate_training_table` (`pquery/predictive_query.py:417`), so
the area snapshots the pquery's graph itself, exactly as
`_LocalTrainingBackend` resolves its own inputs by ID.

```
create()          build_training_task_plan(request) → generate_train_table(plan)
                  record train_table_path, staging_dir, tt_id, model_plan
get_table_data()  the generated parquet paths
get_split_masks() split_counts / split_meta off the builder result
stage_custom_table()
validate_custom_train_table()
```

`get_progress` returns `{}` — generation is synchronous, so the SDK's progress
bar falls through to its non-temporal path, the same accommodation
`_LocalTrainingBackend` already relies on.

### Training area

Two changes in `_LocalTrainingBackend.create`:

- Stop refusing `train_table_job_id` (`training.py:253`). Instead, resolve the
  job and — because the table is already generated and its bundle already
  staged — **skip generation and go straight to `ingest_train_table`**. This
  is the decoupling the cloud advertises, finally paying off locally: a plan
  sweep over one generated table stops regenerating it per run.
- Translate `request.custom_train_table` into a
  `PreparedCustomTrainingTable(parquet_path, weight_column, time_column)` and
  pass it to `from_spec(custom_training_table=...)`.

### `_LocalPQueryBackend.suggest_training_table_plan`

Today it raises (`local/pquery.py:98`), which makes
`generate_training_table()` with no plan fail before reaching the backend.
It returns `TrainingTableGenerationPlan()` instead — every field defaults to
`inferred` or `missing`, and `kumo_core` resolves them downstream. That is
honest: the local backend is not suggesting a tuned plan, it is declining to
override the engine's inference.

______________________________________________________________________

## Normalization, validation, and time

Three responsibilities the cloud platform performs that nothing performs
locally. They are shared by all three `attach` producers, which is why they
are built first.

**What the gate does and does not cover.** It types the roles this area can
name: `ENTITY*`, `TARGET*`, `TIMESTAMP`, the weight, and a split column called
`SPLIT`. It types them against the schema DiskGraph reads rather than the one
pyarrow reports: a dictionary-encoded column is restored by arrow-rs from the
footer's `ARROW:schema`, meets a `cell_to_value` with no `DictionaryArray`
arm, and costs every row — while `pds.dataset` reports a dictionary over ints
as a plain int. A rewrite decodes such a column on the way out, so only a
table read in place is gated on its stored encoding. It does **not** type a
manual split whose column is named anything else. That name lives in the task plan's `split_projection`
(`split_contract.py:72`), and reaching it means building the plan, which reads
column statistics out of the DiskGraph index that staging otherwise never
opens — roughly the cost of a generation run, paid on every `validate` call.
Such a column goes through the same `cell_to_value`, so an unreadable one
routes every row on a null. Closing it is worth doing when a local caller can
actually declare one.

**Row-count parity honours the cloud's escape hatch.**
`KUMO_CUSTOM_TT_ALLOW_ROW_COUNT_MISMATCH` is read by the same name, with the
same `False` default, and downgrades the refusal to a warning exactly as
`custom_train_table.py:684` does. `docker-compose.batch-validation.yml` ships
it set, so a table validated under that configuration would otherwise be
trainable on one backend and refused on the other.

**Stale entity keys are counted, not refused.** An `ENTITY1` the graph snapshot
does not hold is skipped by DiskGraph *"silently skipped (documented
behavior)"* (`custom_tt_chunk_processor.rs:24`) — three rows in, two trained
on, run reports success. Staging counts them against the generated table's own
entity column and warns with the count and a sample. A warning rather than a
refusal because the surviving rows are trainable and only the caller knows
whether losing the rest is acceptable.

**Dtype conversion — settled: there is none either.** A column DiskGraph
would read as nothing is refused, not widened, and the message says *"Cast the
column to the required type"* as hosted's does. Widening `int32` to `int64`
looks harmless and is not: hosted demands exact Arrow parity against the
generated table (`TIMESTAMP` alone relaxed to any timestamp or `int64`), so a
Databricks `INT` table that local widened would train here and be refused on
deploy. That is the same inversion as a local rename, and the worse direction
to fail in — at deploy rather than at author time. There is no vendor
exception: every source now stages to parquet and is gated on the parquet that
arrives, so an `int32` reaches the same check on every connector.

**Column normalization — settled: there is none.** The caller supplies
`ENTITY1`/`TARGET1`, and neither backend converts anything. That is a product
contract, not an implementation gap. `generate_custom_train_table` passes the
path through without renaming, neither generator emits the singular spelling,
and the cloud compares a supplied table against the generated one column by
column, so a table that a local rename made trainable here would fail there.
The refusal names the columns the table actually carries.

**Validation.** `validate_custom_train_table` has no local implementation, and
without it a malformed table fails deep inside the native builder. The local
check runs against the staged parquet's schema and is cheap:

- required columns present after normalization (`ENTITY1`, `TARGET1`,
  `TIMESTAMP` for temporal queries);
- every column named in the `TrainingTableSpec` present;
- `weight_col` numeric, non-negative, non-null — the constraint `update()`
  documents but has never enforced anywhere;
- under `extensive_validation`, row count against the generated table.

**Timestamp unit.** The builder requires `input_timestamp_unit` and applies it
*verbatim* to Int64 `TIMESTAMP` columns; typed Arrow timestamps self-describe
and ignore it. Kumo's own rehydration passes `"nanoseconds"`. A table that
round-tripped through a warehouse unload or through pandas can come back with
a different int64 representation, and the failure mode is not an error — it is
a misread time column, hence a wrong split. Staging reads the parquet schema
and refuses an Int64 `TIMESTAMP` whose inferred unit disagrees with the plan's,
rather than passing the plan's value through unchecked.

______________________________________________________________________

## Implementation, in order

Ordered so each phase is independently useful and the shared work lands first.

**Phase 1 — local generation. Landed.** `_LocalTrainTableBackend`
(`kumoai_studio/backend/local/train_table.py`) resolves a request's query, snapshots
the query's own graph, and calls `build_training_task_plan` then
`generate_train_table`. `suggest_training_table_plan` returns a default plan
rather than raising, so a caller who passes none reaches the backend at all.
The export area writes a generated table through the sink registered for its
connector. `_LocalTrainingBackend` accepts `train_table_job_id` and lifts the
generation plan back in (see below).
*Delivers:* `pquery.generate_training_table()`, `data_df()`, `data_urls()`,
and `training_table.export()` to a warehouse, locally.

*Not yet:* `LocalFileSourceTable` in kumo-api (Phase 2 needs it, this phase
does not), `get_split_masks`, and real bundle reuse -- a run that names a
generated table currently regenerates it rather than calling
`ingest_train_table` on the staged bundle. Regeneration is correct, because
generation is a pure function of the query, snapshot, and plan, and all three
are pinned; it is only wasteful.

**Phase 2 — staging. Partially landed.** `stage_custom_table` exists on the
`TrainTableBackend` protocol and on the local backend, which resolves a
`LocalFileSourceTable` or any connector with a registered reader, refuses a
dtype DiskGraph would read as nothing, and compares the result against the
generated table. `_custom_train_table` is threaded into
`PreparedCustomTrainingTable`, so `fit()` trains on the supplied rows.
`TrainingTable.update` accepts a `LocalFileSourceTable`, which is the
connector-free local spelling.
*Delivers:* weighted training locally.

*Not yet:* `attach` / `transform` / `transform_sql`,
`PredictiveQuery.custom_training_table`, `TrainingTable.job_id` widened to
`Optional`, and cloud-side staging -- `GenerateTrainTableJobAPI.stage_custom_table`
refuses and points at `update()`. The **Public API** section above describes
that target, not this build; the Usage block in it does not run yet.
`ENTITY`/`TARGET` are *not* renamed — see **Column normalization** above,
which is now a contract rather than a deferral.

**Phase 3 — `attach(SourceTable)` and `transform`.** Route a `SourceTable`
through `source.prepare_ingest` into the same staging step. `transform` and
`transform_sql`.
*Delivers:* the cloud flow, unchanged in meaning, now available locally for
any connector with a registered reader.

**Phase 4 — declarative weights.** Extend the existing plan field
`TrainingTableGenerationPlan.weight_col` with a computed form
(`balanced`, `recency`, an expression), so
the common cases need no modified table at all. Separate document; listed
here so the ladder is legible.

______________________________________________________________________

## Constraints worth knowing

**Only Snowflake has a local reader.** `_default_readers()`
(`local/source.py:283`) registers one factory. Every local script in the repo
uses `SnowflakeConnector`. So `attach(SourceTable("x", S3Connector(...)))` —
precisely what the shipped weighted-table example does — raises
`_unsupported('source', 'read from S3')` until a file or object-store reader
exists. This is the concrete reason `attach` accepts a path and a DataFrame
rather than a `SourceTable` alone: on a local backend, requiring a registered
connector to name a file the process can `open()` restates the awkwardness
instead of removing it.

**The training-table plan depends on the model plan, and travels as a base.**
`build_training_task_plan` takes `model_plan` (`workflow.py:307`), which feeds
the split config, downsampling, and `lag_timesteps` -- but that section lives
in `ModelPlan.training_table_generation`, which is listed in
`INTERNAL_ONLY_FIELDS` (`model_plan_conversion.py:24`) and is absent from the
public contract a `TrainingJobRequest` carries. So a run that lifted the public
plan alone would regenerate under the *defaults*, producing a different table
from the one the caller generated, read, and exported.

The conversion already provides the channel: internal-only sections are taken
from its `base`. So the generation job records the section it built under, and
`_LocalTrainingBackend._model_plan` passes it back as the base when a run names
that job. The two dataclasses -- `kumoapi`'s and `kumo_core`'s -- have
identical field sets, so the section crosses as its own JSON rather than field
by field, and a field added to one and not the other fails at the conversion
instead of quietly not travelling.

**`train_table_path_override` is not the mechanism.** The hidden plan field
that points generation at an existing table is consumed by the cloud platform
only; `kumo_core` never reads it. It is prior art for the idea, not a local
implementation of it.

**Databricks is staged like every other warehouse.** It used to be read
natively -- the builder accepted `databricks_source` /
`databricks_config_file` and the engine read the custom table straight from
Databricks -- but that path is retired: the hosted platform now exports
Databricks custom tables to Kumo-owned parquet through the same source reader
Snowflake and BigQuery use, and hands the engine a path. `update()` still
restricts custom SQL sources to Databricks connectors
(`training_table.py:302`), because Databricks is the only connector whose
Custom SQL sources are supported. `stage_custom_table` should stage a
Databricks source like any other.

**Cloud validation is anchored on a generation job.** The endpoint is
`POST /{job_id}/validate_custom_train_table` (`client/jobs.py:454`), so the
un-anchored `pquery.custom_training_table()` path has no server-side check to
call on the cloud backend until the endpoint accepts a `pquery_id`. Until
then it validates at `fit` time, inside the training job, where the error is
later but not lost. Local has no such constraint: its validation is a schema
check against the staged parquet and needs only the plan.

**Only retained custom-table columns are supported.** `TrainingTableSpec` permits an optional `weight_col` and `split_col`. Dropping or reordering rows is unsupported, and the validation above is what turns that from a docstring warning into a refusal. `PreparedCustomTrainingTable` carries the path and optional weight column into a local run.

______________________________________________________________________

## Tests

**Local generation.** Generate a training table from a fixture graph, assert
the parquet exists, `data_df()` returns the expected columns under the
caller-facing names, and split counts sum to the row count.

**Reuse.** Train twice from one generated table; assert the second run ingests
the existing bundle and does not call `generate_train_table`.

**`attach` parity.** The same modification expressed as a DataFrame, as a
path, and as a `SourceTable` produces the same staged parquet and the same
`CustomTrainingTable` modulo the source member.

**Normalization round trip.** A frame read via `data_df()`, modified with
`df['TARGET']`, and attached — the exact shape of the shipped example —
stages a parquet carrying `TARGET1`.

**Validation refusals.** Missing `ENTITY1`; a `weight_col` naming an absent
column; a negative weight; a null weight; a dictionary-encoded `ENTITY1`, over
strings and over ints alike; a null `TARGET1`, which DiskGraph keeps with the
label substituted rather than skipping; a row-count mismatch under
`extensive_validation`. Each with an error naming the offending column, and
each paired with a test running the real binary over the same table to show
what the refusal buys.

**Timestamp unit.** An Int64 `TIMESTAMP` in seconds against a plan expecting
nanoseconds is refused at staging, not misread.

**No generation job.** `pquery.custom_training_table(df, spec)` trains
without a `generate_train_table` call — assert the local run never invokes it,
and that the resulting model's metrics are computed over the supplied rows.

**Cloud unchanged.** Existing `update()` tests pass untouched, and the
deprecated alias emits a warning while behaving identically.

______________________________________________________________________

## Decisions

**Derivation returns a new table rather than mutating.** It makes the
in-memory-only state of today's `update()` a non-issue, lets one generation
feed several experiments, and reads correctly at the call site.

**The backend decides where staged rows live.** A caller who has a DataFrame
should not have to have an opinion about buckets. Making that decision a
backend responsibility is what makes the user-facing code identical across
backends, which matters more than the local speedup.

**`SourceTable` stays a first-class producer.** Building a training table in
Spark or dbt is a real workflow. The change is that it becomes one of three
producers rather than the mandated one.

**The query is the anchor; the generated table is not.** A generated table is
required only when the modification is a function of its rows. Forcing a
generation run for a table the engine will then ignore is a wasted job on
cloud and a wasted bundle locally, so both entry points exist.

**A local file is a source table.** Rather than a special case in the training
path, because the same gap blocks local graph ingest from parquet, and one
union member closes both.
