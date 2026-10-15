# Snowflake ML Jobs — Snowflake CoCo

Read this file when the user asks for a Snowflake ML Job: to score (or train and score) on a
compute pool instead of in the notebook, to run unattended, to repeat a run on a schedule, or to
find out what happened to a job that was already submitted. A training request that says nothing
about jobs never needs it.

Everything in the router still holds: the same intake gates, one artifact, no execution until the
user asks. A job is a different place to run that artifact, not a different workflow. Kumo still
runs locally and in-process, inside the job's container (`adapter.md` §9).

Sources: snowflake-ml-python 2.4.0 (`snowflake.ml.jobs`), the shipped Kumo package, and one
measured job in this account (2026-10-13).

---

## 1. Job or notebook

The notebook is the default. Do not move a request into a job unless the user asks for one.

| Keep the notebook | Propose a job |
|---|---|
| A first training run of a new task | Scoring with a published model, by explicit name and version |
| "Look at the metrics with me", iterate, compare, then decide | Scoring that must finish without the notebook open |
| Explanations, diagnostics, tuning | Scoring the user wants repeated on a schedule (§8) |
| Anything that needs the user between two steps | The user asks to submit the run to a compute pool |

For an interactive request, a job may be mentioned once, as a later option for unattended
scoring after a model is published. Never wrap a fit the user wants to inspect in a job.

## 2. What a job runs

A job runs exactly one of two scripts.

- **Score a pinned model** (`batch_inference`). The model is a registry model with an explicit
  name **and** version. Intake is `kumo-data-science/accept/batch_inference.md`; the load,
  rebuild and predict code is `kumo-model-persistence/references/reuse.md`. The script contains
  no `fit` and no training table. If the user has not named the model and its version, ask
  (batch-inference Node 1a). Never train in its place, and never let the version default:
  `ModelRegistryLocation(version=None)` loads the registry's **default version**, which is the
  first version created unless someone ran `ALTER MODEL ... SET DEFAULT_VERSION`. It is not the
  newest (Source: shipped package (`kumo_snowflake.registry.SnowflakeModelRegistryMixin.fetch_model`)). The
  docstring of `kumoapi.plugin.ModelRegistryLocation` says "latest"; the implementation does not.
- **Train and score**, only when the user asks for training inside the job. It is the training
  notebook as one script, through the same gates: one blocking `fit`, then `predict`.
  Publication, if asked, follows `kumo-model-persistence/SKILL.md`.

## 3. Settings resolved at intake

The job adds settings to the Node 1d restatement. Resolve them as follows. Ask only for what this
table says to ask for.

| Setting | How it is obtained |
|---|---|
| Compute pool | The pool the user named; otherwise one GPU pool proposed `[inferred]` from `SHOW COMPUTE POOLS` (§4) |
| Runtime | `KUMO_JOB_RUNTIME` (`adapter.md` §10). Never the default runtime, and never asked |
| Job stage | The stage the user named; otherwise `<output database>.<output schema>.KUMO_JOB_STAGE` `[inferred]`. Submission creates the stage when it does not exist, so this is a write and is confirmed at the gate |
| Query warehouse | The warehouse the connectors use `[inferred]` |
| Output table | `output_location`, verbatim, as for any batch inference |
| Write mode | The user's words. If unstated, `OVERWRITE [default]` for a one-off run. A recurring run must settle it explicitly (§8) |
| Scoring time | The user's date. Otherwise omit it: the SDK anchors at the latest target time in the snapshot (Source: shipped package (`kumo.engine.PredictiveQuery.generate_prediction_table`)) |

Add these lines to the restatement:

```
- Runs as: Snowflake ML Job on <POOL> (<INSTANCE_FAMILY>, <STATE>) [supplied | inferred], runtime <KUMO_JOB_RUNTIME>
- Job stage: @<DB>.<SCHEMA>.<STAGE> [supplied | inferred]; query warehouse <WH> [inferred]
- Output: <DB.SCHEMA.TABLE>, <OVERWRITE | APPEND> [supplied | default]
- Wait: <the pool is suspended; the first node took about 17 minutes to start when measured | the pool is running>
```

Submitting is execution: it happens only when the user asks for it (router Step 3), after this
confirmation. Generating the script and the submission cell is Step 2.

## 4. Compute pool

Read the pool before relying on it. A pool's name does not establish its hardware.

```sql
SHOW COMPUTE POOLS;   -- name, state, instance_family, auto_suspend_secs
```

An `instance_family` starting with `GPU_` (for example `GPU_NV_S`) is a GPU pool. On this host
Kumo scoring is expected on GPU, and a CPU run has not been verified.

| Situation | Response |
|---|---|
| The user named a GPU pool | Use it |
| The user named a CPU pool | Say it is CPU, that Kumo runs on GPU here and CPU is unverified, and name the GPU pools available. Ask which to use. Never switch pools silently |
| No pool named; GPU pools are available | Propose one `[inferred]` at the gate, preferring one that is already running |
| No pool named; only CPU pools are available | Say so and ask whether to proceed on CPU (unverified) or have a GPU pool provided. Never create a compute pool |

Report the pool's `state`. A `SUSPENDED` pool must start a node before the job runs (§9).

## 5. The job script

The script runs inside the job container. It has no notebook around it: no cells, no display, no
user. Write it as a file next to the notebook (`%%writefile score_job.py` or
`Path('score_job.py').write_text(...)`). For scoring a pinned model:

Every name the script uses is a literal written into the file. `%%writefile` does not
interpolate, and the container shares nothing with the notebook, so a name that is only a
notebook variable is a `NameError` on the first line that reads it. That includes `graph`: the
script rebuilds it.

```python
from snowflake.snowpark.context import get_active_session

session = get_active_session()   # first line of work: before any Kumo connector exists

from pathlib import Path

import kumo.engine as kumo
from kumoapi.jobs import PredictionArtifactType, WriteMode

# Literals from the Node 1d restatement, written into this file (replace each placeholder).
WAREHOUSE = '<warehouse>'
SOURCE_DATABASE, SOURCE_SCHEMA = '<source_db>', '<source_schema>'
REGISTRY_DATABASE, REGISTRY_SCHEMA = '<registry_db>', '<registry_schema>'
OUTPUT_DATABASE, OUTPUT_SCHEMA, OUTPUT_TABLE = '<output_db>', '<output_schema>', '<output_table>'
MODEL_NAME, MODEL_VERSION = '<model_name>', '<model_version>'
WRITE_MODE = WriteMode.OVERWRITE   # or WriteMode.APPEND: the write mode confirmed at the gate (§3)
THRESHOLD = 0.5   # binary classification only: the restated binary_classification_threshold
TRAINED_QUERY = '<the recorded predictive query>'

scratch = Path('/tmp/kumo-job/scratch')
scratch.mkdir(parents=True, exist_ok=True)
kumo.configure(scratch_dir=scratch)

def connect(database, schema_name):
    return kumo.SnowflakeConnector.from_spcs_env(
        warehouse=WAREHOUSE, database=database, schema_name=schema_name)

source = connect(SOURCE_DATABASE, SOURCE_SCHEMA)
registry = connect(REGISTRY_DATABASE, REGISTRY_SCHEMA)
output = connect(OUTPUT_DATABASE, OUTPUT_SCHEMA)

loaded = kumo.Trainer.load_from_connector(
    registry,
    location=kumo.ModelRegistryLocation(model_name=MODEL_NAME, version=MODEL_VERSION),
)
# Rebuild the recorded tables, column lists and edges into `graph` from `source`
# (kumo-model-persistence/references/reuse.md), then:
pq = kumo.PredictiveQuery(graph=graph, query=TRAINED_QUERY)
pq.validate()
snapshot = graph.snapshot()
population = pq.generate_prediction_table(
    snapshot, plan=loaded.training_table_generation_plan)   # add prediction_time= if the user gave one
result = loaded.predict(
    population, snapshot=snapshot, binary_classification_threshold=THRESHOLD,
    output_config=kumo.PredictionOutputConfig(
        connector=output, artifact_type=PredictionArtifactType.PREDICTIONS,
        table_name=OUTPUT_TABLE, write_mode=WRITE_MODE),
)
written = session.table([OUTPUT_DATABASE, OUTPUT_SCHEMA, OUTPUT_TABLE]).count()
print('predicted rows:', result.num_rows, 'table rows:', written)
assert written == result.num_rows, 'write-back row count does not match'
```

The `predict` arguments above are for binary classification. The task-specific ones follow the
trained query: `binary_classification_threshold=THRESHOLD` for binary classification, and
`num_classes_to_return=TOP_K` for link prediction and ranking, with `TOP_K` the `top_k` resolved
at intake. Link prediction fails in `predict` without it
(`kumoai-reference/references/kumo_finetuning_sdk.md` §9).

- **Take the session once, at the top, before any Kumo connector.** The job's launcher already
  holds a Snowpark session, and `from_spcs_env` opens a second one. After that,
  `get_active_session()` and `Session.builder.getOrCreate()` both raise
  `SnowparkSessionException (1409): More than one active session`. This was measured in a job.
  Reuse `session` for every Snowflake call in the script (`kumoai-reference/references/kumo_finetuning_sdk.md` §3.3).
- **Connectors come from `from_spcs_env` only.** The job container mounts the runtime token, and a
  connector built from it worked inside a job. The `connections.toml` and secrets fallbacks of
  `runtime.md` §3 do not apply here. Always pass `warehouse`, `database` and `schema_name`: the
  container's own defaults name the job's schema, not the user's data.
- Everything in the container, `/tmp` included, is gone when the job ends. The output table is the
  only result that survives (`runtime.md` §3.1).
- Let a failure raise. Do not catch broad exceptions, so that a failed scoring run ends the job as
  `FAILED` rather than `DONE`.
- The host's runtime guards still apply (`adapter.md` §1.4), as do every rule of
  `kumoai-reference/SKILL.md` for the Kumo code itself.
- An APPEND run checks the row delta, not the table total (`kumoai-reference/references/outputs.md`).

## 6. Submission, waiting and the result

The submission cell runs in the notebook. That notebook may already hold a Kumo connector, so
pass `session=` explicitly. Without it the submission looks up the active session itself and
fails with "More than one active session is found".

```python
from snowflake.ml import jobs

KUMO_JOB_RUNTIME = '2.9.0'   # adapter.md §10: never leave the runtime to its default
if 'KUMO_JOB_REQUIREMENTS' not in globals():
    raise RuntimeError('Kumo is not enabled for ML Jobs in this environment; '
                       'ask the administrator. Nothing was submitted.')

job = jobs.submit_file(
    'score_job.py', COMPUTE_POOL,
    stage_name=JOB_STAGE,
    query_warehouse=WAREHOUSE,
    runtime_environment=KUMO_JOB_RUNTIME,
    database=JOB_DATABASE, schema=JOB_SCHEMA,
    name=JOB_NAME,
    session=session,
    **KUMO_JOB_REQUIREMENTS,
)
print('submitted:', job.id)

try:
    status = job.wait(timeout=JOB_TIMEOUT_SECONDS)
except TimeoutError:
    status = None
    # The wait gave up; the job did not. It is still running and still spending.
    print(f'ML Job {job.id} is still running after {JOB_TIMEOUT_SECONDS}s. Not cancelled.')
if status is not None:
    print('status:', status)
    if status != 'DONE':
        print(job.get_logs(limit=200))
        raise RuntimeError(f'ML Job {job.id} ended {status}')
    written = session.table([OUTPUT_DATABASE, OUTPUT_SCHEMA, OUTPUT_TABLE]).count()
    print('table rows:', written)
```

- `KUMO_JOB_RUNTIME` and `KUMO_JOB_REQUIREMENTS` are the host's (`adapter.md` §10). Pass them
  through unchanged. Never write a value for `KUMO_JOB_REQUIREMENTS`, add to it, print it or
  explain it.
- Pin the runtime. When `runtime_environment` is omitted, the runtime comes from the environment
  or the pool's default, and the default moves with Snowflake releases (2.10 at the time of
  writing). That runtime carries a torch build outside the range this Kumo build accepts. 2.9.0 is
  the runtime Kumo has run on in a job.
- The image's Python version follows the **submitting** process's Python
  (Source: snowflake-ml-python 2.4.0 (`snowflake.ml.jobs.job_definition`)). The notebook kernel
  is Python 3.12 (`runtime.md` §1.1). Submit from it.
- `compute_pool` is the second positional argument and `stage_name` is required.
  `query_warehouse` defaults to the session's warehouse, so pass it explicitly. `database` and
  `schema` place the job, and default to the session's. A unique suffix is appended to `name`,
  so the job is found later by `job.id`, never by `name` (§7).
- `job.wait()` returns a terminal status: `DONE`, `FAILED`, `CANCELLED`, `INTERNAL_ERROR` or
  `DELETED`. These are Snowflake ML Job statuses (Source: snowflake-ml-python 2.4.0
  (`snowflake.ml.jobs.JOB_STATUS`)). Kumo itself still has none
  (`kumoai-reference/references/kumo_finetuning_sdk.md` §11.3).
- Choose `JOB_TIMEOUT_SECONDS` to cover pool start-up (§9) plus scoring. On `TimeoutError`,
  report the job id and that it is still running. Cancel (`job.cancel()`) only when the user
  asks.
- Report the job id, the final status, the destination `DB.SCHEMA.TABLE` verbatim and the row
  count read back from it.

## 7. A job that already ran: status and logs

```python
from snowflake.ml import jobs

job = jobs.get_job('<JOB_ID>', session=session)   # job.id as printed at submission (§6)
# Without the id: jobs.list_jobs(limit=20, database=JOB_DATABASE, schema=JOB_SCHEMA, session=session)
print(job.status)
print(job.get_logs(limit=500))                 # the script's own output
# job.get_logs(limit=500, verbose=True)        # adds the platform's log lines
```

Read the traceback before you diagnose. The verbose log also carries platform noise, for example
failures to reach `grafana.com`, and Ray start-up lines. That noise is not the failure.

| Log signal | Cause and fix |
|---|---|
| `SnowparkSessionException`, `1409`, "More than one active session" | The script asked for the session after a Kumo connector existed. Take it once at the top (§5) |
| "More than one active session is found. Please specify the session explicitly" at submission | The submission cell omitted `session=` (§6) |
| `ModuleNotFoundError: No module named 'kumo'` | Kumo is not present in the job container: the host's job values are missing or wrong. That is for the administrator, not something to add to the job |
| An error raised importing torch, or from a torch version check | The runtime was not pinned. Use `KUMO_JOB_RUNTIME` |
| The status stays `PENDING` | The pool is starting a node, or it is busy. Not a failure |

Propose the specific fix. Never resubmit on your own: every submission is a new GPU run. Never
delete the job, because deleting it discards its status and logs.

## 8. Recurring runs

Never create a schedule on your own initiative. A request for recurring runs gets a proposal. The
schedule is created only after the user explicitly confirms that proposal, because a task is a
persistent object that spends GPU credits on every run until someone suspends or drops it.

`snowflake.ml.jobs` has no schedule argument. Register the job once as a job definition, a preview
feature of snowflake-ml-python, and run its statement from a Snowflake task:

```python
from snowflake.ml.jobs import MLJobDefinition

job_def = MLJobDefinition.register(
    'score_job.py', COMPUTE_POOL, JOB_STAGE, session=session,
    query_warehouse=WAREHOUSE, runtime_environment=KUMO_JOB_RUNTIME,
    database=JOB_DATABASE, schema=JOB_SCHEMA, name=JOB_NAME,
    **KUMO_JOB_REQUIREMENTS,
)
print(job_def.to_sql())   # CALL SYSTEM$EXECUTE_ML_JOB(...): the task body
```

```sql
CREATE TASK <DB>.<SCHEMA>.<TASK_NAME>
  WAREHOUSE = <WH>
  SCHEDULE = 'USING CRON 0 6 * * * UTC'
  AS <the statement printed by to_sql()>;
ALTER TASK <DB>.<SCHEMA>.<TASK_NAME> RESUME;   -- a new task is suspended until resumed
```

Settle each of these in the proposal before anything is created:

- **Scoring time moves forward.** The script omits `prediction_time`, so each run anchors at the
  latest data. A date fixed in the script would score the same day forever.
- **Write mode for repeated runs.** OVERWRITE keeps only the latest run. APPEND keeps history, but
  a retried run adds a second copy of its rows. If the user already chose a mode, restate it with
  that consequence. Otherwise ask which they want.
- **The model version stays pinned.** A new version reaches the schedule only through an explicit
  change.
- **Cost recurs.** Each run pays for pool start-up, scoring and the idle tail (§9).
- **Timeout.** By default the statement waits for the job, so the task's `USER_TASK_TIMEOUT_MS`
  must cover pool start-up plus scoring.
- **Privileges.** `CREATE TASK` on the schema, `EXECUTE TASK` on the account, and usage on the
  warehouse and the pool.

Suspending (`ALTER TASK … SUSPEND`) or dropping the task is the user's call.

> Not run in this account: a task executing a registered job definition. Say so when proposing it.

## 9. Cost and latency

Do not state a credit figure as fact. Explain what a run is billed for and offer to measure it.

- **GPU node time** for the pool's `instance_family`, at the rate on the user's Snowflake rate
  card.
- **Start-up when the pool is suspended.** On 2026-10-13 a suspended `GPU_NV_S` pool took about 17
  minutes to start a node before the job ran. Whether that wait is billed has not been measured.
- **The job itself.** Container start and the Kumo import took about a minute. Scoring time depends
  on the data and is unknown until measured.
- **The idle tail.** After the job, the pool stays up for its `auto_suspend_secs` (600 s on the
  measured pool).
- **Query warehouse time** for the SQL the job issues.

A notebook on an already-running GPU starts at once. A job on a suspended pool does not, so say
which case applies before the user submits. After a run, cost can be read from
`SNOWFLAKE.ACCOUNT_USAGE.SNOWPARK_CONTAINER_SERVICES_HISTORY` and `METERING_HISTORY`. Both lag
by hours.
