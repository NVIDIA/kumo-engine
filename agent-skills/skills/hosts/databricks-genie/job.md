# Databricks jobs — Genie Code

Read this file when the user asks for a job: to score (or train and score) as a Databricks job
instead of in the notebook, to run unattended, to repeat a run on a schedule, or to find out what
happened to a job run that was already submitted. A training request that says nothing about jobs
never needs it.

Everything in the router still holds: the same intake gates, one artifact, no execution until the
user asks. A job is a different place to run that artifact, not a different workflow. Kumo still
runs locally and in-process, on the job's GPU cluster.

Sources: databricks-sdk (`databricks.sdk.service.jobs`), the shipped Kumo package, and the eval
harness's job runs: generated notebooks submitted with `runs/submit` to an existing T4 cluster,
connecting with `DatabricksIdentity()` (2026-10-21).

---

## 1. Job or notebook

The notebook is the default. Do not move a request into a job unless the user asks for one.

| Keep the notebook | Propose a job |
|---|---|
| A first training run of a new task | Scoring with a published model, by explicit name and version |
| "Look at the metrics with me", iterate, compare, then decide | Scoring that must finish without the notebook open |
| Explanations, diagnostics, tuning | Scoring the user wants repeated on a schedule (§8) |
| Anything that needs the user between two steps | The user asks to submit the run to a cluster |

For an interactive request, a job may be mentioned once, as a later option for unattended
scoring after a model is published. Never wrap a fit the user wants to inspect in a job.

## 2. What a job runs

A job runs one notebook task, a **job notebook** separate from any interactive one.

- **Score a pinned model** (`batch_inference`). The model is a Unity Catalog registered model
  with an explicit name **and** version (`runtime.md` §5.2). Intake is
  `kumo-data-science/accept/batch_inference.md`; the load, rebuild and predict code is
  `kumo-model-persistence/references/reuse.md`. The notebook contains no `fit` and no training
  table. If the user has not named the model and its version, ask (batch-inference Node 1a).
  Never train in its place, and never let the version default: `version=None` loads whichever
  Kumo version is newest when the run starts, so a later publication silently changes what a
  scheduled job scores.
- **Train and score**, only when the user asks for training inside the job. It is the training
  notebook, through the same gates: one blocking `fit`, then `predict`. Publication, if asked,
  follows `runtime.md` §5.1.

## 3. Settings resolved at intake

The job adds settings to the Node 1d restatement. Ask only for what this table says to ask for.

| Setting | How it is obtained |
|---|---|
| Cluster | The cluster the user named; otherwise one GPU cluster proposed `[inferred]` (§4) |
| Job notebook | `<the interactive notebook's folder>/<task>_job` `[inferred]`: a new notebook, never an existing one |
| SQL warehouse | The warehouse the connectors use `[inferred]` |
| Output table | `output_location`, verbatim, as for any batch inference |
| Write mode | The user's words. If unstated, `OVERWRITE [default]` for a one-off run. A recurring run must settle it explicitly (§8) |
| Scoring time | The user's date. Otherwise omit it: the SDK anchors at the latest target time in the snapshot (Source: shipped package (`kumo.engine.PredictiveQuery.generate_prediction_table`)) |

Add these lines to the restatement:

```
- Runs as: Databricks job run on cluster <NAME> (<CLUSTER_ID>; <NODE_TYPE>, <SPARK_VERSION>, <STATE>) [supplied | inferred]
- Job notebook: <workspace path> [inferred]; SQL warehouse <NAME> (<ID>) [inferred]
- Output: <catalog.schema.table>, <OVERWRITE | APPEND> [supplied | default]
- Wait: <the cluster is terminated and starts on submission, typically several minutes | the cluster is running>
```

Submitting is execution: it happens only when the user asks for it (router Step 3), after this
confirmation. Writing the job notebook and the submission cell is Step 2.

## 4. Cluster

The job runs on an **existing** cluster. Never create a cluster or change one's configuration.
Read the cluster before relying on it; a name does not establish its hardware:

```python
from databricks.sdk import WorkspaceClient

c = WorkspaceClient().clusters.get(CLUSTER_ID)
print(c.cluster_name, c.state, c.spark_version, c.node_type_id)
```

A GPU cluster runs a GPU ML runtime: its `spark_version` contains `gpu-ml`. Kumo runs on GPU
here; a CPU run has not been verified.

| Situation | Response |
|---|---|
| The user named a GPU cluster | Use it |
| The user named a cluster without a GPU | Say so, that Kumo runs on GPU here and CPU is unverified, and name the GPU clusters available. Ask which to use. Never switch clusters silently |
| No cluster named; the interactive notebook is on a GPU cluster | Propose that one `[inferred]` |
| No cluster named; other GPU clusters are visible (`clusters.list()`) | Propose one `[inferred]`, preferring one that is running |
| No GPU cluster visible | Say so and stop. Providing one is for the user or an administrator |

Report the cluster's `state`. A `TERMINATED` cluster starts when the run is submitted (§9).

## 5. The job notebook

The job notebook runs with no one watching: no widgets, no display, no user. Every value it uses
is a literal in its config cell; it shares nothing with the interactive notebook, so it rebuilds
the graph itself. For scoring a pinned model, its cells are:

1. The config cell (`runtime.md` §1), with the literals from the Node 1d restatement:
   `CATALOG`, `SCHEMA`, `WAREHOUSE_ID`, `REGISTRY_CATALOG`, `REGISTRY_SCHEMA`, `MODEL_NAME`,
   `MODEL_VERSION` (an integer as a string), `OUTPUT_CATALOG`, `OUTPUT_SCHEMA`, `OUTPUT_TABLE`,
   `WRITE_MODE`, the threshold or `TOP_K`, and `TRAINED_QUERY`.
2. The connect step and `run_sql` (`runtime.md` §2, §3), and the registry connector (§5).
3. `Trainer.load_from_connector` with the explicit version (`runtime.md` §5.2).
4. The recorded tables, column lists and edges rebuilt into `graph`
   (`kumo-model-persistence/references/reuse.md`), then `PredictiveQuery`, `snapshot`, and
   `generate_prediction_table(snapshot, plan=loaded.training_table_generation_plan)` — add
   `prediction_time=` only if the user gave a scoring time.
5. `loaded.predict(...)` with the task's arguments and the `runtime.md` §4 output config, then the
   read-back assertion.

The `predict` arguments follow the trained query: `binary_classification_threshold=THRESHOLD`
for binary classification, and `num_classes_to_return=TOP_K` for link prediction and ranking
(`kumoai-reference/references/kumo_finetuning_sdk.md` §9).

- `DatabricksIdentity()` in a job run takes the credentials of the user the run runs as — the
  submitter for `runs/submit`, the job's owner or `run_as` for a scheduled job.
- Everything on the cluster's local disk is gone when the cluster terminates. The output table is
  the only result that survives.
- Let a failure raise. Do not catch broad exceptions, so that a failed run ends `FAILED`.
- An APPEND run checks the row delta, not the table total (`kumoai-reference/references/outputs.md`).

## 6. Submission, waiting and the result

Submit from the interactive notebook (or Genie Code's own Python), as the user. `runs/submit`
makes a one-time run that is not a saved job:

```python
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

ws = WorkspaceClient()
run_id = ws.jobs.submit(
    run_name=JOB_NAME,
    tasks=[jobs.SubmitTask(
        task_key='score', existing_cluster_id=CLUSTER_ID,
        notebook_task=jobs.NotebookTask(notebook_path=JOB_NOTEBOOK))],
    timeout_seconds=JOB_TIMEOUT_SECONDS,
).run_id
print('submitted run:', run_id)

deadline = time.time() + WAIT_SECONDS
run = ws.jobs.get_run(run_id)
while (run.state.life_cycle_state in (jobs.RunLifeCycleState.PENDING, jobs.RunLifeCycleState.QUEUED,
                                      jobs.RunLifeCycleState.RUNNING, jobs.RunLifeCycleState.TERMINATING,
                                      jobs.RunLifeCycleState.BLOCKED)
       and time.time() < deadline):
    time.sleep(30)
    run = ws.jobs.get_run(run_id)
print(run.run_page_url, run.state.life_cycle_state, run.state.result_state)
```

- A run still going at the deadline is **not** cancelled: report its id and page, and that it is
  still running. Cancel (`ws.jobs.cancel_run(run_id)`) only when the user asks.
- `result_state` `SUCCESS` is the only success. On any other, read the error (§7) before saying
  anything about the cause.
- `timeout_seconds` is the run's own limit, enforced by Databricks: it must cover cluster start-up
  plus scoring.
- Report the run id, its page, the final state, the destination `catalog.schema.table` verbatim
  and the row count read back from it (`runtime.md` §4).

## 7. A run that already happened: state and errors

```python
from databricks.sdk import WorkspaceClient

ws = WorkspaceClient()
run = ws.jobs.get_run(RUN_ID)
print(run.state.life_cycle_state, run.state.result_state, run.state.state_message)
for task in run.tasks or []:
    out = ws.jobs.get_run_output(task.run_id)       # output is per task run
    print(task.task_key, out.error)
    print(out.error_trace)
```

Read the traceback before you diagnose. Without the run id, `ws.jobs.list_runs(job_id=...)` lists
a saved job's runs; a one-time run is found from its page or its id as printed at submission.

| Signal | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'kumo'` | Kumo is not present on that cluster: a different cluster from the one the notebook uses, or one an administrator has not set up. That is for the administrator, not something to add to the job |
| `PERMISSION_DENIED` on a table, the warehouse or the model | The run's identity lacks a grant (§5). Name the object |
| `life_cycle_state` `PENDING` for minutes | The cluster is starting. Not a failure |
| `INTERNAL_ERROR`, or a cluster that failed to start | Platform-side. Report the `state_message` |

Propose the specific fix. Never resubmit on your own: every submission is a new GPU run. Never
delete a run or a job, because that discards its output.

## 8. Recurring runs

Never create a schedule on your own initiative. A request for recurring runs gets a proposal. The
schedule is created only after the user explicitly confirms that proposal, because a scheduled job
is a persistent object that spends GPU time on every run until someone pauses or deletes it.

A schedule is a saved job with a cron trigger. Create it paused, and unpause only on the user's
explicit go-ahead:

```python
job = ws.jobs.create(
    name=JOB_NAME,
    tasks=[jobs.Task(task_key='score', existing_cluster_id=CLUSTER_ID,
                     notebook_task=jobs.NotebookTask(notebook_path=JOB_NOTEBOOK))],
    schedule=jobs.CronSchedule(quartz_cron_expression='0 0 6 * * ?', timezone_id='UTC',
                               pause_status=jobs.PauseStatus.PAUSED),
    max_concurrent_runs=1, timeout_seconds=JOB_TIMEOUT_SECONDS)
print('job id:', job.job_id)
```

The cron is Quartz: seconds first (`0 0 6 * * ?` is every day at 06:00). Settle each of these in
the proposal before anything is created:

- **Scoring time moves forward.** The job notebook omits `prediction_time`, so each run anchors at
  the latest data. A date fixed in the notebook would score the same day forever.
- **Write mode for repeated runs.** OVERWRITE keeps only the latest run. APPEND keeps history, but
  a retried run adds a second copy of its rows. If the user already chose a mode, restate it with
  that consequence. Otherwise ask which they want.
- **The model version stays pinned.** A new version reaches the schedule only through an explicit
  change to the job notebook.
- **Who it runs as.** A saved job runs as its owner by default; that identity needs every grant
  the notebook uses.
- **Cost recurs.** Each run pays for cluster start-up, scoring and the cluster's idle time (§9).

Pausing (`pause_status=PAUSED`) or deleting the job is the user's call.

> Not run in this workspace: a scheduled job of this skill's notebook. Say so when proposing it.

## 9. Cost and latency

Do not state a cost figure as fact. Explain what a run is billed for and offer to measure it.

- **Cluster time** in DBUs for the cluster's node type and runtime, plus the cloud provider's
  instance cost, at the user's rates.
- **Start-up when the cluster is terminated.** A terminated cluster starts before the run; a GPU
  node typically takes several minutes. Not measured for this skill.
- **The idle tail.** After the run, an all-purpose cluster stays up until its auto-termination
  setting (`c.autotermination_minutes`).
- **SQL warehouse time** for the reads, writes and read-back the notebook issues.

After a run, cost can be read from `system.billing.usage` for the cluster, which lags by hours.
