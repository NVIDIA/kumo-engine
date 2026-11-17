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

- **Score a pinned model** (`batch_inference`). The model is a Unity Catalog registered model,
  and the job notebook names an explicit integer version (`runtime.md` §5.2). Intake is
  `kumo-data-science/accept/batch_inference.md`; the load, rebuild and predict code is
  `kumo-model-persistence/references/reuse.md`. The notebook contains no `fit` and no training
  table. Ask for the model if the user has not named it, and for the version only as Node 1a
  says: when the model has more than one loadable version and none was named. A single version
  is restated as `[inferred]` and pinned. Never train in its place, and never let the version
  default: `version=None` loads whichever Kumo version is newest when the run starts, so a later
  publication silently changes what a scheduled job scores.
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
confirmation. Writing the job notebook and the submission cell is Step 2, also after this
confirmation (`adapter.md` §0). A request that already says "write a job" or names the cluster to
run on is not that confirmation: reply with the restatement and stop.

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
| The user asks for a new job cluster ("let it spin up a new cluster") | A job cluster is a cluster created for the run, with a default runtime that has no GPU and no Kumo. Say that this skill runs jobs on an existing GPU cluster, which has both — a correction, not a refusal — propose the GPU cluster from the rows above, and ask to confirm submitting there. Never submit a run that creates its own cluster |

Report the cluster's `state`. A `TERMINATED` cluster starts when the run is submitted (§9).

## 5. The job notebook

The job notebook runs with no one watching: no widgets, no display, no user. Every value it uses
is a literal in its config cell; it shares nothing with the interactive notebook, so it rebuilds
the graph itself. For scoring a pinned model, its cells are:

1. The config cell (`runtime.md` §1), with the literals from the Node 1d restatement:
   `CATALOG`, `SCHEMA`, `WAREHOUSE_ID`, `REGISTRY_CATALOG`, `REGISTRY_SCHEMA`, `MODEL_NAME`,
   `MODEL_VERSION` (an integer as a string), `OUTPUT_CATALOG`, `OUTPUT_SCHEMA`, `OUTPUT_TABLE`,
   `WRITE_MODE`, the scoring options, and `TRAINED_QUERY`.
2. The connect step and `run_sql` (`runtime.md` §2, §3), and the registry connector (§5).
3. `Trainer.load_from_connector` with the explicit version (`runtime.md` §5.2).
4. The bundle's `graph_config` tables, column lists and edges rebuilt into `graph`
   (`kumo-model-persistence/references/reuse.md`), then `PredictiveQuery`, `snapshot`, and
   `generate_prediction_table(snapshot, plan=loaded.training_table_generation_plan)` — add
   `prediction_time=` only if the user gave a scoring time.
5. `loaded.predict(...)` with the task's arguments and the `runtime.md` §4 output config, then the
   read-back assertion.

The `predict` arguments are the scoring options confirmed at Node 1d, by the trained task type,
as the table in `kumo-data-science/accept/batch_inference.md` (*scoring options by task type*)
gives them: a threshold (`binary_classification_threshold`) and a count of results per entity
(`num_classes_to_return`) wherever that table allows one.

- `DatabricksIdentity()` in a job run takes the credentials of the user the run runs as — the
  submitter for `runs/submit`, the job's owner or `run_as` for a scheduled job.
- Everything on the cluster's local disk is gone when the cluster terminates. What survives is
  what the run wrote out: the output table, and a published model or a retained archive if the
  job notebook makes one (`runtime.md` §5, §6).
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
  still running. Cancel (`ws.jobs.cancel_run(run_id)`) only after the user confirms.
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

Propose the specific fix: say what to change in the job notebook and ask only whether to make
that change ("Shall I set `MODEL_VERSION = 1` in the job notebook?"). A yes edits that job
notebook, which this skill generated; it is the one edit of an existing notebook the user can
approve here. Resubmitting is a separate confirmation, asked after the edit; never "then rerun".
Tell a repair from a change first. Correcting `MODEL_VERSION` to the version the user already
confirmed (a typo, a stale value) is a repair: the edit above, then the resubmission ask. Moving
it to a different version is a new scoring request: re-read that version's bundle (`adapter.md` §6.1), show the `adapter.md` §6.1 restatement for it, and resubmit only after
the user confirms that restatement.
Never edit the notebook or resubmit on your own: every submission is a new GPU run. Never delete a run or a job, because
that discards its output.

**"Does the cluster have Kumo? Do I need to add libraries?"** Kumo is present on the GPU cluster
the notebook runs on (`adapter.md`); the job adds no libraries and no requirements. An empty
cluster library list is expected: Kumo is part of the cluster's environment, not a job library.
Answer that, and point to the `ModuleNotFoundError` row above for what a run on a cluster without
it would show.

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
- **Write mode for repeated runs.** OVERWRITE keeps only the latest run. APPEND keeps history:
  each row carries `PREDICTION_ID`, `BUNDLE_ID`, `JOB_TIMESTAMP` and `ANCHOR_TIMESTAMP`, and a
  retried run adds a second copy of its rows under a second `JOB_TIMESTAMP`. An existing table
  without one of those columns fails the write after scoring; settle it at the gate
  (`kumoai-reference/references/outputs.md` *Write predictions*). If the user already chose a
  mode, restate it with that consequence. Otherwise ask which they want.
- **The model version stays pinned.** A new version reaches the schedule only through an explicit
  change to the job notebook.
- **Who it runs as.** A saved job runs as its owner by default; that identity needs every grant
  the notebook uses.
- **Cost recurs.** Each run pays for cluster start-up, scoring and the cluster's idle time (§9).

Deleting the job is the user's call.

A saved job already in the workspace for this scoring — one you find, or one an earlier session
created — gets the same proposal. Changing its schedule, unpausing it or editing its task is
done only after the user confirms that proposal; finding the job is not that confirmation.

### Pausing

A pause the user asks for in plain words ("pause the nightly scoring job") runs at once, with no
confirmation turn, when the request names the job or exactly one job matches. Pausing only stops
future triggers; nothing is lost and unpausing restores it. Set `pause_status=PAUSED` on the
job's existing schedule (or trigger), leaving the rest of it as it is:

```python
j = ws.jobs.get(job_id)                             # a trigger-based job: same, on settings.trigger
prev = j.settings.schedule.pause_status            # UNPAUSED, or already PAUSED
sched = j.settings.schedule
sched.pause_status = jobs.PauseStatus.PAUSED
ws.jobs.update(job_id, new_settings=jobs.JobSettings(schedule=sched))
```

Then report: the job's name and id, its previous state, that it is now paused, and that unpausing
it needs the user's confirmation. If more than one job matches, or none does, list the candidates
and ask; pause nothing. A job you merely find is not a pause request. Everything else about a job
stays confirmed: unpausing, scheduling, editing, repairing a run. Cancelling an active run
discards work, so it is not part of a pause: if the user also asks to stop a running run, pause
the schedule now and ask to confirm cancelling the run.

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
