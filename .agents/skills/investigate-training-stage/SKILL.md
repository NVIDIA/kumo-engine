---
name: investigate-training-stage
description: Use when a training or prediction job is stuck or slow at a particular stage, or its Temporal workflow status needs investigation.
---

# Investigate Training Stage

## When to Use

Use this skill when the user:
- Reports a training or prediction job is "stuck" or "slow"
- Asks about a specific stage (graph refresh, materialization, etc.)
- Needs to understand Temporal workflow status

## Arguments

- Description of the stuck stage (e.g., "data materialization", "graph refresh")
- Canonical job ID and available workflow history or log evidence
- For live logs: the authorized context, namespace, and workload used by `check-logs`

Start from supplied evidence. Cluster or Temporal access is optional and must
refer to a deployment selected for this task. Reuse authorization already given;
missing access does not prevent source analysis. Distinguish a Kubernetes
namespace from a Temporal namespace, and do not infer either from a tenant name.

---

## Temporal Workflow Hierarchy

### Training Jobs

```
Parent: training-wf-{training_job_id}
│
├── 1. GRAPH REFRESH (graph snapshot)
│   └── GraphRefreshWorkflow (read the actual child ID from history)
│       ├── refresh-{table.wf_key()}@ver={version}
│       └── dg-ingest:{graph_key}@ver={version}[:stats-v2]
│
├── 2. TRAINING TABLE GEN (status: "Generating Training Data")
│   └── {generate_train_table_job_id}-wf
│       └── SnapshotActivities.gen_tt activity
│
├── 3. DATA MATERIALIZATION (status: "Data Materialization")
│   └── materialize-training-task-workflow-{job_id}
│       └── materialize_training_task activity
│
└── 4. TRAINING (status: "Training")
    └── training-job-workflow-{job_id}
        └── SnapshotActivities.train activity
```

### Batch Prediction Jobs

```
Parent: batch-prediction-wf-{prediction_job_id}
├── 1. GRAPH REFRESH (same as training)
├── 2. PREDICTION TABLE GEN → {generate_pred_table_job_id}-wf
├── 3. DATA MATERIALIZATION → materialize-prediction-task-workflow-{job_id}
└── 4. BATCH PREDICTION → batch-prediction-job-workflow-{job_id}
```

---

## Workflow ID Patterns

`job_id` means the full canonical job ID, including its type prefix. Table-gen
job IDs are separate fields and their workflow IDs append `-wf`. An existing
table-generation workflow may be reused rather than started as a child.

Graph refresh IDs composed explicitly by the backend use
`refresh-graph:name={graph_key}@ts={timestamp}` or `@ver={version}`. The training
parent currently starts its graph-refresh child without an explicit ID, so
inspect workflow history instead of assuming every graph child has that prefix.

| Stage | Workflow Pattern | Check Activities |
|----------|------------------|------------------|
| Refreshing graph snapshot | Actual `GraphRefreshWorkflow` child ID | table ingestion, DiskGraph ingestion/column stats |
| Generating Training Data | `{generate_train_table_job_id}-wf` | SnapshotActivities.gen_tt |
| Data Materialization | `materialize-*-task-workflow-{job_id}` | materialize_training_task / materialize_prediction_task |
| Training | `training-job-workflow-{job_id}` | SnapshotActivities.train |

---

## Investigation Steps

1. **Map stage** to workflow pattern from hierarchy above

2. **Read supplied logs or query the selected deployment** using the
   [check-logs skill](../check-logs/SKILL.md):
   ```
   /check-logs context={context} namespace={namespace} selector=app=mlworkflow pattern="{job_id}"
   ```
   Follow the actual workflow and child IDs in the evidence. Use `app=databackend`
   for graph refresh orchestration and the relevant pool worker for activities.

3. **Check stuck activities**:
   - Graph Refresh: table refresh and `dg-ingest:*` workflows
   - Training Table: `SnapshotActivities.gen_tt`
   - Materialization: `SnapshotActivities.materialize_training_task` or `materialize_prediction_task`, correlated by canonical job ID
   - Training: `SnapshotActivities.train`; prediction uses `SnapshotActivities.predict`

4. **Possible causes to investigate** (not diagnoses):
   - **Graph Refresh**: Table ingestion and DiskGraph column stats on large tables
   - **Training Table Gen**: Complex PQL, large target generation
   - **Materialization**: Pool activity scheduling, snapshot availability, and task materialization
   - **Training**: Check the assigned pool worker logs, activity retries, and heartbeats

Report observations separately from hypotheses. A heartbeat timeout or retry count
does not establish PQL complexity, worker starvation, or the reason for an earlier
retry. Use attempt-specific history and worker logs to test those explanations;
state what is missing when the supplied evidence cannot distinguish them.

---

## Key Source Files

| Component | File |
|-----------|------|
| Parent workflow IDs | `rest/backends/ml_backend/ml_job_backend.py` (`_training_workflow_id`, `_batch_prediction_workflow_id`) |
| Training workflow | `rest/backends/ml_backend/workflows/training_workflow.py` |
| Graph refresh / table generation dispatch | `rest/backends/ml_backend/workflows/base_training_workflow.py` |
| Table-generation workflow IDs | `statedb/config/common.py` (`GenerateTrainTableJobID.wf_id`, `GeneratePredictionTableJobID.wf_id`) |
| Graph and table refresh | `rest/backends/data_backend/workflows/graph_refresh_workflow.py`, `table_refresh_workflow.py` |
| DiskGraph ingest | `rest/backends/data_backend/workflows/diskgraph_ingest_workflow.py` |
| Training / prediction dispatch | `rest/backends/ml_backend/workflows/training_job_workflow.py`, `batch_prediction_job_workflow.py` |
| Materialization | `rest/backends/ml_backend/workflows/materialize_task_workflows.py` |
| Pool activities | `pool/activities.py` |
| Materialize activity | `rest/backends/ml_backend/activities/materialize_training_task.py` |
