# kumo-core

`kumo-core` is the service-independent Kumo training engine. It contains the original task, sampling, model, Trainer/AutoTrainer, evaluation, explanation, artifact, and prediction implementations without REST, Temporal, StateDB, or Kubernetes runtime dependencies.

There are two supported integration levels:

- parity-preserving modules expose the existing numerical implementation for the Kumo platform and detailed runtime adapters;

- `kumo_core.training.workflow` composes validation, train-table generation, DiskGraph task materialization, default model-plan resolution, and AutoML behind one prepared-graph workflow.

Warehouse packages own only the environment boundaries before and after this workflow: resolving/staging source tables, ingesting the graph snapshot, constructing training bindings, durable storage, and registry publication.

```python
from kumo_core.model_plan import ModelPlan, OptimizationPlan, TrainingJobPlan
from kumo_core.training import CustomTrainingTable
from kumo_core.training.search import AutoMLSearchOptions
from kumo_core.training.workflow import GNNTrainingWorkflow

workflow = GNNTrainingWorkflow(
    connector=environment.connector,
    graph=resolved_graph_definition,
    bindings_factory=environment.training_bindings,
    publisher=environment.model_publisher,
    model_plan=ModelPlan(
        training_job=TrainingJobPlan(metrics=["auroc", "ap"]),
        optimization=OptimizationPlan(max_epochs=10),
    ),
    search=AutoMLSearchOptions(
        run_mode="best",
        num_trials=12,
        seed=42,
    ),
    custom_training_table=CustomTrainingTable(
        "catalog.schema.prebuilt_train_table",
        weight_column="WEIGHT",
    ),
)
result = workflow.train_gnn(query, "KUMO_GNN_CHURN")
```

`CustomTrainingTable` identifies a connector-owned table already shaped with Kumo's `ENTITY` and `TARGET` columns. The connector stages it as parquet and core builds the same DiskGraph train-table artifacts used by generated PQL tables. `AutoMLSearchOptions` controls the original AutoTrainer search; it does not introduce an adapter-specific search loop. Its `num_trials` and `seed` win over the plan's own `training_job.num_experiments` and `manual_seed`, so set the trial count in one place or the other.

The connector stages warehouse tables and prepares DiskGraph. The publisher registers the vital bundle. A remote runtime may wrap `workflow.train_gnn` so the client receives its native asynchronous job handle while the configured core call remains only `query` plus `output_model_name`.

Pass `training_table_generation_plan` separately from `model_plan` to
`GNNTrainingWorkflow`, `prepare_training_request()`, or
`TrainingWorkflowRequest.from_spec()`. The native table planner accepts
this plan directly. Omission from the workflow uses generation defaults;
nondefault settings in the historical internal
`model_plan.training_table_generation` field raise an error directing callers
to the explicit argument. For existing generated rows, pass the table's
saved effective generation settings. Resolved model-plan artifacts retain
the historical section for compatibility.

For a resumable runtime, use `prepare_training_request()`, `build_training_task_plan()`, and `materialize_generated_training_task()` as separate stages. The one-shot and staged forms call the same core functions; adapters must not implement their own task or training behavior.

Materialization accepts the selected graph in `TrainingWorkflowRequest`, a `TaskDatasetMaterializeSpec`, the generated table artifacts, and a per-call `tt_id`. To train existing rows against another graph, build the task-dataset specification with `kumo_core.pquery.build_generated_training_task_dataset()` using that graph and the table's saved generation settings and required split stages. This does not rerun training-table generation or require its original combined plan.

Use `kumo_core.types` for value types, `kumo_core.training` for workflow contracts, and `kumo_core.train` for the numerical stack. These initializers expose deliberate lazy APIs without loading PyTorch during package discovery.

## Development and validation

Run commands from the repository worktree after installing the normal Kumo development environment with `make install-venv`. `make test-kumo-core` runs the complete package-owned CPU suite, and `make check-kumo-core` runs Flake8's Pyflakes rules across the full package, ty across the migrated capability tree, and package metadata validation. The package remains temporarily excluded from the root YAPF/isort/full-Flake8 formatting policy pending the planned repository-wide Ruff migration.

`make test-kumo-core-consumers` exercises the focused Kumo and Kumo Studio integration surface affected by the package cutover, including training and AutoML, evaluation and explanation, worker/image entry points, and retained application consumers.

`make test-kumo-core-artifact` builds both distribution formats, installs the wheel without the source checkout on `PYTHONPATH`, and runs a small CPU train/evaluate/explain round trip against the installed artifact. It uses the worktree environment's already-resolved native dependencies while ensuring the imported `kumo_core` itself comes from the wheel.

On a CUDA-capable development machine, `make test-kumo-core-gpu` selects the retained device-parametrized package cases. The focused, non-required Kumo Core GPU workflow runs on manual dispatch and, for trusted in-repository PRs, when the existing `ml-algorithms` component label triggers a workflow event or a labeled PR is subsequently updated; bot-applied labels may require that subsequent update because `GITHUB_TOKEN` events do not launch workflows. The retired Triton workflow is not part of this package.

Kumo is not yet a uv workspace. These commands operate on `packages/kumo-core` explicitly and do not rely on workspace discovery.
