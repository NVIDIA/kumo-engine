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

`MIPSFaiss` uses PyTorch's `MIPSBruteForce` for CPU tensors on macOS, avoiding
Faiss CPU search and its OpenMP runtime conflict with PyTorch.
Brute-force search materializes the full query-by-key score matrix, so large
CPU workloads need additional memory.
Linux CPU and CUDA tensors retain their Faiss search paths.

For MPS embeddings on Apple Silicon, `MIPSFaiss` uses Faiss's Metal backend.
It requires a Metal-enabled Faiss build, provided by `faiss-cpu==1.14.3` on
macOS ARM64. Keys are copied into the Metal index once; queries pass through
CPU float32 arrays, and scores and indices return to the original MPS device.
Scores retain the input dtype, exclusions are applied on MPS, and query
gradients are computed with PyTorch. This retrieval path does not establish
support for the full training pipeline on MPS.

## Development and validation

Run commands from the repository worktree after installing the normal Kumo development environment with `make install-venv`. `make test-kumo-core` runs the complete package-owned CPU suite, and `make check-kumo-core` runs Flake8's Pyflakes rules across the full package, ty across the migrated capability tree, and package metadata validation. The package remains temporarily excluded from the root YAPF/isort/full-Flake8 formatting policy pending the planned repository-wide Ruff migration.

`make test-kumo-core-consumers` exercises the focused Kumo and Kumo Studio integration surface affected by the package cutover, including training and AutoML, evaluation and explanation, worker/image entry points, and retained application consumers.

`make test-kumo-core-artifact` builds both distribution formats, installs the wheel without the source checkout on `PYTHONPATH`, and runs a small CPU train/evaluate/explain round trip against the installed artifact. It uses the worktree environment's already-resolved native dependencies while ensuring the imported `kumo_core` itself comes from the wheel.

On a CUDA-capable development machine, `make test-kumo-core-gpu` selects the retained device-parametrized package cases. The focused, non-required Kumo Core GPU workflow runs on manual dispatch and, for trusted in-repository PRs, when the existing `ml-algorithms` component label triggers a workflow event or a labeled PR is subsequently updated; bot-applied labels may require that subsequent update because `GITHUB_TOKEN` events do not launch workflows. The retired Triton workflow is not part of this package.

Kumo is not yet a uv workspace. These commands operate on `packages/kumo-core` explicitly and do not rely on workspace discovery.

## Remote batch sampling over Arrow Flight

`diskgraph.SamplingServer` embeds the Rust Flight gateway inside its owning
Python process. Rust calls `diskgraph-runtime` directly and owns
sampling, Arrow encoding, flow control, and cancellation. Python manages the
server handle and consumes Arrow batches through the shared `RuntimeIndex`
wrapper with `sampling_endpoints` selecting its Rust remote backend.

A sampling actor starts the listener over its completed local graph replica:

```python
from pathlib import Path
from kumo_core.diskgraph.sampling_service import SamplingService, SamplingServiceConfig

service = SamplingService(
    SamplingServiceConfig(Path("/local/graph"), host="0.0.0.0", max_threads=2)
)
endpoint = service.endpoint("10.0.0.5")
# Retain service for the actor's lifetime; call service.close() during teardown.
```

The wrapper starts the native listener during construction. Its `health()`
returns native readiness and request/batch counters; its process ID matches the
actor. `close()` cancels waiting streams and joins Rust serving threads without
holding the GIL. Native lifecycle/progress messages use the existing Python
logging bridge and appear in the actor's configured logs.

Before opening epoch streams, the caller must materialize and verify graph and
training-table artifacts on every worker through the artifact distribution path.
Training-table extensions live at the same relative `train_tables/<id>` or
`pred_tables/<id>` path beneath each graph replica. Sampling does not upload or
register artifacts; missing task metadata fails with a preparation error.

Epoch requests carry sampling, projection, seed, and shard settings in a `DoGet`
ticket. Python constructs a native `diskgraph.SamplingRequest` and passes it to
`SamplingClient.open()`; Rust sets the wire identity and serializes the ticket.
Sampling-plan JSON retains the existing native iterator interface. The server reserves a
slot in its bounded output queue before sampling a batch. The Rust remote
backend prefetches one native decoded batch per server and merges them in
round-robin order, with no Python thread pool. gRPC flow control propagates a paused
trainer back to the producer; additional transport buffers and native working
memory are separate from these queue limits. No application credits are sent.
The initial empty Flight schema carries the protocol version and sample/batch
counts in `app_metadata`. Each minibatch starts with a metadata-only FlightData
message carrying its ID, component count and layout. Each component then streams
its own Arrow schema, dictionaries and native record batch; task, graph and
feature components can have different schemas and row counts. Component
application metadata carries only the minibatch and component IDs. `diskgraph.SamplingClient`
handles schema changes in Rust and exposes assembled Arrow objects to Python
through the C Data Interface, without nested IPC streams or Python decoding.
Seed fields and target columns share one record batch; node IDs/seed assignments
and edge endpoints are grouped by channel and type. The layout selects an array
with `column` or a record batch with a half-open `columns` range, preserving the
existing trainer batch structure without target struct columns or `as` markers.
Feature tables and payloads with different row alignment remain separate.
Each logical minibatch is limited to 64 MiB across all component messages.
The native client validates sequence numbers, the received count at Flight EOF, and the
final RPC status; failures propagate without replaying an incomplete epoch. Closing an iterator cancels its
outstanding streams, and data-module teardown closes shared connections. Ordered
RHS catalog scans use one replica to preserve MIPS embedding order.

The standalone `diskgraph-gateway` CLI uses the same Rust server implementation.
Graph preparation belongs to the caller. Use the transport between trusted peers
on a restricted network.


Use `diskgraph.RuntimeIndex(manifest, sampling_endpoints=endpoints)` for remote
sampling or omit the endpoint argument for local sampling. Both modes reuse the
same manifest and option normalization. Rust owns remote connections, replica
sharding, weighted-sidecar path translation and iterator lifetime. Close an epoch
to cancel its prefetch while retaining shared connections; close the index at
job teardown to cancel all epochs and release those connections. A closed index
cannot be reopened. The data module creates a fresh shared handle when needed
after job teardown.


Runtime iterators always provide `close()`. Loader scopes register that method
directly for both backends; local synchronous iterators use a no-op hook and
remote iterators cancel their pending prefetch.
