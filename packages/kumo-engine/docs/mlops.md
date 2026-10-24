# MLOps with Kumo Engine

## Turning MLOps on

`kumo.MLOps` says where artifacts go and what records them. Use it as a block, which notebooks and scheduled scripts alike can do, or pass it as `mlops=` on a single call:

```python
import kumo.engine as kumo

cfg = kumo.MLOps(artifact_root="s3://bucket/kumo")   # or (connector, path), e.g. a volume or stage
with kumo.mlops(cfg):
    snapshot = graph.snapshot()
```

- **`artifact_root`** is required. It is a URI (`s3://`, `gs://`, `file://`, or a local path, including `/Volumes/...` on a Databricks cluster) or a `(connector, path)` pair for a store a URI cannot reach.
- **`export`** names the artifact kinds stored under the root when they are produced: `snapshot`, `training_table`, `model` and `prediction` (default: all).
- **`tracking`** and **`registry`** record fits and prediction runs and register models.
- **Precedence:** a call's `mlops=` replaces the enclosing block, and an inner block replaces an outer one. A block covers the calling thread or asyncio task.
- **Off by default.** Without a block or `mlops=`, every call behaves as it always has. Plain exports (`export(config)`, `export_config=`, `output_config=`) never write to the artifact root.

## Lineage

Every artifact carries a `_provenance.json` manifest, metadata only. The leading underscore keeps Spark and Hive Parquet readers from treating it as data.

A graph snapshot's manifest, next to its index, records each source table's connector, location and version, plus the engine build that wrote it. `graph.snapshot(record_sources=False)` keeps only connector and table names, and keeps Custom SQL only as a hash.

### Training tables

A generated or caller-supplied training table carries `_provenance.json` in its staging directory:
- the query, the effective generation plan and the task type;
- the split seed;
- the snapshot it was generated from (that snapshot's manifest, embedded);
- for `TrainingTable.from_source_table`, the source table's connector, location and version.

`TrainingTable.id` is deterministic: the same snapshot, query, plan, split seed and engine build give the same id. A caller-supplied source without a version gets a random id. An export to files (`TrainingTable.export`) writes the manifest next to the Parquet parts. `TrainingTable.load_from_connector(connector, path=..., pq=..., snapshot=...)` materializes an export again.

With MLOps on, each table is stored once under `<root>/training_tables/<id>/`.

### Models

Every fitted bundle carries `_provenance.json` (`ModelManifest`):
- its `id`, the bundle id (a content hash that leaves the manifest file out);
- when it was fitted and with which fit mode;
- its parents: the training table's manifest, embedded with that table's snapshot, and the warm-start model's manifest, if any.

An embedded model keeps only the ids of its own model parents, so a long chain of warm-start retrains does not grow each manifest. `fit` returns a new `TrainingResult` per call (`.id`, `.manifest`); `Trainer.result` and `Trainer.manifest` describe the latest fit or the loaded bundle.

With MLOps on, each bundle is stored under `<root>/models/<bundle id>/`. A complete copy that is already there is kept; a partial one is replaced.

### Moving snapshots between machines

To build an index on one machine and train or score on another, export it to a file destination and load it there:

```python
from kumo.engine import GraphSnapshot, S3Connector, SnapshotOutputConfig

location = graph.snapshot().export(SnapshotOutputConfig(S3Connector(), "s3://bucket/snapshots"))
# On the training machine:
snapshot = GraphSnapshot.load_from_connector(S3Connector(), path=location)
```

- **Layout:** `export` writes `<path>/<graph id>/<data version>/`, the two halves of `snapshot.id`, and returns that location.
- **Transfer:** files move in parallel with a CRC32 per range. An interrupted export resumes, and a load either completes or leaves nothing.
- **Reuse:** loading again reuses the local copy, unless that copy fails DiskGraph's integrity check; then it is downloaded again.
- **Manifest:** the snapshot's `_provenance.json` travels with the export.
- **Trust:** load only from paths whose writers you trust. Checksums detect corruption, not tampering.
- **Destinations:** any destination whose sink holds files at the path works: S3, GCS, local files, or a volume or stage once those sinks are installed. A warehouse connector raises.

With MLOps on (`graph.snapshot(mlops=...)` or a `kumo.mlops` block), each snapshot is exported under `<root>/snapshots/`, unless a complete export of it is already there.

### Scoring without the graph

A scoring job does not need the training job's `Graph` code. It loads the snapshot the training job exported and the model, then scores:

```python
snapshot = GraphSnapshot.load_from_connector(store, path=location)
trainer = Trainer.load_from_connector(store, path="models/<bundle id>")
result = trainer.predict(snapshot, binary_classification_threshold=0.5)
```

- `predict(snapshot)` reuses the trained query and table-generation plan, and scores its population at `prediction_time`.
- `entities=` scores given rows instead; `table=` scores a prediction table generated beforehand.
- It warns when the snapshot was taken of a different graph than the one the model was trained on.
- `predict(prediction_table, snapshot=...)` keeps working as before.
- `fit` results score too: `result.predict(snapshot, ...)`.
- `PredictiveQuery.from_snapshot(snapshot, query)` binds a query to a snapshot without its `Graph`. Its graph check only means the snapshot belongs to the query's graph, not that it is compatible with a trained model.

## Progress events

`fit(callbacks=[...])` calls each callback with every progress
`kumo.engine.tracking.Event`: `fit_started`, then per search trial
`trial_started`, `epoch_ended` and `trial_ended`, then `fit_ended`. Ended events
carry `params["status"]` and, on failure, a short `params["error"]`. A callback
that raises is logged and not called again; it never fails the fit. For JSON
lines on stdout:

```python
import json

trainer.fit(training_table, snapshot=snapshot,
            callbacks=[lambda e: print(json.dumps(e.to_dict()))])
```
