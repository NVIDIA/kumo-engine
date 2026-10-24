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
