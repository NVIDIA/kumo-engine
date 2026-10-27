# MLOps with Kumo Engine

## Turning MLOps on

`kumo.MLOps` says where artifacts go. Use it as a block, which notebooks and scheduled scripts alike can do, or pass it as `mlops=` on a single call:

```python
import kumo.engine as kumo

cfg = kumo.MLOps(artifact_root="s3://bucket/kumo")   # or (connector, path), e.g. a volume or stage
with kumo.mlops(cfg):
    snapshot = graph.snapshot()
```

- **`artifact_root`** is required. It is a URI (`s3://`, `gs://`, `file://`, or a local path, including `/Volumes/...` on a Databricks cluster) or a `(connector, path)` pair for a store a URI cannot reach.
- **Volumes and stages:** `(DatabricksConnector(...), "/Volumes/<catalog>/<schema>/<volume>/kumo")` and `(SnowflakeConnector(...), "@DB.SCHEMA.STAGE/kumo")` work as roots once the installed kumo-databricks or kumo-snowflake sink holds files at those paths; with an older plugin, building `kumo.MLOps` raises.
- **`export`** names the artifact kinds stored under the root when they are produced: `snapshot`, `training_table` and `model` (default: all).
- **Precedence:** a call's `mlops=` replaces the enclosing block, and an inner block replaces an outer one. A block covers the calling thread or asyncio task.
- **Off by default.** Without a block or `mlops=`, every call behaves as it always has. Plain exports (`export(config)`, `export_config=`, `output_config=`) never write to the artifact root. A path export that finds a complete copy of the same artifact at its destination reuses it instead of uploading again (logged at info): a snapshot with the same bytes, a training table or prediction directory with `_SUCCESS` and the same `_KUMO_EXPORT` identity, or a model bundle with the same files and `_provenance.json`.

## Lineage

Every artifact carries a `_provenance.json` manifest, metadata only. The leading underscore keeps Spark and Hive Parquet readers from treating it as data.

A graph snapshot's manifest, next to its index, records each source table's connector, location and version, plus the engine build that wrote it. `graph.snapshot(record_sources=False)` keeps only connector and table names, plus a SHA-256 of each full location so a source moved elsewhere is still noticed, and keeps Custom SQL only as a hash.

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

An embedded model keeps only the ids of its own model parents, so a long chain of warm-start retrains does not grow each manifest. `fit` returns a new `TrainingResult` per call (`.id`, `.manifest`); `Trainer.result` is the latest fit's result or the loaded bundle's (`trainer.result.id`, `trainer.result.manifest`).

With MLOps on, each bundle is stored under `<root>/models/<bundle id>/`. A complete copy that is already there is kept; a partial one is replaced.

The bundle also carries the published model's holdout predictions (`holdout.parquet`, with its truncation flag), so `result.holdout_df()` works on a model loaded with `Trainer.load` or `Trainer.load_from_connector` (a path or a registry), not only on the result of `fit`. Bundles fitted before this have no copy, and `holdout_df()` raises `FileNotFoundError` for them.

### Moving snapshots between machines

To build an index on one machine and train or score on another, export it to a file destination and load it there:

```python
from kumo.engine import GraphSnapshot, S3Connector, SnapshotOutputConfig

location = graph.snapshot().export(SnapshotOutputConfig(S3Connector(), "s3://bucket/snapshots"))
# On the training machine:
snapshot = GraphSnapshot.load_from_connector(S3Connector(), path=location)
```

- **Layout:** `export` writes `<path>/<graph id>/<data version>/`, the two halves of `snapshot.id`, and returns that location. It holds `kumo_snapshot.json` and one directory per export generation (`index/` and `_provenance.json`).
- **Transfer:** files move in parallel with a CRC32 per range. Each export uploads into a new generation and commits by writing `kumo_snapshot.json` last, naming that generation. Until then the previous export stays loadable, and concurrent exports never mix files: the last commit wins whole. Superseded generations, and abandoned ones over a day old, are removed best-effort. A load either completes or leaves nothing; if a newer export commits while it downloads, it switches to that one.
- **Reuse:** loading again reuses the local copy only if it holds this export's bytes (the same range checksums) and passes DiskGraph's integrity check; otherwise it is downloaded again. Exporting again skips the upload when the destination already holds a complete export (its `kumo_snapshot.json` and `_DONE`) with the local index's bytes, and logs that it reused it.
- **Staleness:** `snapshot.source_changes(graph)` reports, per table of `graph`, whether its source is `unchanged`, `changed` or `unknown` since the snapshot was built. `load_from_connector(..., graph=graph)` warns when the snapshot was taken of another graph, naming the schema changes (tables, columns, types, keys, edges), or when any of its tables changed. A graph id also covers each table's connector and location, so a graph that differs only there is logged at info level, not warned about.
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
- `entities=` scores given rows instead.
- It warns when the snapshot was taken of a different graph than the one the model was trained on, naming the schema changes, for example `Schema changes: events edges.` A graph that differs only in table connectors or locations (a dev graph and a prod graph of the same schema) is logged at info level instead. The model records its training graph from this release on; for an older model the warning cannot say what differs.
- `predict(prediction_table, snapshot=...)` scores a prediction table generated beforehand, as before.
- `fit` results score too: `result.predict(snapshot, ...)`.
