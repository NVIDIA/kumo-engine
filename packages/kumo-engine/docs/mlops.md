# MLOps with Kumo Engine

## Quickstart

With a `graph` and a `pquery` (see the README), turn MLOps on with an artifact root and every snapshot, training table and model is stored under it:

```python
import kumo.engine as kumo

cfg = kumo.MLOps("artifacts")                 # or s3://..., gs://..., /Volumes/..., (connector, "@DB.SCHEMA.STAGE/kumo")
with kumo.mlops(cfg):
    snapshot = graph.snapshot()
    result = kumo.Trainer(pquery.suggest_model_plan(snapshot)).fit(pquery.generate_training_table(snapshot),
        snapshot=snapshot)
```

## How it fits together

- The **artifact root** holds every artifact a block produces, each in its own directory with a `_provenance.json` manifest that embeds its parents' manifests.

| id | names | where to find it |
|---|---|---|
| `snapshot.id` | a graph snapshot | `<root>/snapshots/<graph id>/<data version>/`; `result.snapshot_id` |
| `table.id` | a training table | `<root>/training_tables/<id>/` |
| `result.id` | a model (its bundle id) | `<root>/models/<id>/` |

## Turning MLOps on

`kumo.MLOps` says where artifacts go. Use it as a block, or pass it as `mlops=` on a single call (`fit`, `graph.snapshot`, `generate_training_table`):

```python
cfg = kumo.MLOps("s3://bucket/kumo", on_error="raise")
with kumo.mlops(cfg):
    snapshot = graph.snapshot()
trainer.fit(table, snapshot=snapshot, mlops=cfg)
```

- **`artifact_root`** (required): a URI (`s3://`, `gs://`, `file://`, or a local path, including `/Volumes/...` on a Databricks cluster) or a `(connector, path)` pair for a store a URI cannot reach: a volume through a `DatabricksConnector` (`"/Volumes/<catalog>/<schema>/<volume>/kumo"`) or a named stage through a `SnowflakeConnector` (`"@DB.SCHEMA.STAGE/kumo"`) once the installed sink holds files there.
- **`store`**: the artifact kinds stored when produced: `snapshot`, `training_table` and `model` (default: all).
- **`on_error`**: `"warn"` (default) warns about a failed store and returns; `"raise"`, for scheduled jobs, raises once the call has finished, with the failure as the `__cause__`. After `fit`, `trainer.result` still holds the model.
- **Precedence:** a call's `mlops=` replaces the enclosing block, and an inner block replaces an outer one. A block covers the calling thread or asyncio task.
- **Off by default.** Without a block or `mlops=`, every call behaves as it always has; plain exports (`export(config)`, `export_config=`) never write to the artifact root.
- **Per call, not per model.** MLOps applies only to a call given `mlops=` or run inside `with kumo.mlops(...)`. A later call outside the block, even on a model trained inside one, runs without MLOps, and `on_error="raise"` does not detect this: it covers only MLOps work that ran. There is no per-call off switch; to run one call without MLOps, call it outside the block. For scheduled jobs, wrap the whole job in one block.

### Reading artifacts back

Pass the `kumo.MLOps` where a connector goes, with paths relative to the artifact root:

```python
snapshot = kumo.GraphSnapshot.load_from_connector(cfg, id=result.snapshot_id)
trainer = kumo.Trainer.load_from_connector(cfg, path=f"models/{result.id}")
table = kumo.TrainingTable.load_from_connector(cfg, path=f"training_tables/{table_id}", pquery=pquery, snapshot=snapshot)
```

A loaded model is a `Trainer`; its `result` is the `TrainingResult` (`.id`, `.manifest`, `.metrics`, `.holdout_df()`).

## Score

`predict(snapshot)` scores the trained query's population at its prediction time; `entities=` scores given rows; `predict(prediction_table, snapshot=...)` scores a table generated beforehand. A scoring job needs no `Graph` code: `GraphSnapshot.load_from_connector(cfg, id=...)` and `Trainer.load_from_connector(cfg, ...)`, then `predict`. It warns when the snapshot was taken of a different graph than the model's, naming the schema changes.

## Reference

### Exporting and loading artifacts

Every artifact has one way out and one way back. A `kumo.MLOps` stands in for the connector to read the artifact root.

| Artifact | Export | Load |
|---|---|---|
| Snapshot | `snapshot.export(SnapshotOutputConfig(connector, path))`, returns the location | `GraphSnapshot.load_from_connector(connector, path=location)`, `(cfg, id=...)` |
| Training table | `table.export(TrainingTableOutputConfig(connector, name))` | `TrainingTable.load_from_connector(connector, path=name, pquery=pquery, snapshot=snapshot)` |
| Model | `result.export(ModelOutputConfig(connector, path=... or location=...))`, or `fit(export_config=...)` | `Trainer.load_from_connector(connector, path=... or location=...)` |

Loading needs a file destination; a table export is for consumers outside kumo-engine. Exporting again to a destination that holds a complete copy of the same artifact reuses it (logged at info). `ModelOutputConfig(connector, location=...)` publishes to a connector's registry outside a block.

### Lineage and manifests

Every artifact carries `_provenance.json`, metadata only (the underscore keeps Spark and Hive readers off it). Parents are embedded; a parent without a manifest (a bundle from an older release) is an id stub.

- **Snapshot:** each source table's connector, location and version, and the engine build. `graph.snapshot(record_sources=False)` keeps only connector and table names, a SHA-256 of each location, and Custom SQL as a hash.
- **Training table:** the query, generation plan, task type and split seed, its snapshot, and for `TrainingTable.from_source_table` the source table. `table.id` is deterministic over snapshot, query, plan, seed and engine build; a source without a version gets a random id.
- **Model** (`ModelManifest`, `result.manifest`): the bundle id (a content hash excluding the manifest), fit times and fit mode, and the training table and warm-start model (an embedded model keeps only ids of its own model parents). The bundle carries the holdout predictions (`holdout.parquet`), so `holdout_df()` works on a loaded model.

### Moving snapshots between machines

- **Layout:** `export` writes `<path>/<graph id>/<data version>/`, holding `kumo_snapshot.json` and one directory per export generation (`index/` and `_provenance.json`).
- **Transfer:** parallel, with a CRC32 per range. Each export commits by writing `kumo_snapshot.json` last, so the previous export stays loadable and concurrent exports never mix: the last commit wins. Superseded generations are removed best-effort. A load completes or leaves nothing, and follows a newer export that commits meanwhile.
- **Reuse:** a load reuses the local copy only if it holds the export's bytes and passes DiskGraph's integrity check; an export skips the upload when the destination holds a complete export with the same bytes.
- **Staleness:** `snapshot.source_changes(graph)` reports each table as `unchanged`, `changed` or `unknown`; `load_from_connector(..., graph=graph)` warns about another graph's schema or changed tables.
- **Trust:** load only from paths whose writers you trust; checksums detect corruption, not tampering. A warehouse connector cannot hold a snapshot.
