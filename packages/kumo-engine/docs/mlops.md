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
