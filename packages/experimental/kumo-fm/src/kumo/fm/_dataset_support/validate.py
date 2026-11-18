"""Streaming content checks plus spillable SQL integrity/label validation."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import Config
from .generate import Budget, schema_for
from .storage import (
    ColumnDigests,
    canonical_hash,
    connect,
    file_digest,
    quote,
    safe_path,
)
from .tasks import label_counts


def validate(root: Path, *, allow_incomplete=False):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "fm-dataset-v1":
        raise ValueError("Unsupported bundle format")
    if not allow_incomplete:
        marker = json.loads((root / "_COMPLETE.json").read_text())
        for name, key in [
            ("manifest.json", "manifest_sha256"),
            ("validation.json", "validation_sha256"),
        ]:
            if file_digest(root / name) != marker[key]:
                raise ValueError(f"Completion checksum mismatch: {name}")
    cfg = Config(**manifest["recipe"]["config"])
    imported = manifest["recipe"].get("source_kind") == "relbench"
    if imported:
        from .relbench import imported_schema, validate_recipe, validate_tasks

        cfg.resolve()
        recipe = validate_recipe(root, manifest)
    else:
        recipe = cfg.resolve()
    if recipe != manifest["recipe"] or set(manifest["tables"]) != set(recipe["tables"]):
        raise ValueError("Recipe/schema mismatch")
    budget = Budget(root, cfg)
    report = {
        "status": "pass",
        "tables": {},
        "tasks": {},
        "logical_id": manifest["logical_id"],
    }
    for name, info in manifest["tables"].items():
        table = recipe["tables"][name]
        schema = imported_schema(table) if imported else schema_for(table, cfg)
        if info["schema"] != schema.to_string():
            raise ValueError(f"Manifest schema mismatch: {name}")
        digests, count, previous = ColumnDigests(schema), 0, None
        nulls = {f.name: 0 for f in schema}
        if not info["files"]:
            raise ValueError(f"Missing source files: {name}")
        for item in info["files"]:
            path = safe_path(root, item["path"])
            if (
                file_digest(path) != item["sha256"]
                or path.stat().st_size != item["bytes"]
            ):
                raise ValueError(f"File checksum/size mismatch: {path}")
            pf = pq.ParquetFile(path)
            groups = [
                pf.metadata.row_group(i).num_rows
                for i in range(pf.metadata.num_row_groups)
            ]
            if groups != item["row_groups"] or any(
                n > cfg.row_group_rows for n in groups
            ):
                raise ValueError(f"Row-group mismatch: {name}")
            if not pf.schema_arrow.equals(schema):
                raise ValueError(f"Schema mismatch: {name}")
            for batch in pf.iter_batches(batch_size=65536):
                count += len(batch)
                digests.update(batch)
                for field, array in zip(schema, batch.columns, strict=True):
                    nulls[field.name] += array.null_count
                    if not field.nullable and array.null_count:
                        raise ValueError(f"Null required value: {name}.{field.name}")
                keys = batch.column(schema.get_field_index(table["primary_key"]))
                if len(keys) and not imported:
                    if previous is not None and keys[0].as_py() <= previous:
                        raise ValueError(f"Primary keys not strictly ordered: {name}")
                    if (
                        len(keys) > 1
                        and pc.any(
                            pc.less_equal(keys.slice(1), keys.slice(0, len(keys) - 1))
                        ).as_py()
                    ):
                        raise ValueError(f"Duplicate/unordered primary keys: {name}")
                    previous = keys[-1].as_py()
        if (
            count != table["rows"]
            or count != info["rows"]
            or digests.hexdigest() != info["logical_sha256"]
        ):
            raise ValueError(f"Row count/logical checksum mismatch: {name}")
        report["tables"][name] = {
            "rows": count,
            "null_counts": nulls,
            "relationships": {},
        }
        budget.check()
    logical = canonical_hash(
        {
            n: {k: t[k] for k in ("rows", "schema", "logical_sha256")}
            for n, t in manifest["tables"].items()
        }
    )
    if (
        logical != manifest["logical_id"]
        or canonical_hash(manifest["tables"]) != manifest["snapshot_id"]
    ):
        raise ValueError("Dataset identity mismatch")
    con = connect(root, manifest)
    try:
        for name, table in recipe["tables"].items():
            if imported:
                pk = quote(table["primary_key"])
                total, unique = con.execute(
                    f"SELECT count(*), count(DISTINCT {pk}) FROM {quote(name)}"
                ).fetchall()[0]
                if total != unique:
                    raise ValueError(f"Duplicate/null primary keys: {name}")
                if (
                    table["key_policy"] == "zero-based-pinned-source-row-offset"
                    and total
                ):
                    low, high = con.execute(
                        f"SELECT min({pk}), max({pk}) FROM {quote(name)}"
                    ).fetchall()[0]
                    if (low, high) != (0, total - 1):
                        raise ValueError("Invalid source occurrence IDs")
            for rel in table["fkeys"]:
                fk = quote(rel["column"])
                missing = con.execute(f"""SELECT count(*) FROM {quote(name)} f
                    ANTI JOIN {quote(rel["table"])} p ON f.{fk}=p.{quote(rel["key"])}
                    WHERE f.{fk} IS NOT NULL""").fetchall()[0][0]
                linked, active = con.execute(
                    f"SELECT count({fk}), count(DISTINCT {fk}) FROM {quote(name)}"
                ).fetchall()[0]
                if (
                    missing
                    or (not imported and linked != rel["linked_rows"])
                    or (not imported and active != rel["active_parents"])
                ):
                    raise ValueError(
                        f"Foreign-key integrity/coverage mismatch: {name}.{rel['column']}"
                    )
                degree = con.execute(f"""SELECT min(n), approx_quantile(n, [0.5,0.9,0.99]), max(n)
                    FROM (SELECT count(*) n FROM {quote(name)} WHERE {fk} IS NOT NULL GROUP BY {fk})""").fetchall()[0]
                report["tables"][name]["relationships"][rel["column"]] = {
                    "linked_rows": linked,
                    "active_parents": active,
                    "missing": missing,
                    "nonzero_degree_min": degree[0],
                    "nonzero_degree_quantiles_50_90_99": degree[1],
                    "nonzero_degree_max": degree[2],
                }
            if table["time_column"]:
                bounds = con.execute(
                    f"SELECT min({quote(table['time_column'])})::VARCHAR, max({quote(table['time_column'])})::VARCHAR FROM {quote(name)}"
                ).fetchall()[0]
                report["tables"][name]["time_range"] = list(bounds)
        if imported:
            instance_sets = validate_tasks(root, manifest, con, report)
        else:
            instance_sets = {}
            if set(manifest["tasks"]) != {"event_binary", "event_count"}:
                raise ValueError("Expected both generated tasks")
            for task, spec in manifest["tasks"].items():
                data = {}
                for kind, item in spec["artifacts"].items():
                    path = safe_path(root, item["path"])
                    if file_digest(path) != item["sha256"]:
                        raise ValueError(f"Task checksum mismatch: {task}/{kind}")
                    data[kind] = pq.read_table(path)
                    if len(data[kind]) != item["rows"]:
                        raise ValueError("Task row count mismatch")
                    ids = data[kind]["instance_id"].to_pylist()
                    if len(set(ids)) != len(ids):
                        raise ValueError("Duplicate task instance IDs")
                if "target" in data["predict"].column_names:
                    raise ValueError("Prediction target leakage")
                for kind in ("context", "predict"):
                    instances = data[kind]
                    if not len(instances) or any(
                        instances[c].null_count for c in instances.column_names
                    ):
                        raise ValueError("Empty task or null instance fields")
                    con.register("validate_instances", instances)
                    entity = recipe["entity_table"]
                    entity_key = recipe["tables"][entity]["primary_key"]
                    missing = con.execute(
                        f"SELECT count(*) FROM validate_instances a ANTI JOIN {quote(entity)} e ON a.entity_key=e.{quote(entity_key)}"
                    ).fetchall()[0][0]
                    con.unregister("validate_instances")
                    if missing:
                        raise ValueError("Task references a missing entity")
                    history_end = datetime.fromtimestamp(
                        recipe["history_start_us"] / 1000000, UTC
                    ) + timedelta(days=cfg.history_days)
                    if any(
                        t + timedelta(days=30) > history_end
                        for t in instances["anchor_time"].to_pylist()
                    ):
                        raise ValueError("Incomplete label observation interval")
                    actual = label_counts(con, manifest, data[kind])[
                        "target"
                    ].to_pylist()
                    expected = data["context" if kind == "context" else "labels"][
                        "target"
                    ].to_pylist()
                    if spec["type"] == "classification":
                        actual = [int(v > 0) for v in actual]
                    if actual != expected:
                        raise ValueError(f"Label mismatch: {task}/{kind}")
                if (
                    data["predict"]["instance_id"].to_pylist()
                    != data["labels"]["instance_id"].to_pylist()
                ):
                    raise ValueError("Prediction/label alignment mismatch")
                if (
                    pc.max(data["context"]["label_available_time"]).as_py()
                    > pc.min(data["predict"]["anchor_time"]).as_py()
                ):
                    raise ValueError("Context labels not available at prediction time")
                if any(
                    available != anchor + timedelta(days=30)
                    for anchor, available in zip(
                        data["context"]["anchor_time"].to_pylist(),
                        data["context"]["label_available_time"].to_pylist(),
                        strict=True,
                    )
                ):
                    raise ValueError("Incorrect context label availability")
                instance_sets[task] = {
                    kind: set(data[kind]["instance_id"].to_pylist())
                    for kind in ("context", "predict")
                }
                report["tasks"][task] = {k: len(v) for k, v in data.items()}
        if not manifest["tasks"] or not manifest["workloads"]:
            raise ValueError("Missing tasks/workloads")
        for item in manifest["workloads"]:
            path = safe_path(root, item["path"])
            if file_digest(path) != item["sha256"]:
                raise ValueError("Workload checksum mismatch")
            work = json.loads(path.read_text())
            identifier = work.pop("workload_id")
            if (
                canonical_hash(work) != identifier
                or work["snapshot_id"] != manifest["snapshot_id"]
                or work["logical_id"] != manifest["logical_id"]
            ):
                raise ValueError("Workload identity mismatch")
            if work["task"] not in instance_sets:
                raise ValueError("Unknown workload task")
            for split in ("context", "predict"):
                key = (
                    "context_instance_ids"
                    if split == "context"
                    else "prediction_instance_ids"
                )
                ids = work[key]
                if (
                    not ids
                    or len(set(ids)) != len(ids)
                    or not set(ids) <= instance_sets[work["task"]][split]
                ):
                    raise ValueError("Workload references missing/duplicate instances")
        budget.check()
    finally:
        con.close()
    return report
