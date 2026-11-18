"""Pinned, streaming adapters for the original RelBench F1 and H&M databases."""

import json
import re
import shutil
import urllib.request
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import Config
from .generate import TYPES, Budget
from .storage import (
    ColumnDigests,
    canonical_hash,
    file_digest,
    quote,
    safe_path,
    write_json,
)

REPOSITORY = "stanford-star/relbench-v1"
REVISION = "21d7284df2a9621c71f21bc4a1a30b7e8effb8dd"
TASKS = {
    "rel-f1": {"driver-dnf": 30, "driver-position": 60},
    "rel-hm": {"user-churn": 7},
}
ROW_ID = "_fm_source_row_id"


def yaml_read(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "RelBench imports require PyYAML: install fm-dataloader[relbench]"
        ) from exc

    return yaml.safe_load(path.read_text())


def fetch(cache, relative, budget):
    """Reuse only checksum-verified files under the immutable revision namespace."""
    path = safe_path(cache, relative)
    receipt = path.with_suffix(path.suffix + ".sha256")
    if path.exists() and receipt.exists():
        if file_digest(path) != receipt.read_text().strip():
            raise ValueError(f"Corrupt cached source: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{relative}"
    with (
        urllib.request.urlopen(url, timeout=120) as response,
        temporary.open("wb") as stream,
    ):
        while chunk := response.read(8 * 2**20):
            stream.write(chunk)
            budget.check()
    temporary.replace(path)
    receipt.write_text(file_digest(path) + "\n")
    return path


def normalized_type(dtype):
    if pa.types.is_dictionary(dtype):
        return normalized_type(dtype.value_type)
    if pa.types.is_timestamp(dtype):
        return pa.timestamp("us", tz="UTC")
    if pa.types.is_integer(dtype):
        return pa.int64()
    if pa.types.is_floating(dtype):
        return pa.float64()
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return pa.string()
    if pa.types.is_boolean(dtype) or pa.types.is_date32(dtype):
        return dtype
    raise ValueError(f"Unsupported source type: {dtype}")


def imported_schema(table):
    return pa.schema(
        [pa.field(c["name"], TYPES[c["type"]], c["nullable"]) for c in table["columns"]]
    )


def normalize(batch, schema, offset=0):
    arrays = []
    for field in schema:
        if field.name == ROW_ID:
            arrays.append(pa.array(np.arange(offset, offset + len(batch)), pa.int64()))
        else:
            # safe=True rejects precision loss (including sub-microsecond timestamps).
            arrays.append(
                pc.cast(
                    batch.column(batch.schema.get_field_index(field.name)),
                    field.type,
                    safe=True,
                )
            )
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def artifact(root, path, rows=None):
    result = {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}
    if rows is not None:
        result["rows"] = rows
    return result


def import_relbench(root, cache, dataset, config=None):
    """Import the complete database; never independently sample relational tables."""
    from .validate import validate

    if dataset not in TASKS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    cfg = config or Config()
    # Validate resource controls using the existing configuration validator.
    cfg.resolve()
    if cfg.context_rows < 1 or cfg.prediction_rows < 1:
        raise ValueError("Task sample sizes must be positive")
    root, cache = Path(root).resolve(), Path(cache).resolve() / REVISION
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    cache.mkdir(parents=True, exist_ok=True)
    source_budget = Budget(cache, cfg)
    source_budget.check()
    def get(relative):
        return fetch(cache, relative, source_budget)
    source_manifest = get(f"{dataset}/manifest.yaml")
    meta = yaml_read(source_manifest)
    if meta["name"] != dataset or meta["manifest_version"] != 1:
        raise ValueError("Unexpected upstream dataset manifest")
    root.mkdir(parents=True)
    budget = Budget(root, cfg)
    source_dir = root / "source"
    source_dir.mkdir()
    provenance = []
    for relative in ["README.md", f"{dataset}/README.md", f"{dataset}/manifest.yaml"]:
        original = get(relative)
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        provenance.append(artifact(root, destination))
    recipe = {
        "source_kind": "relbench",
        "adapter_version": 1,
        "config": asdict(cfg),
        "dataset": dataset,
        "tables": {},
        "repository": REPOSITORY,
        "revision": REVISION,
        "val_timestamp": meta["val_timestamp"],
        "test_timestamp": meta["test_timestamp"],
    }
    manifest = {
        "format": "fm-dataset-v1",
        "recipe": recipe,
        "tables": {},
        "tasks": {},
        "workloads": [],
        "source_artifacts": provenance,
        "versions": {"pyarrow": pa.__version__},
        "created_utc": datetime.now(UTC).isoformat(),
    }
    type_names = {v: k for k, v in TYPES.items()}
    for name, spec in meta["tables"].items():
        source = get(f"{dataset}/db/{name}.parquet")
        pf = pq.ParquetFile(source)
        key = spec["pkey"] or ROW_ID
        if ROW_ID in pf.schema_arrow.names:
            raise ValueError("Reserved occurrence-ID column already exists")
        fields = [
            pa.field(f.name, normalized_type(f.type), f.name != key)
            for f in pf.schema_arrow
        ]
        if spec["pkey"] is None:
            fields.insert(0, pa.field(ROW_ID, pa.int64(), False))
        schema = pa.schema(fields)
        table = {
            "rows": pf.metadata.num_rows,
            "primary_key": key,
            "time_column": spec["time_col"],
            "key_policy": "source-key"
            if spec["pkey"]
            else "zero-based-pinned-source-row-offset",
            "source": {
                "path": f"{dataset}/db/{name}.parquet",
                "sha256": file_digest(source),
                "schema": pf.schema_arrow.remove_metadata().to_string(),
            },
            "columns": [
                {
                    "name": f.name,
                    "type": type_names[f.type],
                    "nullable": f.nullable,
                    "role": "primary_key"
                    if f.name == key
                    else "timestamp"
                    if f.name == spec["time_col"]
                    else "feature",
                }
                for f in schema
            ],
            "fkeys": [
                {"column": col, "table": parent, "key": meta["tables"][parent]["pkey"]}
                for col, parent in spec["fkeys"].items()
            ],
        }
        recipe["tables"][name] = table
        folder = root / "tables" / name
        folder.mkdir(parents=True)
        digest, inventory, offset, writer, file_count = (
            ColumnDigests(schema),
            [],
            0,
            None,
            0,
        )

        def close_part(writer, path, inventory):
            writer.close()
            m = pq.ParquetFile(path).metadata
            inventory.append(
                {
                    **artifact(root, path),
                    "bytes": path.stat().st_size,
                    "row_groups": [
                        m.row_group(i).num_rows for i in range(m.num_row_groups)
                    ],
                }
            )

        try:
            for batch in pf.iter_batches(batch_size=cfg.batch_rows):
                chunk = normalize(batch, schema, offset)
                digest.update(chunk)
                offset += len(chunk)
                start = 0
                while start < len(chunk):
                    if writer is None:
                        path = folder / f"part-{len(inventory):05d}.parquet"
                        writer = pq.ParquetWriter(path, schema, compression="zstd")
                        file_count = 0
                    count = min(len(chunk) - start, cfg.file_rows - file_count)
                    writer.write_batch(
                        chunk.slice(start, count), row_group_size=cfg.row_group_rows
                    )
                    start += count
                    file_count += count
                    if file_count == cfg.file_rows:
                        close_part(writer, path, inventory)
                        writer = None
                    budget.check()
            if writer is None and not inventory:
                path = folder / "part-00000.parquet"
                writer = pq.ParquetWriter(path, schema, compression="zstd")
        finally:
            if writer is not None:
                close_part(writer, path, inventory)
        manifest["tables"][name] = {
            "rows": offset,
            "schema": schema.to_string(),
            "logical_sha256": digest.hexdigest(),
            "files": inventory,
        }
        print(
            json.dumps({"dataset": dataset, "table": name, "rows": offset}), flush=True
        )
    manifest["logical_id"] = canonical_hash(
        {
            n: {k: t[k] for k in ("rows", "schema", "logical_sha256")}
            for n, t in manifest["tables"].items()
        }
    )
    manifest["snapshot_id"] = canonical_hash(manifest["tables"])
    build_imported_tasks(root, manifest, get, budget)
    write_json(root / "manifest.json", manifest)
    report = validate(root, allow_incomplete=True)
    write_json(root / "validation.json", report)
    budget.check()
    write_json(
        root / "_COMPLETE.json",
        {
            "manifest_sha256": file_digest(root / "manifest.json"),
            "validation_sha256": file_digest(root / "validation.json"),
        },
    )
    return manifest


def task_rows(path, spec, split, limit=None, cutoff=None):
    """Read bounded samples in pinned source order, retaining source row identity."""
    rows, offset = [], 0
    horizon = timedelta(days=spec["horizon_days"])
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
        for row in batch.to_pylist():
            row_id = offset
            offset += 1
            anchor = row[spec["time_col"]]
            anchor = (
                anchor.replace(tzinfo=UTC)
                if anchor.tzinfo is None
                else anchor.astimezone(UTC)
            )
            if cutoff is not None and anchor + horizon > cutoff:
                continue
            target = row[spec["target_col"]]
            rows.append(
                {
                    "instance_id": row_id
                    + {"train": 0, "val": 10**12, "test": 2 * 10**12}[split],
                    "entity_key": row[spec["entity_col"]],
                    "anchor_time": anchor,
                    "target": target,
                    "label_available_time": anchor + horizon,
                }
            )
            if limit is not None and len(rows) == limit:
                return rows
    return rows


def build_imported_tasks(root, manifest, get, budget):
    recipe, cfg = manifest["recipe"], manifest["recipe"]["config"]
    dataset = recipe["dataset"]
    for task, days in TASKS[dataset].items():
        relative = f"{dataset}/tasks/{task}/manifest.yaml"
        original = get(relative)
        spec = yaml_read(original)
        if spec["kind"] != "forecast" or spec["timedelta"] != f"{days} days":
            raise ValueError("Unexpected upstream task horizon/kind")
        spec["horizon_days"] = days
        folder = root / "tasks" / task
        folder.mkdir(parents=True)
        shutil.copyfile(original, folder / "source-manifest.yaml")
        source_spec = artifact(root, folder / "source-manifest.yaml")
        splits = {}
        for split in ("train", "val", "test"):
            source = get(f"{dataset}/tasks/{task}/{split}.parquet")
            destination = folder / f"source-{split}.parquet"
            shutil.copyfile(source, destination)
            splits[split] = artifact(
                root, destination, pq.ParquetFile(destination).metadata.num_rows
            )
            budget.check()
        predictions = task_rows(
            root / splits["val"]["path"], spec, "val", cfg["prediction_rows"]
        )
        cutoff = min(r["anchor_time"] for r in predictions)
        contexts = task_rows(
            root / splits["train"]["path"], spec, "train", cfg["context_rows"], cutoff
        )
        if not contexts or not predictions:
            raise ValueError("No eligible context/prediction instances")
        target_type = pa.float64() if spec["task_type"] == "regression" else pa.int64()
        key = recipe["tables"][spec["entity_table"]]["primary_key"]
        key_type = (
            imported_schema(recipe["tables"][spec["entity_table"]]).field(key).type
        )
        schema = pa.schema(
            [
                pa.field("instance_id", pa.int64(), False),
                pa.field("entity_key", key_type, False),
                pa.field("anchor_time", pa.timestamp("us", tz="UTC"), False),
                pa.field("target", target_type, False),
                pa.field("label_available_time", pa.timestamp("us", tz="UTC"), False),
            ]
        )
        context, prediction = (
            pa.Table.from_pylist(contexts, schema),
            pa.Table.from_pylist(predictions, schema),
        )
        artifacts = {}
        for kind, data in [
            ("context", context),
            (
                "predict",
                prediction.select(["instance_id", "entity_key", "anchor_time"]),
            ),
            ("labels", prediction.select(["instance_id", "target"])),
        ]:
            path = folder / f"{kind}.parquet"
            pq.write_table(data, path, compression="zstd")
            artifacts[kind] = artifact(root, path, len(data))
        manifest["tasks"][task] = {
            "type": "regression" if target_type == pa.float64() else "classification",
            "entity_table": spec["entity_table"],
            "label_interval": f"(anchor, anchor + {days} days]",
            "source_spec": source_spec,
            "source_splits": splits,
            "source_metadata": spec,
            "prediction_split": "val",
            "selection": "first-eligible-pinned-source-rows-v1",
            "artifacts": artifacts,
        }
        for size in sorted(
            {min(n, len(prediction)) for n in (1, 32, cfg["prediction_rows"])}
        ):
            paths = (
                [
                    ["drivers", "results", "races", "circuits"],
                    ["drivers", "results", "constructors"],
                ]
                if dataset == "rel-f1"
                else [["customer", "transactions", "article"]]
            )
            work = {
                "task": task,
                "logical_id": manifest["logical_id"],
                "snapshot_id": manifest["snapshot_id"],
                "cohort": "official-val",
                "context_instance_ids": context["instance_id"].to_pylist(),
                "prediction_instance_ids": prediction["instance_id"].to_pylist()[:size],
                "paths": paths,
                "event_fanout": 16,
                "forward_fk_fanout": 1,
                "lookback_days": cfg["lookback_days"],
                "history_interval": "[anchor-lookback, anchor)",
                "event_order": ["timestamp DESC", "primary_key ASC"],
                "timeless_dimensions": [
                    n for n, t in recipe["tables"].items() if not t["time_column"]
                ],
                "identity_policy": "deduplicate source IDs within each instance; preserve edges and per-instance occurrences",
                "projection": {
                    n: [c["name"] for c in t["columns"]]
                    for n, t in recipe["tables"].items()
                },
                "feature_columns": {
                    n: [
                        c["name"]
                        for c in t["columns"]
                        if c["role"] != "primary_key"
                        and c["name"] not in {r["column"] for r in t["fkeys"]}
                    ]
                    for n, t in recipe["tables"].items()
                },
            }
            work["workload_id"] = canonical_hash(work)
            path = root / "workloads" / f"{task}-official-val-{size}.json"
            path.parent.mkdir(exist_ok=True)
            write_json(path, work)
            manifest["workloads"].append(artifact(root, path))


def validate_recipe(root, manifest):
    recipe = manifest["recipe"]
    if (
        recipe["repository"] != REPOSITORY
        or recipe["revision"] != REVISION
        or recipe["adapter_version"] != 1
    ):
        raise ValueError("Unrecognized RelBench source pin/adapter")
    dataset = recipe["dataset"]
    if dataset not in TASKS or set(manifest["tasks"]) != set(TASKS[dataset]):
        raise ValueError("Unexpected RelBench dataset/tasks")
    for item in manifest["source_artifacts"]:
        if file_digest(safe_path(root, item["path"])) != item["sha256"]:
            raise ValueError("Source metadata checksum mismatch")
    source = yaml_read(root / "source" / dataset / "manifest.yaml")
    if set(source["tables"]) != set(recipe["tables"]):
        raise ValueError("Source table inventory mismatch")
    for name, spec in source["tables"].items():
        table = recipe["tables"][name]
        expected = [
            {"column": col, "table": parent, "key": source["tables"][parent]["pkey"]}
            for col, parent in spec["fkeys"].items()
        ]
        if (
            table["primary_key"] != (spec["pkey"] or ROW_ID)
            or table["time_column"] != spec["time_col"]
            or table["fkeys"] != expected
        ):
            raise ValueError("Source relational metadata mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", table["source"]["sha256"]):
            raise ValueError("Missing source checksum")
    for key in ("val_timestamp", "test_timestamp"):
        if recipe[key] != source[key]:
            raise ValueError("Source split cutoff mismatch")
    return recipe


def validate_tasks(root, manifest, con, report):
    instance_sets = {}
    recipe, cfg = manifest["recipe"], manifest["recipe"]["config"]
    for task, task_spec in manifest["tasks"].items():
        source_spec = task_spec["source_spec"]
        path = safe_path(root, source_spec["path"])
        if file_digest(path) != source_spec["sha256"]:
            raise ValueError("Source task metadata checksum mismatch")
        spec = yaml_read(path)
        spec["horizon_days"] = TASKS[recipe["dataset"]][task]
        if (
            spec != task_spec["source_metadata"]
            or spec["timedelta"] != f"{spec['horizon_days']} days"
        ):
            raise ValueError("Source task specification mismatch")
        if (
            task_spec["entity_table"] != spec["entity_table"]
            or task_spec["prediction_split"] != "val"
        ):
            raise ValueError("Task mapping mismatch")
        if set(task_spec["source_splits"]) != {"train", "val", "test"}:
            raise ValueError("Missing official task splits")
        for split, item in task_spec["source_splits"].items():
            path = safe_path(root, item["path"])
            pf = pq.ParquetFile(path)
            if (
                file_digest(path) != item["sha256"]
                or pf.metadata.num_rows != item["rows"]
            ):
                raise ValueError("Official task split checksum/count mismatch")
            con.read_parquet(str(path)).create_view("official_split", replace=True)
            entity, col, stamp, target = map(
                quote,
                [
                    spec["entity_table"],
                    spec["entity_col"],
                    spec["time_col"],
                    spec["target_col"],
                ],
            )
            pk = quote(recipe["tables"][spec["entity_table"]]["primary_key"])
            missing = con.execute(
                f"SELECT count(*) FROM official_split s ANTI JOIN {entity} e ON s.{col}=e.{pk}"
            ).fetchall()[0][0]
            nulls = con.execute(
                f"SELECT count(*) FROM official_split WHERE {col} IS NULL OR {stamp} IS NULL OR {target} IS NULL"
            ).fetchall()[0][0]
            val, test = recipe["val_timestamp"], recipe["test_timestamp"]
            condition = (
                f"{stamp} >= ?::TIMESTAMP"
                if split == "train"
                else f"{stamp} < ?::TIMESTAMP OR {stamp} >= ?::TIMESTAMP"
                if split == "val"
                else f"{stamp} < ?::TIMESTAMP"
            )
            params = (
                [val] if split == "train" else [val, test] if split == "val" else [test]
            )
            wrong_split = con.execute(
                f"SELECT count(*) FROM official_split WHERE {condition}", params
            ).fetchall()[0][0]
            if missing or nulls or wrong_split:
                raise ValueError(
                    f"Invalid official split entities/labels/cutoffs: {task}/{split}"
                )
        predictions = task_rows(
            safe_path(root, task_spec["source_splits"]["val"]["path"]),
            spec,
            "val",
            cfg["prediction_rows"],
        )
        cutoff = min(r["anchor_time"] for r in predictions)
        contexts = task_rows(
            safe_path(root, task_spec["source_splits"]["train"]["path"]),
            spec,
            "train",
            cfg["context_rows"],
            cutoff,
        )
        expected = {
            "context": contexts,
            "predict": [
                {k: r[k] for k in ("instance_id", "entity_key", "anchor_time")}
                for r in predictions
            ],
            "labels": [
                {k: r[k] for k in ("instance_id", "target")} for r in predictions
            ],
        }
        if set(task_spec["artifacts"]) != set(expected):
            raise ValueError("Missing task artifacts")
        for kind, item in task_spec["artifacts"].items():
            path = safe_path(root, item["path"])
            data = pq.read_table(path)
            if (
                file_digest(path) != item["sha256"]
                or len(data) != item["rows"]
                or data.to_pylist() != expected[kind]
            ):
                raise ValueError(
                    f"Task differs from official labels/selection: {task}/{kind}"
                )
        instance_sets[task] = {
            kind: {r["instance_id"] for r in expected[kind]}
            for kind in ("context", "predict")
        }
        report["tasks"][task] = {
            "source_splits": {
                k: v["rows"] for k, v in task_spec["source_splits"].items()
            },
            "context": len(contexts),
            "predict": len(predictions),
            "horizon_days": spec["horizon_days"],
            "labels_verified_against": "pinned official splits (not recomputed)",
        }
    return instance_sets
