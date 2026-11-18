"""Batch-independent logical hashes and portable artifact inventories."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes bundle: {relative}")
    return path


class ColumnDigests:
    """Version 1: independent validity/value/length streams in PK source order.

    Separating streams makes the result independent of writer/reader chunk sizes.
    Null values use zero/empty bytes; validity distinguishes null from those values.
    """

    def __init__(self, schema):
        self.schema = schema
        self.streams = [[hashlib.sha256() for _ in range(3)] for _ in schema]

    def update(self, batch):
        for field, array, (valid, values, lengths) in zip(
            self.schema, batch.columns, self.streams, strict=True
        ):
            valid.update(np.asarray(pc.is_valid(array)).astype(np.uint8).tobytes())
            if pa.types.is_string(field.type):
                filled = pc.fill_null(array, "")
                offsets = np.frombuffer(filled.buffers()[1], dtype="<i4")[
                    filled.offset : filled.offset + len(filled) + 1
                ]
                lengths.update(np.diff(offsets).astype("<i8").tobytes())
                values.update(
                    memoryview(filled.buffers()[2])[int(offsets[0]) : int(offsets[-1])]
                )
            else:
                physical = (
                    pc.cast(array, pa.int64())
                    if pa.types.is_timestamp(field.type)
                    else pc.cast(array, pa.int32())
                    if pa.types.is_date32(field.type)
                    else array
                )
                fill = False if pa.types.is_boolean(physical.type) else 0
                data = pc.fill_null(physical, fill).to_numpy(zero_copy_only=False)
                dtype = (
                    "<f8"
                    if pa.types.is_float64(physical.type)
                    else "u1"
                    if pa.types.is_boolean(physical.type)
                    else "<i8"
                )
                values.update(data.astype(dtype).tobytes())

    def hexdigest(self):
        return canonical_hash(
            {
                "format": "column-streams-v1",
                "schema": self.schema.to_string(),
                "columns": [
                    [h.hexdigest() for h in streams] for streams in self.streams
                ],
            }
        )


def quote(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def connect(root, manifest):
    import duckdb

    cfg = manifest["recipe"]["config"]
    scratch = root / ".scratch"
    scratch.mkdir(exist_ok=True)
    available = max(
        1,
        min(
            int(cfg["max_output_gib"] * 1024),
            int((shutil.disk_usage(root).free - cfg["min_free_gib"] * 2**30) / 2**20),
        ),
    )
    con = duckdb.connect(
        config={
            "threads": cfg["threads"],
            "memory_limit": cfg["memory_limit"],
            "temp_directory": str(scratch),
            "max_temp_directory_size": f"{available}MB",
        }
    )
    for name, table in manifest["tables"].items():
        paths = [str(safe_path(root, item["path"])) for item in table["files"]]
        con.read_parquet(paths).create_view(name)
    return con


def diskgraph_config(root, manifest, timestamp_paths=None):
    tables = []
    for name, table in manifest["recipe"]["tables"].items():
        if table["rows"] == 0:
            # Keep empty-table handling explicit across native engine revisions.
            continue
        has_time = any(c["type"] == "time" for c in table["columns"])
        if has_time and (timestamp_paths is None or name not in timestamp_paths):
            raise ValueError(
                "Use prepare-diskgraph to materialize integer timestamp sources for this engine"
            )
        columns = []
        for col in table["columns"]:
            # Preserve exact microseconds with upstream whole-second timestamp storage.
            # prepare_diskgraph explicitly casts the source column before ingestion.
            dtype = "int64" if col["type"] == "time" else col["type"]
            column = {"name": col["name"], "ingested_dtype": dtype}
            columns.append(column)
        tables.append(
            {
                "name": name,
                "primary_key": table["primary_key"],
                "columns": columns,
                "source": {
                    "type": "parquet",
                    "paths": timestamp_paths[name]
                    if has_time
                    else [
                        str(safe_path(root, f["path"]))
                        for f in manifest["tables"][name]["files"]
                    ],
                },
                "fkeys": [
                    {
                        "src_column": r["column"],
                        "dst_table": r["table"],
                        "forward": {},
                        "reverse": {},
                    }
                    for r in table["fkeys"]
                ],
            }
        )
    return {
        "dataset_name": manifest["recipe"].get(
            "dataset", manifest["recipe"]["config"]["preset"]
        ),
        "num_shards": 1,
        "coercion_mode": "lossless",
        "tables": tables,
    }


def prepare_diskgraph(root, output):
    """Stage timestamp-bearing tables with exact integer microseconds.

    Never change canonical Parquet. Record all derived bytes as backend setup.
    Cast the Arrow data, not just the ingest declaration: native timestamp
    storage is whole-second, while old and imported sources may be fractional.
    """
    import time

    import pyarrow.parquet as pq

    from .config import Config
    from .generate import Budget
    from .validate import validate

    root, output = Path(root).resolve(), Path(output).resolve()
    validate(root)
    manifest = json.loads((root / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    cfg = Config(**manifest["recipe"]["config"])
    budget, start = Budget(output, cfg), time.monotonic()
    budget.check()
    paths, inventory = {}, []
    for name, table in manifest["recipe"]["tables"].items():
        timestamp_columns = {c["name"] for c in table["columns"] if c["type"] == "time"}
        if not timestamp_columns or table["rows"] == 0:
            continue
        folder = output / name
        folder.mkdir()
        paths[name] = []
        for i, item in enumerate(manifest["tables"][name]["files"]):
            source = pq.ParquetFile(safe_path(root, item["path"]))
            schema = pa.schema(
                [
                    pa.field(
                        f.name,
                        pa.int64() if f.name in timestamp_columns else f.type,
                        nullable=f.nullable,
                    )
                    for f in source.schema_arrow
                ]
            )
            destination = folder / f"part-{i:05d}.parquet"
            with pq.ParquetWriter(destination, schema, compression="zstd") as writer:
                for chunk in source.iter_batches(batch_size=cfg.batch_rows):
                    arrays = [
                        pc.cast(a, pa.int64()) if f.name in timestamp_columns else a
                        for f, a in zip(chunk.schema, chunk.columns, strict=True)
                    ]
                    writer.write_batch(
                        pa.RecordBatch.from_arrays(arrays, schema=schema),
                        row_group_size=cfg.row_group_rows,
                    )
                    budget.check()
            paths[name].append(str(destination))
            inventory.append(
                {
                    "path": str(destination),
                    "sha256": file_digest(destination),
                    "bytes": destination.stat().st_size,
                }
            )
    config = diskgraph_config(root, manifest, paths)
    write_json(output / "ingest.json", config)
    write_json(
        output / "preparation.json",
        {
            "source_snapshot_id": manifest["snapshot_id"],
            "source_manifest_sha256": file_digest(root / "manifest.json"),
            "timestamp_mapping": "UTC timestamp(us) -> int64 epoch microseconds; no temporal adjacency index",
            "derived_files": inventory,
            "seconds": time.monotonic() - start,
            "ingest_config_sha256": file_digest(output / "ingest.json"),
        },
    )
    budget.check()
    write_json(
        output / "_COMPLETE.json",
        {"preparation_sha256": file_digest(output / "preparation.json")},
    )
    return config
