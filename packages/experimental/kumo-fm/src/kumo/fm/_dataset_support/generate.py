"""Stateless vectorized generation and bounded Parquet writing."""

import hashlib
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import Config
from .storage import ColumnDigests, canonical_hash, file_digest, write_json

DAY_SECONDS = 86400
DAY_US = DAY_SECONDS * 1000000
TYPES = {
    "int64": pa.int64(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
    "string": pa.string(),
    "time": pa.timestamp("us", tz="UTC"),
    "date": pa.date32(),
}


def salt(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def random_bits(ids, key):
    """SplitMix64 with wraparound, keyed by a stable column/relation namespace."""
    with np.errstate(over="ignore"):
        x = np.asarray(ids, dtype=np.uint64) + np.uint64(key)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def uniform(ids, key):
    return (random_bits(ids, key) >> np.uint64(11)).astype(np.float64) / 2**53


def prefix(values, text):
    return pc.binary_join_element_wise(
        pa.scalar(text), pc.cast(pa.array(values), pa.string()), ""
    )


def lead_keys(ids):
    return pc.binary_join_element_wise(
        "L", pc.utf8_lpad(pc.cast(pa.array(ids), pa.string()), 12, "0"), ""
    )


def schema_for(table, config):
    fkeys = {r["column"]: r for r in table["fkeys"]}
    fields = []
    for col in table["columns"]:
        nullable = (
            fkeys[col["name"]]["null_fraction"] > 0
            if col["name"] in fkeys
            else col["role"] not in ("primary_key", "timestamp")
            and config.feature_null_fraction > 0
        )
        fields.append(pa.field(col["name"], TYPES[col["type"]], nullable=nullable))
    return pa.schema(fields)


def fk_values(ids, relation, parent_rows, key):
    active = relation["active_parents"]
    linked = ids < relation["linked_rows"]
    if not active:
        return np.zeros(len(ids), dtype=np.int64), ~linked
    u = uniform(ids, key)
    if relation["distribution"] == "power":
        tail = np.floor(u ** relation["power"] * active).astype(np.int64)
    elif relation["distribution"] == "hotspot":
        hot = max(1, min(active, round(active * relation["hot_fraction"])))
        choices = random_bits(ids, key ^ salt("choices"))
        if hot == active:
            tail = (choices % np.uint64(active)).astype(np.int64)
        else:
            tail = np.where(
                u < relation["hot_share"],
                choices % np.uint64(hot),
                hot + choices % np.uint64(active - hot),
            ).astype(np.int64)
    else:
        tail = (random_bits(ids, key) % np.uint64(active)).astype(np.int64)
    ordinal = np.where(ids < active, ids, tail)
    # A bijection over all parents spreads active parents without collisions.
    # Keep multiplication within int64, even for a very large parent population.
    # Selecting a small coprime multiplier avoids object arrays in the hot path.
    multiplier = 65537
    while math.gcd(multiplier, parent_rows) != 1:
        multiplier += 2
    if parent_rows > (2**63 - 1) // multiplier:
        raise ValueError("Parent population too large for the generator key mapping")
    values = (ordinal * multiplier + key % parent_rows) % parent_rows
    return values, ~linked


def batch(recipe, name, start, size):
    config = Config(**recipe["config"])
    table = recipe["tables"][name]
    ids = np.arange(start, start + size, dtype=np.int64)
    namespace = config.seed ^ salt(name)
    fkeys = {}
    for rel in table["fkeys"]:
        fkeys[rel["column"]] = fk_values(
            ids,
            rel,
            recipe["tables"][rel["table"]]["rows"],
            namespace ^ salt(rel["column"]),
        )
    event = random_bits(ids, namespace ^ salt("event-time")) % np.uint64(
        config.history_days * DAY_SECONDS
    )
    recent = uniform(ids, namespace ^ salt("recent")) < config.recent_share
    recent_days = min(90, config.history_days)
    event = np.where(
        recent,
        (config.history_days - recent_days) * DAY_SECONDS
        + event % np.uint64(recent_days * DAY_SECONDS),
        event,
    ).astype(np.int64)
    # Draw whole-second values; retain timestamp[us] as the physical schema.
    event = event * 1000000 + recipe["history_start_us"]
    arrays = []
    for col in table["columns"]:
        cname, dtype, role = col["name"], col["type"], col["role"]
        key = namespace ^ salt(cname)
        nulls = None
        if role == "primary_key":
            value = lead_keys(ids) if dtype == "string" else pa.array(ids)
        elif cname in fkeys:
            values, nulls = fkeys[cname]
            value = lead_keys(values) if dtype == "string" else pa.array(values)
        elif dtype == "time":
            value = pa.array(event, type=TYPES[dtype])
        elif dtype == "date":
            days = event // DAY_US
            if cname.startswith("first_"):
                days = days - (random_bits(ids, key) % 365).astype(np.int64) - 1
            value = pa.array(days.astype(np.int32), type=pa.date32())
        elif dtype == "bool":
            value = pa.array(uniform(ids, key) < (0.2 if cname == "is_mql" else 0.05))
        elif dtype == "int64":
            # Preserve cumulative Marketo window shapes without Spark dependencies.
            root = (
                cname.rsplit("_", 1)[0]
                if cname.endswith(("_30", "_60", "_90"))
                else cname
            )
            terms = [
                random_bits(ids, namespace ^ salt(root + suffix)) % modulus
                for suffix, modulus in [("a", 5), ("b", 4), ("c", 3), ("d", 6)]
            ]
            count = {"30": 1, "60": 2, "90": 3}.get(cname.rsplit("_", 1)[-1], 4)
            values = sum(terms[:count]).astype(np.int64)
            if cname.startswith("days_since"):
                values = (random_bits(ids, key) % 366).astype(np.int64)
            value = pa.array(values)
        elif dtype == "float64":
            value = pa.array(uniform(ids, key) * 1000)
        elif role == "text":
            # Bounded-cardinality text; lengths/cardinality are explicit compression controls.
            values = random_bits(ids, key) % config.category_cardinality
            value = pc.utf8_slice_codeunits(
                pc.utf8_rpad(prefix(values, "text_"), config.text_length, "x"),
                0,
                config.text_length,
            )
        else:
            identity = (
                fkeys.get("marketo_lead_id", (ids, None))[0]
                if cname.startswith(("account_", "audience_"))
                else ids
            )
            cardinality = config.category_cardinality
            if config.preset == "marketo_v1" and not cname.startswith("extra_"):
                cardinality = (
                    250000
                    if cname in ("account_id", "account_name", "company")
                    else (
                        5000 if cname in ("tactic_name", "tactic_program_name") else 12
                    )
                )
            value = prefix(random_bits(identity, key) % cardinality, cname + "_")
        if role not in ("primary_key", "timestamp") and cname not in fkeys:
            nulls = uniform(ids, key ^ salt("nulls")) < config.feature_null_fraction
        if nulls is not None:
            value = pc.if_else(pa.array(nulls), pa.scalar(None, type=value.type), value)
        arrays.append(value)
    return pa.RecordBatch.from_arrays(arrays, schema=schema_for(table, config))


class Budget:
    def __init__(self, root, config):
        self.root, self.config = root, config

    def check(self):
        size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        free = shutil.disk_usage(self.root).free
        if size > self.config.max_output_gib * 2**30:
            raise RuntimeError(
                "Output disk budget exceeded; incomplete bundle retained for inspection"
            )
        if free < self.config.min_free_gib * 2**30:
            raise RuntimeError(
                "Free-space reserve reached; incomplete bundle retained for inspection"
            )
        return size


def plan(config):
    recipe = config.resolve()
    # An uncompressed estimate, not a promised compressed footprint or peak-RSS bound.
    estimated = 0
    for table in recipe["tables"].values():
        for col in table["columns"]:
            estimated += table["rows"] * (
                config.text_length + 8
                if col["role"] == "text"
                else 40
                if col["type"] == "string"
                else 9
            )
    return {
        "preset": config.preset,
        "rows": {k: v["rows"] for k, v in recipe["tables"].items()},
        "edges": sum(
            r["linked_rows"] for t in recipe["tables"].values() for r in t["fkeys"]
        ),
        "estimated_uncompressed_bytes": estimated,
        "recipe": recipe,
    }


def generate(root: Path, config: Config):
    from .tasks import build_tasks
    from .validate import validate

    root = root.resolve()
    recipe = config.resolve()  # Fail before creating output for invalid configurations.
    root.mkdir(parents=True, exist_ok=False)
    budget = Budget(root, config)
    budget.check()
    started = time.monotonic()
    manifest = {
        "format": "fm-dataset-v1",
        "recipe": recipe,
        "tables": {},
        "tasks": {},
        "created_utc": datetime.now(UTC).isoformat(),
        "versions": {"numpy": np.__version__, "pyarrow": pa.__version__},
    }
    for name, table in recipe["tables"].items():
        directory = root / "tables" / name
        directory.mkdir(parents=True)
        schema = schema_for(table, config)
        digests = ColumnDigests(schema)
        inventory = []
        for shard, file_start in enumerate(
            range(0, max(1, table["rows"]), config.file_rows)
        ):
            path = directory / f"part-{shard:05d}.parquet"
            stop = min(file_start + config.file_rows, table["rows"])
            with pq.ParquetWriter(path, schema, compression="zstd") as writer:
                for group_start in range(file_start, stop, config.row_group_rows):
                    group_stop = min(stop, group_start + config.row_group_rows)
                    chunks = []
                    for start in range(group_start, group_stop, config.batch_rows):
                        chunk = batch(
                            recipe,
                            name,
                            start,
                            min(config.batch_rows, group_stop - start),
                        )
                        digests.update(chunk)
                        chunks.append(chunk)
                    writer.write_table(
                        pa.Table.from_batches(chunks),
                        row_group_size=group_stop - group_start,
                    )
                    budget.check()
            meta = pq.ParquetFile(path).metadata
            inventory.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": file_digest(path),
                    "row_groups": [
                        meta.row_group(i).num_rows for i in range(meta.num_row_groups)
                    ],
                }
            )
            budget.check()
        manifest["tables"][name] = {
            "rows": table["rows"],
            "schema": schema.to_string(),
            "logical_sha256": digests.hexdigest(),
            "files": inventory,
        }
        print(
            json.dumps({"table": name, "rows": table["rows"], "files": len(inventory)}),
            flush=True,
        )
    manifest["logical_id"] = canonical_hash(
        {
            name: {k: v[k] for k in ("rows", "schema", "logical_sha256")}
            for name, v in manifest["tables"].items()
        }
    )
    manifest["snapshot_id"] = canonical_hash(manifest["tables"])
    write_json(root / "manifest.json", manifest)
    build_tasks(root, manifest, budget)
    manifest["generation_seconds"] = time.monotonic() - started
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
