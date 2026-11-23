# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select and load the pinned real-data training windows used by the package suite."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from e2e.datasets import bind_tables
from e2e.datasets.binding_sources import PINS, _sources, full_source
from e2e.datasets.staging import verify

SPLIT_SIZES = {"train": 8192, "valid": 256, "test": 256}


def select_positions(path, split_column, *, sizes=None):
    """Pick one physical row per equal-population stratum within each registered split."""
    sizes = SPLIT_SIZES if sizes is None else sizes
    source = ds.dataset(path)
    rng = np.random.default_rng(17)
    ranks = {}
    seen = dict.fromkeys(sizes, 0)
    for split, size in sizes.items():
        count = source.count_rows(filter=ds.field(split_column) == split)
        if not 0 < size <= count:
            raise ValueError(f"Cannot select {size} distinct {split} rows from {count}")
        edges = np.arange(size + 1, dtype=np.int64) * count // size
        ranks[split] = rng.integers(edges[:-1], edges[1:])
    selected = []
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=[split_column]):
        for split, wanted in ranks.items():
            local = pc.indices_nonzero(pc.equal(batch.column(0), split)).to_numpy()
            start = seen[split]
            stop = start + len(local)
            a, b = wanted.searchsorted([start, stop])
            selected.extend(offset + local[wanted[a:b] - start])
            seen[split] = stop
        offset += len(batch)
    if len(selected) != sum(sizes.values()):
        raise ValueError("Source split counts changed during selection")
    return np.sort(selected)


def upstream_hashes(binding):
    objects = {}
    for name in _sources(binding)[0]:
        objects.update(json.loads((PINS / f"{name}.json").read_text())["objects"])
    return {
        name: objects[name]["sha256"] for name in ("source_manifest", "anchor_manifest", "anchors")
    }


def take_positions(path, positions, columns):
    """Read scattered physical rows without collecting a full weekly anchor table in memory."""
    selected = []
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=columns):
        lo, hi = positions.searchsorted([offset, offset + len(batch)])
        if hi > lo:
            selected.append(batch.take(positions[lo:hi] - offset))
        offset += len(batch)
    return pa.Table.from_batches(selected)


def binding_contract(binding, capacity):
    if binding == "L01":
        from e2e.datasets import aml

        heads = aml._li_targets(binding)
        return aml._binding_contract(capacity, binding, heads, True), "entity_id", tuple(heads)
    if binding in ("M03", "M04"):
        from e2e.datasets import mbd

        categories = binding == "M03"
        return (
            mbd._contract(capacity, categories),
            "root_id",
            mbd._HEADS if categories else ("count",),
        )
    from e2e.datasets import synthetic

    kind = {"S01": "account", "S02": "customer", "S04": "ltv"}[binding]
    root = "account_id" if kind == "account" else "customer_id"
    return synthetic._contract(capacity, kind), root, ("label",) if kind == "ltv" else ()


def extract_population(binding, rows, heads):
    counts = {}
    eligible = {name: {} for name in heads}
    for split in SPLIT_SIZES:
        selected = rows.filter(pc.equal(rows["split"], split))
        counts[split] = len(selected)
        for name in heads:
            eligible[name][split] = len(selected) - selected[name].null_count
    if binding == "M04":
        eligible["generated_count"] = eligible["count"].copy()
    if binding in ("S01", "S02"):
        eligible["masked_values" if binding == "S01" else "next_token"] = None
    return counts, eligible


def load_extract(binding, root):
    manifest = PINS / "throughput" / f"{binding.lower()}.json"
    pin = json.loads(manifest.read_text())
    if pin["upstream_sha256"] != upstream_hashes(binding):
        raise ValueError(f"{binding}: extract no longer matches the registered source and anchors")
    files = verify(root, manifest)
    spec, root_key, heads = binding_contract(binding, pin["capacity"])
    path = files["anchors"]
    rows = pq.read_table(path)
    counts, eligible = extract_population(binding, rows, heads)
    if counts != pin["split_counts"] or eligible != pin["expected_eligible"]:
        raise ValueError(f"{binding}: split or target counts differ from the pinned extract")
    if sum(counts.values()) != len(rows) or not rows["anchor_row_position"].equals(
        pa.chunked_array([pa.array(np.arange(len(rows), dtype=np.int64))])
    ):
        raise ValueError(f"{binding}: invalid extract split or physical row position")
    for key in ("sample_id", "source_row_position"):
        if rows[key].null_count or len(rows[key].unique()) != len(rows):
            raise ValueError(f"{binding}: missing or repeated {key}")
    latest = pc.max(rows["prediction_time_ns"].filter(pc.equal(rows["split"], "train"))).as_py()
    cutoff = latest // 1_000_000_000 - 1
    source_paths = {t.name: files[t.name] for t in spec.schema.tables}
    spec, data, anchors = bind_tables(path.parent, spec, root_key, heads, source_paths=source_paths)
    return spec, data, anchors, files, cutoff, rows.select(["sample_id", "split"]), counts, eligible


def seal(binding, directory, capacity):
    """Pin the prepared files and mix source strata before the native loader's timed prefix."""
    path = directory / "anchors.parquet"
    rows = pq.read_table(path).sort_by("source_row_position")
    rows = rows.take(np.random.default_rng(17).permutation(len(rows)))
    rows = rows.set_column(
        rows.schema.get_field_index("anchor_row_position"),
        "anchor_row_position",
        pa.array(np.arange(len(rows), dtype=np.int64)),
    )
    pq.write_table(rows, path, compression="zstd")
    spec, _, heads = binding_contract(binding, capacity)
    counts, eligible = extract_population(binding, rows, heads)
    if counts != SPLIT_SIZES:
        raise ValueError(f"{binding}: expected {SPLIT_SIZES}, selected {counts}")
    objects = {}
    for name in ["anchors", *(table.name for table in spec.schema.tables)]:
        file = directory / f"{name}.parquet"
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        objects[name] = dict(size=file.stat().st_size, sha256=digest)
    identity = hashlib.sha256(json.dumps(objects, sort_keys=True).encode()).hexdigest()
    prefix = f"datasets/kumo-pytorch/throughput/v1/{binding.lower()}/{identity}"
    for name, obj in objects.items():
        obj["key"] = f"{prefix}/{name}.parquet"
    bucket = json.loads((PINS / f"{_sources(binding)[0][0]}.json").read_text())
    return {
        **{k: bucket[k] for k in ("version", "bucket", "region", "expected_bucket_owner")},
        "upstream_sha256": upstream_hashes(binding),
        "capacity": capacity,
        "split_counts": counts,
        "expected_eligible": eligible,
        "objects": objects,
    }


def build_extract(binding, root, directory):
    """Curate once from verified originals; the runtime only reads the resulting Parquet files."""
    files = full_source(binding, root)
    path = files["anchors"]
    positions = select_positions(path, "forecast_split" if binding == "L01" else "split")
    if binding == "L01":
        from e2e.datasets import aml

        aml.prepare_li_medium(files, directory, anchor_rows=positions)
        anchors_path = directory / "anchors.parquet"
        rows = pq.read_table(anchors_path)
        pq.write_table(
            rows.append_column("source_row_position", rows["anchor_row_position"]), anchors_path
        )
        # Per-account degree in the complete registered source, counting self transfers once.
        raw = ds.dataset(files["transactions"])
        directions = []
        for side in ("from", "to"):
            self_transfer = (ds.field("from_bank") == ds.field("to_bank")) & (
                ds.field("from_account") == ds.field("to_account")
            )
            fields = {key: ds.field(f"{side}_{key}") for key in ("bank", "account")}
            touches = raw.to_table(columns=fields, filter=~self_transfer if side == "to" else None)
            directions.append(touches.group_by(["bank", "account"]).aggregate([([], "count_all")]))
        degrees = (
            pa.concat_tables(directions)
            .group_by(["bank", "account"])
            .aggregate([("count_all", "sum")])
        )
        capacity = pc.max(degrees["count_all_sum"]).as_py()
    else:
        from e2e.datasets import population_tables, synthetic

        directory.mkdir(parents=True)
        if binding in ("M03", "M04"):
            capacity = population_tables.project_mbd(binding, files, positions, directory)
        else:
            kind = {"S01": "account", "S02": "customer", "S04": "ltv"}[binding]
            selected = take_positions(path, positions, synthetic._anchor_columns(kind))
            selected = selected.append_column("source_row_position", pa.array(positions))
            selected = selected.append_column(
                "anchor_row_position", pa.array(np.arange(len(selected)))
            )
            population_tables._synthetic(directory, files, kind, selected=selected)
            capacity = json.loads((directory / "meta.json").read_text())["capacity"]
    return seal(binding, directory, capacity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=build_extract.__doc__)
    parser.add_argument("binding", choices=("L01", "M03", "M04", "S01", "S02", "S04"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pin = build_extract(args.binding, args.source_root, args.output)
    (args.output / "pin.json").write_text(json.dumps(pin, indent=2) + "\n")
