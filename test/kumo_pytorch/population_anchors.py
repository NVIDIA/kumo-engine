# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream original AML anchors into native execution tables without sampling full runs."""

import hashlib
import json

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import kumo.pytorch.api as r

from .aml_inputs import SECOND, SPLITS


def add_identity(digest, sample_id):
    digest.update(json.dumps(sample_id, ensure_ascii=False).encode() + b"\n")


def project_population(
    path, destination, identities, *, graph, native, limit=None, splits=SPLITS
):
    """Keep original order, labels and splits; v2 shifts only graph execution clocks."""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be positive or None for all eligible anchors")
    if not splits or len(set(splits)) != len(splits) or not set(splits) <= set(SPLITS):
        raise ValueError("Select distinct known anchor splits")
    column = "split" if native else "forecast_split"
    columns = ["sample_id", "entity_id", "as_of_time", column, "involvement"]
    if graph:
        columns.append("graph_as_of_time")
    source = pq.ParquetFile(path)
    for name in ("as_of_time", "graph_as_of_time") if graph else ("as_of_time",):
        if source.schema_arrow.field(name).type != pa.timestamp("ns"):
            raise ValueError(f"Expected original nanosecond anchor clock: {name}")
    schema = pa.schema(
        [
            ("sample_id", pa.string()),
            ("entity_id", pa.int64()),
            ("as_of_time", pa.int64()),
            ("split", pa.string()),
            ("label", pa.float32()),
            ("prediction_time_ns", pa.int64()),
            ("registered_entity_id", pa.string()),
            ("anchor_row_position", pa.int64()),
        ]
    )
    counts = dict.fromkeys(splits, 0)
    source_counts = dict.fromkeys(splits, 0)
    hashes = {split: hashlib.sha256() for split in splits}
    seen, offset, train_cutoff = set(), 0, None
    with pq.ParquetWriter(destination, schema, compression="zstd") as writer:
        for batch in source.iter_batches(columns=columns, batch_size=65536):
            for item in pc.value_counts(batch.column(column)).to_pylist():
                if item["values"] in source_counts:
                    source_counts[item["values"]] += item["counts"]
            for name in (
                ("as_of_time", "graph_as_of_time") if graph else ("as_of_time",)
            ):
                batch = batch.set_column(
                    batch.schema.get_field_index(name),
                    name,
                    pc.cast(batch.column(name), pa.int64()),
                )
            records = []
            for position, row in enumerate(batch.to_pylist(), offset):
                split = row[column]
                if split is None:
                    continue
                if split not in SPLITS:
                    raise ValueError(f"Unknown anchor split: {split}")
                if split not in counts:
                    continue
                if limit is not None and counts[split] >= limit:
                    continue
                sample_id, entity = row["sample_id"], row["entity_id"]
                if not sample_id or sample_id in seen:
                    raise ValueError("Missing or duplicate selected sample identity")
                if not entity or entity not in identities:
                    raise ValueError("Selected entity is absent from mapped source")
                stamp = row["as_of_time"]
                if stamp is None or stamp % (86400 * SECOND):
                    raise ValueError(
                        "AML anchors require daily UTC midnight prediction times"
                    )
                if graph and row["graph_as_of_time"] != stamp - SECOND:
                    raise ValueError(
                        "v2 graph cutoff must equal prediction time minus one second"
                    )
                if row["involvement"] not in (None, 0, 1):
                    raise ValueError("Involvement label must be binary or null")
                clock = row["graph_as_of_time"] if graph else stamp
                records.append(
                    {
                        "sample_id": sample_id,
                        "entity_id": identities[entity],
                        "as_of_time": clock // SECOND,
                        "split": split,
                        "label": row["involvement"],
                        "prediction_time_ns": stamp,
                        "registered_entity_id": entity,
                        "anchor_row_position": position,
                    }
                )
                seen.add(sample_id)
                counts[split] += 1
                add_identity(hashes[split], sample_id)
                if split == "train":
                    cutoff = stamp // SECOND - 1
                    train_cutoff = (
                        cutoff if train_cutoff is None else max(train_cutoff, cutoff)
                    )
            if records:
                writer.write_table(pa.Table.from_pylist(records, schema=schema))
            offset += batch.num_rows
            if limit is not None and all(n == limit for n in counts.values()):
                break
    if any(n == 0 or (limit is not None and n != limit) for n in counts.values()):
        raise ValueError("Insufficient required anchors in selected splits")
    if limit is None and (
        offset != source.metadata.num_rows or counts != source_counts
    ):
        raise ValueError("Full projection did not retain every eligible source anchor")
    manifest = {
        "mode": "full" if limit is None else "smoke",
        "split_column": column,
        "counts": counts,
        "ordered_sample_id_sha256": {s: h.hexdigest() for s, h in hashes.items()},
        "encoder_cutoff_seconds": train_cutoff,
        "graph": graph,
        "native": native,
        "source_rows": source.metadata.num_rows,
        "scanned_rows": offset,
        "scanned_source_split_counts": source_counts,
    }
    return r.AnchorTable(
        r.Parquet(str(destination)),
        "entity_id",
        "as_of_time",
        sample_id="sample_id",
        split="split",
        targets={} if native else {"label": "label"},
        metadata=("prediction_time_ns", "registered_entity_id", "anchor_row_position"),
    ), manifest


def projected_rows(path, split):
    """Bounded-memory, independent delivery reference for one original split."""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
        mask = pc.equal(batch.column("split"), split)
        yield from batch.filter(mask).to_pylist()
