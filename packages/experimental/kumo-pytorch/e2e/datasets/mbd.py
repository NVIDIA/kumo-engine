# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MBD's registered 30-day transaction count and historical client graph."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from e2e.datasets import bind_tables
from kumo.pytorch import api

M04_MANIFEST = Path(__file__).parent / "pins" / "mbd-m04.json"
CATEGORIES = (
    "currency",
    "event_type",
    "event_subtype",
    "src_type11",
    "src_type12",
    "dst_type11",
    "dst_type12",
    "src_type21",
    "src_type22",
    "src_type31",
    "src_type32",
)


def read_m04(files):
    """Return original anchors and transactions with their full-source physical identities."""
    pin = json.loads(M04_MANIFEST.read_text())
    published = json.loads(files["anchor_manifest"].read_text())
    if published["files"][0]["sha256"] != pin["objects"]["anchors"]["sha256"]:
        raise ValueError("M04 anchor payload disagrees with its manifest")
    positions = pin["anchor_rows"]
    anchor_columns = [
        "sample_id",
        "entity_id",
        "source_fold",
        "split",
        "as_of_time",
        "graph_as_of_time",
        "outcome_end",
        "future_count",
    ]
    anchors = ds.dataset(files["anchors"]).take(positions, columns=anchor_columns)
    anchors = anchors.append_column("anchor_row_position", pa.array(positions))
    clients = anchors["entity_id"].unique()
    # All clients occur in the monthly anchors. Their sorted IDs match all five membership files.
    batches = pq.ParquetFile(files["anchors"]).iter_batches(columns=["entity_id"])
    names = pc.unique(pa.chunked_array([pc.unique(b.column(0)) for b in batches])).sort()
    encoded = pc.binary_join_element_wise(names, "\n", "").buffers()[2]
    if hashlib.sha256(encoded).hexdigest() != pin["client_ids_sha256"]:
        raise ValueError("M04 client identities differ from the full membership")
    ranks = dict(
        zip(clients.to_pylist(), pc.index_in(clients, value_set=names).to_pylist(), strict=True)
    )

    inventory = json.loads(files["source_manifest"].read_text())["files"]
    membership, pieces = {}, []
    offset = 0
    for entry in sorted(inventory, key=lambda item: item["path"]):
        name = entry["path"]
        source = Path(name).parent
        table = source.parent.as_posix()
        if table not in {"client_split", "detail/trx"} or source.name == "fold=-1":
            continue
        # Keep the full file/row ordinal even when only a few shards are staged.
        base = offset
        if table == "detail/trx":
            offset += entry["rows"]
        if name not in files:
            continue
        if name in pin["objects"]:
            digest = pin["objects"][name]["sha256"]
        else:  # Optional full-source replay supplies extra files outside the nightly pin.
            with files[name].open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"M04 shard disagrees with its published manifest: {name}")
        fold = int(source.name.removeprefix("fold="))
        parquet = pq.ParquetFile(files[name])
        owners = parquet.read(columns=["client_id"])["client_id"]
        selected = pc.indices_nonzero(pc.is_in(owners, value_set=clients))
        if table == "client_split":
            membership.update(dict.fromkeys(owners.take(selected).to_pylist(), fold))
        elif len(selected):
            rows = parquet.read(
                columns=["client_id", "event_time", "fold", "amount", *CATEGORIES]
            ).take(selected)
            owners_match = all(
                membership.get(c) == fold for c in rows["client_id"].unique().to_pylist()
            )
            if not owners_match or rows["fold"].unique().to_pylist() != [fold]:
                raise ValueError("M04 transaction disagrees with its source fold")
            pieces.append(rows.append_column("event_id", pc.add(selected.cast(pa.int64()), base)))
    for row in anchors.to_pylist():
        fold = membership[row["entity_id"]]
        split = ("train", "train", "train", "valid", "test")[fold]
        if (fold, split) != (row["source_fold"], row["split"]):
            raise ValueError("M04 anchor disagrees with its source fold")
    if not pieces:
        raise ValueError("M04 extract contains no transactions for the selected clients")
    transactions = pa.concat_tables(pieces, promote_options="permissive")
    ids = transactions["event_id"].combine_chunks().sort().to_numpy().astype("<i8", copy=False)
    if hashlib.sha256(ids).hexdigest() != pin["event_ids_sha256"]:
        raise ValueError("M04 event IDs differ from the complete published client histories")
    return anchors, transactions, ranks


def _contract(fanout):
    relation = api.RelationSpec(
        "client_transactions", "roots", "root_id", "transactions", "root_id"
    )
    schema = api.DataSchema(
        (
            api.TableSpec("roots", "root_id", time_column=api.NON_TEMPORAL),
            api.TableSpec("transactions", "event_id", time_column="event_time"),
        ),
        (relation,),
    )
    layer = api.GraphLayer((api.GraphSample(relation.name, fanout, time_column="event_time"),))
    future = api.Rows((relation.name,), api.TimeWindow("event_time", 0, 30 * 86400))
    targets = {
        "generated_count": api.Aggregate(future, "count", dtype=api.DType("int64")),
        "count": api.ExternalTarget(api.DType("float32")),
    }
    sample = api.SampleSpec(
        api.EntitySpec("roots", entity_type="client"),
        neighborhoods={"history_graph": api.NeighborhoodSpec((layer,))},
        targets=targets,
    )
    fields = api.Fields(
        (
            api.Encode("amount", api.ZScore()),
            *(api.Encode(n, api.Vocabulary()) for n in CATEGORIES),
            api.Encode("event_time_seconds", api.Raw()),
        )
    )
    graph = api.Input(
        api.NeighborhoodSource("history_graph"), api.Graph({"transactions": fields}), None
    )
    return api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            inputs={"graph": graph},
            targets={name: api.LabelTarget(name, api.Raw()) for name in targets},
        ),
    )


def _category(value):
    if value is None or not math.isfinite(value):
        return None
    return str(int(value)) if int(value) == value else repr(value)


def prepare_m04(tables, directory):
    """Bind projected source rows and v2 anchors for the package test or a benchmark."""
    anchors, transactions, ranks = tables
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    ids = pa.array([ranks[name] for name in transactions["client_id"].to_pylist()], pa.int64())
    seconds = pc.floor_temporal(transactions["event_time"], unit="second").cast(pa.timestamp("s"))
    amounts = transactions["amount"].cast(pa.float64())
    events = pa.table(
        {
            "event_id": transactions["event_id"],
            "root_id": ids,
            "event_time": seconds.cast(pa.timestamp("ms")),
            "event_time_seconds": seconds.cast(pa.int64()),
            "event_time_ns": transactions["event_time"].cast(pa.int64()),
            "amount": pc.if_else(pc.is_finite(amounts), amounts, None),
            **{
                n: pa.array([_category(v) for v in transactions[n].to_pylist()], pa.string())
                for n in CATEGORIES
            },
        }
    )
    projected = pa.table(
        {
            "sample_id": anchors["sample_id"],
            "root_id": [ranks[n] for n in anchors["entity_id"].to_pylist()],
            "registered_entity_id": anchors["entity_id"],
            "prediction_time_ns": anchors["as_of_time"].cast(pa.int64()),
            "as_of_time": anchors["graph_as_of_time"],
            "split": anchors["split"],
            "anchor_row_position": anchors["anchor_row_position"],
            "original_count": anchors["future_count"],
            "count": anchors["future_count"].cast(pa.float32(), safe=False),
        }
    )
    with TemporaryDirectory(dir=directory.parent) as temp:
        work = Path(temp)
        pq.write_table(pa.table({"root_id": sorted(ranks.values())}), work / "roots.parquet")
        pq.write_table(events, work / "transactions.parquet")
        pq.write_table(projected, work / "anchors.parquet")
        work.rename(directory)
    fanout = max(Counter(ids.to_pylist()).values(), default=1)
    return bind_tables(directory, _contract(fanout), "root_id", targets=("count",))
