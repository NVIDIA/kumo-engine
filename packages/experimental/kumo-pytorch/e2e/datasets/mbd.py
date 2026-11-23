# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered MBD transaction forecasts over complete client histories."""

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
from e2e.datasets.staging import verify
from kumo.pytorch import api

M04_MANIFEST = Path(__file__).parent / "pins" / "mbd-m04.json"
M03_MANIFEST = M04_MANIFEST.with_name("mbd-m03.json")
_HEADS = ("event_type", "event_subtype")
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


def load_mbd(root, manifest=M04_MANIFEST):
    """Load a registered forecast from its anchors and the shared MBD sources."""
    return read_mbd(verify(root, manifest), manifest)


def read_mbd(files, manifest=M04_MANIFEST):
    """Return original anchors and transactions with their full-source physical identities."""
    pin = json.loads(manifest.read_text())
    source_pin = pin
    if shared := pin.get("include_objects_from"):
        source_pin = json.loads(manifest.with_name(shared).read_text())
    published = json.loads(files["anchor_manifest"].read_text())
    if published["files"][0]["sha256"] != pin["objects"]["anchors"]["sha256"]:
        raise ValueError("MBD anchor payload disagrees with its manifest")
    positions = pin["anchor_rows"]
    anchor_columns = [
        "sample_id",
        "entity_id",
        "source_fold",
        "split",
        "as_of_time",
    ]
    anchor_data = ds.dataset(files["anchors"])
    if "graph_as_of_time" in anchor_data.schema.names:
        anchor_columns.extend(("graph_as_of_time", "outcome_end", "future_count"))
    else:
        anchor_columns.append("outcome_end")
        anchor_columns.extend(
            f"next_{name}{suffix}" for name in _HEADS for suffix in ("", "_valid")
        )
    anchors = anchor_data.take(positions, columns=anchor_columns)
    anchors = anchors.append_column("anchor_row_position", pa.array(positions))
    clients = anchors["entity_id"].unique()
    # All clients occur in the monthly anchors. Their sorted IDs match all five membership files.
    batches = pq.ParquetFile(files["anchors"]).iter_batches(columns=["entity_id"])
    names = pc.unique(pa.chunked_array([pc.unique(b.column(0)) for b in batches])).sort()
    encoded = pc.binary_join_element_wise(names, "\n", "").buffers()[2]
    if hashlib.sha256(encoded).hexdigest() != source_pin["client_ids_sha256"]:
        raise ValueError("MBD client identities differ from the full membership")
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
        if name in source_pin["objects"]:
            digest = source_pin["objects"][name]["sha256"]
        else:  # Optional full-source replay supplies extra files outside the nightly pin.
            with files[name].open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"MBD shard disagrees with its published manifest: {name}")
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
                raise ValueError("MBD transaction disagrees with its source fold")
            pieces.append(rows.append_column("event_id", pc.add(selected.cast(pa.int64()), base)))
    for row in anchors.to_pylist():
        fold = membership[row["entity_id"]]
        split = ("train", "train", "train", "valid", "test")[fold]
        if (fold, split) != (row["source_fold"], row["split"]):
            raise ValueError("MBD anchor disagrees with its source fold")
    if not pieces:
        raise ValueError("MBD extract contains no transactions for the selected clients")
    transactions = pa.concat_tables(pieces, promote_options="permissive")
    ids = transactions["event_id"].combine_chunks().sort().to_numpy().astype("<i8", copy=False)
    if hashlib.sha256(ids).hexdigest() != pin["event_ids_sha256"]:
        raise ValueError("MBD event IDs differ from the complete published client histories")
    return anchors, transactions, ranks


def _contract(fanout, categories):
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
    if categories:
        return _category_contract(schema, relation.name)
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


def _category_contract(schema, relation):
    event = {name: api.Field(name) for name in ("amount", *CATEGORIES)}
    event.update(age_range=api.EventTimeDelta("seconds"), __event_time=api.Field("event_time"))
    fields = [api.Encode("age_range", api.Buckets(bins=64))]
    fields.extend(api.Encode(name, api.Vocabulary()) for name in CATEGORIES if name not in _HEADS)
    for name, domain in (("hour", range(24)), ("weekday", range(7)), ("day", range(1, 32))):
        event[name] = api.Derived(api.CalendarPart(name), ("__event_time",))
        fields.append(api.Encode(name, api.Vocabulary(values=tuple((i, i + 2) for i in domain))))
    rows = api.Rows((relation,), api.TimeWindow("event_time", stop=-1))
    sample = api.SampleSpec(
        api.EntitySpec("roots", entity_type="client"),
        events={"transactions": api.EventSpec(rows, event)},
        entity_values={"root_id": api.Field("root_id")},
        targets={name: api.ExternalTarget(api.DType("string")) for name in _HEADS},
    )
    history = api.Fields(
        tuple(fields), row=api.Composite(id_space="history"), history=api.Grouped()
    )
    inputs = {
        "history": api.Input(api.EventSource(("transactions",)), history, api.Packed()),
        "events": api.Input(api.EventsOf("history"), api.Fields(tuple(fields)), api.Packed()),
        "identity": api.Input(
            api.EntitySource(), api.Fields((api.Encode("root_id", api.Raw()),)), api.Separate()
        ),
    }
    targets = {name: api.LabelTarget(name, api.Vocabulary()) for name in _HEADS}
    positive = api.ThresholdEventLabel(
        "positive_amount", "amount", api.DType("string"), api.Vocabulary(), threshold=0.0
    )
    targets["matching"] = api.MatchingGroupTarget("identity", ("root_id",))
    targets["aligned"] = api.AlignedEventTarget(
        "events", tuple(api.Encode(name, api.Vocabulary()) for name in _HEADS), (positive,)
    )
    roles = {name: f"[{name.upper()}]" for name in ("pad", "unknown", "mask", "entity", "event")}
    batch = api.BatchSpec(
        inputs,
        targets,
        id_spaces={"history": api.IdSpace(dict(roles, separator="[SEP]"))},
        batching=api.Batching(max_samples=32),
    )
    return api.DataLoaderSpec(schema, sample, batch)


def _category(value):
    if value is None or not math.isfinite(value):
        return None
    return str(int(value)) if int(value) == value else repr(value)


def _category_strings(column):
    codes = pc.dictionary_encode(column.combine_chunks())
    labels = pa.array([_category(v) for v in codes.dictionary.to_pylist()], pa.string())
    return pc.take(labels, codes.indices)


def _project_events(transactions, ids, categories):
    seconds = pc.floor_temporal(transactions["event_time"], unit="second").cast(pa.timestamp("s"))
    nanoseconds = transactions["event_time"].cast(pa.int64())
    event_ids = transactions["event_id"]
    if categories:
        # Native clocks are whole seconds. Preserve original signed ns and file/row order.
        wide = pa.decimal128(20, 0)
        biased_ns = pc.add(nanoseconds.cast(wide), pa.scalar(2**63, wide))
        event_ids = pc.binary_join_element_wise(
            pc.utf8_lpad(biased_ns.cast(pa.string()), 20, "0"),
            pc.utf8_lpad(event_ids.cast(pa.string()), 20, "0"),
            ":",
        )
    amounts = transactions["amount"].cast(pa.float64())
    return pa.table(
        {
            "event_id": event_ids,
            "root_id": ids,
            "event_time": seconds.cast(pa.timestamp("ms")),
            "event_time_seconds": seconds.cast(pa.int64()),
            "event_time_ns": nanoseconds,
            "amount": pc.if_else(pc.is_finite(amounts), amounts, None),
            **{n: _category_strings(transactions[n]) for n in CATEGORIES},
        }
    )


def _project_anchors(anchors, ids, categories):
    if categories:
        labels = {}
        for name in _HEADS:
            valid = anchors[f"next_{name}_valid"]
            if valid.null_count:
                raise ValueError(f"Missing MBD forecast validity: {name}")
            labels[name] = pc.if_else(valid, anchors[f"next_{name}"].cast(pa.string()), None)
    else:
        labels = {
            "original_count": anchors["future_count"],
            "count": anchors["future_count"].cast(pa.float32(), safe=False),
        }
    return pa.table(
        {
            "sample_id": anchors["sample_id"],
            "root_id": ids,
            "registered_entity_id": anchors["entity_id"],
            "prediction_time_ns": anchors["as_of_time"].cast(pa.int64()),
            "as_of_time": anchors["as_of_time" if categories else "graph_as_of_time"],
            "split": anchors["split"],
            "anchor_row_position": anchors["anchor_row_position"],
            **labels,
        }
    )


def prepare_mbd(tables, directory):
    """Bind projected source rows and registered anchors for the package test or a benchmark."""
    anchors, transactions, ranks = tables
    categories = "next_event_type" in anchors.column_names
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    ids = pa.array([ranks[name] for name in transactions["client_id"].to_pylist()], pa.int64())
    events = _project_events(transactions, ids, categories)
    anchor_ids = [ranks[name] for name in anchors["entity_id"].to_pylist()]
    projected = _project_anchors(anchors, anchor_ids, categories)
    with TemporaryDirectory(dir=directory.parent) as temp:
        work = Path(temp)
        pq.write_table(pa.table({"root_id": sorted(ranks.values())}), work / "roots.parquet")
        pq.write_table(events, work / "transactions.parquet")
        pq.write_table(projected, work / "anchors.parquet")
        work.rename(directory)
    targets = _HEADS if categories else ("count",)
    fanout = max(Counter(ids.to_pylist()).values(), default=1)
    return bind_tables(directory, _contract(fanout, categories), "root_id", targets=targets)


read_m04 = read_mbd
prepare_m04 = prepare_mbd
