# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native histories and targets for AML v8 account-daily anchors."""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from operator import index
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from e2e.datasets import CALENDAR, bind_tables, history_batch
from kumo.pytorch import api

E01_MANIFEST = Path(__file__).parent / "pins" / "aml-e01.json"
L01_MANIFEST = E01_MANIFEST.with_name("aml-l01.json")
_CLOCK = pa.timestamp("ms", tz="UTC")
_AMOUNTS = ("amount_paid", "amount_received")
_CATEGORIES = ("payment_currency", "receiving_currency", "payment_format")
_ROLES = ("sender", "receiver")
_TRANSACTIONS = pa.schema(
    {
        "transaction_id": pa.int64(),
        "event_time": _CLOCK,
        "event_time_seconds": pa.int64(),
        "sender_id": pa.int64(),
        "receiver_id": pa.int64(),
        **dict.fromkeys(_AMOUNTS, pa.decimal128(38, 10)),
        **dict.fromkeys(_CATEGORIES, pa.string()),
    }
)


def _select_anchors(path, positions, account_ids, targets, graph_clock):
    source = ds.dataset(path, format="parquet")
    positions = sorted(map(index, positions))
    in_range = positions and positions[0] >= 0 and positions[-1] < source.count_rows()
    if not in_range or len(set(positions)) != len(positions):
        raise ValueError("Select distinct, in-range physical anchor rows")
    field = ds.field
    columns = {
        "sample_id": field("sample_id"),
        "registered_entity_id": field("entity_id"),
        "prediction_time_ns": field("as_of_time").cast(pa.int64()),
        "as_of_time": field("graph_as_of_time" if graph_clock else "as_of_time"),
        "split": field("forecast_split" if targets else "split"),
        **{name: field(column).cast(dtype) for name, (column, dtype, _) in targets.items()},
    }
    rows = source.take(positions, columns=columns).append_column(
        "anchor_row_position", pa.array(positions)
    )
    rows = rows.filter(pc.is_valid(rows["split"]))
    if not len(rows):
        raise ValueError("No eligible forecast anchors in the selection")
    roots = rows["registered_entity_id"].to_pylist()
    rows = rows.append_column("entity_id", pa.array([account_ids[key] for key in roots]))
    return rows, set(roots)


def _write_transactions(path, destination, roots, account_ids):
    # Match account numbers in Arrow, then disambiguate banks using the accepted identity.
    identities = [json.loads(key) for key in roots]
    (variant,) = {key[0] for key in identities}
    numbers = pa.array([key[3] for key in identities])
    root_ids = {account_ids[key] for key in roots}
    degrees = Counter()
    columns = ["timestamp", "from_bank", "from_account", "to_bank", "to_account"]
    columns += [*_AMOUNTS, *_CATEGORIES]

    def endpoint(row, side):
        key = [variant, "account", str(int(row[f"{side}_bank"])), row[f"{side}_account"]]
        return account_ids[json.dumps(key, ensure_ascii=False, separators=(",", ":"))]

    offset = 0
    with pq.ParquetWriter(destination, _TRANSACTIONS) as writer:
        for batch in pq.ParquetFile(path).iter_batches(columns=columns):
            outgoing = pc.is_in(batch["from_account"], numbers)
            incoming = pc.is_in(batch["to_account"], numbers)
            hits = pc.indices_nonzero(pc.or_(outgoing, incoming))
            selected = []
            for i, row in zip(hits.to_pylist(), batch.take(hits).to_pylist(), strict=True):
                pair = (endpoint(row, "from"), endpoint(row, "to"))
                if not root_ids.isdisjoint(pair):
                    selected.append((i, *pair))
                    degrees.update(set(pair))
            if selected:
                positions, senders, receivers = zip(*selected, strict=True)
                rows = batch.take(positions)
                seconds = pc.strptime(rows["timestamp"], format="%Y/%m/%d %H:%M", unit="s")
                data = {
                    "transaction_id": pc.add(pa.array(positions), offset),
                    "event_time": seconds.cast(_CLOCK),
                    "event_time_seconds": seconds.cast(pa.int64()),
                    "sender_id": senders,
                    "receiver_id": receivers,
                    **{name: rows[name].cast(_TRANSACTIONS.field(name).type) for name in _AMOUNTS},
                    **{name: rows[name] for name in _CATEGORIES},
                }
                writer.write_table(pa.table(data, schema=_TRANSACTIONS))
            offset += len(batch)
    return sorted(degrees.keys() | root_ids), max(degrees.values(), default=1)


def _contract(fanout, targets):
    tables = (
        api.TableSpec("accounts", "account_id", time_column="first_observed_time"),
        api.TableSpec("transactions", "transaction_id", time_column="event_time"),
    )
    relations = tuple(
        api.RelationSpec(role, "accounts", "account_id", "transactions", f"{role}_id")
        for role in _ROLES
    )
    outgoing = api.GraphLayer(tuple(api.GraphSample(role, fanout) for role in _ROLES))
    incoming = api.GraphLayer(
        tuple(api.GraphSample(role, 1, direction="reverse") for role in _ROLES)
    )
    graph = api.NeighborhoodSpec((outgoing, incoming))
    fields = api.Fields(
        (
            *(api.Encode(name, api.ZScore()) for name in _AMOUNTS),
            *(api.Encode(name, api.Vocabulary()) for name in _CATEGORIES),
            api.Encode("event_time_seconds", api.Raw()),
        )
    )
    sample = api.SampleSpec(
        api.EntitySpec("accounts", entity_type="account"),
        neighborhoods={"history_graph": graph},
        targets={
            name: api.ExternalTarget(api.DType(dtype)) for name, (_, dtype, _) in targets.items()
        },
    )
    graph_input = api.Input(
        api.NeighborhoodSource("history_graph"), api.Graph({"transactions": fields}), None
    )
    batch = api.BatchSpec(
        inputs={"graph": graph_input},
        targets={name: api.LabelTarget(name, encoder) for name, (_, _, encoder) in targets.items()},
    )
    return api.DataLoaderSpec(api.DataSchema(tables, relations), sample, batch)


def _write_account_events(directory):
    transactions = ds.dataset(directory / "transactions.parquet")
    field = ds.field
    self_transfer = field("sender_id") == field("receiver_id")
    columns = {name: field(name) for name in ("event_time", *_AMOUNTS, *_CATEGORIES)}
    event_id = field("transaction_id") * 2

    def project(owner, identity, direction, include=None):
        projection = dict(columns, account_id=field(owner), event_id=identity, direction=direction)
        return transactions.scanner(columns=projection, filter=include)

    outgoing = project("sender_id", event_id, pc.if_else(self_transfer, "self", "outgoing"))
    incoming = project("receiver_id", event_id + 1, ds.scalar("incoming"), ~self_transfer)
    path = directory / "account_events.parquet"
    with pq.ParquetWriter(path, outgoing.projected_schema) as writer:
        for scan in (outgoing, incoming):
            for batch in scan.to_batches():
                writer.write_batch(batch)


def _history_contract(contract, fanout, binding, graph_clock):
    relation = api.RelationSpec(
        "account_touches", "accounts", "account_id", "account_events", "account_id"
    )
    table = api.TableSpec("account_events", "event_id", time_column="event_time")
    schema = api.DataSchema(
        (*contract.schema.tables, table), (*contract.schema.relations, relation)
    )
    numeric, category = api.Buckets(bins=64), api.Vocabulary()
    encoders = dict(
        direction=category,
        amount_paid=numeric,
        payment_currency=category,
        amount_received=numeric,
        receiving_currency=category,
        payment_format=category,
    )
    fields = {name: api.Field(name) for name in encoders}
    fields["__event_time"] = api.Field("event_time")
    masked = binding == "E03"
    if graph_clock:
        # Graph anchors execute at T-1s, but ages refer to prediction time T.
        fields.update(
            __execution_age=api.EventTimeDelta("seconds"),
            __prediction_offset=api.Constant(1.0, api.DType("float64")),
            age_range=api.Derived("add", ("__execution_age", "__prediction_offset")),
        )
    else:
        fields["age_seconds" if masked else "age_range"] = api.EventTimeDelta("seconds")
    for name in CALENDAR:
        fields[name] = api.Derived(api.CalendarPart(name), ("__event_time",))
    rows = api.Rows((relation.name,), api.TimeWindow("event_time", stop=0 if graph_clock else -1))
    events = {"transactions": api.EventSpec(rows, fields)}
    batch = history_batch(events, encoders, fanout, masked=masked, max_tokens=len(encoders))
    if contract.batch.targets:
        batch = replace(batch, targets=contract.batch.targets)
    sample = replace(contract.sample, events=events)
    if not graph_clock:
        return api.DataLoaderSpec(schema, replace(sample, neighborhoods={}), batch)
    windows = (("all_history", None), ("24h", -86400), ("7d", -7 * 86400), ("30d", -30 * 86400))
    paths = (("total_touching", relation.name), ("sender", "sender"), ("receiver", "receiver"))
    counts = {
        f"{window}_{role}_count": api.Aggregate(
            api.Rows((path,), api.TimeWindow("event_time", start)), "count"
        )
        for window, start in windows
        for role, path in paths
    }
    inputs = dict(
        contract.batch.inputs,
        **batch.inputs,
        aggregates=api.Input(
            api.EntitySource(),
            api.Fields(tuple(api.Encode(name, api.ZScore()) for name in counts)),
            api.Separate(),
        ),
    )
    return api.DataLoaderSpec(
        schema, replace(sample, entity_values=counts), replace(batch, inputs=inputs)
    )


def prepare_hi_small(
    files: Mapping[str, Path], directory: Path, *, anchor_rows: Sequence[int], binding="E01"
):
    """Prepare HI Small graph, combined, masked, next-token or involvement inputs."""
    if binding not in {"E01", "E02", "E03", "E04", "E05"}:
        raise ValueError(f"Unknown HI Small binding: {binding}")
    targets = {} if binding in {"E03", "E04"} else {"label": ("involvement", "float32", api.Raw())}
    return _prepare_aml(files, directory, anchor_rows, targets, binding)


prepare_e01 = prepare_hi_small


def prepare_l01(files: Mapping[str, Path], directory: Path, *, anchor_rows: Sequence[int]):
    """Prepare LI Medium's next outgoing amount and currency."""
    targets = {
        "amount": ("next_amount", "float32", api.Raw()),
        "currency": ("next_currency", "string", api.Vocabulary()),
    }
    return _prepare_aml(files, directory, anchor_rows, targets, "L01")


def _prepare_aml(files, directory, anchor_rows, targets, binding):
    """Write a bounded native view for verified input files into a fresh directory.

    The selected roots keep all touching transactions across time and their counterparties.
    Transaction IDs and account ranks always refer to the complete input population.
    """
    if directory.exists():
        raise FileExistsError(directory)
    mapping = pq.read_table(files["mapping"], columns=["entity_id", "first_observed_time"])
    mapping = mapping.sort_by("entity_id")
    account_ids = {key: i for i, key in enumerate(mapping["entity_id"].to_pylist())}
    graph_clock = binding in {"E01", "E02", "L01"}
    selection, roots = _select_anchors(
        files["anchors"], anchor_rows, account_ids, targets, graph_clock
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent) as tmp:
        work = Path(tmp)
        ids, fanout = _write_transactions(
            files["transactions"], work / "transactions.parquet", roots, account_ids
        )
        contract = _contract(fanout, targets)
        if binding != "E01":
            _write_account_events(work)
            contract = _history_contract(contract, fanout, binding, graph_clock)
        mapping = mapping.take(ids)
        accounts = pa.table(
            {
                "account_id": ids,
                "entity_id": mapping["entity_id"],
                "first_observed_time": mapping["first_observed_time"].cast(_CLOCK),
            }
        )
        pq.write_table(accounts, work / "accounts.parquet")
        pq.write_table(selection, work / "anchors.parquet")
        work.rename(directory)
    return bind_tables(directory, contract, "entity_id", targets)
