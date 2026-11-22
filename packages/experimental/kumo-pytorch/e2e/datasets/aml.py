# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native histories and targets for AML v8 account-daily anchors."""

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import replace
from operator import index
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from torch.nn import functional as F

from e2e import E01_MANIFEST as E01_MANIFEST
from e2e import L01_MANIFEST as L01_MANIFEST
from e2e.datasets import CALENDAR, bind_tables, history_batch
from e2e.datasets.staging import verify
from kumo.pytorch import api

_CLOCK = pa.timestamp("ms", tz="UTC")
_AMOUNTS = ("amount_paid", "amount_received")
_CATEGORIES = ("payment_currency", "receiving_currency", "payment_format")
_ROLES = ("sender", "receiver")
_LI_TARGETS = {
    "L01": {"amount": ("next_amount", "numerical"), "currency": ("next_currency", "categorical")},
    "L02": {"label": ("next_delay_seconds", "numerical")},
    "L03": {"label": ("next_format", "categorical")},
    "L04": {"label": ("involvement", "binary")},
}
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


def _anchor_columns(targets, graph_clock):
    field = ds.field
    return {
        "sample_id": field("sample_id"),
        "registered_entity_id": field("entity_id"),
        "prediction_time_ns": field("as_of_time").cast(pa.int64()),
        "as_of_time": field("graph_as_of_time" if graph_clock else "as_of_time"),
        "split": field("forecast_split" if targets else "split"),
        **{name: field(column).cast(dtype) for name, (column, dtype, _) in targets.items()},
    }


def _select_anchors(path, positions, account_ids, targets, graph_clock):
    source = ds.dataset(path, format="parquet")
    positions = sorted(map(index, positions))
    in_range = positions and positions[0] >= 0 and positions[-1] < source.count_rows()
    if not in_range or len(set(positions)) != len(positions):
        raise ValueError("Select distinct, in-range physical anchor rows")
    columns = _anchor_columns(targets, graph_clock)
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
    pragma = binding in {"E03", "L04"}
    if graph_clock:
        # Graph anchors execute at T-1s, but ages refer to prediction time T.
        fields.update(
            __execution_age=api.EventTimeDelta("seconds"),
            __prediction_offset=api.Constant(1.0, api.DType("float64")),
            age_range=api.Derived("add", ("__execution_age", "__prediction_offset")),
        )
    else:
        fields["age_seconds" if pragma else "age_range"] = api.EventTimeDelta("seconds")
    for name in CALENDAR:
        fields[name] = api.Derived(api.CalendarPart(name), ("__event_time",))
    rows = api.Rows((relation.name,), api.TimeWindow("event_time", stop=0 if graph_clock else -1))
    events = {"transactions": api.EventSpec(rows, fields)}
    batch = history_batch(events, encoders, fanout, masked=pragma, max_tokens=len(encoders))
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
    return _prepare_aml(files, directory, anchor_rows, _hi_targets(binding), binding)


def prepare_hi_small_group(files, directory, positions, roots):
    """Share complete source histories and account ranks across the five HI bindings."""
    mapping, ids = _accounts(files)
    fanout = _write_source_tables(files, directory, mapping, ids, roots, history=True)

    def bind(binding):
        targets = _hi_targets(binding)
        graph = binding in {"E01", "E02"}
        selected, _ = _select_anchors(files["anchors"], positions, ids, targets, graph)
        contract = _binding_contract(fanout, binding, targets, graph)
        path = directory / binding
        path.mkdir()
        pq.write_table(selected, path / "anchors.parquet")
        sources = {t.name: directory / f"{t.name}.parquet" for t in contract.schema.tables}
        return bind_tables(path, contract, "entity_id", targets, source_paths=sources)

    return bind


def _hi_targets(binding):
    if binding not in {"E01", "E02", "E03", "E04", "E05"}:
        raise ValueError(f"Unknown HI Small binding: {binding}")
    return {} if binding in {"E03", "E04"} else {"label": ("involvement", "float32", api.Raw())}


prepare_e01 = prepare_hi_small


def prepare_li_medium(
    files: Mapping[str, Path], directory: Path, *, anchor_rows: Sequence[int], binding="L01"
):
    """Prepare an LI Medium stored-target binding from the supplied source tables."""
    if binding not in _LI_TARGETS:
        raise ValueError(f"Unknown LI Medium binding: {binding}")
    targets = {
        name: (column, "string", api.Vocabulary())
        if kind == "categorical"
        else (column, "float32", api.Raw())
        for name, (column, kind) in _LI_TARGETS[binding].items()
    }
    return _prepare_aml(files, directory, anchor_rows, targets, binding)


prepare_l01 = prepare_li_medium


def prepare_l01_benchmark(source_root, directory):
    """Prepare the first 512 forecast anchors per split from the verified source."""
    files = verify(source_root, L01_MANIFEST)
    remaining = dict.fromkeys(("train", "valid", "test"), 512)
    positions = []
    offset = 0
    for batch in pq.ParquetFile(files["anchors"]).iter_batches(columns=["forecast_split"]):
        for position, split in enumerate(batch.column(0).to_pylist(), offset):
            if remaining.get(split, 0):
                positions.append(position)
                remaining[split] -= 1
        if not any(remaining.values()):
            break
        offset += len(batch)
    else:
        raise ValueError(f"Incomplete L01 benchmark population: missing {remaining}")
    spec, _, _ = prepare_li_medium(files, directory, anchor_rows=positions)
    (directory / "contract.json").write_text(spec.to_json())


def li_head_loss(batch, binding, prediction, name="label"):
    """Return the loss sum and eligible count for one stored LI head."""
    _, kind = _LI_TARGETS[binding][name]
    target = batch.targets[name]
    if kind != "categorical" and prediction.shape != target["values"].shape:
        raise ValueError(f"Expected one scalar {name} prediction per anchor")
    eligible = target["valid"] & ~batch.metadata["is_padding"]
    values, scores = target["values"][eligible], prediction[eligible]
    if kind == "numerical":
        total = F.smooth_l1_loss(scores.double(), values.double(), reduction="sum", beta=1.0)
    elif kind == "categorical":
        total = F.cross_entropy(scores, values, reduction="sum")
    else:
        total = F.binary_cross_entropy_with_logits(scores, values, reduction="sum")
    return total, values.numel()


def l01_loss(batch, amount_prediction, currency_logits):
    """Return the sum of head means, each head's loss sum, and its eligible count."""
    sums, counts = {}, {}
    for name, prediction in (("amount", amount_prediction), ("currency", currency_logits)):
        sums[name], counts[name] = li_head_loss(batch, "L01", prediction, name)
    loss = sum(sums[name] / max(count, 1) for name, count in counts.items())
    return loss, sums, counts


def _accounts(files):
    mapping = pq.read_table(files["mapping"], columns=["entity_id", "first_observed_time"])
    mapping = mapping.sort_by("entity_id")
    return mapping, {key: i for i, key in enumerate(mapping["entity_id"].to_pylist())}


def _write_source_tables(files, directory, mapping, account_ids, roots, *, history):
    ids, fanout = _write_transactions(
        files["transactions"], directory / "transactions.parquet", roots, account_ids
    )
    if history:
        _write_account_events(directory)
    mapping = mapping.take(ids)
    accounts = pa.table(
        {
            "account_id": ids,
            "entity_id": mapping["entity_id"],
            "first_observed_time": mapping["first_observed_time"].cast(_CLOCK),
        }
    )
    pq.write_table(accounts, directory / "accounts.parquet")
    return fanout


def _binding_contract(fanout, binding, targets, graph_clock):
    contract = _contract(fanout, targets)
    if binding not in {"E01", "L03"}:
        contract = _history_contract(contract, fanout, binding, graph_clock)
    return contract


def _prepare_aml(files, directory, anchor_rows, targets, binding):
    """Write a bounded native view for verified input files into a fresh directory.

    The selected roots keep all touching transactions across time and their counterparties.
    Transaction IDs and account ranks always refer to the complete input population.
    """
    if directory.exists():
        raise FileExistsError(directory)
    mapping, account_ids = _accounts(files)
    graph_clock = binding in {"E01", "E02", "L01", "L03"}
    selection, roots = _select_anchors(
        files["anchors"], anchor_rows, account_ids, targets, graph_clock
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent) as tmp:
        work = Path(tmp)
        fanout = _write_source_tables(
            files, work, mapping, account_ids, roots, history=binding not in {"E01", "L03"}
        )
        contract = _binding_contract(fanout, binding, targets, graph_clock)
        pq.write_table(selection, work / "anchors.parquet")
        work.rename(directory)
    return bind_tables(directory, contract, "entity_id", targets)


def prepare_full_aml(files, directory, binding):
    """Project every eligible published anchor and all original AML transactions."""
    directory.mkdir(parents=True)
    mapping, ids = _accounts(files)
    targets = (
        _hi_targets(binding)
        if binding.startswith("E")
        else {
            "amount": ("next_amount", "float32", api.Raw()),
            "currency": ("next_currency", "string", api.Vocabulary()),
        }
    )
    graph = binding in {"E01", "E02", "L01"}
    fanout = _write_source_tables(files, directory, mapping, ids, ids, history=binding != "E01")
    dataset = ds.dataset(files["anchors"])
    offset = 0
    names = mapping["entity_id"].combine_chunks()
    with ExitStack() as stack:
        writer = None
        for batch in dataset.to_batches(columns=_anchor_columns(targets, graph)):
            rows = pa.Table.from_batches([batch]).append_column(
                "anchor_row_position", pa.array(range(offset, offset + len(batch)), type=pa.int64())
            )
            offset += len(batch)
            rows = rows.filter(pc.is_valid(rows["split"]))
            root_ids = pc.index_in(rows["registered_entity_id"], value_set=names)
            if root_ids.null_count:
                raise ValueError("AML anchor account missing from published mapping")
            rows = rows.append_column("entity_id", root_ids.cast(pa.int64()))
            if writer is None:
                writer = stack.enter_context(
                    pq.ParquetWriter(directory / "anchors.parquet", rows.schema)
                )
            writer.write_table(rows)
    contract = _binding_contract(fanout, binding, targets, graph)
    return bind_tables(directory, contract, "entity_id", targets)


def prepare_hi_small_benchmark(source_root, directory):
    """Prepare the full HI Small tables once and two nested training populations per binding."""
    files = verify(source_root, E01_MANIFEST)
    mapping, account_ids = _accounts(files)
    directory.mkdir(parents=True)
    fanout = _write_source_tables(files, directory, mapping, account_ids, account_ids, history=True)

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    identity = {
        "source_manifest_sha256": digest(E01_MANIFEST),
        "tables": {p.stem: digest(p) for p in directory.glob("*.parquet")},
        "bindings": {},
    }
    for binding in ("E01", "E02", "E03", "E04", "E05"):
        targets = _hi_targets(binding)
        graph_clock = binding in {"E01", "E02"}
        spec = _binding_contract(fanout, binding, targets, graph_clock)
        spec, _, _ = bind_tables(directory, spec, "entity_id", targets)
        contract_path = directory / f"{binding}.json"
        contract_path.write_text(spec.to_json())
        split = "forecast_split" if targets else "split"
        splits = pq.read_table(files["anchors"], columns=[split])[split]
        positions = pc.indices_nonzero(pc.equal(splits, "train")).slice(0, 400_000)
        rows, _ = _select_anchors(
            files["anchors"], positions.to_pylist(), account_ids, targets, graph_clock
        )
        if len(rows) != 400_000 or pc.count_distinct(rows["sample_id"]).as_py() != len(rows):
            raise ValueError(f"{binding} needs 400,000 distinct training anchors")
        hashes = {}
        for count in (25_000, 400_000):
            path = directory / f"{binding}-{count}.parquet"
            pq.write_table(rows.slice(0, count), path)
            hashes[str(count)] = digest(path)
        small = pq.read_table(directory / f"{binding}-25000.parquet")
        if not small.equals(ds.dataset(directory / f"{binding}-400000.parquet").head(25_000)):
            raise ValueError(f"{binding} 25k anchors differ from the 400k prefix")
        prefix = hashlib.sha256()
        for sid in small["sample_id"].to_pylist():
            prefix.update(json.dumps(sid, ensure_ascii=False).encode() + b"\n")
        identity["bindings"][binding] = {
            "encoder_cutoff_seconds": pc.max(rows["prediction_time_ns"]).as_py() // 10**9 - 1,
            "contract_sha256": digest(contract_path),
            "anchors_sha256": hashes,
            "prefix_sha256": prefix.hexdigest(),
        }
    (directory / "identity.json").write_text(json.dumps(identity) + "\n")
    return identity
