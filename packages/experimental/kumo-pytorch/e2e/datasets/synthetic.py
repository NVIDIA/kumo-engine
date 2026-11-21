# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native weekly-account and weekly-customer inputs from the Synthetic Medium cohort."""

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from e2e.datasets import bind_tables
from e2e.datasets.staging import verify
from kumo.pytorch import api
from tests.reference_models.batch_definitions import PRAGMA_BATCH

_PINS = Path(__file__).parent / "pins"
_SOURCE = _PINS / "synthetic-medium.json"
_ANCHORS = {kind: _PINS / f"synthetic-weekly-{kind}.json" for kind in ("account", "customer")}

# Physical identity and public input fields, in the S02 declaration order.
_FAMILIES = {
    "card_txns": ("txn_id", ("amount", "mcc", "channel")),
    "transfers": ("transfer_id", ("amount", "transfer_type", "cross_border")),
    "logins": ("login_id", ("ip_region", "success")),
    "profile_changes": ("change_id", ("change_type",)),
    "disputes": ("dispute_id", ("reason",)),
    "lifelong_events": ("lifelong_id", ("event_type",)),
    "plan_events": ("plan_event_id", ("plan",)),
}
_CALENDAR = {"hour": range(24), "weekday": range(7), "day": range(1, 32)}


def _events(family, raw, tables, roots, entity_type):
    key, fields = _FAMILIES[family]
    accounts = tables["accounts"]
    account_view = entity_type == "account"
    if account_view and family != "transfers" and "account_id" not in raw.column_names:
        members = accounts.filter(pc.is_in(accounts["account_id"], value_set=roots))
        raw = raw.join(
            members.select(["customer_id", "account_id", "slot"]),
            "customer_id",
            join_type="inner",
            use_threads=False,
        )
        event_id = pc.add_checked(pc.multiply_checked(raw[key], tables["fanout"]), raw["slot"])
    else:
        event_id = pc.multiply_checked(raw[key], 2)
    values = {name: raw[name] for name in fields}
    values.update(event_id=event_id, source_id=raw[key], source_kind=pa.repeat(family, len(raw)))
    values["event_time"] = raw["event_at"].cast(pa.timestamp("s"))

    def owner(column, *, by_account=False):
        indices = pc.index_in(raw[column], value_set=accounts["account_id"].combine_chunks())
        if pc.any(pc.and_(pc.is_valid(raw[column]), pc.is_null(indices))).as_py():
            raise ValueError(f"Unknown account in {family}.{column}")
        return raw[column] if by_account else accounts["customer_id"].take(indices)

    if family == "transfers":
        src, dst = (
            owner(col, by_account=account_view) for col in ("src_account_id", "dst_account_id")
        )
        same = pc.fill_null(pc.equal(src, dst), False)
        outgoing = pa.table(
            dict(values, root_id=src, direction=pc.if_else(same, "self", "outgoing"))
        )
        values.update(event_id=pc.add_checked(event_id, 1), root_id=dst)
        incoming = pa.table(dict(values, direction=pa.repeat("incoming", len(raw))))
        events = pa.concat_tables([outgoing, incoming.filter(pc.invert(same))])
    else:
        root = raw["customer_id"]
        if "account_id" in raw.column_names and not owner("account_id").equals(root):
            raise ValueError(f"Account ownership disagrees with {family}.customer_id")
        events = pa.table(dict(values, root_id=raw["account_id"] if account_view else root))
    return events.filter(pc.is_in(events["root_id"], value_set=roots))


def _pragma_batch(events, encoders, roles):
    encoders = dict(encoders, reason=api.Text(vocabulary_size=256, min_frequency=2, empty="empty"))
    fields = api.Fields(
        tuple(api.Encode(name, encoder, "history") for name, encoder in encoders.items()),
        row=api.Tokens(),
        field_names="fields",
        field_space="history",
        forbidden_fields=("entity_id", "sample_id", "label", "split"),
    )
    history = replace(
        PRAGMA_BATCH.inputs["history"], source=api.EventSource(tuple(events)), representation=fields
    )
    masked = replace(PRAGMA_BATCH.targets["masked_values"], fields=tuple(encoders))
    return api.BatchSpec(
        {
            "history": history,
            "history_calendar": PRAGMA_BATCH.inputs["history_calendar"],
            "history_timing": PRAGMA_BATCH.inputs["history_timing"],
        },
        {"masked_values": masked},
        id_spaces={"history": api.IdSpace(roles)},
        field_names=PRAGMA_BATCH.field_names,
        batching=api.Batching(max_samples=32),
    )


def _contract(capacity, entity_type):
    tables = [api.TableSpec("roots", "root_id", time_column="available_at")]
    relations, events = [], {}
    buckets = api.Buckets(bins=64)
    account = entity_type == "account"
    encoders = {} if account else {"age_range": buckets}
    for family, (_, fields) in _FAMILIES.items():
        tables.append(api.TableSpec(family, "event_id", time_column="event_time"))
        relation = api.RelationSpec(f"root_{family}", "roots", "root_id", family, "root_id")
        relations.append(relation)
        names = ("source_kind", *fields)
        if family == "transfers":
            names += ("direction",)
        values = {name: api.Field(name) for name in names}
        encoders.update((name, buckets if name == "amount" else api.Vocabulary()) for name in names)
        values["__event_time"] = api.Field("event_time")
        values["age_seconds" if account else "age_range"] = api.EventTimeDelta("seconds")
        for name in _CALENDAR:
            values[name] = api.Derived(api.CalendarPart(name), ("__event_time",))
        events[family] = api.EventSpec(
            api.Rows((relation.name,), api.TimeWindow("event_time", stop=-1)), values
        )
    schema = api.DataSchema(tuple(tables), tuple(relations))
    sample = api.SampleSpec(api.EntitySpec("roots", entity_type=entity_type), events=events)
    roles = {name: f"[{name.upper()}]" for name in ("entity", "event", "mask", "pad", "unknown")}
    if account:
        return api.DataLoaderSpec(schema, sample, _pragma_batch(events, encoders, roles))
    for name, domain in _CALENDAR.items():
        encoders[name] = api.Vocabulary(values=tuple((i, i + 2) for i in domain))
    stream = api.Fields(
        tuple(api.Encode(name, encoder, "history") for name, encoder in encoders.items()),
        row=api.Tokens(missing="error"),
        history=api.Flatten(
            (len(encoders) + 1) * capacity - 1, keep="first", separator="separator"
        ),
    )
    return api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            {"history": api.Input(api.EventSource(tuple(events)), stream, api.Padded())},
            {"next_token": api.NextTokenTarget("history")},
            id_spaces={"history": api.IdSpace(dict(roles, separator="[SEP]"))},
            batching=api.Batching(max_samples=32),
        ),
    )


def prepare_synthetic(tables, directory: Path, entity_type):
    """Project verified original tables to a native dataset, contract and anchor table."""
    if directory.exists():
        raise FileExistsError(directory)
    anchors = tables["anchors"]
    root_key = f"{entity_type}_id"
    clock = {"account": "opened_at", "customer": "onboarded_at"}[entity_type]
    if not anchors[root_key].cast(pa.string()).equals(anchors["entity_id"]):
        raise ValueError(f"Weekly anchor identity disagrees with its {entity_type}")
    roots = anchors[root_key].unique()
    entities = tables[f"{entity_type}s"]
    entities = entities.filter(pc.is_in(entities[root_key], value_set=roots))
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent) as temp:
        work = Path(temp)
        available = entities[clock].cast(pa.timestamp("s"))
        roots_table = pa.table({"root_id": entities[root_key], "available_at": available})
        pq.write_table(roots_table, work / "roots.parquet")
        capacity = 0
        for name in _FAMILIES:
            rows = _events(name, tables[name], tables, roots, entity_type)
            pq.write_table(rows, work / f"{name}.parquet")
            capacity += max(Counter(rows["root_id"].to_pylist()).values(), default=1)
        anchors = anchors.append_column("registered_entity_id", anchors["entity_id"])
        anchors = anchors.append_column(
            "prediction_time_ns", anchors["as_of_time"].cast(pa.int64())
        )
        pq.write_table(anchors, work / "anchors.parquet")
        work.rename(directory)
    return bind_tables(directory, _contract(capacity, entity_type), root_key)


def load_synthetic(root, entity_type):
    """Load the shared verified source shards and this workload's weekly anchors."""
    return read_synthetic(verify(root, _SOURCE) | verify(root, _ANCHORS[entity_type]), entity_type)


def read_synthetic(files, entity_type):
    """Read a cohort from verified files, including any extra shards in a full replay."""
    source_pin = json.loads(_SOURCE.read_text())
    pin = json.loads(_ANCHORS[entity_type].read_text())
    source = json.loads(files["source_manifest"].read_text())
    anchor = json.loads(files["anchor_manifest"].read_text())
    if anchor["definition"]["metadata"]["config_fingerprint"] != source["config_fingerprint"]:
        raise ValueError("Weekly anchors and source use different dataset configurations")
    if anchor["files"][0]["sha256"] != pin["objects"]["anchors"]["sha256"]:
        raise ValueError("Weekly anchor payload disagrees with its published manifest")
    for name, obj in source_pin["objects"].items():
        if "/" in name and source["files"][name] != obj["sha256"]:
            raise ValueError(f"Shard disagrees with the Synthetic Medium source manifest: {name}")
    paths = {}
    for name, path in files.items():
        if "/" in name:
            paths.setdefault(name.split("/")[0], []).append(path)

    positions = pin["anchor_rows"]
    columns = [
        "sample_id",
        "entity_id",
        "customer_id",
        "as_of_time",
        "cutoff_epoch_seconds",
        "split",
    ]
    if entity_type == "account":
        columns.append("account_id")
    anchors = ds.dataset(files["anchors"]).take(positions, columns=columns)
    anchors = anchors.append_column("anchor_row_position", pa.array(positions))
    roots = anchors["customer_id"].unique()
    customer = ds.field("customer_id").isin(roots)
    tables = {
        "anchors": anchors,
        "customers": ds.dataset(paths["customers"]).to_table(
            columns=["customer_id", "onboarded_at"], filter=customer
        ),
    }
    accounts = ds.dataset(paths["accounts"]).to_table(
        columns=["account_id", "customer_id", "opened_at"]
    )
    if entity_type == "account":
        # Rank siblings in the complete ownership table before selecting the cohort.
        accounts = accounts.sort_by([("customer_id", "ascending"), ("account_id", "ascending")])
        _, starts, counts = np.unique(
            accounts["customer_id"].to_numpy(), return_index=True, return_counts=True
        )
        accounts = accounts.append_column(
            "slot", pa.array(np.arange(len(accounts)) - np.repeat(starts, counts))
        )
        tables["fanout"] = int(counts.max())
        owner_index = pc.index_in(
            anchors["account_id"], value_set=accounts["account_id"].combine_chunks()
        )
        if not accounts["customer_id"].take(owner_index).equals(anchors["customer_id"]):
            raise ValueError("Weekly-account anchor disagrees with its registered owner")
    owned = accounts.filter(pc.is_in(accounts["customer_id"], value_set=roots))["account_id"]
    for family, (key, features) in _FAMILIES.items():
        owners, eligible = ["customer_id"], customer
        if family == "transfers":
            owners = ["src_account_id", "dst_account_id"]
            eligible = ds.field(owners[0]).isin(owned) | ds.field(owners[1]).isin(owned)
        elif family in ("card_txns", "disputes"):
            owners.append("account_id")
            eligible = ds.field("account_id").isin(owned)
        tables[family] = ds.dataset(paths[family]).to_table(
            columns=[key, "event_at", *owners, *features], filter=eligible
        )
    transfers = tables["transfers"]
    needed = [owned, transfers["src_account_id"], transfers["dst_account_id"]]
    account_ids = pa.concat_arrays([values.combine_chunks() for values in needed])
    tables["accounts"] = accounts.filter(pc.is_in(accounts["account_id"], value_set=account_ids))
    # Reference IDs were computed by scanning all 1,610 original shards for these roots.
    ids = {
        family: sorted(tables[family][key].to_pylist()) for family, (key, _) in _FAMILIES.items()
    }
    encoded = json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != source_pin["event_ids_sha256"]:
        raise ValueError(
            "Synthetic event IDs differ from the complete published customer histories"
        )
    return tables
