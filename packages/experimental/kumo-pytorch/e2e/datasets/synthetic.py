# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native weekly-customer next-token inputs for the Synthetic Medium S02 cohort."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from kumo.pytorch import api

S02_MANIFEST = Path(__file__).parent / "pins" / "synthetic-s02.json"

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


def _customer_events(family, raw, accounts, roots):
    key, fields = _FAMILIES[family]
    event_id = pc.multiply_checked(raw[key], 2)
    values = {name: raw[name] for name in fields}
    values.update(event_id=event_id, source_id=raw[key], source_kind=pa.repeat(family, len(raw)))
    values["event_time"] = raw["event_at"].cast(pa.timestamp("s"))

    def owner(column):
        indices = pc.index_in(raw[column], value_set=accounts["account_id"].combine_chunks())
        if pc.any(pc.and_(pc.is_valid(raw[column]), pc.is_null(indices))).as_py():
            raise ValueError(f"Unknown account in {family}.{column}")
        return accounts["customer_id"].take(indices)

    if family == "transfers":
        src, dst = owner("src_account_id"), owner("dst_account_id")
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
        events = pa.table(dict(values, root_id=root))
    return events.filter(pc.is_in(events["root_id"], value_set=roots))


def _contract(capacity):
    tables = [api.TableSpec("roots", "root_id", time_column="available_at")]
    relations, events = [], {}
    buckets = api.Buckets(bins=64)
    encoders = {"age_range": buckets}
    for family, (_, fields) in _FAMILIES.items():
        tables.append(api.TableSpec(family, "event_id", time_column="event_time"))
        relation = api.RelationSpec(f"root_{family}", "roots", "root_id", family, "root_id")
        relations.append(relation)
        names = ("source_kind", *fields)
        if family == "transfers":
            names += ("direction",)
        values = {name: api.Field(name) for name in names}
        encoders.update((name, buckets if name == "amount" else api.Vocabulary()) for name in names)
        values.update(__event_time=api.Field("event_time"), age_range=api.EventTimeDelta("seconds"))
        for name in _CALENDAR:
            values[name] = api.Derived(api.CalendarPart(name), ("__event_time",))
        events[family] = api.EventSpec(
            api.Rows((relation.name,), api.TimeWindow("event_time", stop=-1)), values
        )
    for name, domain in _CALENDAR.items():
        encoders[name] = api.Vocabulary(values=tuple((i, i + 2) for i in domain))
    stream = api.Fields(
        tuple(api.Encode(name, encoder, "history") for name, encoder in encoders.items()),
        row=api.Tokens(missing="error"),
        history=api.Flatten(
            (len(encoders) + 1) * capacity - 1, keep="first", separator="separator"
        ),
    )
    roles = {name: f"[{name.upper()}]" for name in ("entity", "event", "mask", "pad", "unknown")}
    return api.DataLoaderSpec(
        api.DataSchema(tuple(tables), tuple(relations)),
        api.SampleSpec(api.EntitySpec("roots", entity_type="customer"), events=events),
        api.BatchSpec(
            {"history": api.Input(api.EventSource(tuple(events)), stream, api.Padded())},
            {"next_token": api.NextTokenTarget("history")},
            id_spaces={"history": api.IdSpace(dict(roles, separator="[SEP]"))},
            batching=api.Batching(max_samples=32),
        ),
    )


def prepare_s02(tables, directory: Path):
    """Project verified original tables to a native dataset, contract and anchor table."""
    if directory.exists():
        raise FileExistsError(directory)
    anchors = tables["anchors"]
    if not anchors["customer_id"].cast(pa.string()).equals(anchors["entity_id"]):
        raise ValueError("Weekly anchor identity disagrees with its customer")
    roots = anchors["customer_id"].unique()
    customers = tables["customers"]
    customers = customers.filter(pc.is_in(customers["customer_id"], value_set=roots))
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent) as temp:
        work = Path(temp)
        available = customers["onboarded_at"].cast(pa.timestamp("s"))
        roots_table = pa.table({"root_id": customers["customer_id"], "available_at": available})
        pq.write_table(roots_table, work / "roots.parquet")
        capacity = 0
        for name in _FAMILIES:
            rows = _customer_events(name, tables[name], tables["accounts"], roots)
            pq.write_table(rows, work / f"{name}.parquet")
            capacity += max(Counter(rows["root_id"].to_pylist()).values(), default=1)
        anchors = anchors.append_column("registered_entity_id", anchors["entity_id"])
        anchors = anchors.append_column(
            "prediction_time_ns", anchors["as_of_time"].cast(pa.int64())
        )
        pq.write_table(anchors, work / "anchors.parquet")
        work.rename(directory)
    dataset = api.Dataset(
        {name: api.Parquet(str(directory / f"{name}.parquet")) for name in ("roots", *_FAMILIES)}
    )
    table = api.AnchorTable(
        api.Parquet(str(directory / "anchors.parquet")),
        "customer_id",
        "as_of_time",
        sample_id="sample_id",
        split="split",
        metadata=("registered_entity_id", "prediction_time_ns", "anchor_row_position"),
    )
    return _contract(capacity).resolve(dataset=dataset), dataset, table


def load_s02(files):
    """Read the pinned weekly-customer cohort, including events beyond its anchor times."""
    pin = json.loads(S02_MANIFEST.read_text())
    source = json.loads(files["source_manifest"].read_text())
    anchor = json.loads(files["anchor_manifest"].read_text())
    if anchor["definition"]["metadata"]["config_fingerprint"] != source["config_fingerprint"]:
        raise ValueError("Weekly-customer anchors and source use different dataset configurations")
    if anchor["files"][0]["sha256"] != pin["objects"]["anchors"]["sha256"]:
        raise ValueError("Weekly-customer anchor payload disagrees with its published manifest")
    for name, obj in pin["objects"].items():
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
    accounts = ds.dataset(paths["accounts"]).to_table(columns=["account_id", "customer_id"])
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
    if hashlib.sha256(encoded).hexdigest() != pin["event_ids_sha256"]:
        raise ValueError("S02 event IDs differ from the complete published customer histories")
    return tables
