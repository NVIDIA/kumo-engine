# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic Medium weekly-customer histories checked against their original source rows."""

import os
from datetime import UTC
from operator import itemgetter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from e2e.datasets.staging import verify
from e2e.datasets.synthetic import S02_MANIFEST, _customer_events, load_s02, prepare_s02
from kumo.pytorch import api
from tests.e2e.history import check_scalar_history


def _customer_histories(tables, dataset, roots):
    accounts = {r["account_id"]: r["customer_id"] for r in tables["accounts"].to_pylist()}
    # Real self transfers use a single account. Project a copy to a sibling account
    # to check that two distinct endpoints still emit one event for their customer.
    transfers = tables["transfers"]
    original = next(
        r
        for r in transfers.to_pylist()
        if r["src_account_id"] is not None and r["src_account_id"] == r["dst_account_id"]
    )
    account = original["src_account_id"]
    owner = accounts[account]
    sibling = next(a for a, c in accounts.items() if c == owner and a != account)
    probe = pa.Table.from_pylist([dict(original, dst_account_id=sibling)], schema=transfers.schema)
    one_root = pa.array([owner])
    (mapped,) = _customer_events("transfers", probe, tables["accounts"], one_root).to_pylist()
    source_id = original["transfer_id"]
    identity = itemgetter("root_id", "event_id", "source_id")
    assert identity(mapped) == (owner, 2 * source_id, source_id) and mapped["direction"] == "self"

    def seconds(clock):
        return clock.replace(tzinfo=UTC).timestamp()

    projected_roots = pq.read_table(dataset.tables["roots"].path).to_pylist()
    assert sorted((r["root_id"], seconds(r["available_at"])) for r in projected_roots) == sorted(
        (r["customer_id"], r["onboarded_at"]) for r in tables["customers"].to_pylist()
    )
    histories = {root: [] for root in roots}
    fields = {"source_kind": api.Vocabulary()}
    for family, key, columns in (
        ("card_txns", "txn_id", ("amount", "mcc", "channel")),
        ("transfers", "transfer_id", ("amount", "transfer_type", "cross_border", "direction")),
        ("logins", "login_id", ("ip_region", "success")),
        ("profile_changes", "change_id", ("change_type",)),
        ("disputes", "dispute_id", ("reason",)),
        ("lifelong_events", "lifelong_id", ("event_type",)),
        ("plan_events", "plan_event_id", ("plan",)),
    ):
        fields.update(
            (n, api.Buckets(bins=64) if n == "amount" else api.Vocabulary()) for n in columns
        )
        expected = []
        for row in tables[family].to_pylist():
            if family == "transfers":
                endpoints = (row["src_account_id"], row["dst_account_id"])
                sender, receiver = (accounts[a] if a is not None else None for a in endpoints)
                owners = (
                    {sender: "outgoing", receiver: "incoming"}
                    if sender != receiver
                    else {sender: "self"}
                )
            else:
                root = row["customer_id"]
                if family in ("card_txns", "disputes"):
                    assert accounts[row["account_id"]] == root
                owners = {root: None}
            for root in owners.keys() & roots:
                direction = owners[root]
                values = {n: direction if n == "direction" else row[n] for n in columns}
                values["source_kind"] = family
                histories[root].append((row["event_at"], values))
                event_id = 2 * row[key] + int(direction == "incoming")
                expected.append((root, event_id, row[key], row["event_at"], direction))
        projected = pq.read_table(dataset.tables[family].path)
        subset = _customer_events(family, tables[family], tables["accounts"], one_root)
        assert subset.equals(
            projected.filter(pc.equal(projected["root_id"], owner)).cast(subset.schema)
        )
        actual = [
            (*identity(r), seconds(r["event_time"]), r.get("direction"))
            for r in projected.to_pylist()
        ]
        assert sorted(actual) == sorted(expected)
    return histories, fields


def test_weekly_customer_next_token(tmp_path):
    root = os.environ.get("KUMO_PYTORCH_DATA_ROOT")
    if root is None:
        pytest.skip("set KUMO_PYTORCH_DATA_ROOT to the staged package datasets")
    tables = load_s02(verify(Path(root), S02_MANIFEST))
    anchors = {r["sample_id"]: r for r in tables["anchors"].to_pylist()}
    spec, dataset, table = prepare_s02(tables, tmp_path / "prepared")
    # No selected event lands exactly at a registered weekly cutoff.
    assert {e.rows.time for e in spec.sample.events.values()} == {
        api.TimeWindow("event_time", stop=-1)
    }
    histories, fields = _customer_histories(
        tables, dataset, {int(r["entity_id"]) for r in anchors.values()}
    )
    cutoff = max(r["cutoff_epoch_seconds"] for r in anchors.values() if r["split"] == "train")
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with (
        api.prepare(spec, dataset, engine) as plan,
        plan.materialize(table, fit_split="train", encoder_cutoff=cutoff - 1) as task,
    ):
        for split in ("train", "valid", "test"):
            (batch,) = task.batches(split=split, batch_size=len(anchors), output_format="model")
            meta = batch.metadata
            samples = meta["sample_ids"]
            assert sorted(samples) == sorted(s for s, r in anchors.items() if r["split"] == split)
            rows = [anchors[s] for s in samples]
            assert list(meta["splits"]) == [split] * len(rows)
            assert list(meta["entity_ids"]) == [int(r["entity_id"]) for r in rows]
            assert meta["as_of_time"].tolist() == [r["cutoff_epoch_seconds"] for r in rows]
            assert not any(meta["is_padding"])
            expected = []
            for row, metadata in zip(rows, meta["request_metadata"], strict=True):
                root, clock = int(row["entity_id"]), row["cutoff_epoch_seconds"]
                assert metadata["anchor_row_position"] == row["anchor_row_position"]
                assert metadata["registered_entity_id"] == row["entity_id"]
                assert metadata["prediction_time_ns"] == clock * 10**9
                expected.append((clock, [(t, v) for t, v in histories[root] if t < clock]))
            assert set(batch.inputs) == {"history"} and set(batch.targets) == {"next_token"}
            check_scalar_history(task, batch, fields, expected)
