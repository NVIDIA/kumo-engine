# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic Medium weekly histories checked against their original source rows."""

import json
import os
from datetime import UTC, datetime
from operator import itemgetter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch

from e2e.datasets.synthetic import _events, load_synthetic, prepare_synthetic
from kumo.pytorch import api
from tests.e2e.history import check_scalar_history, scalar_token


def _source_histories(tables, dataset, roots, entity_type):
    by_account = entity_type == "account"
    accounts = {r["account_id"]: r["customer_id"] for r in tables["accounts"].to_pylist()}
    # Real self transfers use a single account. Project a copy to a sibling account
    # to distinguish account directions from customer-level self ownership.
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
    sides = [account, sibling] if by_account else [owner]
    mapped = _events("transfers", probe, tables, pa.array(sides), entity_type).to_pylist()
    source_id = original["transfer_id"]
    identity = itemgetter("root_id", "event_id", "source_id")
    expected_sides = (
        [(account, 0, "outgoing"), (sibling, 1, "incoming")] if by_account else [(owner, 0, "self")]
    )
    assert sorted((*identity(r), r["direction"]) for r in mapped) == sorted(
        (root, 2 * source_id + side, source_id, direction)
        for root, side, direction in expected_sides
    )

    def seconds(clock):
        return clock.replace(tzinfo=UTC).timestamp()

    root_key, clock_key = (
        ("account_id", "opened_at") if by_account else ("customer_id", "onboarded_at")
    )
    projected_roots = pq.read_table(dataset.tables["roots"].path).to_pylist()
    assert sorted((r["root_id"], seconds(r["available_at"])) for r in projected_roots) == sorted(
        (r[root_key], r[clock_key])
        for r in tables[entity_type + "s"].to_pylist()
        if r[root_key] in roots
    )
    subset_root = sibling if by_account else owner
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
                sender, receiver = (
                    endpoints
                    if by_account
                    else [accounts[a] if a is not None else None for a in endpoints]
                )
                owners = (
                    {sender: ("outgoing", 0), receiver: ("incoming", 1)}
                    if sender != receiver
                    else {sender: ("self", 0)}
                )
            else:
                owner = row["customer_id"]
                if "account_id" in row:
                    assert accounts[row["account_id"]] == owner
                    members = [row["account_id"]] if by_account else [owner]
                else:
                    members = (
                        sorted(a for a, c in accounts.items() if c == owner)
                        if by_account
                        else [owner]
                    )
                owners = {member: (None, slot) for slot, member in enumerate(members)}
            for root in owners.keys() & (roots | {subset_root}):
                direction, side = owners[root]
                values = {n: direction if n == "direction" else row[n] for n in columns}
                values["source_kind"] = family
                if root in roots:
                    histories[root].append((row["event_at"], values))
                # Full ownership has at most two accounts per customer. Direct events retain
                # that multiplier and the account's sibling slot even for a single root.
                expected.append((root, 2 * row[key] + side, row[key], row["event_at"], direction))
        subset = _events(family, tables[family], tables, pa.array([subset_root]), entity_type)
        native = pq.read_table(dataset.tables[family].path)
        if subset_root in roots:
            assert subset.equals(
                native.filter(pc.equal(native["root_id"], subset_root)).cast(subset.schema)
            )
        for selected, projected in ((roots, native), ({subset_root}, subset)):
            actual = [
                (*identity(r), seconds(r["event_time"]), r.get("direction"))
                for r in projected.to_pylist()
            ]
            assert sorted(actual) == sorted(r for r in expected if r[0] in selected)
    return histories, fields


# Expected pieces for every distinct reason, fitted through the selected training clock.
_REASON_PIECES = {
    "item_not_as_described": ("item", "_", "not", "_", "as", "_", "described"),
    "not_received": ("not", "_", "r", "e", "c", "e", "i", "[UNK]", "e", "d"),
}


def _check_account_batch(task, batch, fields, cutoff, rows):
    space = task.spaces["history"]
    bindings = task.output("history").bindings
    assert {n: b.encoder.config for n, b in bindings.items()} == {
        **fields,
        "reason": api.Text(vocabulary_size=256, min_frequency=2, empty="empty"),
    }
    text = bindings["reason"].encoder
    vocab = json.loads(text.tokenizer_json)["model"]["vocab"]
    assert "v" not in vocab  # First appears after training, in not_received.
    names = space.members['["fields","fields"]']
    field_ids = {
        (family, name): names[i]
        for (_, family, name), i in task.fitted_state["field_names"]["fields"]
    }
    packed, target = batch.inputs["history"], batch.targets["masked_values"]
    times = sorted(t for t, _ in rows)
    # Each tied event in this extract belongs to a different family. Its first field
    # identifies the family even when the value itself was replaced.
    records = {(t, field_ids[v["source_kind"], "source_kind"]): v for t, v in rows}
    assert len(records) == len(rows)
    seq = packed["sequence"]
    assert seq["valid"].tolist() == [[True] * len(rows)]
    assert seq["event_indices"].tolist() == [list(range(len(rows)))]
    assert packed["owner"].tolist() == [0] * len(rows)
    assert packed["history_position"].tolist() == list(range(len(rows)))
    assert packed["valid"].all() and packed["field_valid"].all()
    dates = [datetime.fromtimestamp(t, UTC) for t in times]
    for name, expected in (
        ("age_seconds", [cutoff - t for t in times]),
        ("hour", [d.hour for d in dates]),
        ("weekday", [d.weekday() for d in dates]),
        ("day", [d.day for d in dates]),
    ):
        channel = batch.inputs["history_timing" if name == "age_seconds" else "history_calendar"]
        assert torch.equal(channel["valid"], seq["valid"]) and torch.equal(
            channel[name]["valid"], seq["valid"]
        )
        assert channel[name]["values"].tolist() == [expected]

    originals, keys, positions, offsets = [], [], [], [0]
    first_fields = packed["field_ids"][packed["field_offsets"][:-1]].tolist()
    for clock, key in zip(times, first_fields, strict=True):
        row = records.pop((clock, key))
        family = row["source_kind"]
        for name in fields:
            if (value := row.get(name)) is None:
                continue
            if name == "reason":
                pieces = [space.members[text.identity][vocab[p]] for p in _REASON_PIECES[value]]
            else:
                pieces = [scalar_token(bindings[name].encoder, space, value)]
            originals.extend(pieces)
            keys.extend([field_ids[family, name]] * len(pieces))
            positions.extend(range(len(pieces)))
        offsets.append(len(originals))
    assert packed["field_offsets"].tolist() == offsets and target["offsets"].tolist() == offsets
    assert packed["field_ids"].tolist() == keys and packed["position_ids"].tolist() == positions
    before, after = torch.tensor(originals), packed["value_ids"]
    supervised = after == space.special(role="mask")
    unknown = after == space.special(role="unknown")
    assert ((after == before) | supervised | unknown).all()
    if batch.metadata["splits"][0] == "test":
        reason = packed["field_ids"] == field_ids["disputes", "reason"]
        assert (reason & supervised).any()
        assert (reason & unknown & (after != before)).any()  # Unknown replacement.
        assert (reason & unknown & (after == before)).any()  # Original held-out [UNK].
    assert supervised.any() and torch.equal(target["valid"], supervised)
    assert torch.equal(target["values"], torch.where(supervised, before, -100))
    assert torch.equal(target["corruption"], before != after)


@pytest.mark.parametrize(
    "entity_type", ["customer", "account"], ids=["s02-next-token", "s01-masked-values"]
)
def test_weekly_history(tmp_path, entity_type):
    root = os.environ.get("KUMO_PYTORCH_DATA_ROOT")
    if root is None:
        pytest.skip("set KUMO_PYTORCH_DATA_ROOT to the staged package datasets")
    tables = load_synthetic(Path(root), entity_type)
    anchors = {r["sample_id"]: r for r in tables["anchors"].to_pylist()}
    spec, dataset, table = prepare_synthetic(tables, tmp_path / "prepared", entity_type)
    # No selected event lands exactly at a registered weekly cutoff.
    assert {e.rows.time for e in spec.sample.events.values()} == {
        api.TimeWindow("event_time", stop=-1)
    }
    histories, fields = _source_histories(
        tables, dataset, {int(r["entity_id"]) for r in anchors.values()}, entity_type
    )
    cutoff = max(r["cutoff_epoch_seconds"] for r in anchors.values() if r["split"] == "train")
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with (
        api.prepare(spec, dataset, engine) as plan,
        plan.materialize(table, fit_split="train", encoder_cutoff=cutoff - 1) as task,
    ):
        for split in ("train", "valid", "test"):
            (batch,) = task.batches(
                split=split, batch_size=len(anchors), output_format="model", seed=0
            )
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
            if entity_type == "account":
                (sample,) = expected
                _check_account_batch(task, batch, fields, *sample)
            else:
                assert set(batch.inputs) == {"history"} and set(batch.targets) == {"next_token"}
                check_scalar_history(task, batch, fields, expected)
