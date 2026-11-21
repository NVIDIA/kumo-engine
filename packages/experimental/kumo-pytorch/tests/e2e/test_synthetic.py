# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic Medium histories checked against their original source rows."""

import json
import os
from collections import Counter
from datetime import UTC, datetime
from operator import itemgetter
from pathlib import Path
from statistics import mean, pstdev, pvariance

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch

from e2e.datasets.synthetic import _events, load_synthetic, prepare_synthetic
from kumo.pytorch import api
from tests.e2e.history import check_normalized_aggregates, check_scalar_history, scalar_token
from tests.test_inference_lifecycle import assert_same

_SOURCE_FIELDS = {
    "card_txns": ("txn_id", ("amount", "mcc", "channel")),
    "transfers": ("transfer_id", ("amount", "transfer_type", "cross_border", "direction")),
    "logins": ("login_id", ("ip_region", "success")),
    "profile_changes": ("change_id", ("change_type",)),
    "disputes": ("dispute_id", ("reason",)),
    "lifelong_events": ("lifelong_id", ("event_type",)),
    "plan_events": ("plan_event_id", ("plan",)),
}
_event_identity = itemgetter("root_id", "event_id", "source_id")


def _check_sibling_transfer(tables, accounts, by_account):
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
    entity_type = "account" if by_account else "customer"
    mapped = _events("transfers", probe, tables, pa.array(sides), entity_type).to_pylist()
    source_id = original["transfer_id"]
    expected_sides = (
        [(account, 0, "outgoing"), (sibling, 1, "incoming")] if by_account else [(owner, 0, "self")]
    )
    assert sorted((*_event_identity(r), r["direction"]) for r in mapped) == sorted(
        (root, 2 * source_id + side, source_id, direction)
        for root, side, direction in expected_sides
    )
    return sibling if by_account else owner


def _source_histories(tables, dataset, roots, entity_type):
    by_account = entity_type == "account"
    accounts = {r["account_id"]: r["customer_id"] for r in tables["accounts"].to_pylist()}
    subset_root = (
        None if entity_type == "ltv" else _check_sibling_transfer(tables, accounts, by_account)
    )
    checked_roots = roots if subset_root is None else roots | {subset_root}

    def seconds(clock):
        return clock.replace(tzinfo=UTC).timestamp()

    root_key, clock_key = (
        ("account_id", "opened_at") if by_account else ("customer_id", "onboarded_at")
    )
    projected_roots = pq.read_table(dataset.tables["roots"].path).to_pylist()
    assert sorted((r["root_id"], seconds(r["available_at"])) for r in projected_roots) == sorted(
        (r[root_key], r[clock_key])
        for r in tables["accounts" if by_account else "customers"].to_pylist()
        if r[root_key] in roots
    )
    histories = {root: [] for root in roots}
    fields = {"source_kind": api.Vocabulary()}
    for family, (key, columns) in _SOURCE_FIELDS.items():
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
            for root in owners.keys() & checked_roots:
                direction, side = owners[root]
                values = {n: direction if n == "direction" else row[n] for n in columns}
                values.update(
                    source_kind=family,
                    event_id=2 * row[key] + side,
                    event_time_seconds=row["event_at"],
                )
                if root in roots:
                    histories[root].append((row["event_at"], values))
                # Full ownership has at most two accounts per customer. Direct events retain
                # that multiplier and the account's sibling slot even for a single root.
                expected.append((root, 2 * row[key] + side, row[key], row["event_at"], direction))
        native = pq.read_table(dataset.tables[family].path)
        projections = [(roots, native)]
        if subset_root is not None:
            subset = _events(family, tables[family], tables, pa.array([subset_root]), entity_type)
            if subset_root in roots:
                assert subset.equals(
                    native.filter(pc.equal(native["root_id"], subset_root)).cast(subset.schema)
                )
            projections.append(({subset_root}, subset))
        for selected, projected in projections:
            actual = [
                (*_event_identity(r), seconds(r["event_time"]), r.get("direction"))
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


def _check_ltv_graph(task, batch, histories):
    roots = list(batch.metadata["entity_ids"])
    expected = {
        (owner, row["source_kind"], row["event_id"]): row
        for owner, (_, history) in enumerate(histories)
        for _, row in history
    }
    graph = batch.inputs["graph"]
    nodes, edges = graph["nodes"], graph["edges"]
    keys = {}
    for family, node in nodes.items():
        ids, owners = node["ids"], node["batch"]
        assert ids.dtype == owners.dtype == torch.int64 and ids.ndim == owners.ndim == 1
        keys[family] = list(zip(owners.tolist(), ids.tolist(), strict=True))
    root_keys = keys.pop("roots")
    assert not nodes["roots"]["features"] and sorted(root_keys) == list(enumerate(roots))
    assert sorted((o, family, i) for family, rows in keys.items() for o, i in rows) == sorted(
        expected
    )
    assert set(edges) == {f"root_{family}" for family in keys}
    for family, members in keys.items():
        _, columns = _SOURCE_FIELDS[family]
        edge = edges[f"root_{family}"]
        assert (edge["source_type"], edge["destination_type"]) == ("roots", family)
        index = edge["index"]
        assert (
            index.dtype == torch.int64 and index.shape == (2, len(members)) and (index >= 0).all()
        )
        assert sorted((root_keys[a], members[b]) for a, b in index.T.tolist()) == sorted(
            ((owner, roots[owner]), (owner, i)) for owner, i in members
        )
        features = nodes[family]["features"]
        configs = {
            name: api.ZScore() if name == "amount" else api.Vocabulary()
            for name in ("source_kind", *columns)
        }
        configs["event_time_seconds"] = api.Raw()
        bindings = task.output("graph").bindings[family]
        assert {name: binding.encoder.config for name, binding in bindings.items()} == configs
        assert set(features) == set(configs)
        for name, binding in bindings.items():
            encoder = binding.encoder
            raw = [expected[owner, family, i][name] for owner, i in members]
            valid = torch.tensor([v is not None for v in raw], dtype=torch.bool)
            assert_same(features[name]["valid"], valid)
            values = [v for v in raw if v is not None]
            dtype = torch.int64
            if isinstance(encoder.config, api.ZScore):
                values = [(v - encoder.mean) / encoder.scale for v in values]
                dtype = torch.float32
            elif isinstance(encoder.config, api.Vocabulary):
                lookup = dict(encoder.entries)
                values = [lookup.get(v, encoder.config.unknown_id) for v in values]
            assert_same(features[name]["values"][valid], torch.tensor(values, dtype=dtype))


@pytest.mark.parametrize(
    "entity_type",
    ["customer", "account", "ltv"],
    ids=["s02-next-token", "s01-masked-values", "s04-positive-ltv"],
)
def test_registered_history(tmp_path, entity_type):
    ltv = entity_type == "ltv"
    root = os.environ.get("KUMO_PYTORCH_DATA_ROOT")
    if root is None:
        pytest.skip("set KUMO_PYTORCH_DATA_ROOT to the staged package datasets")
    tables = load_synthetic(Path(root), entity_type)
    anchors = {r["sample_id"]: r for r in tables["anchors"].to_pylist()}
    spec, dataset, table = prepare_synthetic(tables, tmp_path / "prepared", entity_type)
    # LTV executes at T-1s; weekly anchors execute at the prediction time T.
    assert {e.rows.time for e in spec.sample.events.values()} == {
        api.TimeWindow("event_time", stop=0 if ltv else -1)
    }
    histories, fields = _source_histories(
        tables, dataset, {int(r["entity_id"]) for r in anchors.values()}, entity_type
    )
    training = [s for s, r in anchors.items() if r["split"] == "train"]
    cutoff = max(anchors[s]["cutoff_epoch_seconds"] for s in training)
    if ltv:
        onboarded = {r["customer_id"]: r["onboarded_at"] for r in tables["customers"].to_pylist()}
        answers = {}
        for sample, row in anchors.items():
            customer, clock = row["customer_id"], row["cutoff_epoch_seconds"]
            assert clock == onboarded[customer] + 28 * 86400
            assert int(row["graph_as_of_time"].replace(tzinfo=UTC).timestamp()) == clock - 1
            assert row["label"] == (row["gross_profit"] > 0)
            answers[sample] = _ltv_answers(spec, clock, histories[customer])
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
            assert meta["as_of_time"].tolist() == [
                r["cutoff_epoch_seconds"] - int(ltv) for r in rows
            ]
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
                check_scalar_history(task, batch, fields, expected)
                if ltv:
                    assert set(batch.inputs) == {
                        "history",
                        "graph",
                        "aggregates",
                        "aggregate_checks",
                    }
                    assert set(batch.targets) == {"label"}
                    labels = torch.tensor(
                        [r["gross_profit"] > 0 for r in rows], dtype=torch.float32
                    )
                    assert_same(
                        batch.targets["label"],
                        {"values": labels, "valid": torch.ones_like(labels, dtype=torch.bool)},
                    )
                    _check_ltv_graph(task, batch, expected)
                    _check_ltv_aggregates(task, batch, answers, training)
                else:
                    assert set(batch.inputs) == {"history"} and set(batch.targets) == {"next_token"}


def _ltv_answers(spec, cutoff, history):
    answers = {}
    assert len(spec.sample.events) == 7
    for window, days in (("all_history", None), ("24h", 1), ("7d", 7), ("30d", 30), ("180d", 180)):
        start = -days * 86400 if days else None
        counts = Counter(
            row["source_kind"]
            for t, row in history
            if t < cutoff and (start is None or t >= cutoff + start)
        )
        for family in spec.sample.events:
            name = f"{window}_{family}_count"
            # At onboarding +28 days, lifetime, 180-day and 30-day counts coincide.
            assert spec.sample.entity_values[name].source == api.Rows(
                (f"root_{family}",), api.TimeWindow("event_time", start)
            )
            answers[name] = counts[family]

    for name, query in spec.sample.entity_values.items():
        if name not in answers and isinstance(query, api.Aggregate):
            source = query.source
            while isinstance(source, api.GroupBy):
                source = source.source
            assert source == api.Rows(
                ("root_card_txns",), api.TimeWindow("event_time", -30 * 86400)
            )
    cards = [
        (t, row)
        for t, row in history
        if cutoff - 30 * 86400 <= t < cutoff and row["source_kind"] == "card_txns"
    ]
    amounts = [row["amount"] for _, row in cards]
    mcc = Counter(row["mcc"] for _, row in cards)
    channels = Counter(row["channel"] for _, row in cards)
    pairs = Counter((row["mcc"], row["channel"]) for _, row in cards)
    for key, groups in (("mcc", mcc), ("channel", channels), ("mcc_channel", pairs)):
        answers[f"card_{key}_groups"] = len(groups)
        answers[f"card_{key}_max_count"] = max(groups.values(), default=None)
    recency = cutoff - max(t for t, _ in cards) if cards else None
    answers.update(
        card_amount_sum=sum(amounts),
        card_amount_mean=mean(amounts) if amounts else None,
        card_amount_min=min(amounts, default=None),
        card_amount_max=max(amounts, default=None),
        card_amount_std_population=pstdev(amounts) if amounts else None,
        card_amount_var_population=pvariance(amounts) if amounts else None,
        card_distinct_mcc=len(mcc),
        card_distinct_channel=len(channels),
        card_channel_mode=min(channels, key=lambda c: (-channels[c], c), default=None),
        card_mean_channels_per_mcc=len(pairs) / len(mcc) if mcc else None,
        card_recency_seconds=recency,
        card_recency_at_execution=recency - 1 if cards else None,
        prediction_offset=1.0,
        recent_card_share=answers["24h_card_txns_count"] / len(cards) if cards else None,
    )
    return answers


def _check_ltv_aggregates(task, batch, answers, training):
    mode = "card_channel_mode"
    samples = batch.metadata["sample_ids"]
    raw = batch.inputs["aggregate_checks"]
    assert raw.keys() == answers[samples[0]].keys()
    # One training customer has cards; the other has no mode.
    (train_mode,) = {answers[s][mode] for s in training} - {None}
    assert dict(task.output("aggregate_checks").field(name=mode).encoder.entries) == {train_mode: 2}
    for name, field in raw.items():
        encoder = task.output("aggregate_checks").field(name=name).encoder
        assert encoder.config == (api.Vocabulary() if name == mode else api.Raw())
        values = [answers[s][name] for s in samples]
        valid = torch.tensor([v is not None for v in values], dtype=torch.bool)
        torch.testing.assert_close(field["valid"], valid)
        expected = [v for v in values if v is not None]
        if name == mode:
            expected = [2 if v == train_mode else 1 for v in expected]
        integral = (
            name == mode
            or name.startswith("card_distinct_")
            or name.endswith(("_count", "_groups"))
        )
        expected = torch.tensor(expected, dtype=torch.int64 if integral else torch.float64)
        tol = 0 if integral else 2e-12
        torch.testing.assert_close(field["values"][valid], expected, rtol=tol, atol=tol, msg=name)
    check_normalized_aggregates(
        task,
        batch,
        answers,
        training,
        raw_only=(mode, "card_recency_at_execution", "prediction_offset"),
    )
