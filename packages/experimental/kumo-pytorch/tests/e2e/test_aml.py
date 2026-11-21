# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AML account-day forecasts checked against the original transaction rows."""

import json
import os
from bisect import bisect_right
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from itertools import accumulate
from pathlib import Path
from statistics import mean, pstdev
from typing import NamedTuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import torch

from e2e.datasets.aml import E01_MANIFEST, L01_MANIFEST, prepare_e01, prepare_l01
from e2e.datasets.staging import verify
from kumo.pytorch import api
from tests.e2e.history import check_scalar_history
from tests.test_inference_lifecycle import assert_same


class Transfer(NamedTuple):
    sender: int
    receiver: int
    time: int
    flagged: bool
    features: dict


def _source_rows(pin, dataset, positions, *targets):
    root = os.environ.get("KUMO_PYTORCH_DATA_ROOT")
    if root is None:
        pytest.skip("set KUMO_PYTORCH_DATA_ROOT to the staged package datasets")
    files = verify(Path(root), pin)
    # Check that the three published manifests describe the same pinned population.
    pins = json.loads(pin.read_text())["objects"]
    manifest = json.loads(files["anchor_manifest"].read_text())
    binding = json.loads(files["mapping_manifest"].read_text())
    definition = manifest["definition"]
    keys = ("dataset_id", "dataset_version", "task_id", "anchor_version")
    expected = (f"aml-data/{dataset}", "v8", "account-daily", "v2")
    assert tuple(definition[k] for k in keys) == expected
    assert definition["metadata"]["source_binding"] == binding
    assert binding["sha256"]["accounts.parquet"] == pins["mapping"]["sha256"]
    source = json.loads(files["source_manifest"].read_text())
    (tx,) = (f for f in source["files"] if f["path"] == f"{dataset}/transactions.parquet")
    (original,) = (f for f in binding["sources"] if f["path"] == tx["path"])
    assert tx["sha256"] == original["expected_sha256"] == pins["transactions"]["sha256"]
    assert tx["rows"] == pq.read_metadata(files["transactions"]).num_rows
    assert manifest["files"][0]["sha256"] == pins["anchors"]["sha256"]
    assert manifest["statistics"]["rows"] == pq.read_metadata(files["anchors"]).num_rows

    columns = ["sample_id", "entity_id", "as_of_time", "graph_as_of_time"]
    columns += ["forecast_split", *targets]
    rows = ds.dataset(files["anchors"]).take(positions, columns=columns).to_pylist()
    return files, dict(zip(positions, rows, strict=True))


def _source_history(files, anchors):
    # The source row number is the transaction ID; the sorted account binding supplies
    # account IDs. Compute the split boundaries from *all* transactions, including ties.
    mapping = pq.read_table(
        files["mapping"], columns=["entity_id", "normalized_bank_id", "account_number"]
    ).sort_by("entity_id")
    roots = [row["entity_id"] for row in anchors.values()]
    ranks = pc.index_in(pa.array(roots), value_set=mapping["entity_id"].combine_chunks())
    account_ids = dict(zip(roots, ranks.to_pylist(), strict=True))
    numbers = mapping["account_number"].take(ranks)
    pairs = zip(
        mapping["normalized_bank_id"].to_pylist(),
        mapping["account_number"].to_pylist(),
        strict=True,
    )
    by_account = {pair: i for i, pair in enumerate(pairs)}

    def endpoint(row, role):
        return by_account[str(int(row[f"{role}_bank"])), row[f"{role}_account"]]

    amounts = ("amount_paid", "amount_received")
    categories = ("payment_currency", "receiving_currency", "payment_format")
    columns = ["timestamp", "from_bank", "from_account", "to_bank", "to_account"]
    columns += ["is_laundering", *amounts, *categories]
    timeline = Counter()
    events = {}
    offset = 0
    for batch in pq.ParquetFile(files["transactions"]).iter_batches(columns=columns):
        for stamp in pc.value_counts(batch["timestamp"]).to_pylist():
            timeline[stamp["values"]] += stamp["counts"]
        indices = pc.indices_nonzero(
            pc.or_(pc.is_in(batch["from_account"], numbers), pc.is_in(batch["to_account"], numbers))
        )
        for index, row in zip(indices.to_pylist(), batch.take(indices).to_pylist(), strict=True):
            t = _seconds(row["timestamp"])
            features = {name: row[name] for name in categories}
            features.update(
                {name: None if row[name] is None else Decimal(row[name]) for name in amounts}
            )
            events[offset + index] = Transfer(
                endpoint(row, "from"),
                endpoint(row, "to"),
                t,
                row["is_laundering"] == "1",
                {**features, "event_time_seconds": t},
            )
        offset += len(batch)
    stamps = sorted(timeline)
    cumulative = list(accumulate(timeline[t] for t in stamps))
    boundaries = [_seconds(stamps[bisect_right(cumulative, offset * p // 10)]) for p in (6, 8)]
    return account_ids, events, [*boundaries, _seconds(stamps[-1])]


def _seconds(stamp):
    return int(datetime.strptime(stamp, "%Y/%m/%d %H:%M").replace(tzinfo=UTC).timestamp())


def _check_graph(task, batch, histories):
    graph = batch.inputs["graph"]
    nodes = graph["nodes"]
    assert set(nodes) == {"accounts", "transactions"} and not nodes["accounts"]["features"]
    assert set(graph["edges"]) == {"sender", "receiver"}
    for node in nodes.values():
        assert node["batch"].dtype == node["ids"].dtype == torch.int64
        assert node["batch"].ndim == node["ids"].ndim == 1
    accounts, transactions = (
        list(zip(node["batch"].tolist(), node["ids"].tolist(), strict=True))
        for node in (nodes["accounts"], nodes["transactions"])
    )
    events = {
        (owner, i): e for owner, (_, _, rows) in enumerate(histories) for i, e in rows.items()
    }
    members = {(owner, root) for owner, (root, _, _) in enumerate(histories)}
    for (owner, _), event in events.items():
        members.update(((owner, event.sender), (owner, event.receiver)))
    assert sorted(accounts) == sorted(members)
    assert sorted(transactions) == sorted(events)
    for role, edge in graph["edges"].items():
        assert (edge["source_type"], edge["destination_type"]) == ("accounts", "transactions")
        indices = edge["index"]
        assert indices.dtype == torch.int64 and indices.shape[0] == 2 and (indices >= 0).all()
        actual = [(accounts[a], transactions[t]) for a, t in indices.T.tolist()]
        expected = [((owner, getattr(e, role)), (owner, i)) for (owner, i), e in events.items()]
        assert sorted(actual) == sorted(expected)

    bindings = task.output("graph").bindings["transactions"]
    configs = {
        "amount_paid": api.ZScore(),
        "amount_received": api.ZScore(),
        "payment_currency": api.Vocabulary(),
        "receiving_currency": api.Vocabulary(),
        "payment_format": api.Vocabulary(),
        "event_time_seconds": api.Raw(),
    }
    assert {name: field.encoder.config for name, field in bindings.items()} == configs
    expected_features = {
        name: binding.encode_values([events[key].features[name] for key in transactions])
        for name, binding in bindings.items()
    }
    assert_same(nodes["transactions"]["features"], expected_features)


def test_hi_small_e01_graph_and_forecast(tmp_path):
    # Split boundary, covered negative, sender-only and receiver-only positives, end of observation.
    files, anchors = _source_rows(
        E01_MANIFEST, "hi-small", (0, 313, 4245, 11800, 409216), "involvement"
    )
    account_ids, events, ends = _source_history(files, anchors)
    spec, data, table = prepare_e01(files, tmp_path / "prepared", anchor_rows=list(anchors))
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with (
        api.prepare(spec, data, engine) as plan,
        plan.materialize(table, fit_split="train") as task,
    ):
        # Two immature forecasts are excluded; each named split has one eligible anchor.
        assert task.count() == 3
        for pos, row in anchors.items():
            root_id = account_ids[row["entity_id"]]
            cutoff = int(row["as_of_time"].replace(tzinfo=UTC).timestamp())
            assert int(row["graph_as_of_time"].replace(tzinfo=UTC).timestamp()) == cutoff - 1
            stage = bisect_right(ends[:2], cutoff)
            split = ("train", "valid", "test")[stage]
            touching = {i: e for i, e in events.items() if root_id in (e.sender, e.receiver)}
            history = {i: e for i, e in touching.items() if e.time < cutoff}
            assert history
            if cutoff + 86400 > ends[stage]:
                assert row["forecast_split"] is None and row["involvement"] is None
                continue
            label = any(e.flagged for e in touching.values() if cutoff <= e.time < cutoff + 86400)
            assert (row["forecast_split"], row["involvement"]) == (split, label)
            (batch,) = task.batches(split=split, batch_size=4, output_format="model")
            assert tuple(batch.metadata["sample_ids"]) == (row["sample_id"],)
            assert list(batch.metadata["entity_ids"]) == [root_id]
            assert batch.metadata["as_of_time"].tolist() == [cutoff - 1]
            metadata = batch.metadata["request_metadata"][0]
            assert metadata["anchor_row_position"] == pos
            assert metadata["registered_entity_id"] == row["entity_id"]
            assert metadata["prediction_time_ns"] == cutoff * 10**9
            assert not any(batch.metadata["is_padding"])
            assert batch.targets["label"]["valid"].tolist() == [True]
            assert batch.targets["label"]["values"].tolist() == [label]
            _check_graph(task, batch, [(root_id, cutoff, history)])


def _check_history(task, batch, histories):
    buckets, category = api.Buckets(bins=64), api.Vocabulary()
    fields = {
        "direction": category,
        "amount_paid": buckets,
        "payment_currency": category,
        "amount_received": buckets,
        "receiving_currency": category,
        "payment_format": category,
    }
    rows = []
    for root, cutoff, history in histories:
        events = []
        for event in history.values():
            direction = "outgoing" if event.sender == root else "incoming"
            features = dict(
                event.features, direction="self" if event.sender == event.receiver else direction
            )
            events.append((event.time, features))
        rows.append((cutoff, events))
    check_scalar_history(task, batch, fields, rows)


def _check_delivered_targets(batch, expected, currency_ids):
    samples = batch.metadata["sample_ids"]
    assert sorted(samples) == sorted(expected)
    amounts, currencies = zip(*(expected[s] for s in samples), strict=True)
    amount, currency = batch.targets["amount"], batch.targets["currency"]
    assert_same(amount["valid"], torch.tensor([v is not None for v in amounts]))
    assert_same(currency["valid"], torch.tensor([v is not None for v in currencies]))
    assert_same(
        amount["values"][amount["valid"]],
        torch.tensor([float(v) for v in amounts if v is not None], dtype=torch.float32),
    )
    ids = [0 if value is None else currency_ids.get(value, 1) for value in currencies]
    assert_same(currency["values"], torch.tensor(ids, dtype=torch.int64))


def _source_counts(spec, root, cutoff, history):
    counts = {}
    for window, days in (("all_history", None), ("24h", 1), ("7d", 7), ("30d", 30)):
        start = None if days is None else -days * 86400
        rows = [e for e in history.values() if start is None or e.time >= cutoff + start]
        for role, count in (
            ("total_touching", len(rows)),
            ("sender", sum(e.sender == root for e in rows)),
            ("receiver", sum(e.receiver == root for e in rows)),
        ):
            name = f"{window}_{role}_count"
            # The LI source spans under 30 days, so lifetime and 30-day values coincide.
            assert spec.sample.entity_values[name].source.time == api.TimeWindow(
                "event_time", start, 0
            )
            counts[name] = count
    return counts


def _check_counts(task, batch, counts, training):
    actual = batch.inputs["aggregates"]
    assert actual.keys() == counts[batch.metadata["sample_ids"][0]].keys()
    for name, field in actual.items():
        train = [counts[sample][name] for sample in training]
        center, scale = mean(train), pstdev(train)
        encoder = task.output("aggregates").field(name=name).encoder
        assert (encoder.mean, encoder.scale) == pytest.approx((center, scale))
        values = [(counts[s][name] - center) / scale for s in batch.metadata["sample_ids"]]
        expected = torch.tensor(values, dtype=torch.float32)
        torch.testing.assert_close(field["values"], expected, rtol=2e-6, atol=2e-6)
        assert_same(field["valid"], torch.ones_like(expected, dtype=torch.bool))


def test_li_medium_l01_combined_forecast(tmp_path):
    # The two train anchors share an account but differ on all twelve historical counts.
    # The other anchors cover a held-out currency, tied transfers and the 24-hour boundary.
    positions = (1302, 31764, 40961, 1338489, 2708071)
    files, anchors = _source_rows(
        L01_MANIFEST, "li-medium", positions, "next_amount", "next_currency"
    )
    account_ids, events, _ = _source_history(files, anchors)
    spec, data, table = prepare_l01(files, tmp_path / "prepared", anchor_rows=positions)
    answers = {"train": {}, "valid": {}, "test": {}}
    histories = {}
    counts = {}
    for row in anchors.values():
        cutoff = int(row["as_of_time"].replace(tzinfo=UTC).timestamp())
        root = account_ids[row["entity_id"]]
        sample = row["sample_id"]
        past = {
            i: e for i, e in events.items() if root in (e.sender, e.receiver) and e.time < cutoff
        }
        histories[sample] = root, cutoff, past
        counts[sample] = _source_counts(spec, root, cutoff, past)
        outgoing = (
            (event.time, index)
            for index, event in events.items()
            if event.sender == root and cutoff <= event.time < cutoff + 86400
        )
        first = min(outgoing, default=None)
        fields = events[first[1]].features if first is not None else {}
        pair = fields.get("amount_paid"), fields.get("payment_currency")
        assert (row["next_amount"], row["next_currency"]) == pair
        answers[row["forecast_split"]][sample] = pair

    (train_currency,) = {currency for _, currency in answers["train"].values()} - {None}
    currency_ids = {train_currency: 2}
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with (
        api.prepare(spec, data, engine) as plan,
        plan.materialize(table, fit_split="train") as task,
    ):
        assert task.count() == len(positions)
        assert dict(task.target("currency").field(name="value").encoder.entries) == currency_ids
        for split, expected in answers.items():
            (batch,) = task.batches(split=split, batch_size=len(positions), output_format="model")
            _check_delivered_targets(batch, expected, currency_ids)
            selected = [histories[s] for s in batch.metadata["sample_ids"]]
            assert set(batch.inputs) == {"graph", "history", "aggregates"}
            assert list(batch.metadata["entity_ids"]) == [root for root, _, _ in selected]
            assert batch.metadata["as_of_time"].tolist() == [
                cutoff - 1 for _, cutoff, _ in selected
            ]
            _check_graph(task, batch, selected)
            _check_history(task, batch, selected)
            _check_counts(task, batch, counts, answers["train"])

        # The source records both labels together. Mask one head at a time to test native delivery.
        projected = pq.read_table(table.source.path)
        row = next(r for r in projected.to_pylist() if r["currency"] in currency_ids)
        probes = [{**row, "sample_id": name, name: None} for name in ("amount", "currency")]
        path = tmp_path / "partial-targets.parquet"
        pq.write_table(pa.Table.from_pylist(probes, schema=projected.schema), path)
        with plan.materialize(
            replace(table, source=api.Parquet(str(path))), fitted=task.fitted_state
        ) as partial:
            (batch,) = partial.batches(batch_size=2, output_format="model")
            _check_delivered_targets(
                batch,
                {"amount": (None, row["currency"]), "currency": (row["amount"], None)},
                currency_ids,
            )
