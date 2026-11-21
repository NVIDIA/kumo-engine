# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HI Small account-day forecasts checked against the original transaction rows."""

import json
import os
from bisect import bisect_right
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from itertools import accumulate
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from e2e.datasets.aml import E01_MANIFEST, prepare_e01
from e2e.datasets.staging import verify
from kumo.pytorch import api
from tests.test_diskgraph_backend import _canonical_graph
from tests.test_inference_lifecycle import assert_same


class Transfer(NamedTuple):
    sender: int
    receiver: int
    time: int
    flagged: bool
    features: dict


def _source_rows(files):
    # Check that the three published manifests describe the same pinned population.
    pins = json.loads(E01_MANIFEST.read_text())["objects"]
    manifest = json.loads(files["anchor_manifest"].read_text())
    binding = json.loads(files["mapping_manifest"].read_text())
    definition = manifest["definition"]
    keys = ("dataset_id", "dataset_version", "task_id", "anchor_version")
    assert tuple(definition[k] for k in keys) == ("aml-data/hi-small", "v8", "account-daily", "v2")
    assert definition["metadata"]["source_binding"] == binding
    assert binding["sha256"]["accounts.parquet"] == pins["mapping"]["sha256"]
    source = json.loads(files["source_manifest"].read_text())
    (tx,) = (f for f in source["files"] if f["path"] == "hi-small/transactions.parquet")
    (original,) = (f for f in binding["sources"] if f["path"] == tx["path"])
    assert tx["sha256"] == original["expected_sha256"] == pins["transactions"]["sha256"]
    assert tx["rows"] == pq.read_metadata(files["transactions"]).num_rows
    assert manifest["files"][0]["sha256"] == pins["anchors"]["sha256"]
    assert manifest["statistics"]["rows"] == pq.read_metadata(files["anchors"]).num_rows

    # Physical positions in the pinned v2 population: split boundary, covered negative,
    # sender-only positive, receiver-only positive, and end of observation.
    positions = (0, 313, 4245, 11800, 409216)
    columns = ["sample_id", "entity_id", "as_of_time", "graph_as_of_time"]
    columns += ["forecast_split", "involvement"]
    rows = ds.dataset(files["anchors"]).take(positions, columns=columns).to_pylist()
    return dict(zip(positions, rows, strict=True))


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


def test_hi_small_e01_graph_and_forecast(tmp_path):
    root = os.environ.get("KUMO_PYTORCH_DATA_ROOT")
    if root is None:
        pytest.skip("set KUMO_PYTORCH_DATA_ROOT to the staged package datasets")
    files = verify(Path(root), E01_MANIFEST)
    anchors = _source_rows(files)
    account_ids, events, ends = _source_history(files, anchors)
    spec, data, table = prepare_e01(files, tmp_path / "prepared", anchor_rows=list(anchors))
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with (
        api.prepare(spec, data, engine) as plan,
        plan.materialize(table, fit_split="train") as task,
    ):
        # Two immature forecasts are excluded; each named split has one eligible anchor.
        assert task.count() == 3
        bindings = task.output("graph").bindings["transactions"]
        expected_encoders = {
            "amount_paid": api.ZScore(),
            "amount_received": api.ZScore(),
            "payment_currency": api.Vocabulary(),
            "receiving_currency": api.Vocabulary(),
            "payment_format": api.Vocabulary(),
            "event_time_seconds": api.Raw(),
        }
        assert {name: field.encoder.config for name, field in bindings.items()} == expected_encoders
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
            graph = batch.inputs["graph"]
            tx = graph["nodes"]["transactions"]
            ids = tx["ids"].tolist()
            accounts = {a for e in history.values() for a in (e.sender, e.receiver)}
            assert _canonical_graph(graph) == {
                "nodes": {
                    "accounts": sorted((a, 0) for a in accounts),
                    "transactions": sorted((i, 0) for i in history),
                },
                "edges": {
                    "sender": sorted((e.sender, i) for i, e in history.items()),
                    "receiver": sorted((e.receiver, i) for i, e in history.items()),
                },
            }
            fields = {
                name: binding.encode_values([history[i].features[name] for i in ids])
                for name, binding in bindings.items()
            }
            assert_same(tx["features"], fields)
