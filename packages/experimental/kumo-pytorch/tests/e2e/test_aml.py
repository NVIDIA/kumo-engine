# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AML account-day histories and forecasts checked against original transaction rows."""

import hashlib
import json
import os
from bisect import bisect_right
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from itertools import accumulate
from math import log
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import torch

from e2e.datasets import bind_tables
from e2e.datasets.aml import (
    E01_MANIFEST,
    L01_MANIFEST,
    l01_loss,
    li_head_loss,
    prepare_hi_small,
    prepare_li_medium,
)
from e2e.datasets.staging import verify
from kumo.pytorch import api
from tests.e2e.binding_report import BindingReport
from tests.e2e.history import (
    check_masked_history,
    check_normalized_aggregates,
    check_scalar_history,
    scalar_token,
)
from tests.e2e.loader_timing import correctness_stage
from tests.test_inference_lifecycle import assert_same, batch_location


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
    transform = manifest["transformation"]
    parent = definition["metadata"]["derived_from"]
    assert parent["identity"]["anchor_version"] == "v1"
    assert transform["original_columns"] == "unchanged Arrow arrays in original row order"
    assert transform["source_payload_sha256_verified"] == parent["payloads"][0]["sha256"]
    assert (
        manifest["statistics"]["rows"]
        == transform["original_rows"]
        == pq.read_metadata(files["anchors"]).num_rows
    )

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
    assert_same(
        nodes["transactions"]["features"],
        expected_features,
        "graph.transactions",
        locate=batch_location(batch, nodes["transactions"]["batch"]),
    )


@pytest.fixture(scope="module")
def li_heads_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("li-heads")
    files = {name: root / f"{name}.parquet" for name in ("anchors", "mapping", "transactions")}
    cutoff = _seconds("2024/03/01 00:00")
    entity = '["li-medium","account","1","A"]'
    clock = pa.timestamp("ns", tz="UTC")
    anchors = {
        "sample_id": ["excluded", "train-0", "train-1", "train-2", "valid-0", "test-0"],
        "entity_id": [entity] * 6,
        "as_of_time": pa.array([cutoff * 10**9] * 6, clock),
        "graph_as_of_time": pa.array([(cutoff - 1) * 10**9] * 6, clock),
        "split": ["train"] * 6,
        "forecast_split": [None, "train", "train", "train", "valid", "test"],
        "next_delay_seconds": [999.0, 0.0, 2.5, None, 86400.0, None],
        "next_format": ["EXCLUDED", "wire", "ach", None, "card", None],
        "involvement": [1, 1, 0, None, 1, None],
    }
    observed = pa.array([(cutoff - 60) * 10**9], clock)
    transactions = {
        "timestamp": ["2024/02/29 23:59", "2024/03/01 00:00"],
        "from_bank": ["1", "1"],
        "to_bank": ["1", "1"],
        "from_account": ["A", "A"],
        "to_account": ["A", "A"],
        "amount_paid": ["1.25", "20"],
        "amount_received": ["1.00", "18"],
        "payment_currency": ["USD", "EUR"],
        "receiving_currency": ["EUR", "USD"],
        "payment_format": ["wire", "card"],
    }
    pq.write_table(pa.table(anchors), files["anchors"])
    pq.write_table(
        pa.table({"entity_id": [entity], "first_observed_time": observed}), files["mapping"]
    )
    pq.write_table(pa.table(transactions), files["transactions"])
    return files, cutoff


@pytest.fixture(scope="module")
def hi_small():
    positions = (0, 313, 2724, 4245, 11800, 409216)
    files, anchors = _source_rows(E01_MANIFEST, "hi-small", positions, "split", "involvement")
    account_ids, events, ends = _source_history(files, anchors)
    histories = {}
    registered = {}
    for pos, row in anchors.items():
        root = account_ids[row["entity_id"]]
        cutoff = int(row["as_of_time"].replace(tzinfo=UTC).timestamp())
        assert int(row["graph_as_of_time"].replace(tzinfo=UTC).timestamp()) == cutoff - 1
        stage = bisect_right(ends[:2], cutoff)
        split = ("train", "valid", "test")[stage]
        assert row["split"] == split
        touching = {i: e for i, e in events.items() if root in (e.sender, e.receiver)}
        past = {i: e for i, e in touching.items() if e.time < cutoff}
        assert past
        if cutoff + 86400 > ends[stage]:
            assert row["forecast_split"] is None and row["involvement"] is None
        else:
            label = any(e.flagged for e in touching.values() if cutoff <= e.time < cutoff + 86400)
            assert (row["forecast_split"], row["involvement"]) == (split, label)
        histories[row["sample_id"]] = root, cutoff, past
        registered[row["sample_id"]] = dict(row, anchor_row_position=pos)
    return files, registered, histories


@pytest.mark.parametrize(
    "binding,inputs,target",
    [
        ("E01", {"graph"}, "label"),
        ("E02", {"graph", "history", "aggregates"}, "label"),
        ("E03", {"history", "history_calendar", "history_timing"}, "masked_values"),
        ("E04", {"history"}, "next_token"),
        pytest.param("E05", {"history"}, "label", marks=pytest.mark.binding_report("histories")),
    ],
    ids=["e01-graph", "e02-combined", "e03-masked-values", "e04-next-token", "e05-involvement"],
)
@pytest.mark.binding_report("correctness")
def test_hi_small_account_history(tmp_path, hi_small, binding, inputs, target):
    files, anchors, histories = hi_small
    positions = [row["anchor_row_position"] for row in anchors.values()]
    spec, data, table = prepare_hi_small(
        files, tmp_path / "prepared", anchor_rows=positions, binding=binding
    )
    split_col = "forecast_split" if target == "label" else "split"
    eligible = {s: r[split_col] for s, r in anchors.items() if r[split_col] is not None}
    training = [s for s, split in eligible.items() if split == "train"]
    population = Counter(eligible.values())
    evidence = BindingReport(
        binding,
        E01_MANIFEST.name,
        "+".join(sorted(inputs)),
        population,
        {target: population if target == "label" else None},
        inputs,
    )
    fit_cutoff = max(histories[s][1] for s in training) - 1
    graph_clock = "graph" in inputs
    if "history" in inputs:
        assert {e.rows.time for e in spec.sample.events.values()} == {
            api.TimeWindow("event_time", stop=0 if graph_clock else -1)
        }
    if "aggregates" in inputs:
        counts = {s: _source_counts(spec, *histories[s]) for s in eligible}
        first, second = (counts[s] for s in training)
        assert all(value != second[name] for name, value in first.items())
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with (
        api.prepare(spec, data, engine) as plan,
        plan.materialize(table, fit_split="train", encoder_cutoff=fit_cutoff) as task,
        correctness_stage("delivery"),
    ):
        assert task.count() == len(eligible)
        for split in ("train", "valid", "test"):
            (batch,) = task.batches(
                split=split, batch_size=len(eligible), seed=1, output_format="model"
            )
            meta = batch.metadata
            samples = meta["sample_ids"]
            assert sorted(samples) == sorted(s for s in eligible if eligible[s] == split)
            assert list(meta["splits"]) == [split] * len(samples)
            selected = [histories[s] for s in samples]
            assert list(meta["entity_ids"]) == [root for root, _, _ in selected]
            assert meta["as_of_time"].tolist() == [t - int(graph_clock) for _, t, _ in selected]
            assert not any(meta["is_padding"])
            for sample, metadata in zip(samples, meta["request_metadata"], strict=True):
                row = anchors[sample]
                assert metadata["anchor_row_position"] == row["anchor_row_position"]
                assert metadata["registered_entity_id"] == row["entity_id"]
                assert metadata["prediction_time_ns"] == histories[sample][1] * 10**9
            assert set(batch.inputs) == inputs and set(batch.targets) == {target}
            if target == "label":
                labels = torch.tensor(
                    [anchors[s]["involvement"] for s in samples], dtype=torch.float32
                )
                assert_same(
                    batch.targets[target],
                    {"values": labels, "valid": torch.ones_like(labels, dtype=torch.bool)},
                )
            elif target == "masked_values" and split == "train":
                masked = batch.targets[target]
                assert (masked["corruption"] & ~masked["valid"]).any()
            if graph_clock:
                _check_graph(task, batch, selected)
            if "history" in inputs:
                _check_history(task, batch, selected)
            if "aggregates" in inputs:
                check_normalized_aggregates(task, batch, counts, training)
            evidence.observe(batch)
    evidence.publish()
    if binding == "E05":
        _check_role_windows(
            spec.schema, data, table, engine, {s: histories[s] for s in eligible}, population
        )


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
    if "masked_values" in batch.targets:
        assert task.contract.batch.targets["masked_values"] == api.MaskedValuesTarget(
            "history", 0.15, 0.10, 0.10, 0.10, fields=tuple(fields)
        )
        check_masked_history(task, batch, fields, rows, family="transactions")
    else:
        check_scalar_history(task, batch, fields, rows)


def _check_l01_targets(batch, expected, currency_ids):
    samples = batch.metadata["sample_ids"]
    assert sorted(samples) == sorted(expected)
    amounts, currencies = zip(*(expected[s] for s in samples), strict=True)
    amount, currency = batch.targets["amount"], batch.targets["currency"]
    assert_same(
        amount["valid"],
        torch.tensor([v is not None for v in amounts]),
        "targets.amount.valid",
        locate=batch_location(batch),
    )
    assert_same(
        currency["valid"],
        torch.tensor([v is not None for v in currencies]),
        "targets.currency.valid",
        locate=batch_location(batch),
    )
    assert_same(
        amount["values"][amount["valid"]],
        torch.tensor([float(v) for v in amounts if v is not None], dtype=torch.float32),
        "targets.amount.values",
        locate=batch_location(batch, amount["valid"].nonzero().flatten()),
    )
    ids = [0 if value is None else currency_ids.get(value, 1) for value in currencies]
    assert_same(
        currency["values"],
        torch.tensor(ids, dtype=torch.int64),
        "targets.currency.values",
        locate=batch_location(batch),
    )

    # One amount score, then distinct currency logits for missing, UNKNOWN and fitted IDs.
    columns = torch.arange(3 + len(currency_ids), dtype=torch.float32)
    scores = torch.nn.Parameter(torch.arange(1, len(samples) + 1)[:, None] * columns)
    prediction, logits = scores[:, 0], scores[:, 1:]
    loss, sums, counts = l01_loss(batch, prediction, logits)
    padding = batch.metadata["is_padding"]
    terms = {"amount": [], "currency": []}
    for i, (value, category) in enumerate(zip(amounts, ids, strict=True)):
        if padding[i]:
            continue
        if value is not None:
            error = abs(prediction[i].item() - amount["values"][i].item())
            terms["amount"].append(error**2 / 2 if error < 1 else error - 0.5)
        if category:
            row = logits[i].detach().double()
            terms["currency"].append((row.logsumexp(0) - row[category]).item())
    assert counts == {name: len(values) for name, values in terms.items()}
    assert sums["amount"].dtype == torch.float64
    for name, values in terms.items():
        assert sums[name].item() == pytest.approx(sum(values), abs=1e-7)
    assert loss.item() == pytest.approx(sum(sum(v) / len(v) for v in terms.values() if v))
    loss.backward()
    assert_same(scores.grad[:, 0] != 0, amount["valid"] & ~padding)
    assert_same(scores.grad[:, 1:].any(dim=1), currency["valid"] & ~padding)
    if not any(counts.values()):
        before = scores.detach().clone()
        torch.optim.SGD([scores], lr=0.1).step()
        assert_same(scores, before)


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
            # Both extracts span under 30 days, so lifetime and 30-day values coincide.
            assert spec.sample.entity_values[name].source.time == api.TimeWindow(
                "event_time", start, 0
            )
            counts[name] = count
    return counts


@pytest.mark.skipif(
    os.getenv("KUMO_PYTORCH_PROFILE_HI_SMALL") != "1", reason="run the HI Small benchmark"
)
def test_hi_small_fitting_memory(request):
    from e2e.datasets.aml import prepare_hi_small_benchmark
    from tests.e2e.host_memory import memory_interval
    from tests.e2e.loader_timing import parquet_metadata

    work = request.config.getoption("--e2e-dir")
    work.mkdir(parents=True, exist_ok=True)
    binding = work.parents[1].name
    prepared = work.parents[2] / "prepared"
    index = work.parent / "index"
    build = work.name == "index-setup"
    if build and not prepared.exists():
        prepare_hi_small_benchmark(Path(os.environ["KUMO_PYTORCH_DATA_ROOT"]), prepared)
    identity = json.loads((prepared / "identity.json").read_text())
    info = identity["bindings"][binding]
    spec_file = (prepared / f"{binding}.json").read_bytes()
    assert hashlib.sha256(spec_file).hexdigest() == info["contract_sha256"]
    spec = api.DataLoaderSpec.from_json(spec_file.decode())
    _, dataset, anchors = bind_tables(prepared, spec, "entity_id", spec.sample.targets)
    torch.set_num_threads(1)
    assert index.exists() == (not build), f"unexpected index state for {work.name}: {index}"
    engine = api.DiskGraphEngine(
        index=index, ingest="if_missing" if build else "never", num_threads=1
    )
    with api.prepare(spec, dataset, engine) as plan:
        report = {
            "binding": binding,
            "pid": os.getpid(),
            "prefix_sha256": info["prefix_sha256"],
            "contract": plan.contract.fingerprint,
            "source_sha256": json.loads((index / "index_manifest.json").read_text())[
                "dataset_fingerprint"
            ],
            "encoder_cutoff": info["encoder_cutoff_seconds"],
        }
        output = work / "results.json"
        request.addfinalizer(lambda: output.write_text(json.dumps(report) + "\n"))
        if build:
            return
        count = int(work.name)
        path = prepared / f"{binding}-{count}.parquet"
        metadata = parquet_metadata(path)
        assert (metadata["rows"], metadata["sha256"]) == (count, info["anchors_sha256"][work.name])
        anchors = replace(anchors, source=api.Parquet(str(path)))
        assert os.environ["KUMO_PYTORCH_HOST_MEMORY"] in ("kernel", "sampled")
        with ExitStack() as stack:
            with memory_interval(report, "materialization", sample_interval_ns=50_000_000):
                task = stack.enter_context(
                    plan.materialize(
                        anchors, fit_split="train", encoder_cutoff=report["encoder_cutoff"]
                    )
                )
            assert task.count() == task.count("train") == count
            for half in ("inputs", "targets"):
                fit = task.fitted_state.fit[half]
                assert (
                    fit["fit_split"] == ["train"] and fit["anchors"] == task.fingerprints["anchors"]
                )
            statistics = task.fitted_state.fit["inputs"].get("statistics")
            if statistics:
                assert statistics["cutoff"] == report["encoder_cutoff"] * 10**9
            report.update(
                anchors=count,
                anchor_sha256=metadata["sha256"],
                fitted_state=task.fitted_state.fingerprint,
            )
        assert report["host_memory"]["complete"]


@pytest.mark.skipif(os.getenv("KUMO_PYTORCH_PROFILE_L01") != "1", reason="run the l01 benchmark")
@pytest.mark.parametrize("mode", ["fresh", "reloaded"])
def test_li_medium_l01_loader(mode, tmp_path, request, native_anchor_delivery):
    import diskgraph

    from kumo.pytorch.engines import diskgraph as backend
    from tests.e2e.file_cache import IndexPageCache
    from tests.e2e.host_memory import memory_interval
    from tests.e2e.loader_timing import parquet_metadata, stage_observer
    from tests.e2e.runs import L01_BATCHING, check_l01_epochs, time_loader

    work = request.config.getoption("--e2e-dir")
    prepared = work.parents[1] / "prepared"
    fresh = work.parent / "fresh" / "l01"
    spec = api.DataLoaderSpec.from_json((prepared / "contract.json").read_text())
    length_planning = os.getenv("KUMO_PYTORCH_L01_LENGTH_PLANNING") == "1"
    if length_planning:
        batching = replace(spec.batch.batching, length_input="history", bucket_by_length=True)
        spec = replace(spec, batch=replace(spec.batch, batching=batching))
    _, data, table = bind_tables(prepared, spec, "entity_id", spec.batch.targets)
    anchors = pq.read_table(table.source.path, columns=["sample_id", "split"])
    splits = Counter(anchors["split"].to_pylist())
    assert splits == dict.fromkeys(("train", "valid", "test"), 512)
    assert pc.count_distinct(anchors["sample_id"]).as_py() == len(anchors)
    expected = anchors.filter(pc.equal(anchors["split"], "train"))["sample_id"].to_pylist()
    delivered = []

    def observe(batch):
        delivered[-1].extend(batch.metadata["sample_ids"])
        return batch

    def batches(task, **options):
        setup["fit_anchors"] = {
            half: [record["anchors"], record["fit_split"]]
            for half, record in task.fitted_state.fit.items()
        }
        delivered.append([])
        return map(observe, native_anchor_delivery(task, seed=0, **options))

    torch.set_num_threads(4)
    torch.manual_seed(0)
    cuda = os.getenv("KUMO_PYTORCH_L01_CUDA") == "1"
    make_step = None
    if cuda:
        from tests.e2e.l01_model import make_l01_step

        make_step = make_l01_step
        assert os.environ["KUMO_PYTORCH_HOST_MEMORY"] == "sampled"
    action = os.getenv("KUMO_PYTORCH_L01_INDEX_ACTION")
    index = (prepared.parent if action else tmp_path) / "index"
    engine = api.DiskGraphEngine(
        index=index,
        ingest="never" if action == "reopen" else "if_missing",
        num_threads=4,
        profile=True,
    )
    population = {
        "source_sha256": hashlib.sha256(L01_MANIFEST.read_bytes()).hexdigest(),
        "anchor_sha256": parquet_metadata(table.source.path)["sha256"],
        "anchor_binding": {k: v for k, v in table.to_dict().items() if k != "source"},
        "split_counts": splits,
    }
    setup = {"pid": os.getpid(), **population}
    with ExitStack() as stack:
        with memory_interval(setup, "index.prepare"), stage_observer() as observer:
            if action:
                observer.wrap(backend, "ensure_diskgraph_index", "index.ensure")
                observer.wrap(diskgraph, "ingest", "index.ingest")
            plan = stack.enter_context(api.prepare(spec, data, engine))
        if action:
            assert observer.stages["index.ensure"]["calls"] == 1
            ingests = observer.stages.get("index.ingest", {}).get("calls", 0)
            assert ingests == int(action == "build"), (
                f"expected {action}, observed {ingests} ingests"
            )
            setup.update(
                index_path=str(index),
                index_action=action,
                index_ingests=ingests,
                index_ensure_ns=observer.stages["index.ensure"]["elapsed_ns"],
            )
        fit = time_loader(
            plan,
            table,
            work / "l01",
            batches,
            setup=setup,
            modes=(mode,),
            artifact=fresh / "artifact",
            cache=IndexPageCache(index) if cuda else None,
            make_step=make_step,
            event_tables=("transactions",),
            **L01_BATCHING,
        )[mode]
    if length_planning:
        assert [Counter(ids) for ids in delivered] == [Counter(expected)] * 2
    else:
        assert delivered == [expected, expected]
    check_l01_epochs(fit["epochs"], cuda=cuda, length_planning=length_planning, where=work)
    for stage in ("input.fit", "target.fit"):
        assert fit["stages"].get(stage, {}).get("calls", 0) == int(mode == "fresh")
    assert fit["stages"].get("artifact.load", {}).get("calls", 0) == int(mode == "reloaded")
    if mode == "reloaded":
        prior = json.loads((fresh / "results.json").read_text())
        assert prior["setup"]["pid"] != setup["pid"]
        for key in (*population, "fit_anchors"):
            assert setup[key] == prior["setup"][key]
        for key in ("fitted_state", "contract"):
            assert prior["fresh"][key] == fit[key]
    else:
        assert setup["fit_anchors"] == dict.fromkeys(
            ("inputs", "targets"), [fit["anchors"], ["train"]]
        )


@pytest.mark.binding_report("correctness")
def test_li_medium_l01_combined_forecast(tmp_path):
    # The two train anchors share an account but differ on all twelve historical counts.
    # The other anchors cover a held-out currency, tied transfers and the 24-hour boundary.
    positions = (1302, 31764, 40961, 1338489, 2708071)
    files, anchors = _source_rows(
        L01_MANIFEST, "li-medium", positions, "next_amount", "next_currency"
    )
    account_ids, events, _ = _source_history(files, anchors)
    spec, data, table = prepare_li_medium(files, tmp_path / "prepared", anchor_rows=positions)
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

    first, second = answers["train"]
    assert all(value != counts[second][name] for name, value in counts[first].items())
    (train_currency,) = {currency for _, currency in answers["train"].values()} - {None}
    currency_ids = {train_currency: 2}
    heads = {
        name: {s: sum(row[i] is not None for row in a.values()) for s, a in answers.items()}
        for i, name in enumerate(("amount", "currency"))
    }
    evidence = BindingReport(
        "L01",
        L01_MANIFEST.name,
        "combined",
        {s: len(a) for s, a in answers.items()},
        heads,
        ("graph", "history", "aggregates"),
    )
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with (
        api.prepare(spec, data, engine) as plan,
        plan.materialize(table, fit_split="train") as task,
        correctness_stage("delivery"),
    ):
        assert task.count() == len(positions)
        assert dict(task.target("currency").field(name="value").encoder.entries) == currency_ids
        for split, expected in answers.items():
            (batch,) = task.batches(split=split, batch_size=len(positions), output_format="model")
            _check_l01_targets(batch, expected, currency_ids)
            selected = [histories[s] for s in batch.metadata["sample_ids"]]
            assert set(batch.inputs) == {"graph", "history", "aggregates"}
            assert list(batch.metadata["entity_ids"]) == [root for root, _, _ in selected]
            assert batch.metadata["as_of_time"].tolist() == [
                cutoff - 1 for _, cutoff, _ in selected
            ]
            _check_graph(task, batch, selected)
            _check_history(task, batch, selected)
            check_normalized_aggregates(task, batch, counts, answers["train"])
            evidence.observe(batch)
        evidence.publish()

        # The short rank gets a copy with valid labels and zero eligible loss.
        (padded,) = task.batches(split="test", rank=1, world_size=2, output_format="model")
        assert padded.metadata["is_padding"].tolist() == [True]
        (labels,) = answers["test"].values()
        _check_l01_targets(padded, {padded.metadata["sample_ids"][0]: labels}, currency_ids)

        # The original labels occur together. Give the two heads different eligible counts.
        projected = pq.read_table(table.source.path)
        row, other = (r for r in projected.to_pylist() if r["split"] == "train")
        probes = [{**row, "sample_id": name, name: None} for name in ("amount", "currency")]
        probes.append(dict(other, currency=None))
        path = tmp_path / "partial-targets.parquet"
        pq.write_table(pa.Table.from_pylist(probes, schema=projected.schema), path)
        with plan.materialize(
            replace(table, source=api.Parquet(str(path))), fitted=task.fitted_state
        ) as partial:
            (batch,) = partial.batches(batch_size=len(probes), output_format="model")
            _check_l01_targets(
                batch,
                {
                    "amount": (None, row["currency"]),
                    "currency": (row["amount"], None),
                    other["sample_id"]: (other["amount"], None),
                },
                currency_ids,
            )


@pytest.mark.parametrize(
    "binding,labels,score,loss_sums",
    [
        ("L02", [0.0, 2.5, 0.0, 86400.0, 0.0], 0.5, [1.625, 86399.0, 0.0]),
        ("L03", [3, 2, 0, 1, 0], [0.0, log(2), log(3), log(4)], [log(25 / 3), log(5), 0.0]),
        ("L04", [1.0, 0.0, 0.0, 1.0, 0.0], log(3), [log(16 / 3), log(4 / 3), 0.0]),
    ],
    ids=["l02-delay", "l03-format", "l04-involvement"],
)
@pytest.mark.binding_report("correctness")
def test_li_medium_stored_head(tmp_path, li_heads_source, binding, labels, score, loss_sums):
    from e2e.datasets import aml
    from tests.e2e.loader_timing import parquet_metadata, stage_observer

    files, cutoff = li_heads_source
    with stage_observer(optional=True) as observer:
        if observer is not None:
            observer.wrap(aml, "_prepare_aml", "source.convert")
        spec, data, table = prepare_li_medium(
            files, tmp_path / "prepared", anchor_rows=range(6), binding=binding
        )
    if observer is not None and binding == "L03":
        for raw, output, rows in (("mapping", "accounts", 1), ("transactions", "transactions", 2)):
            source = parquet_metadata(files[raw])
            converted = parquet_metadata(data.tables[output].path)
            assert source["rows"] == converted["rows"] == rows
            assert source["sha256"] != converted["sha256"]
        assert observer.stages["source.convert"]["calls"] == 1
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with (
        api.prepare(spec, data, engine) as plan,
        plan.materialize(table, fit_split="train", encoder_cutoff=cutoff - 1) as task,
        correctness_stage("delivery"),
    ):
        assert task.count() == 5
        if binding == "L03":
            assert dict(task.target("label").field(name="value").encoder.entries) == {
                "ach": 2,
                "wire": 3,
            }
        if binding == "L04":
            # All six fields of the prior self-transfer survive without corruption.
            fields = task.output("history").bindings.values()
            source = ("self", 1.25, "USD", 1.0, "EUR", "wire")
            tokens = [
                scalar_token(field.encoder, task.spaces["history"], value)
                for field, value in zip(fields, source, strict=True)
            ]
        valid = torch.tensor([True, True, False, True, False])
        splits = (("train", slice(3)), ("valid", slice(3, 4)), ("test", slice(4, 5)))
        evidence = BindingReport(
            binding,
            "six-row fixture",
            {"L02": "nuformer", "L03": "graphsage", "L04": "pragma"}[binding],
            {s: len(labels[r]) for s, r in splits},
            {"label": {s: int(valid[r].sum()) for s, r in splits}},
            ("graph" if binding == "L03" else "history",),
        )
        for (split, rows), expected_loss in zip(splits, loss_sums, strict=True):
            (batch,) = task.batches(split=split, batch_size=3, output_format="model")
            size = len(labels[rows])
            meta = batch.metadata
            assert list(meta["sample_ids"]) == [f"{split}-{i}" for i in range(size)]
            assert meta["as_of_time"].tolist() == [cutoff - int(binding == "L03")] * size
            requests = meta["request_metadata"]
            assert [r["anchor_row_position"] for r in requests] == list(range(1, 6)[rows])
            assert {r["prediction_time_ns"] for r in requests} == {cutoff * 10**9}
            assert_same(
                batch.targets,
                {"label": {"values": torch.tensor(labels[rows]), "valid": valid[rows]}},
            )
            if binding == "L02":
                assert set(batch.inputs) == {"history"}
                assert batch.inputs["history"]["valid"].sum(dim=1).tolist() == [10] * size
            elif binding == "L03":
                assert set(batch.inputs) == {"graph"}
                assert batch.inputs["graph"]["nodes"]["transactions"]["ids"].tolist() == [0] * size
            else:
                assert set(batch.inputs) == {"history", "history_calendar", "history_timing"}
                assert batch.inputs["history"]["value_ids"].tolist() == tokens * size

            prediction = torch.tensor([score] * size, requires_grad=True)
            total, count = li_head_loss(batch, binding, prediction)
            assert count == valid[rows].sum().item()
            assert total.item() == pytest.approx(expected_loss)
            if binding == "L02":
                assert total.dtype == torch.float64
            total.backward()
            assert_same(prediction.grad.reshape(size, -1).any(dim=1), valid[rows])
            evidence.observe(batch)
        evidence.publish()


def _check_role_windows(schema, data, table, engine, histories, population):
    # At prediction time T, senders use all prior events and receivers use [T-24h, T-1h].
    events = {
        role: api.EventSpec(
            api.Rows((role,), api.TimeWindow("event_time", *window)),
            {"id": api.Field("transaction_id"), "role": api.Constant(i, api.DType("int64"))},
        )
        for i, (role, window) in enumerate((("sender", (None, -1)), ("receiver", (-86401, -3600))))
    }
    fields = api.Fields(tuple(api.Encode(name, api.Raw()) for name in ("role", "id")))
    inputs = {
        f"{keep}_{order}": api.Input(
            api.EventSource(
                tuple(events), max_events=limit, keep=keep if limit else "first", order=order
            ),
            fields,
            api.Packed(),
        )
        for keep, limit in (("all", None), ("first", 3), ("last", 3))
        for order in ("ascending", "descending")
    }
    spec = api.DataLoaderSpec(
        schema, api.SampleSpec(api.EntitySpec("accounts"), events=events), api.BatchSpec(inputs)
    )
    evidence = BindingReport(
        "E05", E01_MANIFEST.name, "packed", population, {}, inputs, mode="histories"
    )
    with (
        api.prepare(spec, data, engine) as plan,
        plan.materialize(replace(table, targets={}), fit_split="train") as task,
        correctness_stage("delivery"),
    ):
        seen = []
        for batch in task.batches(batch_size=len(histories), output_format="model"):
            assert batch.inputs.keys() == inputs.keys()
            seen.extend(batch.metadata["sample_ids"])
            for owner, sample in enumerate(batch.metadata["sample_ids"]):
                root, cutoff, past = histories[sample]
                times = {(0, i): e.time for i, e in past.items() if e.sender == root}
                times.update(
                    {
                        (1, i): e.time
                        for i, e in past.items()
                        if e.receiver == root and cutoff - 86400 <= e.time <= cutoff - 3600
                    }
                )
                ordered = sorted(times.values())
                for name, stream in batch.inputs.items():
                    keep, order = name.split("_")
                    rows = slice(*stream["offsets"][owner : owner + 2].tolist())
                    roles, ids = (stream[key]["values"][rows].tolist() for key in ("role", "id"))
                    actual = list(zip(roles, ids, strict=True))
                    assert len(set(actual)) == len(actual)
                    # Equal-time transfers can exchange places, including at the third event.
                    selected = {"all": ordered, "first": ordered[:3], "last": ordered[-3:]}[keep]
                    assert [times[key] for key in actual] == (
                        selected if order == "ascending" else selected[::-1]
                    )
            evidence.observe(batch)
        assert sorted(seen) == sorted(histories)
        evidence.publish()
