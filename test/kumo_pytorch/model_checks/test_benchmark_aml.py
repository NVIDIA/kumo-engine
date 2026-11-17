"""Exercise the real AML benchmark binding using tiny local source files."""

import hashlib
import json
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from kumo.pytorch.api import Batch

from test.kumo_pytorch import aml_inputs
from test.kumo_pytorch.aml_benchmark import counts, factory
from test.kumo_pytorch.benchmarking import run
from test.kumo_pytorch.benchmarking.campaign import compare_states
from test.kumo_pytorch.data import ROOT

from .test_aml_source import account, source_files

CASES = [
    (s["selector"]["workload_id"], f)
    for s in json.loads((ROOT / "workloads/aml.json").read_text())
    for f in s["models"]
]


def test_pdf_token_example_excludes_padding_targets_and_parallel_ids():
    valid = torch.arange(200) < 110
    indices = torch.cat((torch.arange(100) // 10, torch.full((100,), -1)))
    batch = Batch(
        inputs={
            "history": {
                "value_ids": torch.ones(200, dtype=torch.long),
                "valid": valid,
                "field_ids": torch.ones(200, dtype=torch.long),
                "source_event_indices": indices,
            }
        },
        targets={"extra": {"value_ids": torch.ones(900, dtype=torch.long)}},
        metadata={"sample_ids": ("sample",), "is_padding": torch.tensor([False])},
    )
    observed = counts(batch)
    assert observed.tokens == 110
    assert observed.events == 10
    assert observed.samples == 1


def test_zscored_numbers_are_not_tokens():
    batch = Batch(
        inputs={
            "features": {
                "values": torch.ones(3),
                "valid": torch.ones(3, dtype=torch.bool),
            }
        },
        targets={},
        metadata={
            "sample_ids": ("a", "b", "c"),
            "is_padding": torch.zeros(3, dtype=torch.bool),
        },
    )
    observed = counts(batch)
    assert observed.samples == 3
    assert observed.tokens is None


@pytest.fixture
def source(tmp_path, monkeypatch):
    tx, binding, _ = source_files(tmp_path)
    files = {}
    for original, key in (
        (tx, "datasets/aml-data/v8/hi-small/transactions.parquet"),
        (binding, "anchors/aml-data/v8/hi-small/_bindings/v1/accounts.parquet"),
    ):
        target = tmp_path / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
        files[key] = {
            "key": key,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "bytes": target.stat().st_size,
        }
    # Source checksum verification and anchor reading have independent tests.
    # Here supply explicit tiny anchors to exercise the binding and real source mapper.
    groups = {
        split: [
            {
                "sample_id": split,
                "entity_id": account(1, "0007"),
                "as_of_time": int(datetime(2022, 9, day, tzinfo=UTC).timestamp())
                * 10**9,
                "split": split,
                "forecast_split": split,
                "involvement": 1,
                "physical_row_position": day,
            }
        ]
        for day, split in enumerate(("train", "valid", "test"), 2)
    }
    graph_groups = {
        split: [{**row, "graph_as_of_time": row["as_of_time"] - 10**9} for row in rows]
        for split, rows in groups.items()
    }
    for version, source_groups in (("v1", groups), ("v2", graph_groups)):
        path = (
            tmp_path
            / "anchors/aml-data/v8/hi-small/account-daily"
            / version
            / "anchors.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(
            [row for rows in source_groups.values() for row in rows]
        )
        for column in (
            ("as_of_time", "graph_as_of_time") if version == "v2" else ("as_of_time",)
        ):
            table = table.set_column(
                table.schema.get_field_index(column),
                column,
                table[column].cast(pa.timestamp("ns")),
            )
        pq.write_table(table, path)
    monkeypatch.setattr(aml_inputs, "verify_inputs", lambda directory: files)
    monkeypatch.setattr(
        aml_inputs,
        "load_selections",
        lambda directory, limit: {
            "v1": {"split": groups, "forecast_split": groups},
            "v2": {"forecast_split": graph_groups},
        },
    )
    return tmp_path


@pytest.mark.parametrize(
    "fault", ["native-fallback", "cache-reset", "missing-history-plan"]
)
def test_native_path_and_cache_regressions_fail(source, monkeypatch, fault):
    from kumo.pytorch.engines.native_anchors import NativeAnchors

    if fault == "native-fallback":
        # A materialized table alone must not qualify native execution.
        monkeypatch.setattr(NativeAnchors, "_typed_batches", lambda *a, **kw: iter(()))
        message = "bypassed the native AnchorTable model iterator"
    elif fault == "cache-reset":
        original = NativeAnchors._iter

        def reset(self, *args, **kwargs):
            if kwargs.get("epoch") == 1:
                self._length_caches.clear()
                for path in self.path.glob("lengths-*.bin"):
                    path.unlink()
            return original(self, *args, **kwargs)

        monkeypatch.setattr(NativeAnchors, "_iter", reset)
        message = "replaced the native per-anchor length cache"
    else:
        from test.kumo_pytorch.benchmarking.kumo_observers import KumoObservers

        original = KumoObservers._native_iterator

        def missing_history(self, native):
            observed = original(self, native)

            def call(owner, stage, **kwargs):
                kwargs["history"] = None
                return observed(owner, stage, **kwargs)

            return call

        monkeypatch.setattr(KumoObservers, "_native_iterator", missing_history)
        message = "omitted the contract's merged-history plan"
    result = run(
        factory,
        {
            "data_dir": str(source),
            "family": "nuformer",
            "workload": "next-token-pretrain",
            "anchors_per_split": 1,
            "batch_size": 1,
            "threads": 1,
            "training": False,
        },
        source / "regression",
        repetitions=1,
        epochs=2,
    )
    assert result["execution_status"] == "failed"
    assert any(message in row["traceback"] for row in result["errors"]), result[
        "errors"
    ]
    assert not result["native_iterations"][-1]["complete"]


def test_native_lengths_cover_every_anchor_across_partial_batches(source):
    groups = aml_inputs.load_selections(source, limit=1)["v1"]["split"]
    original = groups["train"][0]
    groups["train"].extend(
        {
            **original,
            "sample_id": f"extra-{i}",
            "as_of_time": original["as_of_time"] + i * 86400 * 10**9,
        }
        for i in (1, 2)
    )
    path = source / "anchors/aml-data/v8/hi-small/account-daily/v1/anchors.parquet"
    table = pa.Table.from_pylist([row for rows in groups.values() for row in rows])
    table = table.set_column(
        table.schema.get_field_index("as_of_time"),
        "as_of_time",
        table["as_of_time"].cast(pa.timestamp("ns")),
    )
    pq.write_table(table, path)
    result = run(
        factory,
        {
            "data_dir": str(source),
            "family": "nuformer",
            "workload": "next-token-pretrain",
            "anchors_per_split": 3,
            "batch_size": 2,
            "threads": 1,
            "training": False,
        },
        source / "lengths",
        repetitions=1,
        epochs=2,
    )
    assert result["execution_status"] == "completed", result["errors"]
    for epoch, evidence in enumerate(result["native_iterations"]):
        [iterator] = evidence["iterators"]
        assert iterator["anchors"] == 3 and iterator["batches"] == 2
        assert iterator["merged_history_batches"] == 2
        assert iterator["lengths_computed"] == (0 if epoch else 3)
        assert iterator["length_cache_hits"] == (3 if epoch else 0)
        assert iterator["length_cache_reused"] == bool(epoch)


@pytest.mark.parametrize("workload,family", CASES)
def test_training_counts_and_saved_state_reuse(source, workload, family):
    config = {
        "data_dir": str(source),
        "family": family,
        "workload": workload,
        "anchors_per_split": 1,
        "batch_size": 1,
        "threads": 1,
        "save_fitted_state": str(source / "fitted-state"),
    }
    fresh = run(factory, config, source / "fresh", repetitions=1, epochs=2)
    assert fresh["execution_status"] == "completed", fresh["errors"]
    for epoch, evidence in enumerate(fresh["native_iterations"]):
        assert evidence["complete"] and evidence["typed_calls"] == 1
        [iterator] = evidence["iterators"]
        assert iterator["complete"] and iterator["batches"] == 1
        graph = family in {"graphsage", "combined"}
        assert iterator["merged_history_batches"] == int(family != "graphsage")
        assert iterator["length_planning"] == (not graph)
        if not graph:
            assert iterator["lengths_computed"] == int(epoch == 0)
            assert iterator["length_cache_hits"] == int(epoch > 0)
            assert iterator["length_cache_reused"] == (epoch > 0)
    coverage = json.loads((source / "fresh/coverage.json").read_text())["metrics"]
    # CPU single-batch fixture cannot measure Linux peaks or later-batch latency.
    expected_gaps = {
        "ingestion.peak_anonymous_bytes",
        "materialization.peak_anonymous_bytes",
        "batch_production.peak_anonymous_bytes",
        "ingestion.peak_host_bytes",
        "materialization.peak_host_bytes",
        "batch_production.peak_host_bytes",
        "later_batch_latency_p50_ns",
        "later_batch_latency_p95_ns",
    }
    assert {
        name for name, row in coverage.items() if row["status"] != "complete"
    } == expected_gaps
    for name in (
        "native.sample.elapsed_ns",
        "native.retrieve.elapsed_ns",
        "rows.examined",
        "rows.decoded",
    ):
        assert all(row["status"] == "available" for row in fresh["measurements"][name])
        assert any(row["value"] > 0 for row in fresh["measurements"][name])
    for epoch in fresh["epochs"]:
        values = epoch["metrics"]
        assert values["real_samples"] == 1
        assert values["optimizer_updates"] == 1
        assert values["events"] == 1
        assert values["output_bytes"] > 0
        if family != "graphsage":
            assert values["tokens"] > 0
    reused_config = {k: v for k, v in config.items() if k != "save_fitted_state"}
    reused_config["load_fitted_state"] = config["save_fitted_state"]
    reused_config["prepared_dir"] = str(source / "fresh/work/0/prepared")
    reused_config["index_dir"] = str(source / "fresh/work/0/index")
    reused = run(factory, reused_config, source / "reused", repetitions=1, epochs=2)
    assert reused["execution_status"] == "completed", reused["errors"]
    assert compare_states(fresh, reused)["status"] == "measured"
    assert "index.reopen.elapsed_ns" in reused["measurements"]
    assert not reused["measurements"].get("input.fit.elapsed_ns")
    assert not reused["measurements"].get("target.fit.elapsed_ns")
    # Original training population/cutoff is retained for graph-v2 and sequence-v1.
    fresh_rows = [
        json.loads(line)
        for line in (source / "fresh/observations.jsonl").read_text().splitlines()
    ]
    inputs = next(row for row in fresh_rows if row["kind"] == "dataset_inputs")
    assert inputs["anchors"][0]["split"] == "train"
    assert inputs["history_capacity"] == 1


def test_build_once_and_two_reopens_measure_all_stages(source):
    config = {
        "data_dir": str(source),
        "family": "graphsage",
        "workload": "account-involvement-next24h",
        "anchors_per_split": 1,
        "batch_size": 1,
        "threads": 1,
        "training": False,
        "index_dir": str(source / "shared-index"),
        "prepared_dir": str(source / "shared-prepared"),
    }
    for i, action in enumerate(("build", "reopen", "reopen")):
        result = run(
            factory,
            dict(config, index_action=action),
            source / f"lifecycle-{i}",
            repetitions=1,
            epochs=2,
        )
        assert result["execution_status"] == "completed", result["errors"]
        measurements = result["measurements"]
        assert len(measurements[f"index.{action}.elapsed_ns"]) == 1
        opposite = "reopen" if action == "build" else "build"
        assert not measurements.get(f"index.{opposite}.elapsed_ns")
        for stage in ("ingestion", "materialization", "batch_production"):
            assert all(
                r["status"] == "available" for r in measurements[f"{stage}.elapsed_ns"]
            )
    with pytest.raises(ValueError, match="Build requires"):
        factory(dict(config, index_action="build"), source / "bad-build")
