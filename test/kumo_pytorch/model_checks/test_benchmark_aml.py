"""Exercise the real AML benchmark binding using tiny local source files."""

import hashlib
import json
from datetime import UTC, datetime

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
    coverage = json.loads((source / "fresh/coverage.json").read_text())["metrics"]
    # CPU single-batch fixture cannot measure Linux peaks or later-batch latency.
    expected_gaps = {
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
