"""Independent timing/count oracles and unrelated adapters for the generic API."""

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from test.kumo_pytorch.benchmarking import BatchCounts, Workload, run
from test.kumo_pytorch.benchmarking.metrics import METRICS
from test.kumo_pytorch.benchmarking.recording import Recorder


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 100
        return self.value


def tabular(config, workdir):
    """A plain numeric dataset with neither graph nor token representation."""
    batches = [[2, 4, 6], [8]]
    return Workload(
        identity={
            "dataset": "numeric-fixture",
            "workload": "numeric-delivery",
            "model": None,
        },
        prepare=lambda recorder: recorder.measure("source.rows", 4),
        batches=lambda epoch: iter(batches),
        count=lambda batch: BatchCounts(
            len(batch), 0, output_elements=len(batch), output_bytes=8 * len(batch)
        ),
        inapplicable={
            name: "This numeric workload has no token sequence"
            for name in ("tokens", "tokens_per_second")
        },
    )


def graph(config, workdir):
    """Different payload and custom training consumer, with request padding."""

    def step(batch, recorder):
        with recorder.span("model.forward"):
            batch["parameter"] -= 0.1
        return batch["samples"] > batch["padding"]

    return Workload(
        identity={
            "dataset": "relationships-fixture",
            "workload": "edge-regression",
            "model": "custom-scalar",
        },
        prepare=lambda recorder: None,
        batches=lambda epoch: iter(
            [
                {
                    "samples": 3,
                    "padding": 1,
                    "nodes": 11,
                    "edges": 13,
                    "parameter": 1.0,
                },
                {"samples": 2, "padding": 0, "nodes": 5, "edges": 7, "parameter": 1.0},
            ]
        ),
        count=lambda batch: BatchCounts(
            batch["samples"],
            batch["padding"],
            nodes=batch["nodes"],
            edges=batch["edges"],
        ),
        step=step,
    )


def execute(tmp_path, factory=tabular, **kwargs):
    return run(
        factory, {}, tmp_path / "run", repetitions=1, epochs=1, clock=Clock(), **kwargs
    )


def test_numeric_and_graph_adapters_have_independent_counts(tmp_path):
    numeric = execute(tmp_path / "numeric")
    trained = execute(tmp_path / "graph", graph)
    a, b = [r["epochs"][0] for r in (numeric, trained)]
    assert a["metrics"]["real_samples"] == 4
    assert a["metrics"]["output_bytes"] == 32
    assert "tokens_per_second" not in a["metrics"]
    assert b["metrics"]["real_samples"] == 4
    assert b["metrics"]["samples"] == 5
    assert b["metrics"]["nodes"] == 16
    assert b["metrics"]["edges"] == 20
    assert b["metrics"]["steps"] == b["metrics"]["optimizer_updates"] == 2
    assert b["metrics"]["samples_per_second"] == 4 * 1e9 / b["elapsed_ns"]
    assert b["metrics"]["edges_per_second"] == 20 * 1e9 / b["elapsed_ns"]
    assert numeric["status"] == trained["status"] == "incomplete"
    assert numeric["execution_status"] == trained["execution_status"] == "completed"


def test_complete_catalog_and_raw_evidence_in_reports(tmp_path):
    result = execute(tmp_path)
    directory = tmp_path / "run"
    assert {p.name for p in directory.iterdir() if p.is_file()} == {
        "config.json",
        "results.json",
        "coverage.json",
        "observations.jsonl",
        "report.md",
    }
    coverage = json.loads((directory / "coverage.json").read_text())
    assert set(coverage["metrics"]) == set(METRICS)
    assert coverage["metrics"]["ingestion.peak_host_bytes"]["status"] == "incomplete"
    assert coverage["metrics"]["tokens"]["cases"][0]["status"] == "not_applicable"
    report = (directory / "report.md").read_text()
    assert all(name in report for name in METRICS)
    requests = result["measurements"]["batch.request.elapsed_ns"]
    assert [row["details"]["outcome"] for row in requests] == [
        "produced",
        "produced",
        "exhausted",
    ]
    assert len(result["epochs"]) == 1
    assert "unavailable" in report


def test_request_clock_excludes_record_delivery(tmp_path):
    # Counting and recording advance the clock but may not enter request latency.
    result = execute(tmp_path)
    epoch = result["epochs"][0]
    assert epoch["metrics"]["batch_latency_p50_ns"] == 100
    assert epoch["metrics"]["batch_latency_p95_ns"] == 100
    assert epoch["metrics"]["batch_wait_ns"] == 300  # includes exhaustion
    assert epoch["elapsed_ns"] > epoch["metrics"]["batch_wait_ns"]
    assert epoch["metrics"]["first_batch_ns"] > epoch["metrics"]["first_request_ns"]


def test_percentiles_have_independent_oracle(tmp_path):
    clock = Clock()

    def build(config, directory):
        workload = tabular(config, directory)

        def batches(epoch):
            for duration in (1000, 2000, 9000, 3000):
                clock.value += duration
                yield [1]

        workload.batches = batches
        return workload

    result = run(build, {}, tmp_path / "run", repetitions=1, epochs=1, clock=clock)
    values = result["epochs"][0]["metrics"]
    assert values["batch_latency_p50_ns"] == 2600
    assert values["batch_latency_p95_ns"] == 9100
    assert values["later_batch_latency_p50_ns"] == 3100


def test_empty_input_does_not_manufacture_latency(tmp_path):
    def build(config, directory):
        workload = tabular(config, directory)
        workload.batches = lambda epoch: iter(())
        return workload

    result = execute(tmp_path, build)
    values = result["epochs"][0]["metrics"]
    assert values["samples_per_second"] == values["real_samples"] == 0
    assert "first_batch_ns" not in values
    assert "batch_latency_p95_ns" not in values


def test_partial_missing_counts_do_not_produce_partial_rate(tmp_path):
    def build(config, directory):
        workload = tabular(config, directory)
        workload.inapplicable = {}
        workload.count = lambda batch: BatchCounts(
            len(batch), 0, tokens=6 if len(batch) == 3 else None
        )
        return workload

    result = execute(tmp_path, build)
    assert "tokens" not in result["epochs"][0]["metrics"]
    assert "tokens_per_second" not in result["epochs"][0]["metrics"]


def test_missing_metric_in_later_epoch_fails_coverage(tmp_path):
    def build(config, directory):
        workload = tabular(config, directory)
        workload.inapplicable = {}
        state = {"epoch": None}

        def batches(epoch):
            state["epoch"] = epoch
            return iter([[1]])

        workload.batches = batches
        workload.count = lambda batch: BatchCounts(
            1, 0, tokens=2 if state["epoch"] == 0 else None
        )
        return workload

    run(build, {}, tmp_path / "run", repetitions=1, epochs=2)
    item = json.loads((tmp_path / "run/coverage.json").read_text())["metrics"]["tokens"]
    assert item["status"] == "incomplete"
    assert [case["status"] for case in item["cases"]] == ["measured", "missing"]


def test_reference_fitting_belongs_to_setup_not_every_epoch(tmp_path):
    def build(config, directory):
        workload = tabular(config, directory)

        def prepare(recorder):
            with recorder.span("reference.fit"):
                pass

        workload.prepare = prepare
        return workload

    run(build, {}, tmp_path / "run", repetitions=1, epochs=2)
    item = json.loads((tmp_path / "run/coverage.json").read_text())["metrics"][
        "reference.fit.elapsed_ns"
    ]
    assert item["status"] == "complete"
    assert len(item["cases"]) == 1


def test_training_time_is_not_reported_as_loader_time(tmp_path):
    clock = Clock()

    def build(config, directory):
        workload = graph(config, directory)

        def step(batch, recorder):
            clock.value += 1_000_000
            return True

        workload.step = step
        return workload

    result = run(build, {}, tmp_path / "run", repetitions=1, epochs=1, clock=clock)
    elapsed = result["epochs"][0]["elapsed_ns"]
    production = result["measurements"]["batch_production.elapsed_ns"][0]
    assert elapsed >= 2_000_000
    assert production["value"] == 400
    assert production["details"]["timing_scope"] == "iterator_and_request_intervals"


def test_device_adapter_requires_completion_callback(tmp_path):
    def build(config, directory):
        workload = tabular(config, directory)
        workload.device = "custom-accelerator"
        return workload

    result = execute(tmp_path, build)
    assert result["execution_status"] == "failed"
    assert "completion callback" in result["errors"][0]["traceback"]


def test_device_completion_is_inside_window_stop(tmp_path):
    clock, fences = Clock(), []

    def build(config, directory):
        workload = graph(config, directory)
        workload.device = "test-accelerator"

        def synchronize():
            fences.append(clock.value)
            clock.value += 1_000_000

        workload.synchronize = synchronize
        return workload

    result = run(build, {}, tmp_path / "run", repetitions=1, epochs=1, clock=clock)
    assert len(fences) == 3  # start, warmup boundary, end; no per-batch fence
    assert 2_000_000 < result["epochs"][0]["elapsed_ns"] < 3_000_000


def test_post_warmup_window_excludes_first_batches(tmp_path):
    def build(config, directory):
        workload = tabular(config, directory)
        workload.batches = lambda epoch: iter([[1] * n for n in (100, 100, 1, 2, 3, 4)])
        return workload

    result = execute(tmp_path, build)
    (window,) = result["steady_windows"]
    assert window["real_samples"] == 10
    assert window["measured_batches"] == 4
    assert window["samples_per_second"] == 10 * 1e9 / window["elapsed_ns"]
    assert result["experiments"]["steady_state"]["status"] == "measured"


@pytest.mark.parametrize(
    "failure", ["prepare", "batches", "count", "step", "synchronize", "close"]
)
def test_failure_retains_reports_and_closes(tmp_path, failure):
    closed = []

    def fail(*args):
        raise ValueError("deliberate failure")

    def build(config, directory):
        workload = tabular(config, directory)

        def close():
            closed.append(True)
            if failure == "close":
                fail()

        workload.close = close
        if failure != "close":
            setattr(workload, failure, fail)
        return workload

    result = execute(tmp_path, build)
    assert result["status"] == "failed"
    assert closed == [True]
    assert "deliberate failure" in (tmp_path / "run/report.md").read_text()
    if failure in {"batches", "count", "step"}:
        assert not result["epochs"][0]["complete"]
        assert "samples_per_second" not in result["epochs"][0]["metrics"]


def test_repetitions_get_separate_workloads_and_epochs(tmp_path):
    calls = []

    def build(config, directory):
        calls.append(directory)
        return tabular(config, directory)

    result = run(build, {}, tmp_path / "run", repetitions=2, epochs=2, clock=Clock())
    assert len(set(calls)) == 2
    assert [
        (r["scope"]["repetition"], r["scope"]["epoch"]) for r in result["epochs"]
    ] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert result["experiments"]["independent_repeated_runs"]["status"] == "pending"


def test_existing_output_preserved(tmp_path):
    execute(tmp_path)
    before = (tmp_path / "run/results.json").read_bytes()
    with pytest.raises(FileExistsError):
        execute(tmp_path)
    assert (tmp_path / "run/results.json").read_bytes() == before


@pytest.mark.parametrize(
    "values",
    [
        {"samples": True, "padding_samples": 0},
        {"samples": 1, "padding_samples": 2},
        {"samples": 1, "padding_samples": 0, "tokens": -1},
    ],
)
def test_counts_reject_invalid_values(values):
    with pytest.raises(ValueError):
        BatchCounts(**values)


def test_nested_spans_keep_identity_and_do_not_add_times():
    stream = io.StringIO()
    recorder = Recorder(stream, clock=Clock())
    with recorder.span("materialization"), recorder.span("input.fit"):
        pass
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert rows[0]["parent"] == rows[1]["details"]["span_id"]
    assert rows[0]["value"] < rows[1]["value"]


@pytest.mark.parametrize(
    "name,value",
    [
        ("samples", float("nan")),
        ("samples", 1.5),
        ("source.bytes", -1),
        ("gpu.utilization_percent", 101),
    ],
)
def test_invalid_readings_rejected(name, value):
    with pytest.raises(ValueError):
        Recorder(io.StringIO()).measure(name, value)


def test_applicability_cannot_hide_sample_requirement(tmp_path):
    def build(config, directory):
        workload = tabular(config, directory)
        workload.inapplicable = {"samples_per_second": "not implemented"}
        return workload

    result = execute(tmp_path, build)
    assert result["status"] == "failed"


def test_cli_nonzero_on_missing_coverage(tmp_path):
    settings = tmp_path / "config.json"
    settings.write_text("{}")
    root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable,
        "-m",
        "test.kumo_pytorch.benchmark",
        "--adapter",
        "test.kumo_pytorch.checks.test_benchmark:tabular",
        "--config",
        str(settings),
        "--output",
        str(tmp_path / "run"),
        "--epochs",
        "1",
        "--repetitions",
        "1",
    ]
    completed = subprocess.run(
        command, cwd=root, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 2, completed.stderr
    assert "incomplete" in completed.stdout
    assert (tmp_path / "run/report.md").is_file()
