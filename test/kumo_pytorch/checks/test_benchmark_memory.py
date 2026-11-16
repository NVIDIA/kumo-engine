"""Control-flow tests for kernel peak collection; no Linux peak claims on macOS."""

import io
import json
import threading

import pytest

from test.kumo_pytorch.benchmarking.memory import CgroupMemory, SampledCgroupMemory
from test.kumo_pytorch.benchmarking.recording import Recorder


def fake_kernel(monkeypatch, *, read=b"8192\n", reset_error=None):
    events = []
    monkeypatch.setattr(
        CgroupMemory, "validate", lambda self: events.append("validate")
    )
    monkeypatch.setattr(
        "test.kumo_pytorch.benchmarking.memory.os.open",
        lambda *a: events.append("open") or 91,
    )

    def write(fd, value):
        events.append(("reset", fd, value))
        if reset_error:
            raise reset_error
        return len(value)

    monkeypatch.setattr("test.kumo_pytorch.benchmarking.memory.os.write", write)
    monkeypatch.setattr(
        "test.kumo_pytorch.benchmarking.memory.os.lseek",
        lambda *a: events.append(("seek", *a)),
    )
    monkeypatch.setattr(
        "test.kumo_pytorch.benchmarking.memory.os.read",
        lambda fd, size: events.append(("read", fd)) or read,
    )
    monkeypatch.setattr(
        "test.kumo_pytorch.benchmarking.memory.os.close",
        lambda fd: events.append(("close", fd)),
    )
    return events


def test_same_descriptor_reset_read_and_stage_scope(monkeypatch, tmp_path):
    stream = io.StringIO()
    with monkeypatch.context() as patches:
        events = fake_kernel(patches)
        recorder = Recorder(stream, observers=(CgroupMemory(tmp_path),))
        with recorder.span("materialization"):
            events.append("operation")
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    peak = next(
        row for row in rows if row["metric"] == "materialization.peak_host_bytes"
    )
    assert peak["value"] == 8192
    assert (
        peak["details"]["memory_scope"]
        == "cgroup_and_descendants_including_file_cache_and_kernel_charges"
    )
    assert events == [
        "validate",
        "open",
        ("reset", 91, b"0"),
        "operation",
        "validate",
        ("seek", 91, 0, 0),
        ("read", 91),
        ("close", 91),
    ]


@pytest.mark.parametrize("failure", ("reset", "read", "operation"))
def test_failure_closes_descriptor_and_does_not_emit_false_peak(
    monkeypatch, tmp_path, failure
):
    stream = io.StringIO()
    with monkeypatch.context() as patches:
        events = fake_kernel(
            patches,
            read=b"bad" if failure == "read" else b"999",
            reset_error=PermissionError("read only") if failure == "reset" else None,
        )
        recorder = Recorder(stream, observers=(CgroupMemory(tmp_path),))
        if failure == "operation":
            with (
                pytest.raises(ValueError, match="original"),
                recorder.span("ingestion"),
            ):
                raise ValueError("original")
        else:
            with recorder.span("ingestion"):
                pass
    assert ("close", 91) in events
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    peak = next(row for row in rows if row["metric"] == "ingestion.peak_host_bytes")
    assert peak["value"] is None
    assert peak["status"] == ("error" if failure == "operation" else "unavailable")


def test_invalid_cgroup_does_not_write_files(tmp_path):
    stream = io.StringIO()
    recorder = Recorder(stream, observers=(CgroupMemory(tmp_path),))
    with recorder.span("batch.request"):
        pass
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert rows[-1]["metric"] == "batch_production.peak_host_bytes"
    assert rows[-1]["status"] == "unavailable"
    assert list(tmp_path.iterdir()) == []


def test_deferred_observations_are_serialized_after_request():
    stream = io.StringIO()
    recorder = Recorder(stream)
    with recorder.defer():
        with recorder.span("batch.request"):
            with recorder.span("python.encode"):
                recorder.measure("rows.decoded", 7)
            assert stream.getvalue() == ""
        assert stream.getvalue() == ""
    assert [json.loads(line)["metric"] for line in stream.getvalue().splitlines()] == [
        "rows.decoded",
        "python.encode.elapsed_ns",
        "batch.request.elapsed_ns",
    ]


def test_deferral_preserves_failure_evidence():
    stream = io.StringIO()
    recorder = Recorder(stream)
    with pytest.raises(ValueError), recorder.defer(), recorder.span("batch.request"):
        raise ValueError("failure")
    assert json.loads(stream.getvalue())["status"] == "error"
    assert recorder.pending == []
    assert recorder.deferral == 0


def test_sampled_peak_captures_interior_read_and_resets_between_stages(
    monkeypatch, tmp_path
):
    seen = threading.Event()
    main = threading.current_thread()
    current = [10]

    def read(self):
        value = current[0]
        if threading.current_thread() is not main and value == 1000:
            seen.set()
        return value

    monkeypatch.setattr(CgroupMemory, "validate", lambda self: None)
    monkeypatch.setattr(SampledCgroupMemory, "_read_current", read)
    collector = SampledCgroupMemory(tmp_path, interval=0.001)
    stream = io.StringIO()
    recorder = Recorder(stream, observers=(collector,))
    with recorder.span("ingestion"):
        current[0] = 1000
        assert seen.wait(2), "Sampler did not observe the held allocation"
        current[0] = 10
    with recorder.span("materialization"):
        pass
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    peaks = [row for row in rows if row["metric"].endswith(".peak_host_bytes")]
    assert [row["value"] for row in peaks] == [1000, 10]
    first = peaks[0]["details"]
    assert first["method"] == "sampled_cgroup_v2_memory_current"
    assert first["precision"] == "sampled_maximum_not_kernel_high_water_mark"
    assert first["periodic_sample_count"] >= 1
    assert first["samples"][0]["value"] == first["samples"][-1]["value"] == 10
    assert first["operation_boundaries"]["outcome"] == "completed"
    assert peaks[1]["details"]["sample_count"] >= 2


@pytest.mark.parametrize("failure", ("read", "capacity", "operation"))
def test_sampled_failure_never_becomes_a_successful_peak(
    monkeypatch, tmp_path, failure
):
    monkeypatch.setattr(CgroupMemory, "validate", lambda self: None)

    def read(self):
        if failure == "read":
            raise PermissionError("denied")
        return 10

    monkeypatch.setattr(SampledCgroupMemory, "_read_current", read)
    collector = SampledCgroupMemory(tmp_path)
    stream = io.StringIO()
    recorder = Recorder(stream, observers=(collector,))
    if failure == "operation":
        with pytest.raises(ValueError, match="original"), recorder.span("ingestion"):
            raise ValueError("original")
    else:
        with recorder.span("ingestion"):
            if failure == "capacity":
                collector.max_samples = 1
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    row = next(row for row in rows if row["metric"] == "ingestion.peak_host_bytes")
    assert row["status"] == ("error" if failure == "operation" else "unavailable")
    assert row["value"] is None


def test_sampled_method_is_visible_in_report_and_configuration(monkeypatch, tmp_path):
    from test.kumo_pytorch.benchmarking import BatchCounts, Workload, run

    monkeypatch.setattr(CgroupMemory, "validate", lambda self: None)
    monkeypatch.setattr(SampledCgroupMemory, "_read_current", lambda self: 8192)

    def factory(config, workdir):
        def prepare(recorder):
            with recorder.span("ingestion"):
                pass

        return Workload(
            identity={"dataset": "fixture", "workload": "sampling"},
            prepare=prepare,
            batches=lambda epoch: iter([1]),
            count=lambda batch: BatchCounts(1, 0),
        )

    output = tmp_path / "run"
    result = run(
        factory,
        {},
        output,
        repetitions=1,
        epochs=1,
        memory_cgroup=tmp_path,
        memory_method="sampled",
        memory_interval=0.005,
    )
    assert result["measurements"]["ingestion.peak_host_bytes"][0]["value"] == 8192
    assert (
        "Sampled maximum; between-sample peaks may be missed"
        in (output / "report.md").read_text()
    )
    assert (
        json.loads((output / "config.json").read_text())["memory_interval_seconds"]
        == 0.005
    )
