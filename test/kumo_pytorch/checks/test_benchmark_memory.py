"""Control-flow tests for kernel peak collection; no Linux peak claims on macOS."""

import io
import json

import pytest

from test.kumo_pytorch.benchmarking.memory import CgroupMemory
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
