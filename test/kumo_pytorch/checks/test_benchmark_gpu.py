# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent fake-device checks; these do not establish real CUDA execution."""

import io
import json
import sys
import threading
import time
from test.kumo_pytorch.benchmarking import BatchCounts, Workload, run
from test.kumo_pytorch.benchmarking.gpu import GPUCollector, TorchCudaBackend
from test.kumo_pytorch.benchmarking.recording import Recorder
from types import SimpleNamespace

import pytest


class Backend:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []
        self.sampled = threading.Event()
        self.samples = 0
        self.resets = 0

    def ready(self):
        self.calls.append("ready")
        if self.failure == "ready":
            raise RuntimeError("no GPU")
        return {"device": "cuda:2", "allocator": "native"}

    def peaks(self):
        self.calls.append("peaks")
        return {
            "gpu.peak_allocated_bytes": 4096 if self.resets else 999999,
            "gpu.peak_reserved_bytes": 8192 if self.resets else 9999999,
        }

    def reset(self):
        self.calls.append("reset")
        if self.failure == "reset":
            raise RuntimeError("reset denied")
        self.resets += 1

    def start_utilization(self):
        if self.failure == "nvml":
            raise RuntimeError("NVML unavailable")
        return "GPU-explicit-uuid"

    def utilization(self):
        self.samples += 1
        if self.samples > 1:
            self.sampled.set()
        if self.failure == "utilization":
            raise RuntimeError("driver read failed")
        return 60

    def start_transfer(self):
        self.calls.append("transfer_begin")
        return object()

    def end_transfer(self, events):
        self.calls.append("transfer_end")

    def transfer_ns(self, events):
        self.calls.append("resolve")
        if self.failure == "pending":
            raise RuntimeError("event still pending")
        return 1250000, 7

    def close(self):
        self.calls.append("close")


def observe(backend, *, failure=False):
    stream = io.StringIO()
    recorder = Recorder(
        stream, observers=(GPUCollector("cuda:2", interval=0.001, backend=backend),)
    )
    with recorder.span("execution"):
        with recorder.span("transfer.host"):
            pass
        if backend.failure not in {"ready", "nvml"}:
            assert backend.sampled.wait(1)
            time.sleep(0.003)
        backend.calls.append("device_complete")
        if failure:
            raise ValueError("workload failed")
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def metric(rows, name):
    return next(row for row in rows if row.get("metric") == name)


def test_peaks_events_and_utilization_have_distinct_scopes():
    backend = Backend()
    rows = observe(backend)
    assert metric(rows, "gpu.peak_allocated_bytes")["value"] == 4096
    assert metric(rows, "gpu.peak_reserved_bytes")["value"] == 8192
    assert metric(rows, "transfer.device_ns")["value"] == 1250000
    utilization = metric(rows, "gpu.utilization_percent")
    assert utilization["value"] == 60
    assert utilization["details"]["sample_count"] > 0
    assert utilization["details"]["uuid"] == "GPU-explicit-uuid"
    assert 0 <= utilization["details"]["sample_span_fraction"] <= 1
    assert backend.calls.index("reset") < backend.calls.index("transfer_begin")
    assert backend.calls.index("device_complete") < backend.calls.index("resolve")
    assert backend.calls[-1] == "close"
    assert (
        next(row for row in rows if row["kind"] == "gpu_pre_reset")["peaks"][
            "gpu.peak_allocated_bytes"
        ]
        == 999999
    )


@pytest.mark.parametrize(
    "failure,name",
    [
        ("ready", "gpu.peak_allocated_bytes"),
        ("reset", "gpu.peak_reserved_bytes"),
        ("nvml", "gpu.utilization_percent"),
        ("utilization", "gpu.utilization_percent"),
        ("pending", "transfer.device_ns"),
    ],
)
def test_missing_gpu_evidence_never_becomes_zero(failure, name):
    backend = Backend(failure)
    row = metric(observe(backend), name)
    assert row["status"] == "unavailable"
    assert row["value"] is None
    assert row["reason"]
    assert backend.calls[-1] == "close"


def test_failed_window_closes_and_releases_process_guard():
    backend = Backend()
    with pytest.raises(ValueError, match="workload failed"):
        observe(backend, failure=True)
    assert backend.calls[-1] == "close"
    assert (
        metric(observe(Backend()), "gpu.peak_allocated_bytes")["status"] == "available"
    )


def test_transfer_capacity_error_is_explicit():
    stream = io.StringIO()
    backend = Backend()
    recorder = Recorder(
        stream, observers=(GPUCollector("cuda:2", backend=backend, max_transfers=0),)
    )
    with (
        pytest.raises(RuntimeError, match="capacity"),
        recorder.span("execution"),
        recorder.span("transfer.host"),
    ):
        pass
    assert (
        metric(
            [json.loads(line) for line in stream.getvalue().splitlines()],
            "gpu.peak_allocated_bytes",
        )["status"]
        == "unavailable"
    )


def test_torch_backend_requires_initialized_cuda_without_initializing(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)),
    )
    with pytest.raises(RuntimeError, match="already be initialized"):
        TorchCudaBackend("cuda:0").ready()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("GPU-abc", "GPU-abc"),
        (
            SimpleNamespace(bytes=list(range(16))),
            "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f",
        ),
    ],
)
def test_nvml_identity_uses_uuid_not_cuda_ordinal(monkeypatch, raw, expected):
    calls = []
    cuda = SimpleNamespace(
        is_initialized=lambda: True,
        current_device=lambda: 3,
        get_allocator_backend=lambda: "native",
        get_device_properties=lambda device: SimpleNamespace(
            uuid=raw, name="test", total_memory=100
        ),
    )
    nvml = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlDeviceGetHandleByUUID=lambda uuid: calls.append(uuid) or "handle",
        nvmlDeviceGetUUID=lambda handle: expected.encode("ascii"),
        nvmlShutdown=lambda: calls.append("close"),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    backend = TorchCudaBackend("cuda")
    assert backend.ready()["device"] == "cuda:3"
    assert backend.start_utilization() == expected
    backend.close()
    assert calls == ["init", expected, "close"]


def test_runner_collects_each_gpu_epoch_and_reports_coverage(monkeypatch, tmp_path):
    backend = Backend()
    monkeypatch.setattr(
        "test.kumo_pytorch.benchmarking.gpu.TorchCudaBackend", lambda device: backend
    )

    def factory(config, directory):
        def step(batch, recorder):
            with recorder.span("transfer.host"):
                pass
            backend.sampled.clear()
            assert backend.sampled.wait(1)
            time.sleep(0.003)
            return True

        return Workload(
            identity={"dataset": "GPU-interface-fixture", "workload": "custom-model"},
            prepare=lambda recorder: None,
            batches=lambda epoch: iter([[1]]),
            count=lambda batch: BatchCounts(1, 0),
            step=step,
            device="cuda:2",
            synchronize=lambda: backend.calls.append("device_complete"),
        )

    result = run(
        factory, {}, tmp_path / "run", repetitions=1, epochs=2, gpu_interval=0.001
    )
    assert result["execution_status"] == "completed", result["errors"]
    assert result["status"] == "incomplete"
    assert backend.resets == 2
    coverage = json.loads((tmp_path / "run/coverage.json").read_text())
    for name in (
        "gpu.peak_allocated_bytes",
        "gpu.peak_reserved_bytes",
        "gpu.utilization_percent",
        "transfer.device_ns",
    ):
        assert coverage["metrics"][name]["status"] == "complete"
    assert len(result["gpu_observations"]) == 4
    assert "GPU sampling coverage" in (tmp_path / "run/report.md").read_text()


def test_event_end_precedes_transfer_record_serialization():
    backend = Backend()

    class Stream(io.StringIO):
        def write(self, value):
            if '"metric": "transfer.host.elapsed_ns"' in value:
                assert "transfer_end" in backend.calls
            return super().write(value)

    recorder = Recorder(Stream(), observers=(GPUCollector("cuda:2", backend=backend),))
    with recorder.span("execution"), recorder.span("transfer.host"):
        pass


@pytest.mark.parametrize("interval", (0, -1, float("nan"), float("inf")))
def test_invalid_sampling_interval(interval):
    with pytest.raises(ValueError):
        GPUCollector("cuda:0", interval=interval)


@pytest.mark.parametrize(
    "failure", (None, "missing_epoch", "unavailable", "lifetime_peak")
)
def test_live_validation_rejects_incomplete_or_lifetime_evidence(tmp_path, failure):
    from test.kumo_pytorch.benchmarking.validate_gpu import check

    rows = []
    for epoch in range(2):
        if failure == "missing_epoch" and epoch == 1:
            continue
        for metric, values in {
            "gpu.peak_allocated_bytes": [
                (80 if epoch == 0 or failure == "lifetime_peak" else 20) * 1024**2
            ],
            "gpu.peak_reserved_bytes": [100 * 1024**2],
            "gpu.utilization_percent": [50],
            "transfer.device_ns": [1000] * 256,
        }.items():
            for value in values:
                rows.append(
                    {
                        "scope": {"epoch": epoch},
                        "metric": metric,
                        "value": value,
                        "status": "unavailable"
                        if failure == "unavailable"
                        else "available",
                    }
                )
    (tmp_path / "observations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows)
    )
    result = {"execution_status": "completed"}
    if failure:
        with pytest.raises(AssertionError):
            check(tmp_path, result)
    else:
        check(tmp_path, result)
