"""CUDA window peaks, deferred transfer events and UUID-matched NVML samples.

Measurements follow Arena's boundary rules without an Arena dependency. No
collector initializes CUDA or synchronizes a batch; the workload owns completion.
"""

import importlib
import itertools
import math
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from uuid import UUID

_ACTIVE = threading.Lock()


class TorchCudaBackend:
    def __init__(self, device):
        self.device = device
        self.nvml = None

    def ready(self):
        if self.device.split(":")[0] != "cuda":
            raise RuntimeError("CUDA collector requires a CUDA device")
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_initialized():
            raise RuntimeError("CUDA must already be initialized by the workload")
        self.cuda = torch.cuda
        if self.device == "cuda":
            self.device = f"cuda:{self.cuda.current_device()}"
        self.properties = self.cuda.get_device_properties(self.device)
        self.allocator = self.cuda.get_allocator_backend()
        return {
            "device": self.device,
            "name": self.properties.name,
            "allocator": self.allocator,
            "total_memory": self.properties.total_memory,
            "scope": "selected_device_current_process_all_threads_and_streams",
        }

    def peaks(self):
        if self.allocator != "native":
            raise RuntimeError(
                f"Allocator peak semantics not validated for {self.allocator}"
            )
        stats = self.cuda.memory_stats(self.device)
        result = {}
        for name, field in {
            "gpu.peak_allocated_bytes": "allocated_bytes.all.peak",
            "gpu.peak_reserved_bytes": "reserved_bytes.all.peak",
        }.items():
            value = stats.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"Missing/invalid CUDA allocator statistic: {field}")
            result[name] = value
        return result

    def reset(self):
        self.cuda.reset_peak_memory_stats(self.device)

    def start_utilization(self):
        raw = getattr(self.properties, "uuid", None)
        # PyTorch exposes CUDA's CUuuid object (16 bytes), not always a str.
        if raw is not None and hasattr(raw, "bytes"):
            raw = str(UUID(bytes=bytes(raw.bytes)))
        if isinstance(raw, bytes):
            raw = raw.decode("ascii")
        if not isinstance(raw, str) or not raw:
            raise RuntimeError(
                "CUDA device does not expose a UUID; ordinal matching is forbidden"
            )
        uuid = raw if raw.startswith(("GPU-", "MIG-")) else "GPU-" + raw
        nvml = importlib.import_module("pynvml")
        nvml.nvmlInit()
        self.nvml = nvml
        self.handle = nvml.nvmlDeviceGetHandleByUUID(uuid)
        observed = nvml.nvmlDeviceGetUUID(self.handle)
        if isinstance(observed, bytes):
            observed = observed.decode("ascii")
        if observed != uuid:
            raise RuntimeError("NVML returned a different GPU UUID")
        return uuid

    def utilization(self):
        return self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu

    def start_transfer(self):
        stream = self.cuda.current_stream(self.device)
        begin = self.cuda.Event(enable_timing=True)
        end = self.cuda.Event(enable_timing=True)
        begin.record(stream)
        return stream, begin, end

    def end_transfer(self, events):
        stream, _, end = events
        if self.cuda.current_stream(self.device).cuda_stream != stream.cuda_stream:
            raise RuntimeError("Transfer changed the observed CUDA stream")
        end.record(stream)

    def transfer_ns(self, events):
        stream, begin, end = events
        if not end.query():
            raise RuntimeError("Transfer event is pending after workload completion")
        return begin.elapsed_time(end) * 1e6, int(stream.cuda_stream)

    def close(self):
        if self.nvml is not None:
            nvml, self.nvml = self.nvml, None
            nvml.nvmlShutdown()


class GPUCollector:
    def __init__(
        self,
        device,
        *,
        interval=1.0,
        backend=None,
        max_samples=100000,
        max_transfers=10000,
    ):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("GPU sampling interval must be positive and finite")
        self.device = device
        self.interval = interval
        self.backend = backend or TorchCudaBackend(device)
        self.max_samples = max_samples
        self.max_transfers = max_transfers
        self.active = False

    def _poll(self):
        while not self.stop.is_set():
            if len(self.samples) >= self.max_samples:
                self.sampling_error = "GPU sample capacity exceeded"
                return
            row = {"start_ns": time.monotonic_ns()}
            try:
                value = self.backend.utilization()
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not 0 <= value <= 100
                ):
                    raise ValueError("Invalid GPU utilization reading")
                row.update(value=value, status="available")
            except Exception as error:  # noqa: BLE001 -- driver failures are retained observations
                row.update(
                    value=None,
                    status="error",
                    reason=f"{type(error).__name__}: {error}",
                )
            row["end_ns"] = time.monotonic_ns()
            self.samples.append(row)
            self.stop.wait(self.interval)

    @contextmanager
    def stage(self, operation, recorder):
        if operation == "transfer.host":
            with recorder.defer(), self._transfer(recorder) as details:
                yield details
            return
        if operation != "execution":
            yield
            return
        if not _ACTIVE.acquire(blocking=False):
            raise RuntimeError("GPU observation window already active in this process")
        details, identity = {}, {"device": self.device}
        self.active = True
        self.samples, self.transfers = [], []
        self.stop = threading.Event()
        self.sampling_error = self.ready_error = self.peak_error = None
        thread, uuid = None, None
        completed = False
        try:
            try:
                identity.update(self.backend.ready())
            except Exception as error:  # noqa: BLE001 -- unavailable SDK/device is a coverage failure
                self.ready_error = f"{type(error).__name__}: {error}"
            if self.ready_error is None:
                try:
                    before = self.backend.peaks()
                    recorder.write("gpu_pre_reset", identity=identity, peaks=before)
                    self.backend.reset()
                except Exception as error:  # noqa: BLE001 -- do not substitute lifetime peaks
                    self.peak_error = f"{type(error).__name__}: {error}"
                try:
                    uuid = self.backend.start_utilization()
                    thread = threading.Thread(target=self._poll, daemon=True)
                    thread.start()
                except Exception as error:  # noqa: BLE001 -- NVML may be unavailable independently
                    self.sampling_error = f"{type(error).__name__}: {error}"
            yield details
            completed = details.get("outcome") == "completed"
        finally:
            try:
                self.stop.set()
                if thread is not None:
                    thread.join(timeout=5)
                    if thread.is_alive():
                        raise RuntimeError(
                            "NVML sampling call did not finish; GPU recording failed"
                        )
                self._finish(recorder, details, identity, uuid, completed)
            finally:
                self.active = False
                try:
                    if thread is None or not thread.is_alive():
                        self.backend.close()
                finally:
                    _ACTIVE.release()

    def _finish(self, recorder, window, identity, uuid, completed):
        reason = (
            self.ready_error
            or self.peak_error
            or (None if completed else "Execution window failed")
        )
        peaks = {}
        if reason is None:
            try:
                peaks = self.backend.peaks()
            except Exception as error:  # noqa: BLE001 -- preserve absent counters
                reason = f"{type(error).__name__}: {error}"
        for metric in ("gpu.peak_allocated_bytes", "gpu.peak_reserved_bytes"):
            recorder.measure(
                metric,
                peaks.get(metric) if reason is None else None,
                status="available" if reason is None else "unavailable",
                reason=reason,
                identity=identity,
                includes_live_allocations=True,
                window_start_ns=window.get("start_ns"),
                window_end_ns=window.get("end_ns"),
            )
        start, end = window.get("start_ns", 0), window.get("end_ns", 0)
        rows = [
            row
            for row in self.samples
            if start <= row["start_ns"] <= row["end_ns"] <= end
        ]
        good = [row for row in rows if row["status"] == "available"]
        points = [row["start_ns"] for row in good]
        coverage = {
            "sample_count": len(good),
            "failed_samples": len(rows) - len(good),
            "interval_seconds": self.interval,
            "window_start_ns": start,
            "window_end_ns": end,
            "leading_gap_ns": points[0] - start if points else None,
            "trailing_gap_ns": end - good[-1]["end_ns"] if points else None,
            "maximum_gap_ns": max(
                (b - a for a, b in itertools.pairwise(points)), default=None
            ),
            "sample_span_fraction": (good[-1]["end_ns"] - good[0]["start_ns"])
            / (end - start)
            if points and end > start
            else None,
            "scope": "device_wide_driver_samples; not a time_integral or steady_state_claim",
            "uuid": uuid,
        }
        recorder.write(
            "gpu_utilization_samples", samples=self.samples, coverage=coverage
        )
        reason = (
            self.ready_error
            or self.sampling_error
            or ("Execution window failed" if not completed else None)
            or ("No in-window utilization samples" if not good else None)
            or ("Some utilization reads failed" if len(good) != len(rows) else None)
        )
        recorder.measure(
            "gpu.utilization_percent",
            statistics.mean(row["value"] for row in good) if reason is None else None,
            status="available" if reason is None else "unavailable",
            reason=reason,
            **coverage,
        )
        for events, host, error in self.transfers:
            stream = value = None
            reason = error or (None if completed else "Execution window failed")
            if reason is None:
                try:
                    value, stream = self.backend.transfer_ns(events)
                except Exception as caught:  # noqa: BLE001 -- never synchronize to rescue an unfinished event
                    reason = f"{type(caught).__name__}: {caught}"
            recorder.measure(
                "transfer.device_ns",
                value if reason is None else None,
                status="available" if reason is None else "unavailable",
                reason=reason,
                device=self.device,
                stream=stream,
                host_interval=host,
                timing_scope="CUDA current-stream events; not DMA busy time or additive with host",
            )

    @contextmanager
    def _transfer(self, recorder):
        host = {}
        if not self.active:
            try:
                yield host
            finally:
                recorder.measure(
                    "transfer.device_ns",
                    status="unavailable",
                    reason="Transfer outside an execution window",
                )
            return
        if len(self.transfers) >= self.max_transfers:
            raise RuntimeError("CUDA transfer event capacity exceeded")
        events, reason = None, self.ready_error
        if reason is None:
            try:
                events = self.backend.start_transfer()
            except Exception as error:  # noqa: BLE001 -- preserve event failures without hiding host execution
                reason = f"{type(error).__name__}: {error}"
        try:
            yield host
        finally:
            if events is not None:
                try:
                    self.backend.end_transfer(events)
                except Exception as error:  # noqa: BLE001 -- event failure is reported after the window
                    reason = f"{type(error).__name__}: {error}"
            if host.get("outcome") != "completed":
                reason = "Host transfer failed"
            self.transfers.append((events, host, reason))
