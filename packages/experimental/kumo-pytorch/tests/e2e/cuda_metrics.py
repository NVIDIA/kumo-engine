# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe the selected device during a complete CUDA epoch in the package runner."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from itertools import pairwise
from statistics import mean
from threading import Event, Lock
from time import perf_counter_ns

import torch


class _Window:
    def __init__(self, device, data, sample):
        self.device, self.data, self.sample = device, data, sample
        self.events = []

    def transfer(self, batch, *, start_steady=False):
        """Measure the host dispatch and queue an event pair on the actual copy stream."""
        if start_steady:
            self.data["steady_start_ns"] = perf_counter_ns()
            self.sample("steady_start")
        if len(self.events) >= 10_000:
            raise RuntimeError("CUDA transfer event capacity exceeded")
        stream = torch.cuda.current_stream(self.device)
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record(stream)
        record = {"stream": stream.cuda_stream}
        self.data["transfers"].append(record)
        start = perf_counter_ns()
        try:
            result = batch.to(self.device, non_blocking=True)
        finally:
            record["dispatch_ns"] = perf_counter_ns() - start
        if torch.cuda.current_stream(self.device) != stream:
            raise RuntimeError("CUDA stream changed during batch transfer")
        end.record(stream)
        self.events.append((begin, end))
        return result


@contextmanager
def cuda_epoch(device, row):
    """Reset native allocator peaks and resolve transfer events after epoch synchronization."""
    data = row["cuda"] = {
        "complete": False,
        "transfers": [],
        "samples": [],
        "sample_interval_ns": 100_000_000,
    }
    samples = data["samples"]
    try:
        import pynvml as nvml

        device = torch.cuda.current_stream(device).device
        props = torch.cuda.get_device_properties(device)
        raw = props.uuid
        raw = raw.decode("ascii") if isinstance(raw, bytes) else str(raw)
        uuid = raw if raw.startswith(("GPU-", "MIG-")) else "GPU-" + raw
        allocator = torch.cuda.get_allocator_backend()
        data.update(
            device=str(device),
            device_uuid=uuid,
            name=props.name,
            allocator=allocator,
            utilization_scope="device-wide NVML snapshots",
        )
        if allocator != "native":
            raise RuntimeError(
                f"CUDA allocator peak measurement requires native, found {allocator}"
            )
        with ExitStack() as resources:
            nvml.nvmlInit()
            resources.callback(nvml.nvmlShutdown)
            handle = nvml.nvmlDeviceGetHandleByUUID(uuid)
            found = nvml.nvmlDeviceGetUUID(handle)
            if (found.decode("ascii") if isinstance(found, bytes) else found) != uuid:
                raise RuntimeError(f"NVML UUID {found!r} differs from CUDA UUID {uuid}")
            lock, stop = Lock(), Event()

            def sample(kind):
                with lock:
                    if len(samples) >= 100_000:
                        raise RuntimeError("NVML sample capacity exceeded")
                    reading = {"kind": kind, "start_ns": perf_counter_ns()}
                    samples.append(reading)
                    try:
                        value = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        if not 0 <= value <= 100:
                            raise ValueError(f"Invalid NVML utilization: {value}")
                        reading["gpu_percent"] = value
                    except Exception as error:
                        reading["error"] = str(error)
                        raise
                    finally:
                        reading["end_ns"] = perf_counter_ns()

            def poll():
                while not stop.wait(data["sample_interval_ns"] / 1e9):
                    sample("periodic")

            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            data.update(
                reset_allocated_bytes=torch.cuda.max_memory_allocated(device),
                reset_reserved_bytes=torch.cuda.max_memory_reserved(device),
                start_ns=perf_counter_ns(),
            )
            sample("start")
            window = _Window(device, data, sample)
            pool = resources.enter_context(ThreadPoolExecutor(max_workers=1))
            polling = pool.submit(poll)
            resources.callback(stop.set)
            yield window
            sample("dispatch_end")
            torch.cuda.synchronize(device)
            data["end_ns"] = perf_counter_ns()
            stop.set()
            polling.result()
            data.update(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
            for (begin, end), transfer in zip(window.events, data["transfers"], strict=True):
                if not begin.query() or not end.query():
                    raise RuntimeError("Unresolved CUDA transfer event after epoch synchronization")
                transfer["event_ms"] = begin.elapsed_time(end)
        valid = [
            s for s in samples if data["start_ns"] <= s["start_ns"] <= s["end_ns"] <= data["end_ns"]
        ]
        steady = data["steady_start_ns"]
        data.update(
            in_window_samples=len(valid),
            warmup_samples=sum(s["end_ns"] <= steady for s in valid),
            steady_samples=sum(s["start_ns"] >= steady for s in valid),
            utilization_percent=mean(s["gpu_percent"] for s in valid),
            leading_gap_ns=valid[0]["start_ns"] - data["start_ns"],
            trailing_gap_ns=data["end_ns"] - valid[-1]["end_ns"],
            max_gap_ns=max((b["start_ns"] - a["end_ns"] for a, b in pairwise(valid)), default=0),
            complete=True,
        )
    except BaseException as error:
        data["error"] = f"{type(error).__name__}: {error}"
        raise
