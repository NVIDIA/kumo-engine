# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource measurements for the manual DiskGraph benchmark."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from threading import Event

from e2e.datasets.throughput import pin_path


@contextmanager
def _poll(interval, sample):
    stop = Event()

    def poll():
        while not stop.wait(interval):
            sample()

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(poll)
        try:
            yield
        finally:
            stop.set()
        worker.result()


@contextmanager
def training_gpu_window(device):
    """Sample the selected device's driver-reported utilization during timed training."""
    metrics = {}
    if device.type != "cuda":
        yield metrics
        return
    import pynvml as nvml
    import torch

    raw = torch.cuda.get_device_properties(device).uuid
    uuid = raw.decode("ascii") if isinstance(raw, bytes) else str(raw)
    uuid = uuid if uuid.startswith(("GPU-", "MIG-")) else "GPU-" + uuid
    count = total = 0
    nvml.nvmlInit()
    try:
        handle = nvml.nvmlDeviceGetHandleByUUID(uuid)

        def sample():
            nonlocal count, total
            value = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
            if not 0 <= value <= 100:
                raise ValueError(f"Invalid NVML utilization: {value}")
            count += 1
            total += value

        with _poll(0.1, sample):
            yield metrics
        metrics.update(
            device_uuid=uuid,
            samples=count,
            interval_ms=100,
            mean_percent=total / count if count else None,
        )
    finally:
        nvml.nvmlShutdown()


def _allocated_bytes(paths):
    seen, size = set(), 0
    for path in paths:
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue  # Index construction can rename or remove a temporary file between samples.
        key = info.st_dev, info.st_ino
        if key not in seen:
            seen.add(key)
            size += info.st_blocks * 512
    return size


def _process_io():
    path = Path("/proc/self/io")
    if not path.exists():
        return {}
    counters = dict(line.split(": ") for line in path.read_text().splitlines())
    return {name: int(counters[name]) for name in ("read_bytes", "write_bytes")}


@contextmanager
def workload_resources(workload, root, index):
    """Observe the trainer process and this workload's staged files and fresh index."""
    import resource

    pins = json.loads(pin_path(workload).read_text())["objects"]
    staged_files = tuple(root / pin["key"] for pin in pins.values())
    staged = _allocated_bytes(staged_files)
    metrics = dict(
        staged_bytes=staged, peak_sampled_scratch_bytes=staged, scratch_sample_interval_ms=1000
    )

    def sample():
        allocated = _allocated_bytes(chain(staged_files, (index,), index.rglob("*")))
        metrics["peak_sampled_scratch_bytes"] = max(
            metrics["peak_sampled_scratch_bytes"], allocated
        )

    before = _process_io()
    with _poll(1.0, sample):
        yield metrics
    sample()
    for name, value in _process_io().items():
        metrics["process_" + name] = value - before[name]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metrics["peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
