# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure total and anonymous cgroup charges for the loader worker and descendants."""

import os
from contextlib import contextmanager, nullcontext
from itertools import pairwise
from pathlib import Path
from threading import Event, Thread
from time import perf_counter_ns

from tests.e2e.memory_worker import validate_membership

SAMPLE_INTERVAL_NS = 10_000_000
MAX_SAMPLES = 100_000


def _byte_count(value):
    count = int(value)
    if count < 0:
        raise ValueError(f"Negative cgroup memory counter: {count}")
    return count


@contextmanager
def memory_interval(record, operation, *, sample_interval_ns=SAMPLE_INTERVAL_NS):
    """Measure an operation when the supervisor has selected a host-memory mode."""
    method = os.getenv("KUMO_PYTORCH_HOST_MEMORY")
    if not method:
        yield
        return
    if method not in ("kernel", "sampled"):
        raise ValueError(f"Unknown host memory method: {method}")
    path = Path(os.environ["KUMO_PYTORCH_MEMORY_CGROUP"])
    stage = record.setdefault(
        "host_memory",
        {
            "method": method,
            "cgroup": str(path),
            "sample_interval_ns": sample_interval_ns,
            "complete": True,
            "intervals": [],
        },
    )
    samples = []
    interval = {"operation": operation, "start_ns": perf_counter_ns(), "samples": samples}
    stage["intervals"].append(interval)
    stop = Event()
    poll_errors = []

    def sample(kind):
        if len(samples) >= MAX_SAMPLES:
            raise RuntimeError("Host memory sample capacity exceeded")
        start = perf_counter_ns()
        fields = dict(line.split() for line in (path / "memory.stat").read_text().splitlines())
        values = {"anonymous_bytes": _byte_count(fields["anon"])}
        if method == "sampled":
            values["current_bytes"] = _byte_count((path / "memory.current").read_text())
        samples.append({"kind": kind, "start_ns": start, "end_ns": perf_counter_ns(), **values})

    def poll():
        try:
            while not stop.wait(sample_interval_ns / 1e9):
                sample("periodic")
        except Exception as error:
            poll_errors.append(error)

    try:
        validate_membership(path)
        peak_file = path / "memory.peak"
        handle = peak_file.open("r+b", buffering=0) if method == "kernel" else nullcontext()
        with handle as peak:
            if peak is not None:
                peak.write(b"0")
            sample("start")
            thread = Thread(target=poll)
            thread.start()
            try:
                interval["operation_start_ns"] = perf_counter_ns()
                yield
            finally:
                interval["operation_end_ns"] = perf_counter_ns()
                stop.set()
                thread.join()
            if poll_errors:
                raise poll_errors[0]
            sample("end")
            if peak is None:
                total = max(s["current_bytes"] for s in samples)
            else:
                peak.seek(0)
                total = _byte_count(peak.read())
            validate_membership(path)
        interval["peak_host_bytes"] = total
        interval["peak_anonymous_bytes"] = max(s["anonymous_bytes"] for s in samples)
        if stage["complete"]:
            for key in ("peak_host_bytes", "peak_anonymous_bytes"):
                stage[key] = max(stage.get(key, 0), interval[key])
    except BaseException as error:
        interval["error"] = f"{type(error).__name__}: {error}"
        stage["complete"] = False
        stage.pop("peak_host_bytes", None)
        stage.pop("peak_anonymous_bytes", None)
        raise
    finally:
        interval.update(
            end_ns=perf_counter_ns(),
            sample_count=len(samples),
            max_gap_ns=max((b["start_ns"] - a["end_ns"] for a, b in pairwise(samples)), default=0),
        )
