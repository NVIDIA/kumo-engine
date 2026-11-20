# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream observations without retaining batches or constructing rate estimates."""

import json
import math
import time
from contextlib import ExitStack, contextmanager

from .metrics import EXTRAS, METRICS


class Recorder:
    def __init__(self, stream, *, clock=time.monotonic_ns, observers=()):
        self.stream = stream
        self.clock = clock
        self.scope = {}
        self.parents = []
        self.serial = 0
        self.span_serial = 0
        self.pending = []
        self.deferral = 0
        self.observers = tuple(observers)

    def write(self, kind, **fields):
        self.serial += 1
        record = dict(id=self.serial, kind=kind, scope=dict(self.scope), **fields)
        if self.deferral:
            if len(self.pending) >= 10000:
                raise RuntimeError(
                    "One request exceeded the deferred observation limit (10000)"
                )
            self.pending.append(record)
        else:
            self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        return record

    @contextmanager
    def defer(self):
        """Keep encoding/writing outside the individually timed loader call."""
        self.deferral += 1
        try:
            yield
        finally:
            self.deferral -= 1
            if not self.deferral:
                records, self.pending = self.pending, []
                for record in records:
                    self.stream.write(json.dumps(record, allow_nan=False) + "\n")

    def measure(self, name, value=None, *, status="available", reason=None, **details):
        metric = METRICS.get(name) or EXTRAS.get(name)
        if metric is None:
            raise ValueError(f"Unknown metric: {name}")
        if status not in {"available", "unavailable", "error"}:
            raise ValueError(f"Invalid measurement status: {status}")
        if status == "available":
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} requires a finite nonnegative number")
            if metric.unit in {"bytes", "count", "rows"} and type(value) is not int:
                raise ValueError(f"{name} requires an integer")
            if metric.unit == "percent" and value > 100:
                raise ValueError(f"{name} exceeds 100 percent")
        elif value is not None or not reason:
            raise ValueError(
                "Missing measurements require a reason and no numeric value"
            )
        return self.write(
            "measurement",
            metric=name,
            unit=metric.unit,
            value=value,
            status=status,
            reason=reason,
            details=details,
            timestamp_ns=self.clock(),
            parent=self.parents[-1] if self.parents else None,
        )

    @contextmanager
    def span(self, operation):
        with ExitStack() as stack:
            observations = []
            for observer in self.observers:
                observations.append(
                    stack.enter_context(observer.stage(operation, self))
                )
            details = {}
            try:
                with self._span(operation) as details:
                    yield details
            finally:
                for observation in observations:
                    if isinstance(observation, dict):
                        observation.update(details)

    @contextmanager
    def _span(self, operation):
        name = operation + ".elapsed_ns"
        if name not in METRICS and name not in EXTRAS:
            raise ValueError(f"Unknown operation: {operation}")
        self.span_serial += 1
        span_id = self.span_serial
        self.parents.append(span_id)
        details = {"outcome": "completed"}
        status, reason = "available", None
        start = self.clock()
        details["start_ns"] = start
        try:
            yield details
        except BaseException as error:
            details["outcome"] = "failed"
            status, reason = "error", f"{type(error).__name__}: {error}"
            raise
        finally:
            end = self.clock()
            self.parents.pop()
            details.update(start_ns=start, end_ns=end, elapsed_ns=end - start)
            self.measure(
                name,
                end - start if status == "available" else None,
                status=status,
                reason=reason,
                span_id=span_id,
                **details,
            )
