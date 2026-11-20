# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-volume elapsed-time diagnostics for correctness campaigns."""

import time
from contextlib import contextmanager


class StageTiming:
    """Accumulate explicit spans without per-batch log records or profiler hooks."""

    def __init__(self, report, persist=None, *, checkpoint=False):
        self.report = report
        self.persist = persist
        self.checkpoint = checkpoint
        self.stages = report.setdefault("stage_timings", {})

    @contextmanager
    def span(self, name, *, checkpoint=None):
        # Callers use non-overlapping spans; total campaign time remains separate.
        row = self.stages.setdefault(name, {"seconds": 0.0, "calls": 0, "failures": 0})
        checkpoint = self.checkpoint if checkpoint is None else checkpoint
        if checkpoint:
            self.report["active_timing_stage"] = name
            if self.persist:
                self.persist()
        started = time.perf_counter()
        try:
            yield row
        except BaseException:
            row["failures"] += 1
            raise
        finally:
            row["seconds"] += time.perf_counter() - started
            row["calls"] += 1
            if checkpoint:
                self.report.pop("active_timing_stage", None)
                if self.persist:
                    self.persist()
