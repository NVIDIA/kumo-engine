# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monotonic workload intervals and optional physical-I/O trace markers."""

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from native_io_resources import cgroup_path

from io_utils import fsutil

PHASE_IDS = {'fit': 1, 'training': 2, 'predict': 3, 'repeat': 4, 'reload': 5,
             'e2e': 6}
MARKER_ORDER = [(6, 1), (1, 1), (2, 1), (2, 0), (1, 0), (3, 1), (3, 0),
                (6, 0), (4, 1), (4, 0), (5, 1), (5, 0)]


def parse_device_io(text: str) -> dict[str, dict[str, int]]:
    devices = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] in devices:
            raise ValueError('Duplicate block device in io.stat')
        devices[fields[0]] = {key: int(value) for key, value in
                              (item.split('=') for item in fields[1:])}
    return devices


@dataclass(frozen=True)
class DeviceSnapshot:
    before_ns: int
    after_ns: int
    raw: str
    devices: dict[str, dict[str, int]]


class PhaseRecorder:
    def __init__(self, marker: Callable[[int, bool], dict[str, int]] | None = None,
                 io_path: str | None = None):
        self.marker = marker
        self.io_path = io_path
        self.intervals: dict[str, dict[str, int]] = {}
        self.markers: list[dict[str, int]] = []
        self.device_io: dict[str, dict[str, DeviceSnapshot]] = {}

    @classmethod
    def from_environment(cls) -> 'PhaseRecorder':
        enabled = os.environ.get('DG_IOPS_PHASE_MARKERS', '0')
        if enabled not in {'0', '1'}:
            raise ValueError('DG_IOPS_PHASE_MARKERS must be 0 or 1')
        if enabled == '0':
            return cls()
        from native_io_trace import mark_phase, prepare_marker
        prepare_marker()
        io_path = cgroup_path() + '/io.stat'
        if not fsutil.exists(io_path):
            raise RuntimeError('IOPS markers require actual cgroup I/O accounting')
        return cls(mark_phase, io_path)

    def snapshot(self, phase: str, position: str) -> None:
        if self.io_path is None:
            return
        before_ns = time.perf_counter_ns()
        raw = fsutil.read_text(self.io_path)
        after_ns = time.perf_counter_ns()
        self.device_io.setdefault(phase, {})[position] = DeviceSnapshot(
            before_ns, after_ns, raw, parse_device_io(raw))

    def begin(self, phase: str) -> int:
        if phase not in PHASE_IDS or phase in self.intervals:
            raise ValueError(f'Unexpected or repeated phase: {phase}')
        if phase == 'training' and 'end_ns' in self.intervals.get('fit', {'end_ns': 0}):
            raise ValueError('Training must start inside fit')
        self.snapshot(phase, 'before')
        if self.marker is not None:
            self.markers.append(self.marker(PHASE_IDS[phase], True))
        started = time.perf_counter_ns()
        self.intervals[phase] = {'start_ns': started}
        return started

    def end(self, phase: str) -> int:
        if phase not in self.intervals or 'end_ns' in self.intervals[phase]:
            raise ValueError(f'Phase is not active: {phase}')
        ended = time.perf_counter_ns()
        self.intervals[phase]['end_ns'] = ended
        if self.marker is not None:
            self.markers.append(self.marker(PHASE_IDS[phase], False))
        self.snapshot(phase, 'after')
        return ended

    def validate(self) -> None:
        if set(self.intervals) != set(PHASE_IDS):
            raise ValueError('Incomplete workload phase intervals')
        for interval in self.intervals.values():
            if interval.get('end_ns', 0) <= interval['start_ns']:
                raise ValueError('Incomplete or nonpositive workload phase')
        edges = sorted((moment, PHASE_IDS[phase], int(edge == 'start_ns'))
                       for phase, interval in self.intervals.items()
                       for edge, moment in interval.items())
        if [(phase, entering) for _, phase, entering in edges] != MARKER_ORDER:
            raise ValueError('Unexpected workload phase nesting')
        if self.marker is not None:
            order = [(item['phase_id'], item['entering']) for item in self.markers]
            if order != MARKER_ORDER:
                raise ValueError('Unexpected physical-I/O marker sequence')
