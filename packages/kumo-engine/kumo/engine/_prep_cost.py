# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""What building a snapshot or training table cost."""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Set, Tuple

from kumo_core.summaries import Summary, Table, Tag

from kumo.engine._manifest import BuildCost

PREP_TABLE = 'data_preparation'

Start = Tuple[datetime, float]

_built_here: Set[str] = set()


def start() -> Start:
    return datetime.now(timezone.utc), time.monotonic()


def build_cost(id: str, started: Start, disk_bytes: int) -> BuildCost:
    at, clock = started
    _built_here.add(id)
    return BuildCost(
        started_at=at.isoformat(timespec='seconds'),
        finished_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        seconds=time.monotonic() - clock, disk_bytes=disk_bytes)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob('*')
               if item.is_file())


def prep_summaries(*manifests: Any) -> List[Summary]:
    r"""What building each of ``manifests`` cost, as a brief table and tags."""
    rows = tuple((manifest.kind, manifest.build.seconds,
                  manifest.build.disk_bytes, manifest.id not in _built_here)
                 for manifest in manifests
                 if getattr(manifest, 'build', None) is not None)
    if not rows:
        return []
    return [Table(PREP_TABLE, ('step', 'seconds', 'bytes', 'reused'), rows,
                  brief=True),
            *(Tag(f'prep.{step}.{name}', str(value))
              for step, seconds, size, _ in rows
              for name, value in (('seconds', seconds), ('bytes', size)))]
