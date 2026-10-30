# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""How long building a snapshot or a training table took and what it
wrote, shown on the runs that use it."""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from kumo_core.summaries import Table

from kumo.engine._manifest import BuildCost

PREP_TABLE = 'data_preparation'

Start = Tuple[datetime, float]

#: Ids of the artifacts this process built.
_built_here: Set[str] = set()


def start() -> Start:
    return datetime.now(timezone.utc), time.monotonic()


def build_cost(id: str, started: Start, disk_bytes: int) -> BuildCost:
    r"""The cost of building artifact ``id``, ``started`` at `start`, which
    wrote ``disk_bytes``."""
    at, clock = started
    _built_here.add(id)
    return BuildCost(
        started_at=at.isoformat(timespec='seconds'),
        finished_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        seconds=time.monotonic() - clock, disk_bytes=disk_bytes)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob('*')
               if item.is_file())


def prep_summaries(*manifests: Any) -> List[Table]:
    r"""A brief table of what building each of ``manifests`` (a snapshot's,
    a training table's) cost, for those that recorded it, and whether an
    earlier process built it."""
    rows = tuple((manifest.kind, manifest.build.seconds,
                  manifest.build.disk_bytes, manifest.id not in _built_here)
                 for manifest in manifests
                 if getattr(manifest, 'build', None) is not None)
    return ([Table(PREP_TABLE, ('step', 'seconds', 'bytes', 'reused'), rows,
                   brief=True)] if rows else [])


def prep_params(table: Table) -> Dict[str, str]:
    r"""``table``, a `prep_summaries` table, as run params
    ``prep.<step>.seconds`` and ``prep.<step>.bytes``."""
    return {f'prep.{step}.{name}': str(value)
            for step, seconds, size, _ in table.rows
            for name, value in (('seconds', seconds), ('bytes', size))}
