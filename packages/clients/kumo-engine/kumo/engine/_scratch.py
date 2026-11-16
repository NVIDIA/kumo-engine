# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Scratch-directory naming and configuration.

There is no central registry of `Trainer`/`Graph` objects in this package,
so scratch-directory names must be collision-free on their own:

* Each :class:`~kumo.engine.trainer.Trainer` gets a random UUID at construction,
  used as its scratch-dir prefix.
* Each :class:`~kumo.engine.graph.Graph` computes a content-fingerprint id
  (`Graph.id`) from its tables' `(connector.id, table_name, definition)`
  and its edges -- the same approach the cloud pipeline uses for its
  `graph_key` -- used, together with a fingerprint of the tables' data
  versions, as the :class:`~kumo.engine.graph.GraphSnapshot` scratch-dir prefix
  (`Graph.snapshot()`'s own docstring). Two structurally identical graphs
  converge on the same `Graph.id`, which combined with an unchanged data
  version is what lets `Graph.snapshot()` reuse an existing build instead
  of re-ingesting.

The root all of these nest under is `kumo.settings.scratch_dir()`.
"""

import hashlib
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, Optional

from kumo_core.chunked_fs import set_spool_dir

import kumo.settings

# Tests pin the root by patching this.
_root: Optional[Path] = None
_scoped_root: ContextVar[Optional[Path]] = ContextVar('kumo_scratch_root',
                                                      default=None)


def get_scratch_dir() -> Path:
    return _scoped_root.get() or _root or kumo.settings.scratch_dir()


@contextmanager
def scoped_scratch_dir(path: Path) -> Iterator[None]:
    r"""Nests this context's new scratch directories under ``path``."""
    token = _scoped_root.set(path)
    try:
        yield
    finally:
        _scoped_root.reset(token)


set_spool_dir(lambda: str(get_scratch_dir() / "spool"))


def new_run_id() -> str:
    r"""A fresh, random id for one run -- a :class:`~kumo.engine.trainer.Trainer`
    instance, a `Trainer.predict` call, a `PredictiveQuery.
    generate_training_table` call, and so on.
    """
    return uuid.uuid4().hex


def fingerprint(*parts: bytes) -> str:
    r"""A stable content-fingerprint id from the given byte fragments. Used
    by `Connector.id` and `Graph.id`.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def trainer_scratch_dir(trainer_id: str) -> Path:
    return get_scratch_dir() / "trainers" / trainer_id


def snapshot_scratch_dir(graph_key: str) -> Path:
    r"""Scratch space for one `Graph`, keyed by `Graph.id` alone (not the
    data version) -- used only for staging raw source-table reads before
    ingest (`Graph.snapshot()`'s `persist_dir`). The materialized index
    itself is keyed by `(graph_key, data_version)` via `kumo_core.diskgraph.
    snapshot.LocalSnapshotStore`, nested under this same root.
    """
    return get_scratch_dir() / "snapshots" / graph_key


def train_table_scratch_dir(run_id: str) -> Path:
    return get_scratch_dir() / "train_tables" / run_id


def prediction_table_scratch_dir(run_id: str) -> Path:
    return get_scratch_dir() / "prediction_tables" / run_id
