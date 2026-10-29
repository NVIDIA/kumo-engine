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

The root all of these nest under defaults to `~/.cache/kumo-engine`, but is
overridable via `kumo.engine.configure(scratch_dir=...)` (`kumo/engine/config.py`) --
e.g. to point `Graph.snapshot()`'s DiskGraph ingest and `Trainer`'s training
runs at a local NVMe SSD mount instead of the home directory, matching
`kumo_core.config.NativeRuntimeConfig`'s own `scratch_root`/`artifact_root`
split ("so adapters can map them to local NVMe and durable storage
respectively"). A process-wide, mutable module global, not threaded through
every constructor -- the same tradeoff the cloud SDK's own `Storage` object
makes, just set once via a function call instead of passed to a `Backend`
constructor (this package has no `Backend`). Like that object, `configure()`
is meant to be called once, early -- before constructing any `Graph`/
`Trainer`/etc. -- not toggled mid-session: nothing here re-reads it after a
path has already been computed and used.
"""

import hashlib
import uuid
from pathlib import Path
from typing import Union

_DEFAULT_ROOT = Path.home() / ".cache" / "kumo-engine"
_root = _DEFAULT_ROOT


def set_scratch_dir(path: Union[str, Path]) -> None:
    r"""Overrides the scratch-directory root every helper below nests under.
    Backs `kumo.engine.configure(scratch_dir=...)`.

    Args:
        path: a local filesystem path -- DiskGraph ingest and GNN training
            read and write it directly, the same constraint the cloud SDK's
            `Storage.scratch` enforces on its own scratch path.

    Raises:
        ValueError: if `path` isn't a local filesystem path (e.g. a
            Snowflake stage URI or an `s3://`/`gs://` URI).
    """
    text = str(path)
    if text.startswith('@') or '://' in text:
        raise ValueError(
            f"scratch_dir must be a local filesystem path, not {text!r} -- "
            "DiskGraph ingest and GNN training read and write it directly.")
    global _root
    _root = Path(path).expanduser()


def get_scratch_dir() -> Path:
    return _root


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
