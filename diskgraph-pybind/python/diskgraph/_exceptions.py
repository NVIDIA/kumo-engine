"""Exception classes used by the task-dataset rework modules.

Re-exports the Rust-backed `DiskgraphDataError` / `DiskgraphConfigError`
/ `DiskgraphSystemError` triple from `._c` so the new task-dataset
modules raise the *same* class objects callers already catch elsewhere
in the codebase.  Adds `DiskgraphTaskError` as a Python sibling for
task-runtime failures (dataset reopen, runtime open, batch iteration);
this can be promoted to a PyO3 exception alongside the others when the
task-runtime surface lands in Rust.
"""
from __future__ import annotations

from ._c import (
    DiskgraphConfigError,
    DiskgraphDataError,
    DiskgraphSystemError,
)


class DiskgraphTaskError(Exception):
    """Raised for task-runtime failures: dataset reopen, runtime open,
    input lookup, batch iteration.  Distinct from the data / config /
    system triad so callers can predict-catch task-runtime failures
    without intercepting unrelated errors."""


# Backward-compat alias. Legacy kumo callers (e.g. pool/activities.py
# snapshot residency code) import ``DiskgraphCorruptionError`` from this
# module. The TT-NG rearchitecture folded corruption-detection failures
# into ``DiskgraphDataError``; aliasing here keeps single-binary ABI
# compatibility for kumo branches that are dual-pathing across wheels.
DiskgraphCorruptionError = DiskgraphDataError


__all__ = [
    "DiskgraphConfigError",
    "DiskgraphDataError",
    "DiskgraphSystemError",
    "DiskgraphTaskError",
    "DiskgraphCorruptionError",
]
