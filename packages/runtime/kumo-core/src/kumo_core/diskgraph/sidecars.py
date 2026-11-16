# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable identity and path rules for DiskGraph job sidecars."""

from __future__ import annotations

import os
import re

_SNAPSHOT_ID = re.compile(r"^graph-([a-f0-9]{32})@(.+)$")


def canonical_graph_snapshot_id(snapshot_id: str) -> str:
    if not isinstance(snapshot_id,
                      str) or _SNAPSHOT_ID.fullmatch(snapshot_id) is None:
        raise ValueError(
            "snapshot_id must already use canonical graph-<fpid>@<version> "
            f"form; got {snapshot_id!r}")
    return snapshot_id


def graph_snapshot_parts(snapshot_id: str) -> tuple[str, str]:
    canonical = canonical_graph_snapshot_id(snapshot_id)
    match = _SNAPSHOT_ID.fullmatch(canonical)
    assert match is not None
    return match.group(1), match.group(2)


def job_sidecar_root(snapshot_id: str, data_persist_dir: str) -> str:
    graph_id, version = graph_snapshot_parts(snapshot_id)
    return os.path.join(data_persist_dir, "diskgraph_job_sidecars",
                        f"NA-{graph_id}", version)


def safe_relative_path(relpath: str) -> str:
    normalized = os.path.normpath(relpath)
    if (normalized in (".", "..") or normalized.startswith(f"..{os.sep}")
            or os.path.isabs(normalized)):
        raise ValueError(f"Unsafe relative path: {relpath!r}")
    return normalized
