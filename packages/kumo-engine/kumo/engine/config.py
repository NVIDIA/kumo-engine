# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-wide scratch and compute configuration via ``init``.

``configure`` remains available for shared ``kumo.settings`` scratch configuration.
"""

from pathlib import Path

from kumo.engine.cluster import ArtifactRoot, ComputeCluster, ComputeClusterConfig
from kumo.engine.connector.base import Connector

_cluster: ComputeCluster | None = None


def init(
    *,
    scratch_dir: str | Path | None = None,
    cluster_config: ComputeClusterConfig | None = None,
    artifact_root: ArtifactRoot | str | tuple[Connector, str] | None = None,
) -> None:
    """Configure scratch space and the process-wide compute cluster.

    Call before creating snapshots or trainers. Reinitializing releases the old
    workers and invalidates their remote handles, preserving cached files and Ray.
    Omitting cluster_config selects local execution. Scratch overrides remain
    applied if worker startup fails.

    Args:
        scratch_dir: Local driver scratch path; None keeps the current path.
        cluster_config: Remote worker placement, or None for local execution.
        artifact_root: Durable storage shared by workers. Accepts an ArtifactRoot,
            a URI/path, or a (connector, path) pair with connector authentication.
            Required for remote execution; independent of MLOps configuration.

    Raises:
        ValueError: If scratch_dir is remote or cluster/artifact settings conflict.
        RuntimeError: If Ray is uninitialized or worker placement fails.
        PermissionError: If the artifact root is not writable.
    """
    if cluster_config is not None and artifact_root is None:
        raise ValueError("ComputeCluster requires artifact_root in kumo.init")
    if cluster_config is None and artifact_root is not None:
        raise ValueError("artifact_root requires cluster_config")
    if isinstance(artifact_root, str):
        artifact_root = ArtifactRoot(artifact_root)
    elif isinstance(artifact_root, tuple):
        connector, location = artifact_root
        artifact_root = ArtifactRoot(location, connector)
    candidate = (
        ComputeCluster(cluster_config, artifact_root)
        if cluster_config is not None and artifact_root is not None
        else None
    )
    configure(scratch_dir=scratch_dir)
    global _cluster
    if _cluster is not None:
        _cluster.close()
    _cluster = None
    if candidate is not None:
        candidate.start()
    _cluster = candidate


def _get_compute_cluster() -> ComputeCluster | None:
    return _cluster

import kumo.settings


def configure(
    *,
    scratch_dir: str | Path | None = None,
) -> None:
    r"""Sets process-wide configuration for this package. Same as
    `kumo.settings.configure`.

    Args:
        scratch_dir: where `Graph.snapshot()`'s DiskGraph ingest and
            `Trainer.fit`/`predict`'s training/prediction runs read and
            write their working data -- e.g. a local NVMe SSD mount, for
            faster ingest/training than the default `~/.cache/kumo-engine`
            (typically the boot disk). Must be a local filesystem path --
            DiskGraph and GNN training read and write it directly, so a
            remote URI (an `s3://`/`gs://` path, a Snowflake stage) isn't
            accepted here. `None` leaves the current setting unchanged.

    Raises:
        ValueError: if `scratch_dir` isn't a local filesystem path.
    """
    kumo.settings.configure(scratch_dir=scratch_dir)
