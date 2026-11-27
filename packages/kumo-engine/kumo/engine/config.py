# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry points and Engine's live compute-pool lifecycle."""

import os

import kumo.settings
from kumo.engine.cluster import ComputeCluster
from kumo.settings import ArtifactRoot, ComputeClusterConfig, _Unchanged

_cluster: ComputeCluster | None = None


def configure(
    *,
    scratch_dir: str | os.PathLike[str] | None = None,
    cluster_config: ComputeClusterConfig | None | _Unchanged = _Unchanged.CLUSTER,
    artifact_root: ArtifactRoot | str | tuple[object, str] | None = None,
) -> None:
    """Forward configuration to ``kumo.settings.configure``.

    Args:
        scratch_dir: Local working directory; None keeps the current setting.
        cluster_config: Worker placement. Omit to preserve the pool; None closes it.
        artifact_root: Shared storage as an ArtifactRoot, URI/path, or
            (connector, path) pair; required when selecting a cluster.

    Returns:
        None.

    Raises:
        ValueError: If scratch is remote or cluster/storage settings conflict.
        TypeError: If compute settings have unsupported types.
        RuntimeError: If Ray is uninitialized or worker placement fails.
        PermissionError: If shared storage is not writable.
    """
    kumo.settings.configure(
        scratch_dir=scratch_dir,
        cluster_config=cluster_config,
        artifact_root=artifact_root,
    )


def init(
    *,
    scratch_dir: str | os.PathLike[str] | None = None,
    cluster_config: ComputeClusterConfig | None = None,
    artifact_root: ArtifactRoot | str | tuple[object, str] | None = None,
) -> None:
    """Forward initialization to settings, selecting local execution by default.

    Args:
        scratch_dir: Local working directory; None keeps the current setting.
        cluster_config: Worker placement, or None to release the current pool.
        artifact_root: Shared storage; required when selecting a cluster.

    Returns:
        None.

    Raises:
        ValueError: If scratch is remote or cluster/storage settings conflict.
        TypeError: If compute settings have unsupported types.
        RuntimeError: If Ray is uninitialized or worker placement fails.
        PermissionError: If shared storage is not writable.
    """
    configure(
        scratch_dir=scratch_dir,
        cluster_config=cluster_config,
        artifact_root=artifact_root,
    )


def _configure_cluster(
    cluster_config: ComputeClusterConfig | None, artifact_root: ArtifactRoot | None
) -> None:
    """Replace Engine's worker pool after shared settings have been validated."""
    candidate = (
        ComputeCluster(cluster_config, artifact_root)
        if cluster_config is not None and artifact_root is not None
        else None
    )
    global _cluster
    if _cluster is not None:
        _cluster.close()
    _cluster = None
    if candidate is not None:
        candidate.start()
    _cluster = candidate


def _get_compute_cluster() -> ComputeCluster | None:
    return _cluster
