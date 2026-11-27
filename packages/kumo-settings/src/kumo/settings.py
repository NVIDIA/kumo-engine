# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-wide settings shared by the Kumo packages."""

import math
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_DEFAULT_SCRATCH_DIR = Path.home() / ".cache" / "kumo-engine"
_configured: Path | None = None


class _Unchanged(Enum):
    """Distinguish omitted compute settings from an explicit return to local mode."""

    CLUSTER = "unchanged"


@dataclass(frozen=True)
class ComputeClusterConfig:
    """Placement and local storage settings for disk-equipped Ray workers.

    Args:
        num_workers: Distinct remote nodes to select; None selects all eligible nodes.
        worker_root: Persistent worker-local directory reused across sessions.
        resources: Custom Ray resources required by each worker.
        cpus_per_worker: Reserved CPUs; None reserves each node's full capacity.

    Raises:
        ValueError: If worker count, CPU capacity or worker_root is invalid.
    """

    num_workers: int | None = None
    worker_root: str = "/tmp/kumo-compute"
    resources: dict[str, float] | None = None
    cpus_per_worker: float | None = None

    def __post_init__(self) -> None:
        """Reject invalid worker capacity or a nonlocal scratch directory."""
        if (
            self.num_workers is not None
            and (type(self.num_workers) is not int or self.num_workers < 1)
        ) or (
            self.cpus_per_worker is not None
            and (not math.isfinite(self.cpus_per_worker) or self.cpus_per_worker <= 0)
        ):
            raise ValueError("Worker count and CPUs per worker must be positive")
        if not Path(self.worker_root).is_absolute() or "://" in self.worker_root:
            raise ValueError("worker_root must be an absolute worker-local directory")


@dataclass(frozen=True)
class ArtifactRoot:
    """Durable cluster storage with optional connector authentication.

    Args:
        location: Local/shared path or object-store URI; connector-relative when
            a connector is supplied.
        connector: File/object-store connector passed unchanged to the runtime.
    """

    location: str
    connector: object | None = None


def configure(
    *,
    scratch_dir: str | os.PathLike[str] | None = None,
    cluster_config: ComputeClusterConfig | None | _Unchanged = _Unchanged.CLUSTER,
    artifact_root: ArtifactRoot | str | tuple[object, str] | None = None,
) -> None:
    """Set scratch placement and optionally replace the process-wide worker pool.

    Scratch-only calls do not import Engine or Ray. Cluster setup lazily loads
    the Engine runtime; initialize Ray first. Replacement releases old workers
    and invalidates their handles, preserving cached files and Ray. An explicit
    scratch override remains applied if worker startup fails.

    Args:
        scratch_dir: Local working directory; None keeps the current setting.
        cluster_config: Worker placement. Omit to preserve the pool; None closes it
            and selects local execution.
        artifact_root: Shared storage, as an ArtifactRoot, URI/path, or
            (connector, path) pair. Required when selecting a cluster.

    Returns:
        None.

    Raises:
        ValueError: If scratch is remote or cluster/storage settings conflict.
        TypeError: If compute settings have unsupported types.
        ImportError: If cluster setup is requested without Engine installed.
        RuntimeError: If Ray is uninitialized or worker placement fails.
        PermissionError: If shared storage is not writable.
    """
    global _configured
    path = _configured
    if scratch_dir is not None:
        text = os.fspath(scratch_dir)
        if text.startswith("@") or "://" in text:
            raise ValueError(
                f"scratch_dir must be a local filesystem path, not {text!r} -- "
                "DiskGraph ingest and training read and write it directly."
            )
        path = Path(text).expanduser()
    if cluster_config is _Unchanged.CLUSTER:
        if artifact_root is not None:
            raise ValueError("artifact_root requires cluster_config")
    else:
        if cluster_config is not None and not isinstance(
            cluster_config, ComputeClusterConfig
        ):
            raise TypeError("cluster_config must be a ComputeClusterConfig or None")
        if cluster_config is not None and artifact_root is None:
            raise ValueError("ComputeCluster requires artifact_root")
        if cluster_config is None and artifact_root is not None:
            raise ValueError("artifact_root requires cluster_config")
        if isinstance(artifact_root, str):
            artifact_root = ArtifactRoot(artifact_root)
        elif isinstance(artifact_root, tuple):
            connector, location = artifact_root
            artifact_root = ArtifactRoot(location, connector)
        elif artifact_root is not None and not isinstance(artifact_root, ArtifactRoot):
            raise TypeError("artifact_root must be an ArtifactRoot, path, or pair")

    _configured = path
    if cluster_config is _Unchanged.CLUSTER:
        return
    if cluster_config is None and "kumo.engine.config" not in sys.modules:
        return
    from kumo.engine.config import _configure_cluster

    _configure_cluster(cluster_config, artifact_root)


def scratch_dir() -> Path:
    """Return the local directory used for ingest, training, and prediction.

    Returns:
        The configured scratch path, or the default cache directory.
    """
    return _configured or _DEFAULT_SCRATCH_DIR
