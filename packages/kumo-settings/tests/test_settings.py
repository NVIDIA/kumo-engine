# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import Mock

import pytest

import kumo.settings


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.setattr(kumo.settings, "_configured", None)
    monkeypatch.setattr(kumo.settings, "_DEFAULT_SCRATCH_DIR", tmp_path / "default")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def test_configure_sets_the_scratch_dir_and_creates_nothing(tmp_path):
    assert kumo.settings.scratch_dir() == tmp_path / "default"
    kumo.settings.configure(scratch_dir=Path("~/configured"))
    assert kumo.settings.scratch_dir() == tmp_path / "home" / "configured"
    kumo.settings.configure(scratch_dir=None)
    assert kumo.settings.scratch_dir() == tmp_path / "home" / "configured"
    assert not (tmp_path / "home").exists()


@pytest.mark.parametrize("path", ["s3://bucket/scratch", "@stage/scratch"])
def test_remote_paths_are_rejected(path):
    with pytest.raises(ValueError, match="scratch_dir must be a local filesystem path"):
        kumo.settings.configure(scratch_dir=path)


def test_compute_cluster_defaults() -> None:
    """Placement settings default to all eligible nodes and their full CPUs."""
    placement = kumo.settings.ComputeClusterConfig()
    assert placement.num_workers is None
    assert placement.cpus_per_worker is None
    assert placement.resources is None
    assert placement.worker_root == "/tmp/kumo-compute"


@pytest.mark.parametrize(
    "workers,cpus",
    [
        (0, None),
        (-1, 1),
        (True, 1),
        (1, 0),
        (1, -1),
        (1, float("inf")),
        (1, float("nan")),
    ],
)
def test_invalid_compute_capacity(workers: int | None, cpus: float | None) -> None:
    """Invalid capacity fails before a compute runtime is needed."""
    with pytest.raises(ValueError, match="positive"):
        kumo.settings.ComputeClusterConfig(num_workers=workers, cpus_per_worker=cpus)


@pytest.mark.parametrize("root", ["relative", "s3://bucket/workers"])
def test_compute_workers_require_local_absolute_paths(root: str) -> None:
    """Worker scratch placement cannot use a relative path or object storage."""
    with pytest.raises(ValueError, match="absolute worker-local"):
        kumo.settings.ComputeClusterConfig(worker_root=root)


@pytest.fixture
def compute_runtime(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace the optional Engine runtime without installing its dependencies."""
    runtime = ModuleType("kumo.engine.config")
    configure_cluster = Mock()
    setattr(runtime, "_configure_cluster", configure_cluster)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    return configure_cluster


@pytest.mark.parametrize("root", ["artifacts", kumo.settings.ArtifactRoot("artifacts")])
def test_configure_normalizes_cluster_storage(
    root: str | kumo.settings.ArtifactRoot,
    tmp_path: Path,
    compute_runtime: Mock,
) -> None:
    """Settings normalize storage and apply scratch before runtime startup."""
    placement = kumo.settings.ComputeClusterConfig()
    kumo.settings.configure(
        scratch_dir=tmp_path, cluster_config=placement, artifact_root=root
    )
    compute_runtime.assert_called_once_with(
        placement, kumo.settings.ArtifactRoot("artifacts")
    )
    assert kumo.settings.scratch_dir() == tmp_path


def test_configure_preserves_connector_authentication(compute_runtime: Mock) -> None:
    """Connector objects reach the runtime unchanged without settings inspecting them."""
    connector = object()
    placement = kumo.settings.ComputeClusterConfig()
    kumo.settings.configure(
        cluster_config=placement, artifact_root=(connector, "artifacts")
    )
    compute_runtime.assert_called_once_with(
        placement, kumo.settings.ArtifactRoot("artifacts", connector)
    )


def test_omitted_cluster_configuration_preserves_the_pool(
    tmp_path: Path, compute_runtime: Mock
) -> None:
    """Scratch-only and empty calls preserve workers; explicit None releases them."""
    placement = kumo.settings.ComputeClusterConfig()
    kumo.settings.configure(cluster_config=placement, artifact_root="artifacts")
    kumo.settings.configure(scratch_dir=tmp_path)
    kumo.settings.configure()
    compute_runtime.assert_called_once_with(
        placement, kumo.settings.ArtifactRoot("artifacts")
    )
    kumo.settings.configure(cluster_config=None)
    assert compute_runtime.call_count == 2
    compute_runtime.assert_called_with(None, None)


@pytest.mark.parametrize(
    "placement,root,error",
    [
        (kumo.settings.ComputeClusterConfig(), None, ValueError),
        (None, "artifacts", ValueError),
        (cast(kumo.settings.ComputeClusterConfig, object()), "artifacts", TypeError),
        (
            kumo.settings.ComputeClusterConfig(),
            cast(kumo.settings.ArtifactRoot, object()),
            TypeError,
        ),
    ],
)
def test_invalid_cluster_settings_preserve_existing_state(
    placement: kumo.settings.ComputeClusterConfig | None,
    root: str | kumo.settings.ArtifactRoot | None,
    error: type[Exception],
    tmp_path: Path,
    compute_runtime: Mock,
) -> None:
    """An invalid storage/placement pair cannot change scratch or touch workers."""
    before = kumo.settings.scratch_dir()
    with pytest.raises(error):
        kumo.settings.configure(
            scratch_dir=tmp_path, cluster_config=placement, artifact_root=root
        )
    assert kumo.settings.scratch_dir() == before
    compute_runtime.assert_not_called()


def test_artifact_root_requires_explicit_cluster_configuration(
    compute_runtime: Mock,
) -> None:
    """Storage alone cannot silently reconfigure an existing pool."""
    with pytest.raises(ValueError, match="requires cluster_config"):
        kumo.settings.configure(artifact_root="artifacts")
    compute_runtime.assert_not_called()


def test_local_configuration_needs_no_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Selecting local execution without an Engine runtime keeps settings lightweight."""
    monkeypatch.delitem(sys.modules, "kumo.engine.config", raising=False)
    kumo.settings.configure(scratch_dir=tmp_path, cluster_config=None)
    assert "kumo.engine.config" not in sys.modules
    assert kumo.settings.scratch_dir() == tmp_path
