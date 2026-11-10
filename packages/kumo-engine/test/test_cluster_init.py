# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Global compute setup and head-node snapshot coordination."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

import kumo.engine as kumo
from kumo.engine import config
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.cluster import _ComputeWorker
from kumo.engine.graph import GraphSnapshot, _SnapshotSources


def test_init_and_return_to_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reinitializing closes the pool while retaining configure compatibility."""
    factory = Mock()
    monkeypatch.setattr(config, "ComputeCluster", factory)
    monkeypatch.setattr(config, "_cluster", None)
    previous = get_scratch_dir()
    try:
        placement = kumo.ComputeClusterConfig()
        kumo.init(
            scratch_dir=tmp_path,
            cluster_config=placement,
            artifact_root=str(tmp_path / "store"),
        )
        factory.assert_called_once_with(
            placement, kumo.ArtifactRoot(str(tmp_path / "store"))
        )
        factory.return_value.start.assert_called_once_with()
        assert get_scratch_dir() == tmp_path
        assert config._get_compute_cluster() is factory.return_value
        kumo.init()
        factory.return_value.close.assert_called_once_with()
        assert config._get_compute_cluster() is None
        kumo.configure(scratch_dir=tmp_path / "other")
        assert get_scratch_dir() == tmp_path / "other"
    finally:
        kumo.configure(scratch_dir=previous)


def test_init_failure_keeps_scratch_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed placement retains the override without exposing an incomplete pool."""
    factory = Mock()
    factory.return_value.start.side_effect = RuntimeError("placement failed")
    monkeypatch.setattr(config, "ComputeCluster", factory)
    monkeypatch.setattr(config, "_cluster", None)
    monkeypatch.setattr("kumo.engine._scratch._root", tmp_path / "before")
    with pytest.raises(RuntimeError, match="placement failed"):
        kumo.init(
            scratch_dir=tmp_path,
            cluster_config=kumo.ComputeClusterConfig(),
            artifact_root=str(tmp_path / "store"),
        )
    assert config._get_compute_cluster() is None
    assert get_scratch_dir() == tmp_path


@pytest.mark.parametrize(
    "arguments,error",
    [
        ({"cluster_config": kumo.ComputeClusterConfig()}, ValueError),
        ({"artifact_root": "/unused"}, ValueError),
        ({"scratch_dir": "s3://unused"}, ValueError),
    ],
)
def test_init_rejects_invalid_configuration(
    arguments: dict[str, object],
    error: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid setup fails before closing an existing pool."""
    previous = Mock()
    monkeypatch.setattr(config, "_cluster", previous)
    with pytest.raises(error):
        kumo.init(**arguments)
    previous.close.assert_not_called()


def test_init_connector_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Authenticated connectors are forwarded without introducing credential fields."""
    factory = Mock()
    monkeypatch.setattr(config, "ComputeCluster", factory)
    monkeypatch.setattr(config, "_cluster", None)
    connector = kumo.FileConnector(root_dir=str(tmp_path))
    placement = kumo.ComputeClusterConfig()
    kumo.init(cluster_config=placement, artifact_root=(connector, "artifacts"))
    factory.assert_called_once_with(
        placement, kumo.ArtifactRoot("artifacts", connector)
    )


@pytest.mark.parametrize(
    "scenario",
    [
        (True, [False, False], None),
        (False, [False, True], 1),
        (False, [True, False], 0),
        (False, [False, False], 0),
    ],
)
def test_snapshot_selects_producer_after_storage_and_cache_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: tuple[bool, list[bool], int | None],
) -> None:
    """Durable hits download; cache hits upload; a total miss produces exactly once."""
    import ray

    durable, cached, producer = scenario
    cluster = kumo.ComputeCluster(
        kumo.ComputeClusterConfig(), kumo.ArtifactRoot(str(tmp_path))
    )
    graph_key, version = "a" * 64, "b" * 64
    descriptor = GraphSnapshot(
        f"{graph_key}@{version[:16]}", Path("/cache") / version, {}
    )
    events: list[str] = []

    def _produce(*args: object) -> GraphSnapshot:
        events.append("upload")
        return descriptor

    def _download(*args: object) -> GraphSnapshot:
        events.append("download")
        return descriptor

    actors = [Mock(), Mock()]
    for index, actor in enumerate(actors):
        actor.stage_snapshot.remote.return_value = (descriptor, {})
        actor.has_snapshot.remote.return_value = cached[index]
        actor.produce_snapshot.remote.side_effect = _produce
        actor.download_snapshot.remote.side_effect = _download
    cluster._actors, cluster._active = actors, True
    monkeypatch.setattr(ray, "get", lambda value: value)
    monkeypatch.setattr("kumo.engine.cluster.random.choice", lambda peers: peers[0])
    read = Mock(
        return_value=("", {"snapshot_id": descriptor.id, "data_version": version})
    )
    if not durable:
        read.side_effect = FileNotFoundError("no export")
    monkeypatch.setattr("kumo.engine._snapshot_io.read", read)
    graph, options = Mock(), {"timestamp_precision": "seconds", "record_sources": True}
    result = cluster._graph_snapshot(graph, options)
    assert result.id == descriptor.id and result._compute is cluster
    read.assert_called_once()
    actors[0].stage_snapshot.remote.assert_called_once_with(graph, options)
    actors[1].stage_snapshot.remote.assert_not_called()
    for index, actor in enumerate(actors):
        if durable:
            actor.has_snapshot.remote.assert_not_called()
        else:
            actor.has_snapshot.remote.assert_called_once_with(descriptor)
        if producer == index:
            actor.produce_snapshot.remote.assert_called_once_with(
                (graph, options, descriptor), cluster._settings, {}
            )
        else:
            actor.produce_snapshot.remote.assert_not_called()
        if producer == index:
            actor.download_snapshot.remote.assert_not_called()
        else:
            actor.download_snapshot.remote.assert_called_once_with(
                descriptor, cluster._settings, {}
            )
    assert events == (["download", "download"] if durable else ["upload", "download"])
    actors[0].discard_staging.remote.assert_called_once_with(descriptor.id)


def test_snapshot_storage_error_stops_before_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Permission and corruption errors do not masquerade as cache misses."""
    import ray

    cluster = kumo.ComputeCluster(
        kumo.ComputeClusterConfig(), kumo.ArtifactRoot(str(tmp_path))
    )
    actor = Mock()
    descriptor = GraphSnapshot("id", Path("/cache/version"), {})
    actor.stage_snapshot.remote.return_value = (descriptor, {})
    actor.discard_staging.remote.side_effect = OSError("cleanup failed")
    cluster._actors, cluster._active = [actor], True
    monkeypatch.setattr(ray, "get", lambda value: value)
    monkeypatch.setattr(
        "kumo.engine._snapshot_io.read", Mock(side_effect=PermissionError("denied"))
    )
    with pytest.raises(PermissionError, match="denied"):
        cluster._graph_snapshot(Mock(), {})
    actor.produce_snapshot.remote.assert_not_called()
    actor.download_snapshot.remote.assert_not_called()
    actor.discard_staging.remote.assert_called_once_with(descriptor.id)
    assert cluster._failed
    assert "Could not clean snapshot source staging" in caplog.text


def test_cached_worker_skips_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached producer uploads its index without reading source rows again."""
    worker = object.__new__(_ComputeWorker)
    descriptor = GraphSnapshot("id", tmp_path / "index", {})
    graph = Mock()
    worker.snapshots = {}
    monkeypatch.setattr(worker, "_local_snapshot", Mock(return_value=descriptor))
    manifest = Mock()
    monkeypatch.setattr("kumo.engine.graph._write_snapshot_manifest", manifest)
    store = Mock()
    monkeypatch.setattr("kumo.engine._snapshot_io.store", store)
    assert worker.produce_snapshot((graph, {}, descriptor), Mock(), {}) == descriptor
    graph._snapshot_no_lock.assert_not_called()
    graph._stage_snapshot_sources.assert_not_called()
    store.assert_called_once()
    manifest.assert_called_once_with(descriptor, descriptor.id, {}, True, fresh=False)


def test_download_skips_existing_local_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker with the exact valid version never downloads it again."""
    worker = object.__new__(_ComputeWorker)
    descriptor = GraphSnapshot("id", tmp_path / "index", {})
    worker.snapshots = {}
    monkeypatch.setattr(worker, "_local_snapshot", Mock(return_value=descriptor))
    manifest = Mock()
    monkeypatch.setattr("kumo.engine.graph._write_snapshot_manifest", manifest)
    load = Mock()
    monkeypatch.setattr(GraphSnapshot, "load_from_connector", load)
    assert worker.download_snapshot(descriptor, Mock(), {}) == descriptor
    load.assert_not_called()
    manifest.assert_called_once_with(descriptor, descriptor.id, {}, True, fresh=False)
    assert worker.snapshots[descriptor.id] == descriptor


def test_source_version_identity_matches_local_semantics() -> None:
    """Schema, source version and clock changes cannot reuse a different index."""
    sources = _SnapshotSources({"users": "v1"}, {}, {})
    assert sources.data_version("seconds") == replace(sources).data_version("seconds")
    assert sources.data_version("seconds") != sources.data_version("milliseconds")
    assert sources.data_version("seconds") != _SnapshotSources(
        {"users": "v2"}, {}, {}
    ).data_version("seconds")


def test_remote_operations_use_initialized_storage(
    snapshot: GraphSnapshot,
    tmp_path: Path,
) -> None:
    """Remote work neither resolves MLOps contexts nor accepts storage overrides."""
    from kumoapi.model_plan import ModelPlan

    cluster = Mock(spec=kumo.ComputeCluster)
    snapshot = replace(snapshot, _compute=cluster)
    query = kumo.PredictiveQuery._from_snapshot(
        snapshot, "PREDICT users.label FOR EACH users.id"
    )
    trainer = kumo.Trainer(ModelPlan())
    settings = kumo.MLOps(str(tmp_path))
    with kumo.mlops(settings):
        table = query.generate_training_table(snapshot)
        trainer.prepare_artifacts(table, snapshot=snapshot)
    cluster._training_table_generate.assert_called_once_with(query, snapshot, None)
    assert len(cluster.prepare_training.call_args.args) == 3
    with pytest.raises(ValueError, match="kumo.init"):
        query.generate_training_table(snapshot, mlops=settings)
    with pytest.raises(ValueError, match="kumo.init"):
        trainer.fit(table, snapshot=snapshot, mlops=settings)


def test_existing_tables_keep_their_producer(
    snapshot: GraphSnapshot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later snapshot cache hit cannot send an older table to a different owner."""
    cluster = kumo.ComputeCluster(
        kumo.ComputeClusterConfig(), kumo.ArtifactRoot(str(tmp_path))
    )
    snapshot = replace(snapshot, _compute=cluster)
    first, second = Mock(), Mock()
    cluster._active = True
    cluster._actors = [first, second]
    cluster._snapshots[snapshot.id] = first
    from kumo_core.model_plan import TrainingTableGenerationPlan

    table = kumo.TrainingTable(
        "query",
        tmp_path,
        tmp_path,
        tmp_path,
        (),
        TrainingTableGenerationPlan(),
        id="table",
        _compute=cluster,
    )
    import ray

    from kumo.engine.cluster import _TrainingTablePublication

    first.describe_training_table.remote.return_value = _TrainingTablePublication(
        table, snapshot.id, str(tmp_path), None
    )
    first.has_training_table.remote.return_value = False
    second.has_training_table.remote.return_value = False
    monkeypatch.setattr(ray, "get", lambda value: value)
    call = Mock(return_value=table)
    monkeypatch.setattr(cluster, "_call", call)
    cluster._training_table_generate(Mock(query="query"), snapshot, None)
    cluster._snapshots[snapshot.id] = second
    cluster.prepare_training(table, snapshot, Mock())
    assert call.call_args.args[1] is first


def test_worker_restart_preserves_disk_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actor restarts on the same physical node reuse a persistent scratch root."""
    from types import SimpleNamespace

    import ray

    nodes = iter(("first-node", "second-node"))
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: next(nodes)),
    )
    scratch = Mock()
    monkeypatch.setattr("kumo.engine._scratch.set_scratch_dir", scratch)
    first = _ComputeWorker(str(tmp_path))
    marker = first.root / "cached-index"
    marker.write_bytes(b"persisted")
    second = _ComputeWorker(str(tmp_path))
    assert first.root == second.root == tmp_path
    assert first.node_id != second.node_id
    assert marker.read_bytes() == b"persisted"
    scratch.assert_called_with(second.root)
