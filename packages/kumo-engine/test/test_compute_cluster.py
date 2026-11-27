# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact orchestration and integrity without provisioning a Ray cluster."""

import hashlib
import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kumo_core.diskgraph.sampling_artifacts import ArtifactBundle, ArtifactFile

import kumo.engine as kumo
from kumo.engine._mlops import resolve
from kumo.engine.cluster import (
    _ComputeWorker,
    _Replicator,
    _TrainingTableRequest,
)


def _worker(tmp_path: Path) -> _ComputeWorker:
    worker = object.__new__(_ComputeWorker)
    worker.root = tmp_path / "worker"
    worker.root.mkdir()
    worker.node_id = "worker-node"
    worker._downloads = {}
    worker.snapshots = {}
    worker.tables = {}
    worker._staged = {}
    return worker


def test_mlops_context_restores_nested_settings(tmp_path: Path) -> None:
    """The existing mlops wrapper restores settings after nested contexts."""
    outer = kumo.MLOps(str(tmp_path / "outer"))
    inner = kumo.MLOps(str(tmp_path / "inner"))
    assert resolve() is None
    with kumo.mlops(outer) as active:
        assert active is outer and resolve() is outer
        with kumo.mlops(outer):
            with kumo.mlops(inner):
                assert resolve() is inner
            assert resolve() is outer
        assert resolve() is outer
    assert resolve() is None


@pytest.mark.parametrize(
    "query_text",
    [
        "PREDICT COUNT(events.*, 0, 7, days) > 0 FOR EACH users.ID",
        (
            "PREDICT SUM(events.VALUE, 0, 7, days) FORECAST 3 TIMEFRAMES "
            "FOR EACH users.ID"
        ),
    ],
)
def test_downloaded_training_rows_bypass_connector_staging(
    graph: kumo.Graph,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    query_text: str,
) -> None:
    """Restore portable rows unchanged and reuse complete worker-local staging."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kumo_core.training import table as native_table
    from kumo_core.training.export import export_schema, read_spec, stage_parts
    from kumoapi.typing import Stype

    from kumo.engine._scratch import scoped_scratch_dir
    from kumo.engine.cluster import _TrainingTablePublication

    events_path = tmp_path / "data/events.parquet"
    events = pq.read_table(events_path)
    events = events.append_column(
        "VALUE",
        pa.array([1.0000000000000002 + index / 7 for index in range(len(events))]),
    )
    pq.write_table(events, events_path)
    event_table = kumo.Table.from_source_table(
        graph.tables["events"].connector["events.parquet"],
        primary_key="EID",
        time_column="TS",
    ).with_column("USER_ID", stype=Stype.ID)
    graph = kumo.Graph(
        tables={**graph.tables, "events": event_table}, edges=graph.edges
    )
    producer = tmp_path / "producer"
    settings = kumo.MLOps(str(tmp_path / "artifacts"))
    with scoped_scratch_dir(producer):
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(graph=graph, query=query_text)
        table = query.generate_training_table(snapshot, mlops=settings)
    worker = _worker(tmp_path)
    worker.snapshots[snapshot.id] = snapshot
    publication = _TrainingTablePublication(
        table, snapshot.id, str(producer), table.manifest
    )
    stage = Mock(side_effect=AssertionError("Downloaded rows were staged again"))
    monkeypatch.setattr(
        "kumo.engine.connector.file._FileSourceBackend.prepare_ingest", stage
    )
    materialize = Mock(wraps=native_table.generate_custom_train_table)
    monkeypatch.setattr(native_table, "generate_custom_train_table", materialize)

    with scoped_scratch_dir(worker.root):
        restored = worker.download_training_table(publication, settings)
        cached = worker.download_training_table(publication, settings)

    stage.assert_not_called()
    materialize.assert_called_once()
    assert cached.staging_dir == restored.staging_dir
    assert restored.id == table.id
    assert restored.generation_plan == table.generation_plan
    assert restored.required_stages == table.required_stages
    ingested = native_table.ingest_train_table(
        staging_dir=str(restored.staging_dir),
        index_path=str(snapshot.index_path),
        tt_id="downloaded-rows",
    )
    assert Path(ingested.manifest_path).is_file()
    expected_parts = stage_parts(read_spec(str(table.staging_dir)))
    actual_parts = stage_parts(read_spec(str(restored.staging_dir)))
    assert actual_parts.keys() == expected_parts.keys()
    for split, parts in expected_parts.items():
        expected = pa.concat_tables(
            [pq.read_table(table.staging_dir / part) for part in parts]
        )
        columns = export_schema(expected.schema).names[:-1]
        expected = expected.select(columns)
        actual = pa.concat_tables(
            [pq.read_table(restored.staging_dir / part) for part in actual_parts[split]]
        ).select(columns)
        assert actual.schema == expected.schema
        order = [("ENTITY1", "ascending"), ("TIMESTAMP", "ascending")]
        np.testing.assert_equal(
            actual.sort_by(order).to_pydict(), expected.sort_by(order).to_pydict()
        )


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
def test_invalid_worker_configuration(workers: int | None, cpus: float | None) -> None:
    """Resource errors fail before connecting to Ray."""
    with pytest.raises(ValueError, match="positive"):
        kumo.ComputeClusterConfig(num_workers=workers, cpus_per_worker=cpus)


def test_receive_verifies_before_commit_and_relocates_metadata(
    tmp_path: Path,
) -> None:
    """Files arrive early but readiness requires the exact complete inventory."""
    worker = _worker(tmp_path)
    worker.begin("publication", kumo.MLOps(str(tmp_path / "store")))
    source = tmp_path / "store" / "objects"
    source.mkdir(parents=True)
    data = json.dumps({"path": "/producer/graph/index.bin"}).encode("utf-8")
    (source / "metadata.json").write_bytes(data)
    entry = ArtifactFile("metadata.json", len(data), hashlib.sha256(data).hexdigest())
    bundle = ArtifactBundle("/producer", (entry,))
    with pytest.raises(RuntimeError, match="incomplete"):
        worker.commit("publication", "graph", bundle)
    worker.receive("publication", "graph", str(source), entry)
    assert "publication" in worker._downloads
    replica = worker.commit("publication", "graph", bundle)
    assert replica.publication == "publication"
    assert "publication" not in worker._downloads
    assert json.loads((worker.root / "graph/metadata.json").read_bytes()) == {
        "path": str(worker.root / "graph/index.bin")
    }


def test_receive_rejects_corruption_without_publishing(tmp_path: Path) -> None:
    """Corrupt data leaves diagnostic partial files and never becomes ready."""
    worker = _worker(tmp_path)
    worker.begin("bad", kumo.MLOps(str(tmp_path / "store")))
    source = tmp_path / "store"
    source.mkdir()
    (source / "index.bin").write_bytes(b"bad")
    entry = ArtifactFile("index.bin", 3, hashlib.sha256(b"yes").hexdigest())
    worker.receive("bad", "graph", str(source), entry)
    with pytest.raises(ValueError, match="checksum"):
        worker.commit("bad", "graph", ArtifactBundle("/producer", (entry,)))
    assert not worker._downloads
    assert not (worker.root / "graph/index.bin").exists()
    assert list((worker.root / "graph").glob("*.partial"))


@pytest.mark.parametrize("name", ["../escape", "/escape", "a/../../escape"])
def test_receive_rejects_paths_outside_worker(tmp_path: Path, name: str) -> None:
    """An inventory cannot install objects outside its worker root."""
    worker = _worker(tmp_path)
    with pytest.raises(ValueError):
        worker.receive("bad", "graph", "unused", ArtifactFile(name, 0, ""))


def test_notifications_start_before_publication_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slow downloads never make upload notifications wait for replica completion."""
    import ray

    peer = SimpleNamespace(begin=Mock(), receive=Mock(), commit=Mock())
    peer.receive.remote = Mock(side_effect=["one", "two", "three"])
    peer.commit.remote = Mock(return_value="ready")
    get = Mock(
        side_effect=lambda value: ["replica"] if isinstance(value, list) else None
    )
    monkeypatch.setattr(ray, "get", get)
    transfer = _Replicator([peer], "graph", "publication", Mock(spec=kumo.MLOps))
    get.reset_mock()
    first = ArtifactFile("first.bin", 1, "a")
    second = ArtifactFile("second.bin", 1, "b")
    transfer.uploaded("store/objects", first)
    peer.receive.remote.assert_called_once_with(
        "publication", "graph", "store/objects", first
    )
    peer.commit.remote.assert_not_called()
    transfer.uploaded("store/objects", second)
    transfer.uploaded("store/objects", ArtifactFile("third.bin", 1, "c"))
    get.assert_not_called()
    assert peer.receive.remote.call_count == 3
    assert transfer.finish(Path("/producer")) == ("replica",)
    assert get.call_count == 2
    assert not transfer.pending


def test_cluster_failure_prevents_subsequent_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed publication poisons the cluster instead of using partial replicas."""
    import ray

    cluster = kumo.ComputeCluster(
        kumo.ComputeClusterConfig(), kumo.ArtifactRoot(str(tmp_path))
    )
    cluster._active = True
    actor = Mock()
    cluster._actors = [actor]
    monkeypatch.setattr(ray, "get", Mock(side_effect=OSError("download failed")))
    with pytest.raises(OSError, match="download failed"):
        cluster._call("prepare", actor)
    with pytest.raises(RuntimeError, match="failed"):
        cluster._call("prepare", actor)
    assert actor.prepare.remote.call_count == 1


def test_prepare_checks_table_snapshot_ownership(tmp_path: Path) -> None:
    """Mixing snapshots cannot ingest a table against another graph version."""
    cluster = kumo.ComputeCluster(
        kumo.ComputeClusterConfig(), kumo.ArtifactRoot(str(tmp_path / "store"))
    )
    snapshot = SimpleNamespace(_compute=cluster, id="snapshot")
    table = SimpleNamespace(_compute=cluster, id="table")
    cluster._tables["table"] = "other-snapshot"
    with pytest.raises(ValueError, match="must belong"):
        cluster.prepare_training(table, snapshot, Mock())


def test_trainer_prepares_before_stopping_at_execution_boundary(
    tmp_path: Path,
) -> None:
    """fit prepares remotely and exposes the handoff without reading local indexes."""
    cluster = Mock()
    prepared = Mock()
    cluster.prepare_training.return_value = prepared
    snapshot = SimpleNamespace(_compute=cluster)
    from kumoapi.model_plan import ModelPlan

    trainer = kumo.Trainer(ModelPlan())
    with pytest.raises(NotImplementedError, match="Remote artifacts are ready"):
        trainer.fit(Mock(), snapshot=snapshot)
    assert trainer.prepared_artifacts is prepared
    cluster.prepare_training.assert_called_once()


def test_native_artifacts_are_produced_once_and_replicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise real graph build, TT generation and ingestion with synchronous RPCs."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import ray
    from kumo_core.model_plan import ModelPlan

    from kumo.engine._scratch import scoped_scratch_dir

    source = tmp_path / "source"
    source.mkdir()
    pq.write_table(
        pa.table({"id": range(64), "target": [i % 2 == 0 for i in range(64)]}),
        source / "rows.parquet",
    )
    graph = kumo.Graph(
        tables={
            "users": kumo.Table.from_source_table(
                kumo.FileConnector(root_dir=str(source))["rows.parquet"],
                primary_key="id",
            ).infer_metadata()
        },
        edges=[],
    )
    producer_dir, replica_dir = tmp_path / "producer", tmp_path / "replica"
    producer_dir.mkdir()
    replica_dir.mkdir()
    producer, replica = _worker(producer_dir), _worker(replica_dir)
    settings = kumo.MLOps(str(producer_dir / "store"))

    peer = SimpleNamespace(
        download_training_table=Mock(),
        begin=SimpleNamespace(remote=replica.begin),
        receive=SimpleNamespace(remote=replica.receive),
        commit=SimpleNamespace(remote=replica.commit),
        abort=SimpleNamespace(remote=replica.abort),
    )
    monkeypatch.setattr(ray, "get", lambda value: value)
    with scoped_scratch_dir(producer.root):
        from kumo_core.diskgraph import ingest

        build_index = Mock(wraps=ingest.execute_diskgraph_ingest)
        monkeypatch.setattr(ingest, "execute_diskgraph_ingest", build_index)
        options = {"timestamp_precision": "seconds", "record_sources": True}
        descriptor, source_tables = producer.stage_snapshot(graph, options)
        snapshot = producer.produce_snapshot(
            (graph, options, descriptor), settings, source_tables
        )
        cached_descriptor, _ = producer.stage_snapshot(graph, options)
        assert cached_descriptor._data_version == snapshot._data_version
        producer.discard_staging(cached_descriptor.id)
        with scoped_scratch_dir(replica.root):
            replica.download_snapshot(descriptor, settings, source_tables)
            assert replica.has_snapshot(descriptor)
            cached = replica.produce_snapshot(
                (graph, options, descriptor), settings, source_tables
            )
            assert cached.id == snapshot.id
            replica.download_snapshot(descriptor, settings, source_tables)
        assert build_index.call_count == 1
        index = replica.root / snapshot.index_path.relative_to(producer.root)
        assert (index / "_SUCCESS").is_file()
        assert str(producer.root) not in (index / "diskgraph_config.json").read_text()
        for path in snapshot.index_path.rglob("*.bin"):
            copied = replica.root / path.relative_to(producer.root)
            assert copied.read_bytes() == path.read_bytes()
        from kumo.engine import _training_table_io

        publication = Mock(wraps=_training_table_io.store_prepared)
        monkeypatch.setattr(_training_table_io, "store_prepared", publication)
        portable = Mock(
            side_effect=AssertionError("Published a second row representation")
        )
        monkeypatch.setattr(kumo.TrainingTable, "export", portable)
        from kumo.engine.pquery import predictive_query

        generate_rows = Mock(wraps=predictive_query.generate_train_table)
        monkeypatch.setattr(predictive_query, "generate_train_table", generate_rows)
        table = producer.training_table_generate(
            _TrainingTableRequest(
                snapshot.id, "PREDICT users.target FOR EACH users.id", None
            ),
            settings,
        )
        assert table.id is not None
        from copy import copy

        restarted = copy(producer)
        restarted.tables = {}
        query = kumo.PredictiveQuery._from_snapshot(
            snapshot, "PREDICT users.target FOR EACH users.id"
        )
        generation_request = _TrainingTableRequest(snapshot.id, query.query, None)
        descriptor = restarted.describe_training_table(generation_request)
        assert descriptor.table.id == table.id
        assert restarted.has_training_table(descriptor)
        repeated = restarted.training_table_generate(generation_request, settings)
        assert repeated.id == table.id
        assert repeated.generation_plan == table.generation_plan
        assert generate_rows.call_count == 1
        publication.assert_not_called()
        portable.assert_not_called()
        store = Path(settings.artifact_root)
        stored = store / "training_tables" / table.id
        assert (stored / "_SUCCESS").is_file()
        assert not (stored / "native").exists()
        assert "SPLIT" in pq.read_schema(next(stored.glob("*.parquet"))).names
        peer.download_training_table.remote.assert_not_called()
        assert table.id not in replica.tables
        assert not (
            replica.root / table.staging_dir.relative_to(producer.root)
        ).exists()
        from kumo_core.training import workflow

        ingest_rows = Mock(wraps=workflow.ingest_train_table)
        monkeypatch.setattr(workflow, "ingest_train_table", ingest_rows)
        prepared = producer.prepare(
            (snapshot.id, table.id, "first-fit"), ModelPlan(), settings, [peer]
        )
        first_extension = Path(prepared.materialized.train_table_extension_path)
        before = {
            path.relative_to(first_extension): path.read_bytes()
            for path in first_extension.rglob("*")
            if path.is_file()
        }
        second_plan = ModelPlan()
        second_plan.optimization.max_epochs = 2
        next_fit = producer.prepare(
            (snapshot.id, table.id, "second-fit"),
            second_plan,
            settings,
            [peer],
        )
        assert next_fit.materialized.train_table_extension_path != str(first_extension)
        assert ingest_rows.call_count == 2
        assert next_fit.materialized.model_plan.optimization.max_epochs == 2
        assert before == {
            path.relative_to(first_extension): path.read_bytes()
            for path in first_extension.rglob("*")
            if path.is_file()
        }
        empty_store = kumo.MLOps(str(tmp_path / "empty-artifact-root"))
        republished = restarted.training_table_generate(generation_request, empty_store)
        assert republished.id == table.id
        assert generate_rows.call_count == 1
        assert ingest_rows.call_count == 2

    extension = Path(prepared.materialized.train_table_extension_path)
    remote_extension = replica.root / extension.relative_to(producer.root)
    assert (remote_extension / "diskgraph_manifest.json").is_file()
    assert len(prepared.replicas) == 2
    assert prepared.replicas[0].publication == prepared.replicas[1].publication
    assert prepared.snapshot_id == snapshot.id and prepared.table_id == table.id
    for path in extension.rglob("*.bin"):
        assert (
            replica.root / path.relative_to(producer.root)
        ).read_bytes() == path.read_bytes()
    for path in (replica.root / "fits/first-fit").rglob("*.json"):
        assert str(producer.root) not in path.read_text()


@pytest.mark.parametrize(
    "workers,cpus,expected",
    [
        (None, None, [("a" * 56, 4), ("b" * 56, 8)]),
        (None, 2, [("a" * 56, 2), ("b" * 56, 2)]),
        (None, 6, [("b" * 56, 6)]),
        (1, None, [("a" * 56, 4)]),
        (2, None, [("a" * 56, 4), ("b" * 56, 8)]),
    ],
)
def test_cluster_owns_only_remote_actors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workers: int | None,
    cpus: float | None,
    expected: list[tuple[str, float]],
) -> None:
    """Placement excludes the trainer and closing preserves the Ray runtime."""
    import ray

    actors = [Mock() for _ in expected]
    actor_type = Mock()
    actor_type.options.return_value = actor_type
    actor_type.remote.side_effect = actors
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "driver"),
    )
    monkeypatch.setattr(
        ray,
        "nodes",
        lambda: (
            [
                {
                    "Alive": True,
                    "NodeID": name,
                    "Resources": {"CPU": capacity, "disk": 1},
                }
                for name, capacity in (
                    ("driver", 16),
                    ("empty", 0),
                    ("a" * 56, 4),
                    ("b" * 56, 8),
                )
            ]
            + [
                {
                    "Alive": False,
                    "NodeID": "dead",
                    "Resources": {"CPU": 8, "disk": 1},
                },
                {"Alive": True, "NodeID": "no-disk", "Resources": {"CPU": 8}},
            ]
        ),
    )
    monkeypatch.setattr(ray, "remote", Mock(return_value=actor_type))
    monkeypatch.setattr(
        ray, "get", Mock(return_value=[("node", "root")] * len(expected))
    )
    kill, shutdown = Mock(), Mock()
    monkeypatch.setattr(ray, "kill", kill)
    monkeypatch.setattr(ray, "shutdown", shutdown)
    cluster = kumo.ComputeCluster(
        kumo.ComputeClusterConfig(
            resources={"disk": 1},
            **({"num_workers": workers} if workers is not None else {}),
            **({"cpus_per_worker": cpus} if cpus is not None else {}),
        ),
        kumo.ArtifactRoot(str(tmp_path / "store")),
    )
    cluster.start()
    try:
        assert cluster._active
        placements = [
            (
                call.kwargs["scheduling_strategy"].node_id,
                call.kwargs["num_cpus"],
            )
            for call in actor_type.options.call_args_list
        ]
        assert placements == expected
        assert all(len(call.args) == 1 for call in actor_type.remote.call_args_list)
    finally:
        cluster.close()
    assert not cluster._active and not cluster._actors
    assert kill.call_count == len(expected)
    shutdown.assert_not_called()


@pytest.mark.parametrize("workers,available", [(None, 0), (1, 0), (3, 2)])
def test_cluster_requires_enough_eligible_nodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workers: int | None, available: int
) -> None:
    """An empty default selection or unmet explicit count fails before actors start."""
    import ray

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "driver"),
    )
    monkeypatch.setattr(
        ray,
        "nodes",
        lambda: [
            {"Alive": True, "NodeID": name, "Resources": {"CPU": 4}}
            for name in ["driver", *[str(index) for index in range(available)]]
        ],
    )
    start = Mock()
    monkeypatch.setattr(ray, "remote", start)

    cluster = kumo.ComputeCluster(
        kumo.ComputeClusterConfig(num_workers=workers), kumo.ArtifactRoot(str(tmp_path))
    )
    with pytest.raises(RuntimeError, match="Not enough distinct remote"):
        cluster.start()

    start.assert_not_called()


def test_publication_failure_never_writes_commit_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replica failures propagate and prevent a durable compute manifest."""
    import ray

    worker = _worker(tmp_path)
    source = worker.root / "task"
    source.mkdir()
    (source / "data.bin").write_bytes(b"data")
    peer = SimpleNamespace(begin=Mock(), receive=Mock(), abort=Mock())
    peer.receive.remote = Mock(return_value="failed-download")
    monkeypatch.setattr(
        ray, "get", Mock(side_effect=[None, OSError("download failed"), None])
    )
    settings = kumo.MLOps(str(tmp_path / "store"))
    with (
        pytest.raises(OSError, match="download failed"),
        _Replicator([peer], "", "publication", settings) as transfer,
    ):
        from kumo.engine._training_table_io import store_prepared

        store_prepared(
            worker.root,
            (source,),
            settings,
            transfer.uploaded,
            lambda: transfer.finish(worker.root),
        )
    peer.abort.remote.assert_called_once_with("publication")
    assert not list((tmp_path / "store").rglob("manifest.json"))


def test_real_ray_workers_replicate_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross process production keeps indexes off the driver and shares fit versions."""
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq
    import ray

    if ray.is_initialized():
        pytest.skip("This test owns an isolated Ray runtime")
    source = tmp_path / "source"
    source.mkdir()
    pq.write_table(
        pa.table({"id": range(64), "target": [i % 2 == 0 for i in range(64)]}),
        source / "rows.parquet",
    )
    graph = kumo.Graph(
        tables={
            "users": kumo.Table.from_source_table(
                kumo.FileConnector(root_dir=str(source))["rows.parquet"],
                primary_key="id",
            ).infer_metadata()
        },
        edges=[],
    )
    monkeypatch.setenv("RAY_AUTH_MODE", "disabled")
    monkeypatch.setenv("RAY_USAGE_STATS_ENABLED", "0")
    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        object_store_memory=80 << 20,
        _node_ip_address="127.0.0.1",
        _temp_dir=tempfile.mkdtemp(prefix="kcc-"),
    )
    actors = []
    cluster = None
    try:
        root = tmp_path / "artifacts"
        actor_type = ray.remote(num_cpus=1)(_ComputeWorker)
        actors = [
            actor_type.remote(str(tmp_path / "workers" / str(index)))
            for index in range(2)
        ]
        cluster = kumo.ComputeCluster(
            kumo.ComputeClusterConfig(num_workers=2), kumo.ArtifactRoot(str(root))
        )
        # This local runtime has one node; remote placement is tested separately.
        cluster._actors = actors
        cluster._active = True
        monkeypatch.setattr(
            kumo.Graph,
            "_snapshot_no_lock",
            Mock(side_effect=AssertionError("Driver tried to build an index")),
        )
        monkeypatch.setattr(
            "kumo.engine.pquery.predictive_query.generate_train_table",
            Mock(side_effect=AssertionError("Driver tried to generate rows")),
        )
        monkeypatch.setattr(
            "kumo_core.training.workflow.materialize_generated_training_task",
            Mock(side_effect=AssertionError("Driver tried to ingest rows")),
        )
        monkeypatch.setattr("kumo.engine.config._cluster", cluster)
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(
            graph=graph, query="PREDICT users.target FOR EACH users.id"
        )
        table = query.generate_training_table(snapshot)
        generated_marker = root / "training_tables" / table.id / "_SUCCESS"
        generated_time = generated_marker.stat().st_mtime_ns
        assert query.generate_training_table(snapshot).id == table.id
        assert generated_marker.stat().st_mtime_ns == generated_time
        trainer = kumo.Trainer(query.suggest_model_plan(snapshot))
        prepared = trainer.prepare_artifacts(table, snapshot=snapshot)
        second_plan = query.suggest_model_plan(snapshot)
        second_plan.optimization.max_epochs = 2
        repeated = kumo.Trainer(second_plan).prepare_artifacts(table, snapshot=snapshot)
        assert (
            repeated.materialized.train_table_extension_path
            != prepared.materialized.train_table_extension_path
        )
        assert repeated.run_id != prepared.run_id
        assert repeated.materialized.model_plan.optimization.max_epochs == 2
        assert not (root / "training_tables/cache").exists()
        assert {path.name for path in root.iterdir()} == {
            "snapshots",
            "training_tables",
        }
        assert len(prepared.replicas) == 2
        assert prepared.replicas[0].root != prepared.replicas[1].root
        assert prepared.replicas[0].publication == prepared.replicas[1].publication
        assert prepared.table_id == table.id
        assert prepared.snapshot_id == snapshot.id
    finally:
        if cluster is not None:
            cluster.close()
        else:
            for actor in actors:
                ray.kill(actor, no_restart=True)
        ray.shutdown()


def test_snapshot_uses_global_cluster(
    graph: kumo.Graph,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialization routes snapshots to the configured pool without call overrides."""
    from kumo.engine._scratch import scoped_scratch_dir

    local = Mock()
    build_local = Mock(return_value=local)
    monkeypatch.setattr(kumo.Graph, "_snapshot_no_lock", build_local)
    monkeypatch.setattr("kumo.engine.config._cluster", None)
    with scoped_scratch_dir(tmp_path / "scratch"):
        assert graph.snapshot() is local
    cluster = Mock(spec=kumo.ComputeCluster)
    monkeypatch.setattr("kumo.engine.config._cluster", cluster)
    result = graph.snapshot(record_sources=False)
    assert result is cluster._graph_snapshot.return_value
    cluster._graph_snapshot.assert_called_once_with(
        graph, {"timestamp_precision": "seconds", "record_sources": False}
    )
    with pytest.raises(ValueError, match="kumo.settings.configure"):
        graph.snapshot(mlops=Mock())
    with pytest.raises(ValueError, match="callbacks"):
        graph.snapshot(callbacks=[Mock()])
    assert build_local.call_count == 1


def test_downloads_overlap_with_a_bounded_pool_and_commit_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Notifications stay nonblocking while a bounded pool downloads multiple files."""
    worker = _worker(tmp_path)
    monkeypatch.setattr("kumo.engine.cluster.download_workers", lambda: 2)
    saturated, release = Event(), Event()
    lock = Lock()
    active = peak = completed = 0

    def download(
        settings: kumo.MLOps, directory: str, entry: ArtifactFile, root: Path
    ) -> None:
        nonlocal active, peak, completed
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                saturated.set()
        if not release.wait(timeout=10):
            raise TimeoutError("Test did not release the blocked downloads")
        with lock:
            active -= 1
            completed += 1

    def relocate(bundle: ArtifactBundle, root: Path, *, destination_root: Path) -> None:
        assert completed == len(bundle.files)

    monkeypatch.setattr(worker, "_download", download)
    relocation = Mock(side_effect=relocate)
    monkeypatch.setattr("kumo.engine.cluster.relocate_artifacts", relocation)
    worker.begin("publication", kumo.MLOps(str(tmp_path / "store")))
    entries = tuple(ArtifactFile(f"file-{index}", 0, "") for index in range(3))
    try:
        for entry in entries:
            worker.receive("publication", "graph", "objects", entry)
        assert saturated.wait(timeout=10)
        with ThreadPoolExecutor(max_workers=1) as barrier:
            ready = barrier.submit(
                worker.commit,
                "publication",
                "graph",
                ArtifactBundle("/producer", entries),
            )
            try:
                assert not ready.done()
                relocation.assert_not_called()
            finally:
                release.set()
            assert ready.result(timeout=10).publication == "publication"
        assert peak == 2
        assert not worker._downloads
        relocation.assert_called_once()
    finally:
        release.set()
        worker.abort("publication")


def test_failed_download_drains_the_pool_without_relocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed file reaches commit as an error after all pool work has stopped."""
    worker = _worker(tmp_path)
    monkeypatch.setattr("kumo.engine.cluster.download_workers", lambda: 1)
    download = Mock(side_effect=[OSError("download failed"), None])
    monkeypatch.setattr(worker, "_download", download)
    relocation = Mock()
    monkeypatch.setattr("kumo.engine.cluster.relocate_artifacts", relocation)
    worker.begin("publication", kumo.MLOps(str(tmp_path / "store")))
    entries = tuple(ArtifactFile(f"file-{index}", 0, "") for index in range(2))
    for entry in entries:
        worker.receive("publication", "graph", "objects", entry)
    futures = [future for _, future in worker._downloads["publication"].files.values()]
    with pytest.raises(OSError, match="download failed"):
        worker.commit("publication", "graph", ArtifactBundle("/producer", entries))
    assert all(future.done() for future in futures)
    relocation.assert_not_called()
    assert not worker._downloads


def test_abort_cancels_queued_downloads_and_waits_for_active_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aborting a publication leaves no queued or active work behind."""
    worker = _worker(tmp_path)
    monkeypatch.setattr("kumo.engine.cluster.download_workers", lambda: 1)
    started, release, cancelled = Event(), Event(), Event()

    def download(
        settings: kumo.MLOps, directory: str, entry: ArtifactFile, root: Path
    ) -> None:
        started.set()
        if not release.wait(timeout=10):
            raise TimeoutError("Test did not release the active download")

    def observe_cancellation(future: Future[None]) -> None:
        if future.cancelled():
            cancelled.set()

    transfer = Mock(side_effect=download)
    monkeypatch.setattr(worker, "_download", transfer)
    worker.begin("publication", kumo.MLOps(str(tmp_path / "store")))
    entries = tuple(ArtifactFile(f"file-{index}", 0, "") for index in range(2))
    try:
        worker.receive("publication", "graph", "objects", entries[0])
        assert started.wait(timeout=10)
        worker.receive("publication", "graph", "objects", entries[1])
        futures = [
            future for _, future in worker._downloads["publication"].files.values()
        ]
        futures[1].add_done_callback(observe_cancellation)
        with ThreadPoolExecutor(max_workers=1) as cleanup:
            stopped = cleanup.submit(worker.abort, "publication")
            try:
                assert cancelled.wait(timeout=10)
                assert not stopped.done()
                assert not futures[0].done()
            finally:
                release.set()
            stopped.result(timeout=10)
        assert all(future.done() for future in futures)
        transfer.assert_called_once()
        assert not worker._downloads
        worker.abort("publication")
    finally:
        release.set()
        worker.abort("publication")


def test_replication_cleanup_preserves_the_original_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed abort is logged while the original publication error propagates."""
    import ray
    from ray.exceptions import RayError

    peer = SimpleNamespace(begin=Mock(), abort=Mock())
    monkeypatch.setattr(
        ray, "get", Mock(side_effect=[None, RayError("cleanup failed")])
    )
    with (
        pytest.raises(ValueError, match="publication failed"),
        _Replicator([peer], "", "publication", Mock()),
    ):
        raise ValueError("publication failed")
    peer.abort.remote.assert_called_once_with("publication")
    assert "Could not abort replica downloads" in caplog.text


def test_duplicate_publications_and_files_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicate notifications cannot replace a pool or race writes to one file."""
    worker = _worker(tmp_path)
    monkeypatch.setattr(worker, "_download", Mock())
    settings = kumo.MLOps(str(tmp_path / "store"))
    worker.begin("publication", settings)
    try:
        with pytest.raises(RuntimeError, match="already open"):
            worker.begin("publication", settings)
        entry = ArtifactFile("file", 0, "")
        worker.receive("publication", "graph", "objects", entry)
        with pytest.raises(ValueError, match="already scheduled"):
            worker.receive("publication", "graph", "objects", entry)
    finally:
        worker.abort("publication")


def test_async_download_failure_prevents_durable_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queued download failure aborts the producer's final manifest publication."""
    import ray

    from kumo.engine._training_table_io import store_prepared

    replica = _worker(tmp_path)
    monkeypatch.setattr(
        replica, "_download", Mock(side_effect=OSError("download failed"))
    )
    peer = SimpleNamespace(
        begin=SimpleNamespace(remote=replica.begin),
        receive=SimpleNamespace(remote=replica.receive),
        commit=SimpleNamespace(remote=replica.commit),
        abort=SimpleNamespace(remote=replica.abort),
    )
    monkeypatch.setattr(ray, "get", lambda value: value)
    source = tmp_path / "task"
    source.mkdir()
    (source / "data.bin").write_bytes(b"data")
    settings = kumo.MLOps(str(tmp_path / "store"))
    with (
        pytest.raises(OSError, match="download failed"),
        _Replicator([peer], "", "publication", settings) as transfer,
    ):
        store_prepared(
            tmp_path,
            (source,),
            settings,
            transfer.uploaded,
            lambda: transfer.finish(tmp_path),
        )
    assert not list((tmp_path / "store").rglob("manifest.json"))
    assert not replica._downloads


def test_replication_scope_aborts_an_unfinished_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leaving a replication scope before finish releases its peer download pools."""
    import ray

    peer = SimpleNamespace(begin=Mock(), abort=Mock())
    monkeypatch.setattr(ray, "get", lambda value: value)
    with _Replicator([peer], "", "publication", Mock()) as transfer:
        assert transfer.publication == "publication"
    peer.abort.remote.assert_called_once_with("publication")
