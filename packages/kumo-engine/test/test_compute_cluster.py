# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact orchestration and integrity without provisioning a Ray cluster."""

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import kumo.engine as kumo
import orjson
import pytest
from kumo.engine._mlops import resolve
from kumo.engine.cluster import _ComputeWorker, _Replicator, _TrainingTableRequest
from kumo_core.diskgraph.sampling_artifacts import ArtifactBundle, ArtifactFile


def _worker(tmp_path: Path) -> _ComputeWorker:
    worker = object.__new__(_ComputeWorker)
    worker.root = tmp_path / "worker"
    worker.root.mkdir()
    worker.node_id = "worker-node"
    worker.received = {}
    worker._stores = {}
    worker.snapshots = {}
    worker.tables = {}
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
    "method,kind",
    [
        ("graph_snapshot", "snapshot"),
        ("training_table_generate", "training_table"),
        ("prepare", "training_table"),
    ],
)
@pytest.mark.parametrize(
    "store", [True, False, {"snapshot"}, {"training_table"}, {"model"}]
)
def test_cluster_validates_only_operation_artifact_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    kind: str,
    store: bool | set[str],
) -> None:
    """Validate each publication's kind without enabling unrelated exports."""
    import ray

    cluster = kumo.ComputeCluster()
    settings = kumo.MLOps(str(tmp_path), store=store)
    actor = Mock()
    cluster._actors = [actor]
    cluster._active = True
    monkeypatch.setattr(ray, "get", Mock(return_value="result"))
    remote = getattr(actor, method).remote
    if kind not in settings.kinds:
        with pytest.raises(ValueError, match=f"{method} requires MLOps storing {kind}"):
            cluster._call(method, settings)
        remote.assert_not_called()
        assert not cluster._failed
        return

    assert cluster._call(method, settings) == "result"
    remote.assert_called_once()
    worker_settings, peers = remote.call_args.args
    assert worker_settings.kinds == settings.kinds
    assert worker_settings.artifact_root == settings.artifact_root
    assert peers == []


def test_cluster_requires_settings_for_each_operation(tmp_path: Path) -> None:
    """A previously active context cannot supply a later operation's settings."""
    with kumo.mlops(kumo.MLOps(str(tmp_path))):
        cluster = kumo.ComputeCluster()
    with pytest.raises(ValueError, match="active kumo.mlops"):
        cluster._call("graph_snapshot", None)
    assert not cluster._failed


@pytest.mark.parametrize("override", [False, True])
def test_table_operations_resolve_settings_at_call_time(
    snapshot: kumo.GraphSnapshot, tmp_path: Path, override: bool
) -> None:
    """Generation and ingestion resolve independently, preserving overrides."""
    from kumoapi.model_plan import ModelPlan

    cluster = Mock(spec=kumo.ComputeCluster)
    snapshot = replace(snapshot, _compute=cluster)
    old, generation, ingestion, explicit = [
        kumo.MLOps(str(tmp_path / name))
        for name in ("old", "generation", "ingestion", "explicit")
    ]
    with kumo.mlops(old):
        query = kumo.PredictiveQuery._from_snapshot(
            snapshot, "PREDICT users.label FOR EACH users.id"
        )
        trainer = kumo.Trainer(ModelPlan())
    kwargs = {"mlops": explicit} if override else {}
    with kumo.mlops(generation):
        table = query.generate_training_table(snapshot, **kwargs)
    cluster._training_table_generate.assert_called_once_with(
        query, snapshot, None, explicit if override else generation
    )
    with kumo.mlops(ingestion):
        trainer.prepare_artifacts(table, snapshot=snapshot, **kwargs)
    assert cluster.prepare_training.call_args.args[-1] is (
        explicit if override else ingestion
    )
    cluster.prepare_training.reset_mock()
    with (
        kumo.mlops(ingestion),
        pytest.raises(NotImplementedError, match="Remote artifacts are ready"),
    ):
        trainer.fit(table, snapshot=snapshot, **kwargs)
    assert cluster.prepare_training.call_args.args[-1] is (
        explicit if override else ingestion
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
        kumo.ComputeCluster(num_workers=workers, cpus_per_worker=cpus)


def test_receive_verifies_before_commit_and_relocates_metadata(tmp_path: Path) -> None:
    """Files arrive early but readiness requires the exact complete inventory."""
    worker = _worker(tmp_path)
    worker.begin("publication", kumo.MLOps(str(tmp_path / "store")))
    source = tmp_path / "store" / "objects"
    source.mkdir(parents=True)
    data = orjson.dumps({"path": "/producer/graph/index.bin"})
    (source / "metadata.json").write_bytes(data)
    entry = ArtifactFile("metadata.json", len(data), hashlib.sha256(data).hexdigest())
    bundle = ArtifactBundle("/producer", (entry,))
    with pytest.raises(RuntimeError, match="incomplete"):
        worker.commit("publication", "graph", bundle)
    worker.receive("publication", "graph", str(source), entry)
    assert "publication" in worker.received
    replica = worker.commit("publication", "graph", bundle)
    assert replica.publication == "publication"
    assert "publication" not in worker.received
    assert "publication" not in worker._stores
    assert orjson.loads((worker.root / "graph/metadata.json").read_bytes()) == {
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
    with pytest.raises(ValueError, match="checksum"):
        worker.receive("bad", "graph", str(source), entry)
    assert not worker.received
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed publication poisons the cluster instead of using partial replicas."""
    import ray

    cluster = kumo.ComputeCluster()
    cluster._active = True
    actor = SimpleNamespace(graph_snapshot=Mock())
    actor.graph_snapshot.remote = Mock(return_value="operation")
    cluster._actors = [actor]
    monkeypatch.setattr(ray, "get", Mock(side_effect=OSError("download failed")))
    with pytest.raises(OSError, match="download failed"):
        cluster._call("graph_snapshot", kumo.MLOps(str(tmp_path)))
    with pytest.raises(RuntimeError, match="failed"):
        cluster._call("graph_snapshot", kumo.MLOps(str(tmp_path)))
    assert actor.graph_snapshot.remote.call_count == 1


def test_prepare_checks_table_snapshot_ownership(tmp_path: Path) -> None:
    """Mixing snapshots cannot ingest a table against another graph version."""
    cluster = kumo.ComputeCluster()
    snapshot = SimpleNamespace(_compute=cluster, id="snapshot")
    table = SimpleNamespace(_compute=cluster, id="table")
    cluster._tables["table"] = "other-snapshot"
    with pytest.raises(ValueError, match="must belong"):
        cluster.prepare_training(table, snapshot, Mock(), None)


def test_trainer_prepares_before_stopping_at_execution_boundary(tmp_path: Path) -> None:
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
    from kumo.engine._scratch import scoped_scratch_dir
    from kumo_core.model_plan import ModelPlan

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
        begin=SimpleNamespace(remote=replica.begin),
        receive=SimpleNamespace(remote=replica.receive),
        commit=SimpleNamespace(remote=replica.commit),
    )
    monkeypatch.setattr(ray, "get", lambda value: value)
    with scoped_scratch_dir(producer.root):
        snapshot = producer.graph_snapshot(graph, {}, settings, [peer])
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
        table = producer.training_table_generate(
            _TrainingTableRequest(
                snapshot.id, "PREDICT users.target FOR EACH users.id", None
            ),
            settings,
            [peer],
        )
        assert table.id is not None
        publication.assert_not_called()
        portable.assert_not_called()
        store = Path(settings.artifact_root)
        stored = store / "training_tables" / table.id
        assert (stored / "kumo_training_table.json").is_file()
        assert not list(stored.glob("part-*.parquet"))
        assert not (store / "compute").exists()
        marker = orjson.loads((stored / "kumo_training_table.json").read_bytes())
        generated_parts = [
            path for path in table.staging_dir.rglob("*.parquet") if path.is_file()
        ]
        assert sum(path.is_file() for path in stored.rglob("*.parquet")) == len(
            generated_parts
        )
        for entry in marker["files"]:
            original = table.staging_dir / entry["name"]
            copied = replica.root / original.relative_to(producer.root)
            assert copied.read_bytes() == original.read_bytes()
        query = kumo.PredictiveQuery._from_snapshot(snapshot, table.query)
        restored = kumo.TrainingTable.load_from_connector(
            settings,
            path=f"training_tables/{table.id}",
            pquery=query,
            snapshot=snapshot,
        )
        assert restored.id == table.id and restored.tt_id == table.tt_id
        assert restored.required_stages == table.required_stages
        producer.tables[table.id] = restored
        prepared = producer.prepare(
            (snapshot.id, table.id, "first-fit"), ModelPlan(), settings, [peer]
        )
        first_extension = Path(prepared.materialized.train_table_extension_path)
        before = {
            path.relative_to(first_extension): path.read_bytes()
            for path in first_extension.rglob("*")
            if path.is_file()
        }
        next_fit = producer.prepare(
            (snapshot.id, table.id, "second-fit"), ModelPlan(), settings, [peer]
        )
        assert next_fit.materialized.train_table_extension_path != str(first_extension)
        assert before == {
            path.relative_to(first_extension): path.read_bytes()
            for path in first_extension.rglob("*")
            if path.is_file()
        }
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
def test_context_owns_only_remote_actors(
    monkeypatch: pytest.MonkeyPatch,
    workers: int | None,
    cpus: float | None,
    expected: list[tuple[str, float]],
) -> None:
    """Placement excludes the trainer and context exit preserves the Ray runtime."""
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
                {"Alive": False, "NodeID": "dead", "Resources": {"CPU": 8, "disk": 1}},
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
        resources={"disk": 1},
        **({"num_workers": workers} if workers is not None else {}),
        **({"cpus_per_worker": cpus} if cpus is not None else {}),
    )
    with cluster as active:
        assert active is cluster and cluster._active
        placements = [
            (call.kwargs["scheduling_strategy"].node_id, call.kwargs["num_cpus"])
            for call in actor_type.options.call_args_list
        ]
        assert placements == expected
        assert all(len(call.args) == 1 for call in actor_type.remote.call_args_list)
    assert not cluster._active and not cluster._actors
    assert kill.call_count == len(expected)
    shutdown.assert_not_called()


@pytest.mark.parametrize("workers,available", [(None, 0), (1, 0), (3, 2)])
def test_context_requires_enough_eligible_nodes(
    monkeypatch: pytest.MonkeyPatch, workers: int | None, available: int
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

    with (
        pytest.raises(RuntimeError, match="Not enough distinct remote"),
        kumo.ComputeCluster(num_workers=workers),
    ):
        pytest.fail("Started without enough eligible worker nodes")

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
    peer = SimpleNamespace(begin=Mock(), receive=Mock())
    peer.receive.remote = Mock(return_value="failed-download")
    monkeypatch.setattr(
        ray, "get", Mock(side_effect=[None, OSError("download failed")])
    )
    settings = kumo.MLOps(str(tmp_path / "store"))
    with pytest.raises(OSError, match="download failed"):
        transfer = _Replicator([peer], "", "publication", settings)
        from kumo.engine._training_table_io import store_prepared

        store_prepared(
            worker.root,
            (source,),
            settings,
            transfer.uploaded,
            lambda: transfer.finish(worker.root),
        )
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
        graph_settings, table_settings, fit_settings = [
            kumo.MLOps(str(tmp_path / name), store={kind}, on_error="raise")
            for name, kind in (
                ("graph-artifacts", "snapshot"),
                ("table-artifacts", "training_table"),
                ("fit-artifacts", "training_table"),
            )
        ]
        actor_type = ray.remote(num_cpus=1)(_ComputeWorker)
        actors = [actor_type.remote(str(tmp_path / "workers")) for _ in range(2)]
        cluster = kumo.ComputeCluster(num_workers=2)
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
        with kumo.mlops(graph_settings):
            snapshot = graph.snapshot(cluster=cluster)
        query = kumo.PredictiveQuery(
            graph=graph, query="PREDICT users.target FOR EACH users.id"
        )
        with kumo.mlops(table_settings):
            table = query.generate_training_table(snapshot)
        trainer = kumo.Trainer(query.suggest_model_plan(snapshot))
        with kumo.mlops(fit_settings):
            prepared = trainer.prepare_artifacts(table, snapshot=snapshot)
        for settings, kind in (
            (graph_settings, "snapshots"),
            (table_settings, "training_tables"),
            (fit_settings, "training_tables"),
        ):
            root = Path(settings.artifact_root)
            assert {path.name for path in root.iterdir()} == {kind}
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


@pytest.mark.parametrize("override", [False, True])
def test_snapshot_uses_cluster_only_when_passed_explicitly(
    graph: kumo.Graph,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: bool,
) -> None:
    """Omitting cluster builds locally; an explicit cluster receives the request."""
    from kumo.engine._scratch import scoped_scratch_dir

    cluster = Mock(spec=kumo.ComputeCluster)
    local = Mock()
    build_local = Mock(return_value=local)
    monkeypatch.setattr(kumo.Graph, "_snapshot_no_lock", build_local)
    with scoped_scratch_dir(tmp_path / "scratch"):
        assert graph.snapshot() is local
    build_local.assert_called_once()
    cluster._graph_snapshot.assert_not_called()

    settings = kumo.MLOps(str(tmp_path / "active"))
    explicit = kumo.MLOps(str(tmp_path / "explicit"))
    with kumo.mlops(settings):
        result = graph.snapshot(
            cluster=cluster,
            record_sources=False,
            **({"mlops": explicit} if override else {}),
        )

    assert result is cluster._graph_snapshot.return_value
    cluster._graph_snapshot.assert_called_once_with(
        graph,
        {"timestamp_precision": "seconds", "record_sources": False},
        explicit if override else settings,
    )
    assert build_local.call_count == 1
