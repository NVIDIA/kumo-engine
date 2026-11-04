# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ray-owned artifact production and pipelined, verified worker replicas."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self
from uuid import uuid4

from kumo_core.diskgraph.sampling_artifacts import (
    ArtifactBundle,
    ArtifactFile,
    download_artifact,
    relative_artifact_path,
)
from kumo_core.model_plan import ModelPlan
from kumo_core.training.workflow import MaterializedTrainingTask

from kumo.engine._mlops import MLOps
from kumo.engine.graph import Graph, GraphSnapshot
from kumo.engine.pquery import PredictiveQuery, TrainingTable


@dataclass(frozen=True)
class ArtifactReplica:
    """One ready worker copy of an immutable publication.

    Args:
        node_id: Ray node owning this copy.
        root: Worker-local scratch root; never a driver-local path.
        publication: Unique identity shared by every replica of this publication.
    """

    node_id: str
    root: str
    publication: str


@dataclass(frozen=True)
class PreparedTrainingArtifacts:
    """Ready graph and fit-specific table extensions, without training execution.

    Args:
        snapshot_id: Graph identity against which the table was ingested.
        table_id: Identity of the generated training rows.
        run_id: Unique fit identity, also used for its native table extension.
        replicas: Worker roots holding the identical committed publication.
        materialized: Producer-local task metadata for a future training runtime.
    """

    snapshot_id: str
    table_id: str
    run_id: str
    replicas: tuple[ArtifactReplica, ...]
    materialized: MaterializedTrainingTask


@dataclass(frozen=True)
class _TrainingTableRequest:
    """Query and graph identity sent to the worker owning the graph."""

    snapshot_id: str
    query: str
    plan: Any


class _Replicator:
    def __init__(
        self,
        peers: list[Any],
        destination: str,
        publication: str,
        settings: MLOps,
    ) -> None:
        import ray

        self.ray = ray
        self.peers = peers
        self.destination = destination
        self.publication = publication
        self.pending: list[Any] = []
        self.files: list[ArtifactFile] = []
        self.ray.get(
            [peer.begin.remote(publication, settings) for peer in peers]
        )

    def uploaded(self, directory: str, entry: ArtifactFile) -> None:
        """Queue replica downloads after upload close, without waiting for them."""
        self.files.append(entry)
        for peer in self.peers:
            self.pending.append(
                peer.receive.remote(
                    self.publication, self.destination, directory, entry
                )
            )

    def finish(self, source: Path) -> tuple[ArtifactReplica, ...]:
        """Wait for every file and commit readiness on every worker."""
        self.ray.get(self.pending)
        self.pending.clear()
        bundle = ArtifactBundle(str(source), tuple(self.files))
        return tuple(
            self.ray.get(
                [
                    peer.commit.remote(
                        self.publication, self.destination, bundle
                    )
                    for peer in self.peers
                ]
            )
        )


class _ComputeWorker:
    def __init__(self, root: str) -> None:
        import ray

        from kumo.engine._scratch import set_scratch_dir

        self.root = Path(root) / uuid4().hex
        self.root.mkdir(parents=True, exist_ok=False)
        set_scratch_dir(self.root)
        self.node_id = ray.get_runtime_context().get_node_id()
        self.snapshots: dict[str, GraphSnapshot] = {}
        self.tables: dict[str, TrainingTable] = {}
        self.received: dict[str, dict[str, ArtifactFile]] = {}
        self._stores: dict[str, MLOps] = {}

    def identity(self) -> tuple[str, str]:
        """Return placement metadata without reading any graph files."""
        return self.node_id, str(self.root)

    def begin(self, publication: str, settings: MLOps) -> None:
        """Bind storage settings to this publication before receiving its files."""
        self._stores[publication] = settings

    def receive(
        self,
        publication: str,
        destination: str,
        directory: str,
        entry: ArtifactFile,
    ) -> None:
        """Verify a downloaded object before installing it in an unpublished root."""
        from kumo.engine.connector import file_store

        base = (
            self.root / relative_artifact_path(destination)
            if destination
            else self.root
        )
        relative_artifact_path(entry.name)
        if not base.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Artifact destination escapes worker scratch")
        settings = self._stores[publication]
        filesystem, _ = file_store(
            settings.root.connector, settings.root.location("training_tables")
        )
        download_artifact(filesystem, directory, entry, base)
        self.received.setdefault(publication, {})[entry.name] = entry

    def commit(
        self, publication: str, destination: str, bundle: ArtifactBundle
    ) -> ArtifactReplica:
        """Expose a replica only after all expected files have arrived."""
        from kumo_core.diskgraph.sampling_artifacts import relocate_artifacts

        inventory = {entry.name: entry for entry in bundle.files}
        if self.received.get(publication, {}) != inventory:
            raise RuntimeError(
                "Replica has an incomplete publication inventory"
            )
        root = (
            self.root / relative_artifact_path(destination)
            if destination
            else self.root
        )
        relocate_artifacts(bundle, root, destination_root=self.root)
        self.received.pop(publication, None)
        self._stores.pop(publication)
        return ArtifactReplica(self.node_id, str(self.root), publication)

    def graph_snapshot(
        self,
        graph: Graph,
        options: dict[str, Any],
        settings: MLOps,
        peers: list[Any],
    ) -> GraphSnapshot:
        """Build once and reuse the MLOps exporter to stream files to replicas."""
        from kumo.engine._snapshot_io import store

        snapshot = graph.snapshot(**options)
        destination = snapshot.index_path.relative_to(self.root).as_posix()
        publication = uuid4().hex
        transfer = _Replicator(peers, destination, publication, settings)

        def finish() -> None:
            transfer.finish(self.root)

        store(
            snapshot,
            settings,
            on_file_uploaded=transfer.uploaded,
            on_files_uploaded=finish,
        )
        self.snapshots[snapshot.id] = snapshot
        return snapshot

    def training_table_generate(
        self, request: _TrainingTableRequest, settings: MLOps, peers: list[Any]
    ) -> TrainingTable:
        """Generate rows on their graph owner, export them, and replicate staging."""
        snapshot = self.snapshots[request.snapshot_id]
        table = PredictiveQuery._from_snapshot(
            snapshot, request.query
        ).generate_training_table(
            snapshot, request.plan, mlops=replace(settings, store=False)
        )
        publication = uuid4().hex
        destination = table.staging_dir.relative_to(self.root).as_posix()
        transfer = _Replicator(peers, destination, publication, settings)

        def finish() -> None:
            transfer.finish(self.root)

        table._store(
            settings,
            on_file_uploaded=transfer.uploaded,
            on_files_uploaded=finish,
        )
        assert table.id is not None
        self.tables[table.id] = table
        return table

    def prepare(
        self,
        identity: tuple[str, str, str],
        model_plan: ModelPlan,
        settings: MLOps,
        peers: list[Any],
    ) -> PreparedTrainingArtifacts:
        """Ingest generated rows once and replicate the complete fitted extension."""
        from kumo_core.training.preparation import prepare_training_task
        from kumo_core.training.table import GeneratedTrainTableResult
        from kumo_core.training.workflow import (
            PreparedGraph,
            TrainingWorkflowRequest,
        )

        from kumo.engine._training_table_io import store_prepared

        snapshot_id, table_id, run_id = identity
        snapshot = self.snapshots[snapshot_id]
        table = self.tables[table_id]
        output = self.root / "fits" / run_id
        request = TrainingWorkflowRequest.from_spec(
            graph=snapshot.graph_config,
            predictive_query=table.query,
            prepared_graph=PreparedGraph(
                snapshot_id, str(snapshot.index_path)
            ),
            run_id=run_id,
            scratch_dir=str(output),
            artifact_dir=str(output),
            model_plan=model_plan,
            training_table_generation_plan=table.generation_plan,
        )
        materialized = prepare_training_task(
            request,
            GeneratedTrainTableResult(
                str(table.train_table_path),
                str(table.manifest_path),
                str(table.staging_dir),
            ),
            table.required_stages,
        )
        publication = uuid4().hex
        transfer = _Replicator(peers, "", publication, settings)
        replicas: tuple[ArtifactReplica, ...] = ()

        def finish() -> None:
            nonlocal replicas
            replicas = transfer.finish(self.root)

        store_prepared(
            self.root,
            (Path(materialized.train_table_extension_path), output),
            settings,
            transfer.uploaded,
            finish,
        )
        replicas = (
            ArtifactReplica(self.node_id, str(self.root), publication),
            *replicas,
        )
        return PreparedTrainingArtifacts(
            snapshot_id, table_id, run_id, replicas, materialized
        )


class ComputeCluster:
    """Produce artifacts on disk-equipped Ray nodes using an MLOps artifact root.

    Each operation uses its method's explicit MLOps settings or the settings
    active at that call. The cluster does not retain an MLOps configuration.

    Args:
        num_workers: Number of distinct remote nodes holding each publication.
            Defaults to all eligible remote nodes present when entering the context.
        worker_root: Local disk directory available on every selected worker.
        resources: Ray custom resources required by each worker actor.
        cpus_per_worker: CPUs reserved per actor; must be positive if supplied.
            Defaults to each selected node's full Ray CPU capacity.

    Raises:
        ValueError: If settings disable required artifacts or placement is invalid.
        RuntimeError: If Ray is uninitialized, placement fails, or a replica fails.
    """

    def __init__(
        self,
        num_workers: int | None = None,
        worker_root: str = "/tmp/kumo-compute",
        *,
        resources: dict[str, float] | None = None,
        cpus_per_worker: float | None = None,
    ) -> None:
        if (
            num_workers is not None
            and (type(num_workers) is not int or num_workers < 1)
        ) or (
            cpus_per_worker is not None
            and (not math.isfinite(cpus_per_worker) or cpus_per_worker <= 0)
        ):
            raise ValueError(
                "Worker count and CPUs per worker must be positive"
            )
        if not Path(worker_root).is_absolute() or "://" in worker_root:
            raise ValueError(
                "worker_root must be an absolute worker-local directory"
            )
        self.num_workers = num_workers
        self.worker_root = worker_root
        self.resources = dict(resources or {})
        self.cpus_per_worker = cpus_per_worker
        self._actors: list[Any] = []
        self._active = False
        self._snapshots: set[str] = set()
        self._tables: dict[str, str] = {}
        self._failed = False

    def __enter__(self) -> Self:
        """Place workers on distinct live Ray nodes, excluding the driver node."""
        import ray
        from ray.util.scheduling_strategies import (
            NodeAffinitySchedulingStrategy,
        )

        if self._active:
            raise RuntimeError("ComputeCluster is already active")
        if not ray.is_initialized():
            raise RuntimeError("Initialize Ray before entering ComputeCluster")
        driver = ray.get_runtime_context().get_node_id()
        nodes = [
            node
            for node in ray.nodes()
            if node["Alive"]
            and node["NodeID"] != driver
            and node["Resources"].get("CPU", 0) > 0
            and (
                self.cpus_per_worker is None
                or node["Resources"]["CPU"] >= self.cpus_per_worker
            )
            and all(
                node["Resources"].get(k, 0) >= v
                for k, v in self.resources.items()
            )
        ]
        if not nodes or (
            self.num_workers is not None and len(nodes) < self.num_workers
        ):
            raise RuntimeError(
                "Not enough distinct remote Ray nodes with the requested resources"
            )
        actor_type = ray.remote(_ComputeWorker)
        try:
            for node in nodes[: self.num_workers]:
                self._actors.append(
                    actor_type.options(
                        num_cpus=(
                            node["Resources"]["CPU"]
                            if self.cpus_per_worker is None
                            else self.cpus_per_worker
                        ),
                        resources=self.resources,
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            node["NodeID"], soft=False
                        ),
                    ).remote(self.worker_root)
                )
            ray.get(
                [actor.identity.remote() for actor in self._actors], timeout=60
            )
        except BaseException:
            self.close()
            raise
        self._failed = False
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the worker actors owned by this cluster."""
        self.close()

    def close(self) -> None:
        """Release worker actors; preserve local files and the caller's Ray runtime."""
        import ray

        self._active = False
        for actor in self._actors:
            ray.kill(actor, no_restart=True)
        self._actors.clear()
        self._snapshots.clear()
        self._tables.clear()

    def _call(
        self,
        method: Literal[
            "graph_snapshot", "training_table_generate", "prepare"
        ],
        settings: MLOps | None,
        *args: Any,
    ) -> Any:
        import ray

        if settings is None:
            raise ValueError(
                "ComputeCluster operations require active kumo.mlops(...) settings "
                "or an explicit mlops= argument"
            )
        kind = "snapshot" if method == "graph_snapshot" else "training_table"
        if not settings.wants(kind):
            raise ValueError(
                f"ComputeCluster {method} requires MLOps storing {kind}"
            )
        if not self._actors or not self._active or self._failed:
            raise RuntimeError("ComputeCluster is closed, inactive, or failed")
        worker_settings = replace(
            settings,
            tracking=None,
            registry=None,
            on_error="raise",
        )
        try:
            return ray.get(
                getattr(self._actors[0], method).remote(
                    *args, worker_settings, self._actors[1:]
                )
            )
        except BaseException:
            self._failed = True
            raise

    def _graph_snapshot(
        self,
        graph: Graph,
        options: dict[str, Any],
        settings: MLOps | None,
    ) -> GraphSnapshot:
        result = self._call("graph_snapshot", settings, graph, options)
        self._snapshots.add(result.id)
        return replace(result, _compute=self)

    def _training_table_generate(
        self,
        query: PredictiveQuery,
        snapshot: GraphSnapshot,
        plan: Any,
        settings: MLOps | None,
    ) -> TrainingTable:
        if snapshot.id not in self._snapshots or snapshot._compute is not self:
            raise ValueError("Snapshot does not belong to this ComputeCluster")
        table = self._call(
            "training_table_generate",
            settings,
            _TrainingTableRequest(snapshot.id, query.query, plan),
        )
        self._tables[table.id] = snapshot.id
        return replace(table, _compute=self)

    def prepare_training(
        self,
        table: TrainingTable,
        snapshot: GraphSnapshot,
        model_plan: ModelPlan,
        settings: MLOps | None,
    ) -> PreparedTrainingArtifacts:
        """Ingest and replicate a fresh fit extension without executing training.

        Args:
            table: Generated rows owned by this cluster.
            snapshot: The exact graph against which the rows were generated.
            model_plan: Model settings used to resolve and ingest the task.
            settings: Effective MLOps configuration resolved for this operation.
        Returns:
            Metadata and worker locations after every replica is ready.
        Raises:
            ValueError: If graph and table identities do not match this cluster.
            RuntimeError: If the cluster is closed or any worker operation fails.
        """
        if (
            table._compute is not self
            or snapshot._compute is not self
            or self._tables.get(table.id) != snapshot.id
        ):
            raise ValueError(
                "Training table and snapshot must belong to this ComputeCluster"
            )
        return self._call(
            "prepare",
            settings,
            (snapshot.id, table.id, uuid4().hex),
            model_plan,
        )
