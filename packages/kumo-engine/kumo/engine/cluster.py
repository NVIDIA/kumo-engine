# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ray-owned artifact production and pipelined, verified worker replicas."""

from __future__ import annotations

import json
import logging
import math
import random
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal
from uuid import uuid4

from kumo_core.diskgraph.sampling_artifacts import (
    ArtifactBundle,
    ArtifactFile,
    download_artifact,
    relative_artifact_path,
    relocate_artifacts,
)
from kumo_core.model_plan import ModelPlan
from kumo_core.training.workflow import MaterializedTrainingTask
from kumoapi.model_plan import TrainingTableGenerationPlan

from kumo.engine._manifest import (
    SourceTableEntry,
    TrainingTableManifest,
    read_manifest,
    write_manifest,
)
from kumo.engine._mlops import MLOps
from kumo.engine.connector.base import Connector
from kumo.engine.graph import Graph, GraphSnapshot, _SnapshotSources
from kumo.engine.pquery import PredictiveQuery, TrainingTable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactRoot:
    """Durable cluster storage, with authentication supplied by a connector.

    Args:
        location: Local/shared path or object-store URI; connector-relative when
            a connector is supplied.
        connector: Optional file/object-store connector, including its identity.
    """

    location: str
    connector: Connector | None = None

    def _settings(self) -> MLOps:
        root = (
            self.location if self.connector is None else (self.connector, self.location)
        )
        return MLOps(root, store={"snapshot", "training_table"}, on_error="raise")


@dataclass(frozen=True)
class ComputeClusterConfig:
    """Placement for disk-equipped Ray workers initialized by ``kumo.init``.

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
        run_id: Unique fit identity, also used for its fresh native table extension.
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
    plan: TrainingTableGenerationPlan | None
    expected: _TrainingTablePublication | None = None


@dataclass(frozen=True)
class _TrainingTablePublication:
    table: TrainingTable
    snapshot_id: str
    source_root: str
    manifest: TrainingTableManifest | None


def _relocate_tree(source: Path, destination: Path, replacement: Path) -> None:
    """Rebase copied metadata while retaining its binary artifact files."""
    bundle = ArtifactBundle(
        str(source),
        tuple(
            ArtifactFile(path.relative_to(destination).as_posix(), 0, "")
            for path in destination.rglob("*")
            if (path.suffix == ".json" or path.name == "_SUCCESS") and path.is_file()
        ),
    )
    relocate_artifacts(bundle, destination, destination_root=replacement)


def _install_training_table(table: TrainingTable, target: Path) -> TrainingTable:
    source = table.staging_dir
    if source == target:
        return table
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.rename(target.with_name(f".{target.name}.{uuid4().hex}.incomplete"))
    source.rename(target)
    _relocate_tree(source, target, target)
    return replace(
        table,
        staging_dir=target,
        train_table_path=target / table.train_table_path.relative_to(source),
        manifest_path=target / table.manifest_path.relative_to(source),
    )


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
        self.ray.get([peer.begin.remote(publication, settings) for peer in peers])

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
                    peer.commit.remote(self.publication, self.destination, bundle)
                    for peer in self.peers
                ]
            )
        )


class _ComputeWorker:
    def __init__(self, root: str) -> None:
        import ray

        from kumo.engine._scratch import set_scratch_dir

        self.node_id = ray.get_runtime_context().get_node_id()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        set_scratch_dir(self.root)
        self.snapshots: dict[str, GraphSnapshot] = {}
        self.tables: dict[str, TrainingTable] = {}
        self.received: dict[str, dict[str, ArtifactFile]] = {}
        self._stores: dict[str, MLOps] = {}
        self._staged: dict[str, tuple[Path, _SnapshotSources]] = {}

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
        # Resolve authentication from the artifact root; directory supplies the path.
        filesystem, _ = file_store(
            settings.root.connector, settings.root.location("")
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
            raise RuntimeError("Replica has an incomplete publication inventory")
        root = (
            self.root / relative_artifact_path(destination)
            if destination
            else self.root
        )
        relocate_artifacts(bundle, root, destination_root=self.root)
        self.received.pop(publication, None)
        self._stores.pop(publication)
        return ArtifactReplica(self.node_id, str(self.root), publication)

    def stage_snapshot(
        self,
        graph: Graph,
        options: dict[str, Any],
    ) -> tuple[GraphSnapshot, dict[str, SourceTableEntry]]:
        """Stage sources once and return their exact index identity, without ingest.

        Args:
            graph: Graph whose connectors are readable on this worker.
            options: Timestamp precision and source-recording choices.
        Returns:
            Snapshot descriptor and source provenance; the index may not exist yet.
        Raises:
            ValueError: If the graph or timestamp precision is invalid.
            OSError: If source staging fails.
        """
        from kumo_core.diskgraph.snapshot import DiskgraphSnapshotHandle
        from kumo_core.sources.transfer import discard_destination
        from kumoapi.typing import validate_timestamp_precision

        from kumo.engine.graph import _snapshot_id, _store

        precision = options["timestamp_precision"]
        validate_timestamp_precision(precision)
        graph.validate()
        for table in graph._named_tables.values():
            table.connector.validate_scratch_dir(self.root)
        persist_dir = self.root / "staging" / uuid4().hex
        try:
            sources = graph._stage_snapshot_sources(
                persist_dir, options["record_sources"]
            )
        except BaseException:
            discard_destination(str(persist_dir))
            raise
        version = sources.data_version(precision)
        snapshot = GraphSnapshot(
            _snapshot_id(graph.id, version),
            Path(_store().path(DiskgraphSnapshotHandle(graph.id, version))),
            graph._graph_config,
            record_sources=options["record_sources"],
        )
        self._staged[snapshot.id] = persist_dir, sources
        return snapshot, sources.tables

    def discard_staging(self, snapshot_id: str) -> None:
        """Discard staged source rows after their snapshot operation completes.

        Args:
            snapshot_id: Identity returned by stage_snapshot.
        Raises:
            OSError: If staged files cannot be removed.
        """
        from kumo_core.sources.transfer import discard_destination

        staged = self._staged.pop(snapshot_id, None)
        if staged is not None:
            discard_destination(str(staged[0]))

    def _local_snapshot(self, snapshot: GraphSnapshot) -> GraphSnapshot | None:
        from kumo_core.diskgraph.snapshot import DiskgraphSnapshotHandle

        from kumo.engine.graph import _loaded, _lock, _store

        handle = DiskgraphSnapshotHandle(snapshot._graph_key, snapshot._data_version)
        with _lock(snapshot._graph_key):
            result = _loaded(_store(), handle)
        if result is None:
            return None
        return replace(snapshot, index_path=Path(result.index_path), _compute=None)

    def has_snapshot(self, snapshot: GraphSnapshot) -> bool:
        """Check a worker's persistent index cache for the exact snapshot version.

        Args:
            snapshot: Snapshot identity requested by the coordinator.
        Returns:
            Whether a complete, valid local index exists.
        """
        return self._local_snapshot(snapshot) is not None

    def download_snapshot(
        self,
        snapshot: GraphSnapshot,
        settings: MLOps,
        source_tables: dict[str, SourceTableEntry],
    ) -> GraphSnapshot:
        """Reuse a valid local index or download its committed durable export.

        Args:
            snapshot: Exact graph/data-version descriptor.
            settings: Cluster storage configuration.
            source_tables: Current source provenance for the requested privacy setting.
        Returns:
            Worker-local snapshot ready for table generation.
        Raises:
            OSError: If download fails.
            ValueError: If the export is corrupt or has another identity.
        """
        from kumo.engine._mlops import snapshot_path
        from kumo.engine.graph import _write_snapshot_manifest

        local = self._local_snapshot(snapshot)
        if local is None:
            local = GraphSnapshot.load_from_connector(
                settings, path=snapshot_path(snapshot)
            )
            if local.id != snapshot.id or local._data_version != snapshot._data_version:
                raise ValueError("Downloaded snapshot has a different data version")
            output = Path(
                json.loads((local.index_path / "ingest_stats.json").read_bytes())[
                    "stats"
                ]["output"]
            )
            _relocate_tree(output.parents[2], local.index_path, self.root)
        _write_snapshot_manifest(
            local,
            snapshot.id,
            source_tables,
            snapshot.record_sources is not False,
            fresh=False,
        )
        local = replace(local, record_sources=snapshot.record_sources)
        self.snapshots[local.id] = local
        return local

    def produce_snapshot(
        self,
        request: tuple[Graph, dict[str, Any], GraphSnapshot],
        settings: MLOps,
        source_tables: dict[str, SourceTableEntry],
    ) -> GraphSnapshot:
        """Reuse or build the selected version, then commit it to cluster storage.

        Args:
            request: Graph, snapshot options and staged version descriptor.
            settings: Cluster storage configuration.
            source_tables: Current source provenance for the requested privacy setting.
        Returns:
            Produced worker-local snapshot after durable upload completes.
        Raises:
            RuntimeError: If production would require restaging on another worker.
            OSError: If production or upload fails.
        """
        from kumo.engine._snapshot_io import store
        from kumo.engine.graph import _lock, _write_snapshot_manifest

        graph, options, descriptor = request
        snapshot = self._local_snapshot(descriptor)
        if snapshot is None:
            if descriptor.id not in self._staged:
                raise RuntimeError(
                    "Selected producer lost its cached index; retry the snapshot"
                )
            persist_dir, sources = self._staged[descriptor.id]
            with _lock(graph.id):
                snapshot = graph._snapshot_no_lock(
                    graph.id, persist_dir, **options, timed=True, prepared=sources
                )
        else:
            _write_snapshot_manifest(
                snapshot,
                snapshot.id,
                source_tables,
                snapshot.record_sources is not False,
                fresh=False,
            )
        store(snapshot, settings)
        self.snapshots[snapshot.id] = snapshot
        return snapshot

    def describe_training_table(
        self, request: _TrainingTableRequest
    ) -> _TrainingTablePublication:
        """Resolve the existing table ID and generation settings without producing rows.

        Args:
            request: Graph snapshot, predictive query and requested generation plan.
        Returns:
            A descriptor keyed by the standard SDK training-table identity.
        Raises:
            ValueError: If the query or generation settings are invalid.
        """
        snapshot = self.snapshots[request.snapshot_id]
        query = PredictiveQuery._from_snapshot(snapshot, request.query)
        table = query._describe_training_table(snapshot, request.plan)
        return _TrainingTablePublication(table, snapshot.id, str(self.root), None)

    def _local_training_table(
        self, publication: _TrainingTablePublication
    ) -> TrainingTable | None:
        from kumo_core.artifacts import PROVENANCE_FILE
        from kumo_core.training.export import read_spec, stage_parts

        original = publication.table
        relative = original.staging_dir.relative_to(publication.source_root)
        staging = self.root / relative_artifact_path(relative.as_posix())
        manifest = read_manifest(staging / PROVENANCE_FILE, TrainingTableManifest)
        if manifest is None or manifest.id != original.id:
            return None
        table = replace(
            original,
            staging_dir=staging,
            train_table_path=staging / "tt.parquet",
            manifest_path=staging / "manifest.json",
            _compute=None,
        )
        if not table.train_table_path.exists() or not table.manifest_path.is_file():
            return None
        try:
            parts = stage_parts(read_spec(str(staging)))
            if not all(
                (staging / relative_artifact_path(name)).is_file()
                for paths in parts.values()
                for name in paths
            ):
                return None
        except FileNotFoundError:
            return None
        except (KeyError, TypeError, ValueError) as error:
            logger.warning(
                "Ignoring incomplete training table %s: %s", original.id, error
            )
            return None
        assert table.id is not None
        self.tables[table.id] = table
        return table

    def has_training_table(self, publication: _TrainingTablePublication) -> bool:
        """Check for completed local staging under the standard table ID.

        Args:
            publication: Expected training-table identity and generation settings.
        Returns:
            Whether the table's provenance, ingestion specification and parts exist.
        """
        return self._local_training_table(publication) is not None

    def training_table_generate(
        self,
        request: _TrainingTableRequest,
        settings: MLOps,
    ) -> TrainingTable:
        """Produce missing rows and persist staging for owner-local ingestion."""
        publication = request.expected or self.describe_training_table(request)
        snapshot = self.snapshots[request.snapshot_id]
        table = self._local_training_table(publication)
        if table is None:
            query = PredictiveQuery._from_snapshot(snapshot, request.query)
            table = query.generate_training_table(
                snapshot, request.plan, mlops=replace(settings, store=False)
            )
            if table.id != publication.table.id:
                raise RuntimeError(
                    "Generated training-table identity differs from its plan"
                )
            target = self.root / relative_artifact_path(
                publication.table.staging_dir.relative_to(
                    publication.source_root
                ).as_posix()
            )
            table = _install_training_table(table, target)
        table._store(settings)
        assert table.id is not None
        self.tables[table.id] = table
        return table

    def download_training_table(
        self,
        publication: _TrainingTablePublication,
        settings: MLOps,
    ) -> TrainingTable:
        """Restore local ingestion staging from portable Parquet and its descriptor.

        Args:
            publication: Original query, generation settings, identity and provenance.
            settings: Cluster storage holding the completed portable export.
        Returns:
            Ready worker-local table, reusing a valid staging copy when available.
        Raises:
            ValueError: If downloaded rows or the generation settings are invalid.
            OSError: If the export is incomplete or cannot be downloaded.
        """
        import pyarrow.parquet as parquet
        from kumo_core.artifacts import PROVENANCE_FILE
        from kumo_core.io.json import from_json_dict, to_json_dict
        from kumo_core.sources.object_store import _source_version
        from kumo_core.training.custom_table import contract
        from kumoapi.model_plan import TrainingTableGenerationPlan
        from kumoapi.train import TrainingTableSpec

        from kumo.engine import _provenance as provenance
        from kumo.engine.connector import file_store
        from kumo.engine.pquery.training_table import (
            _listed,
            _record_sources,
            _signature,
        )
        from kumo.engine.trainer.trainer import _require_success, fetch_parquet

        original = publication.table
        assert original.id is not None
        cached = self._local_training_table(publication)
        if cached is not None:
            return cached
        snapshot = self.snapshots[publication.snapshot_id]
        path = settings.root.location(f"training_tables/{original.id}")
        filesystem, root = file_store(settings.root.connector, path)
        _require_success(filesystem, root, path)
        files = _listed(filesystem, root)
        parts = [info for info in files if info.path.endswith(".parquet")]
        version = _source_version(root, parts)
        query = PredictiveQuery._from_snapshot(snapshot, original.query)
        plan = from_json_dict(
            to_json_dict(original.generation_plan), TrainingTableGenerationPlan
        )
        relative = original.staging_dir.relative_to(publication.source_root)
        target = self.root / relative_artifact_path(relative.as_posix())
        with TemporaryDirectory(dir=self.root) as temporary:
            rows = Path(temporary) / "rows"
            fetch_parquet(
                filesystem,
                root,
                rows,
                parts,
            )
            if _signature(_listed(filesystem, root)) != _signature(files):
                raise RuntimeError(
                    "Portable training-table export changed while loading; retry"
                )
            weight = contract()["weight_output_column"]
            schema = parquet.read_schema(next(rows.glob("*.parquet")))
            restored = TrainingTable._from_rows(
                rows,
                TrainingTableSpec(
                    split_col="SPLIT",
                    weight_col=weight if weight in schema.names else None,
                ),
                pq=query,
                snapshot=snapshot,
                plan=plan,
                mlops=replace(settings, store=False),
                version=version,
                source=provenance.stored_table(
                    settings.root.connector,
                    path,
                    version,
                    record_sources=_record_sources(snapshot),
                ),
            )
            restored = _install_training_table(restored, target)
        restored = replace(
            restored,
            id=original.id,
            generation_plan=original.generation_plan,
            required_stages=original.required_stages,
        )
        manifest = publication.manifest or restored.manifest
        if manifest is not None:
            write_manifest(target / PROVENANCE_FILE, replace(manifest, id=original.id))
        self.tables[original.id] = restored
        return restored

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
            prepared_graph=PreparedGraph(snapshot_id, str(snapshot.index_path)),
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
    """Remote worker pool owned by ``kumo.init``.

    Args:
        config: Worker placement and persistent local disk configuration.
        artifact_root: Durable storage shared by all workers.
    """

    def __init__(
        self, config: ComputeClusterConfig, artifact_root: ArtifactRoot
    ) -> None:
        self.config = config
        self._settings = artifact_root._settings()
        self._actors: list[Any] = []
        self._active = False
        self._snapshots: dict[str, Any] = {}
        self._tables: dict[str, str] = {}
        self._table_owners: dict[str, Any] = {}
        self._failed = False
        self._artifact_lock = threading.Lock()

    def start(self) -> None:
        """Place workers on distinct live Ray nodes, excluding the driver node.

        Raises:
            RuntimeError: If Ray is uninitialized or placement fails.
        """
        import ray
        from ray.util.scheduling_strategies import (
            NodeAffinitySchedulingStrategy,
        )

        if self._active:
            raise RuntimeError("ComputeCluster is already active")
        if not ray.is_initialized():
            raise RuntimeError("Initialize Ray before kumo.init(cluster_config=...)")
        driver = ray.get_runtime_context().get_node_id()
        nodes = [
            node
            for node in ray.nodes()
            if node["Alive"]
            and node["NodeID"] != driver
            and node["Resources"].get("CPU", 0) > 0
            and (
                self.config.cpus_per_worker is None
                or node["Resources"]["CPU"] >= self.config.cpus_per_worker
            )
            and all(
                node["Resources"].get(k, 0) >= v
                for k, v in (self.config.resources or {}).items()
            )
        ]
        if not nodes or (
            self.config.num_workers is not None and len(nodes) < self.config.num_workers
        ):
            raise RuntimeError(
                "Not enough distinct remote Ray nodes with the requested resources"
            )
        actor_type = ray.remote(_ComputeWorker)
        try:
            for node in nodes[: self.config.num_workers]:
                self._actors.append(
                    actor_type.options(
                        num_cpus=(
                            node["Resources"]["CPU"]
                            if self.config.cpus_per_worker is None
                            else self.config.cpus_per_worker
                        ),
                        resources=self.config.resources or {},
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            node["NodeID"], soft=False
                        ),
                    ).remote(self.config.worker_root)
                )
            ray.get([actor.identity.remote() for actor in self._actors], timeout=60)
        except BaseException:
            self.close()
            raise
        self._failed = False
        self._active = True

    def close(self) -> None:
        """Release worker actors; preserve local files and the caller's Ray runtime."""
        import ray

        self._active = False
        for actor in self._actors:
            ray.kill(actor, no_restart=True)
        self._actors.clear()
        self._snapshots.clear()
        self._tables.clear()
        self._table_owners.clear()

    def _require_active(self) -> None:
        if not self._actors or not self._active or self._failed:
            raise RuntimeError("ComputeCluster is closed, inactive, or failed")

    def _call(
        self,
        method: Literal["training_table_generate", "prepare"],
        owner: Any,
        *args: Any,
    ) -> Any:
        import ray

        self._require_active()
        try:
            kwargs = (
                {"peers": [peer for peer in self._actors if peer != owner]}
                if method == "prepare"
                else {}
            )
            return ray.get(
                getattr(owner, method).remote(
                    *args,
                    self._settings,
                    **kwargs,
                )
            )
        except BaseException:
            self._failed = True
            raise

    def _graph_snapshot(self, graph: Graph, options: dict[str, Any]) -> GraphSnapshot:
        with self._artifact_lock:
            return self._coordinate_snapshot(graph, options)

    def _coordinate_snapshot(
        self, graph: Graph, options: dict[str, Any]
    ) -> GraphSnapshot:
        import ray

        from kumo.engine._mlops import snapshot_path
        from kumo.engine._snapshot_io import read

        self._require_active()
        staging_worker = random.choice(self._actors)
        descriptor = None
        try:
            descriptor, source_tables = ray.get(
                staging_worker.stage_snapshot.remote(graph, options)
            )
            location = self._settings.root.location(snapshot_path(descriptor))
            try:
                _, manifest = read(self._settings.root.connector, location)
            except FileNotFoundError:
                manifest = None
            result = None
            if manifest is not None:
                if (
                    manifest["snapshot_id"] != descriptor.id
                    or manifest["data_version"] != descriptor._data_version
                ):
                    raise ValueError("Stored snapshot has a different data version")
                owner = staging_worker
            else:
                cached = ray.get(
                    [peer.has_snapshot.remote(descriptor) for peer in self._actors]
                )
                owner = next(
                    (peer for peer, ready in zip(self._actors, cached) if ready),
                    staging_worker,
                )
                result = ray.get(
                    owner.produce_snapshot.remote(
                        (graph, options, descriptor), self._settings, source_tables
                    )
                )
            receivers = (
                self._actors
                if result is None
                else [peer for peer in self._actors if peer != owner]
            )
            replicas = ray.get(
                [
                    peer.download_snapshot.remote(
                        descriptor, self._settings, source_tables
                    )
                    for peer in receivers
                ]
            )
            if result is None:
                result = replicas[self._actors.index(owner)]
            self._snapshots[result.id] = owner
            return replace(result, _compute=self)
        except BaseException:
            self._failed = True
            raise
        finally:
            if descriptor is not None:
                try:
                    ray.get(staging_worker.discard_staging.remote(descriptor.id))
                except (ray.exceptions.RayError, OSError):
                    if not self._failed:
                        self._failed = True
                        raise
                    logger.error(
                        "Could not clean snapshot source staging", exc_info=True
                    )

    def _training_table_generate(
        self,
        query: PredictiveQuery,
        snapshot: GraphSnapshot,
        plan: TrainingTableGenerationPlan | None,
    ) -> TrainingTable:
        if snapshot.id not in self._snapshots or snapshot._compute is not self:
            raise ValueError("Snapshot does not belong to this ComputeCluster")
        with self._artifact_lock:
            self._require_active()
            import ray
            from kumo_core.artifacts import PROVENANCE_FILE
            from pyarrow.fs import FileType

            from kumo.engine._snapshot_io import _read_json
            from kumo.engine.connector import file_store

            default = self._snapshots[snapshot.id]
            request = _TrainingTableRequest(snapshot.id, query.query, plan)
            try:
                publication = ray.get(default.describe_training_table.remote(request))
                table_id = publication.table.id
                filesystem, root = file_store(
                    self._settings.root.connector,
                    self._settings.root.location(f"training_tables/{table_id}"),
                )
                if filesystem.get_file_info(f"{root}/_SUCCESS").type is FileType.File:
                    metadata = _read_json(filesystem, f"{root}/{PROVENANCE_FILE}")
                    manifest = (
                        None
                        if metadata is None
                        else TrainingTableManifest.from_dict(metadata)
                    )
                    if manifest is not None and manifest.id != table_id:
                        raise ValueError(
                            "Stored training-table identity differs from its path"
                        )
                    publication = replace(publication, manifest=manifest)
                    owner = default
                    table = ray.get(
                        owner.download_training_table.remote(
                            publication, self._settings
                        )
                    )
                else:
                    cached = ray.get(
                        [
                            actor.has_training_table.remote(publication)
                            for actor in self._actors
                        ]
                    )
                    owner = next(
                        (actor for actor, ready in zip(self._actors, cached) if ready),
                        default,
                    )
                    table = self._call(
                        "training_table_generate",
                        owner,
                        replace(request, expected=publication),
                    )
                self._tables[table.id] = snapshot.id
                self._table_owners[table.id] = owner
                return replace(table, _compute=self)
            except BaseException:
                self._failed = True
                raise

    def prepare_training(
        self,
        table: TrainingTable,
        snapshot: GraphSnapshot,
        model_plan: ModelPlan,
    ) -> PreparedTrainingArtifacts:
        """Ingest and replicate a fresh fit extension without executing training.

        Args:
            table: Generated rows owned by this cluster.
            snapshot: The exact graph against which the rows were generated.
            model_plan: Current model settings, resolved separately for each fit.
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
        with self._artifact_lock:
            self._require_active()
            try:
                return self._call(
                    "prepare",
                    self._table_owners[table.id],
                    (snapshot.id, table.id, uuid4().hex),
                    model_plan,
                )
            except BaseException:
                self._failed = True
                raise
