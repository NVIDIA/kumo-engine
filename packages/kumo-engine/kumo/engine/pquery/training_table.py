# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.model_plan import TrainingTableGenerationPlan
from kumoapi.model_plan import \
    TrainingTableGenerationPlan as PublicGenerationPlan
from kumoapi.train import TrainingTableSpec

from kumo.engine import _provenance as provenance
from kumo.engine._manifest import (
    SourceTableEntry,
    TrainingTableManifest,
    TrainingTableParents,
    read_manifest,
    stamp,
    write_manifest,
)
from kumo.engine.connector.source_table import SourceTable
from kumo.engine.graph import GraphSnapshot

if TYPE_CHECKING:
    from kumo.engine.trainer.output_config import TrainingTableOutputConfig

STAGING_MANIFEST_FILE = PROVENANCE_FILE
EXPORT_MANIFEST_FILE = PROVENANCE_FILE
_RANDOM_SPLITS = ('RandomSplit', 'RandomLinkSplit')


@dataclass(frozen=True)
class TrainingTable:
    r"""An immutable training table materialized from a predictive query.

    `Trainer.fit` consumes the generated paths, `required_stages`, and
    `generation_plan` without regenerating the table. It independently builds
    task metadata from the graph snapshot selected for that fit. This keeps
    the physical rows, anchors, splits, and effective forecast settings owned
    by the generated table rather than by a graph-bound training plan.
    """

    query: str
    train_table_path: Path
    manifest_path: Path
    staging_dir: Path
    #: Split artifacts that must exist in the generated staging directory.
    required_stages: tuple[str, ...]
    #: Effective generation settings carried forward into ``Trainer.fit``.
    generation_plan: TrainingTableGenerationPlan
    #: The artifact id: the same snapshot, query, plan, split seed and
    #: engine build give the same id. None for a hand-built table.
    id: Optional[str] = field(default=None, compare=False)

    @property
    def tt_id(self) -> str:
        """The id DiskGraph assigned to this table at generation."""
        from kumo_core.training.export import read_spec

        return read_spec(str(self.staging_dir))['tt_id']

    @property
    def manifest(self) -> Optional[TrainingTableManifest]:
        """Lineage recorded at generation; None for older tables."""
        return read_manifest(self.staging_dir / STAGING_MANIFEST_FILE,
                             TrainingTableManifest)

    def export(self, output_config: "TrainingTableOutputConfig") -> str:
        r"""Write the staged rows with a ``SPLIT`` column (``0`` train, ``1``
        validation, ``2`` test) for :meth:`from_source_table`. Returns the
        destination."""
        import tempfile

        from kumo_core.training.export import (
            export_train_table,
            identity,
            read_spec,
        )

        from kumo.engine._scratch import get_scratch_dir

        connector = output_config.connector
        write_files = getattr(connector.sink_backend, "write_files", None)
        if not callable(write_files):
            raise NotImplementedError(
                f"{type(connector).__name__} cannot export a training table")
        export_id = identity(read_spec(str(self.staging_dir)))
        root = get_scratch_dir()
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as output:
            files = export_train_table(str(self.staging_dir), output)
            return write_files(connector, output_config.table_name, files,
                               output_config.write_mode, export_id=export_id,
                               **self._export_sidecars(connector,
                                                       output_config))

    def _export_sidecars(self, connector, output_config) -> Dict[str, Any]:
        from kumo_core.sources import layout

        from kumo.engine.connector import is_file_destination

        manifest = self.staging_dir / STAGING_MANIFEST_FILE
        name = output_config.table_name.rstrip('/')
        try:
            if (not is_file_destination(connector, output_config.table_name)
                    or layout.file_type(name) == layout.CSV
                    or not manifest.is_file()):
                return {}
            return {'sidecars': {EXPORT_MANIFEST_FILE: manifest.read_bytes()}}
        except Exception as error:
            warnings.warn(
                f"Exporting without {EXPORT_MANIFEST_FILE}: {error}",
                UserWarning, stacklevel=3)
            return {}

    @classmethod
    def load_from_connector(cls, connector: Any, *, path: str,
                            pquery: "PredictiveQuery",  # noqa: F821
                            snapshot: GraphSnapshot) -> "TrainingTable":
        r"""Materialize a table written by :meth:`export` (a directory with a
        ``SPLIT`` column) against ``snapshot``, as :meth:`from_source_table`
        with ``split_col='SPLIT'``. A `kumo.MLOps` ``connector`` reads
        ``path`` under its artifact root (``training_tables/<id>``)."""
        from kumo.engine._mlops import on_root

        connector, path = on_root(connector, path)
        return cls.from_source_table(connector[path],
                                     TrainingTableSpec(split_col='SPLIT'),
                                     pq=pquery, snapshot=snapshot)

    def _store(self, settings: Any) -> None:
        from pyarrow.fs import FileType

        from kumo.engine.connector import file_store
        from kumo.engine.trainer.output_config import TrainingTableOutputConfig

        path = settings.root.location(f'training_tables/{self.id}/')
        try:
            settings.locate(self.staging_dir / STAGING_MANIFEST_FILE,
                            TrainingTableManifest, f'training_tables/{self.id}')
            filesystem, root = file_store(settings.root.connector, path)
            if filesystem.get_file_info(
                    f'{root}/_SUCCESS').type is not FileType.File:
                self.export(
                    TrainingTableOutputConfig(settings.root.connector, path))
        except Exception as error:
            raise RuntimeError(
                f"Could not store training table {self.id} under the "
                f"artifact root: {error}") from error

    @classmethod
    def from_source_table(
        cls,
        source_table: SourceTable,
        spec: TrainingTableSpec,
        *,
        pq: "PredictiveQuery",  # noqa: F821
        snapshot: GraphSnapshot,
        plan: Optional[PublicGenerationPlan] = None,
        mlops: Optional[Any] = None,
    ) -> "TrainingTable":
        r"""Materialize caller-supplied labeled rows as a training table.

        The supplied table must contain the query's standard ``ENTITY1``,
        ``TARGET1`` and, for temporal tasks, ``TIMESTAMP`` columns. ``spec``
        may name a weight column and a manual split column. Without a manual
        split column, the supplied ``plan`` split (or the task's normal
        inferred split) assigns the caller's rows to stages.

        ``snapshot`` supplies the index used to validate and normalize the
        rows. The returned object retains only the materialized rows and
        generation metadata; :meth:`Trainer.fit` independently selects its
        graph through its own required ``snapshot=`` argument. With MLOps on
        (``mlops=`` or a ``kumo.mlops`` block), the table is also stored
        under the artifact root.
        """
        # Local imports keep the immutable descriptor cheap to import and
        # avoid pquery -> trainer -> pquery cycles.
        from kumo_core.io.json import from_json_dict
        from kumo_core.pquery import build_native_pquery_task_plan
        from kumo_core.training.table import (
            PreparedCustomTrainingTable,
            generate_custom_train_table,
        )
        from kumo_core.training.workflow import (
            PreparedGraph,
            TrainingWorkflowRequest,
        )
        from kumoapi.json_serde import to_json_dict

        from kumo.engine._embeddings import no_encoder_inference_warnings
        from kumo.engine._scratch import new_run_id, train_table_scratch_dir
        from kumo.engine.pquery._custom_source import (
            stage_source_table,
            validate_training_source,
        )
        from kumo.engine.pquery.predictive_query import PredictiveQuery

        if not isinstance(pq, PredictiveQuery):
            raise TypeError("pq must be a kumo.engine PredictiveQuery")
        if not isinstance(spec, TrainingTableSpec):
            raise TypeError("spec must be a TrainingTableSpec")
        pq._require_matching_snapshot(snapshot)

        public_plan = plan if plan is not None else PublicGenerationPlan()
        split_column = (spec.split_col or "").strip()
        if split_column:
            public_plan = replace(
                public_plan,
                split=(f"ManualSplit(col={split_column!r}, train_value=0, "
                       "val_value=1, test_value=2)"),
            )
        run_id = new_run_id()
        run_dir = train_table_scratch_dir(run_id)
        rows_path, version = stage_source_table(source_table,
                                                run_dir / "source-table")
        generation_plan = from_json_dict(
            to_json_dict(public_plan),
            TrainingTableGenerationPlan,
        )
        workflow = TrainingWorkflowRequest.from_spec(
            graph=snapshot.graph_config,
            predictive_query=pq.query,
            prepared_graph=PreparedGraph(
                index_id=snapshot.id,
                index_path=str(snapshot.index_path),
            ),
            run_id=run_id,
            scratch_dir=str(run_dir),
            artifact_dir=str(run_dir),
            training_table_generation_plan=generation_plan,
        )
        validate_training_source(rows_path, workflow.query, workflow.graph,
                                 spec)
        with no_encoder_inference_warnings():
            task_plan = build_native_pquery_task_plan(
                query=workflow.query,
                graph=workflow.graph,
                training_table_generation_plan=generation_plan,
                run_id=workflow.run_id,
                graph_index_path=workflow.prepared_graph.index_path,
                scratch_dir=workflow.scratch_dir,
                artifact_dir=workflow.artifact_dir,
                graph_index_id=workflow.prepared_graph.index_id,
                supplied_rows=True,
            )
        generated = generate_custom_train_table(
            task_plan.train_table,
            PreparedCustomTrainingTable(
                parquet_path=str(rows_path),
                weight_column=spec.weight_col,
                timestamp_unit=spec.timestamp_unit,
            ),
        )
        if workflow.query.query_type == "temporal":
            from kumo_core.model_plan import inferred_or_missing_type_to_none
            value = inferred_or_missing_type_to_none(
                generation_plan.timeframe_step)
            offset = workflow.query.query.get_combined_date_offset_range()
            generation_plan = replace(
                generation_plan,
                timeframe_step=(int(value) if value is not None else
                                int(offset.end) - int(offset.start or 0)),
            )
        if workflow.query.task_type == "forecasting":
            generation_plan = replace(
                generation_plan,
                forecast_length=workflow.query.forecast_length,
                lag_timesteps=task_plan.train_table.lag_timesteps,
                year_over_year=("yoy_offsets_secs" in
                                (task_plan.train_table.compiler_options or {})),
            )
        table = cls(
            query=pq.query,
            train_table_path=Path(generated.train_table_path),
            manifest_path=Path(generated.manifest_path),
            staging_dir=Path(generated.staging_dir),
            required_stages=task_plan.task_dataset.required_stages,
            generation_plan=generation_plan,
        )
        return _finish(
            table, snapshot, task_plan.train_table, mlops,
            source_table=provenance.source_table(
                source_table.connector, source_table.table_name,
                source_table.sql, version,
                record_sources=_record_sources(snapshot)),
            source_version=version)


def _record_sources(snapshot: GraphSnapshot) -> bool:
    if snapshot.record_sources is not None:
        return snapshot.record_sources
    manifest = snapshot.manifest
    return manifest is not None and manifest.record_sources is True


def _finish(table: TrainingTable, snapshot: GraphSnapshot, request: Any,
            mlops: Optional[Any], *,
            source_table: Optional[SourceTableEntry] = None,
            source_version: Any = None) -> TrainingTable:
    from kumo.engine._mlops import resolve

    seed = None
    if getattr(request, 'split_family', None) in _RANDOM_SPLITS:
        seed = (request.split_config or {}).get('rnd_seed') or 0
    table = replace(table, id=_table_id(table, snapshot, seed, source_version,
                                        custom=source_table is not None))
    _record_manifest(table, snapshot, request, seed, source_table=source_table)
    settings = resolve(mlops)
    if settings is not None and settings.wants('training_table'):
        from kumo.engine._mlops import raise_on_error

        try:
            table._store(settings)
        except Exception as error:
            raise_on_error(settings, error, 'Training table generation',
                           stacklevel=3)
    return table


def _table_id(table: TrainingTable, snapshot: GraphSnapshot,
              seed: Optional[int], source_version: Any, *,
              custom: bool) -> str:
    from kumo_core.io.json import to_json_dict

    from kumo.engine._provenance import environment

    if custom and source_version is None:
        from kumo.engine._scratch import new_run_id
        return f'tt-{new_run_id()}'
    build = environment()
    key = json.dumps([
        snapshot.id, table.query,
        to_json_dict(table.generation_plan), seed,
        None if source_version is None else str(source_version),
        build.kumo_engine, build.build_commit, build.diskgraph,
        build.diskgraph_build_commit
    ], sort_keys=True, default=str)
    return f'tt-{hashlib.sha256(key.encode()).hexdigest()[:32]}'


def _record_manifest(table: TrainingTable, snapshot: GraphSnapshot,
                     request: Any, seed: Optional[int], *,
                     source_table: Optional[SourceTableEntry] = None) -> None:
    from kumo.engine._manifest import embedded, stub

    path = table.staging_dir / STAGING_MANIFEST_FILE
    try:
        from kumo_core.io.json import to_json_dict

        parent = snapshot.manifest
        write_manifest(
            path,
            TrainingTableManifest(
                id=table.id,
                parents=TrainingTableParents(
                    snapshot=embedded(parent) if parent is not None else
                    stub('snapshot', snapshot.id),
                    source_table=source_table),
                query=table.query,
                generation_plan=to_json_dict(table.generation_plan),
                task_type=request.task_type,
                split_seed=seed,
                **stamp(),
            ))
    except Exception as error:
        warnings.warn(
            f"Could not write the training-table manifest at {path}: "
            f"{error}", UserWarning, stacklevel=3)
