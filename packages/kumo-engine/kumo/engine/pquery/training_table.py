# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from kumo_core.model_plan import TrainingTableGenerationPlan
from kumoapi.model_plan import \
    TrainingTableGenerationPlan as PublicGenerationPlan
from kumoapi.train import TrainingTableSpec

from kumo.engine._manifest import new_manifest, read_manifest, write_manifest
from kumo.engine.connector.source_table import SourceTable
from kumo.engine.graph import GraphSnapshot

if TYPE_CHECKING:
    from kumo.engine.trainer.output_config import TrainingTableOutputConfig

logger = logging.getLogger(__name__)

STAGING_MANIFEST_FILE = 'training-table-manifest.json'
EXPORT_MANIFEST_FILE = '_kumo_manifest.json'
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

    @property
    def tt_id(self) -> str:
        """The id DiskGraph assigned to this table at generation."""
        from kumo_core.training.export import read_spec

        return read_spec(str(self.staging_dir))['tt_id']

    @property
    def manifest(self) -> Optional[Dict[str, Any]]:
        """Lineage recorded at generation; None for older tables."""
        return read_manifest(self.staging_dir / STAGING_MANIFEST_FILE)

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
            payload = self._export_manifest(connector, output_config, files)
            extra = ({} if payload is None else {
                'sidecars': {
                    EXPORT_MANIFEST_FILE: payload
                }
            })
            return write_files(connector, output_config.table_name, files,
                               output_config.write_mode, export_id=export_id,
                               **extra)

    def _export_manifest(self, connector, output_config,
                         files: Sequence[str]) -> Optional[bytes]:
        from kumo_core.object_store_sink import ObjectStoreSinkBackend
        from kumo_core.sources import layout

        name = output_config.table_name
        if (not isinstance(connector.sink_backend, ObjectStoreSinkBackend)
                or not (name.endswith('/') or len(files) > 1)
                or layout.file_type(name.rstrip('/')) not in (
                    None, layout.PARQUET)):
            logger.debug('Training-table manifests are only exported to '
                         'Parquet directories on files or object stores.')
            return None
        manifest = self.manifest
        if manifest is None:
            return None
        return (json.dumps(manifest, indent=2, sort_keys=True) +
                '\n').encode('utf-8')

    @classmethod
    def from_source_table(
        cls,
        source_table: SourceTable,
        spec: TrainingTableSpec,
        *,
        pq: "PredictiveQuery",  # noqa: F821
        snapshot: GraphSnapshot,
        plan: Optional[PublicGenerationPlan] = None,
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
        graph through its own required ``snapshot=`` argument.
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
        rows_path = stage_source_table(source_table, run_dir / "source-table")
        lineage = _source_lineage(source_table)
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
        _record_manifest(table, snapshot, task_plan.train_table, 'custom',
                         source_table=lineage)
        return table


def _record_manifest(table: TrainingTable, snapshot: GraphSnapshot,
                     request: Any, source: str, *,
                     source_table: Optional[Dict[str, Any]] = None) -> None:
    path = table.staging_dir / STAGING_MANIFEST_FILE
    try:
        from kumo_core.io.json import to_json_dict
        from kumo_core.training.export import read_spec

        spec = read_spec(str(table.staging_dir))
        seed = None
        if request.split_family in _RANDOM_SPLITS:
            seed = (request.split_config or {}).get('rnd_seed') or 0
        parents: Dict[str, Any] = {'snapshot': snapshot.id}
        if source_table is not None:
            parents['source_table'] = source_table
        write_manifest(
            path,
            new_manifest(
                'training_table', spec['tt_id'], parents,
                query=table.query,
                generation_plan=to_json_dict(table.generation_plan),
                task_type=request.task_type,
                rows={
                    stage['key']: stage['row_count']
                    for stage in spec.get('split_stages') or []
                },
                split_seed=seed,
                source=source,
            ))
    except Exception as error:
        warnings.warn(
            f"Could not write the training-table manifest at {path}: "
            f"{error}", UserWarning, stacklevel=3)


def _source_lineage(source_table: SourceTable) -> Dict[str, Optional[str]]:
    connector, name, sql = (source_table.connector, source_table.table_name,
                            source_table.sql)
    lineage: Dict[str, Optional[str]] = {'location': None, 'version': None}
    try:
        lineage['location'] = _location(connector, name, sql)
    except Exception as error:
        warnings.warn(
            f"Could not record where the custom table was loaded from: "
            f"{error}", UserWarning, stacklevel=3)
    if sql is None:
        try:
            version = connector.source_backend.get_data_version(
                connector, name)
            lineage['version'] = None if version is None else str(version)
        except Exception as error:
            warnings.warn(
                f"Could not record the custom table's data version: {error}",
                UserWarning, stacklevel=3)
    return lineage


def _location(connector: Any, name: str, sql: Optional[str]) -> str:
    import hashlib
    import posixpath

    from kumo_core.sources.object_store_sessions import ObjectStoreSessions

    if sql is not None:
        return 'sql:' + hashlib.sha256(sql.encode()).hexdigest()
    root = getattr(connector.config, 'root_dir', None)
    if root is not None:
        return str(Path(root, name))
    backend = connector.source_backend
    if isinstance(backend, ObjectStoreSessions):
        _, prefix, key, _ = backend._placement(connector, name)
        return f'{backend.scheme}://{posixpath.join(prefix, key).rstrip("/")}'
    namespace = (getattr(connector, field, None)
                 for field in ('database', 'catalog', 'project_id',
                               'schema_name', 'dataset_id'))
    return '.'.join([*filter(None, namespace), name])
