# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Materialized prediction tables and their shared compile path."""

import warnings
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

from kumo_core.model_plan import \
    TrainingTableGenerationPlan as CoreGenerationPlan
from kumo_core.pquery import (
    ValidatedNativeQuery,
    build_native_pquery_task_plan,
    latest_temporal_timestamp,
    normalize_graph,
    resolve_query_forecast,
)
from kumo_core.prediction import (
    GeneratedPredictionTable,
    PredictionTableRequest,
    PreparedCustomPredictionTable,
    generate_custom_prediction_table,
    generate_prediction_table,
)
from kumoapi.model_plan import TrainingTableGenerationPlan

from kumo.engine._embeddings import no_encoder_inference_warnings
from kumo.engine.connector.source_table import SourceTable
from kumo.engine.graph import GraphSnapshot


@dataclass(frozen=True)
class PredictionTable:
    r"""An immutable, materialized prediction table, compiled from a
    `PredictiveQuery` against a `GraphSnapshot` -- produced by
    `PredictiveQuery.generate_prediction_table(snapshot)`, the prediction
    counterpart of `TrainingTable`/`generate_training_table`.

    `Trainer.predict` consumes this table's generated rows directly while
    building task metadata from the graph snapshot selected for that call.
    It reconstructs a
    `kumo_core.prediction.PredictionTableRequest`/`GeneratedPredictionTable`
    from the fields below and feeds them to `kumo_core.prediction.
    materialize_prediction_task` -- rather than recompiling the query
    against the original snapshot or resolving a fresh anchor timestamp,
    which would
    silently ignore the `prediction_time` this table was actually generated
    with.

    The table deliberately retains no graph snapshot, graph index, or task
    plan. Its generated population, anchors, required split stages, and exact
    forecast compiler settings remain independent of the graph whose features
    are selected later for scoring.
    """

    query: str
    table_path: Path
    staging_dir: Path
    #: Generated-row count and stable staging identifier. Prediction derives
    #: a separate ingest identifier for each call.
    row_count: int
    tt_id: str
    #: Anchor timestamps resolved when the table was generated.
    timestamps: Tuple[int, ...]
    #: Split artifacts that must exist in the generated staging directory.
    required_stages: Tuple[str, ...]
    #: Forecast compiler options resolved when this table was generated.
    compiler_options: Optional[Mapping[str, Any]] = None
    #: Effective temporal step used by the fitted forecast task.
    forecast_timeframe_step: Optional[int] = None
    source: Literal['pql', 'custom'] = 'pql'
    skipped_rows: int = 0
    skipped_sample: Tuple[Any, ...] = ()
    skip_reason: Optional[str] = None

    @property
    def requested_rows(self) -> int:
        r"""Rows asked for: the scored rows plus the skipped ones."""
        return self.row_count + self.skipped_rows

    @classmethod
    def from_source_table(
        cls,
        source_table: SourceTable,
        *,
        pq: "PredictiveQuery",  # noqa: F821
        snapshot: GraphSnapshot,
        plan: Optional[TrainingTableGenerationPlan] = None,
    ) -> "PredictionTable":
        r"""Materialize caller-supplied entities and anchor times.

        The source must contain the query's standard ``ENTITY1`` column and,
        when the task samples against seed times (a temporal query, or a
        static one on a timestamped entity), ``TIMESTAMP``. Those rows define
        the exact population to score. ``snapshot`` is used to validate and
        normalize the rows; :meth:`Trainer.predict` independently selects the
        graph to score through its own required ``snapshot=`` argument.
        """
        from kumo.engine._scratch import (
            new_run_id,
            prediction_table_scratch_dir,
        )
        from kumo.engine.pquery._custom_source import (
            stage_source_table,
            validate_prediction_source,
        )
        from kumo.engine.pquery.predictive_query import PredictiveQuery

        if not isinstance(pq, PredictiveQuery):
            raise TypeError("pq must be a kumo.engine PredictiveQuery")
        pq._require_matching_snapshot(snapshot)
        run_id = new_run_id()
        run_dir = prediction_table_scratch_dir(run_id)
        rows_path, _ = stage_source_table(source_table,
                                          run_dir / "source-table")
        request, _ = _build_prediction_table_request(
            snapshot,
            pq._parsed,
            prediction_time=None,
            run_id=run_id,
            scratch_dir=run_dir,
            artifact_dir=run_dir,
            generation_plan=plan,
            supplied_rows=True,
        )
        # Custom rows own their individual anchor times. Planning resolves only
        # graph/task/forecast metadata and must not replace the supplied
        # parquet's TIMESTAMP column.
        request = replace(request, timestamps=())
        validate_prediction_source(rows_path, request.query, request.graph)
        generated = generate_custom_prediction_table(
            request,
            PreparedCustomPredictionTable(str(rows_path)),
        )
        skipped = skipped_entities(generated.row_failures, rows_path)
        return cls(
            query=pq.query,
            table_path=Path(generated.table_path),
            staging_dir=Path(generated.staging_dir),
            row_count=generated.row_count,
            tt_id=generated.tt_id,
            timestamps=(),
            required_stages=request.task_dataset.required_stages,
            compiler_options=request.compiler_options,
            forecast_timeframe_step=request.forecast_timeframe_step,
            source='custom',
            **skipped,
        )


def _build_prediction_table_request(
    snapshot: GraphSnapshot,
    parsed: ValidatedNativeQuery,
    *,
    prediction_time: Optional[datetime],
    run_id: str,
    scratch_dir: Path,
    artifact_dir: Path,
    generation_plan: Optional[TrainingTableGenerationPlan] = None,
    supplied_rows: bool = False,
) -> Tuple[PredictionTableRequest, str]:
    """Resolve task metadata shared by generated and supplied populations."""
    native_graph = normalize_graph(snapshot.graph_config)
    from kumo_core.prediction_options import prediction_timestamps

    if prediction_time is None:
        timestamps = ()
        if parsed.query_type == 'temporal' and not supplied_rows:
            # Prediction does not need the future label window that bounds
            # training anchors. The latest target source defines the default
            # source clock, regardless of PQL aggregation order.
            timestamps = (latest_temporal_timestamp(
                parsed, str(snapshot.index_path)), )
    else:
        timestamps = prediction_timestamps(prediction_time)
        if parsed.query_type == 'static':
            raise ValueError('prediction_time is not supported for static queries')

    options = CoreGenerationPlan()
    if generation_plan is not None:
        from kumo_core.io.json import from_json_dict
        from kumoapi.json_serde import to_json_dict
        options = from_json_dict(
            to_json_dict(generation_plan),
            CoreGenerationPlan,
        )

    parsed = resolve_query_forecast(parsed, native_graph, options)
    with no_encoder_inference_warnings():
        plan = build_native_pquery_task_plan(
            query=parsed,
            graph=native_graph,
            training_table_generation_plan=options,
            run_id=run_id,
            graph_index_path=str(snapshot.index_path),
            scratch_dir=str(scratch_dir),
            artifact_dir=str(artifact_dir),
            graph_index_id=snapshot.id,
            timestamps=timestamps,
            supplied_rows=supplied_rows,
        )
    forecast_timeframe_step = None
    if parsed.task_type == 'forecasting' and generation_plan is not None:
        from kumo_core.model_plan import inferred_or_missing_type_to_none
        value = inferred_or_missing_type_to_none(options.timeframe_step)
        offset = parsed.query.get_combined_date_offset_range()
        forecast_timeframe_step = int(
            value if value is not None else int(offset.end) -
            int(offset.start or 0))
    return PredictionTableRequest(
        query=parsed,
        graph=native_graph,
        timestamps=timestamps,
        index_path=str(snapshot.index_path),
        tt_id=f'pred_{run_id}',
        task_dataset=plan.task_dataset,
        compiler_options=plan.train_table.compiler_options,
        forecast_timeframe_step=forecast_timeframe_step,
    ), plan.graph_index_id


def _compile_prediction_table(
    snapshot: GraphSnapshot,
    parsed: ValidatedNativeQuery,
    *,
    prediction_time: Optional[datetime],
    run_id: str,
    scratch_dir: Path,
    artifact_dir: Path,
    generation_plan: Optional[TrainingTableGenerationPlan] = None,
) -> Tuple[PredictionTableRequest, GeneratedPredictionTable, str]:
    r"""Compile and generate a prediction table against ``snapshot``.

    Returns:
        The request, generated artifacts, and graph-index identifier.
    """
    request, graph_index_id = _build_prediction_table_request(
        snapshot,
        parsed,
        prediction_time=prediction_time,
        run_id=run_id,
        scratch_dir=scratch_dir,
        artifact_dir=artifact_dir,
        generation_plan=generation_plan,
    )
    generated = generate_prediction_table(request)
    return request, generated, graph_index_id


SKIPPED_SAMPLE = 20
_SKIP_REASONS = {
    'unknown_entity': 'not in snapshot',
    'non_canonical_entity': 'not a canonical node in the snapshot',
    'null_entity': 'null entity id',
    'null_timestamp': 'null anchor time',
}


def skipped_entities(report: Optional[Mapping[str, Any]],
                     rows_path: Path) -> Dict[str, Any]:
    r"""Skipped-row fields from DiskGraph's row_failures; warns once."""
    skipped = int((report or {}).get('skipped_rows') or 0)
    if not skipped:
        return {}
    conditions = report.get('by_condition') or []
    reasons = [(_SKIP_REASONS.get(entry.get('condition'),
                                  str(entry.get('condition')).replace(
                                      '_', ' ')), int(entry.get('count') or 0))
               for entry in conditions]
    reason = (reasons[0][0] if len(reasons) == 1 else '; '.join(
        f'{name} ({count:,})' for name, count in reasons)) or None
    requested = int(report.get('rows_read') or 0)
    warnings.warn(
        f'Prediction skipped {skipped:,} of {requested:,} requested entities '
        f'({skipped / requested if requested else 1:.1%}): {reason}',
        UserWarning, stacklevel=3)
    return {
        'skipped_rows': skipped,
        'skipped_sample': _sample_ids(conditions, rows_path),
        'skip_reason': reason,
    }


def _sample_ids(conditions: Any, rows_path: Path) -> Tuple[Any, ...]:
    import re

    import pyarrow.parquet as pq

    ids: List[Any] = []
    try:
        for entry in conditions:
            for location in entry.get('samples') or ():
                found = re.fullmatch(r'(.+) row group (\d+) row (\d+)',
                                     str(location))
                if found is None or len(ids) >= SKIPPED_SAMPLE:
                    continue
                part = Path(rows_path)
                part = part / found.group(1) if part.is_dir() else part
                reader = pq.ParquetFile(part)
                group, row = int(found.group(2)), int(found.group(3))
                while row >= reader.metadata.row_group(group).num_rows:
                    row -= reader.metadata.row_group(group).num_rows
                    group += 1
                column = reader.read_row_group(
                    group, columns=[entry['column']]).column(0)
                ids.append(column[row].as_py())
    except Exception:
        pass
    return tuple(ids)
