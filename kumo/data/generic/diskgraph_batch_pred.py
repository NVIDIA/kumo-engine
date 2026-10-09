# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import tempfile
from typing import List, NoReturn, Optional, Tuple

import kumo_core.types as core_types
import pyarrow as pa
import pyarrow.parquet as pq
from diskgraph import TimeUnit
from kumoapi.pquery import QueryType
from kumoapi.pquery.validated_predictive_query import ValidatedPredictiveQuery
from pandas import Timestamp

from common_utils.kumo_timing import kumo_timing
from io_utils import fsutil, logging
from kumo.data.generic.diskgraph_staging import (
    predict_staging_tt_id,
    ttng_persisted_pred_staging_dir,
    ttng_staging_dir,
)
from kumo.graph import Graph
from kumo.pquery.generator_utils import map_to_api_graph_definition
from kumo.pquery.utils import (
    get_lag_timestamps,
    get_rhs_lp_table,
    get_yoy_timestamps,
)
from kumo.validation.train_table_validation import TrainingTableGenerationError
from temporalio_commons.process_pool_utils import diskgraph_subprocess

logger = logging.getLogger(__name__)

LP_BP_LAYOUT_HSTACK = 'hstack'
LP_BP_LAYOUT_SEPARATE_LHS_RHS = 'separate_lhs_rhs'
LP_BP_LAYOUTS = {LP_BP_LAYOUT_HSTACK, LP_BP_LAYOUT_SEPARATE_LHS_RHS}


def _resolve_lp_bp_layout(
    list_distinct_target_is_array: bool,
    lp_bp_layout: Optional[str],
) -> str:
    if lp_bp_layout is not None and lp_bp_layout not in LP_BP_LAYOUTS:
        raise ValueError(
            f"lp_bp_layout must be one of {sorted(LP_BP_LAYOUTS)}, "
            f"got {lp_bp_layout!r}")
    if list_distinct_target_is_array:
        # Array-target LP has exactly one valid layout (separate LHS/RHS); a
        # persisted/deserialized hstack must be normalized, never silently used.
        return LP_BP_LAYOUT_SEPARATE_LHS_RHS
    return lp_bp_layout if lp_bp_layout is not None else LP_BP_LAYOUT_HSTACK


def _raise_empty_prediction_table(
    *,
    is_remote: bool,
    local_path: str,
    tmp_dir: Optional[str],
) -> NoReturn:
    try:
        if is_remote and tmp_dir is not None:
            fsutil.rm(tmp_dir)
        elif fsutil.exists(local_path):
            fsutil.rm(local_path)
    except Exception:
        logger.warning("Failed to clean empty prediction table output",
                       exc_info=True)

    raise TrainingTableGenerationError(
        "No rows to predict",
        "Kumo found no rows that need predictions. This usually means the "
        "column you asked Kumo to predict is already filled in for every row, "
        "or your filters removed every row. To continue, leave the target "
        "value blank for rows you want Kumo to fill, loosen the filters, or "
        "provide a prediction table with the rows to score.",
    )


@diskgraph_subprocess
def generate_batch_pred_table_via_diskgraph(
    query: ValidatedPredictiveQuery,
    task_type: core_types.TaskType,
    graph: Graph,
    timestamps: List[int],
    forecast_length: int,
    lag_timesteps: Optional[int],
    use_yoy_labels: bool,
    list_distinct_target_is_array: bool,
    index_path: str,
    output_root: str,
    input_timestamp_unit: TimeUnit,
    target_stype: str,
    entity_time_col: Optional[str],
    target_output_root: Optional[str] = None,
    lp_bp_layout: Optional[str] = None,
) -> Tuple[int, int]:
    """Compile and execute a diskgraph Batch Prediction plan.

    Handles Node Prediction, LP (separate or HStack), and Forecasting.
    Returns (total_rows, total_partitions).
    """
    try:
        from diskgraph import DiskGraphPlanCompiler
    except Exception as e:
        raise RuntimeError('Failed to import diskgraph runtime.') from e

    if not index_path:
        raise RuntimeError('diskgraph index path is required')

    with kumo_timing("batch_pred_table.create_plan_compiler"):
        compiler = DiskGraphPlanCompiler()

    # Determine execution mode
    if task_type == core_types.TaskType.FORECASTING:
        return _execute_forecasting(compiler, query, task_type, timestamps,
                                    lag_timesteps, forecast_length,
                                    use_yoy_labels, index_path, output_root,
                                    input_timestamp_unit=input_timestamp_unit,
                                    target_stype=target_stype,
                                    entity_time_col=entity_time_col)

    if task_type.is_link_pred:
        return _execute_link_prediction(
            compiler, query, graph, task_type, timestamps,
            list_distinct_target_is_array, lp_bp_layout, index_path,
            output_root, target_output_root,
            input_timestamp_unit=input_timestamp_unit,
            target_stype=target_stype, entity_time_col=entity_time_col)

    # Default: Node Prediction
    return _execute_node_prediction(compiler, query, task_type, timestamps,
                                    index_path, output_root,
                                    input_timestamp_unit=input_timestamp_unit,
                                    target_stype=target_stype,
                                    entity_time_col=entity_time_col)


def _execute_node_prediction(
    compiler,
    query: ValidatedPredictiveQuery,
    task_type: core_types.TaskType,
    timestamps: List[int],
    index_path: str,
    output_root: str,
    input_timestamp_unit: TimeUnit,
    target_stype: str,
    entity_time_col: Optional[str],
) -> Tuple[int, int]:
    with kumo_timing("batch_pred_table.node.compile_plan"):
        plan_json = compiler.compile(
            query=query,
            timestamps=timestamps,
            task_type=task_type.value,
            input_timestamp_unit=input_timestamp_unit,
            is_prediction=True,
            target_stype=target_stype,
            entity_time_col=entity_time_col,
        )

    output_path = osp.join(output_root, "part-00000.parquet")
    with kumo_timing("batch_pred_table.node.stream_to_parquet"):
        return _stream_to_parquet(plan_json, index_path, output_path,
                                  query_type=query.query_type,
                                  task_type=task_type)


def _execute_forecasting(
    compiler,
    query: ValidatedPredictiveQuery,
    task_type: core_types.TaskType,
    timestamps: List[int],
    lag_timesteps: Optional[int],
    forecast_length: int,
    use_yoy_labels: bool,
    index_path: str,
    output_root: str,
    input_timestamp_unit: TimeUnit,
    target_stype: str,
    entity_time_col: Optional[str],
) -> Tuple[int, int]:
    # Rebuild the anchor pd.Timestamp in its own unit so non-nanos callers
    # don't anchor to 1970.
    from diskgraph import TimeUnit
    _pd_unit = {
        TimeUnit.Nanoseconds: 'ns',
        TimeUnit.Microseconds: 'us',
        TimeUnit.Milliseconds: 'ms',
        TimeUnit.Seconds: 's',
    }[input_timestamp_unit]
    pt = Timestamp(timestamps[0], unit=_pd_unit)

    # Same timeframe step logic as Spark's batch_pred_table_generator.py.
    time_range = query.get_combined_date_offset_range()
    step = time_range.timeframe
    assert step is not None

    lag_ts = get_lag_timestamps(pt, lag_timesteps, step)
    lag_offsets_secs = [int((t - pt).total_seconds()) for t in lag_ts]

    yoy_offsets_secs = None
    if use_yoy_labels:
        yoy_ts, _ = get_yoy_timestamps(pt, forecast_length, step)
        yoy_offsets_secs = [int((t - pt).total_seconds()) for t in yoy_ts]

    plan_json = compiler.compile(
        query=query,
        timestamps=timestamps,
        task_type=task_type.value,
        input_timestamp_unit=input_timestamp_unit,
        is_prediction=True,
        target_stype=target_stype,
        entity_time_col=entity_time_col,
        lag_offsets_secs=lag_offsets_secs,
        yoy_offsets_secs=yoy_offsets_secs,
    )

    output_path = osp.join(output_root, "part-00000.parquet")
    return _stream_to_parquet(
        plan_json,
        index_path,
        output_path,
        query_type=query.query_type,
        task_type=task_type,
    )


def _execute_link_prediction(
    compiler,
    original_query: ValidatedPredictiveQuery,
    graph: Graph,
    task_type: core_types.TaskType,
    timestamps: List[int],
    list_distinct_target_is_array: bool,
    lp_bp_layout: Optional[str],
    index_path: str,
    output_root: str,
    target_output_root: Optional[str],
    input_timestamp_unit: TimeUnit,
    target_stype: str,
    entity_time_col: Optional[str],
) -> Tuple[int, int]:
    from diskgraph import execute_tt_plan
    from kumopql.rewriting import BPFilterTransformer

    tt_str = task_type.value
    resolved_lp_bp_layout = _resolve_lp_bp_layout(
        list_distinct_target_is_array,
        lp_bp_layout,
    )
    uses_separate_lhs_rhs = (
        resolved_lp_bp_layout == LP_BP_LAYOUT_SEPARATE_LHS_RHS)

    lhs_plan = compiler.compile(
        query=original_query,
        timestamps=timestamps,
        task_type=tt_str,
        input_timestamp_unit=input_timestamp_unit,
        is_prediction=True,
        target_stype=target_stype,
        entity_time_col=entity_time_col,
        lp_prediction_mode="LHS",
        list_distinct_target_is_array=list_distinct_target_is_array,
        lp_bp_layout=resolved_lp_bp_layout,
    )
    # Sanitize the query AST for RHS candidate extraction.
    bp_transformer = BPFilterTransformer()
    graph_def = map_to_api_graph_definition(graph)
    transformed_query = bp_transformer.generate_bp_query(
        original_query, task_type, graph_def)

    dst_table = get_rhs_lp_table(transformed_query, graph)
    target_table = dst_table.split('.')[-1]
    rhs_entity_pkey = graph.table(dst_table).pkey
    target_pk = rhs_entity_pkey.name

    rhs_plan = compiler.compile(
        query=transformed_query,
        timestamps=timestamps,
        task_type=tt_str,
        input_timestamp_unit=input_timestamp_unit,
        is_prediction=True,
        target_stype=target_stype,
        entity_time_col=entity_time_col,
        lp_prediction_mode="RHS",
        lp_rhs_entity_table=target_table,
        lp_rhs_entity_column=target_pk,
        list_distinct_target_is_array=list_distinct_target_is_array,
        lp_bp_layout=resolved_lp_bp_layout,
    )
    if uses_separate_lhs_rhs:
        output_path = osp.join(output_root, "part-00000.parquet")
        _stream_to_parquet(lhs_plan, index_path, output_path,
                           query_type=original_query.query_type,
                           task_type=task_type,
                           list_distinct_target_is_array=True)

        if target_output_root is not None:
            _stream_to_parquet(rhs_plan, index_path, target_output_root,
                               query_type=original_query.query_type,
                               task_type=task_type, is_rhs=True)

        return 0, 1

    # Legacy HStack mode. Drain RHS before opening the LHS iterator: two live
    # TTPlanIterators deadlock (each __next__ joins a GIL-needing thread).
    output_path = osp.join(output_root, "part-00000.parquet")

    # ParquetWriter is local-only: write to a temp file then copy when remote.
    is_remote = not fsutil.is_local(output_path)
    if is_remote:
        tmp_dir = tempfile.mkdtemp(prefix='diskgraph_bp_')
        local_path = osp.join(tmp_dir, "part-00000.parquet")
    else:
        local_path = output_path
        fsutil.makedirs(output_root, exist_ok=True)

    rhs_batches = list(execute_tt_plan(rhs_plan, index_path))

    if not rhs_batches or sum(b.num_rows for b in rhs_batches) == 0:
        raise TrainingTableGenerationError(
            "No candidates to rank",
            "Link Prediction batch prediction produced no candidates for the "
            "requested time window. This usually means your query matched no "
            "candidate items, or your filters removed every candidate. To "
            "continue, loosen the candidate-side filters or widen the "
            "prediction time window.",
        )

    first_rhs_batch = rhs_batches[0]
    first_rhs_val = first_rhs_batch.column(
        'ENTITY')[0] if first_rhs_batch.num_rows > 0 else None
    rhs_iter = iter(rhs_batches)

    # --- Now create LHS iterator (streams, not buffered) ---
    lhs_iter = execute_tt_plan(lhs_plan, index_path)

    try:
        first_lhs_batch = next(lhs_iter)

        def iter_with_first(first, rest):
            yield first
            yield from rest

        lhs_iter = iter_with_first(first_lhs_batch, lhs_iter)
        first_lhs_val = first_lhs_batch.column(
            'ENTITY')[0] if first_lhs_batch.num_rows > 0 else None
    except StopIteration:
        raise TrainingTableGenerationError(
            "No entities to predict",
            "Link Prediction batch prediction produced no entities for the "
            "requested time window. This usually means your query matched no "
            "entities, or your filters removed every entity. To continue, "
            "loosen the entity-side filters or widen the prediction time "
            "window.",
        )

    writer = None
    total_rows = 0
    global_arange_offset = 0

    lhs_exhausted = False
    rhs_exhausted = False

    current_lhs_batch = None
    current_rhs_batch = None
    lhs_offset = 0
    rhs_offset = 0

    try:
        while not (lhs_exhausted and rhs_exhausted):
            if current_lhs_batch is None or lhs_offset >= current_lhs_batch.num_rows:
                try:
                    current_lhs_batch = next(lhs_iter)
                    lhs_offset = 0
                except StopIteration:
                    lhs_exhausted = True
                    current_lhs_batch = None

            if current_rhs_batch is None or rhs_offset >= current_rhs_batch.num_rows:
                try:
                    current_rhs_batch = next(rhs_iter)
                    rhs_offset = 0
                except StopIteration:
                    rhs_exhausted = True
                    current_rhs_batch = None

            if lhs_exhausted and rhs_exhausted:
                break

            lhs_avail = current_lhs_batch.num_rows - lhs_offset if current_lhs_batch else float(
                'inf')
            rhs_avail = current_rhs_batch.num_rows - rhs_offset if current_rhs_batch else float(
                'inf')

            if lhs_exhausted:
                batch_size = rhs_avail
            elif rhs_exhausted:
                batch_size = lhs_avail
            else:
                batch_size = min(lhs_avail, rhs_avail)

            # Build the base batch from LHS and add TARGET from RHS
            if lhs_exhausted:
                # Pad LHS with first LHS value
                pad_val = first_lhs_val.as_py(
                ) if first_lhs_val is not None else None
                lhs_array = pa.array([pad_val] * batch_size,
                                     type=first_lhs_batch.column('ENTITY').type)

                # Construct a dummy batch with the required columns
                base_arrays = [lhs_array]
                base_names = ['ENTITY']

                # Forward fill the timestamp if it exists in the first batch
                if 'TIMESTAMP' in first_lhs_batch.schema.names and original_query.query_type != QueryType.STATIC:
                    ts_type = first_lhs_batch.column('TIMESTAMP').type
                    ts_val = first_lhs_batch.column('TIMESTAMP')[0].as_py()
                    base_arrays.append(
                        pa.array([ts_val] * batch_size, type=ts_type))
                    base_names.append('TIMESTAMP')

                # Forward-fill PSEUDO_TIMESTAMP on padded batches so the writer
                # schema (set from the first real batch) stays consistent.
                if 'PSEUDO_TIMESTAMP' in first_lhs_batch.schema.names:
                    pts_type = first_lhs_batch.column('PSEUDO_TIMESTAMP').type
                    pts_val = first_lhs_batch.column(
                        'PSEUDO_TIMESTAMP')[0].as_py()
                    base_arrays.append(
                        pa.array([pts_val] * batch_size, type=pts_type))
                    base_names.append('PSEUDO_TIMESTAMP')

                base_batch = pa.RecordBatch.from_arrays(base_arrays,
                                                        names=base_names)
            else:
                # Slice the entire batch from LHS
                base_batch = current_lhs_batch.slice(lhs_offset, batch_size)
                lhs_offset += batch_size

            # Get the TARGET column from RHS
            if rhs_exhausted:
                pad_val = first_rhs_val.as_py(
                ) if first_rhs_val is not None else None
                rhs_array = pa.array([pad_val] * batch_size,
                                     type=first_rhs_batch.column('ENTITY').type)
            else:
                rhs_array = current_rhs_batch.column('ENTITY').slice(
                    rhs_offset, batch_size)
                rhs_offset += batch_size

            # Legacy LP uses PSEUDO_TIMESTAMP, so the TIMESTAMP check is a no-op
            # today; kept for safety.
            drop_cols = []
            if 'TARGET' in base_batch.schema.names:
                drop_cols.append('TARGET')
            if (original_query.query_type == QueryType.STATIC
                    and task_type.is_link_pred
                    and 'TIMESTAMP' in base_batch.schema.names):
                drop_cols.append('TIMESTAMP')
            if drop_cols:
                base_batch = base_batch.drop_columns(drop_cols)

            # Add the TARGET column explicitly from RHS
            base_batch = base_batch.append_column('TARGET', rhs_array)

            arange_array = pa.array(
                range(global_arange_offset, global_arange_offset + batch_size),
                type=pa.int64())
            base_batch = base_batch.append_column('arange', arange_array)
            global_arange_offset += batch_size

            if writer is None:
                # Padded batches (on side exhaustion) are nullable even though
                # DG columns are not-null.
                nullable_schema = pa.schema(
                    [f.with_nullable(True) for f in base_batch.schema])
                writer = pq.ParquetWriter(local_path, nullable_schema,
                                          compression='snappy')

            if base_batch.schema != writer.schema:
                base_batch = base_batch.cast(writer.schema)
            writer.write_batch(base_batch)
            total_rows += batch_size
    finally:
        if writer:
            writer.close()

    if is_remote:
        fsutil.makedirs(output_root, exist_ok=True)
        fsutil.copy_file(local_path, output_path, overwrite=True)
        fsutil.rm(tmp_dir)

    return total_rows, 1


# DG emits slot-0 ENTITY1/TARGET1; kumo's customer-facing pred table names them
# ENTITY/TARGET (a temporary alias until kumo adopts DG's multi-entity schema).
_DG_TO_KUMO_COL = {'ENTITY1': 'ENTITY', 'TARGET1': 'TARGET'}


def _rename_dg_to_kumo(batch: pa.RecordBatch) -> pa.RecordBatch:
    new_names = [_DG_TO_KUMO_COL.get(n, n) for n in batch.schema.names]
    if new_names == batch.schema.names:
        return batch
    new_schema = pa.schema([
        batch.schema.field(i).with_name(new_names[i])
        for i in range(len(new_names))
    ])
    return pa.RecordBatch.from_arrays(batch.columns, schema=new_schema)


def _postprocess_bp_batch(
    batch: pa.RecordBatch,
    *,
    total_rows: int,
    query_type: Optional[QueryType],
    task_type: Optional[core_types.TaskType],
    list_distinct_target_is_array: bool,
    is_rhs: bool,
) -> pa.RecordBatch:
    """Apply the shared batch-prediction post-processing to one DG batch:
    slot-0 rename, per-task column drops, and the ``arange`` row index. Shared by
    the streaming (``_stream_to_parquet*``) and the bundle-replay
    (``_write_serving_pred_parquet_from_bundle``) paths so the customer-facing
    pred-table schema is identical regardless of how the table was produced."""
    batch = _rename_dg_to_kumo(batch)
    num_rows = batch.num_rows

    if is_rhs:
        if 'ENTITY' in batch.schema.names:
            idx = batch.schema.get_field_index('ENTITY')
            batch = batch.set_column(idx, 'TARGET', batch.column('ENTITY'))
    else:
        # Drop TARGET for Node/Ranking/Forecasting or modern-LP LHS.
        should_drop_target = False
        if task_type and not task_type.is_link_pred:
            should_drop_target = True
        elif list_distinct_target_is_array:
            should_drop_target = True
        if should_drop_target and 'TARGET' in batch.schema.names:
            batch = batch.drop_columns(['TARGET'])

        # Drop TIMESTAMP only for static array-target LP ("modern LP", GE
        # final_cols = {ENTITY, TARGET}). Static node-pred keeps it; static
        # legacy LP names it PSEUDO_TIMESTAMP so won't match here.
        is_static_lp_modern = (query_type and query_type.name == "STATIC"
                               and task_type and task_type.is_link_pred
                               and list_distinct_target_is_array)
        if is_static_lp_modern and 'TIMESTAMP' in batch.schema.names:
            batch = batch.drop_columns(['TIMESTAMP'])

    arange_array = pa.array(range(total_rows, total_rows + num_rows),
                            type=pa.int64())
    return batch.append_column("arange", arange_array)


def _write_serving_pred_parquet_from_bundle(
    *,
    bundle_tt_parquet: str,
    output_root: str,
    query_type: Optional[QueryType],
    task_type: Optional[core_types.TaskType],
    list_distinct_target_is_array: bool,
) -> int:
    """Write the customer-facing serving pred table to
    ``<output_root>/part-00000.parquet`` by replaying the already-persisted
    gen bundle ``tt.parquet`` through the same post-processing the streaming
    path applies (rename + per-task drops + ``arange``).

    The TT-NG predict gen persists only the internal ``.ttng_staging`` bundle
    (materialize ingests it directly); nothing is written at ``output_root``, so
    ``GET /genpredtable_jobs/{id}/table_data`` -- which presigns/lists parquet
    under the pred-table path -- returned ``[]``. This materializes the readable
    pred table there so the public SDK contract holds, reusing the bundle (no
    second ``execute_tt_gen``). LHS-only: RHS candidates live in the ingest
    bundle, so ``is_rhs`` never applies here."""
    from diskgraph import read_tt_parquet_batches

    output_path = osp.join(output_root, "part-00000.parquet")
    is_remote = not fsutil.is_local(output_path)
    tmp_dir = None
    if is_remote:
        tmp_dir = tempfile.mkdtemp(prefix='diskgraph_bp_serving_')
        local_path = osp.join(tmp_dir, "part-00000.parquet")
    else:
        local_path = output_path
        fsutil.makedirs(output_root, exist_ok=True)

    writer = None
    total_rows = 0
    try:
        for batch in read_tt_parquet_batches(bundle_tt_parquet):
            batch = _postprocess_bp_batch(
                batch,
                total_rows=total_rows,
                query_type=query_type,
                task_type=task_type,
                list_distinct_target_is_array=list_distinct_target_is_array,
                is_rhs=False,
            )
            if writer is None:
                writer = pq.ParquetWriter(local_path, batch.schema,
                                          compression='snappy')
            writer.write_batch(batch)
            total_rows += batch.num_rows
    finally:
        if writer:
            writer.close()

    if is_remote:
        fsutil.makedirs(output_root, exist_ok=True)
        fsutil.copy_file(local_path, output_path, overwrite=True)
        fsutil.rm(tmp_dir, ignore_errors=True)
    return total_rows


def _stream_to_parquet(
    plan_json: str,
    index_path: str,
    output_path: str,
    query_type: Optional[QueryType] = None,
    task_type: Optional[core_types.TaskType] = None,
    list_distinct_target_is_array: bool = False,
    is_rhs: bool = False,
) -> Tuple[int, int]:
    """Helper to stream a single-pass plan to parquet with arange."""
    from diskgraph import execute_tt_plan

    # Write a file inside a directory output path.
    if output_path.endswith('/'):
        output_path = osp.join(output_path, "part-00000.parquet")
    elif not output_path.endswith('.parquet'):
        output_path = osp.join(output_path, "part-00000.parquet")

    # ParquetWriter is local-only: write to a temp file then copy when remote.
    is_remote = not fsutil.is_local(output_path)
    tmp_dir = None
    if is_remote:
        tmp_dir = tempfile.mkdtemp(prefix='diskgraph_bp_')
        local_path = osp.join(tmp_dir, "part-00000.parquet")
    else:
        local_path = output_path
        fsutil.makedirs(osp.dirname(output_path), exist_ok=True)

    writer = None
    total_rows = 0
    batch_count = 0
    try:
        with kumo_timing("batch_pred_table.stream.execute_tt_plan"):
            for batch in execute_tt_plan(plan_json, index_path):
                num_rows = batch.num_rows
                batch_count += 1

                if is_rhs:
                    if 'ENTITY' in batch.schema.names:
                        idx = batch.schema.get_field_index('ENTITY')
                        batch = batch.set_column(idx, 'TARGET',
                                                 batch.column('ENTITY'))
                else:
                    # Drop TARGET for Node/Ranking/Forecasting or modern-LP LHS.
                    should_drop_target = False
                    if task_type and not task_type.is_link_pred:
                        should_drop_target = True
                    elif list_distinct_target_is_array:
                        should_drop_target = True

                    if should_drop_target:
                        drop_cols = []
                        if 'TARGET' in batch.schema.names:
                            drop_cols.append('TARGET')
                        if drop_cols:
                            batch = batch.drop_columns(drop_cols)

                    # Drop TIMESTAMP only for static array-target LP
                    # ("modern LP", GE final_cols = {ENTITY, TARGET}). Static
                    # node-pred keeps it; static legacy LP names it
                    # PSEUDO_TIMESTAMP so won't match here.
                    is_static_lp_modern = (query_type
                                           and query_type.name == "STATIC"
                                           and task_type
                                           and task_type.is_link_pred
                                           and list_distinct_target_is_array)
                    if (is_static_lp_modern
                            and 'TIMESTAMP' in batch.schema.names):
                        batch = batch.drop_columns(['TIMESTAMP'])

                # Inject arange
                arange_array = pa.array(
                    range(total_rows, total_rows + num_rows), type=pa.int64())
                batch = batch.append_column("arange", arange_array)

                if writer is None:
                    writer = pq.ParquetWriter(local_path, batch.schema,
                                              compression='snappy')

                writer.write_batch(batch)
                total_rows += num_rows
    finally:
        if writer:
            writer.close()

    if total_rows == 0:
        _raise_empty_prediction_table(
            is_remote=is_remote,
            local_path=local_path,
            tmp_dir=tmp_dir,
        )

    if is_remote:
        with kumo_timing("batch_pred_table.stream.copy_remote_output",
                         total_rows=total_rows, batch_count=batch_count):
            fsutil.makedirs(osp.dirname(output_path), exist_ok=True)
            fsutil.copy_file(local_path, output_path, overwrite=True)
            fsutil.rm(tmp_dir)

    return total_rows, 1


# ---------------------------------------------------------------------------
# TT-NG sibling functions. Separate from the master helpers above (which keep
# working against the master wheel) so both paths coexist during the dual-path
# window.
# ---------------------------------------------------------------------------


@diskgraph_subprocess
def generate_batch_pred_table_via_diskgraph_ttng(
    query: ValidatedPredictiveQuery,
    task_type: core_types.TaskType,
    graph: Graph,
    timestamps: List[int],
    forecast_length: int,
    lag_timesteps: Optional[int],
    use_yoy_labels: bool,
    list_distinct_target_is_array: bool,
    index_path: str,
    output_root: str,
    input_timestamp_unit: TimeUnit,
    target_stype: str,
    entity_time_col: Optional[str],
    target_output_root: Optional[str] = None,
    lp_bp_layout: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Tuple[int, int]:
    """TT-NG sibling of :func:`generate_batch_pred_table_via_diskgraph`.

    ``job_id`` names the kept predict staging dir so ``materialize_prediction_task``
    ingests this gen instead of re-running it (one-gen predict, like train).
    """
    try:
        from diskgraph import DiskGraphPlanCompiler
    except Exception as e:
        raise RuntimeError('Failed to import diskgraph runtime.') from e

    if not index_path:
        raise RuntimeError('diskgraph index path is required')

    compiler = DiskGraphPlanCompiler()

    if task_type == core_types.TaskType.FORECASTING:
        return _execute_forecasting_ttng(
            compiler, query, task_type, timestamps, lag_timesteps,
            forecast_length, use_yoy_labels, index_path, output_root,
            input_timestamp_unit=input_timestamp_unit,
            target_stype=target_stype, entity_time_col=entity_time_col,
            job_id=job_id)

    if task_type.is_link_pred:
        return _execute_link_prediction_ttng(
            query, graph, task_type, timestamps, list_distinct_target_is_array,
            lp_bp_layout, index_path, input_timestamp_unit=input_timestamp_unit,
            target_stype=target_stype, entity_time_col=entity_time_col,
            job_id=job_id, output_root=output_root)

    # Default: Node Prediction
    return _execute_node_prediction_ttng(
        query, task_type, timestamps, index_path,
        input_timestamp_unit=input_timestamp_unit, target_stype=target_stype,
        entity_time_col=entity_time_col, graph=graph, job_id=job_id,
        output_root=output_root)


def _execute_node_prediction_ttng(
    query: ValidatedPredictiveQuery,
    task_type: core_types.TaskType,
    timestamps: List[int],
    index_path: str,
    input_timestamp_unit: TimeUnit,
    target_stype: str,
    entity_time_col: Optional[str],
    graph: Graph,
    job_id: Optional[str] = None,
    output_root: Optional[str] = None,
) -> Tuple[int, int]:
    """TT-NG sibling of :func:`_execute_node_prediction` — one-gen predict into
    the KEPT staging dir named by ``job_id``; materialize ingests it directly."""
    if not job_id:
        raise RuntimeError(
            "TT-NG node-prediction batch prediction requires job_id to name "
            "the kept predict staging dir (one-gen handoff to materialize); "
            f"got {job_id!r}.")

    # Node-pred: no lp_resolution; tt-ingest synthesizes ENTITY1 -> entity_node_id.
    plan_json = build_ttng_predict_plan(
        query,
        task_type=task_type,
        graph=graph,
        timestamps=timestamps,
        list_distinct_target_is_array=False,
        input_timestamp_unit=input_timestamp_unit,
        target_stype=target_stype,
        entity_time_col=entity_time_col,
    )

    return _gen_kept_staging_ttng(plan_json, index_path,
                                  tt_id=predict_staging_tt_id(job_id),
                                  output_root=output_root,
                                  query_type=query.query_type,
                                  task_type=task_type,
                                  list_distinct_target_is_array=False)


def _execute_forecasting_ttng(
    compiler,
    query: ValidatedPredictiveQuery,
    task_type: core_types.TaskType,
    timestamps: List[int],
    lag_timesteps: Optional[int],
    forecast_length: int,
    use_yoy_labels: bool,
    index_path: str,
    output_root: str,
    input_timestamp_unit: TimeUnit,
    target_stype: str,
    entity_time_col: Optional[str],
    job_id: Optional[str] = None,
) -> Tuple[int, int]:
    """TT-NG forecasting predict into the kept staging dir for materialize."""
    if not job_id:
        raise RuntimeError(
            "TT-NG forecasting batch prediction requires job_id to name the "
            "kept predict staging dir (one-gen handoff to materialize); got "
            f"{job_id!r}.")

    # Rebuild the anchor pd.Timestamp in its own unit so non-nanos callers
    # don't anchor to 1970.
    from diskgraph import TimeUnit
    _pd_unit = {
        TimeUnit.Nanoseconds: 'ns',
        TimeUnit.Microseconds: 'us',
        TimeUnit.Milliseconds: 'ms',
        TimeUnit.Seconds: 's',
    }[input_timestamp_unit]
    pt = Timestamp(timestamps[0], unit=_pd_unit)
    time_range = query.get_combined_date_offset_range()
    step = time_range.timeframe
    assert step is not None

    lag_ts = get_lag_timestamps(pt, lag_timesteps, step)
    lag_offsets_secs = [int((t - pt).total_seconds()) for t in lag_ts]

    yoy_offsets_secs = None
    if use_yoy_labels:
        yoy_ts, _ = get_yoy_timestamps(pt, forecast_length, step)
        yoy_offsets_secs = [int((t - pt).total_seconds()) for t in yoy_ts]

    plan_json = compiler.compile(
        query=query,
        timestamps=timestamps,
        task_type=task_type.value,
        input_timestamp_unit=input_timestamp_unit,
        is_prediction=True,
        target_stype=target_stype,
        entity_time_col=entity_time_col,
        lag_offsets_secs=lag_offsets_secs,
        yoy_offsets_secs=yoy_offsets_secs,
    )

    return _gen_kept_staging_ttng(plan_json, index_path,
                                  tt_id=predict_staging_tt_id(job_id),
                                  output_root=output_root,
                                  query_type=query.query_type,
                                  task_type=task_type,
                                  list_distinct_target_is_array=False)


def _execute_link_prediction_ttng(
    original_query: ValidatedPredictiveQuery,
    graph: Graph,
    task_type: core_types.TaskType,
    timestamps: List[int],
    list_distinct_target_is_array: bool,
    lp_bp_layout: Optional[str],
    index_path: str,
    input_timestamp_unit: TimeUnit,
    target_stype: str,
    entity_time_col: Optional[str],
    job_id: Optional[str] = None,
    output_root: Optional[str] = None,
) -> Tuple[int, int]:
    """TT-NG sibling of :func:`_execute_link_prediction`. FK-free (array-target)
    LP only: gens LHS predict seeds once into the kept staging dir named by
    ``job_id``. RHS candidates are seeded at ingest, so no RHS parquet or HStack
    cross-product is materialized; non-array layouts raise.
    """
    resolved_lp_bp_layout = _resolve_lp_bp_layout(
        list_distinct_target_is_array,
        lp_bp_layout,
    )
    uses_separate_lhs_rhs = (
        resolved_lp_bp_layout == LP_BP_LAYOUT_SEPARATE_LHS_RHS)
    if not uses_separate_lhs_rhs:
        raise NotImplementedError(
            "TT-NG link-prediction batch prediction requires the separate "
            "LHS/RHS (FK-free / array-target) layout; the legacy HStack "
            "cross-product layout is not supported on TT-NG. Enable array "
            "targets (ALLOW_ARRAY_TARGETS / list_distinct_target_is_array), or "
            f"use the non-TT-NG path. Got lp_bp_layout={resolved_lp_bp_layout!r}."
        )

    if not job_id:
        raise RuntimeError(
            "TT-NG LP batch prediction requires job_id to name the kept "
            "predict staging dir (one-gen handoff to materialize); got "
            f"{job_id!r}.")

    lhs_plan = build_ttng_predict_plan(
        original_query,
        task_type=task_type,
        graph=graph,
        timestamps=timestamps,
        list_distinct_target_is_array=list_distinct_target_is_array,
        input_timestamp_unit=input_timestamp_unit,
        target_stype=target_stype,
        entity_time_col=entity_time_col,
    )

    return _gen_kept_staging_ttng(
        lhs_plan, index_path, tt_id=predict_staging_tt_id(job_id),
        output_root=output_root, query_type=original_query.query_type,
        task_type=task_type,
        list_distinct_target_is_array=(list_distinct_target_is_array))


def _gen_kept_staging_ttng(
    plan_json: str,
    index_path: str,
    *,
    tt_id: str,
    output_root: Optional[str] = None,
    query_type: Optional[QueryType] = None,
    task_type: Optional[core_types.TaskType] = None,
    list_distinct_target_is_array: bool = False,
) -> Tuple[int, int]:
    """Run ``execute_tt_gen`` into the kept staging dir and return the row count.
    The kept staging dir is what materialize ingests directly (one-gen predict).
    Shared by node-pred + LP predict.

    When ``output_root`` (the pred-table path) is given: (1) copy the staging
    bundle to the durable ``ttng_persisted_pred_staging_dir`` beside the pred
    table so any pod can re-ingest later, and (2) materialize the customer-facing
    serving pred table at ``output_root/part-00000.parquet`` (replaying the
    bundle through the shared BP post-processing) so
    ``GET /genpredtable_jobs/{id}/table_data`` surfaces a readable path.
    Ephemeral-only (no persist, no serving table) without ``output_root``.
    """
    from diskgraph import execute_tt_gen

    # Do NOT delete the staging dir — materialize ingests it.
    gen_result = execute_tt_gen(plan_json, index_path, tt_id)
    total_rows = int(gen_result.get('row_count') or 0)
    if total_rows == 0:
        _raise_empty_prediction_table(
            is_remote=False,
            local_path='',
            tmp_dir=None,
        )

    if output_root:
        ephemeral_staging = ttng_staging_dir(index_path, tt_id)
        bundle_tt_parquet = osp.join(ephemeral_staging, 'tt.parquet')
        if not fsutil.exists(bundle_tt_parquet):
            raise RuntimeError(
                'TT-NG predict: execute_tt_gen produced no staging parquet to '
                f'persist durably; expected {bundle_tt_parquet!r}')
        durable_staging = ttng_persisted_pred_staging_dir(output_root)
        # Copy the whole bundle (ingest reads tt_ingest_spec.json / manifest.json
        # / stats alongside tt.parquet); it embeds no absolute paths.
        from kumo.data.generic import diskgraph_train_table
        diskgraph_train_table.persist_durable_ttng_bundle(
            ephemeral_staging, durable_staging)
        logger.info(
            'TT-NG predict: persisted durable gen→ingest staging %s -> %s',
            ephemeral_staging, durable_staging)

        # Materialize the customer-facing serving pred table beside the bundle so
        # the public /table_data endpoint returns a readable parquet path (the
        # gen bundle alone is internal: raw ENTITY1/TARGET1, no arange).
        serving_rows = _write_serving_pred_parquet_from_bundle(
            bundle_tt_parquet=bundle_tt_parquet,
            output_root=output_root,
            query_type=query_type,
            task_type=task_type,
            list_distinct_target_is_array=list_distinct_target_is_array,
        )
        logger.info('TT-NG predict: wrote serving pred table (%d rows) at %s',
                    serving_rows, output_root)

    return total_rows, 1


def _stream_to_parquet_ttng(
    plan_json: str,
    index_path: str,
    output_path: str,
    query_type: Optional[QueryType] = None,
    task_type: Optional[core_types.TaskType] = None,
    list_distinct_target_is_array: bool = False,
    is_rhs: bool = False,
) -> Tuple[int, int]:
    """TT-NG sibling of :func:`_stream_to_parquet`, iterating the wheel's
    ``tt.parquet`` and applying BP post-processing (column drops + arange) per
    batch. Renames slot-0 ENTITY1/TARGET1 -> ENTITY/TARGET (a temporary alias
    that disappears once kumo adopts DG's multi-entity schema).
    """
    from diskgraph import execute_tt_gen, read_tt_parquet_batches

    # Write a file inside a directory output path.
    if output_path.endswith('/'):
        output_path = osp.join(output_path, "part-00000.parquet")
    elif not output_path.endswith('.parquet'):
        output_path = osp.join(output_path, "part-00000.parquet")

    # ParquetWriter is local-only: write to a temp file then copy when remote.
    is_remote = not fsutil.is_local(output_path)
    tmp_dir = None
    if is_remote:
        tmp_dir = tempfile.mkdtemp(prefix='diskgraph_bp_')
        local_path = osp.join(tmp_dir, "part-00000.parquet")
    else:
        local_path = output_path
        fsutil.makedirs(osp.dirname(output_path), exist_ok=True)

    # The DG-side parquet lives in a tempdir cleaned up in the outer finally.
    dg_tmpdir = tempfile.mkdtemp(prefix='diskgraph_bp_stream_')
    dg_tt_out_dir = None

    writer = None
    total_rows = 0
    try:
        # Rename slot-0 ENTITY1/TARGET1 -> ENTITY/TARGET (see
        # _diskgraph_legacy_graphstore_bridge.py for the matching TT-path rename).
        _DG_TO_KUMO_COL = {'ENTITY1': 'ENTITY', 'TARGET1': 'TARGET'}

        def _rename_dg_to_kumo(batch: pa.RecordBatch) -> pa.RecordBatch:
            new_names = [_DG_TO_KUMO_COL.get(n, n) for n in batch.schema.names]
            if new_names == batch.schema.names:
                return batch
            new_schema = pa.schema([
                batch.schema.field(i).with_name(new_names[i])
                for i in range(len(new_names))
            ])
            return pa.RecordBatch.from_arrays(batch.columns, schema=new_schema)

        # 3rd arg is a tt_id, not a path — artifacts go to train_tables/<tt_id>/
        tt_id = osp.basename(dg_tmpdir.rstrip('/'))
        result = execute_tt_gen(plan_json, index_path, tt_id)
        dg_tt_out_dir = osp.dirname(result['tt_parquet_path'])
        for batch in read_tt_parquet_batches(result['tt_parquet_path']):
            batch = _rename_dg_to_kumo(batch)
            num_rows = batch.num_rows

            if is_rhs:
                if 'ENTITY' in batch.schema.names:
                    idx = batch.schema.get_field_index('ENTITY')
                    batch = batch.set_column(idx, 'TARGET',
                                             batch.column('ENTITY'))
            else:
                # Drop TARGET for Node/Ranking/Forecasting or modern-LP LHS.
                should_drop_target = False
                if task_type and not task_type.is_link_pred:
                    should_drop_target = True
                elif list_distinct_target_is_array:
                    should_drop_target = True

                if should_drop_target:
                    drop_cols = []
                    if 'TARGET' in batch.schema.names:
                        drop_cols.append('TARGET')
                    if drop_cols:
                        batch = batch.drop_columns(drop_cols)

                # Drop TIMESTAMP only for static array-target LP ("modern LP",
                # GE final_cols = {ENTITY, TARGET}). Static node-pred keeps it;
                # static legacy LP names it PSEUDO_TIMESTAMP so won't match here.
                is_static_lp_modern = (query_type
                                       and query_type.name == "STATIC"
                                       and task_type and task_type.is_link_pred
                                       and list_distinct_target_is_array)
                if (is_static_lp_modern and 'TIMESTAMP' in batch.schema.names):
                    batch = batch.drop_columns(['TIMESTAMP'])

            # Inject arange
            arange_array = pa.array(range(total_rows, total_rows + num_rows),
                                    type=pa.int64())
            batch = batch.append_column("arange", arange_array)

            if writer is None:
                writer = pq.ParquetWriter(local_path, batch.schema,
                                          compression='snappy')

            writer.write_batch(batch)
            total_rows += num_rows
    finally:
        if writer:
            writer.close()
        # Clean up the DG executor tempdir; the output already lives at local_path.
        for _td in (dg_tmpdir, dg_tt_out_dir):
            if _td:
                try:
                    fsutil.rm(_td)
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass

    if total_rows == 0:
        _raise_empty_prediction_table(
            is_remote=is_remote,
            local_path=local_path,
            tmp_dir=tmp_dir,
        )

    if is_remote:
        fsutil.makedirs(osp.dirname(output_path), exist_ok=True)
        fsutil.copy_file(local_path, output_path, overwrite=True)
        fsutil.rm(tmp_dir)

    return total_rows, 1


def build_ttng_predict_plan(
    query: ValidatedPredictiveQuery,
    *,
    task_type: core_types.TaskType,
    graph: Graph,
    timestamps: List[int],
    list_distinct_target_is_array: bool,
    input_timestamp_unit: TimeUnit,
    target_stype: str,
    entity_time_col: Optional[str],
) -> str:
    """Compile the TT-NG ``predict`` plan (single stage, no split).

    Every task declares ``lp_resolution`` for the LHS entity. LP has no TARGET
    at prediction time, so both node and link plans resolve only ENTITY1.
    """
    from diskgraph import DiskGraphPlanCompiler
    from kumo_core.training.table import apply_reference_resolution

    compiler = DiskGraphPlanCompiler()
    tt_str = task_type.value
    is_link_pred = bool(getattr(task_type, 'is_link_pred', False))

    if is_link_pred:
        if not list_distinct_target_is_array:
            raise RuntimeError(
                'TT-NG LP predict requires array-target resolution')
        # Array-target resolution is the ONLY LP mode DG (tt-ng) supports; under
        # TT-NG ``allow_array_targets()`` is always True, so this flag arrives
        # True from config and the predict plan gets FK-free ENTITY1 resolution.
        # LHS-only plan: seed set is the LHS anchors, no RHS/TARGET (predict has
        # no positives).
        plan_json = compiler.compile(
            query=query,
            timestamps=timestamps,
            task_type=tt_str,
            input_timestamp_unit=input_timestamp_unit,
            is_prediction=True,
            target_stype=target_stype,
            entity_time_col=entity_time_col,
            lp_prediction_mode="LHS",
            list_distinct_target_is_array=list_distinct_target_is_array,
        )
        # Entity-only resolution (ENTITY1 -> entity_node_id);
        # lp_rhs_entity_table=None declares only the LHS spec. Derive
        # entity_table from the query so it doesn't depend on the plan's shape.
        entity_table = query.entity_column.split('.')[0]
        plan_json = apply_reference_resolution(
            plan_json,
            entity_node_type=entity_table,
            rhs_node_type=None,
            resolve_link_targets=False,
            is_static_link=False,
            rhs_is_temporal=False,
        )
        return plan_json

    plan_json = compiler.compile(
        query=query,
        timestamps=timestamps,
        task_type=tt_str,
        input_timestamp_unit=input_timestamp_unit,
        is_prediction=True,
        target_stype=target_stype,
        entity_time_col=entity_time_col,
    )
    entity_table = query.entity_column.split('.')[0]
    return apply_reference_resolution(
        plan_json,
        entity_node_type=entity_table,
        rhs_node_type=None,
        resolve_link_targets=False,
        is_static_link=False,
        rhs_is_temporal=False,
    )
