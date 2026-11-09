# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow.parquet as pq
import pytest
from diskgraph import TimeUnit
from kumo_core.train.autoregression import AUTOREGRESSION_COL, YOY_COL
from kumo_core.types import TaskType
from kumoapi.pquery import QueryType
from kumoapi.pquery.AST import Aggregation, Column
from kumoapi.typing import AggregationType

from kumo.data.generic import generate_current_pred_table as current_pred
from kumo.data.generic.diskgraph_batch_pred import (
    LP_BP_LAYOUT_HSTACK,
    LP_BP_LAYOUT_SEPARATE_LHS_RHS,
    _resolve_lp_bp_layout,
    generate_batch_pred_table_via_diskgraph,
)
from kumo.pquery.batch_pred_table_generator import PredictionTableGeneratorV2
from kumo.pquery.predictive_query import PredictiveQueryV2
from kumo.validation import ValidationResponse
from kumo.validation.train_table_validation import TrainingTableGenerationError


@pytest.fixture
def auto_setup_kumo_backend():
    return False


class _StaticPredictionTableGenerator(PredictionTableGeneratorV2):
    @property
    def query_type(self):
        return QueryType.STATIC


# Mocked AST components
def get_mock_query():
    aggr = Aggregation(target=Column(fqn="items.item_id"),
                       aggr=AggregationType.LIST_DISTINCT)

    query = MagicMock()
    query.entity_table = "users"
    query.entity_column = "users.user_id"
    query.entity_ast = Column(fqn="users.user_id")
    query.query_type = QueryType.STATIC
    query.list_distinct_target_is_array = False
    query.whatif_ast = None

    query.target_ast = aggr

    # Mock getting the final target
    query.get_final_target_aggregation.return_value = aggr
    query.get_single_target_fkey.return_value = "items.item_id"

    # Mock time offset logic for forecasting
    mock_range = MagicMock()
    mock_range.unit.value = "days"
    query.get_combined_date_offset_range.return_value = mock_range
    return query


def get_mock_graph():
    from kumo.graph.column import ForeignKey
    from kumo.graph.table import Table

    g = MagicMock()

    # Mock column_from_fqn to return a ForeignKey
    fk_mock = MagicMock(spec=ForeignKey)
    fk_mock.dst_table = "items"
    g.column_from_fqn.return_value = fk_mock

    # Mock table to return a Table with a pkey
    table_mock = MagicMock(spec=Table)
    table_mock.pkey = MagicMock()
    table_mock.pkey.name = "item_id"
    g.table.return_value = table_mock

    return g


@pytest.fixture
def setup_dirs(tmpdir):
    idx_path = os.path.join(tmpdir, "index")
    out_path = os.path.join(tmpdir, "output")
    os.makedirs(idx_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(idx_path, "index_metadata.json"), "w") as f:
        f.write("{}")
    return str(idx_path), str(out_path)


@patch('diskgraph.execute_tt_plan')
def test_node_prediction(mock_execute, setup_dirs):
    idx_path, out_path = setup_dirs
    query = get_mock_query()

    # Mock diskgraph iterator
    import pyarrow as pa
    mock_batch = pa.RecordBatch.from_arrays([pa.array(["u1", "u2"])],
                                            names=["ENTITY"])
    mock_execute.return_value = iter([mock_batch])
    total_rows, parts = generate_batch_pred_table_via_diskgraph(
        query=query, task_type=TaskType.BINARY_CLASSIFICATION,
        graph=get_mock_graph(), timestamps=[0], forecast_length=1,
        lag_timesteps=None, use_yoy_labels=False,
        list_distinct_target_is_array=False, index_path=idx_path,
        output_root=out_path, input_timestamp_unit=TimeUnit.Nanoseconds,
        target_stype='numerical', entity_time_col=None)

    assert total_rows == 2
    assert parts == 1

    # Verify file was written. With entity_time_col=None and the mock batch
    # only emitting ENTITY (no TIMESTAMP), there's no TIMESTAMP column to
    # carry through; output is ENTITY + arange.
    df = pq.read_table(os.path.join(out_path, "part-00000.parquet")).to_pandas()
    assert len(df) == 2
    assert list(df.columns) == ["ENTITY", "arange"]
    assert list(df['arange']) == [0, 1]


@patch('diskgraph.execute_tt_plan')
def test_node_prediction_empty_batches_raise_validation_error(
        mock_execute, setup_dirs):
    idx_path, out_path = setup_dirs
    mock_execute.return_value = iter([])

    with pytest.raises(
            TrainingTableGenerationError,
            match="No rows to predict.*leave the target value "
            "blank"):
        generate_batch_pred_table_via_diskgraph(
            query=get_mock_query(), task_type=TaskType.BINARY_CLASSIFICATION,
            graph=get_mock_graph(), timestamps=[0], forecast_length=1,
            lag_timesteps=None, use_yoy_labels=False,
            list_distinct_target_is_array=False, index_path=idx_path,
            output_root=out_path, input_timestamp_unit=TimeUnit.Nanoseconds,
            target_stype='numerical', entity_time_col=None)


@patch('kumopql.rewriting.BPFilterTransformer', create=True)
@patch('diskgraph.execute_tt_plan')
def test_link_prediction_hstack(mock_execute, mock_transformer, setup_dirs):
    idx_path, out_path = setup_dirs
    query = get_mock_query()

    # Mock transformer to just return the query
    mock_instance = MagicMock()
    mock_instance.generate_bp_query.return_value = query
    mock_transformer.return_value = mock_instance

    import pyarrow as pa

    # LHS returns users and native pseudo time in two asymmetric chunks
    lhs_batch_1 = pa.RecordBatch.from_arrays(
        [pa.array(["u1", "u2"]),
         pa.array([100000000000000, 100000000000000])],
        names=["ENTITY", "PSEUDO_TIMESTAMP"])
    lhs_batch_2 = pa.RecordBatch.from_arrays(
        [pa.array(["u3"]), pa.array([100000000000000])],
        names=["ENTITY", "PSEUDO_TIMESTAMP"])

    # RHS returns candidate items in two asymmetric chunks
    rhs_batch_1 = pa.RecordBatch.from_arrays([pa.array(["i1"])],
                                             names=["ENTITY"])
    rhs_batch_2 = pa.RecordBatch.from_arrays([pa.array(["i2"])],
                                             names=["ENTITY"])

    # execute_tt_plan is called twice: first for RHS (drained eagerly),
    # then for LHS (streamed).
    mock_execute.side_effect = [
        iter([rhs_batch_1, rhs_batch_2]),
        iter([lhs_batch_1, lhs_batch_2])
    ]

    total_rows, parts = generate_batch_pred_table_via_diskgraph(
        query=query, task_type=TaskType.STATIC_LINK_PREDICTION,
        graph=get_mock_graph(), timestamps=[0], forecast_length=1,
        lag_timesteps=None, use_yoy_labels=False,
        list_distinct_target_is_array=False, index_path=idx_path,
        output_root=out_path, input_timestamp_unit=TimeUnit.Nanoseconds,
        target_stype='numerical', entity_time_col=None)

    assert total_rows == 3

    df = pq.read_table(os.path.join(out_path, "part-00000.parquet")).to_pandas()
    assert len(df) == 3
    # Static legacy LP: column is PSEUDO_TIMESTAMP, not TIMESTAMP. The drop
    # only fires for static-LP-modern, so PSEUDO_TIMESTAMP is preserved.
    assert list(
        df.columns) == ["ENTITY", "PSEUDO_TIMESTAMP", "TARGET", "arange"]
    # The implementation fills min(lhs_avail, rhs_avail).
    # Row 1: u1 -> i1
    # Row 2: u2 -> i2 (from RHS chunk 2)
    # Row 3: u3 -> i1 (RHS is exhausted, pads with first RHS value)
    assert list(df['ENTITY']) == ["u1", "u2", "u3"]
    assert list(df['TARGET']) == ["i1", "i2", "i1"]
    assert list(df['arange']) == [0, 1, 2]


@patch('diskgraph.execute_tt_plan')
def test_temporal_node_prediction(mock_execute, setup_dirs):
    idx_path, out_path = setup_dirs
    query = get_mock_query()
    query.query_type = QueryType.TEMPORAL

    import pyarrow as pa
    mock_batch = pa.RecordBatch.from_arrays(
        [pa.array(["u1", "u2"]), pa.array([100, 200])],
        names=["ENTITY", "TIMESTAMP"])
    mock_execute.return_value = iter([mock_batch])
    total_rows, parts = generate_batch_pred_table_via_diskgraph(
        query=query, task_type=TaskType.BINARY_CLASSIFICATION,
        graph=get_mock_graph(), timestamps=[100, 200], forecast_length=1,
        lag_timesteps=None, use_yoy_labels=False,
        list_distinct_target_is_array=False, index_path=idx_path,
        output_root=out_path, input_timestamp_unit=TimeUnit.Nanoseconds,
        target_stype='numerical', entity_time_col=None)

    assert total_rows == 2
    df = pq.read_table(os.path.join(out_path, "part-00000.parquet")).to_pandas()
    # Temporal node-pred: TIMESTAMP carries the anchor and is preserved.
    assert list(df.columns) == ['ENTITY', 'TIMESTAMP', 'arange']


@patch('kumopql.rewriting.BPFilterTransformer', create=True)
@patch('diskgraph.execute_tt_plan')
def test_static_link_prediction_modern(mock_execute, mock_transformer,
                                       setup_dirs):
    idx_path, out_path = setup_dirs
    target_out_path = out_path + "_target"
    os.makedirs(target_out_path, exist_ok=True)

    query = get_mock_query()
    query.query_type = QueryType.STATIC

    mock_instance = MagicMock()
    mock_instance.generate_bp_query.return_value = query
    mock_transformer.return_value = mock_instance

    import pyarrow as pa

    # Modern static LHS emits TIMESTAMP (0) which is then dropped
    lhs_batch = pa.RecordBatch.from_arrays(
        [pa.array(["u1", "u2"]), pa.array([0, 0])],
        names=["ENTITY", "TIMESTAMP"])
    rhs_batch = pa.RecordBatch.from_arrays([pa.array(["i1", "i2"])],
                                           names=["ENTITY"])

    mock_execute.side_effect = [iter([lhs_batch]), iter([rhs_batch])]

    total_rows, parts = generate_batch_pred_table_via_diskgraph(
        query=query, task_type=TaskType.STATIC_LINK_PREDICTION,
        graph=get_mock_graph(), timestamps=[0], forecast_length=1,
        lag_timesteps=None, use_yoy_labels=False,
        list_distinct_target_is_array=True, index_path=idx_path,
        output_root=out_path, target_output_root=target_out_path,
        input_timestamp_unit=TimeUnit.Nanoseconds, target_stype='numerical',
        entity_time_col=None)

    # Static LP modern: this is the only case where the drop should fire —
    # GE-side final_cols is {ENTITY, TARGET}. The mock LHS emits TIMESTAMP=0
    # (executor placeholder), and the writer drops it.
    df_lhs = pq.read_table(os.path.join(out_path,
                                        "part-00000.parquet")).to_pandas()
    assert list(df_lhs.columns) == ["ENTITY", "arange"]

    df_rhs = pq.read_table(os.path.join(target_out_path,
                                        "part-00000.parquet")).to_pandas()
    assert list(df_rhs.columns) == ["TARGET", "arange"]


@patch('kumopql.rewriting.BPFilterTransformer', create=True)
@patch('diskgraph.execute_tt_plan')
def test_temporal_link_prediction_modern(mock_execute, mock_transformer,
                                         setup_dirs):
    idx_path, out_path = setup_dirs
    target_out_path = out_path + "_target"
    os.makedirs(target_out_path, exist_ok=True)

    query = get_mock_query()
    query.query_type = QueryType.TEMPORAL

    mock_instance = MagicMock()
    mock_instance.generate_bp_query.return_value = query
    mock_transformer.return_value = mock_instance

    import pyarrow as pa
    lhs_batch = pa.RecordBatch.from_arrays(
        [pa.array(["u1", "u2"]), pa.array([100, 200])],
        names=["ENTITY", "TIMESTAMP"])
    rhs_batch = pa.RecordBatch.from_arrays([pa.array(["i1", "i2"])],
                                           names=["ENTITY"])

    mock_execute.side_effect = [iter([lhs_batch]), iter([rhs_batch])]

    total_rows, parts = generate_batch_pred_table_via_diskgraph(
        query=query, task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        graph=get_mock_graph(), timestamps=[100, 200], forecast_length=1,
        lag_timesteps=None, use_yoy_labels=False,
        list_distinct_target_is_array=True, index_path=idx_path,
        output_root=out_path, target_output_root=target_out_path,
        input_timestamp_unit=TimeUnit.Nanoseconds, target_stype='numerical',
        entity_time_col=None)

    df_lhs = pq.read_table(os.path.join(out_path,
                                        "part-00000.parquet")).to_pandas()
    assert list(df_lhs.columns) == ["ENTITY", "TIMESTAMP", "arange"]

    df_rhs = pq.read_table(os.path.join(target_out_path,
                                        "part-00000.parquet")).to_pandas()
    assert list(df_rhs.columns) == ["TARGET", "arange"]


@patch('kumopql.rewriting.BPFilterTransformer', create=True)
@patch('diskgraph.execute_tt_plan')
def test_temporal_link_prediction_hstack(mock_execute, mock_transformer,
                                         setup_dirs):
    idx_path, out_path = setup_dirs
    query = get_mock_query()
    query.query_type = QueryType.TEMPORAL

    mock_instance = MagicMock()
    mock_instance.generate_bp_query.return_value = query
    mock_transformer.return_value = mock_instance

    import pyarrow as pa
    lhs_batch_1 = pa.RecordBatch.from_arrays(
        [pa.array(["u1", "u2"]), pa.array([100, 200])],
        names=["ENTITY", "TIMESTAMP"])
    lhs_batch_2 = pa.RecordBatch.from_arrays(
        [pa.array(["u3"]), pa.array([300])], names=["ENTITY", "TIMESTAMP"])
    rhs_batch_1 = pa.RecordBatch.from_arrays([pa.array(["i1"])],
                                             names=["ENTITY"])
    rhs_batch_2 = pa.RecordBatch.from_arrays([pa.array(["i2"])],
                                             names=["ENTITY"])

    mock_execute.side_effect = [
        iter([rhs_batch_1, rhs_batch_2]),
        iter([lhs_batch_1, lhs_batch_2])
    ]

    total_rows, parts = generate_batch_pred_table_via_diskgraph(
        query=query, task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        graph=get_mock_graph(), timestamps=[100, 200, 300], forecast_length=1,
        lag_timesteps=None, use_yoy_labels=False,
        list_distinct_target_is_array=False, index_path=idx_path,
        output_root=out_path, input_timestamp_unit=TimeUnit.Nanoseconds,
        target_stype='numerical', entity_time_col=None)

    assert total_rows == 3

    df = pq.read_table(os.path.join(out_path, "part-00000.parquet")).to_pandas()
    assert len(df) == 3
    assert list(df.columns) == ["ENTITY", "TIMESTAMP", "TARGET", "arange"]
    assert list(df['ENTITY']) == ["u1", "u2", "u3"]
    assert list(df['TARGET']) == ["i1", "i2", "i1"]
    assert list(df['TIMESTAMP']) == [100, 200, 300]
    assert list(df['arange']) == [0, 1, 2]


@patch('diskgraph.execute_tt_plan')
def test_forecasting_preserves_native_diskgraph_timestamps(
        mock_execute, setup_dirs):
    idx_path, out_path = setup_dirs
    query = get_mock_query()
    query.query_type = QueryType.TEMPORAL

    import pyarrow as pa

    mock_range = MagicMock()
    mock_range.timeframe = pd.Timedelta(days=1)
    query.get_combined_date_offset_range.return_value = mock_range

    mock_batch = pa.RecordBatch.from_arrays(
        [
            pa.array(["u1", "u2"]),
            pa.array([100_000_000_000, 200_000_000_000],
                     type=pa.timestamp('ns')),
            pa.array([[10.0, 20.0], [30.0, 40.0]]),
            pa.array([[100.0], [200.0]]),
        ],
        names=[
            "ENTITY",
            "TIMESTAMP",
            AUTOREGRESSION_COL,
            YOY_COL,
        ],
    )
    mock_execute.return_value = iter([mock_batch])

    total_rows, parts = generate_batch_pred_table_via_diskgraph(
        query=query, task_type=TaskType.FORECASTING, graph=get_mock_graph(),
        timestamps=[100, 200
                    ], forecast_length=1, lag_timesteps=1, use_yoy_labels=True,
        list_distinct_target_is_array=False, index_path=idx_path,
        output_root=out_path, input_timestamp_unit=TimeUnit.Seconds,
        target_stype='numerical', entity_time_col=None)

    assert total_rows == 2
    assert parts == 1
    df = pq.read_table(os.path.join(out_path, "part-00000.parquet")).to_pandas()
    assert list(df.columns) == [
        "ENTITY",
        "TIMESTAMP",
        AUTOREGRESSION_COL,
        YOY_COL,
        "arange",
    ]
    assert list(df["arange"]) == [0, 1]
    assert str(df["TIMESTAMP"].dtype) == "datetime64[ns]"
    assert list(df["TIMESTAMP"].astype("int64")) == [
        100_000_000_000,
        200_000_000_000,
    ]


# Regression: static node-pred where the entity table has a time_col.
# Mirror of `train_table_generator.maybe_add_entity_time_col_for_static_query`:
# the diskgraph executor emits a TIMESTAMP column populated with real
# per-row entity dates, and the BP writer must NOT drop it. Predict-time
# code reads `train_table.source_name` and projects TIMESTAMP, so dropping
# it produces a `pyarrow.lib.ArrowInvalid: No match for FieldRef.Name(TIMESTAMP)`
# at predict time (the bug this fix addresses).
@patch('diskgraph.execute_tt_plan')
def test_static_node_pred_with_entity_time_col_keeps_timestamp(
        mock_execute, setup_dirs):
    idx_path, out_path = setup_dirs
    query = get_mock_query()
    # Static + entity has a time_col → executor emits real per-row dates.
    query.query_type = QueryType.STATIC

    import pyarrow as pa
    mock_batch = pa.RecordBatch.from_arrays(
        [pa.array(["u1", "u2"]),
         pa.array([1700000000, 1700086400])], names=["ENTITY", "TIMESTAMP"])
    mock_execute.return_value = iter([mock_batch])

    total_rows, parts = generate_batch_pred_table_via_diskgraph(
        query=query, task_type=TaskType.BINARY_CLASSIFICATION,
        graph=get_mock_graph(), timestamps=[0], forecast_length=1,
        lag_timesteps=None, use_yoy_labels=False,
        list_distinct_target_is_array=False, index_path=idx_path,
        output_root=out_path, input_timestamp_unit=TimeUnit.Nanoseconds,
        target_stype='numerical', entity_time_col='DATE')

    assert total_rows == 2
    df = pq.read_table(os.path.join(out_path, "part-00000.parquet")).to_pandas()
    # The previous (buggy) behavior dropped TIMESTAMP for any STATIC
    # non-legacy-LP. With the fix, the drop only fires for static-LP-modern,
    # so static node-pred preserves TIMESTAMP — matching the GE final_cols
    # contract `{ENTITY, TIMESTAMP}` for BC.
    assert list(df.columns) == ["ENTITY", "TIMESTAMP", "arange"]
    assert list(df['TIMESTAMP']) == [1700000000, 1700086400]


# Verify the writer-side drop still fires for static-LP-modern. The mock
# emits TIMESTAMP=0 (executor placeholder for the LHS pass when
# entity_time_col is None), and the writer must drop it because GE's
# final_cols for static LP modern is {ENTITY, TARGET}.
@patch('kumopql.rewriting.BPFilterTransformer', create=True)
@patch('diskgraph.execute_tt_plan')
def test_static_link_pred_modern_drops_timestamp(mock_execute, mock_transformer,
                                                 setup_dirs):
    idx_path, out_path = setup_dirs
    target_out_path = out_path + "_target_drop"
    os.makedirs(target_out_path, exist_ok=True)

    query = get_mock_query()
    query.query_type = QueryType.STATIC

    mock_instance = MagicMock()
    mock_instance.generate_bp_query.return_value = query
    mock_transformer.return_value = mock_instance

    import pyarrow as pa

    # Even when entity_time_col is provided (entity has a time_col) the
    # drop must still fire for static LP modern, since GE's final_cols is
    # {ENTITY, TARGET}.
    lhs_batch = pa.RecordBatch.from_arrays(
        [pa.array(["u1", "u2"]), pa.array([0, 0])],
        names=["ENTITY", "TIMESTAMP"])
    rhs_batch = pa.RecordBatch.from_arrays([pa.array(["i1", "i2"])],
                                           names=["ENTITY"])
    mock_execute.side_effect = [iter([lhs_batch]), iter([rhs_batch])]

    generate_batch_pred_table_via_diskgraph(
        query=query, task_type=TaskType.STATIC_LINK_PREDICTION,
        graph=get_mock_graph(), timestamps=[0], forecast_length=1,
        lag_timesteps=None, use_yoy_labels=False,
        list_distinct_target_is_array=True, index_path=idx_path,
        output_root=out_path, target_output_root=target_out_path,
        input_timestamp_unit=TimeUnit.Nanoseconds, target_stype='numerical',
        entity_time_col='DATE')

    df_lhs = pq.read_table(os.path.join(out_path,
                                        "part-00000.parquet")).to_pandas()
    assert list(df_lhs.columns) == ["ENTITY", "arange"]


# Verify the entity_time_col argument is forwarded through to
# DiskGraphPlanCompiler.compile() for node-pred. Without this signal the
# diskgraph executor emits TIMESTAMP=0 placeholders even when the entity
# table has a real time_column — which silently corrupts time-aware
# sampling at predict time.
@patch('diskgraph.DiskGraphPlanCompiler')
@patch('diskgraph.execute_tt_plan')
def test_entity_time_col_is_forwarded_to_compiler(mock_execute,
                                                  mock_compiler_cls,
                                                  setup_dirs):
    idx_path, out_path = setup_dirs
    query = get_mock_query()
    query.query_type = QueryType.STATIC

    mock_compiler = MagicMock()
    mock_compiler.compile.return_value = '{}'
    mock_compiler_cls.return_value = mock_compiler

    import pyarrow as pa
    mock_execute.return_value = iter(
        [pa.RecordBatch.from_arrays([pa.array(["u1"])], names=["ENTITY"])])

    generate_batch_pred_table_via_diskgraph(
        query=query, task_type=TaskType.BINARY_CLASSIFICATION,
        graph=get_mock_graph(), timestamps=[0], forecast_length=1,
        lag_timesteps=None, use_yoy_labels=False,
        list_distinct_target_is_array=False, index_path=idx_path,
        output_root=out_path, input_timestamp_unit=TimeUnit.Nanoseconds,
        target_stype='numerical', entity_time_col='ROW_DATE')

    mock_compiler.compile.assert_called_once()
    compile_kwargs = mock_compiler.compile.call_args.kwargs
    assert compile_kwargs.get('entity_time_col') == 'ROW_DATE'


@patch('kumopql.rewriting.BPFilterTransformer', create=True)
@patch('diskgraph.DiskGraphPlanCompiler')
@patch('diskgraph.execute_tt_plan')
def test_lp_bp_layout_decouples_separate_output_from_train_target_flag(
        mock_execute, mock_compiler_cls, mock_transformer, setup_dirs):
    idx_path, out_path = setup_dirs
    target_out_path = out_path + "_target_decoupled"
    os.makedirs(target_out_path, exist_ok=True)

    query = get_mock_query()
    query.query_type = QueryType.STATIC

    mock_instance = MagicMock()
    mock_instance.generate_bp_query.return_value = query
    mock_transformer.return_value = mock_instance

    mock_compiler = MagicMock()
    mock_compiler.compile.side_effect = ['lhs-plan', 'rhs-plan']
    mock_compiler_cls.return_value = mock_compiler

    import pyarrow as pa
    lhs_batch = pa.RecordBatch.from_arrays(
        [pa.array(["u1", "u2"]), pa.array([0, 0])],
        names=["ENTITY", "TIMESTAMP"])
    rhs_batch = pa.RecordBatch.from_arrays([pa.array(["i1", "i2"])],
                                           names=["ENTITY"])
    mock_execute.side_effect = [iter([lhs_batch]), iter([rhs_batch])]

    generate_batch_pred_table_via_diskgraph(
        query=query, task_type=TaskType.STATIC_LINK_PREDICTION,
        graph=get_mock_graph(), timestamps=[0], forecast_length=1,
        lag_timesteps=None, use_yoy_labels=False,
        list_distinct_target_is_array=False, index_path=idx_path,
        output_root=out_path, target_output_root=target_out_path,
        input_timestamp_unit=TimeUnit.Nanoseconds, target_stype='numerical',
        entity_time_col=None, lp_bp_layout='separate_lhs_rhs')

    compile_calls = mock_compiler.compile.call_args_list
    assert compile_calls[0].kwargs['lp_bp_layout'] == 'separate_lhs_rhs'
    assert compile_calls[1].kwargs['lp_bp_layout'] == 'separate_lhs_rhs'
    assert compile_calls[0].kwargs['list_distinct_target_is_array'] is False

    df_lhs = pq.read_table(os.path.join(out_path,
                                        "part-00000.parquet")).to_pandas()
    assert list(df_lhs.columns) == ["ENTITY", "arange"]

    df_rhs = pq.read_table(os.path.join(target_out_path,
                                        "part-00000.parquet")).to_pandas()
    assert list(df_rhs.columns) == ["TARGET", "arange"]


def test_resolve_lp_bp_layout_defaults_to_legacy_branch():
    assert _resolve_lp_bp_layout(True, None) == LP_BP_LAYOUT_SEPARATE_LHS_RHS
    assert _resolve_lp_bp_layout(False, None) == LP_BP_LAYOUT_HSTACK


def test_resolve_lp_bp_layout_passthrough():
    # Non-array (scalar) targets keep the requested layout unchanged.
    assert _resolve_lp_bp_layout(
        False, LP_BP_LAYOUT_SEPARATE_LHS_RHS) == LP_BP_LAYOUT_SEPARATE_LHS_RHS
    assert _resolve_lp_bp_layout(False,
                                 LP_BP_LAYOUT_HSTACK) == LP_BP_LAYOUT_HSTACK


def test_resolve_lp_bp_layout_array_target_normalizes_to_separate():
    # Array-target LP has exactly one valid layout: a requested hstack is
    # normalized to separate_lhs_rhs, never passed through. (Under TT-NG-only
    # LP is always array-target, so hstack is never actually produced.)
    assert _resolve_lp_bp_layout(
        True, LP_BP_LAYOUT_HSTACK) == LP_BP_LAYOUT_SEPARATE_LHS_RHS
    assert _resolve_lp_bp_layout(
        True, LP_BP_LAYOUT_SEPARATE_LHS_RHS) == LP_BP_LAYOUT_SEPARATE_LHS_RHS


def test_resolve_lp_bp_layout_rejects_unknown():
    with pytest.raises(ValueError, match="lp_bp_layout must be one of"):
        _resolve_lp_bp_layout(False, "nonsense")


@pytest.mark.parametrize('lp_bp_layout',
                         [LP_BP_LAYOUT_SEPARATE_LHS_RHS, LP_BP_LAYOUT_HSTACK])
def test_current_pred_passes_runtime_options_to_generator(
        monkeypatch, tmp_path, lp_bp_layout):
    # On DiskGraph the executor honors lp_bp_layout internally; the pred-table
    # entrypoint only forwards the runtime options to generate_table.
    config = MagicMock()
    config.graph.tables = []
    config.year_over_year = False

    generator = _StaticPredictionTableGenerator.__new__(
        _StaticPredictionTableGenerator)
    generator.generate_table = MagicMock(return_value=pd.DataFrame())
    generator.get_validate_train_table_response = MagicMock(
        return_value=ValidationResponse())

    pquery = PredictiveQueryV2.__new__(PredictiveQueryV2)
    pquery.lag_timesteps = None
    pquery.list_distinct_target_is_array = False
    pquery.lp_bp_layout = lp_bp_layout
    pquery.get_batch_pred_table_generator = MagicMock(return_value=generator)
    monkeypatch.setattr(PredictiveQueryV2, 'from_config',
                        classmethod(lambda cls, cfg: pquery))

    current_pred.generate_current_pred_table_generic(
        predictive_query_config=config,
        output_file_path=str(tmp_path / 'pred_table'),
        pred_table_validation_response_path=str(tmp_path / 'validation'),
        destination=str(tmp_path),
        target_file_path=str(tmp_path / 'targets'),
    )

    kwargs = generator.generate_table.call_args.kwargs
    assert kwargs['lp_bp_layout'] == lp_bp_layout
    assert kwargs['list_distinct_target_is_array'] is False


# ---------------------------------------------------------------------------
# Serving pred-table materialization from the persisted TT-NG gen bundle.
# ---------------------------------------------------------------------------


def _write_raw_bundle_parquet(path: str) -> None:
    """Write a raw DG gen-bundle ``tt.parquet`` (slot-0 ENTITY1/TARGET1, no
    arange) -- the internal schema the predict gen persists."""
    import pyarrow as pa
    table = pa.table({
        'ENTITY1': pa.array([10, 11, 12], type=pa.int64()),
        'TIMESTAMP': pa.array([100, 200, 300], type=pa.int64()),
        'TARGET1': pa.array([1, 0, 1], type=pa.int64()),
    })
    os.makedirs(path)
    pq.write_table(table, os.path.join(path, 'part-00000.parquet'))


def test_write_serving_pred_parquet_node_pred_drops_target(tmp_path):
    """Node-pred serving table: ENTITY1->ENTITY, TARGET dropped, arange added."""
    from kumo.data.generic import diskgraph_batch_pred
    bundle = str(tmp_path / 'tt.parquet')
    _write_raw_bundle_parquet(bundle)
    out_root = str(tmp_path / 'pred_table')

    rows = diskgraph_batch_pred._write_serving_pred_parquet_from_bundle(
        bundle_tt_parquet=bundle,
        output_root=out_root,
        query_type=QueryType.TEMPORAL,
        task_type=TaskType.BINARY_CLASSIFICATION,
        list_distinct_target_is_array=False,
    )

    assert rows == 3
    df = pd.read_parquet(os.path.join(out_root, 'part-00000.parquet'))
    assert set(df.columns) == {'ENTITY', 'TIMESTAMP', 'arange'}
    assert df['arange'].tolist() == [0, 1, 2]
    assert df['ENTITY'].tolist() == [10, 11, 12]


def test_write_serving_pred_parquet_lp_keeps_target(tmp_path):
    """LP serving table: TARGET1->TARGET kept, arange added."""
    from kumo.data.generic import diskgraph_batch_pred
    bundle = str(tmp_path / 'tt.parquet')
    _write_raw_bundle_parquet(bundle)
    out_root = str(tmp_path / 'pred_table')

    rows = diskgraph_batch_pred._write_serving_pred_parquet_from_bundle(
        bundle_tt_parquet=bundle,
        output_root=out_root,
        query_type=QueryType.TEMPORAL,
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        list_distinct_target_is_array=False,
    )

    assert rows == 3
    df = pd.read_parquet(os.path.join(out_root, 'part-00000.parquet'))
    assert set(df.columns) == {'ENTITY', 'TIMESTAMP', 'TARGET', 'arange'}
    assert df['TARGET'].tolist() == [1, 0, 1]
