# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock

import pandas as pd
import pyarrow.parquet as pq
import pytest
from kumo_core.types import TaskType
from kumoapi.pquery import QueryType

from kumo.data.generic import generate_current_pred_table as current_pred
from kumo.data.generic.diskgraph_batch_pred import (
    LP_BP_LAYOUT_HSTACK,
    LP_BP_LAYOUT_SEPARATE_LHS_RHS,
    _resolve_lp_bp_layout,
)
from kumo.pquery.batch_pred_table_generator import PredictionTableGeneratorV2
from kumo.pquery.predictive_query import PredictiveQueryV2
from kumo.validation import ValidationResponse


@pytest.fixture
def auto_setup_kumo_backend():
    return False


class _StaticPredictionTableGenerator(PredictionTableGeneratorV2):
    @property
    def query_type(self):
        return QueryType.STATIC


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
