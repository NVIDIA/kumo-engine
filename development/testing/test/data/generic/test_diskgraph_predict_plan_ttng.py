# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TT-NG predict-plan builder.

TT-NG only. Verifies that ``build_ttng_predict_plan`` declares the FK-free
``lp_resolution`` for link prediction (LHS entity ONLY — no TARGET positive at
predict time) and declares NONE for node prediction (so the wheel falls back to
FK-edge ingest).

The diskgraph ``DiskGraphPlanCompiler`` is stubbed so the test exercises the
kumo-side plan post-processing, not the compiler.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from test.data.generic.ttng_v3_test_bundle import write_v3_bundle
from types import SimpleNamespace

import pandas as pd
import pytest
from kumo_core.types import TaskType
from kumoapi.pquery import QueryType


@pytest.fixture
def stub_compiler(monkeypatch):
    """Stub ``diskgraph.DiskGraphPlanCompiler`` to return a plan JSON with an
    ``entity`` block (so ``_apply_lp_resolution_plan`` can read the entity
    table off the plan) and record the compile kwargs."""
    captured = {}

    class _StubCompiler:
        def __init__(self, *, index_path):
            captured['index_path'] = index_path

        def compile(self, **kwargs):
            captured['kwargs'] = kwargs
            plan = {
                'entity': {
                    'table': 'users'
                },
                'task_type': kwargs.get('task_type'),
                'is_prediction': kwargs.get('is_prediction'),
            }
            return json.dumps(plan)

    # diskgraph is imported lazily inside the helper as
    # ``from diskgraph import DiskGraphPlanCompiler``; patch the module attr.
    diskgraph_mod = sys.modules.get('diskgraph')
    if diskgraph_mod is None:
        diskgraph_mod = SimpleNamespace()
        monkeypatch.setitem(sys.modules, 'diskgraph', diskgraph_mod)
    monkeypatch.setattr(diskgraph_mod, 'DiskGraphPlanCompiler', _StubCompiler,
                        raising=False)
    return captured


def _build(task_type, *, list_array, stub_compiler):
    bp = importlib.import_module('kumo.data.generic.diskgraph_batch_pred')
    plan_json = bp.build_ttng_predict_plan(
        SimpleNamespace(
            entity_column='users.user_id'),  # only entity_column read
        index_path='/index',
        task_type=task_type,
        graph=SimpleNamespace(),
        timestamps=[],
        list_distinct_target_is_array=list_array,
        input_timestamp_unit=object(),
        target_stype='ID',
        entity_time_col=None,
    )
    return json.loads(plan_json), stub_compiler['kwargs']


def test_node_prediction_plan_has_entity_resolution(stub_compiler):
    plan, kwargs = _build(TaskType.BINARY_CLASSIFICATION, list_array=False,
                          stub_compiler=stub_compiler)
    assert plan['lp_resolution'] == [{
        'src_col': 'ENTITY1',
        'dst_table': 'users',
        'kind': 'scalar',
        'node_id_col': 'entity_node_id',
    }]
    assert kwargs['is_prediction'] is True
    # Node prediction does not pass an LP prediction mode.
    assert 'lp_prediction_mode' not in kwargs


def test_lp_plan_carries_entity_only_resolution(stub_compiler):
    plan, kwargs = _build(TaskType.STATIC_LINK_PREDICTION, list_array=True,
                          stub_compiler=stub_compiler)
    assert kwargs['lp_prediction_mode'] == 'LHS'
    assert kwargs['list_distinct_target_is_array'] is True

    res = plan['lp_resolution']
    # Exactly one spec — the LHS entity. No TARGET resolution at predict time
    # (no positive), so no FK edge / no leak.
    assert len(res) == 1
    spec = res[0]
    assert spec['src_col'] == 'ENTITY1'
    assert spec['node_id_col'] == 'entity_node_id'
    assert spec['kind'] == 'scalar'
    assert spec['dst_table'] == 'users'  # read off the plan's entity block


def test_temporal_lp_plan_carries_entity_only_resolution(stub_compiler):
    plan, _ = _build(TaskType.TEMPORAL_LINK_PREDICTION, list_array=True,
                     stub_compiler=stub_compiler)
    res = plan['lp_resolution']
    assert [s['src_col'] for s in res] == ['ENTITY1']


def test_lp_plan_honors_array_flag_from_config(stub_compiler):
    """Array-mode is decided upstream by ``allow_array_targets()`` (always True
    under TT-NG) and threaded through as the config value, so the predict-plan
    builder simply honors it and threads it into the compiler + resolution."""
    plan, kwargs = _build(TaskType.STATIC_LINK_PREDICTION, list_array=True,
                          stub_compiler=stub_compiler)
    assert kwargs['list_distinct_target_is_array'] is True
    res = plan['lp_resolution']
    assert [s['src_col'] for s in res] == ['ENTITY1']
    assert res[0]['node_id_col'] == 'entity_node_id'


def test_lp_predict_fails_loud_when_resolution_missing(stub_compiler):
    """A deserialized-False LP config that reaches the predict-plan builder must
    NOT silently emit a plan with no FK-free lp_resolution -- it must fail loud
    (the retained downstream guard for the removed fail-loud check)."""
    with pytest.raises(RuntimeError, match='requires array-target resolution'):
        _build(TaskType.STATIC_LINK_PREDICTION, list_array=False,
               stub_compiler=stub_compiler)


def test_resolve_lp_bp_layout_normalizes_persisted_hstack_under_array():
    """A persisted lp_bp_layout='hstack' under array-mode LP must be normalized
    to the separate LHS/RHS (array) layout, never silently used."""
    bp = importlib.import_module('kumo.data.generic.diskgraph_batch_pred')
    resolved = bp._resolve_lp_bp_layout(
        True,  # list_distinct_target_is_array (array-mode / TT-NG LP)
        bp.LP_BP_LAYOUT_HSTACK,
    )
    assert resolved == bp.LP_BP_LAYOUT_SEPARATE_LHS_RHS
    # Non-array (legacy) still honors an explicit persisted hstack.
    assert bp._resolve_lp_bp_layout(
        False, bp.LP_BP_LAYOUT_HSTACK) == bp.LP_BP_LAYOUT_HSTACK
    # Invalid persisted value still rejected.
    with pytest.raises(ValueError, match='lp_bp_layout must be one of'):
        bp._resolve_lp_bp_layout(True, 'bogus')


def test_forecasting_ttng_prediction_uses_kept_staging(monkeypatch, tmp_path):
    bp = importlib.import_module('kumo.data.generic.diskgraph_batch_pred')
    captured = {}

    class _TimeUnit:
        Nanoseconds = 'Nanoseconds'
        Microseconds = 'Microseconds'
        Milliseconds = 'Milliseconds'
        Seconds = 'Seconds'

    class _Compiler:
        def __init__(self, *, index_path):
            captured['compiler_index_path'] = index_path

        def compile(self, **kwargs):
            captured['compile'] = kwargs
            return json.dumps({'task_type': kwargs['task_type']})

    def _execute_tt_gen(plan_json, index_path, tt_id):
        captured['execute'] = {
            'plan_json': plan_json,
            'index_path': index_path,
            'tt_id': tt_id,
        }
        # Write the current producer contract. A v1-shaped empty manifest
        # would correctly fail the strict-v3 publication boundary.
        expected_staging = bp.ttng_staging_dir(index_path, tt_id)
        staging = write_v3_bundle(
            os.path.dirname(expected_staging),
            tt_id,
            rows=7,
            ingest_spec_schema_version='diskgraph.tt_ingest_spec.v3',
        )
        assert staging == expected_staging
        return {'row_count': 7}

    def _read_tt_parquet_batches(path):
        import pyarrow.parquet as pq
        yield from pq.ParquetDataset(path).read().to_batches()

    diskgraph_mod = SimpleNamespace(
        DiskGraphPlanCompiler=_Compiler,
        TimeUnit=_TimeUnit,
        execute_tt_gen=_execute_tt_gen,
        read_tt_parquet_batches=_read_tt_parquet_batches,
    )
    monkeypatch.setitem(sys.modules, 'diskgraph', diskgraph_mod)
    monkeypatch.setattr(
        bp,
        'get_lag_timestamps',
        lambda pt, length, step: [pt - step],
    )
    monkeypatch.setattr(bp, 'get_yoy_timestamps', lambda *_a, **_kw: ([], None))

    index_path = str(tmp_path / 'idx')
    output_root = str(tmp_path / 'home' / 'pred_table')
    query = SimpleNamespace(
        query_type=QueryType.TEMPORAL,
        get_combined_date_offset_range=lambda: SimpleNamespace(
            timeframe=pd.Timedelta(days=5),
            unit=SimpleNamespace(value='days'),
        ),
    )
    rows, parts = bp.generate_batch_pred_table_via_diskgraph_ttng(
        query=query,
        task_type=TaskType.FORECASTING,
        graph=SimpleNamespace(),
        timestamps=[1_700_000_000_000_000_000],
        forecast_length=3,
        lag_timesteps=1,
        use_yoy_labels=False,
        list_distinct_target_is_array=False,
        index_path=index_path,
        output_root=output_root,
        input_timestamp_unit=_TimeUnit.Nanoseconds,
        target_stype='numerical',
        entity_time_col='ts',
        job_id='bp-job',
    )

    assert (rows, parts) == (7, 1)
    assert captured['execute']['index_path'] == index_path
    assert captured['compiler_index_path'] == index_path
    assert captured['execute']['tt_id'] == 'pred_bp-job'
    assert captured['compile']['is_prediction'] is True
    assert captured['compile']['task_type'] == TaskType.FORECASTING.value
    assert captured['compile']['lag_offsets_secs'] == [-432000]
    # The whole staging bundle was persisted durably beside the pred table.
    durable = bp.ttng_persisted_pred_staging_dir(output_root)
    from kumo.data.generic.diskgraph_train_table import (
        resolve_durable_ttng_bundle,
    )
    assert os.path.isfile(os.path.join(durable, '_current.json'))
    durable = resolve_durable_ttng_bundle(durable)
    assert os.path.isdir(os.path.join(durable, 'tt.parquet'))
    for name in ('tt_ingest_spec.json', 'manifest.json'):
        assert os.path.isfile(os.path.join(durable, name)), name
