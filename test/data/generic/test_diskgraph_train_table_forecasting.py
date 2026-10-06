# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from kumo.data.generic.diskgraph_train_table import (
    generate_train_table_via_diskgraph,
    generate_train_table_via_diskgraph_ttng,
)


@pytest.fixture
def auto_setup_kumo_backend():
    return False


def test_forecasting_train_generation_passes_history_offsets(
        monkeypatch, tmp_path):
    captured = {}

    class _SplitSpec:
        def __init__(self, **kwargs):
            self.kind = kwargs['kind']

    class _Compiler:
        def compile(self, **kwargs):
            captured.update(kwargs)
            return '{"time_col_name": "TIMESTAMP"}'

    diskgraph = SimpleNamespace(
        DiskGraphPlanCompiler=lambda: _Compiler(),
        SplitSpec=_SplitSpec,
        execute_tt_plan_with_artifacts=lambda **_kwargs: {
            'manifest_path': str(tmp_path / 'manifest.json'),
        },
    )
    monkeypatch.setitem(sys.modules, 'diskgraph', diskgraph)

    import kumo_core.training.table as core_train_table

    from kumo.data.generic import diskgraph_train_table as dtt
    monkeypatch.setattr(core_train_table, 'diskgraph', diskgraph)
    monkeypatch.setattr(dtt.fsutil, 'exists', lambda _path: True)
    monkeypatch.setattr(
        dtt,
        'get_lag_timestamps',
        lambda pt, length, step:
        [pt - ((length - i) * step) for i in range(length)],
    )
    monkeypatch.setattr(
        dtt,
        'get_yoy_timestamps',
        lambda pt, length, step:
        ([pt - pd.Timedelta(days=364) + (i * step)
          for i in range(length)], None),
    )

    query = MagicMock()
    query.entity_column = 'users.user_id'
    query.year_over_year = True
    query.get_combined_date_offset_range.return_value = SimpleNamespace(
        timeframe=pd.Timedelta(days=7))

    generate_train_table_via_diskgraph(
        query=query,
        graph=None,
        timestamps=[0],
        task_type='forecasting',
        forecast_length=3,
        lag_timesteps=2,
        list_distinct_target_is_array=False,
        include_frequency=False,
        entity_time_col=None,
        index_path=str(tmp_path / 'index'),
        output_path=str(tmp_path / 'out'),
        split_config={'ratio': [0.8, 0.1, 0.1]},
        split_family='RandomSplit',
        downsampling=None,
        train_table_edges_config=None,
        input_timestamp_unit=object(),
        target_stype='numerical',
    )

    assert 'future_offsets_secs' not in captured
    assert captured['lag_offsets_secs'] == [-1209600, -604800]
    assert captured['yoy_offsets_secs'] == [
        -31449600,
        -30844800,
        -30240000,
    ]

    captured.clear()
    query.year_over_year = False
    generate_train_table_via_diskgraph(
        query=query,
        graph=None,
        timestamps=[0],
        task_type='forecasting',
        forecast_length=3,
        lag_timesteps=2,
        list_distinct_target_is_array=False,
        include_frequency=False,
        entity_time_col=None,
        index_path=str(tmp_path / 'index'),
        output_path=str(tmp_path / 'out'),
        split_config={'ratio': [0.8, 0.1, 0.1]},
        split_family='RandomSplit',
        downsampling=None,
        train_table_edges_config=None,
        input_timestamp_unit=object(),
        target_stype='numerical',
    )

    assert 'future_offsets_secs' not in captured
    assert captured['lag_offsets_secs'] == [-1209600, -604800]
    assert captured['yoy_offsets_secs'] is None

    captured.clear()
    generate_train_table_via_diskgraph(
        query=query,
        graph=None,
        timestamps=[0],
        task_type='regression',
        forecast_length=1,
        lag_timesteps=None,
        list_distinct_target_is_array=False,
        include_frequency=False,
        entity_time_col=None,
        index_path=str(tmp_path / 'index'),
        output_path=str(tmp_path / 'out'),
        split_config={'ratio': [0.8, 0.1, 0.1]},
        split_family='RandomSplit',
        downsampling=None,
        train_table_edges_config=None,
        input_timestamp_unit=object(),
        target_stype='numerical',
    )

    assert 'future_offsets_secs' not in captured


def test_forecasting_ttng_train_generation_passes_offsets(
        monkeypatch, tmp_path):
    captured = {}

    class _SplitSpec:
        def __init__(self, **kwargs):
            self.kind = kwargs['kind']

    class _Compiler:
        def compile(self, **kwargs):
            captured.update(kwargs)
            return '{"time_col_name": "TIMESTAMP"}'

    diskgraph = SimpleNamespace(
        DiskGraphPlanCompiler=lambda: _Compiler(),
        SplitSpec=_SplitSpec,
        execute_tt_gen=lambda *_args: {
            'staging_dir': str(tmp_path / 'staging'),
            'tt_parquet_path': str(tmp_path / 'staging' / 'tt.parquet'),
            'row_count': 0,
        },
    )
    monkeypatch.setitem(sys.modules, 'diskgraph', diskgraph)

    import kumo_core.training.table as core_train_table

    from kumo.data.generic import diskgraph_train_table as dtt
    monkeypatch.setattr(core_train_table, 'diskgraph', diskgraph)
    monkeypatch.setattr(dtt.fsutil, 'exists', lambda _path: True)
    monkeypatch.setattr(
        dtt,
        'get_lag_timestamps',
        lambda pt, length, step:
        [pt - ((length - i) * step) for i in range(length)],
    )
    monkeypatch.setattr(
        dtt,
        'get_yoy_timestamps',
        lambda pt, length, step:
        ([pt - pd.Timedelta(days=364) + (i * step)
          for i in range(length)], None),
    )

    query = MagicMock()
    query.entity_column = 'users.user_id'
    query.get_combined_date_offset_range.return_value = SimpleNamespace(
        timeframe=pd.Timedelta(days=7))

    generate_train_table_via_diskgraph_ttng(
        query=query,
        graph=None,
        timestamps=[0],
        task_type='forecasting',
        forecast_length=3,
        lag_timesteps=2,
        year_over_year=True,
        list_distinct_target_is_array=False,
        include_frequency=False,
        entity_time_col=None,
        index_path=str(tmp_path / 'index'),
        output_path=str(tmp_path / 'out'),
        tt_id='tt-job',
        split_config={'ratio': [0.8, 0.1, 0.1]},
        split_family='RandomSplit',
        downsampling=None,
        input_timestamp_unit=object(),
        target_stype='numerical',
    )

    assert captured['future_offsets_secs'] == [0, 604800, 1209600]
    assert captured['lag_offsets_secs'] == [-1209600, -604800]
    assert captured['yoy_offsets_secs'] == [
        -31449600,
        -30844800,
        -30240000,
    ]

    captured.clear()
    generate_train_table_via_diskgraph_ttng(
        query=query,
        graph=None,
        timestamps=[0],
        task_type='forecasting',
        forecast_length=3,
        lag_timesteps=2,
        year_over_year=False,
        list_distinct_target_is_array=False,
        include_frequency=False,
        entity_time_col=None,
        index_path=str(tmp_path / 'index'),
        output_path=str(tmp_path / 'out'),
        tt_id='tt-job',
        split_config={'ratio': [0.8, 0.1, 0.1]},
        split_family='RandomSplit',
        downsampling=None,
        input_timestamp_unit=object(),
        target_stype='numerical',
    )

    assert captured['future_offsets_secs'] == [0, 604800, 1209600]
    assert captured['lag_offsets_secs'] == [-1209600, -604800]
    assert captured['yoy_offsets_secs'] is None


def test_ttng_link_prediction_keeps_atemporal_rhs_flag_false(
        monkeypatch, tmp_path):
    captured = {}

    class _SplitSpec:
        def __init__(self, **kwargs):
            self.kind = kwargs['kind']

    class _Compiler:
        def compile(self, **_kwargs):
            return json.dumps({'entity': {'table': 'Customer'}})

    def _execute_tt_gen(plan_json, *_args):
        captured.update(json.loads(plan_json))
        return {
            'staging_dir': str(tmp_path / 'staging'),
            'tt_parquet_path': str(tmp_path / 'staging' / 'tt.parquet'),
            'row_count': 0,
        }

    diskgraph = SimpleNamespace(
        DiskGraphPlanCompiler=lambda: _Compiler(),
        SplitSpec=_SplitSpec,
        execute_tt_gen=_execute_tt_gen,
    )
    monkeypatch.setitem(sys.modules, 'diskgraph', diskgraph)

    import kumo_core.training.table as core_train_table

    from kumo.data.generic import diskgraph_train_table as dtt
    monkeypatch.setattr(core_train_table, 'diskgraph', diskgraph)
    monkeypatch.setattr(dtt.fsutil, 'exists', lambda _path: True)
    monkeypatch.setattr(
        dtt,
        'get_rhs_lp_table',
        lambda *_args, **_kwargs: 'Product',
    )

    class _AtemporalTable:
        pkey_name = 'id'

        @staticmethod
        def has_time_col():
            return False

        @staticmethod
        def has_end_time_col():
            return False

    generate_train_table_via_diskgraph_ttng(
        query=SimpleNamespace(entity_column='Customer.id'),
        graph={
            'Customer': _AtemporalTable(),
            'Product': _AtemporalTable(),
        },
        timestamps=[0],
        task_type='link_prediction',
        forecast_length=1,
        lag_timesteps=None,
        list_distinct_target_is_array=True,
        include_frequency=False,
        entity_time_col=None,
        index_path=str(tmp_path / 'index'),
        output_path=str(tmp_path / 'out'),
        tt_id='tt-job',
        split_config={'ratio': [0.8, 0.1, 0.1]},
        split_family='RandomSplit',
        downsampling=None,
        input_timestamp_unit=object(),
        target_stype='categorical',
    )

    assert captured['lp_rhs_is_temporal'] is False
