# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kumoapi.model_plan import TrainingTableGenerationPlan

from kumo.engine.pquery import prediction_table


def test_compile_carries_resolved_forecast_options(monkeypatch, tmp_path):
    parsed = SimpleNamespace(query_type='temporal', task_type='forecasting')
    snapshot = SimpleNamespace(graph_config=object(),
                               index_path=tmp_path / 'index', id='snapshot')
    compiler_options = {
        'lag_offsets_secs': (-86400, ),
        'future_offsets_secs': (0, 86400, 172800),
    }
    task_dataset = object()
    plan = SimpleNamespace(
        graph_index_id='snapshot',
        task_dataset=task_dataset,
        train_table=SimpleNamespace(
            compiler_options=compiler_options,
            timestamps=(
                1_735_603_200_000_000_000,
                1_735_516_800_000_000_000,
            ),
        ),
    )
    monkeypatch.setattr(prediction_table, 'normalize_graph',
                        lambda config: 'graph')
    source_maximum = 1_735_689_600_000_000_000
    monkeypatch.setattr(prediction_table, 'latest_temporal_timestamp',
                        lambda query, index_path: source_maximum)
    resolve_timestamp = Mock(return_value=(123, ))
    monkeypatch.setattr('kumo_core.prediction_options.prediction_timestamps',
                        resolve_timestamp)
    build_args = {}

    def build(**kwargs):
        build_args.update(kwargs)
        return plan

    monkeypatch.setattr(prediction_table, 'build_native_pquery_task_plan', build)
    captured = []
    monkeypatch.setattr(
        prediction_table, 'generate_prediction_table',
        lambda request: captured.append(request) or SimpleNamespace())

    request, _, graph_id = prediction_table._compile_prediction_table(
        snapshot,
        parsed,
        prediction_time=None,
        run_id='forecast',
        scratch_dir=Path(tmp_path),
        artifact_dir=Path(tmp_path),
    )

    assert graph_id == 'snapshot'
    assert request.timestamps == (source_maximum, )
    assert build_args['timestamps'] == (source_maximum, )
    resolve_timestamp.assert_not_called()
    assert request.task_dataset is task_dataset
    assert request.compiler_options == compiler_options
    assert captured == [request]


def test_compile_preserves_explicit_prediction_time(monkeypatch, tmp_path):
    parsed = SimpleNamespace(query_type='temporal', task_type='forecasting')
    snapshot = SimpleNamespace(graph_config=object(),
                               index_path=tmp_path / 'index', id='snapshot')
    anchor = datetime(2024, 12, 31)
    plan = SimpleNamespace(
        graph_index_id='snapshot',
        task_dataset=object(),
        train_table=SimpleNamespace(
            compiler_options=None,
            timestamps=(1_735_603_200_000_000_000, ),
        ),
    )
    resolve_timestamp = Mock(return_value=(123, ))
    monkeypatch.setattr(prediction_table, 'normalize_graph',
                        lambda config: 'graph')
    monkeypatch.setattr('kumo_core.prediction_options.prediction_timestamps',
                        resolve_timestamp)
    monkeypatch.setattr(prediction_table, 'build_native_pquery_task_plan',
                        lambda **kwargs: plan)

    request, graph_id = prediction_table._build_prediction_table_request(
        snapshot,
        parsed,
        prediction_time=anchor,
        run_id='explicit-anchor',
        scratch_dir=Path(tmp_path),
        artifact_dir=Path(tmp_path),
    )

    assert graph_id == 'snapshot'
    assert request.timestamps == (123, )
    resolve_timestamp.assert_called_once_with(anchor)


def test_compile_marks_caller_rows_as_supplied(monkeypatch, tmp_path):
    parsed = SimpleNamespace(query_type='temporal', task_type='forecasting')
    snapshot = SimpleNamespace(graph_config=object(),
                               index_path=tmp_path / 'index', id='snapshot')
    plan = SimpleNamespace(
        graph_index_id='snapshot',
        task_dataset=object(),
        train_table=SimpleNamespace(compiler_options=None, timestamps=()),
    )
    build_args = {}

    def build(**kwargs):
        build_args.update(kwargs)
        return plan

    monkeypatch.setattr(prediction_table, 'normalize_graph',
                        lambda config: 'graph')
    monkeypatch.setattr(prediction_table, 'latest_temporal_timestamp',
                        lambda *args: pytest.fail('resolved supplied rows'))
    monkeypatch.setattr(prediction_table, 'build_native_pquery_task_plan', build)

    request, _ = prediction_table._build_prediction_table_request(
        snapshot,
        parsed,
        prediction_time=None,
        run_id='supplied-rows',
        scratch_dir=Path(tmp_path),
        artifact_dir=Path(tmp_path),
        supplied_rows=True,
    )

    assert build_args['supplied_rows'] is True
    assert build_args['timestamps'] == ()
    assert request.timestamps == ()


def test_compile_keeps_omitted_static_timestamps_empty(monkeypatch, tmp_path):
    parsed = SimpleNamespace(query_type='static', task_type='node_prediction')
    snapshot = SimpleNamespace(graph_config=object(),
                               index_path=tmp_path / 'index', id='snapshot')
    plan = SimpleNamespace(
        graph_index_id='snapshot',
        task_dataset=object(),
        train_table=SimpleNamespace(compiler_options=None, timestamps=()),
    )
    build_args = {}

    def build(**kwargs):
        build_args.update(kwargs)
        return plan

    monkeypatch.setattr(prediction_table, 'normalize_graph',
                        lambda config: 'graph')
    monkeypatch.setattr(prediction_table, 'latest_temporal_timestamp',
                        lambda *args: pytest.fail('resolved a static query'))
    monkeypatch.setattr(prediction_table, 'build_native_pquery_task_plan', build)

    request, _ = prediction_table._build_prediction_table_request(
        snapshot,
        parsed,
        prediction_time=None,
        run_id='static',
        scratch_dir=Path(tmp_path),
        artifact_dir=Path(tmp_path),
    )

    assert build_args['timestamps'] == ()
    assert request.timestamps == ()


def test_compile_uses_effective_forecast_generation_plan(monkeypatch, tmp_path):
    parsed = SimpleNamespace(
        query_type='temporal',
        task_type='forecasting',
        query=SimpleNamespace(get_combined_date_offset_range=lambda:
                              SimpleNamespace(start=0, end=1)),
    )
    snapshot = SimpleNamespace(graph_config=object(),
                               index_path=tmp_path / 'index', id='snapshot')
    compiler_options = {
        'lag_offsets_secs': (-172800, -345600, -518400),
        'yoy_offsets_secs': (-31536000, ),
        'future_offsets_secs': (0, 172800),
    }
    captured = {}

    def build(**kwargs):
        generation = kwargs['model_plan'].training_table_generation
        captured['generation'] = generation
        return SimpleNamespace(
            graph_index_id='snapshot',
            task_dataset=object(),
            train_table=SimpleNamespace(
                compiler_options=compiler_options,
                timestamps=(123, ),
            ),
        )

    monkeypatch.setattr(prediction_table, 'normalize_graph',
                        lambda config: 'graph')
    monkeypatch.setattr(prediction_table, 'latest_temporal_timestamp',
                        lambda query, index_path: 123)
    monkeypatch.setattr('kumo_core.prediction_options.prediction_timestamps',
                        lambda value: (123, ))
    monkeypatch.setattr(prediction_table, 'build_native_pquery_task_plan',
                        build)
    monkeypatch.setattr(prediction_table, 'generate_prediction_table',
                        lambda request: SimpleNamespace())

    request, _, _ = prediction_table._compile_prediction_table(
        snapshot,
        parsed,
        prediction_time=None,
        run_id='forecast',
        scratch_dir=Path(tmp_path),
        artifact_dir=Path(tmp_path),
        generation_plan=TrainingTableGenerationPlan(
            lag_timesteps=3,
            year_over_year=True,
            timeframe_step=2,
        ),
    )

    generation = captured['generation']
    assert generation.lag_timesteps == 3
    assert generation.year_over_year is True
    assert generation.timeframe_step == 2
    assert request.compiler_options == compiler_options
    assert request.forecast_timeframe_step == 2
