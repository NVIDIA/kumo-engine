# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise production error swallowing with actual producer feature batches."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
from kumo_core.encoders.tensor_frame import get_tf
from kumo_core.encoders.tensor_mapper import (
    MultiCategoricalTensorMapper,
    NumericalTensorMapper,
    TimestampTensorMapper,
)
from kumo_core.explain.column_analysis import ColumnAnalysis
from kumo_core.explain.subgraph_collector import SubgraphCollector
from kumo_core.io import filesystem
from kumo_core.io.json import load_json
from kumo_core.types import TaskType
from kumoapi.explain import ColumnAnalysisOutput
from kumoapi.train import TrainingStage
from torch_geometric.data import HeteroData


@pytest.fixture(autouse=True)
def production_errors(monkeypatch):
    monkeypatch.setenv('KUMO_EXPLAINER_CONTINUE_ON_ERROR', 'true')
    monkeypatch.delenv('KUMO_EXPLAINER_RAISE_ON_ERROR', raising=False)


def _batch(entity, *, user_value=1.0):
    mapper = NumericalTensorMapper()
    batch = HeteroData()
    for table in ['USER', 'FACT']:
        batch[table].tf = get_tf(pa.RecordBatch.from_arrays(
            [pa.array([user_value if table == 'USER' else 1.0])], names=['VALUE']), {'VALUE': mapper})
        batch[table].batch_size = 1
        batch[table].batch = torch.tensor([0])
        batch[table].num_sampled_nodes = [1]
        batch[table].pkey = np.array([entity])
    batch['USER'].input_id = torch.tensor([entity])
    store = SimpleNamespace(
        train_table_name='TRAIN', table_names=['USER', 'FACT'],
        get_tensor_mapper=Mock(return_value=mapper),
        get_pkey_name=lambda table: 'ID',
        get_time_col_name=lambda table: None)
    schema = SimpleNamespace(get_foreign_keys=lambda table: [])
    return batch, SimpleNamespace(data=(store, schema))


def _module():
    return SimpleNamespace(
        encoder=SimpleNamespace(exclude_cols_dict={}),
        target_encoder=SimpleNamespace(decode_tensor=lambda values: values))


@pytest.mark.parametrize('failed', [{0}, {1}, {2}, {0, 1, 2}])
@pytest.mark.parametrize('partial', [False, True])
def test_failed_cohort_batches_never_publish(tmp_path, failed, partial):
    analyzer = ColumnAnalysis(TaskType.REGRESSION, str(tmp_path), normalize=False)
    module = _module()
    analyzer.explain_start(module, None, TrainingStage.PRED)
    for idx in range(3):
        batch, loader = _batch(idx)
        if idx in failed:
            mapper = loader.data[0].get_tensor_mapper.return_value

            def feature_mapper(table, column):
                if not partial or table == 'FACT':
                    raise RuntimeError('feature batch failed')
                return mapper

            if partial:
                collect = analyzer.collect_numerical
                collected = 0

                def collect_feature(*args, **kwargs):
                    nonlocal collected
                    collected += 1
                    if collected == 2:
                        raise RuntimeError('partially collected batch failed')
                    return collect(*args, **kwargs)

                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(analyzer, 'collect_numerical', collect_feature)
                    analyzer.explain_post_step(batch, module, loader, {},
                                               TrainingStage.PRED, idx)
            else:
                loader.data[0].get_tensor_mapper.side_effect = feature_mapper
                analyzer.explain_post_step(batch, module, loader, {},
                                           TrainingStage.PRED, idx)
        else:
            analyzer.explain_post_step(batch, module, loader, {},
                                       TrainingStage.PRED, idx)
    analyzer.explain_end(module, None, {'pred': torch.tensor([10., 20., 30.])},
                         TrainingStage.PRED)
    assert analyzer.failure is not None
    assert analyzer.processed_rows == 3
    assert not analyzer._dfs_dict
    assert not analyzer._cols_dict
    assert not analyzer.completed
    assert not filesystem.exists(str(tmp_path / 'pred/column_analysis.json'))


def test_healthy_cohort_values(tmp_path):
    analyzer = ColumnAnalysis(TaskType.REGRESSION, str(tmp_path), normalize=False)
    module = _module()
    analyzer.explain_start(module, None, TrainingStage.PRED)
    for idx in range(3):
        batch, loader = _batch(idx)
        analyzer.explain_post_step(batch, module, loader, {}, TrainingStage.PRED,
                                   idx)
    analyzer.explain_end(module, None, {'pred': torch.tensor([10., 20., 30.])},
                         TrainingStage.PRED)
    result = load_json(str(tmp_path / 'pred/column_analysis.json'),
                       ColumnAnalysisOutput)
    assert result is not None
    assert result['USER', 0, 'VALUE'].predictions == [20.0]
    assert analyzer.failure is None
    assert analyzer.processed_rows == 3
    assert analyzer.completed


@pytest.mark.parametrize('null_rows', [{0, 1, 2}, {1}])
def test_null_numeric_values_preserve_other_cohorts(tmp_path, null_rows):
    analyzer = ColumnAnalysis(TaskType.REGRESSION, str(tmp_path), normalize=False)
    module = _module()
    analyzer.explain_start(module, None, TrainingStage.PRED)
    for idx in range(3):
        value = float('nan') if idx in null_rows else 1.0
        batch, loader = _batch(idx, user_value=value)
        analyzer.explain_post_step(batch, module, loader, {}, TrainingStage.PRED,
                                   idx)
    analyzer.explain_end(module, None, {'pred': torch.tensor([10., 20., 30.])},
                         TrainingStage.PRED)
    result = load_json(str(tmp_path / 'pred/column_analysis.json'),
                       ColumnAnalysisOutput)
    assert analyzer.completed
    assert analyzer.failure is None
    assert result is not None
    assert result['FACT', 0, 'VALUE'].predictions == [20.0]
    if len(null_rows) == 3:
        with pytest.raises(KeyError):
            result['USER', 0, 'VALUE']
    else:
        assert result['USER', 0, 'VALUE'].predictions == [20.0]
    assert all(cohort.cohorts for cohort in result.cohorts)


@pytest.mark.parametrize('values', ['empty', 'missing', 'future', 'populated'])
@pytest.mark.parametrize('stage', [TrainingStage.PRED, TrainingStage.VAL])
def test_empty_categories_preserve_cohorts(tmp_path, values, stage):
    analyzer = ColumnAnalysis(TaskType.REGRESSION, str(tmp_path), normalize=False)
    module = _module()
    analyzer.explain_start(module, None, stage)
    numerical = NumericalTensorMapper()
    mapper = (TimestampTensorMapper() if values == 'future' else
              MultiCategoricalTensorMapper([['known', 1], ['unused', 1]],
                                           sep=None))
    for idx in range(3):
        batch, loader = _batch(idx)
        # Only USER nodes are prediction seeds and supply their seed time.
        del batch['FACT'].batch_size
        if values == 'future':
            feature = pa.array(['2025-01-01'])
            batch['USER'].seed_time = torch.tensor([
                pd.Timestamp('2024-01-01').value,
            ])
        else:
            value = (None if values == 'missing' else
                     ['known'] if values == 'populated' and idx == 1 else [])
            feature = pa.array([value], type=pa.list_(pa.string()))
        batch['USER'].tf = get_tf(
            pa.RecordBatch.from_arrays([pa.array([1.0]), feature],
                                       names=['VALUE', 'CATEGORY']),
            {'VALUE': numerical, 'CATEGORY': mapper})
        loader.data[0].get_tensor_mapper.side_effect = (
            lambda table, column: mapper if column == 'CATEGORY' else numerical)
        analyzer.explain_post_step(batch, module, loader, {}, stage, idx)

    outputs = {'pred': torch.tensor([10., 20., 30.])}
    if stage != TrainingStage.PRED:
        outputs['target'] = torch.tensor([1., 2., 3.])
    analyzer.explain_end(module, None, outputs, stage)
    result = load_json(str(tmp_path / stage / 'column_analysis.json'),
                       ColumnAnalysisOutput)
    assert analyzer.completed
    assert analyzer.failure is None
    assert result['USER', 0, 'VALUE'].predictions == [20.0]
    if values == 'populated':
        cohort = result['USER', 0, 'CATEGORY']
        assert cohort.cohorts == ['known']
        assert cohort.populations == [1.0]
        assert cohort.predictions == [20.0]
        assert cohort.targets == ([2.0] if stage != TrainingStage.PRED else None)
    else:
        with pytest.raises(KeyError):
            result['USER', 0, 'CATEGORY']
    assert all(cohort.cohorts for cohort in result.cohorts)
    for cohort in result.cohorts:
        assert sum(cohort.populations) == pytest.approx(1.0)
        assert np.isfinite(cohort.predictions).all()


def test_empty_population_is_not_a_failure(tmp_path):
    analyzer = ColumnAnalysis(TaskType.REGRESSION, str(tmp_path))
    module = _module()
    analyzer.explain_start(module, None, TrainingStage.PRED)
    analyzer.explain_end(module, None, {'pred': torch.empty(0)}, TrainingStage.PRED)
    result = load_json(str(tmp_path / 'pred/column_analysis.json'),
                       ColumnAnalysisOutput)
    assert result.cohorts == []
    assert analyzer.completed
    assert analyzer.failure is None


def test_aggregate_error_does_not_complete(tmp_path, monkeypatch):
    analyzer = ColumnAnalysis(TaskType.REGRESSION, str(tmp_path))
    module = _module()
    analyzer.explain_start(module, None, TrainingStage.PRED)
    batch, loader = _batch(0)
    analyzer.explain_post_step(batch, module, loader, {}, TrainingStage.PRED, 0)
    monkeypatch.setattr(analyzer, 'aggregate_range',
                        Mock(side_effect=RuntimeError('aggregate failed')))
    analyzer.explain_end(module, None, {'pred': torch.tensor([10.])},
                         TrainingStage.PRED)
    assert 'aggregate failed' in analyzer.failure
    assert not analyzer._dfs_dict
    assert not analyzer._cols_dict
    assert not analyzer.completed
    assert not filesystem.exists(str(tmp_path / 'pred/column_analysis.json'))


@pytest.mark.parametrize('failed', [{0}, {1}, {2}, {0, 1, 2}])
@pytest.mark.parametrize('feature_error', [False, True])
def test_collector_failures_leave_identity_gaps(tmp_path, failed, feature_error,
                                                monkeypatch):
    collector = SubgraphCollector(TaskType.REGRESSION, str(tmp_path), ['csv'])
    collector.explain_start(None, None, TrainingStage.PRED)
    write = collector.format_csv
    for idx in range(3):
        batch, loader = _batch(100 + idx)
        if idx in failed:
            if feature_error:
                loader.data[0].get_tensor_mapper.side_effect = RuntimeError(
                    'feature extraction failed')
                writer = write
            else:
                writer = Mock(side_effect=RuntimeError('serialization failed'))
        else:
            writer = write
        monkeypatch.setattr(collector, 'format_csv', writer)
        collector.explain_post_step(batch, _module(), loader, {},
                                    TrainingStage.PRED, idx)
    collector.explain_end(None, None, {}, TrainingStage.PRED)
    assert collector._idx == 3
    assert collector.failure is not None
    assert not collector.completed
    for idx in range(3):
        path = str(tmp_path / 'pred/csv' / str(idx) / 'USER.csv')
        assert filesystem.exists(path) == (idx not in failed)
        if idx not in failed:
            with filesystem.open_file(path, 'rt') as source:
                assert pd.read_csv(source)['ID'].tolist() == [100 + idx]
