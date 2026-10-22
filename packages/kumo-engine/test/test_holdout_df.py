# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pandas as pd
import pytest
from kumo_core.train.output import EvalOutput
from kumoapi.graph import Edge
from kumoapi.model_plan import ModelPlan
from kumoapi.train import TrainingStage
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.trainer.trainer import _find_holdout


@pytest.fixture(scope='module')
def graph(tmp_path_factory):
    root = tmp_path_factory.mktemp('graph')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    rng = np.random.default_rng(0)
    data = root / 'data'
    data.mkdir()
    pd.DataFrame({'ID': np.arange(50)}).to_parquet(data / 'ents.parquet')
    pd.DataFrame({'IID': np.arange(20)}).to_parquet(data / 'items.parquet')
    pd.DataFrame({
        'EID':
        np.arange(3000),
        'ENT_ID':
        rng.integers(0, 50, 3000),
        'ITEM_ID':
        rng.integers(0, 20, 3000),
        'TS':
        pd.Timestamp('2024-01-01') +
        pd.to_timedelta(rng.integers(0, 120, 3000), unit='D'),
        'V':
        rng.normal(size=3000),
    }).to_parquet(data / 'events.parquet')
    connector = kumo.FileConnector(root_dir=str(data))
    ents = kumo.Table.from_source_table(connector['ents.parquet'],
                                        primary_key='ID').infer_metadata()
    items = kumo.Table.from_source_table(connector['items.parquet'],
                                         primary_key='IID').infer_metadata()
    events = kumo.Table.from_source_table(
        connector['events.parquet'], primary_key='EID',
        time_column='TS').infer_metadata().with_column(
            'ENT_ID', stype=Stype.ID).with_column('ITEM_ID', stype=Stype.ID)
    graph = kumo.Graph(
        tables={
            'ents': ents,
            'items': items,
            'events': events
        },
        edges=[
            Edge(src_table='events', fkey='ENT_ID', dst_table='ents'),
            Edge(src_table='events', fkey='ITEM_ID', dst_table='items'),
        ])
    yield graph, graph.snapshot()
    kumo.configure(scratch_dir=previous)


QUERY = 'PREDICT COUNT(events.*, 0, 7, days) > 0 FOR EACH ents.ID'


def _fit(graph, query, refit_trainval=True, refit_full=False, explain=False):
    graph, snapshot = graph
    pquery = kumo.PredictiveQuery(graph=graph, query=query)
    plan = ModelPlan()
    plan.training_job.disable_explain = not explain
    plan.training_job.refit_trainval = refit_trainval
    plan.training_job.refit_full = refit_full
    plan.optimization.max_epochs = 1
    # refit_full splits each epoch's batches across 3 stages.
    plan.optimization.max_steps_per_epoch = 3 if refit_full else 2
    trainer = kumo.Trainer(model_plan=plan)
    return trainer.fit(pquery.generate_training_table(snapshot),
                       snapshot=snapshot)


def test_holdout_df_after_fit(graph):
    result = _fit(graph, 'PREDICT COUNT(events.*, 0, 7, days) > 0 '
                  'FOR EACH ents.ID')

    frame = result.holdout_df()

    assert result.holdout_path.name == 'test_df.parquet'
    assert not result.holdout_truncated
    assert len(frame) > 0
    assert {'ENTITY', 'TARGET_PRED'} <= set(frame.columns)
    assert not (result.bundle_dir / 'test_df.parquet').exists()
    holdout = result.provenance.holdout
    assert holdout.rows == len(frame)
    assert holdout.label_positive_rate == frame['TARGET'].mean()
    assert holdout.truncated is False


@pytest.mark.parametrize('refit_trainval,refit_full,explain,model', [
    (False, False, False, 'fit'),
    (True, False, True, 'refit_trainval'),
    (True, True, True, None),
])
def test_holdout_is_the_published_models(graph, refit_trainval, refit_full,
                                         explain, model):
    result = _fit(graph, QUERY, refit_trainval, refit_full, explain)

    assert result.provenance.fit_mode == (model or 'refit_full')
    if model is None:
        assert result.holdout_path is None
        assert result.provenance.holdout is None
    else:
        suffix = '' if model == 'fit' else '_refit_trainval'
        assert re.fullmatch(rf'version_\d+{suffix}',
                            result.holdout_path.parent.name)
        assert result.provenance.holdout.rows == len(result.holdout_df())


def test_no_holdout(graph):
    result = _fit(
        graph, 'PREDICT LIST_DISTINCT(events.ITEM_ID, 0, 7, days) '
        'RANK TOP 3 FOR EACH ents.ID')

    assert result.holdout_path is None
    assert not result.holdout_truncated
    assert result.provenance.holdout is None
    with pytest.raises(FileNotFoundError, match='No holdout'):
        result.holdout_df()


def test_loaded_trainer_has_no_holdout(graph):
    graph, snapshot = graph
    result = _fit((graph, snapshot), 'PREDICT COUNT(events.*, 0, 7, days) > 0 '
                  'FOR EACH ents.ID')

    loaded = kumo.Trainer.load(result.bundle_dir)

    assert loaded._training_result.holdout_path is None
    assert (loaded._training_result.provenance.holdout ==
            result.provenance.holdout)


def test_truncation_is_recorded(tmp_path):
    frame = pd.DataFrame({'ENTITY': np.arange(1000), 'x': np.zeros(1000)})
    output = EvalOutput(metrics={}, df=frame)

    output.save_df(str(tmp_path), TrainingStage.TEST,
                   max_size=int(frame.memory_usage(deep=True).sum() / 4))
    path, truncated = _find_holdout(str(tmp_path))

    assert truncated
    assert 0 < len(pd.read_parquet(path)) < len(frame)

    output.save_df(str(tmp_path), TrainingStage.TEST)
    _, truncated = _find_holdout(str(tmp_path))

    assert not truncated


def test_fit_reports_truncation(graph, monkeypatch, caplog):
    monkeypatch.setattr('kumo_core.train.output._HOLDOUT_MAX_BYTES', 1000)

    result = _fit(graph, QUERY)

    assert result.holdout_truncated is True
    assert result.provenance.holdout.truncated is True
    with caplog.at_level('WARNING'):
        assert len(result.holdout_df()) > 0
    assert 'truncated' in caplog.text


@pytest.mark.parametrize('content', ['{not json', '{}', None])
def test_unreadable_size_record_does_not_fail_fit(tmp_path, caplog, content):
    EvalOutput(metrics={}, df=pd.DataFrame({'ENTITY': [1, 2]})).save_df(
        str(tmp_path), TrainingStage.TEST)
    meta = tmp_path / 'test_df_meta.json'
    if content is None:
        meta.unlink()
    else:
        meta.write_text(content)

    path, truncated = _find_holdout(str(tmp_path))

    assert truncated is None
    assert len(pd.read_parquet(path)) == 2
    assert 'truncation is unknown' in caplog.text
