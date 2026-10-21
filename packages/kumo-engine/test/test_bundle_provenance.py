# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import shutil
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from kumo_core.artifacts import add_bundle_file
from kumoapi.graph import Edge
from kumoapi.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.trainer._spec import MANIFEST_FILE
from kumo.engine.trainer.trainer import TrainingResult, _published_fit_mode


@pytest.fixture(scope='module')
def two_trials(tmp_path_factory):
    root = tmp_path_factory.mktemp('provenance')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    try:
        rng = np.random.default_rng(0)
        data = root / 'data'
        data.mkdir()
        pd.DataFrame({'ID': np.arange(50)}).to_parquet(data / 'ents.parquet')
        pd.DataFrame({
            'EID':
            np.arange(3000),
            'ENT_ID':
            rng.integers(0, 50, 3000),
            'TS':
            pd.Timestamp('2024-01-01') +
            pd.to_timedelta(rng.integers(0, 120, 3000), unit='D'),
        }).to_parquet(data / 'events.parquet')
        connector = kumo.FileConnector(root_dir=str(data))
        events = kumo.Table.from_source_table(
            connector['events.parquet'], primary_key='EID',
            time_column='TS').infer_metadata().with_column(
                'ENT_ID', stype=Stype.ID)
        graph = kumo.Graph(
            tables={
                'ents':
                kumo.Table.from_source_table(connector['ents.parquet'],
                                             primary_key='ID').infer_metadata(),
                'events':
                events,
            },
            edges=[Edge(src_table='events', fkey='ENT_ID', dst_table='ents')])
        snapshot = graph.snapshot()
        pquery = kumo.PredictiveQuery(
            graph=graph, query='PREDICT COUNT(events.*, 0, 7, days) > 0 '
            'FOR EACH ents.ID')
        plan = ModelPlan()
        plan.training_job.disable_explain = True
        plan.training_job.num_experiments = 2
        plan.optimization.max_epochs = 1
        plan.optimization.max_steps_per_epoch = 2
        table = pquery.generate_training_table(snapshot)
        trainer = kumo.Trainer(model_plan=plan)
        result = trainer.fit(table, snapshot=snapshot)
        yield (trainer, result, table, snapshot,
               pquery.generate_prediction_table(snapshot))
    finally:
        kumo.configure(scratch_dir=previous)


def _predict(trainer, table, snapshot):
    result = trainer.predict(table, snapshot=snapshot,
                             binary_classification_threshold=.5)
    return result.frame.sort_values('ENTITY').reset_index(drop=True)


def _roles(bundle):
    manifest = json.loads((bundle / MANIFEST_FILE).read_text())
    return {entry['role']: entry for entry in manifest['artifacts']}


def _strip_provenance(bundle):
    manifest = json.loads((bundle / MANIFEST_FILE).read_text())
    manifest['artifacts'] = [
        entry for entry in manifest['artifacts']
        if entry['role'] != 'provenance'
    ]
    (bundle / MANIFEST_FILE).write_text(json.dumps(manifest))
    (bundle / 'provenance.json').unlink()


def test_fit_records_listed_provenance(two_trials):
    trainer, result, table, snapshot, _ = two_trials
    assert _roles(result.bundle_dir)['provenance']['relative_path'] == (
        'provenance.json')
    provenance = result.provenance
    assert provenance['kind'] == 'model_bundle'
    assert provenance['parents'] == {
        'training_table': table.tt_id,
        'snapshot': snapshot.id,
        'warm_start': None,
    }
    assert provenance['training_table'] == {
        'source': 'pql',
        'snapshot': snapshot.id
    }
    assert provenance['source_tables'] == (
        snapshot.manifest['parents']['source_tables'])
    assert provenance['fit']['seconds'] >= 0
    assert provenance['fit_mode'] == 'refit_trainval'
    assert provenance['environment']['kumo_engine'] == kumo.__version__
    assert trainer.bundle_id == result.bundle_id


def test_bundle_id_is_content_identity(two_trials, tmp_path):
    trainer, result, *_ = two_trials
    assert re.fullmatch('[0-9a-f]{64}', result.bundle_id)
    loaded = kumo.Trainer.load(result.bundle_dir)
    assert loaded.bundle_id == result.bundle_id
    assert loaded.provenance == result.provenance
    copy = tmp_path / 'copy'
    shutil.copytree(result.bundle_dir, copy)
    assert kumo.Trainer.load(copy).bundle_id == result.bundle_id

    provenance = json.loads((copy / 'provenance.json').read_text())
    provenance['fit']['seconds'] = -1
    (copy / 'provenance.json').write_text(json.dumps(provenance))
    add_bundle_file(copy, 'provenance', 'provenance.json')
    reloaded = kumo.Trainer.load(copy)
    assert reloaded.bundle_id == result.bundle_id
    assert reloaded.provenance['fit']['seconds'] == -1

    config = copy / 'model' / 'config.json'
    config.write_text(config.read_text() + ' ')
    add_bundle_file(copy, 'trainer_config', 'model/config.json')
    assert kumo.Trainer.load(copy).bundle_id != result.bundle_id


def test_bundle_without_provenance_loads_and_predicts_identically(
        two_trials, tmp_path):
    trainer, result, _, snapshot, prediction_table = two_trials
    old = tmp_path / 'old'
    shutil.copytree(result.bundle_dir, old)
    _strip_provenance(old)

    restored = kumo.Trainer.load(old)
    assert restored.bundle_id == result.bundle_id
    assert restored.provenance is None
    pd.testing.assert_frame_equal(_predict(restored, prediction_table,
                                           snapshot),
                                  _predict(trainer, prediction_table, snapshot))

    (old / 'provenance.json').write_text(json.dumps(result.provenance))
    stray = kumo.Trainer.load(old)
    assert stray.provenance is None
    assert stray.bundle_id == result.bundle_id


def _finish_stub_fit(tmp_path, trainer, manifest=None):
    bundle = tmp_path / 'bundle'
    (bundle / 'model').mkdir(parents=True)
    (bundle / 'model' / 'config.json').write_text('{}')
    (bundle / MANIFEST_FILE).write_text(json.dumps({'artifacts': []}))
    add_bundle_file(bundle, 'trainer_config', 'model/config.json')
    table = type('Table', (), {'tt_id': 'tt-1', 'manifest': manifest})()
    return trainer._finish_fit(
        TrainingResult({}, {}, bundle), predictive_query='PREDICT x',
        snapshot_id='snap', export_config=None,
        training_table_generation=TrainingTableGenerationPlan(),
        provenance=dict(train_table=table, snapshot_id='snap',
                        warm_start_bundle_id=None,
                        started_at=datetime.now(timezone.utc),
                        fit_mode=None))


def test_provenance_without_a_table_manifest(tmp_path):
    result = _finish_stub_fit(tmp_path, kumo.Trainer(ModelPlan()))
    assert result.provenance['parents']['training_table'] == 'tt-1'
    assert result.provenance['training_table'] == {
        'source': None,
        'snapshot': None
    }
    assert result.provenance['source_tables'] is None


def test_provenance_records_custom_table_source(tmp_path):
    source_table = {
        'connector': 'S3',
        'location': 's3://bucket/exports/tt',
        'version': '17'
    }
    manifest = {
        'source': 'custom',
        'query': 'PREDICT x',
        'parents': {
            'snapshot': 'snap',
            'source_table': source_table
        },
    }
    result = _finish_stub_fit(tmp_path, kumo.Trainer(ModelPlan()), manifest)
    assert result.provenance['training_table'] == {
        'source': 'custom',
        'snapshot': 'snap',
        **source_table,
    }


@pytest.mark.parametrize('flags,mode', [
    ({}, None),
    ({'fitted': True}, 'fit'),
    ({'fitted': True, 'refitted_trainval': True}, 'refit_trainval'),
    ({'fitted': True, 'refitted_full': True}, 'refit_full'),
    ({'fitted': True, 'refitted_trainval': True, 'refitted_full': True},
     'refit_full'),
])
def test_published_fit_mode_follows_refit_flags(flags, mode):
    trainer = SimpleNamespace(refit_cfg=SimpleNamespace(**flags))
    assert _published_fit_mode(trainer) == mode


def test_provenance_failure_warns_and_still_publishes(tmp_path, monkeypatch):
    from kumo.engine.trainer import _provenance

    def fail(*args):
        raise RuntimeError('unreadable manifest')

    monkeypatch.setattr(_provenance, '_training_table', fail)
    trainer = kumo.Trainer(ModelPlan())
    with pytest.warns(UserWarning, match='unreadable manifest'):
        result = _finish_stub_fit(tmp_path, trainer)
    assert 'provenance' not in _roles(result.bundle_dir)
    assert result.provenance is None
    assert re.fullmatch('[0-9a-f]{64}', result.bundle_id)
    assert trainer._training_result is result
