# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise shared replay and read-only source data with tiny real CPU models."""

import importlib.util
import os
import shutil
import subprocess
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pytorch_lightning import Callback
from pytorch_lightning import Trainer as LightningTrainer

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_check.py'
SPEC = importlib.util.spec_from_file_location('weekly_shared_helpers', SOURCE)
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)
case_runner = shared.sibling('nightly_case')


@contextmanager
def readonly_mount(root):
    result = subprocess.run(
        ['mount', '--bind', str(root), str(root)], capture_output=True,
        text=True)
    if result.returncode:
        pytest.skip('Read-only bind mount unavailable: ' +
                    result.stderr.strip())
    try:
        subprocess.run(['mount', '-o', 'remount,bind,ro',
                        str(root)], check=True, capture_output=True)
        assert os.statvfs(root).f_flag & os.ST_RDONLY
        yield
    finally:
        subprocess.run(['umount', str(root)], check=True, capture_output=True)


@pytest.mark.parametrize('read_only', [False, True])
def test_shared_replay_loads_real_checkpoint_without_fitting(
        fitted, tmp_path, monkeypatch, read_only):
    trainer, prediction, snapshot = fitted
    archive = tmp_path / 'archive'
    archive.mkdir()
    fitted_result = trainer._training_result
    shutil.copytree(fitted_result.bundle_dir, archive / 'model')
    checkpoint = shared.digest(
        fitted_result.artifact_paths['selected_checkpoint'])
    population = pd.read_parquet(
        prediction.table_path).rename(columns={'ENTITY': 'ENTITY1'})

    def predict(model):
        frame = model.predict(prediction, snapshot=snapshot,
                              binary_classification_threshold=.5).frame
        frame = frame.set_index('ENTITY').loc[population.ENTITY1]
        return frame[['False_PROB', 'True_PROB']].to_numpy(), \
            frame.TARGET_PRED.to_numpy()

    before, decisions = predict(trainer)
    population.to_parquet(archive / 'population.parquet', index=False)
    case_runner.write_prediction(archive / 'predictions.parquet', population,
                                 before, decisions, 'binary')
    replay = dict(
        model=dict(
            path='model', files={
                path.relative_to(archive / 'model').as_posix():
                shared.digest(path)
                for path in (archive / 'model').rglob('*') if path.is_file()
            }), checkpoint_sha256=checkpoint)
    for name in ('population', 'predictions'):
        replay[name] = dict(path=name + '.parquet',
                            sha256=shared.digest(archive / (name + '.parquet')))
    state = dict(prediction_batches={})

    def reject_fit(*args, **kwargs):
        pytest.fail('Reference replay attempted a fit')

    monkeypatch.setattr(kumo.Trainer, 'fit', reject_fit)
    initialize = LightningTrainer.__init__

    class RestoredCheckpoint(Callback):
        def on_predict_start(self, trainer, module):
            state['replay_checks'] += 1
            state['replay_restored_tensors'] += shared.verify_checkpoint(
                module, trainer.ckpt_path, checkpoint)

    def initialize_with_check(self, *args, callbacks=None, **kwargs):
        initialize(self, *args,
                   callbacks=[*(callbacks or []),
                              RestoredCheckpoint()], **kwargs)

    monkeypatch.setattr(LightningTrainer, '__init__', initialize_with_check)
    output = tmp_path / 'output'
    output.mkdir()

    def replay_predict(model, stage, frame):
        assert stage == 'replay'
        pd.testing.assert_frame_equal(frame, population)
        return predict(model)

    with readonly_mount(archive) if read_only else nullcontext():
        result = case_runner.replay_reference(
            replay, archive, replay_predict, state, dict(atol=.005, rtol=0),
            task='binary', classes=[], helpers=SimpleNamespace(**vars(shared)),
            output=output, population=population)
    assert result['status'] == 'PASS'
    assert state['replay_checks'] > 0
    assert state['replay_restored_tensors'] > 0


@pytest.fixture
def isolated_scratch(tmp_path):
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        yield
    finally:
        kumo.configure(scratch_dir=previous)


@pytest.mark.parametrize('ranking', [False, True])
def test_real_fit_and_predict_keep_mounted_sources_read_only(
        tmp_path, ranking, isolated_scratch):
    from kumoapi.graph import Edge
    from kumoapi.model_plan import ModelPlan
    from kumoapi.typing import Stype

    data = tmp_path / 'data'
    data.mkdir()
    rng = np.random.default_rng(0)
    pd.DataFrame(dict(ID=range(50))).to_parquet(data / 'ents.parquet')
    pd.DataFrame(dict(ID=range(20))).to_parquet(data / 'items.parquet')
    pd.DataFrame(
        dict(
            EID=range(3000), ENT_ID=rng.integers(0, 50, 3000),
            ITEM_ID=rng.integers(0, 20, 3000), TS=pd.Timestamp('2024-01-01') +
            pd.to_timedelta(rng.integers(0, 120, 3000), unit='D'))).to_parquet(
                data / 'events.parquet')
    before = {path.name: shared.digest(path) for path in data.iterdir()}
    with readonly_mount(data):
        files = kumo.FileConnector(root_dir=str(data))
        events = kumo.Table.from_source_table(
            files['events.parquet'], primary_key='EID',
            time_column='TS').infer_metadata().with_column(
                'ENT_ID', stype=Stype.ID).with_column('ITEM_ID', stype=Stype.ID)
        graph = kumo.Graph(
            tables=dict(
                ents=kumo.Table.from_source_table(
                    files['ents.parquet'], primary_key='ID').infer_metadata(),
                items=kumo.Table.from_source_table(
                    files['items.parquet'], primary_key='ID').infer_metadata(),
                events=events), edges=[
                    Edge(src_table='events', fkey='ENT_ID', dst_table='ents'),
                    Edge(src_table='events', fkey='ITEM_ID', dst_table='items')
                ])
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(
            graph,
            ('PREDICT LIST_DISTINCT(events.ITEM_ID, 0, 7, days) RANK TOP 3 '
             if ranking else 'PREDICT COUNT(events.*, 0, 7, days) > 0 ') +
            'FOR EACH ents.ID')
        plan = ModelPlan()
        plan.training_job.disable_explain = True
        plan.training_job.num_experiments = 1
        plan.optimization.max_epochs = 1
        plan.optimization.max_steps_per_epoch = 2
        plan.optimization.max_val_steps = 2
        trainer = kumo.Trainer(model_plan=plan)
        trainer.fit(query.generate_training_table(snapshot), snapshot=snapshot)
        result = trainer.predict(query.generate_prediction_table(snapshot),
                                 snapshot=snapshot)
        assert result.num_rows > 0
        assert {
            path.name: shared.digest(path)
            for path in data.iterdir()
        } == before
