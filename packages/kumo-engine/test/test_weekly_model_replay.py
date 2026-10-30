# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the weekly replay with real public model loading on CPU."""

import importlib.util
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

import kumo.engine as kumo

SCRIPT = Path(__file__).parents[1] / 'scripts/weekly_worker.py'
SPEC = importlib.util.spec_from_file_location('weekly_worker', SCRIPT)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)
weekly = worker.weekly


def test_saved_model_replay_uses_frozen_native_inputs_without_a_fit(
        fitted, tmp_path, monkeypatch):
    trainer, prediction, snapshot = fitted
    [runtime_path] = trainer.scratch_dir.glob(
        'runs/*-fit-*/task_dataset/runtime_manifest.json')
    native_specs = [
        weekly.read(path)
        for path in trainer.scratch_dir.parents[1].rglob('tt_ingest_spec.json')
    ]
    [native_spec] = [
        value for value in native_specs
        if {v['key']
            for v in (value['split_stages'] or [])} == {'train', 'val', 'test'}
    ]
    counts = {v['key']: v['row_count'] for v in native_spec['split_stages']}
    contract = worker.check_native_contract(
        snapshot.graph_config, weekly.read(runtime_path),
        weekly.read(runtime_path.with_name('feature_schema.json')),
        weekly.read(trainer._training_result.artifact_paths['trainer_config']),
        counts, 'binary_classification')
    assert contract['status'] == 'PASS'
    inputs = tmp_path / 'inputs'
    index = inputs / 'index'
    for directory, dirs, names in os.walk(snapshot.index_path):
        directory = Path(directory)
        if directory == snapshot.index_path:
            dirs[:] = [name for name in dirs if name != 'train_tables']
        destination = index / directory.relative_to(snapshot.index_path)
        destination.mkdir(parents=True)
        for name in names:
            os.link(directory / name, destination / name)
    shutil.copytree(prediction.staging_dir, inputs / 'prediction')
    weekly.save(
        inputs / 'snapshot.json',
        dict(id=snapshot.id, index_path='index',
             graph_config=snapshot.graph_config))
    weekly.save(
        inputs / 'prediction.json',
        dict(
            staging_dir='prediction', table_path=str(
                Path('prediction') /
                prediction.table_path.relative_to(prediction.staging_dir)),
            row_count=prediction.row_count, tt_id=prediction.tt_id,
            timestamps=list(prediction.timestamps),
            required_stages=list(prediction.required_stages)))
    files = weekly.inventory(inputs)
    case = dict(query=prediction.query, snapshot='snapshot.json',
                prediction_table='prediction.json')
    imported_snapshot, imported_table = worker.prepared(inputs, files, case,
                                                        tmp_path / 'scratch')

    def predict(model, table, graph):
        result = model.predict(table, snapshot=graph,
                               binary_classification_threshold=.5)
        frame = pd.concat(pd.read_parquet(path)
                          for path in result.files).sort_values('ENTITY')
        return dict(entity_ids=frame.ENTITY.tolist(),
                    values=[[v] for v in frame.True_PROB.tolist()],
                    decisions=frame.TARGET_PRED.tolist())

    before = predict(trainer, prediction, snapshot)

    def reject_fit(*args, **kwargs):
        pytest.fail('Reference replay attempted an extra fit')

    monkeypatch.setattr(kumo.Trainer, 'fit', reject_fit)
    # The shared fixture already completed its small CPU fit. The replay only loads and predicts.
    bundle = trainer._training_result.bundle_dir
    result = weekly.replay_reference(
        bundle, before, lambda path: kumo.Trainer.load(path, accelerator='cpu'),
        lambda model: predict(model, imported_table, imported_snapshot),
        atol=1e-4, rtol=1e-5)
    assert result['status'] == 'PASS'
    assert result['entities'] == len(before['entity_ids'])
    assert weekly.inventory(inputs) == files
    assert not (index / 'train_tables').exists()
    assert (imported_snapshot.index_path / 'train_tables').is_dir()
    broken = tmp_path / 'unsupported-bundle'
    shutil.copytree(bundle, broken)
    specification = weekly.read(broken / 'training-spec.json')
    specification['spec_version'] = 999
    weekly.save(broken / 'training-spec.json', specification)
    with pytest.raises(ValueError):
        weekly.replay_reference(
            broken, before,
            lambda path: kumo.Trainer.load(path, accelerator='cpu'),
            lambda model: predict(model, imported_table, imported_snapshot),
            atol=1e-4, rtol=1e-5)
