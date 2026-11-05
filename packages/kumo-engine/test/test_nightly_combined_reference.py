# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference cases create their answers without an earlier answer bundle."""
import copy
import json
from contextlib import contextmanager

import pandas as pd
import pytest
import test_nightly_reference_answers as previous

case_runner = previous.case_runner
future_targets = previous.future_targets
answer_inputs = previous.answer_inputs


@pytest.mark.parametrize('task', ['regression', 'forecasting', 'static_link'])
def test_current_prediction_population_contains_only_observed_values(
        case_runner, task):
    truth = pd.DataFrame(dict(ENTITY1=[2, 1], TARGET1=[8., 9.], SPLIT=[2, 2]))
    if task != 'static_link':
        truth['TIMESTAMP'] = pd.to_datetime(['2024-01-01'] * 2)
    if task == 'forecasting':
        truth['autoregression_labels'] = [[1., 2.], [3., 4.]]
        truth['future_labels'] = [[8., 8.], [9., 9.]]
    frame = case_runner.prediction_population(truth, task)
    assert frame.ENTITY1.tolist() == [1, 2]
    assert not {'SPLIT', 'TARGET1', 'future_labels'} & set(frame)
    if task == 'forecasting':
        assert frame.autoregression_labels.tolist() == [[3., 4.], [1., 2.]]


def test_reference_control_needs_no_saved_answers(case_runner, answer_inputs,
                                                  tmp_path):
    root, manifest, normal, control, query, snapshot, helpers = answer_inputs
    control.pop('control_rows')
    control['control_recipe'] = dict(method='stage-permutation-v1', seed=0)
    _, stages, proof = case_runner.generate_training_table(
        root, manifest, control, query, snapshot, tmp_path / 'work',
        helpers=helpers, bootstrap=True)
    assert set(stages) == {'train', 'val', 'test'}
    assert proof['answer_policy'] == 'record_reference'
    assert proof['control_recipe'] == control['control_recipe']


def test_reference_records_tables_before_any_model_fit(case_runner,
                                                       answer_inputs, tmp_path):
    root, manifest, normal, _, query, snapshot, helpers = answer_inputs
    output = tmp_path / 'evidence'
    _, _, proof = case_runner.generate_training_table(
        root, manifest, normal, query, snapshot, tmp_path / 'work',
        helpers=helpers, bootstrap=True, reference_output=output)
    receipt = json.loads((output / 'reference-data.json').read_text())
    assert receipt['status'] == 'PREPARED'
    assert receipt['stages'] == proof['stages']
    assert receipt['files']
    for name, expected in receipt['files'].items():
        assert helpers.digest(output / name) == expected['sha256']
        assert (output / name).stat().st_size == expected['bytes']


def test_reference_answers_survive_failure_before_training(
        case_runner, answer_inputs, tmp_path, monkeypatch):
    import kumo.engine as kumo

    root, manifest, normal, _, query, snapshot, helpers = answer_inputs
    pd.DataFrame(dict(id=[10, 20, 30, 40])).to_parquet(root / 'catalog.parquet')
    manifest['files']['catalog.parquet'] = helpers.digest(root /
                                                          'catalog.parquet')
    normal['catalog'] = dict(path='catalog.parquet', column='id')
    monkeypatch.setattr(kumo, 'PredictiveQuery', lambda *args: query)

    @contextmanager
    def fail_before_model(*args, **kwargs):
        raise RuntimeError('deliberate failure before training')
        yield

    monkeypatch.setattr(case_runner, 'observe_cuda', fail_before_model)
    output = tmp_path / 'result'
    with pytest.raises(RuntimeError, match='deliberate failure'):
        case_runner.run_case(root, manifest, normal, object(),
                             snapshot, output, tmp_path / 'scratch',
                             dict(atol=None), helpers=helpers,
                             measure_only=True)
    receipt = json.loads(
        (output / 'reference-data/reference-data.json').read_text())
    assert receipt['status'] == 'PREPARED'
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'FAIL'
    assert result['physical_fits'] == 0
    for name, expected in receipt['files'].items():
        assert helpers.digest(output / 'reference-data' /
                              name) == expected['sha256']


def test_source_only_candidate_control_compares_approved_answers(
        case_runner, answer_inputs, tmp_path):
    root, manifest, normal, control, query, snapshot, helpers = answer_inputs
    control.pop('control_rows')
    control['control_recipe'] = dict(method='stage-permutation-v1', seed=0)
    _, _, reference = case_runner.generate_training_table(
        root, manifest, control, query, snapshot, tmp_path / 'reference',
        helpers=helpers, bootstrap=True)
    normal['stage_identity'] = reference['normal_stages']
    control['stage_identity'] = reference['stages']
    _, _, candidate = case_runner.generate_training_table(
        root, manifest, control, query, snapshot, tmp_path / 'candidate',
        helpers=helpers)
    assert candidate['stages'] == reference['stages']
    control['stage_identity'] = copy.deepcopy(reference['stages'])
    control['stage_identity']['test']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Candidate control'):
        case_runner.generate_training_table(root, manifest, control, query,
                                            snapshot, tmp_path / 'corrupt',
                                            helpers=helpers)


def test_reference_control_reuses_current_ordered_tables(
        case_runner, answer_inputs, tmp_path, monkeypatch):
    root, manifest, _, control, query, snapshot, helpers = answer_inputs
    sorts = []
    original = helpers.ordered_batches

    def observed_sort(files, *args, **kwargs):
        sorts.append(tuple(files))
        yield from original(files, *args, **kwargs)

    monkeypatch.setattr(helpers, 'ordered_batches', observed_sort)
    _, _, proof = case_runner.generate_training_table(root, manifest, control,
                                                      query, snapshot,
                                                      tmp_path / 'work',
                                                      helpers=helpers,
                                                      bootstrap=True)
    assert proof['control_recipe'] == control['control_recipe']
    assert all(stage['rows'] == 2 for stage in proof['stages'].values())
    assert len(sorts) == 2


def test_failed_durable_save_prevents_model_work(case_runner, answer_inputs,
                                                 tmp_path, monkeypatch):
    from types import SimpleNamespace

    import kumo.engine as kumo

    root, manifest, normal, _, query, snapshot, helpers = answer_inputs
    pd.DataFrame(dict(id=[10, 20, 30, 40])).to_parquet(root / 'catalog.parquet')
    manifest['files']['catalog.parquet'] = helpers.digest(root /
                                                          'catalog.parquet')
    normal['catalog'] = dict(path='catalog.parquet', column='id')
    monkeypatch.setattr(kumo, 'PredictiveQuery', lambda *args: query)
    calls = []
    sibling = helpers.sibling

    def publish(directory, destination):
        assert (directory / 'reference-data/reference-data.json').is_file()
        assert (directory / 'current-population.parquet').is_file()
        assert destination == 's3://test-bucket/proposals'
        calls.append('publish')
        raise OSError('durable storage unavailable')

    monkeypatch.setattr(
        helpers, 'sibling', lambda name: SimpleNamespace(publish=publish)
        if name == 'nightly_evidence' else sibling(name))

    def forbid_model(*args, **kwargs):
        pytest.fail('Model work started before durable data preservation')

    monkeypatch.setattr(case_runner, 'observe_cuda', forbid_model)
    with pytest.raises(OSError, match='durable storage unavailable'):
        case_runner.run_case(root, manifest, normal, object(), snapshot,
                             tmp_path / 'result', tmp_path / 'scratch',
                             dict(atol=None), helpers=helpers,
                             measure_only=True,
                             reference_destination='s3://test-bucket/proposals')
    assert calls == ['publish']
