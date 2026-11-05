# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference runs record answers; candidate runs compare approved answers."""

import copy
import importlib.util
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_nightly_case

SOURCE = test_nightly_case.SOURCE
case_runner = test_nightly_case.case_runner
future_targets = test_nightly_case.future_targets


@pytest.fixture
def answer_inputs(case_runner, future_targets, tmp_path, monkeypatch):
    from kumoapi.json_serde import to_json_dict

    query, snapshot, plan, _, _, rows = future_targets
    root = tmp_path / 'inputs'
    root.mkdir()
    (root / 'normal.json').write_text(
        json.dumps(
            dict(query=query.query, query_generation_plan=to_json_dict(plan))))
    (root / 'staging').mkdir()
    pq.write_table(rows, root / 'control.parquet')
    spec = importlib.util.spec_from_file_location(
        'reference_answer_helpers', SOURCE.with_name('nightly_check.py'))
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    def reject_saved_normal(*args):
        pytest.fail('Reference generation read saved normal answers')

    monkeypatch.setattr(helpers, 'load_table', reject_saved_normal)
    stale = {
        name: dict(rows=999, sha256='0' * 64)
        for name in ('train', 'val', 'test')
    }
    normal = dict(scale='100M', task='temporal_link', arm='normal', seed=0,
                  descriptor='normal.json', stage_identity=stale)
    control = dict(normal, arm='control', control_rows='control.parquet',
                   control_recipe=dict(method='stage-permutation-v1', seed=0),
                   stage_identity=copy.deepcopy(stale))
    manifest = dict(
        cases=[normal, control], files={
            name: helpers.digest(root / name)
            for name in ('normal.json', 'control.parquet')
        })
    return root, manifest, normal, control, query, snapshot, helpers


@pytest.mark.parametrize('arm', ['normal', 'control'])
def test_reference_records_current_answers_despite_stale_saved_hashes(
        case_runner, answer_inputs, tmp_path, arm):
    root, manifest, normal, control, query, snapshot, helpers = answer_inputs
    case = normal if arm == 'normal' else control
    _, stages, proof = case_runner.generate_training_table(
        root, manifest, case, query, snapshot, tmp_path / 'reference',
        helpers=helpers, bootstrap=True)
    assert set(stages) == {'train', 'val', 'test'}
    assert {
        name: identity['rows']
        for name, identity in proof['stages'].items()
    } == dict(train=2, val=2, test=2)
    assert all(identity['sha256'] != '0' * 64
               for identity in proof['stages'].values())
    assert proof['answer_policy'] == 'record_reference'


def test_reference_does_not_load_saved_normal_payload_when_hashes_are_absent(
        case_runner, answer_inputs, tmp_path):
    root, manifest, normal, _, query, snapshot, helpers = answer_inputs
    normal.pop('stage_identity')
    _, _, proof = case_runner.generate_training_table(root, manifest, normal,
                                                      query, snapshot,
                                                      tmp_path / 'reference',
                                                      helpers=helpers,
                                                      bootstrap=True)
    assert proof['normal_stages']['train']['rows'] == 2


@pytest.mark.parametrize('arm', ['normal', 'control'])
def test_candidate_still_rejects_stale_answer_hashes(case_runner, answer_inputs,
                                                     tmp_path, arm):
    root, manifest, normal, control, query, snapshot, helpers = answer_inputs
    case = normal if arm == 'normal' else control
    if arm == 'control':
        normal['stage_identity'] = case_runner.stage_identities(
            [str(root / 'control.parquet')], 'temporal_link',
            tmp_path / 'normal-identity', helpers=helpers)
    with pytest.raises(ValueError,
                       match=f'{arm} stage counts or labels differ'):
        case_runner.generate_training_table(root, manifest, case, query,
                                            snapshot, tmp_path / 'candidate',
                                            helpers=helpers)


def test_candidate_still_requires_expected_answers(case_runner, answer_inputs,
                                                   tmp_path):
    root, manifest, normal, _, query, snapshot, helpers = answer_inputs
    normal.pop('stage_identity')
    (root / 'staging').rmdir()
    with pytest.raises(ValueError, match='reviewed normal-stage hashes'):
        case_runner.generate_training_table(root, manifest, normal, query,
                                            snapshot, tmp_path / 'candidate',
                                            helpers=helpers)


def test_candidate_control_rejects_changed_input_keys(case_runner,
                                                      answer_inputs, tmp_path):
    root, manifest, normal, control, query, snapshot, helpers = answer_inputs
    normal['stage_identity'] = case_runner.stage_identities(
        [str(root / 'control.parquet')], 'temporal_link',
        tmp_path / 'normal-identity', helpers=helpers)
    rows = pq.read_table(root / 'control.parquet')
    entities = rows['ENTITY1'].to_pylist()
    entities[0] = 999
    rows = rows.set_column(rows.schema.get_field_index('ENTITY1'), 'ENTITY1',
                           pa.array(entities, rows['ENTITY1'].type))
    pq.write_table(rows, root / 'control.parquet')
    manifest['files']['control.parquet'] = helpers.digest(root /
                                                          'control.parquet')
    with pytest.raises(ValueError, match='entity, time, or split keys differ'):
        case_runner.generate_training_table(root, manifest, control, query,
                                            snapshot, tmp_path / 'reference',
                                            helpers=helpers)


def test_reference_control_still_detects_answers_lost_during_import(
        case_runner, answer_inputs, tmp_path, monkeypatch):
    import kumo.engine as kumo

    root, manifest, _, control, query, snapshot, helpers = answer_inputs
    original = kumo.TrainingTable.from_source_table

    def legacy_import(*args, **kwargs):
        kwargs['allow_new_target_entities'] = False
        return original(*args, **kwargs)

    monkeypatch.setattr(kumo.TrainingTable, 'from_source_table',
                        staticmethod(legacy_import))
    with pytest.raises(ValueError, match='control.*labels differ'):
        case_runner.generate_training_table(root, manifest, control, query,
                                            snapshot, tmp_path / 'reference',
                                            helpers=helpers, bootstrap=True)
