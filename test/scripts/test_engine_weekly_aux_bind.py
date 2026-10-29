# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze runtime fields without changing the admitted auxiliary workload."""

import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve(
).parents[2] / 'packages/kumo-engine/scripts/weekly_aux_bind.py'


def runtime():
    return dict(mode='reference', expected_prefix='/new/venv',
                wheel=dict(path='/wheel.whl', sha256='1' * 64, version='1.0.0'),
                dependency_profile=dict(path='/deps.json', sha256='2' * 64),
                host_lock='/host.lock')


def template():
    return dict(schema=1, check='P1', mode='reference',
                expected_prefix='/old/venv', host_lock='/host.lock',
                wheel=dict(path='/old.whl', sha256='1' * 64, version='1.0.0'),
                dependency_profile=dict(path='/deps.json', sha256='2' * 64),
                source_sha256='3' * 64, data_sha256='4' * 64,
                work=dict(n=300000), worker=dict(index='/full/index'),
                inputs={'/full/index/meta.json': '5' * 64})


def render(*args, **kwargs):
    return runpy.run_path(str(SCRIPT))['render'](*args, **kwargs)


def test_runtime_binding_preserves_frozen_data_source_and_work():
    original = template()
    bound = render(original, runtime())
    assert bound['expected_prefix'] == '/new/venv'
    for key in ('source_sha256', 'data_sha256', 'work', 'worker', 'inputs'):
        assert bound[key] == original[key]
    assert original['expected_prefix'] == '/old/venv'


@pytest.mark.parametrize(
    'change',
    ['unknown', 'nested_unknown', 'lock', 'reference_version', 'dependencies'])
def test_runtime_binding_rejects_unapproved_field_or_environment_drift(change):
    selected = runtime()
    if change == 'unknown':
        selected['work'] = dict(n=1)
    elif change == 'nested_unknown':
        selected['wheel']['model_plan'] = '/other/plan.json'
    elif change == 'lock':
        selected['host_lock'] = '/other.lock'
    elif change == 'reference_version':
        selected['wheel']['version'] = '2.0.0'
    else:
        selected['dependency_profile']['sha256'] = 'f' * 64
    with pytest.raises(ValueError):
        render(template(), selected)


def retained():
    return dict(
        path='/case/weekly-case.json', sha256='6' * 64, mode='reference',
        version='1.0.0', wheel_sha256='1' * 64,
        result=dict(fit_id='new-P2-fit',
                    expected_reload_checkpoint_sha256='7' * 64),
        device=dict(gpu='NVIDIA L4'), case_spec=dict(
            workload='P2', population=dict(expected_prediction_count=200000),
            worker=dict(path='/frozen/model-worker.py', sha256='8' * 64)),
        model_bundle='/case/worker/saved-model', prediction_batch_size=1024,
        model_files={'model/checkpoint': '7' * 64})


def test_b9_binding_uses_actual_retained_model_and_preserves_static_sources():
    source = template()
    source['check'] = 'B9'
    source['prediction_batch_size'] = 1024
    source['worker'] = dict(b9_helper=dict(path='/frozen/b9.py', sha256='9' *
                                           64))
    bound = render(source, runtime(), retained())
    assert bound['retained_fit_id'] == 'new-P2-fit'
    assert bound['population'] == retained()['case_spec']['population']
    assert bound['model_files'] == retained()['model_files']
    assert bound['worker']['b9_helper'] == source['worker']['b9_helper']
    assert bound['worker']['model_helper'] == retained()['case_spec']['worker']
    assert bound['source_sha256'] == source['source_sha256']


@pytest.mark.parametrize(
    'change', ['missing', 'wheel', 'workload', 'overwritten_input', 'batch'])
def test_b9_binding_rejects_missing_or_foreign_retained_evidence(change):
    source, evidence = template(), retained()
    source['check'] = 'B9'
    source['prediction_batch_size'] = 1024
    if change == 'missing':
        evidence = None
    elif change == 'wheel':
        evidence['wheel_sha256'] = 'f' * 64
    elif change == 'workload':
        evidence['case_spec']['workload'] = 'tiny'
    elif change == 'batch':
        source['prediction_batch_size'] = 512
    else:
        source['inputs'][evidence['path']] = 'f' * 64
    original = copy.deepcopy(source)
    with pytest.raises(ValueError):
        render(source, runtime(), evidence)
    assert source == original


@pytest.mark.parametrize('override,expected', [(None, 512), ({}, 512),
                                               ({
                                                   'batch_size': 256
                                               }, 256)])
def test_prediction_batch_size_uses_hash_pinned_saved_native_config(
        tmp_path, override, expected):
    bundle = tmp_path / 'saved-model'
    (bundle / 'model').mkdir(parents=True)
    config = bundle / 'model/config.json'
    config.write_text(
        json.dumps(
            dict(loader_kwargs=dict(batch_size=512),
                 eval_loader_kwargs=override)))
    files = {
        'model/config.json': hashlib.sha256(config.read_bytes()).hexdigest()
    }
    assert runpy.run_path(str(SCRIPT))['prediction_batch_size'](
        bundle, files) == expected
    config.write_text('{}')
    with pytest.raises(ValueError):
        runpy.run_path(str(SCRIPT))['prediction_batch_size'](bundle, files)


@pytest.mark.parametrize('changed',
                         [None, 'template_bytes', 'capacity_graph', 'missing'])
def test_static_auxiliary_admission_rejects_drift_before_model_work(
        tmp_path, changed):
    api = runpy.run_path(str(SCRIPT))

    def record(path, value):
        path.write_text(json.dumps(value))
        return dict(path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    runner = SCRIPT.with_name('weekly_aux.py')
    runner_record = dict(path=str(runner),
                         sha256=hashlib.sha256(runner.read_bytes()).hexdigest())
    primary = dict(workload='P2', identity=dict(data_sha256='a' * 64),
                   population=dict(expected_prediction_count=200000),
                   hardware_profile='scale-l4')
    bindings = {}
    for check in ('P1', 'B9'):
        capacity = dict(status='ADMITTED', check=check,
                        native_capacity_checked=True,
                        identity=copy.deepcopy(primary['identity']),
                        population=primary['population'],
                        hardware_profile='scale-l4', prediction_batch_size=1024)
        if changed == 'capacity_graph' and check == 'B9':
            capacity['identity']['data_sha256'] = 'f' * 64
        source = dict(
            schema=1, check=check, host_lock='/host.lock',
            source_sha256=runner_record['sha256'], inputs={},
            capacity=record(tmp_path / f'{check}-capacity.json',
                            capacity), prediction_batch_size=1024)
        bindings[check] = dict(
            runner=runner_record,
            template=record(tmp_path / f'{check}-template.json', source))
    if changed == 'template_bytes':
        Path(bindings['P1']['template']['path']).write_text('{}')
    elif changed == 'missing':
        del bindings['B9']
    if changed:
        with pytest.raises(ValueError):
            api['admit_bindings'](bindings, '/host.lock', primary)
    else:
        api['admit_bindings'](bindings, '/host.lock', primary)
