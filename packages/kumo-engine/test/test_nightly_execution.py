# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep independent cases running when a GPU boundary fails before output."""

import hashlib
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_check.py'
SPEC = importlib.util.spec_from_file_location('nightly_execution', SOURCE)
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)


def measurement():
    return dict(
        format='kumo-engine-nightly-resources-v1', status='MEASURED',
        task='binary', arm='normal', input_signature={},
        input_manifest_sha256='a' * 64, wheel_sha256='a' * 64,
        harness_sha256='b' * 64, gpu_class='RTX PRO 6000', gpus=1, seconds=9.1,
        peaks=dict(cpu_cores=2.1, ram_bytes=100, scratch_bytes=200,
                   gpu_memory_bytes=50))


def test_resource_minima_come_from_measured_peaks_with_headroom():
    profile = nightly.resource_requirements(measurement())
    assert profile['minimum'] == dict(cpus=3, ram_bytes=125, scratch_bytes=250,
                                      gpus=1, gpu_memory_bytes=63,
                                      time_seconds=12)
    assert profile['measurement']['gpu_class'] == 'RTX PRO 6000'
    assert nightly.admit_resources(profile,
                                   profile['minimum']) == profile['minimum']
    for field in ('cpus', 'ram_bytes', 'scratch_bytes', 'gpus',
                  'gpu_memory_bytes', 'time_seconds'):
        capacity = dict(profile['minimum'])
        capacity[field] -= 1
        with pytest.raises(ValueError, match=field):
            nightly.admit_resources(profile, capacity)
        with pytest.raises(ValueError, match=field):
            nightly.admit_resources(
                profile, dict(profile['minimum'], **{field: float('nan')}))
    assert nightly.admit_resources(
        profile, dict(profile['minimum'], gpu_class='NVIDIA A10G'))


@pytest.mark.parametrize('available,gpus,memory,passes', [
    (True, 1, 24 * 1024**3, True),
    (False, 1, 24 * 1024**3, False),
    (True, 0, 24 * 1024**3, False),
    (True, 2, 24 * 1024**3, False),
    (True, 1, 0, False),
])
def test_runtime_uses_cuda_capacity_without_requiring_a_gpu_model(
        monkeypatch, available, gpus, memory, passes):
    import torch

    import kumo.engine as kumo

    prefix = nightly.sys.prefix
    monkeypatch.setattr(nightly, 'sys',
                        SimpleNamespace(platform='linux',
                                        version_info=(3, 12), prefix=prefix))
    monkeypatch.setattr(nightly, 'platform',
                        SimpleNamespace(machine=lambda: 'x86_64'))
    monkeypatch.setattr(kumo, '__file__',
                        str(Path(prefix) / 'lib/kumo/engine/__init__.py'))
    metadata = SimpleNamespace(
        distributions=lambda: [], version=lambda name: kumo.__version__,
        distribution=lambda name: SimpleNamespace(read_text=lambda filename:
            json.dumps(dict(archive_info=dict(hashes=dict(sha256='a' * 64))))))
    monkeypatch.setattr(nightly, 'importlib', SimpleNamespace(metadata=metadata))
    monkeypatch.setattr(nightly, 'subprocess',
                        SimpleNamespace(check_output=lambda *args, **kwargs:
                                        '580.0\n'))
    properties = SimpleNamespace(name='NVIDIA A10G', total_memory=memory,
                                 uuid='GPU-12345678-1234-1234-1234-123456789abc',
                                 major=8, minor=6)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: available)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: gpus)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda i: properties)
    if passes:
        runtime = nightly.verify_runtime(kumo.__version__, None)
        assert runtime['gpu'] == 'NVIDIA A10G'
        assert runtime['gpu_memory_bytes'] == 24 * 1024**3
        assert runtime['compute_capability'] == [8, 6]
    else:
        with pytest.raises(ValueError):
            nightly.verify_runtime(kumo.__version__, None)


@pytest.mark.parametrize('damage', [
    dict(status='INCOMPLETE'),
    dict(wheel_sha256='unbound'),
    dict(peaks=dict(cpu_cores=0, ram_bytes=100, scratch_bytes=200,
                    gpu_memory_bytes=50))
])
def test_unmeasured_resources_cannot_be_admitted(damage):
    with pytest.raises(ValueError):
        nightly.resource_requirements(dict(measurement(), **damage))
    with pytest.raises(ValueError):
        nightly.admit_resources(None, {})


def prepared_run(tmp_path, monkeypatch, failure=None):
    root, output, scratch = (tmp_path / name
                             for name in ('inputs', 'evidence', 'scratch'))
    root.mkdir()
    wheel = tmp_path / 'candidate.whl'
    wheel.write_bytes(b'test wheel boundary')
    cases, previous, learning, files = [], [], {}, {}
    for task, definition in nightly.smoke.TASKS.items():
        lower = task in ('regression', 'forecasting')
        for arm in ('normal', 'control'):
            case = dict(task=task, arm=arm, seed=0, graph=definition['graph'],
                        descriptor=task + '-' + arm + '.json',
                        prediction_input=task + '-full.parquet',
                        partial_input=task + '-partial.parquet',
                        requested_plan_sha256='d' * 64)
            for field in ('descriptor', 'prediction_input', 'partial_input'):
                files[case[field]] = 'e' * 64
            measured = dict(
                measurement(), task=task, arm=arm,
                input_signature=nightly.input_signature({'files': files}, case))
            case['requirements'] = nightly.resource_requirements(measured)
            cases.append(case)
            previous.append(
                dict(task=task, arm=arm, baseline=1. if lower else .5,
                     score=1. if lower else .5))
        learning[task] = dict(min_normal_gain=.1, max_control_drift=.02)
    graphs = {
        case['graph']:
        dict(schema='schema.json', profile=case['graph'], static=False,
             sources={})
        for case in cases
    }
    nightly.save(root / 'schema.json', {key: {} for key in graphs})
    manifest = dict(cases=cases, graphs=graphs, files=files)
    capacity = cases[0]['requirements']['minimum']
    if failure == 'requirements':
        del cases[0]['requirements']
    elif failure == 'identity':
        cases[0]['requirements']['measurement']['arm'] = 'control'
    policy = dict(learning=learning,
                  predictions={task: dict(atol=.0004)
                               for task in learning})
    monkeypatch.setattr(nightly, 'load_inputs', lambda *args: manifest)
    monkeypatch.setattr(nightly, 'load_reference',
                        lambda args: {'cases': previous})
    monkeypatch.setattr(nightly, 'reference_cases', lambda *args: None)
    monkeypatch.setattr(nightly, 'load_policy', lambda *args: policy)
    runtime = dict(wheel_sha256=nightly.digest(wheel), gpu='RTX PRO 6000',
                   gpu_count=1)
    monkeypatch.setattr(nightly, 'verify_runtime',
                        lambda *args, **kwargs: runtime)
    monkeypatch.setattr(nightly, 'verify_installed_files', lambda *args: 1)
    monkeypatch.setattr(nightly, 'check_install', lambda *args, **kwargs: None)
    monkeypatch.setattr(nightly, 'check_snapshot', lambda *args: {})
    monkeypatch.setattr(nightly, 'available_capacity', lambda *args: capacity)
    monkeypatch.setattr(nightly, 'track_resources',
                        lambda *args: nullcontext(measurement()))
    if failure == 'measurement':

        def observe(output, *args):
            result = measurement()
            if output.name == 'binary-normal':
                result['peaks']['cpu_cores'] = 0
            return nullcontext(result)

        monkeypatch.setattr(nightly, 'track_resources', observe)
    built = []

    def build(*args, **kwargs):
        built.append(object())
        if failure == 'snapshot' and len(built) == 1:
            raise RuntimeError('ingest failed before creating case output')
        return built[-1], built[-1]

    def execute(root, manifest, case, graph, snapshot, directory, *args,
                **kwargs):
        if failure == 'case' and case is cases[0]:
            raise RuntimeError('case failed before creating output')
        lower = case['task'] in ('regression', 'forecasting')
        row = dict(
            task=case['task'], arm=case['arm'], seed=0, status='MEASURED',
            physical_fits=1, fit_id=case['task'] + '-' + case['arm'],
            checkpoint_sha256='a' * 64, seconds=1.,
            prediction_checks=dict(status='PASS'),
            prediction_batches={stage: 1
                                for stage in nightly.PREDICTIONS},
            score=(1. if lower else .5) if case['arm'] == 'control' else
            (.7 if lower else .8))
        nightly.save(directory / 'result.json', row)
        return row

    monkeypatch.setattr(nightly, 'build_graph', build)
    monkeypatch.setattr(nightly, 'run_case', execute)
    args = SimpleNamespace(input_root=root, output_dir=output, scratch=scratch,
                           input_sha256='a' * 64, reference_sha256='b' * 64,
                           policy=tmp_path / 'policy.json',
                           policy_sha256='c' * 64, reference_only=False,
                           expected_version='1.0', wheel=wheel)
    return args, built, cases


@pytest.mark.parametrize(
    'failure', ['snapshot', 'case', 'requirements', 'identity', 'measurement'])
def test_pre_output_failure_preserves_every_other_case(tmp_path, monkeypatch,
                                                       failure):
    args, built, _ = prepared_run(tmp_path, monkeypatch, failure)
    assert nightly.run(args) is False
    result = nightly.read(args.output_dir / 'nightly-result.json')
    assert len(result['runs']) == 16
    assert result['runs'][0]['status'] == 'FAIL'
    assert all(row['status'] == 'MEASURED' for row in result['runs'][1:])
    assert len(built) == (15 if failure in ('requirements', 'identity') else 16)
    assert nightly.read(args.output_dir / 'comparison.json')['status'] == 'FAIL'


def test_measurement_mode_proposes_requirements_without_approving_a_case(
        tmp_path, monkeypatch):
    args, built, cases = prepared_run(tmp_path, monkeypatch)
    del cases[0]['requirements']
    args.case = 'binary-normal'
    args.measure_resources = True
    args.reference_only = True
    args.time_budget_seconds = 12
    assert nightly.run(args) is True
    assert len(built) == 1
    report = nightly.read(args.output_dir / 'nightly-result.json')
    assert report['status'] == 'MEASURED'
    assert nightly.read(args.output_dir /
                        'comparison.json')['status'] == 'MEASURED'
    proposal = nightly.read(args.output_dir /
                            'binary-normal/requirements-proposal.json')
    assert proposal['measurement']['input_manifest_sha256'] == 'a' * 64
    assert proposal['minimum']['time_seconds'] == 12


@pytest.mark.parametrize('dimension', [
    'cpus', 'ram_bytes', 'scratch_bytes', 'gpus',
    'gpu_memory_bytes', 'time_seconds'
])
def test_insufficient_capacity_stops_before_snapshot_build(
        tmp_path, monkeypatch, dimension):
    args, built, cases = prepared_run(tmp_path, monkeypatch)
    args.case = 'binary-normal'
    capacity = dict(cases[0]['requirements']['minimum'])
    capacity[dimension] -= 1
    monkeypatch.setattr(nightly, 'available_capacity', lambda *args: capacity)
    assert nightly.run(args) is False
    assert built == []
    [result] = nightly.read(args.output_dir / 'nightly-result.json')['runs']
    assert result['error'].startswith('ValueError: Insufficient ' + dimension)


def test_different_gpu_model_does_not_block_case_with_enough_capacity(
        tmp_path, monkeypatch):
    args, built, cases = prepared_run(tmp_path, monkeypatch)
    args.case = 'binary-normal'
    capacity = dict(cases[0]['requirements']['minimum'], gpu_class='NVIDIA A10G')
    monkeypatch.setattr(nightly, 'available_capacity', lambda *args: capacity)
    assert nightly.run(args) is True
    assert len(built) == 1


def test_standalone_case_records_its_declared_time_limit(tmp_path, monkeypatch):
    args, built, cases = prepared_run(tmp_path, monkeypatch)
    args.case = 'binary-normal'
    required = cases[0]['requirements']['minimum']
    monkeypatch.setattr(
        nightly, 'available_capacity',
        lambda scratch, runtime, budget: dict(required, time_seconds=budget))
    assert nightly.run(args) is True
    assert len(built) == 1
    [result] = nightly.read(args.output_dir / 'nightly-result.json')['runs']
    assert result['admission']['capacity']['time_seconds'] == required[
        'time_seconds']


def test_reviewed_case_requirements_are_bound_by_checksum(
        tmp_path, monkeypatch):
    args, built, cases = prepared_run(tmp_path, monkeypatch)
    profile = cases[0].pop('requirements')
    path = tmp_path / 'requirements.json'
    path.write_text(json.dumps(profile))
    args.case = 'binary-normal'
    args.requirements = path
    args.requirements_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    assert nightly.run(args) is True
    assert len(built) == 1
    assert nightly.read(args.output_dir /
                        'nightly-result.json')['status'] == 'PASS'
    path.write_text('{}')
    args.output_dir = tmp_path / 'tampered-evidence'
    args.scratch = tmp_path / 'tampered-scratch'
    with pytest.raises(ValueError, match='checksum'):
        nightly.run(args)
    assert len(built) == 1


def test_caller_uses_repository_checksums_instead_of_forwarded_overrides(
        tmp_path, monkeypatch):
    args, built, cases = prepared_run(tmp_path, monkeypatch)
    profile = cases[0].pop('requirements')
    requirements = tmp_path / 'requirements.json'
    requirements.write_text(json.dumps(profile))
    build = tmp_path / 'build.json'
    build.write_text(json.dumps(dict(version='1.0')))
    config = tmp_path / 'inputs.json'
    config.write_text(
        json.dumps(
            dict(
                format='kumo-engine-nightly-job-v1', min_free_bytes=1,
                input_manifest_sha256='a' * 64,
                inputs=dict(bucket='shared', region='us-west-2',
                            key='inputs.tar', sha256='d' * 64),
                reference=dict(path='reference.json', sha256='b' * 64),
                policy=dict(path='policy.json', sha256='c' * 64))))
    spec = importlib.util.spec_from_file_location(
        'job', SOURCE.with_name('nightly_job.py'))
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    monkeypatch.setattr(job, 'load_script', lambda path: nightly)
    result = job.main([
        'run', '--configuration',
        str(config), '--build-receipt',
        str(build), '--input-root',
        str(args.input_root), '--output-dir',
        str(args.output_dir), '--scratch',
        str(args.scratch), '--wheel',
        str(args.wheel), '--case', 'binary-normal', '--requirements',
        str(requirements), '--requirements-sha256',
        hashlib.sha256(requirements.read_bytes()).hexdigest(), '--input-sha256',
        'f' * 64
    ])
    assert result == 0
    assert len(built) == 1
    report = nightly.read(args.output_dir / 'nightly-result.json')
    assert report['input_manifest_sha256'] == 'a' * 64
