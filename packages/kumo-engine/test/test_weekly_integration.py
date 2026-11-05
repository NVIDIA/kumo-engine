# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check weekly admission and delegation without starting a GPU campaign."""

import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).parents[1] / 'scripts'


def script(name):
    spec = importlib.util.spec_from_file_location('weekly_test_' + name,
                                                  SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def configuration(tmp_path):
    shared = script('nightly_check')
    value = shared.read(SCRIPTS.parent / 'test/nightly/inputs.json')
    value['format'] = 'kumo-engine-weekly-job-v2'
    value['scales'] = {
        scale:
        dict(inputs=None, input_manifest_sha256=None,
             pending='Freeze the matching seven-family archive')
        for scale in ('10B', '25B')
    }
    for name in ('reference', 'policy', 'requirements'):
        path = tmp_path / (name + '.json')
        shared.save(path, dict(status='PENDING_MEASUREMENT', cases=[]))
        value[name] = dict(path=path.name, sha256=shared.digest(path))
    path = tmp_path / 'inputs.json'
    shared.save(path, value)
    return path, value


def test_weekly_configuration_declares_matching_forty_eight_cases(
        configuration):
    path, _ = configuration
    shared = script('nightly_check')
    value = shared.load_configuration(path)
    cases = shared.expected_cases(value)
    names = {
        'binary', 'multiclass', 'multilabel', 'multilabel_ranking',
        'regression', 'forecasting', 'static_link', 'temporal_link',
        'streaming', 'community', 'payments', 'fraud'
    }
    assert {(case['scale'], shared.case_name(case))
            for case in cases} == {(scale, name + '-' + arm)
                                   for scale in ('10B', '25B')
                                   for name in names
                                   for arm in ('normal', 'control')}
    assert len(cases) == 48


def test_committed_records_bind_all_weekly_cases_without_approving_numbers():
    shared = script('nightly_check')
    path = SCRIPTS.parent / 'test/weekly/inputs.json'
    value = shared.load_configuration(path)
    cases = shared.expected_cases(value)
    assert len(cases) == 48
    assert {row['scale'] for row in cases} == {'10B', '25B'}
    records = shared.runtime_records(path, value, mode='reference')
    gates = shared.sibling('nightly_gates')
    assert {gates.case_key(row)
            for row in records['reference']['cases']
            } == {gates.case_key(row)
                  for row in cases}
    assert set(records['requirements']['cases']) == {
        gates.case_label(row)
        for row in cases
    }
    assert all(row['score'] is None and row['baseline'] is None
               for row in records['reference']['cases'])
    assert all(v is None for v in records['requirements']['cases'].values())
    gates.validate_policy(records['policy'], cases)
    assert all(record['status'] == 'PENDING_MEASUREMENT'
               for record in records.values())
    with pytest.raises(ValueError, match='approved'):
        shared.runtime_records(path, value, mode='candidate')


def test_weekly_entry_preserves_failed_admission_before_gpu_work(
        configuration, tmp_path):
    path, _ = configuration
    output = tmp_path / 'output'
    completed = subprocess.run([
        sys.executable, '-B',
        str(SCRIPTS / 'weekly_check.py'), 'run', '--configuration',
        str(path), '--scale', '10B', '--case', 'streaming-normal', '--mode',
        'reference', '--wheel',
        str(tmp_path / 'reference.whl'), '--input-root',
        str(tmp_path / 'data'), '--scratch',
        str(tmp_path / 'scratch'), '--output-dir',
        str(output), '--cgroup-root',
        str(tmp_path / 'cgroups'), '--time-budget-seconds', '60',
        '--reference-output-s3', 's3://test-bucket/proposals'
    ], capture_output=True, text=True)
    assert completed.returncode == 1, completed.stderr
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'FAIL'
    assert result['scale'] == '10B'
    assert result['variant'] == 'streaming'
    assert 'frozen' in result['error']
    assert 'runtime' not in result


def test_shared_entry_rejects_weakening_the_weekly_tier(configuration,
                                                        tmp_path):
    path, _ = configuration
    shared = script('nightly_check')
    output = tmp_path / 'output'
    code = shared.main([
        'combine', '--configuration',
        str(path), '--tier', 'nightly', '--mode', 'candidate', '--wheel',
        str(tmp_path / 'candidate.whl'), '--results',
        str(tmp_path / 'result.json'), '--output-dir',
        str(output)
    ])
    assert code == 1
    result = shared.read(output / 'comparison.json')
    assert any('tier' in error.lower() for error in result['errors'])


@pytest.mark.parametrize('task,baseline,reference,passing,failing', [
    ('binary', .5, .8, .725, .70),
    ('regression', 4., 3., 3.25, 3.30),
])
def test_weekly_reference_retention_is_enforced_in_both_directions(
        task, baseline, reference, passing, failing):
    gates = script('nightly_gates')
    row = dict(task=task, arm='normal', score=passing)
    previous = dict(baseline=baseline, score=reference)
    limits = dict(min_normal_gain=.01, max_control_drift=.01)
    assert gates.case_learning(row, previous, limits,
                               tier='weekly')['status'] == 'PASS'
    row['score'] = failing
    assert gates.case_learning(row, previous, limits,
                               tier='weekly')['status'] == 'FAIL'


def test_pending_values_are_not_approved_by_checksum_binding(configuration):
    path, _ = configuration
    shared = script('nightly_check')
    value = shared.load_configuration(path)
    records = shared.runtime_records(path, value, mode='reference')
    assert records['reference']['status'] == 'PENDING_MEASUREMENT'
    with pytest.raises(ValueError, match='approved'):
        shared.runtime_records(path, value, mode='candidate')


def test_shared_weekly_run_cannot_skip_reference_replay(configuration):
    _, value = configuration
    args = SimpleNamespace(tier='weekly', command='run', replay_reference=False)
    script('nightly_check').check_tier(value, args)
    assert args.replay_reference is True


@pytest.mark.parametrize('bad_format', [[], {}, None])
def test_bad_configuration_format_fails_validation(configuration, bad_format):
    _, value = configuration
    value['format'] = bad_format
    with pytest.raises(ValueError):
        script('nightly_job').configuration(json.dumps(value))


def test_weekly_rejects_writable_inputs_before_gpu_work(configuration,
                                                        tmp_path):
    path, value = configuration
    shared = script('nightly_check')
    value['scales']['10B'] = dict(
        input_manifest_sha256='1' * 64,
        inputs=dict(bucket='shared', region='us-west-2', key='inputs.tar',
                    sha256='2' * 64))
    shared.save(path, value)
    inputs = tmp_path / 'data'
    inputs.mkdir()
    output = tmp_path / 'output'
    completed = subprocess.run([
        sys.executable, '-B',
        str(SCRIPTS / 'weekly_check.py'), 'run', '--configuration',
        str(path), '--scale', '10B', '--case', 'binary-normal', '--mode',
        'reference', '--wheel',
        str(tmp_path / 'reference.whl'), '--input-root',
        str(inputs), '--scratch',
        str(tmp_path / 'scratch'), '--output-dir',
        str(output), '--cgroup-root',
        str(tmp_path / 'cgroups'), '--time-budget-seconds', '60',
        '--reference-output-s3', 's3://test-bucket/proposals'
    ], capture_output=True, text=True)
    assert completed.returncode == 1, completed.stderr
    result = shared.read(output / 'result.json')
    assert 'read-only' in result['error']
    assert 'runtime' not in result


def workload_bindings():
    return {
        'manifest':
        dict(bucket='shared', region='us-west-2', key='25B/manifest.json',
             sha256='1' * 64),
        'by_graph': {
            name:
            dict(bucket='shared', region='us-west-2', key=f'25B/{name}.tar',
                 sha256='2' * 64)
            for name in ('adtech', 'delivery', 'commerce', 'streaming',
                         'community', 'payments', 'fraud')
        }
    }


def test_weekly_workload_binding_selects_the_cases_graph(configuration):
    _, value = configuration
    value['scales']['25B'] = dict(inputs=workload_bindings(),
                                  input_manifest_sha256='1' * 64)
    job = script('nightly_job')
    parsed = job.configuration(json.dumps(value))
    assert job.selected_workload(parsed, 'binary-control') == 'adtech'
    assert job.selected_workload(parsed, 'streaming-normal') == 'streaming'
    with pytest.raises(ValueError, match='case'):
        job.selected_workload(parsed, None)
    with pytest.raises(ValueError, match='case'):
        job.selected_workload(parsed, 'missing-normal')
    del value['scales']['25B']['inputs']['by_graph']['fraud']
    with pytest.raises(ValueError):
        job.configuration(json.dumps(value))


def test_prepare_extracts_only_the_selected_workload(configuration, tmp_path):
    _, value = configuration
    shared = script('nightly_check')
    repository = tmp_path / 'repository'
    repository.mkdir()
    metadata = {}
    for name in ('schema.json', 'dependencies.json', 'query.json'):
        shared.save(repository / name, {})
        metadata[name] = shared.digest(repository / name)
    sources = {}
    for name in ('adtech', 'fraud'):
        payload = repository / f'{name}.parquet'
        payload.write_bytes(name.encode())
        archive_path = repository / f'{name}.tar'
        with tarfile.open(archive_path, 'w') as archive:
            archive.add(payload, arcname=f'{name}/source.parquet')
            for item in metadata:
                archive.add(repository / item, arcname=item)
        sources[name] = dict(path=archive_path.name,
                             sha256=shared.digest(archive_path))
    value['scales']['25B'] = dict(inputs=dict(by_graph=sources),
                                  input_manifest_sha256='1' * 64)
    manifest = repository / 'manifest.json'
    shared.save(
        manifest,
        dict(
            scale='25B', reference_dependencies='dependencies.json', files=dict(
                metadata, **{
                    'fraud/source.parquet':
                    shared.digest(repository / 'fraud.parquet')
                }), graphs={
                    'fraud':
                    dict(profile='fraud', schema='schema.json',
                         sources={'events': 'fraud/source.parquet'})
                }, cases=[
                    dict(task='binary', variant='fraud', arm='control',
                         graph='fraud', descriptor='query.json')
                ]))
    value['scales']['25B']['inputs']['manifest'] = dict(
        path=manifest.name, sha256=shared.digest(manifest))
    for name in ('reference', 'policy', 'requirements'):
        path = repository / (name + '.json')
        shared.save(path, dict(status='PENDING_MEASUREMENT'))
        value[name] = dict(path=path.name, sha256=shared.digest(path))
    work = tmp_path / 'work'
    work.mkdir()
    script('nightly_job').prepare(value, work, scale='25B',
                                  local_root=repository, case='fraud-control')
    assert (work / 'inputs/fraud/source.parquet').read_bytes() == b'fraud'
    assert not (work / 'inputs/adtech').exists()
    assert (work / 'inputs/manifest.json').read_bytes() == manifest.read_bytes()


def selected_manifest(shared, value, root, selected_graph):
    cases = [
        dict(row) for row in shared.expected_cases(value)
        if row['scale'] == '25B'
    ]
    files, graphs = {}, {}
    for case in cases:
        graph = shared.task_profile(value, case)['graph']
        case.update(
            graph=graph, descriptor=f'{graph}/descriptor.json',
            prediction_input=f'{graph}/population.parquet',
            native_expectations=dict(
                key_failures={name: 0
                              for name in shared.KEY_FAILURES}))
        if case['arm'] == 'control':
            case['control_rows'] = f'{graph}/control.parquet'
        if case['task'] in ('static_link', 'temporal_link'):
            case.update(
                catalog=dict(path=f'{graph}/catalog.parquet', column='id'))
        key = graph + ('-static' if case['task'] == 'static_link' else '')
        case['graph'] = key
        graphs[key] = dict(profile=graph, static=case['task'] == 'static_link',
                           schema=f'{graph}/schema.json',
                           sources=dict(events=f'{graph}/events'),
                           row_counts=dict(events=1))
        for suffix in ('descriptor.json', 'population.parquet',
                       'control.parquet', 'catalog.parquet', 'schema.json',
                       'events/part.parquet'):
            name = f'{graph}/{suffix}'
            files[name] = '0' * 64
            if graph == selected_graph:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'payload')
                files[name] = shared.digest(path)
    dependencies = root / 'dependencies.json'
    shared.save(dependencies, {})
    files[dependencies.name] = shared.digest(dependencies)
    manifest = dict(format='kumo-engine-nightly-inputs-v2', scale='25B',
                    files=files, graphs=graphs, cases=cases,
                    reference_dependencies=dependencies.name)
    shared.save(root / 'manifest.json', manifest)
    return manifest


@pytest.mark.parametrize('case,graph', [('binary-control', 'adtech'),
                                        ('streaming-control', 'streaming')])
def test_selected_inputs_keep_full_roster_and_verify_selected_payloads(
        configuration, tmp_path, case, graph):
    _, value = configuration
    shared = script('nightly_check')
    root = tmp_path / 'data'
    root.mkdir()
    manifest = selected_manifest(shared, value, root, graph)
    checksum = shared.digest(root / 'manifest.json')
    loaded = shared.load_inputs(root, checksum, '25B', value,
                                selected_case=case)
    assert len(loaded['cases']) == 24
    assert not (root / 'fraud').exists()
    (root / graph / 'control.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        shared.load_inputs(root, checksum, '25B', value, selected_case=case)
    (root / graph / 'control.parquet').write_bytes(b'payload')
    manifest['cases'].pop()
    shared.save(root / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='every configured case'):
        shared.load_inputs(root, shared.digest(root / 'manifest.json'), '25B',
                           value, selected_case=case)


def test_selected_inputs_verify_bootstrap_normal_payload(
        configuration, tmp_path):
    _, value = configuration
    shared = script('nightly_check')
    root = tmp_path / 'data'
    root.mkdir()
    manifest = selected_manifest(shared, value, root, 'adtech')
    path = root / 'adtech/normal.parquet'
    path.write_bytes(b'normal payload')
    manifest['files']['adtech/normal.parquet'] = shared.digest(path)
    for row in manifest['cases']:
        if row['graph'] == 'adtech':
            row['normal_rows'] = ['adtech/normal.parquet']
    shared.save(root / 'manifest.json', manifest)
    checksum = shared.digest(root / 'manifest.json')
    shared.load_inputs(root, checksum, '25B', value,
                       selected_case='binary-normal')
    path.write_bytes(b'changed normal payload')
    with pytest.raises(ValueError, match='checksum'):
        shared.load_inputs(root, checksum, '25B', value,
                           selected_case='binary-normal')
