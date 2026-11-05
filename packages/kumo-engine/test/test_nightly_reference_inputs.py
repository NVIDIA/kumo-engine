# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admit pinned source inputs without reading saved reference answers."""

import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_check.py'
SPEC = importlib.util.spec_from_file_location('reference_inputs', SOURCE)
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)


@pytest.fixture
def source_inputs(tmp_path):
    configuration = dict(scales={'100M': {}})
    root = tmp_path / 'inputs'
    root.mkdir()
    nightly.save(root / 'schema.json', dict(adtech={}, delivery={},
                                            commerce={}))
    nightly.save(root / 'dependencies.json', {})
    cases = nightly.expected_cases(configuration)
    graphs = {}
    for case in cases:
        profile = nightly.task_profile(configuration, case)
        family = profile['graph']
        graph = family + ('-static' if case['task'] == 'static_link' else '')
        raw = f'raw/{family}/part.parquet'
        path = root / raw
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'Pinned source table')
        graphs[graph] = dict(profile=family,
                             static=case['task'] == 'static_link',
                             schema='schema.json',
                             sources={'table': f'raw/{family}'})
        name = case['task'] + '/' + case['arm']
        case.update(
            graph=graph, descriptor=f'cases/{name}/query.json',
            prediction_input=f'cases/{name}/full.parquet',
            native_expectations=dict(
                key_failures=dict.fromkeys(nightly.KEY_FAILURES, 0)))
        if case['arm'] == 'normal':
            nightly.save(root / case['descriptor'],
                         dict(query=profile['query'], query_generation_plan={}))
            case['normal_rows'] = [f'cases/{name}/answers.parquet']
        else:
            case['control_rows'] = [f'cases/{name}/answers.parquet']
            case['control_recipe'] = dict(
                method=('entity-trajectory-permutation-v1' if case['task']
                        == 'static_link' else 'stage-permutation-v1'), seed=0)
        if case['task'].endswith('link'):
            catalog = f'catalogs/{family}.parquet'
            path = root / catalog
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'Pinned catalog')
            case['catalog'] = dict(path=catalog, column='ID')
    files = {
        path.relative_to(root).as_posix(): nightly.digest(path)
        for path in root.rglob('*') if path.is_file()
    }
    for case in cases:
        for key in ('descriptor', 'prediction_input'):
            files.setdefault(case[key], 'a' * 64)
        for key in ('normal_rows', 'control_rows'):
            for name in case.get(key, []):
                files[name] = 'a' * 64
        staging = str(Path(case['descriptor']).parent / 'staging/tt.parquet')
        files[staging] = 'a' * 64
    manifest = dict(format='kumo-engine-nightly-inputs-v2', scale='100M',
                    files=files, cases=cases, graphs=graphs,
                    reference_dependencies='dependencies.json')
    nightly.save(root / 'manifest.json', manifest)
    return root, configuration, manifest


def admit(inputs, **options):
    root, configuration, _ = inputs
    return nightly.load_inputs(root, nightly.digest(root / 'manifest.json'),
                               '100M', configuration, **options)


@pytest.mark.parametrize('case', [None, 'binary-normal', 'binary-control'])
def test_reference_admits_sources_without_saved_answer_or_prediction_bytes(
        source_inputs, case):
    loaded = admit(source_inputs, selected_case=case, reference_only=True)
    assert len(loaded['cases']) == 16


def test_reference_admits_a_manifest_without_saved_answer_declarations(
        source_inputs):
    root, _, manifest = source_inputs
    for case in manifest['cases']:
        case.pop('prediction_input')
        case.pop('normal_rows', None)
        case.pop('control_rows', None)
        if case['arm'] == 'control':
            case['descriptor'] = case['descriptor'].replace(
                '/control/', '/normal/')
    manifest['files'] = {
        name: checksum
        for name, checksum in manifest['files'].items()
        if (root / name).is_file()
    }
    nightly.save(root / 'manifest.json', manifest)
    assert len(admit(source_inputs, reference_only=True)['cases']) == 16


@pytest.mark.parametrize('damage', ['missing', 'corrupt', 'extra'])
def test_reference_still_rejects_damaged_source_tables(source_inputs, damage):
    root, _, _ = source_inputs
    path = root / 'raw/adtech/part.parquet'
    if damage == 'missing':
        path.unlink()
    elif damage == 'corrupt':
        path.write_bytes(b'Changed source table')
    else:
        path.with_name('extra.parquet').write_bytes(b'Unpinned source table')
    with pytest.raises(ValueError):
        admit(source_inputs, selected_case='binary-normal', reference_only=True)


def test_reference_selected_family_does_not_require_other_source_bytes(
        source_inputs):
    root, _, _ = source_inputs
    (root / 'raw/delivery/part.parquet').unlink()
    assert len(
        admit(source_inputs, selected_case='binary-normal',
              reference_only=True)['cases']) == 16


@pytest.mark.parametrize(
    'recipe', [None, dict(method='stage-permutation-v1', seed=1)])
def test_reference_rejects_missing_or_invalid_pinned_control_recipe(
        source_inputs, recipe):
    root, _, manifest = source_inputs
    control = next(case for case in manifest['cases']
                   if case['task'] == 'binary' and case['arm'] == 'control')
    if recipe is None:
        control.pop('control_recipe')
    else:
        control['control_recipe'] = recipe
    nightly.save(root / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='control recipe'):
        admit(source_inputs, selected_case='binary-normal', reference_only=True)


def test_candidate_still_requires_declared_saved_inputs(source_inputs):
    with pytest.raises(ValueError, match='missing|checksum'):
        admit(source_inputs, selected_case='binary-normal')


def test_reference_still_requires_the_normal_query_descriptor(source_inputs):
    root, _, _ = source_inputs
    (root / 'cases/binary/normal/query.json').unlink()
    with pytest.raises(ValueError, match='missing|checksum'):
        admit(source_inputs, selected_case='binary-control',
              reference_only=True)


def test_generated_prediction_hash_replaces_saved_population_binding():
    case = dict(scale='100M', task='binary', arm='normal', seed=0,
                descriptor='query.json')
    manifest = dict(files={'query.json': 'a' * 64})
    assert nightly.input_signature(manifest, case, prediction='c' * 64) == dict(
        descriptor='a' * 64, prediction='c' * 64, scale='100M', task='binary',
        arm='normal', seed=0)
    case['prediction_input'] = 'full.parquet'
    manifest['files']['full.parquet'] = 'b' * 64
    assert nightly.input_signature(manifest, case)['prediction'] == 'b' * 64


def test_runner_identity_binds_the_control_generation_helper(
        tmp_path, monkeypatch):
    for name in ('nightly_check.py', 'nightly_case.py', 'nightly_gates.py',
                 'nightly_predictions.py', 'nightly_resources.py',
                 'nightly_job.py', 'nightly_build.py', 'nightly_controls.py',
                 'nightly_sources.py', 'nightly_evidence.py', 'nightly_host.py',
                 'wheel_gpu_smoke.py'):
        (tmp_path / name).write_text(name)
    monkeypatch.setattr(nightly, '__file__', str(tmp_path / 'nightly_check.py'))
    monkeypatch.setattr(nightly, 'HELPER', tmp_path / 'wheel_gpu_smoke.py')
    before = nightly.runner_identity()
    (tmp_path / 'nightly_controls.py').write_text('Changed control generation')
    assert nightly.runner_identity() != before


@pytest.mark.parametrize('case_name', ['binary-normal', 'binary-control'])
@pytest.mark.parametrize('recipe_state', ['valid', 'missing', 'bad_seed'])
def test_reference_worker_admits_sources_inside_resource_observation(
        source_inputs, tmp_path, monkeypatch, case_name, recipe_state):
    from contextlib import contextmanager
    from types import SimpleNamespace

    root, configuration, manifest = source_inputs
    control = next(case for case in manifest['cases']
                   if case['task'] == 'binary' and case['arm'] == 'control')
    if recipe_state == 'missing':
        control.pop('control_recipe')
    elif recipe_state == 'bad_seed':
        control['control_recipe']['seed'] = 1
    nightly.save(root / 'manifest.json', manifest)
    (root / 'raw/delivery/part.parquet').unlink()
    configuration['scales']['100M'].update(
        inputs={}, input_manifest_sha256=nightly.digest(root / 'manifest.json'))
    configuration['reference_identity'] = dict(version='1.0.0',
                                               wheel_sha256='a' * 64)
    wheel = tmp_path / 'reference.whl'
    wheel.write_bytes(b'reference wheel')
    output = tmp_path / 'output'
    output.mkdir()
    args = SimpleNamespace(scale='100M', case=case_name, mode='reference',
                           input_root=root, output_dir=output,
                           scratch=tmp_path / 'scratch', wheel=wheel,
                           build_receipt=None, time_budget_seconds=60)
    initial = dict(scale='100M', task='binary', arm=case_name.rsplit('-', 1)[1],
                   seed=0, status='RUNNING', mode='reference', bindings={})
    resources = nightly.sibling('nightly_resources')
    observed = dict(errors=[], status='MEASURED', peaks={})
    active = []

    @contextmanager
    def track(*arguments):
        active.append(True)
        try:
            yield observed
        finally:
            active.pop()

    def stop_at_snapshot(*arguments, **options):
        assert active == [True]
        raise ValueError('Reached fresh graph inside resource observation')

    monkeypatch.setattr(resources, 'join_cgroup', lambda scope: None)
    monkeypatch.setattr(resources, 'track_resources', track)
    monkeypatch.setattr(resources, 'available_capacity', lambda *arguments: {})
    monkeypatch.setattr(
        nightly, 'verify_runtime', lambda *arguments, **options: dict(
            version='1.0.0', wheel_sha256=nightly.digest(wheel)))
    monkeypatch.setattr(nightly, 'verify_installed_files', lambda *arguments: 1)
    monkeypatch.setattr(nightly, 'check_install',
                        lambda *arguments, **options: {})
    monkeypatch.setattr(nightly, 'runner_identity', lambda: 'b' * 64)
    monkeypatch.setattr(nightly, 'build_graph', stop_at_snapshot)
    result = nightly._worker(args, configuration, {}, initial,
                             tmp_path / 'cgroup')
    if recipe_state == 'valid':
        assert result['error'] == (
            'ValueError: Reached fresh graph inside resource observation')
    else:
        assert 'control recipe' in result['error']
    assert result['resources']['errors'] == []
