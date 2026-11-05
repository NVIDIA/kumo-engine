# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the shared nightly admission and reference bootstrap contracts."""
import hashlib
import importlib.util
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_check.py'
SPEC = importlib.util.spec_from_file_location('nightly_configuration', SOURCE)
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)


def configuration(tmp_path):
    for name in ('reference', 'policy', 'requirements'):
        nightly.save(tmp_path / (name + '.json'),
                     dict(status='PENDING_MEASUREMENT', cases=[]))
    value = dict(
        format='kumo-engine-nightly-job-v2', min_free_bytes=1,
        reference_identity=dict(version='1.0.0', wheel_sha256='a' * 64),
        scales={
            scale:
            dict(
                input_manifest_sha256='b' * 64,
                inputs=dict(bucket='shared', region='us-west-2',
                            key=scale + '/inputs.tar', sha256='c' * 64))
            for scale in ('100M', '1B')
        }, archive=None, **{
            name:
            dict(path=name + '.json',
                 sha256=nightly.digest(tmp_path / (name + '.json')))
            for name in ('reference', 'policy', 'requirements')
        })
    path = tmp_path / 'inputs.json'
    nightly.save(path, value)
    return path, value


def variant_configuration(tmp_path):
    path, value = configuration(tmp_path)
    value['variants'] = {
        'streaming':
        dict(task='temporal_link', graph='streaming', metric='recall@50',
             query='streaming query', top_k=50, raw_limit=51),
        'community':
        dict(task='regression', graph='community', metric='mae',
             query='community query'),
        'payments':
        dict(task='binary', graph='payments', metric='auroc',
             query='payments query', min_hops=4, max_hops=6),
        'fraud':
        dict(task='binary', graph='fraud', metric='auroc', query='fraud query',
             min_hops=4, max_hops=6)
    }
    nightly.save(path, value)
    return path, value


def test_all_seven_families_have_both_scales_and_both_answer_conditions(
        tmp_path):
    path, _ = variant_configuration(tmp_path)
    cases = nightly.expected_cases(nightly.load_configuration(path))
    assert len(cases) == 48
    assert len(
        {nightly.sibling('nightly_gates').case_key(row)
         for row in cases}) == 48
    assert sum('variant' not in row for row in cases) == 32
    for name in ('streaming', 'community', 'payments', 'fraud'):
        assert {(row['scale'], row['arm'])
                for row in cases if row.get('variant') == name
                } == {(scale, arm)
                      for scale in ('100M', '1B')
                      for arm in ('normal', 'control')}
    binary = [
        row for row in cases if row['task'] == 'binary'
        and row['scale'] == '100M' and row['arm'] == 'normal'
    ]
    assert len(binary) == 3


def test_variant_recipe_identity_binds_query_scoring_and_depth(tmp_path):
    _, value = variant_configuration(tmp_path)
    before = nightly.recipe_identity(value)
    value['variants']['streaming']['query'] += ' changed'
    assert nightly.recipe_identity(value) != before


def test_variant_selection_is_resolved_before_pending_inputs_fail(tmp_path):
    path, value = variant_configuration(tmp_path)
    value['scales']['1B'].update(inputs=None, input_manifest_sha256=None,
                                 pending='Prepare all seven families')
    nightly.save(path, value)
    output = tmp_path / 'variant-output'
    assert nightly.main([
        'run', '--configuration',
        str(path), '--scale', '1B', '--case', 'streaming-control', '--mode',
        'reference', '--input-root',
        str(tmp_path / 'inputs'), '--wheel',
        str(tmp_path / 'wheel.whl'), '--scratch',
        str(tmp_path / 'scratch'), '--output-dir',
        str(output), '--cgroup-root', '/sys/fs/cgroup', '--time-budget-seconds',
        '100', '--reference-output-s3', 's3://test-bucket/proposals'
    ]) == 1
    result = nightly.read(output / 'result.json')
    assert result['variant'] == 'streaming'
    assert result['task'] == 'temporal_link'
    assert 'frozen' in result['error']


def test_variant_plan_fields_follow_the_bound_definition(tmp_path):
    _, value = variant_configuration(tmp_path)
    case = dict(scale='100M', task='temporal_link', arm='normal', seed=0,
                variant='streaming')
    selected = nightly.bind_case_definition(case, value,
                                            dict(query='streaming query'))
    assert selected['top_k'] == 50
    assert selected['raw_limit'] == 51
    assert selected['metric'] == 'recall@50'
    with pytest.raises(ValueError, match='query'):
        nightly.bind_case_definition(case, value, dict(query='commerce query'))
    with pytest.raises(ValueError, match='semantic|task'):
        nightly.bind_case_definition(dict(case, task='binary'), value,
                                     dict(query='streaming query'))


def test_frozen_scale_manifest_requires_all_twenty_four_declared_cases(
        tmp_path):
    _, config = variant_configuration(tmp_path)
    data = tmp_path / 'frozen'
    data.mkdir()
    for name in ('schema.json', 'dependencies.json'):
        nightly.save(data / name, {})
    (data / 'source.parquet').write_bytes(b'Frozen data fixture')
    files = {path.name: nightly.digest(path) for path in data.iterdir()}
    cases = [
        dict(case) for case in nightly.expected_cases(config)
        if case['scale'] == '100M'
    ]
    graphs = {}
    for case in cases:
        case['graph'] = case.get('variant', case['task'])
        case['native_expectations'] = dict(
            key_failures=dict.fromkeys(nightly.KEY_FAILURES, 0))
        graphs[case['graph']] = dict(
            profile=nightly.task_profile(config, case)['graph'],
            static=case['task'] == 'static_link', schema='schema.json',
            sources={'table': 'source.parquet'})
    value = dict(format='kumo-engine-nightly-inputs-v2', scale='100M',
                 files=files, cases=cases, graphs=graphs,
                 reference_dependencies='dependencies.json')
    path = data / 'manifest.json'
    nightly.save(path, value)
    assert len(
        nightly.load_inputs(data, nightly.digest(path), '100M',
                            config)['cases']) == 24
    value['cases'].pop()
    nightly.save(path, value)
    with pytest.raises(ValueError, match='every configured case'):
        nightly.load_inputs(data, nightly.digest(path), '100M', config)


def frozen_family_payload(tmp_path, family='adtech'):
    _, config = variant_configuration(tmp_path)
    data = tmp_path / 'frozen'
    data.mkdir()
    payloads = {
        'graph-schema.json': b'{}',
        'reference-dependencies.json': b'{}',
    }
    cases = [
        dict(case) for case in nightly.expected_cases(config)
        if case['scale'] == '100M'
    ]
    graphs = {}
    for case in cases:
        profile = nightly.task_profile(config, case)['graph']
        graph = profile + ('-static' if case['task'] == 'static_link' else '')
        source = f'raw/{profile}/100M/table/part.parquet'
        payloads[source] = profile.encode()
        payloads[f'source-manifests/{profile}.json'] = b'{}'
        graphs[graph] = dict(profile=profile,
                             static=case['task'] == 'static_link',
                             schema='graph-schema.json',
                             sources={'table': str(Path(source).parent)})
        name = case.get('variant', case['task'])
        descriptor = f'cases/{name}/query.json'
        population = f'cases/{name}/population.parquet'
        normal = f'cases/{name}/normal/part.parquet'
        control = f'cases/{name}/control/part.parquet'
        probe = f'cases/{name}/bounded-import-probe.json'
        for path in (descriptor, population, normal, control, probe):
            payloads[path] = path.encode()
        case.update(
            graph=graph, descriptor=descriptor, prediction_input=population,
            normal_rows=[normal], preparation_probe=probe,
            native_expectations=dict(
                key_failures=dict.fromkeys(nightly.KEY_FAILURES, 0)))
        if case['arm'] == 'control':
            case['control_rows'] = [control]
    value = dict(
        format='kumo-engine-nightly-inputs-v2', scale='100M', files={
            name: hashlib.sha256(content).hexdigest()
            for name, content in payloads.items()
        }, cases=cases, graphs=graphs,
        reference_dependencies='reference-dependencies.json')
    workloads = {
        case.get('variant', case['task'])
        for case in cases if graphs[case['graph']]['profile'] == family
    }
    for name, content in payloads.items():
        if (name in ('graph-schema.json', 'reference-dependencies.json')
                or name.startswith(f'raw/{family}/')
                or name == f'source-manifests/{family}.json' or any(
                    name.startswith(f'cases/{workload}/')
                    for workload in workloads)):
            path = data / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    path = data / 'manifest.json'
    nightly.save(path, value)
    return data, config, value, path


@pytest.mark.parametrize('case,family', [('binary-normal', 'adtech'),
                                         ('binary-control', 'adtech'),
                                         ('payments-control', 'payments'),
                                         ('forecasting-control', 'delivery'),
                                         ('static_link-normal', 'commerce')])
def test_selected_family_payload_retains_the_complete_parent_manifest(
        tmp_path, case, family):
    data, config, manifest, path = frozen_family_payload(tmp_path, family)
    checksum = nightly.digest(path)
    loaded = nightly.load_inputs(data, checksum, '100M', config,
                                 selected_case=case)
    assert loaded == manifest
    assert len(loaded['cases']) == 24
    assert nightly.digest(path) == checksum
    assert not (data / 'raw/fraud').exists()


def test_parent_header_can_be_validated_before_payload_admission(tmp_path):
    data, config, manifest, path = frozen_family_payload(tmp_path)
    for payload in data.rglob('*'):
        if payload.is_file() and payload != path:
            payload.unlink()
    assert nightly.load_input_manifest(data, nightly.digest(path), '100M',
                                       config) == manifest


def test_selected_family_requires_its_control_answers_even_for_normal(tmp_path):
    data, config, _, path = frozen_family_payload(tmp_path)
    (data / 'cases/binary/control/part.parquet').unlink()
    with pytest.raises(ValueError):
        nightly.load_inputs(data, nightly.digest(path), '100M', config,
                            selected_case='binary-normal')


def test_full_payload_validation_still_requires_other_families(tmp_path):
    data, config, _, path = frozen_family_payload(tmp_path)
    with pytest.raises(ValueError, match='missing|checksum'):
        nightly.load_inputs(data, nightly.digest(path), '100M', config)


@pytest.mark.parametrize('damage', ['missing', 'corrupt', 'extra', 'symlink'])
def test_selected_family_rejects_bad_graph_payloads(tmp_path, damage):
    data, config, _, path = frozen_family_payload(tmp_path)
    source = data / 'raw/adtech/100M/table/part.parquet'
    if damage == 'missing':
        source.unlink()
    elif damage == 'corrupt':
        source.write_bytes(b'Changed frozen data')
    elif damage == 'extra':
        source.with_name('extra.parquet').write_bytes(b'Unlisted data')
    else:
        source.unlink()
        source.symlink_to(data / 'reference-dependencies.json')
    with pytest.raises(ValueError):
        nightly.load_inputs(data, nightly.digest(path), '100M', config,
                            selected_case='binary-normal')


@pytest.mark.parametrize('damage',
                         ['roster', 'profile', 'native', 'hash', 'path'])
def test_selected_family_still_validates_absent_family_metadata(
        tmp_path, damage):
    data, config, manifest, path = frozen_family_payload(tmp_path)
    fraud = next(case for case in manifest['cases']
                 if case.get('variant') == 'fraud')
    if damage == 'roster':
        manifest['cases'].remove(fraud)
    elif damage == 'profile':
        manifest['graphs']['fraud']['profile'] = 'community'
    elif damage == 'native':
        fraud['native_expectations']['key_failures']['key_lookup_failures'] = 1
    elif damage == 'hash':
        manifest['files']['raw/fraud/100M/table/part.parquet'] = 'invalid'
    else:
        manifest['files']['../outside.parquet'] = 'a' * 64
    nightly.save(path, manifest)
    with pytest.raises(ValueError):
        nightly.load_inputs(data, nightly.digest(path), '100M', config,
                            selected_case='binary-normal')


def test_selected_family_rejects_an_unknown_case(tmp_path):
    data, config, _, path = frozen_family_payload(tmp_path)
    with pytest.raises(ValueError):
        nightly.load_inputs(data, nightly.digest(path), '100M', config,
                            selected_case='unknown-normal')


def test_configuration_declares_two_scales_and_thirty_two_unique_cases(
        tmp_path):
    path, _ = configuration(tmp_path)
    config = nightly.load_configuration(path)
    cases = nightly.expected_cases(config)
    assert len(cases) == 32
    assert len({(c['scale'], c['task'], c['arm'], c['seed'])
                for c in cases}) == 32
    assert sum(c['scale'] == '100M' for c in cases) == 16
    assert sum(c['scale'] == '1B' for c in cases) == 16


def test_reference_bootstrap_does_not_require_approved_reference(tmp_path):
    path, _ = configuration(tmp_path)
    config = nightly.load_configuration(path)
    data = nightly.runtime_records(path, config, mode='reference')
    assert data['reference']['status'] == 'PENDING_MEASUREMENT'
    with pytest.raises(ValueError, match='approved'):
        nightly.runtime_records(path, config, mode='candidate')


def test_compact_record_hash_is_checked_even_during_measurement(tmp_path):
    path, _ = configuration(tmp_path)
    (tmp_path / 'policy.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        nightly.runtime_records(path, nightly.load_configuration(path),
                                mode='reference')


def test_unfrozen_one_billion_inputs_fail_before_model_work(tmp_path):
    path, value = configuration(tmp_path)
    value['scales']['1B']['input_manifest_sha256'] = None
    value['scales']['1B']['inputs'] = None
    value['scales']['1B']['pending'] = 'Freeze the one-billion-row inputs'
    nightly.save(path, value)
    config = nightly.load_configuration(path)
    with pytest.raises(ValueError, match='frozen'):
        nightly.scale_configuration(config, '1B')


def test_configured_reference_identity_is_distinct_from_candidate_receipt():
    runtime = dict(version='1.0.0', wheel_sha256='a' * 64, installed_files=10)
    approved = dict(version='1.0.0', wheel_sha256='a' * 64)
    assert nightly.check_install(None, runtime, reference_only=True,
                                 reference_identity=approved) == dict(
                                     format='reference-wheel', **approved)
    with pytest.raises(ValueError, match='receipt'):
        nightly.check_install(None, runtime, reference_only=False)
    with pytest.raises(ValueError):
        nightly.check_install(None, dict(runtime, wheel_sha256='b' * 64),
                              reference_only=True, reference_identity=approved)


def test_reference_preserves_verified_clean_build_receipt(tmp_path):
    identity = dict(version='1.0.0+aaaaaaa', wheel_sha256='c' * 64)
    receipt = dict(identity, revision='a' * 40, tree='b' * 40, harness='d' * 40)
    path = tmp_path / 'build.json'
    nightly.save(path, receipt)
    assert nightly.check_install(None, dict(identity, installed_files=10),
                                 reference_only=True,
                                 reference_identity=identity,
                                 build_receipt=path) == receipt


@pytest.mark.parametrize('field,value', [('version', '1.0.0+ddddddd'),
                                         ('wheel_sha256', 'd' * 64),
                                         ('revision', 'not-a-commit'),
                                         ('tree', 'not-a-tree')])
def test_reference_rejects_mismatched_or_invalid_build_receipt(
        tmp_path, field, value):
    identity = dict(version='1.0.0+aaaaaaa', wheel_sha256='c' * 64)
    receipt = dict(identity, revision='a' * 40, tree='b' * 40, harness='d' * 40)
    receipt[field] = value
    path = tmp_path / 'build.json'
    nightly.save(path, receipt)
    with pytest.raises(ValueError, match='identity differs'):
        nightly.check_install(None, dict(identity, installed_files=10),
                              reference_only=True, reference_identity=identity,
                              build_receipt=path)


def test_single_case_admission_failure_is_saved(tmp_path):
    path, value = configuration(tmp_path)
    value['scales']['1B'].update(inputs=None, input_manifest_sha256=None,
                                 pending='Freeze inputs before execution')
    nightly.save(path, value)
    output = tmp_path / 'output'
    code = nightly.main([
        'run', '--configuration',
        str(path), '--scale', '1B', '--case', 'binary-normal', '--mode',
        'reference', '--input-root',
        str(tmp_path / 'inputs'), '--wheel',
        str(tmp_path / 'wheel.whl'), '--scratch',
        str(tmp_path / 'scratch'), '--output-dir',
        str(output), '--cgroup-root', '/sys/fs/cgroup', '--time-budget-seconds',
        '100', '--reference-output-s3', 's3://test-bucket/proposals'
    ])
    assert code == 1
    result = nightly.read(output / 'result.json')
    assert result['status'] == 'FAIL'
    assert result['scale'] == '1B'
    assert 'frozen' in result['error']


def test_reference_case_requires_explicit_temporary_budget(tmp_path):
    path, _ = configuration(tmp_path)
    with pytest.raises(SystemExit):
        nightly.main([
            'run', '--configuration',
            str(path), '--scale', '100M', '--case', 'binary-normal', '--mode',
            'reference', '--input-root',
            str(tmp_path), '--wheel',
            str(tmp_path / 'wheel.whl'), '--scratch',
            str(tmp_path / 'scratch'), '--output-dir',
            str(tmp_path / 'output'), '--cgroup-root', '/sys/fs/cgroup'
        ])


def approved_records(tmp_path):
    path, config = configuration(tmp_path)
    for name in ('reference', 'policy', 'requirements'):
        record = nightly.read(tmp_path / (name + '.json'))
        record['status'] = 'APPROVED'
        if name == 'reference':
            record['reference_identity'] = dict(config['reference_identity'])
            record['input_manifest_sha256'] = {
                scale: item['input_manifest_sha256']
                for scale, item in config['scales'].items()
            }
            record['recipe_sha256'] = nightly.signature(
                nightly.sibling('nightly_case').CAPS)
        nightly.save(tmp_path / (name + '.json'), record)
        config[name]['sha256'] = nightly.digest(tmp_path / (name + '.json'))
    nightly.save(path, config)
    return path, config


def test_approved_reference_matches_configured_wheel_and_input_identities(
        tmp_path):
    path, config = approved_records(tmp_path)
    records = nightly.runtime_records(path, config, mode='candidate')
    assert records['reference']['reference_identity'] == config[
        'reference_identity']


@pytest.mark.parametrize('damage', [
    lambda record: record.pop('reference_identity'),
    lambda record: record['reference_identity'].update(version='1.0.1'),
    lambda record: record['reference_identity'].update(wheel_sha256='f' * 64),
    lambda record: record.pop('input_manifest_sha256'),
    lambda record: record['input_manifest_sha256'].pop('1B'),
    lambda record: record['input_manifest_sha256'].update({'1B': 'f' * 64}),
    lambda record: record.update(recipe_sha256='f' * 64),
])
def test_candidate_rejects_stale_reference_provenance_even_with_updated_hash(
        tmp_path, damage):
    path, config = approved_records(tmp_path)
    reference = nightly.read(tmp_path / 'reference.json')
    damage(reference)
    nightly.save(tmp_path / 'reference.json', reference)
    config['reference']['sha256'] = nightly.digest(tmp_path / 'reference.json')
    nightly.save(path, config)
    with pytest.raises(ValueError,
                       match='identity|manifest|reference|Reference'):
        nightly.runtime_records(path, config, mode='candidate')


def archived_records(tmp_path):
    path, config = configuration(tmp_path)
    archive_root = tmp_path / 'archive'
    archive_root.mkdir()
    manifest = dict(
        format='kumo-engine-nightly-archive-v1',
        reference_identity=dict(config['reference_identity']),
        input_manifest_sha256={
            scale: item['input_manifest_sha256']
            for scale, item in config['scales'].items()
        },
        recipe_sha256=nightly.signature(nightly.sibling('nightly_case').CAPS),
        files={}, cases=[])
    descriptor = tmp_path / 'archive-manifest.json'
    nightly.save(descriptor, manifest)
    config['archive'] = dict(path=descriptor.name,
                             sha256=nightly.digest(descriptor))
    nightly.save(path, config)
    return path, config, archive_root, manifest


def test_archive_is_optional_until_a_case_requests_archived_proof(tmp_path):
    path, config = configuration(tmp_path)
    assert nightly.load_archive(config, path, None) is None


def test_archive_descriptor_is_bound_to_inputs_reference_and_recipe(tmp_path):
    path, config, root, manifest = archived_records(tmp_path)
    assert nightly.load_archive(config, path, root) == manifest


@pytest.mark.parametrize('damage', [
    lambda manifest: manifest.update(format='old-format'),
    lambda manifest: manifest['reference_identity'].update(version='9.0.0'),
    lambda manifest: manifest['reference_identity'].update(wheel_sha256='f' * 64
                                                           ),
    lambda manifest: manifest['input_manifest_sha256'].update(
        {'100M': 'f' * 64}),
    lambda manifest: manifest['input_manifest_sha256'].pop('1B'),
    lambda manifest: manifest.update(recipe_sha256='f' * 64),
])
def test_archive_rejects_stale_manifest_bindings_after_descriptor_hash_update(
        tmp_path, damage):
    path, config, root, manifest = archived_records(tmp_path)
    damage(manifest)
    nightly.save(tmp_path / config['archive']['path'], manifest)
    config['archive']['sha256'] = nightly.digest(tmp_path /
                                                 config['archive']['path'])
    with pytest.raises(ValueError):
        nightly.load_archive(config, path, root)


def test_archive_checks_descriptor_bytes_and_rejects_missing_payload_root(
        tmp_path):
    path, config, root, _ = archived_records(tmp_path)
    with pytest.raises(ValueError, match='root|directory|archive|Archive'):
        nightly.load_archive(config, path, None)
    (tmp_path / config['archive']['path']).write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        nightly.load_archive(config, path, root)


def test_archive_rejects_descriptor_escape_and_symlink(tmp_path):
    path, config, root, _ = archived_records(tmp_path)
    escaped = dict(config, archive=dict(config['archive'],
                                        path='../manifest.json'))
    with pytest.raises(ValueError):
        nightly.load_archive(escaped, path, root)
    link = tmp_path / 'link.json'
    link.symlink_to(tmp_path / config['archive']['path'])
    linked = dict(config, archive=dict(config['archive'], path=link.name))
    with pytest.raises(ValueError):
        nightly.load_archive(linked, path, root)


def replay_case(tmp_path):
    _, _, root, archive = archived_records(tmp_path)
    case = dict(scale='100M', task='binary', arm='normal', seed=0)
    paths = ['model/metadata.json', 'population.json', 'predictions.json']
    for name in paths:
        nightly.save(root / name, {})
    archive['files'] = {name: nightly.digest(root / name) for name in paths}
    replay = dict(
        model=dict(path='model',
                   files={'metadata.json': archive['files'][paths[0]]}),
        checkpoint_sha256='a' * 64,
        population=dict(path=paths[1], sha256=archive['files'][paths[1]]),
        predictions=dict(path=paths[2], sha256=archive['files'][paths[2]]))
    archive['cases'] = [
        dict(case, training_table=dict(stages={}), replay=replay)
    ]
    return root, archive, case, dict(case, score=.8, baseline=.5)


def test_archived_case_combines_scores_with_verified_replay_payload(tmp_path):
    root, archive, case, reference = replay_case(tmp_path)
    result = nightly.archived_case(archive, case, reference, root, replay=True)
    assert result['score'] == reference['score']
    assert result['baseline'] == reference['baseline']
    assert result['replay'] == archive['cases'][0]['replay']


@pytest.mark.parametrize('model_path', ['model', '../outside', '/tmp/outside'])
def test_archived_model_needs_nonempty_verified_file_membership(
        tmp_path, model_path):
    root, archive, case, reference = replay_case(tmp_path)
    archive['cases'][0]['replay']['model'].update(path=model_path, files={})
    with pytest.raises(ValueError):
        nightly.archived_case(archive, case, reference, root, replay=True)


@pytest.mark.parametrize('linked', [False, True])
def test_archived_model_rejects_extra_and_linked_unlisted_files(
        tmp_path, linked):
    root, archive, case, reference = replay_case(tmp_path)
    extra = root / 'model/extra.json'
    if linked:
        extra.symlink_to(root / 'population.json')
    else:
        nightly.save(extra, {})
    with pytest.raises(ValueError):
        nightly.archived_case(archive, case, reference, root, replay=True)


@pytest.mark.parametrize('relationship',
                         ['output-under-input', 'input-under-output'])
def test_case_evidence_and_frozen_inputs_cannot_contain_each_other(
        tmp_path, relationship):
    path, _ = configuration(tmp_path)
    if relationship == 'output-under-input':
        source, output = tmp_path / 'inputs', tmp_path / 'inputs/evidence'
    else:
        source, output = tmp_path / 'evidence/inputs', tmp_path / 'evidence'
    code = nightly.main([
        'run', '--configuration',
        str(path), '--scale', '100M', '--case', 'binary-normal', '--mode',
        'reference', '--input-root',
        str(source), '--wheel',
        str(tmp_path / 'missing.whl'), '--scratch',
        str(tmp_path / 'scratch'), '--output-dir',
        str(output), '--cgroup-root', '/sys/fs/cgroup', '--time-budget-seconds',
        '100', '--reference-output-s3', 's3://test-bucket/proposals'
    ])
    assert code == 1
    result = nightly.read(output / 'result.json')
    assert result['status'] == 'FAIL'
    assert 'input' in result['error'].lower()
    assert ('evidence' in result['error'].lower()
            or 'output' in result['error'].lower())


def test_input_identity_does_not_require_obsolete_partial_or_plan_fields():
    manifest = dict(
        scale='1B', files={
            'descriptor.json': 'a' * 64,
            'prediction.parquet': 'b' * 64
        })
    case = dict(scale='1B', task='binary', arm='normal', seed=0,
                descriptor='descriptor.json',
                prediction_input='prediction.parquet')
    identity = nightly.input_signature(manifest, case)
    assert identity['descriptor'] == 'a' * 64
    assert identity['prediction'] == 'b' * 64
    assert 'partial' not in identity
    assert 'requested_plan' not in identity


@pytest.mark.parametrize('failure', ['exit', 'timeout'])
@pytest.mark.parametrize('measurement_status', ['MEASURED', 'INCOMPLETE'])
def test_supervisor_retains_worker_failure_details(tmp_path, monkeypatch,
                                                   failure, measurement_status):
    path, _ = configuration(tmp_path)
    output = tmp_path / 'output'
    monkeypatch.delenv('NIGHTLY_CASE_CGROUP', raising=False)
    monkeypatch.setattr(nightly, 'execution_identity',
                        lambda config, wheel: dict(wheel_sha256='a' * 64))
    gates = nightly.sibling('nightly_gates')
    monkeypatch.setattr(nightly.sibling('nightly_host'), 'require_local_ssd',
                        lambda path: dict(mount_target=str(tmp_path)))
    monkeypatch.setattr(nightly.sibling('nightly_host'),
                        'require_frozen_inputs',
                        lambda path: dict(read_only=True))
    monkeypatch.setattr(gates, 'validate_policy', lambda *args: None)

    @contextmanager
    def dedicated_scope(root):
        yield tmp_path / 'scope'

    def failed_worker(command, group, seconds, **kwargs):
        nightly.save(
            output / 'result.json',
            dict(status='FAIL', scale='100M', task='binary', arm='normal',
                 seed=0, error='specific wheel mismatch',
                 traceback='worker trace'))
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, seconds)
        return subprocess.CompletedProcess(command, 1)

    resources = SimpleNamespace(
        dedicated_cgroup=dedicated_scope, run_with_deadline=failed_worker,
        finalize_resources=lambda *args, **kwargs: dict(
            status=measurement_status, errors=['resource failure']
            if measurement_status == 'INCOMPLETE' else []))
    sibling = nightly.sibling
    monkeypatch.setattr(
        nightly, 'sibling', lambda name: resources
        if name == 'nightly_resources' else sibling(name))
    code = nightly.main([
        'run', '--configuration',
        str(path), '--scale', '100M', '--case', 'binary-normal', '--mode',
        'reference', '--input-root',
        str(tmp_path / 'inputs'), '--wheel',
        str(tmp_path / 'missing.whl'), '--scratch',
        str(tmp_path / 'scratch'), '--output-dir',
        str(output), '--cgroup-root', '/sys/fs/cgroup', '--time-budget-seconds',
        '100', '--reference-output-s3', 's3://test-bucket/proposals'
    ])
    assert code == 1
    result = nightly.read(output / 'result.json')
    assert result['status'] == 'FAIL'
    assert result['error'] == 'specific wheel mismatch'
    assert result['traceback'] == 'worker trace'
    assert ('code 1' if failure == 'exit' else
            'TimeoutExpired') in result['supervisor_error']


def test_nightly_rejects_writable_inputs_before_engine_work(tmp_path):
    path, _ = configuration(tmp_path)
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    output = tmp_path / 'output'
    code = nightly.main([
        'run', '--configuration',
        str(path), '--scale', '100M', '--case', 'binary-normal', '--mode',
        'reference', '--input-root',
        str(inputs), '--wheel',
        str(tmp_path / 'reference.whl'), '--scratch',
        str(tmp_path / 'scratch'), '--output-dir',
        str(output), '--cgroup-root',
        str(tmp_path / 'groups'), '--time-budget-seconds', '60',
        '--reference-output-s3', 's3://test-bucket/proposals'
    ])
    assert code == 1
    result = nightly.read(output / 'result.json')
    assert 'read-only' in result['error']
    assert 'runtime' not in result
