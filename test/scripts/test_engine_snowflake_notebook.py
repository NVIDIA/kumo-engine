# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submission and result guards; these tests do not qualify live Snowflake UI."""
import copy
import importlib
import json
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve(
).parents[2] / 'packages/kumo-engine/test/snowflake_notebook'


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    return tuple(
        importlib.import_module(name)
        for name in ('launch', 'prepare', 'verify'))


def completed():
    return dict(
        status='blocked', engine_checks=dict(
            status='passed', phases=dict.fromkeys(
                ('fit', 'score', 'evaluate', 'negative'), 'passed'),
            remaining_requirements={
                'exported_epoch_history': {
                    'status': 'not implemented'
                },
                'ui_review': {
                    'status': 'blocked'
                }
            }), stages={
                'preflight': {
                    'status': 'passed'
                },
                'lifecycle': {
                    'status': 'blocked'
                }
            })


def test_selected_engine_result_preserves_full_qualification_gaps(modules):
    _, _, verify = modules
    result = verify.verdict(completed())
    assert result['engine_checks'] == 'passed'
    assert result['full_qualification'] == 'incomplete'
    assert set(result['remaining_requirements']) == verify.KNOWN_GAPS


@pytest.mark.parametrize('change', [
    'timeout', 'missing_phase', 'preflight', 'cleanup', 'collection',
    'unknown_gap', 'unexplained_block'
])
def test_incomplete_or_erroneous_execution_cannot_pass(modules, change):
    _, _, verify = modules
    report = completed()
    if change == 'timeout':
        report['stages']['lifecycle']['error'] = {
            'message': 'statement timeout'
        }
    elif change == 'missing_phase':
        del report['engine_checks']['phases']['negative']
    elif change == 'preflight':
        report['stages']['preflight']['status'] = 'blocked'
    elif change == 'cleanup':
        report['close_error'] = {'message': 'close failed'}
    elif change == 'collection':
        report['stages']['lifecycle']['collection_error'] = {
            'message': 'checksum mismatch'
        }
    elif change == 'unknown_gap':
        report['engine_checks']['remaining_requirements']['other_failure'] = {
            'status': 'blocked'
        }
    else:
        report['engine_checks']['remaining_requirements'] = {}
    with pytest.raises(ValueError):
        verify.verdict(report)


def test_plan_keeps_large_assets_remote_and_rejects_changed_sources(
        modules, tmp_path):
    launch, prepare, _ = modules
    payload = tmp_path / 'payload'
    manifest = prepare.prepare(
        payload,
        Path(__file__).parent / 'data/snowflake-candidate.json')
    for mode in ('preflight', 'lifecycle'):
        plan = launch.prepare_payload(tmp_path / mode, 'V2.9-CPU-PY3.12',
                                      payload / mode, approved=manifest)
        assert plan['remote_assets'] == manifest['remote_assets']
        assert not (tmp_path / mode / 'payload/wheelhouse.tar').exists()
        assert len(plan['remote_assets']) == 2
    (payload /
     'lifecycle/wheel_mlops_lifecycle.py').write_text('changed source')
    with pytest.raises(ValueError, match='content differs'):
        launch.prepare_payload(tmp_path / 'changed', 'V2.9-CPU-PY3.12',
                               payload / 'lifecycle', approved=manifest)


def test_changed_remote_asset_is_rejected_without_download_or_upload(
        modules, monkeypatch):
    launch, _, _ = modules
    asset = json.loads((Path(__file__).parent / 'data/snowflake-candidate.json'
                        ).read_text())['assets']['wheelhouse.tar']
    observed = copy.deepcopy(asset['remote'])
    observed['md5'] = 'changed-same-size-object'
    monkeypatch.setattr(launch, 'list_object', lambda *_: observed)
    with pytest.raises(ValueError, match='changed or disappeared'):
        launch.existing_remote_asset(object(), 'wheelhouse.tar', asset)
    monkeypatch.setattr(launch, 'list_object', lambda *_: None)
    with pytest.raises(ValueError, match='changed or disappeared'):
        launch.existing_remote_asset(object(), 'wheelhouse.tar', asset)


def test_foreign_stage_reference_is_rejected_before_cloud_access(modules):
    launch, _, _ = modules
    asset = json.loads((Path(__file__).parent / 'data/snowflake-candidate.json'
                        ).read_text())['assets']['wheelhouse.tar']
    asset['location'] = '@PRODUCTION.PUBLIC.STAGE/private-data'
    with pytest.raises(ValueError, match='Invalid staged asset'):
        launch.existing_remote_asset(object(), 'wheelhouse.tar', asset)


def test_ci_authentication_uses_encrypted_secret_without_local_cache(
        modules, monkeypatch):
    launch, _, _ = modules
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import (
        BestAvailableEncryption,
        Encoding,
        PrivateFormat,
        load_der_private_key,
    )
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8,
                            BestAvailableEncryption(b'synthetic-ci-passphrase'))
    monkeypatch.setenv('SNOWFLAKE_PRIVATE_KEY', pem.decode())
    monkeypatch.setenv('SNOWFLAKE_PRIVATE_KEY_PASSPHRASE',
                       'synthetic-ci-passphrase')
    resources = importlib.import_module('resources')
    monkeypatch.setenv('SNOWFLAKE_TEST_CONFIG',
                       json.dumps(resources.expected()))
    config = launch.connection_parameters(None)
    actual = load_der_private_key(config['private_key'], password=None)
    assert actual.public_key().public_numbers() == key.public_key(
    ).public_numbers()
    assert config['authenticator'] == 'SNOWFLAKE_JWT'
    assert config['user'] == 'KUMO_SNOWFLAKE_CI'


def test_engine_only_verifier_never_claims_unchecked_full_qualification(
        modules):
    _, _, verify = modules
    report = completed()
    report['status'] = 'passed'
    report['engine_checks']['remaining_requirements'] = {}
    with pytest.raises(ValueError):
        verify.verdict(report)


def test_collector_accepts_only_expected_gap_exit_after_validating_receipts(
        modules, tmp_path):
    launch, _, _ = modules
    phases = ('fit', 'score', 'evaluate', 'negative')
    summary = dict(
        status='blocked',
        phases={phase: dict(status='passed', returncode=0)
                for phase in phases},
        required_gaps=completed()['engine_checks']['remaining_requirements'])
    paths = {}
    for phase in phases:
        path = tmp_path / (phase + '.json')
        path.write_text(
            json.dumps(dict(status='passed', environment={'sha256': 'a' * 64})))
        paths['lifecycle/' + phase + '.json'] = path
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps(summary))
    paths['lifecycle/summary.json'] = path
    receipt = dict(status='blocked', lifecycle=dict(returncode=1,
                                                    summary=summary))
    manifest = dict(mode='lifecycle', wheel_sha256='a' * 64)
    launch.validate_lifecycle_exit(receipt, paths, manifest)
    receipt['lifecycle']['returncode'] = 2
    with pytest.raises(AssertionError):
        launch.validate_lifecycle_exit(receipt, paths, manifest)
    receipt['lifecycle']['returncode'] = 1
    summary['phases']['fit']['returncode'] = 1
    path.write_text(json.dumps(summary))
    with pytest.raises(AssertionError):
        launch.validate_lifecycle_exit(receipt, paths, manifest)


def test_lifecycle_evidence_checks_survive_python_optimization(
        modules, tmp_path):
    launch, _, _ = modules
    namespace = dict(__file__=launch.__file__, __name__='optimized_launcher')
    exec(
        compile(
            Path(launch.__file__).read_text(), launch.__file__, 'exec',
            optimize=2), namespace)
    phases = ('fit', 'score', 'evaluate', 'negative')
    paths = {}
    summary = {
        'phases': {
            name: dict(status='passed', returncode=1)
            for name in phases
        }
    }
    for name, value in [
        ('summary', summary),
            *[(name, dict(status='passed', environment={'sha256': 'wheel'}))
              for name in phases]
    ]:
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps(value))
        paths['lifecycle/' + name + '.json'] = path
    with pytest.raises(AssertionError, match='did not exit successfully'):
        namespace['lifecycle_verdict'](paths, {'wheel_sha256': 'wheel'})


def test_public_result_does_not_print_provider_error_text(modules, capsys):
    launch, _, _ = modules
    private = 'private-' + 'account.snowflakecomputing.com'
    error = dict(type='OperationalError', message=private)
    report = dict(status='failed',
                  stages={'lifecycle': dict(error=error, reason=private)},
                  error=error, close_error=error)
    launch.print_result(report, 'summary.json')
    output = capsys.readouterr().out
    assert private not in output
    assert 'inspect private evidence' in output


def custom_workload():
    return dict(file='wheel_mlops_pytorch_snowflake.py',
                helpers=['wheel_mlops_pytorch.py'], phases=['train', 'reload'],
                sha256='c' * 64, wheels=[])


@pytest.mark.parametrize('fault',
                         [None, 'worker_sha', 'summary_sha', 'exit', 'gap'])
def test_customer_lifecycle_requires_its_declared_receipts_and_workload_identity(
        modules, tmp_path, fault):
    launch, _, verify = modules
    custom = custom_workload()
    names = ('platform-prerequisite', *custom['phases'])
    paths, recorded = {}, {}
    for name in names:
        value = dict(status='passed', environment=dict(sha256='a' * 64),
                     workload=dict(sha256=custom['sha256']))
        if fault == 'worker_sha' and name == 'reload':
            value['workload']['sha256'] = 'd' * 64
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps(value))
        paths['lifecycle/' + name + '.json'] = path
        recorded[name] = dict(
            value, returncode=1 if fault == 'exit' and name == 'train' else 0)
    summary = dict(
        status='blocked', phases=recorded,
        workload=dict(sha256='d' *
                      64 if fault == 'summary_sha' else custom['sha256']),
        required_gaps=dict(ui_review=dict(status='blocked')))
    if fault == 'gap':
        summary['required_gaps']['exported_epoch_history'] = dict(
            status='not implemented')
    summary_file = tmp_path / 'summary.json'
    summary_file.write_text(json.dumps(summary))
    paths['lifecycle/summary.json'] = summary_file
    manifest = dict(mode='lifecycle', wheel_sha256='a' * 64, workload=custom)
    receipt = dict(status='blocked', lifecycle=dict(returncode=1,
                                                    summary=summary))
    if fault is not None:
        with pytest.raises(AssertionError):
            launch.validate_lifecycle_exit(receipt, paths, manifest)
        return
    launch.validate_lifecycle_exit(receipt, paths, manifest)
    checks = launch.lifecycle_verdict(paths, manifest)
    assert checks['phases'] == dict.fromkeys(names, 'passed')
    assert checks['workload'] == custom
    report = completed()
    report['engine_checks'] = checks
    assert verify.verdict(report)['engine_checks'] == 'passed'


@pytest.mark.parametrize('dirty', [False, True])
def test_customer_payload_pins_committed_sources_and_dependency_inventory(
        modules, monkeypatch, tmp_path, dirty):
    launch, prepare, _ = modules
    repo = DIRECTORY.parents[3]
    workload = DIRECTORY.parent / 'wheel_mlops_pytorch_snowflake.py'
    helper = DIRECTORY.parent / 'wheel_mlops_pytorch.py'
    descriptor = json.loads(
        (Path(__file__).parent / 'data/snowflake-candidate.json').read_text())
    wheel = 'diskgraph-0.2.0-cp311-abi3-linux_x86_64.whl'
    descriptor['workload_wheels'] = [wheel]
    descriptor['dependencies']['workload_wheels'] = [wheel]
    descriptor['dependencies']['wheels'][wheel] = dict(sha256='d' * 64, size=1,
                                                       name='diskgraph',
                                                       version='0.2.0')
    candidate = tmp_path / 'candidate.json'
    candidate.write_text(json.dumps(descriptor))

    def committed(command, **kwargs):
        path = repo / command[-1].removeprefix('HEAD:')
        return b'different committed source' if dirty else path.read_bytes()

    monkeypatch.setattr(prepare.subprocess, 'check_output', committed)
    copyfile = prepare.shutil.copyfile

    def copy_frozen(source, destination):
        assert source not in (workload,
                              helper), 'Do not reread admitted customer sources'
        return copyfile(source, destination)

    monkeypatch.setattr(prepare.shutil, 'copyfile', copy_frozen)
    output = tmp_path / 'payload'
    if dirty:
        with pytest.raises(ValueError,
                           match='Commit the exact workload source'):
            prepare.prepare(output, candidate, workload, [helper])
        assert not output.exists()
        return
    approved = prepare.prepare(output, candidate, workload, [helper])
    for mode in ('preflight', 'negative', 'lifecycle'):
        plan = launch.prepare_payload(tmp_path / mode, 'V2.9-CPU-PY3.12',
                                      output / mode, approved=approved)
        assert plan['workload']['phases'] == ['train', 'reload']
        assert plan['workload']['wheels'] == [wheel]
        assert (tmp_path / mode / 'payload' / workload.name).is_file()
        assert (tmp_path / mode / 'payload' / helper.name).is_file()
    config_path = output / 'lifecycle/lifecycle-config.json'
    config = json.loads(config_path.read_text())
    config['workload']['phases'] = ['unapproved']
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='Workload differs'):
        launch.prepare_payload(tmp_path / 'changed', 'V2.9-CPU-PY3.12',
                               output / 'lifecycle', approved=approved)
