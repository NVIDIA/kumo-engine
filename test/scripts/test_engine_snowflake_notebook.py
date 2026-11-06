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
    manifest = prepare.prepare(payload, DIRECTORY / 'candidate.json')
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
    asset = json.loads(
        (DIRECTORY / 'candidate.json').read_text())['assets']['wheelhouse.tar']
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
    asset = json.loads(
        (DIRECTORY / 'candidate.json').read_text())['assets']['wheelhouse.tar']
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


def test_environment_guard_rejects_unrestricted_or_other_branches(modules):
    guard = importlib.import_module('check_environment')
    environment = {
        'can_admins_bypass':
        False,
        'protection_rules': [{
            'type':
            'required_reviewers',
            'reviewers': [{
                'type': 'Team',
                'reviewer': {
                    'id': guard.TEAM_ID
                }
            }]
        }],
        'deployment_branch_policy': {
            'protected_branches': False,
            'custom_branch_policies': True
        }
    }
    branch = {'name': guard.BRANCH, 'type': 'branch'}
    guard.validate(environment, {'branch_policies': [branch]})
    for invalid in [[], [dict(branch, name='main')], [dict(branch, type='tag')],
                    [branch, dict(branch, name='another-branch')]]:
        with pytest.raises(ValueError):
            guard.validate(environment, {'branch_policies': invalid})
    with pytest.raises(ValueError):
        guard.validate({'deployment_branch_policy': None},
                       {'branch_policies': [branch]})
