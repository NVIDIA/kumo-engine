# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify native environment protection and explicit test-resource admission."""
import copy
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'packages/kumo-engine/test/snowflake_notebook'


@pytest.fixture
def environment_guard(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    guard = importlib.import_module('check_environment')
    environment = dict(
        can_admins_bypass=False, protection_rules=[
            dict(type='required_reviewers', prevent_self_review=True,
                 reviewers=[dict(type='Team', reviewer=dict(id=guard.TEAM_ID))])
        ], deployment_branch_policy=dict(protected_branches=False,
                                         custom_branch_policies=True))
    policies = dict(branch_policies=[
        dict(name='main', type='branch'),
        dict(name='feature/platform-trial', type='branch')
    ])
    return guard, environment, policies


def test_existing_maintainer_environment_is_sufficient(environment_guard):
    guard, environment, policies = environment_guard
    guard.validate(environment, policies)


@pytest.mark.parametrize(
    'change', ['bypass', 'reviewer', 'branch', 'no-reviewer', 'self-review'])
def test_unprotected_environment_is_rejected(environment_guard, change):
    guard, environment, policies = environment_guard
    if change == 'bypass':
        environment['can_admins_bypass'] = True
    elif change == 'reviewer':
        environment['protection_rules'][0]['reviewers'][0]['reviewer'][
            'id'] = 123
    elif change == 'no-reviewer':
        environment['protection_rules'] = []
    elif change == 'self-review':
        environment['protection_rules'][0]['prevent_self_review'] = False
    else:
        policies['branch_policies'][0]['name'] = 'untrusted'
    with pytest.raises(ValueError):
        guard.validate(environment, policies)


@pytest.fixture
def resources(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    return importlib.import_module('resources')


def test_live_resources_require_explicit_configuration(resources, monkeypatch):
    monkeypatch.delenv('SNOWFLAKE_TEST_CONFIG', raising=False)
    with pytest.raises(ValueError, match='required for live execution'):
        resources.require()


@pytest.mark.parametrize('field', [
    'database', 'warehouse', 'pool', 'stage', 'account', 'user', 'browser_user'
])
def test_configuration_cannot_select_nontest_resources(resources, monkeypatch,
                                                       field):
    value = copy.deepcopy(resources.expected())
    value[field] = 'production'
    monkeypatch.setenv('SNOWFLAKE_TEST_CONFIG', json.dumps(value))
    with pytest.raises(ValueError, match='dedicated test resources'):
        resources.require()


def test_changed_payload_requires_environment_config_update(
        resources, monkeypatch):
    value = resources.expected()
    value['manifests']['graph-assets.json'] = '0' * 64
    monkeypatch.setenv('SNOWFLAKE_TEST_CONFIG', json.dumps(value))
    with pytest.raises(ValueError, match='exact graph asset manifest'):
        resources.require()


def test_github_cannot_fall_back_to_local_keys(resources, monkeypatch):
    launch = importlib.import_module('launch')
    monkeypatch.setenv('SNOWFLAKE_TEST_CONFIG',
                       json.dumps(resources.expected()))
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.delenv('SNOWFLAKE_PRIVATE_KEY', raising=False)
    with pytest.raises(ValueError, match='environment RSA secrets'):
        launch.connection_parameters(Path('/not-a-key'))


def test_github_cannot_fall_back_to_local_browser_credentials(
        resources, monkeypatch):
    ui = importlib.import_module('ui')
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.delenv('SNOWFLAKE_UI_CREDENTIALS', raising=False)
    with pytest.raises(ui.UIBlocked, match='environment browser secret'):
        ui.credentials(Path('/not-a-credential-file'))


@pytest.mark.parametrize('feature',
                         ['feature/platform-trial', 'feature/other-trial'])
def test_main_only_environment_survives_feature_branch_removal(
        environment_guard, feature):
    guard, environment, _ = environment_guard
    policies = dict(branch_policies=[dict(name='main', type='branch')])
    guard.validate(environment, policies, branch=feature)
    with pytest.raises(ValueError, match='selected branch'):
        guard.validate(environment, policies, branch=feature,
                       selected_branch=feature)


@pytest.mark.parametrize('name,kind', [('*', 'branch'), ('feature/*', 'branch'),
                                       ('main', 'tag'),
                                       ('untrusted', 'branch')])
def test_unapproved_deployment_rules_are_rejected(environment_guard, name,
                                                  kind):
    guard, environment, policies = environment_guard
    policies['branch_policies'].append(dict(name=name, type=kind))
    with pytest.raises(ValueError):
        guard.validate(environment, policies)


@pytest.mark.parametrize('environment_name,feature',
                         [('snowflake-mlops-test', 'feature/platform-trial'),
                          ('other-platform-test', 'feature/other-trial')])
@pytest.mark.parametrize(
    'ref,event,repository,allowed',
    [('refs/heads/main', 'workflow_dispatch', 'NVIDIA/kumo-engine', True),
     ('feature', 'workflow_dispatch', 'NVIDIA/kumo-engine', True),
     ('refs/heads/untrusted', 'workflow_dispatch', 'NVIDIA/kumo-engine', False),
     ('refs/tags/main', 'workflow_dispatch', 'NVIDIA/kumo-engine', False),
     ('refs/heads/main', 'push', 'NVIDIA/kumo-engine', False),
     ('refs/heads/main', 'pull_request', 'NVIDIA/kumo-engine', False),
     ('refs/heads/main', 'workflow_dispatch', 'other/repository', False)])
def test_environment_guard_checks_main_and_feature_before_api_access(
        environment_guard, monkeypatch, environment_name, feature, ref, event,
        repository, allowed):
    import io

    guard, environment, policies = environment_guard
    policies['branch_policies'][1]['name'] = feature
    monkeypatch.setenv('GITHUB_REPOSITORY', repository)
    monkeypatch.setenv('GITHUB_REF',
                       'refs/heads/' + feature if ref == 'feature' else ref)
    monkeypatch.setenv('GITHUB_EVENT_NAME', event)
    monkeypatch.setenv('GITHUB_TOKEN', 'synthetic-token')
    calls = []

    def get(request, timeout):
        calls.append(request.full_url)
        assert '/environments/' + environment_name in request.full_url
        return io.BytesIO(
            json.dumps(policies if 'deployment-branch-policies' in
                       request.full_url else environment).encode())

    monkeypatch.setattr(guard.urllib.request, 'urlopen', get)
    if allowed:
        guard.main(environment=environment_name, branch=feature)
        assert len(calls) == 2
    else:
        with pytest.raises(SystemExit, match='manual dispatch'):
            guard.main(environment=environment_name, branch=feature)
        assert calls == []


@pytest.mark.parametrize('case', [
    'match', 'foreign', 'missing', 'empty', 'extra', 'scalar', 'query-error',
    'close-error'
])
def test_sql_connection_checks_account_before_any_test_writes(
        resources, monkeypatch, case):
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    launch = importlib.import_module('launch')
    target = resources.expected()
    monkeypatch.setenv('SNOWFLAKE_TEST_CONFIG', json.dumps(target))
    monkeypatch.setattr(launch, 'connection_parameters', lambda _: {})
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = {
        'match': (target['account_locator'], ),
        'foreign': ('PRIVATE_FOREIGN_ACCOUNT', ),
        'missing': None,
        'empty': (),
        'extra': (target['account_locator'], 'PRIVATE'),
        'scalar': target['account_locator'],
        'query-error': None,
        'close-error': None,
    }[case]
    if case == 'query-error':
        cursor.execute.side_effect = RuntimeError('PRIVATE credential response')
    if case == 'close-error':
        connection.close.side_effect = RuntimeError('PRIVATE close response')
    connector = SimpleNamespace(connect=lambda **_: connection)
    monkeypatch.setitem(sys.modules, 'snowflake',
                        SimpleNamespace(connector=connector))
    monkeypatch.setitem(sys.modules, 'snowflake.connector', connector)
    if case == 'match':
        assert launch.connect(None) is connection
        connection.close.assert_not_called()
    else:
        with pytest.raises(ValueError, match='Cannot verify') as error:
            launch.connect(None)
        assert 'PRIVATE' not in str(error.value)
        assert error.value.__suppress_context__
        connection.close.assert_called_once_with()
    cursor.execute.assert_called_once_with('SELECT CURRENT_ACCOUNT()')


def test_production_entrypoint_reads_policy_then_rejects_unapproved_branch(
        environment_guard, monkeypatch):
    import io
    guard, environment, policies = environment_guard
    monkeypatch.setenv('GITHUB_REPOSITORY', 'NVIDIA/kumo-engine')
    monkeypatch.setenv('GITHUB_REF', 'refs/heads/untrusted')
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'workflow_dispatch')
    monkeypatch.setenv('GITHUB_TOKEN', 'synthetic-token')
    calls = []

    def get(request, timeout):
        calls.append(request.full_url)
        value = policies if 'deployment-branch-policies' in request.full_url else environment
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(guard.urllib.request, 'urlopen', get)
    with pytest.raises(ValueError, match='selected branch'):
        guard.main()
    assert len(calls) == 2


@pytest.mark.parametrize('selected', ['main', 'feature-test'])
@pytest.mark.parametrize('prevent_self_review', [False, True])
def test_trial_self_approval_keeps_main_independent(environment_guard, selected,
                                                    prevent_self_review):
    guard, environment, policies = environment_guard
    environment['protection_rules'][0][
        'prevent_self_review'] = prevent_self_review
    policies['branch_policies'] = [
        dict(name='main', type='branch'),
        dict(name='feature-test', type='branch')
    ]
    policies['total_count'] = 2
    if selected == 'main' and not prevent_self_review:
        with pytest.raises(ValueError, match='self-review'):
            guard.validate(environment, policies, branch='feature-test',
                           selected_branch=selected)
    else:
        guard.validate(environment, policies, branch='feature-test',
                       selected_branch=selected)


def test_legacy_dependency_hashes_do_not_block_new_candidates(resources, monkeypatch):
    value = resources.expected()
    value['manifests'].update({'dependencies.json': 'a' * 64,
                               'dependency-asset.json': 'b' * 64})
    monkeypatch.setenv('SNOWFLAKE_TEST_CONFIG', json.dumps(value))
    assert resources.require() == resources.expected()
