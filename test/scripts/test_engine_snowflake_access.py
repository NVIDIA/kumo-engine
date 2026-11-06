# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual pre-checkout workflow gate without cloud credentials."""
import copy
import importlib
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'packages/kumo-engine/test/snowflake_notebook'
WORKFLOW = ROOT / '.github/workflows/kumo-engine-snowflake-notebook.yml'
SECRETS = {
    'KUMO_ENGINE_TEAM_READ_TOKEN', 'SNOWFLAKE_TEST_CONFIG',
    'SNOWFLAKE_PRIVATE_KEY', 'SNOWFLAKE_PRIVATE_KEY_PASSPHRASE',
    'SNOWFLAKE_UI_CREDENTIALS'
}


@pytest.fixture
def gate(monkeypatch):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    step = workflow['jobs']['engine-lifecycle']['steps'][0]
    assert step == workflow['jobs']['notebook-ui']['steps'][0]
    script = '\n'.join(step['run'].splitlines()[1:-1])
    for name, value in dict(TEAM_READ_TOKEN='synthetic-token',
                            GITHUB_ACTOR='maintainer',
                            GITHUB_TRIGGERING_ACTOR='maintainer').items():
        monkeypatch.setenv(name, value)
    environment = dict(
        can_admins_bypass=False, protection_rules=[
            dict(type='required_reviewers',
                 reviewers=[dict(type='Team', reviewer=dict(id=19956283))])
        ], deployment_branch_policy=dict(protected_branches=False,
                                         custom_branch_policies=True))
    state = dict(
        environment=environment, names=set(SECRETS), actors={}, calls=[],
        policies=[dict(name='egu/mlops-platform-integration', type='branch')])

    def get(request, timeout):
        assert request.full_url.startswith('https://api.github.com/')
        assert request.get_header('Authorization') == 'Bearer synthetic-token'
        state['calls'].append(request.full_url)
        if '/memberships/' in request.full_url:
            actor = request.full_url.rsplit('/', 1)[1]
            value = state['actors'].get(actor, dict(state='active'))
            if isinstance(value, int):
                raise urllib.error.HTTPError(request.full_url, value, 'denied',
                                             {}, None)
        elif '/secrets?' in request.full_url:
            value = dict(secrets=[dict(name=name) for name in state['names']])
        elif '/deployment-branch-policies?' in request.full_url:
            value = dict(branch_policies=state['policies'])
        else:
            value = state['environment']
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(urllib.request, 'urlopen', get)

    def run():
        namespace = {}
        exec(compile(script, '<workflow-authorization>', 'exec'), namespace)

    return state, run


def test_maintainer_gate_passes_only_after_all_checks(gate):
    state, run = gate
    run()
    assert len(state['calls']) == 5
    assert state['calls'][-1].endswith('/secrets?per_page=100')


@pytest.mark.parametrize('who', ['GITHUB_ACTOR', 'GITHUB_TRIGGERING_ACTOR'])
@pytest.mark.parametrize('membership', [403, 404, dict(state='pending')])
def test_nonmember_or_unverifiable_actor_is_denied(gate, monkeypatch, who,
                                                   membership):
    state, run = gate
    monkeypatch.setenv(who, 'outsider')
    state['actors']['outsider'] = membership
    with pytest.raises(SystemExit):
        run()
    assert not any('/environments/' in url for url in state['calls'])


def test_missing_membership_credential_makes_no_requests(gate, monkeypatch):
    state, run = gate
    monkeypatch.delenv('TEAM_READ_TOKEN')
    with pytest.raises(SystemExit, match='Missing environment secret'):
        run()
    assert not state['calls']


@pytest.mark.parametrize('missing', sorted(SECRETS))
def test_repository_secret_fallback_cannot_satisfy_gate(gate, missing):
    state, run = gate
    state['names'].remove(missing)
    with pytest.raises(SystemExit, match='must exist in snowflake-mlops-test'):
        run()


@pytest.mark.parametrize('change', ['bypass', 'reviewer', 'branch'])
def test_removed_protection_is_rechecked_in_the_cloud_job(gate, change):
    state, run = gate
    if change == 'bypass':
        state['environment']['can_admins_bypass'] = True
    elif change == 'reviewer':
        state['environment']['protection_rules'][0]['reviewers'][0]['reviewer'][
            'id'] = 123
    else:
        state['policies'][0]['name'] = 'main'
    with pytest.raises(SystemExit):
        run()


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
    value['manifests']['candidate.json'] = '0' * 64
    monkeypatch.setenv('SNOWFLAKE_TEST_CONFIG', json.dumps(value))
    with pytest.raises(ValueError, match='exact candidate/asset manifests'):
        resources.require()


def test_authorized_resources_roundtrip(resources, monkeypatch):
    value = resources.expected()
    monkeypatch.setenv('SNOWFLAKE_TEST_CONFIG', json.dumps(value))
    assert resources.require() == value


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


@pytest.mark.parametrize('raw', ['not-json', 'null', '{}', '[]'])
def test_malformed_or_incomplete_configuration_fails_closed(
        resources, monkeypatch, raw):
    monkeypatch.setenv('SNOWFLAKE_TEST_CONFIG', raw)
    with pytest.raises(ValueError, match='SNOWFLAKE_TEST_CONFIG'):
        resources.require()
