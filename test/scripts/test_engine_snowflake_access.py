# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify native environment protection and explicit test-resource admission."""
import copy
import importlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'packages/kumo-engine/test/snowflake_notebook'
WORKFLOW = ROOT / '.github/workflows/kumo-engine-snowflake-notebook.yml'


@pytest.fixture
def environment_guard(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    guard = importlib.import_module('check_environment')
    environment = dict(
        can_admins_bypass=False, protection_rules=[
            dict(type='required_reviewers',
                 reviewers=[dict(type='Team', reviewer=dict(id=guard.TEAM_ID))])
        ], deployment_branch_policy=dict(protected_branches=False,
                                         custom_branch_policies=True))
    policies = dict(branch_policies=[dict(name=guard.BRANCH, type='branch')])
    return guard, environment, policies


def test_existing_maintainer_environment_is_sufficient(environment_guard):
    guard, environment, policies = environment_guard
    guard.validate(environment, policies)


@pytest.mark.parametrize('change',
                         ['bypass', 'reviewer', 'branch', 'no-reviewer'])
def test_unprotected_environment_is_rejected(environment_guard, change):
    guard, environment, policies = environment_guard
    if change == 'bypass':
        environment['can_admins_bypass'] = True
    elif change == 'reviewer':
        environment['protection_rules'][0]['reviewers'][0]['reviewer'][
            'id'] = 123
    elif change == 'no-reviewer':
        environment['protection_rules'] = []
    else:
        policies['branch_policies'][0]['name'] = 'main'
    with pytest.raises(ValueError):
        guard.validate(environment, policies)


def test_every_cloud_job_rechecks_protection_with_builtin_token():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    gate = workflow['jobs']['configuration']['steps'][1]
    assert gate['env'] == {'GITHUB_TOKEN': '${{ github.token }}'}
    for name in ('engine-lifecycle', 'notebook-ui'):
        job = workflow['jobs'][name]
        assert job['environment'] == 'snowflake-mlops-test'
        assert job['permissions'] == {'contents': 'read', 'actions': 'read'}
        assert job['steps'][1] == gate
        assert 'secrets.' not in json.dumps(job['steps'][:2])
    assert set(workflow[True]) == {'workflow_dispatch'}
    assert 'KUMO_ENGINE_TEAM_READ_TOKEN' not in WORKFLOW.read_text()


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
