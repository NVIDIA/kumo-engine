# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from kumoapi.data_source import DEFAULT_GROUP_ID, DEFAULT_PROJECT_ID

import kumoai_studio
from kumoai_studio import global_state
from kumoai_studio.client.client import KumoClient
from kumoai_studio.client.projects import ProjectsAPI
from kumoai_studio.exceptions import HTTPException
from kumoai_studio.project import Group, Project

PROJECT_A_ID = 'project_' + 'a' * 32
PROJECT_B_ID = 'project_' + 'b' * 32
DEFAULT_PROJECT = Project(DEFAULT_PROJECT_ID, 'default', 'Default',
                          DEFAULT_GROUP_ID)


@pytest.fixture(autouse=True)
def _reset_global_state():
    global_state.clear()
    yield
    global_state.clear()


def _client(projects=(), projects_by_id=None):
    client = MagicMock(spec=KumoClient)
    client.projects_api.list_projects.return_value = list(projects)
    by_id = {project.id: project for project in projects}
    by_id.update(projects_by_id or {})
    client.projects_api.get_project.side_effect = by_id.__getitem__
    by_slug = {project.slug: project for project in projects}

    def get_by_slug(slug):
        try:
            return by_slug[slug]
        except KeyError:
            raise HTTPException(404, 'project unavailable') from None

    client.projects_api.get_project_by_slug.side_effect = get_by_slug
    return client


def test_init_has_no_group_parameter():
    assert 'group' not in inspect.signature(kumoai_studio.init).parameters


def test_canonical_default_project_id_is_literal_default():
    assert DEFAULT_PROJECT_ID == 'default'


def test_default_project_uses_only_canonical_id_lookup():
    client = _client(projects_by_id={DEFAULT_PROJECT_ID: DEFAULT_PROJECT})

    kumoai_studio._init_project(client, None)

    assert kumoai_studio.current_project() == DEFAULT_PROJECT
    client.projects_api.get_project.assert_called_once_with(DEFAULT_PROJECT_ID)
    client.projects_api.list_projects.assert_not_called()


def test_explicit_default_project_id_precedes_slug_and_name_resolution():
    client = _client(projects_by_id={DEFAULT_PROJECT_ID: DEFAULT_PROJECT})

    kumoai_studio._init_project(client, 'default')

    assert kumoai_studio.current_project() == DEFAULT_PROJECT
    client.projects_api.get_project.assert_called_once_with('default')
    client.projects_api.get_project_by_slug.assert_not_called()
    client.projects_api.list_projects.assert_not_called()


def test_explicit_project_id_uses_direct_lookup_without_fallback():
    project = Project(PROJECT_A_ID, 'analytics', 'Analytics',
                      'group_' + 'a' * 32)
    client = _client(projects_by_id={PROJECT_A_ID: project})

    kumoai_studio._init_project(client, PROJECT_A_ID)

    assert kumoai_studio.current_project() == project
    client.projects_api.get_project.assert_called_once_with(PROJECT_A_ID)
    client.projects_api.list_projects.assert_not_called()


def test_project_name_uses_normalized_visible_lookup():
    project = Project(PROJECT_A_ID, 'fraud-detection', 'Fraud Detection',
                      'group_' + 'a' * 32)
    client = _client([project])

    kumoai_studio._init_project(client, '  ＦＲＡＵＤ ｄｅｔｅｃｔｉｏｎ  ')

    assert kumoai_studio.current_project() == project
    client.projects_api.list_projects.assert_called_once_with()
    client.projects_api.get_project.assert_not_called()


def test_project_slug_uses_exact_endpoint_and_sets_current_slug():
    project = Project(PROJECT_A_ID, 'fraud-detection', 'Fraud Detection',
                      'group_' + 'a' * 32)
    client = _client([project])

    kumoai_studio._init_project(client, 'fraud-detection')

    assert kumoai_studio.current_project() == project
    assert kumoai_studio.current_project().slug == 'fraud-detection'
    client.projects_api.get_project_by_slug.assert_called_once_with(
        'fraud-detection')
    client.projects_api.list_projects.assert_called_once_with()


def test_project_slug_and_display_name_ambiguity_fails_safely():
    slug_project = Project(PROJECT_A_ID, 'analytics', 'Analytics Workspace',
                           'group_' + 'a' * 32)
    name_project = Project(PROJECT_B_ID, 'other-project', 'analytics',
                           'group_' + 'b' * 32)
    client = _client([slug_project, name_project])

    with pytest.raises(RuntimeError, match='ambiguous'):
        kumoai_studio._init_project(client, 'analytics')


def test_missing_project_slug_falls_back_to_exact_normalized_display_name():
    project = Project(PROJECT_A_ID, 'other-project', 'analytics',
                      'group_' + 'a' * 32)
    client = _client([project])

    kumoai_studio._init_project(client, 'analytics')

    assert kumoai_studio.current_project() == project
    client.projects_api.get_project_by_slug.assert_called_once_with(
        'analytics')


def test_missing_project_id_fails_without_name_or_default_fallback():
    client = _client(projects_by_id={DEFAULT_PROJECT_ID: DEFAULT_PROJECT})

    with pytest.raises(KeyError):
        kumoai_studio._init_project(client, PROJECT_A_ID)

    assert global_state._project_id is None
    client.projects_api.list_projects.assert_not_called()
    client.projects_api.get_project.assert_called_once_with(PROJECT_A_ID)


def test_missing_project_name_fails_without_default_fallback():
    client = _client([])

    with pytest.raises(ValueError, match='not found or is not accessible'):
        kumoai_studio._init_project(client, 'Missing')

    assert global_state._project_id is None
    client.projects_api.get_project.assert_not_called()


@pytest.mark.parametrize('lookup', [
    'Fraud Detection',
    '  ＦＲＡＵＤ ｄｅｔｅｃｔｉｏｎ  ',
])
def test_get_project_resolves_visible_name_without_changing_active_project(
        lookup):
    project = Project(PROJECT_A_ID, 'fraud-detection', 'Fraud Detection',
                      'group_' + 'a' * 32)
    client = _client([project])
    kumoai_studio._set_active_project(DEFAULT_PROJECT)

    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)):
        result = kumoai_studio.get_project(lookup)

    assert result == project
    assert kumoai_studio.current_project() == DEFAULT_PROJECT
    client.projects_api.list_projects.assert_called_once_with()
    client.projects_api.get_project.assert_not_called()


def test_get_project_id_uses_direct_endpoint_without_changing_active_project():
    project = Project(PROJECT_A_ID, 'fraud-detection', 'Fraud Detection',
                      'group_' + 'a' * 32)
    client = _client(projects_by_id={PROJECT_A_ID: project})
    kumoai_studio._set_active_project(DEFAULT_PROJECT)

    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)):
        result = kumoai_studio.get_project(PROJECT_A_ID)

    assert result == project
    assert kumoai_studio.current_project() == DEFAULT_PROJECT
    client.projects_api.get_project.assert_called_once_with(PROJECT_A_ID)
    client.projects_api.list_projects.assert_not_called()


def test_get_project_default_id_uses_direct_endpoint_before_slug():
    client = _client(projects_by_id={DEFAULT_PROJECT_ID: DEFAULT_PROJECT})
    other = Project(PROJECT_A_ID, 'analytics', 'Analytics',
                    'group_' + 'a' * 32)
    kumoai_studio._set_active_project(other)

    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)):
        result = kumoai_studio.get_project('default')

    assert result == DEFAULT_PROJECT
    assert kumoai_studio.current_project() == other
    client.projects_api.get_project.assert_called_once_with('default')
    client.projects_api.get_project_by_slug.assert_not_called()
    client.projects_api.list_projects.assert_not_called()


@pytest.mark.parametrize('project', [None, '', '   '])
def test_get_project_rejects_missing_or_empty_input(project):
    with pytest.raises(ValueError,
                       match='non-empty ID, slug, or display name'):
        kumoai_studio.get_project(project)  # type: ignore[arg-type]


def test_get_project_missing_or_inaccessible_name_is_value_error():
    client = _client([DEFAULT_PROJECT])
    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)):
        with pytest.raises(ValueError, match='not found or is not accessible'):
            kumoai_studio.get_project('Missing')


@pytest.mark.parametrize('status_code', [403, 404])
def test_get_project_preserves_http_error_for_inaccessible_or_missing_id(
        status_code):
    client = _client()
    expected = HTTPException(status_code, 'project unavailable')
    client.projects_api.get_project.side_effect = expected
    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)):
        with pytest.raises(HTTPException) as caught:
            kumoai_studio.get_project(PROJECT_A_ID)

    assert caught.value is expected
    assert caught.value.status_code == status_code
    client.projects_api.list_projects.assert_not_called()


def test_get_project_rejects_duplicate_normalized_visible_names():
    client = _client([
        Project(PROJECT_A_ID, 'fraud-a', 'Fraud Detection',
                'group_' + 'a' * 32),
        Project(PROJECT_B_ID, 'fraud-b', ' ＦＲＡＵＤ ｄｅｔｅｃｔｉｏｎ ',
                'group_' + 'b' * 32),
    ])
    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)):
        with pytest.raises(RuntimeError, match='duplicate project name'):
            kumoai_studio.get_project('fraud detection')


def test_project_client_exposes_only_safe_project_metadata():
    response = MagicMock(ok=True)
    response.json.return_value = {
        'id': PROJECT_A_ID,
        'slug': 'archived-project',
        'display_name': 'Archived project',
        'group_id': 'group_' + 'a' * 32,
        'status': 'archived',
        'identity_ids': ['identity_secret'],
        'connectors': [{
            'credentials': 'must-not-surface'
        }],
    }
    client = MagicMock(spec=KumoClient)
    client._request.return_value = response

    project = ProjectsAPI(client).get_project(PROJECT_A_ID)

    assert project == Project(PROJECT_A_ID, 'archived-project',
                              'Archived project', 'group_' + 'a' * 32)
    assert vars(project) == {
        'id': PROJECT_A_ID,
        'slug': 'archived-project',
        'name': 'Archived project',
        'group_id': 'group_' + 'a' * 32,
    }


def test_failed_init_is_atomic_and_retryable():
    client = MagicMock(spec=KumoClient)
    client._url = 'http://example.test/api'
    client._api_key = 'deployment:key'
    client.projects_api.get_project.side_effect = [
        ValueError('not authorized'),
        DEFAULT_PROJECT,
    ]

    with patch('kumoai_studio.KumoClient', return_value=client):
        with pytest.raises(ValueError, match='not authorized'):
            kumoai_studio.init(url=client._url, api_key=client._api_key)

        assert not global_state.initialized
        assert global_state._project_id is None

        kumoai_studio.init(url=client._url, api_key=client._api_key)

    assert global_state.initialized
    assert kumoai_studio.current_project() == DEFAULT_PROJECT
    assert client.authenticate.call_count == 2


def test_successful_init_does_not_log_endpoint_secrets(caplog):
    client = MagicMock(spec=KumoClient)
    client._url = 'https://example.test/api'
    client._api_key = 'deployment:key'
    client.projects_api.get_project.return_value = DEFAULT_PROJECT
    endpoint = 'https://user:password@example.test/api?token=secret'

    with caplog.at_level(logging.INFO, logger='kumoai_studio'), \
            patch('kumoai_studio.KumoClient', return_value=client):
        kumoai_studio.init(url=endpoint, api_key=client._api_key)

    assert 'password' not in caplog.text
    assert 'token=secret' not in caplog.text
    assert 'example.test' not in caplog.text


def test_set_project_id_uses_direct_lookup():
    project = Project(PROJECT_B_ID, 'risk', 'Risk', 'group_' + 'b' * 32)
    client = _client(projects_by_id={PROJECT_B_ID: project})
    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)):
        kumoai_studio.set_project(PROJECT_B_ID)

    assert kumoai_studio.current_project() == project
    client.projects_api.get_project.assert_called_once_with(PROJECT_B_ID)
    client.projects_api.list_projects.assert_not_called()


def test_set_project_default_id_uses_direct_lookup_and_updates_current():
    client = _client(projects_by_id={DEFAULT_PROJECT_ID: DEFAULT_PROJECT})
    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)):
        kumoai_studio.set_project('default')

    assert kumoai_studio.current_project() == DEFAULT_PROJECT
    assert kumoai_studio.current_project().id == 'default'
    client.projects_api.get_project.assert_called_once_with('default')
    client.projects_api.get_project_by_slug.assert_not_called()
    client.projects_api.list_projects.assert_not_called()


def test_set_project_name_uses_normalized_lookup():
    project = Project(PROJECT_B_ID, 'risk-models', 'Risk Models',
                      'group_' + 'b' * 32)
    client = _client([project])
    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)):
        kumoai_studio.set_project('  RISK models ')

    assert kumoai_studio.current_project() == project


def test_set_project_string_uses_public_get_project_resolution():
    project = Project(PROJECT_B_ID, 'risk-models', 'Risk Models',
                      'group_' + 'b' * 32)
    with patch('kumoai_studio.get_project',
               return_value=project) as get_project:
        kumoai_studio.set_project('  RISK models ')

    get_project.assert_called_once_with('  RISK models ')
    assert kumoai_studio.current_project() == project


def test_set_project_accepts_project_without_another_lookup():
    project = Project(PROJECT_B_ID, 'risk-models', 'Risk Models',
                      'group_' + 'b' * 32)
    with patch('kumoai_studio.get_project') as get_project:
        kumoai_studio.set_project(project)

    get_project.assert_not_called()
    assert kumoai_studio.current_project() == project


def test_get_project_is_public():
    assert 'get_project' in kumoai_studio.__all__


def test_get_project_has_required_dual_form_project_parameter():
    signature = inspect.signature(kumoai_studio.get_project)
    assert list(signature.parameters) == ['project']
    parameter = signature.parameters['project']
    assert parameter.default is inspect.Parameter.empty
    assert parameter.annotation is str


def test_create_project_without_group_uses_canonical_default():
    project = Project(PROJECT_A_ID, 'analytics', 'Analytics', DEFAULT_GROUP_ID)
    client = MagicMock()
    client.projects_api.create_project.return_value = project
    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)), \
            patch('kumoai_studio.list_groups') as list_groups:
        assert kumoai_studio.create_project('Analytics') == project

    list_groups.assert_not_called()
    client.projects_api.create_project.assert_called_once_with(
        name='Analytics', group_id=DEFAULT_GROUP_ID, is_private=False,
        slug=None)


def test_create_project_resolves_explicit_group_by_normalized_name():
    group = Group('group_' + 'a' * 32, 'data-science', 'Data Science')
    project = Project(PROJECT_A_ID, 'analytics', 'Analytics', group.id)
    client = MagicMock()
    client.projects_api.create_project.return_value = project
    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)), \
            patch('kumoai_studio.list_groups', return_value=[group]):
        assert kumoai_studio.create_project('Analytics',
                                            ' data SCIENCE ') == project

    client.projects_api.create_project.assert_called_once_with(
        name='Analytics', group_id=group.id, is_private=False, slug=None)


def test_create_project_resolves_group_slug_and_forwards_project_slug():
    group = Group('group_' + 'a' * 32, 'risk-team', 'Risk Team')
    project = Project(PROJECT_A_ID, 'fraud-detection', 'Fraud Detection',
                      group.id)
    client = MagicMock()
    client.groups_api.get_group_by_slug.return_value = group
    client.projects_api.create_project.return_value = project
    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)), \
            patch('kumoai_studio.list_groups', return_value=[]):
        result = kumoai_studio.create_project(
            'Fraud Detection',
            group='risk-team',
            slug='fraud-detection',
        )

    assert result.slug == 'fraud-detection'
    client.groups_api.get_group_by_slug.assert_called_once_with('risk-team')
    client.projects_api.create_project.assert_called_once_with(
        name='Fraud Detection', group_id=group.id, is_private=False,
        slug='fraud-detection')


def test_create_project_resolves_group_id_before_other_selectors():
    group = Group('group_' + 'a' * 32, 'risk-team', 'Risk Team')
    project = Project(PROJECT_A_ID, 'analytics', 'Analytics', group.id)
    client = MagicMock()
    client.projects_api.create_project.return_value = project
    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)), \
            patch('kumoai_studio.list_groups', return_value=[group]) as list_groups:
        result = kumoai_studio.create_project('Analytics', group=group.id)

    assert result == project
    list_groups.assert_called_once_with()
    client.groups_api.get_group_by_slug.assert_not_called()


def test_group_slug_and_display_name_ambiguity_fails_safely():
    slug_group = Group('group_' + 'a' * 32, 'risk-team', 'Risk Owners')
    name_group = Group('group_' + 'b' * 32, 'other-team', 'risk-team')
    client = MagicMock()
    client.groups_api.get_group_by_slug.return_value = slug_group
    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)), \
            patch('kumoai_studio.list_groups', return_value=[name_group]):
        with pytest.raises(RuntimeError, match='ambiguous'):
            kumoai_studio.create_project('Analytics', group='risk-team')


def test_client_injects_project_but_never_group():
    global_state._project_id = PROJECT_A_ID
    client = KumoClient('http://example', 'api-key')

    result = client._inject_project_param('/tables',
                                          {'params': {
                                              'other': 'value'
                                          }})

    assert result['params'] == {
        'other': 'value',
        'project_id': PROJECT_A_ID,
    }
    assert not hasattr(global_state, '_group_id')

    path_scoped = client._inject_project_param(
        f'/projects/{PROJECT_B_ID}/connectors',
        {'params': {
            'table_name': 'deep/table.parquet'
        }})
    assert path_scoped['params'] == {'table_name': 'deep/table.parquet'}

    explicit = client._inject_project_param(
        '/tables', {'params': {
            'project_id': PROJECT_B_ID
        }})
    assert explicit['params']['project_id'] == PROJECT_B_ID


def test_get_identity_matches_normalized_display_name():
    production = SimpleNamespace(name='Production AWS')
    kumoai_studio._set_active_project(DEFAULT_PROJECT)
    with patch('kumoai_studio.list_identities', return_value=[production]):
        assert kumoai_studio.get_identity('  ＰＲＯＤＵＣＴＩＯＮ ａｗｓ  ') is production


def test_get_identity_reports_unavailable_name():
    kumoai_studio._set_active_project(DEFAULT_PROJECT)
    with patch('kumoai_studio.list_identities', return_value=[]):
        with pytest.raises(ValueError, match='not available'):
            kumoai_studio.get_identity('Missing')


def test_get_identity_resolves_exact_slug_with_project_scope():
    identity = SimpleNamespace(
        id='identity_' + 'a' * 32,
        slug='prod-aws',
        name='Production AWS',
    )
    client = MagicMock()
    client.identity_api.get_by_slug.return_value = identity
    kumoai_studio._set_active_project(DEFAULT_PROJECT)
    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)), \
            patch('kumoai_studio.list_identities', return_value=[]):
        result = kumoai_studio.get_identity('prod-aws')

    assert result is identity
    client.identity_api.get_by_slug.assert_called_once_with(
        DEFAULT_PROJECT_ID, 'prod-aws')


def test_get_identity_resolves_opaque_id_before_other_selectors():
    identity_id = 'identity_' + 'a' * 32
    identity = SimpleNamespace(id=identity_id, slug='prod-aws',
                               name='Production AWS')
    client = MagicMock()
    kumoai_studio._set_active_project(DEFAULT_PROJECT)
    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)), \
            patch('kumoai_studio.list_identities',
                  return_value=[identity]) as list_identities:
        result = kumoai_studio.get_identity(identity_id)

    assert result is identity
    list_identities.assert_called_once_with()
    client.identity_api.get_by_slug.assert_not_called()


def test_identity_slug_and_display_name_ambiguity_fails_safely():
    slug_identity = SimpleNamespace(id='identity_' + 'a' * 32, slug='prod-aws',
                                    name='Production AWS')
    name_identity = SimpleNamespace(id='identity_' + 'b' * 32, slug='other',
                                    name='prod-aws')
    client = MagicMock()
    client.identity_api.get_by_slug.return_value = slug_identity
    kumoai_studio._set_active_project(DEFAULT_PROJECT)
    with patch.object(type(global_state), 'client',
                      new=property(lambda self: client)), \
            patch('kumoai_studio.list_identities', return_value=[name_identity]):
        with pytest.raises(RuntimeError, match='ambiguous'):
            kumoai_studio.get_identity('prod-aws')
