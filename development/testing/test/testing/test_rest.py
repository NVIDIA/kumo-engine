# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import kumo.testing.rest as rest_helpers


class _ParamsThatMustNotBeMutated:
    def merge(self, *args, **kwargs):
        raise AssertionError('test client defaults must not be mutated')


def test_post_success_omits_skip_cache_by_default(monkeypatch):
    captured = {}
    response = object()

    class _Client:
        def post(self, _path, **kwargs):
            captured.update(kwargs)
            return response

    monkeypatch.setattr(rest_helpers, '_dump_dataclass', lambda payload: payload)
    monkeypatch.setattr(rest_helpers, '_check_status_code', lambda *_: None)
    monkeypatch.setattr(rest_helpers, 'parse_dataclass',
                        lambda _response_type, value: value)

    assert rest_helpers.post_and_expect_success(
        _Client(), '/strict-post', {}, object) is response
    assert captured['params'] == {}


def test_get_success_omits_skip_cache_by_default(monkeypatch):
    captured = {}
    response = object()

    class _Client:
        def get(self, _path, **kwargs):
            captured.update(kwargs)
            return response

    monkeypatch.setattr(rest_helpers, '_check_status_code', lambda *_: None)
    monkeypatch.setattr(rest_helpers, 'parse_dataclass',
                        lambda _response_type, value: value)

    assert rest_helpers.get_and_expect_success(
        _Client(), '/strict-get', {}, object) is response
    assert captured['params'] == {}


def test_create_identity_backed_connector_does_not_mutate_client_defaults(
        monkeypatch):
    client = SimpleNamespace(params=_ParamsThatMustNotBeMutated())
    binding = rest_helpers.IdentityBindingSetup(
        identity_id='identity_00000000000000000000000000000000',
        group_id='group_00000000000000000000000000000001',
    )
    monkeypatch.setattr(rest_helpers, 'create_identity_binding',
                        lambda *args, **kwargs: binding)
    monkeypatch.setattr(
        rest_helpers,
        'create_project_for_group',
        lambda *args, **kwargs: 'project_00000000000000000000000000000001',
    )
    monkeypatch.setattr(rest_helpers, '_api_external_config',
                        lambda config: config)
    monkeypatch.setattr(rest_helpers.data_source_api, 'CreateConnectorArgs',
                        lambda **kwargs: kwargs)
    monkeypatch.setattr(
        rest_helpers,
        'post_and_expect_success',
        lambda *args, **kwargs: SimpleNamespace(
            id='connector_00000000000000000000000000000000'),
    )

    setup = rest_helpers.create_identity_backed_connector(
        client,
        identity_config=SimpleNamespace(),
        credentials=SimpleNamespace(),
        connector_config=SimpleNamespace(name='warehouse'),
    )

    assert setup.project_id == 'project_00000000000000000000000000000001'
    assert setup.connector_id == 'connector_00000000000000000000000000000000'
