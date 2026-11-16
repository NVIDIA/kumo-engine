# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
from kumoapi.data_source import (
    DataSourceType,
    S3ConnectorResourceConfig,
    S3RootedConnectorTarget,
)
from kumoapi.plugin import SourceBackend, SourceConnector
from kumoapi.source_table import (
    FileType,
    S3RelativeSource,
    S3SourceTable,
    SourceTableConfigResponse,
    SourceTableListResponse,
)

from kumoai_studio.backend.cloud import RestSourceBackend
from kumoai_studio.connector.base import Connector


class _Connector(Connector):
    r"""Minimal connector used to exercise the source area."""
    def __init__(self, project_id='project-owned'):
        self._kumo_project_id = project_id

    @property
    def id(self):
        return 'connector-1'

    @property
    def name(self):
        return 'test'

    @property
    def source_type(self):
        return DataSourceType.S3

    @property
    def config(self):
        return S3ConnectorResourceConfig(target=S3RootedConnectorTarget(
            bucket='company-data', prefix='events'))


def _config():
    return SourceTableConfigResponse(source_table=S3SourceTable(
        connector_id='connector-1',
        source=S3RelativeSource(path='events'),
        file_type=FileType.PARQUET,
    ))


def test_rest_backend_implements_the_protocol(protocol_members):
    missing = [
        attr for attr in protocol_members(SourceBackend)
        if not hasattr(RestSourceBackend(), attr)
    ]
    assert not missing


def test_connector_satisfies_the_structural_contract(protocol_members):
    r"""The source area declares what it may rely on any connector to expose,
    so `kumoai_studio.backend` need not import `kumoai_studio.connector`.
    """
    connector = _Connector()
    missing = [
        attr for attr in protocol_members(SourceConnector)
        if not hasattr(connector, attr)
    ]
    assert not missing


def test_table_names_read_through_the_source_area(mock_backend):
    mock_backend.source.list_tables.return_value = SourceTableListResponse(
        table_names=['events', 'users'])
    connector = _Connector()

    assert connector.table_names() == ['events', 'users']
    mock_backend.source.list_tables.assert_called_once_with(connector)


def test_table_lookup_reads_through_the_source_area(mock_backend):
    mock_backend.source.get_table_config.return_value = _config()
    connector = _Connector()

    table = connector.table('events')

    assert table.name == 'events'
    mock_backend.source.get_table_config.assert_called_once_with(
        connector, 'events')


@pytest.mark.parametrize('sample_rows,method', [
    (0, 'get_columns'),
    (5, 'get_sample'),
])
def test_table_data_selects_columns_or_sample(mock_backend, sample_rows,
                                              method):
    r"""Sampling is requested only when rows are actually asked for."""
    response = object()
    getattr(mock_backend.source, method).return_value = response
    connector = _Connector()

    assert connector._get_table_data(['events'], sample_rows) == [response]

    assert getattr(mock_backend.source, method).call_count == 1
    other = 'get_sample' if method == 'get_columns' else 'get_columns'
    getattr(mock_backend.source, other).assert_not_called()


def test_download_urls_pass_the_connector_and_project(mock_backend):
    mock_backend.source.get_download_urls.return_value = ['https://signed']
    connector = _Connector()

    assert connector._get_download_urls('events') == ['https://signed']

    mock_backend.source.get_download_urls.assert_called_once_with(
        connector, 'events', project_id=None)


def test_download_urls_accept_an_explicit_project(mock_backend):
    r"""Callers holding an exact project pass it rather than re-deriving it."""
    mock_backend.source.get_download_urls.return_value = []
    connector = _Connector()

    connector._get_download_urls('events', project_id='project-other')

    mock_backend.source.get_download_urls.assert_called_once_with(
        connector, 'events', project_id='project-other')


def test_rest_backend_defaults_to_the_connector_project():
    source = RestSourceBackend()
    connector = _Connector(project_id='project-owned')

    with mock.patch('kumoai_studio.global_state') as state:
        source.list_tables(connector)

    state.client.source_table_api.list_tables.assert_called_once_with(
        'connector-1', project_id='project-owned')


def test_rest_backend_honours_an_explicit_project():
    source = RestSourceBackend()
    connector = _Connector(project_id='project-owned')

    with mock.patch('kumoai_studio.global_state') as state:
        source.get_download_urls(connector, 'events',
                                 project_id='project-other')

    state.client.source_table_api._get_download_urls.assert_called_once_with(
        'connector-1', 'events', project_id='project-other')


def test_rest_backend_resolves_the_client_per_call():
    source = RestSourceBackend()
    connector = _Connector()

    with mock.patch('kumoai_studio.global_state') as state:
        state.client.source_table_api.list_tables.return_value = 'first'
        assert source.list_tables(connector) == 'first'
        state.client.source_table_api.list_tables.return_value = 'second'
        assert source.list_tables(connector) == 'second'
