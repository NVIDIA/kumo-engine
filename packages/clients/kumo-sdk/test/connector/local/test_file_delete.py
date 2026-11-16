# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from kumoapi.data_source import UploadFileType

from kumoai_studio.connector.file_upload_connector import FileUploadConnector


@pytest.fixture
def connector():
    state = MagicMock()
    state._project_id = 'project-owned'
    response = state.client.connector_api.get_upload_connector.return_value
    response.connector_id = 'connector-project-uploads'
    with patch('kumoai_studio.connector.file_upload_connector.global_state',
               state):
        yield FileUploadConnector(file_type='parquet',
                                  project_id='project-owned'), state


def test_upload_connector_uses_project_hidden_connector(connector):
    upload_connector, _ = connector

    assert upload_connector.id == 'connector-project-uploads'
    assert upload_connector.name == 'uploads'


def test_upload_connector_snapshots_project_before_lookup():
    state = MagicMock()
    state._project_id = 'project-b'
    response = state.client.connector_api.get_upload_connector.return_value
    response.connector_id = 'connector-project-uploads'

    def switch_global_project(*, project_id):
        assert project_id == 'project-b'
        state._project_id = 'project-a'
        return response

    state.client.connector_api.get_upload_connector.side_effect = (
        switch_global_project)
    with patch('kumoai_studio.connector.file_upload_connector.global_state', state), \
            patch('kumoai_studio._require_project'):
        upload_connector = FileUploadConnector(file_type='parquet')

    assert upload_connector.kumo_project_id == 'project-b'
    state.client.connector_api.get_upload_connector.assert_called_once_with(
        project_id='project-b')


def test_delete_sends_format_but_no_global_connector_id(connector):
    upload_connector, _ = connector
    with patch('kumoai_studio.connector.utils.global_state') as state:
        upload_connector.delete(name='users')

    request = state.client.connector_api.delete_file_upload.call_args.args[0]
    assert request.source_table_name == 'users'
    assert request.file_type == UploadFileType.PARQUET
    assert not hasattr(request, 'connector_id')
    state.client.connector_api.delete_file_upload.assert_called_once_with(
        request, project_id='project-owned')


def test_upload_keeps_connector_project_after_global_switch(connector):
    upload_connector, _ = connector
    with patch('kumoai_studio.connector.file_upload_connector.upload_table'
               ) as upload:
        upload_connector.upload(name='users', path='/tmp/users.parquet')

    upload.assert_called_once_with(
        path='/tmp/users.parquet',
        name='users',
        file_type='parquet',
        auto_partition=True,
        partition_size_mb=250,
        project_id='project-owned',
    )
