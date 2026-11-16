# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from test.rest.integ.black_box.project_upload_helpers import upload_project_csv

from fastapi.testclient import TestClient


def test_completed_project_upload_has_safe_source_table_config(
    public_api_client: TestClient,
    tmp_path,
) -> None:
    projects = public_api_client.get('/v1/projects')
    assert projects.status_code == HTTPStatus.OK, projects.text
    project_id = str(projects.json()[0]['id'])
    source_path = tmp_path / 'orders.csv'
    source_path.write_text('order_id\n1\n', encoding='utf-8')
    connector_id = upload_project_csv(
        public_api_client,
        project_id,
        source_path,
        api_prefix='/v1',
    )

    response = public_api_client.get(
        f'/v1/projects/{project_id}/connectors/{connector_id}/'
        'source-tables/orders/config')

    assert response.status_code == HTTPStatus.OK, response.text
    assert response.json() == {
        'source_table': {
            'connector_id': connector_id,
            'source_table_name': 'orders',
            'file_type': 'CSV',
            'data_source_type': 'S3',
        },
    }
