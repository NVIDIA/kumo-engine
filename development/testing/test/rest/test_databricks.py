# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import kumoapi.source_table as source_api
from fastapi.testclient import TestClient

from kumo.testing.databricks import withDatabricksOauthCredentials
from kumo.testing.rest import (
    create_identity_backed_connector,
    databricks_test_identity,
    get_and_expect_success,
    hm_databricks_config,
)


@withDatabricksOauthCredentials
def test_list_databricks_tables_with_schema_async(client: TestClient):
    identity_config, credentials = databricks_test_identity()
    setup = create_identity_backed_connector(
        client,
        identity_config=identity_config,
        credentials=credentials,
        connector_config=dataclasses.replace(
            hm_databricks_config('test_databricks'),
            schema_name='test_schema',
        ),
    )

    resp = get_and_expect_success(
        client,
        f"/v0/projects/{setup.project_id}/connectors/"
        f"{setup.connector_id}/source-tables",
        {"group_id": setup.group_id},
        source_api.SourceTableListResponse,
    )

    # Discovery is scoped by the schema required on the connector config.
    assert len(resp.table_names) >= 3
