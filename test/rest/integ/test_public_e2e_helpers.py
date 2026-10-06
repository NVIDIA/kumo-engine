# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from test.rest.integ import test_e2e_public_api
from unittest.mock import MagicMock

from kumo_core.types import TaskType
from kumoapi.pquery import QueryType

from rest.config.predictive_query_v2 import PQueryMetadataResponse


def test_task_type_lookup_uses_explicit_project_scope(monkeypatch) -> None:
    response = PQueryMetadataResponse(
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    get_and_expect_success = MagicMock(return_value=response)
    monkeypatch.setattr(test_e2e_public_api, 'get_and_expect_success',
                        get_and_expect_success)
    private_api_client = MagicMock()

    assert test_e2e_public_api._get_task_type_v1_compat(
        private_api_client,
        'pquery-test',
        project_id='project-test',
    ) == TaskType.REGRESSION
    get_and_expect_success.assert_called_once_with(
        private_api_client,
        '/v1/predictive_queries/pquery-test/metadata',
        {'project_id': 'project-test'},
        PQueryMetadataResponse,
    )
