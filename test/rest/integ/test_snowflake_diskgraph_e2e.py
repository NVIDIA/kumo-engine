# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snowflake diskgraph end-to-end training test.

Exercises the full diskgraph training pipeline backed by a real Snowflake
connector (account xva19026, schema FINANCIAL_PYTEST) and validates:
1. Training completes successfully (status == DONE)
2. Model metrics exist and are reasonable
3. Diskgraph train table parquet exists with 'arange' column
4. JobConfig has diskgraph_index_path set

Prerequisites:
    - Temporal dev server running: make up-local-temporal
    - diskgraph Python package installed
    - DEPLOY_ENV_CONFIG_FILE pointing to a diskgraph deploy config
    - Snowflake credentials available (via AWS Secrets Manager or env vars)

Usage:
    python -m pytest test/rest/integ/test_snowflake_diskgraph_e2e.py -xvs \\
        --timeout=1800 -m snowflake
"""
import logging
import os
from pathlib import Path
from test.rest.integ.test_diskgraph_e2e import (
    _check_diskgraph_train_table,
    _check_job_config_diskgraph,
    _check_metrics,
    _setup_and_train,
    _wait_for_training,
)

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from io_utils.json import load_json
from kumo.connector import snowflake_connector
from kumo.testing import withSnowflakeCredentials
from kumo.testing.rest import (
    create_identity_backed_connector,
    financial_snowflake_config,
    randomize_fixture_ids_v2,
    remap_fixture_connector_ids_v2,
    snowflake_test_identity,
)
from rest.config.api_command import FixtureDefinition
from rest.config.credentials import KeyPair
from rest.testing.integration import wait_for_training_workflow_to_finish
from statedb.config.common import JobStatus

log = logging.getLogger(__name__)


FIXTURE_PATH = Path(
    __file__).parent / "test_cases" / "test_snowflake_e2e_fixtures.json"
PQL_TEST_CASE = "snowflake_regression_base"

# Pool fixtures (_diskgraph_deploy_config, _pool_subprocesses) are pulled in
# transitively via pytestmark — same pattern as test_diskgraph_e2e.py.
# storage_backend is pinned to s3 to exercise Kumo-owned remote persistence
# before the pool-local diskgraph download.
pytestmark = [
    pytest.mark.ci_gate('snowflake-e2e'),
    pytest.mark.usefixtures("_pool_subprocesses"),
    pytest.mark.parametrize("storage_backend", [
        pytest.param("s3", marks=pytest.mark.private_resource('aws')),
    ], indirect=True),
]


def _register_connector(client: TestClient, fixture: FixtureDefinition):
    connector_name = str(
        next(iter(fixture.graphs[0].tables.values())).get_connector_id())
    key_pair = KeyPair(
        user=SecretStr(os.environ[snowflake_connector._ENV_SNOWFLAKE_USER]),
        private_key=SecretStr(
            os.environ[snowflake_connector._ENV_SNOWFLAKE_PRIVATE_KEY]),
    )
    identity_config, credentials = snowflake_test_identity(key_pair)
    return create_identity_backed_connector(
        client,
        identity_config=identity_config,
        credentials=credentials,
        connector_config=financial_snowflake_config(connector_name),
    )


@pytest.fixture
def test_cleanup_options():
    """Preserve state across the test — don't reset DB between operations."""
    from rest.testing.config import TestFunctionCleanupOptions
    return TestFunctionCleanupOptions(
        should_reset_rest_db=False,
        should_reset_ml_db=False,
        should_reset_data_backend_cache=True,
        should_delete_data_persist_dir=True,
    )


@pytest.mark.snowflake
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@withSnowflakeCredentials
async def test_snowflake_diskgraph_e2e_training(
    client: TestClient,
    temporal_client,
):
    """Full diskgraph training pipeline backed by a real Snowflake connector.

    Verifies that SnowflakeConnector.export_table() streams parquet into
    Kumo-owned persistence and connector.download() copies it to the local
    diskgraph ingest root, completing the training pipeline.
    """
    fixture = load_json(str(FIXTURE_PATH), FixtureDefinition)
    fixture.queries = [
        q for q in fixture.queries if q.pquery_name == PQL_TEST_CASE
    ]
    assert len(
        fixture.queries) == 1, f"Query {PQL_TEST_CASE!r} not found in fixture"
    fixture = randomize_fixture_ids_v2(fixture)
    randomized_name = fixture.queries[0].pquery_name
    temporary_connector_id = str(
        next(iter(fixture.graphs[0].tables.values())).get_connector_id())
    setup = _register_connector(client, fixture)
    fixture = remap_fixture_connector_ids_v2(
        fixture, {temporary_connector_id: setup.connector_id})

    pquery_id, training_job_id = await _setup_and_train(
        client,
        fixture,
        randomized_name,
        provision_connectors=False,
        project_id=setup.project_id,
    )
    log.info("Training started: pquery_id=%s, training_job_id=%s", pquery_id,
             training_job_id)

    job = _wait_for_training(
        client,
        training_job_id,
        project_id=setup.project_id,
    )
    assert job.execution.status == JobStatus.DONE, (
        f"Training failed: status={job.execution.status}, "
        f"results={job.execution.results}")
    await wait_for_training_workflow_to_finish(training_job_id,
                                               temporal_client)
    log.info("Training completed successfully.")

    # 1. Validate model metrics
    results = job.execution.results
    assert results is not None, "Training job completed but has no results"
    _check_metrics(results)

    # 2. Validate diskgraph train table artifact exists
    _check_diskgraph_train_table(training_job_id)

    # 3. Validate JobConfig has diskgraph backend settings
    _check_job_config_diskgraph(training_job_id)

    log.info("Snowflake diskgraph e2e training test passed.")
