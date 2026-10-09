# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core.sources.types import ReadMode

from kumo.engine.connector.bigquery import BigQueryConnector
from kumo.engine.connector.identity import BigQueryIdentity


@pytest.mark.parametrize(
    'staging_uri, require_staged, expected_mode',
    [
        (None, False, None),
        ('gs://staging/kumo', False, None),
        ('gs://staging/kumo', True, ReadMode.STAGED),
    ],
)
def test_read_configuration_reaches_the_cached_backend(staging_uri,
                                                       require_staged,
                                                       expected_mode):
    identity = BigQueryIdentity()
    connector = BigQueryConnector(
        project_id='data-project',
        dataset_id='analytics',
        identity=identity,
        staging_uri=staging_uri,
        require_staged=require_staged,
    )

    backend = connector.source_backend

    assert connector.source_backend is backend
    assert backend._identity is identity
    assert backend.staging_uri == staging_uri
    assert backend.mode is expected_mode


@pytest.mark.parametrize('staging_uri', [None, ''])
def test_strict_staging_requires_a_uri(staging_uri):
    with pytest.raises(ValueError, match='require_staged needs a staging_uri'):
        BigQueryConnector(
            project_id='data-project',
            dataset_id='analytics',
            staging_uri=staging_uri,
            require_staged=True,
        )


def test_sink_backend_is_cached_and_carries_configuration():
    from kumo_bigquery.sink import BigQuerySinkBackend

    identity = BigQueryIdentity()
    connector = BigQueryConnector(
        project_id='data-project',
        dataset_id='analytics',
        identity=identity,
        staging_uri='gs://staging/kumo',
    )

    sink = connector.sink_backend

    assert connector.sink_backend is sink
    assert isinstance(sink, BigQuerySinkBackend)
    assert sink._identity is identity
    assert sink.staging_uri == 'gs://staging/kumo'


def test_fit_rejects_model_export_to_bigquery_up_front():
    from kumoapi.model_plan import ModelPlan

    from kumo.engine.trainer.output_config import ModelOutputConfig
    from kumo.engine.trainer.trainer import Trainer

    connector = BigQueryConnector(project_id='data-project',
                                  dataset_id='analytics')

    with pytest.raises(NotImplementedError, match='cannot publish models'):
        Trainer(ModelPlan()).fit(
            None, snapshot=None,
            export_config=ModelOutputConfig(connector, path='m'))
