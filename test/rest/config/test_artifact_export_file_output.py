# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from kumoapi.jobs import (
    FileTrainingTableOutputConfig,
    GCSPredictionOutput,
    PredictionArtifactType,
    S3PredictionOutput,
    TrainingTableArtifactType,
)

from kumo.connector.any_uri_adls_connector import AnyURIADLSConnector
from kumo.connector.any_uri_gcs_connector import AnyURIGCSConnector
from kumo.connector.any_uri_s3_connector import AnyURIS3Connector
from kumo.connector.composite_file_connector import CompositeFileConnector
from rest.config.artifact_export import (
    ArtifactExportInput,
    FileOutputConfig,
    require_output_path_within_connector,
    resolve_file_output_path,
)
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.prediction_output import \
    S3PredictionOutput as RestS3PredictionOutput
from statedb.config.common import ConnectorID, ProjectID
from temporalio_commons.dataclass_converter import KumoPayloadConverter

REFERENCE = ConnectorExecutionRef(project_id=ProjectID.generate(),
                                  connector_id=ConnectorID.generate())


@pytest.mark.parametrize(('field', 'coordinate'), [
    ('output_path', 'gs://bucket/output'),
    ('path', 'output'),
    ('path', 'gs://bucket/root/output'),
])
def test_artifact_output_payload_roundtrip(field, coordinate):
    output = FileOutputConfig(REFERENCE, **{field: coordinate})
    artifact = PredictionArtifactType.PREDICTIONS
    value = ArtifactExportInput({artifact: '/internal'}, {artifact: [output]})
    converter = KumoPayloadConverter()
    payloads = converter.to_payloads([value])
    restored = converter.from_payloads(payloads, [ArtifactExportInput])[0]
    restored_output = restored.output_config[artifact][0]
    actual = (getattr(restored_output, 'output_path', None)
              or getattr(restored_output, 'path', None))
    assert (restored_output.connector, actual) == (REFERENCE, coordinate)


@pytest.mark.parametrize(('output', 'artifact', 'expected'), [
    (S3PredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS, connector_id=str(
            REFERENCE.connector_id),
        file_path='s3://other/output/'), PredictionArtifactType.PREDICTIONS,
     FileOutputConfig(REFERENCE, output_path='s3://other/output/')),
    (GCSPredictionOutput(
        artifact_type=PredictionArtifactType.EMBEDDINGS, connector_id=str(
            REFERENCE.connector_id),
        file_path='nested/output/'), PredictionArtifactType.EMBEDDINGS,
     FileOutputConfig(REFERENCE, path='nested/output/')),
    (FileTrainingTableOutputConfig(connector_id=str(REFERENCE.connector_id),
                                   path='s3://other/training/'),
     TrainingTableArtifactType.FULL_TRAIN_TABLE,
     FileOutputConfig(REFERENCE, output_path='s3://other/training/')),
    (FileTrainingTableOutputConfig(connector_id=str(REFERENCE.connector_id),
                                   path='gs://other/training/'),
     TrainingTableArtifactType.FULL_TRAIN_TABLE,
     FileOutputConfig(REFERENCE, output_path='gs://other/training/')),
])
def test_api_output_preserves_string_coordinate(output, artifact, expected):
    export = ArtifactExportInput.from_out_config_n_spec({artifact: '/internal'},
                                                        output, REFERENCE)
    assert export.output_config[artifact] == [expected]
    assert 'credential' not in repr(export).casefold()
    assert 'session' not in repr(export).casefold()


def test_prediction_artifact_preserves_multiple_destinations():
    first = ConnectorExecutionRef(
        project_id=REFERENCE.project_id,
        connector_id=ConnectorID.generate(),
    )
    second = ConnectorExecutionRef(
        project_id=REFERENCE.project_id,
        connector_id=ConnectorID.generate(),
    )
    outputs = [
        RestS3PredictionOutput(
            artifact_type=PredictionArtifactType.PREDICTIONS,
            connector_id=str(first.connector_id),
            file_path='s3://first/predictions/',
        ),
        RestS3PredictionOutput(
            artifact_type=PredictionArtifactType.PREDICTIONS,
            connector_id=str(second.connector_id),
            file_path='s3://second/predictions/',
        ),
    ]
    export = ArtifactExportInput.from_bp_input(
        SimpleNamespace(
            config=SimpleNamespace(outputs=outputs),
            output_connectors={
                first.connector_id: first,
                second.connector_id: second,
            },
        ),
        {PredictionArtifactType.PREDICTIONS: '/internal'},
    )

    assert [
        output.output_path
        for output in export.output_config[PredictionArtifactType.PREDICTIONS]
    ] == [
        's3://first/predictions/',
        's3://second/predictions/',
    ]


@pytest.mark.parametrize(('connector', 'config', 'expected'), [
    (CompositeFileConnector(name='s3', root_dir='s3://bucket/root'),
     FileOutputConfig(REFERENCE, path='output/'), 's3://bucket/root/output/'),
    (CompositeFileConnector(name='gcs', root_dir='gs://bucket/root'),
     FileOutputConfig(REFERENCE, path='output/'), 'gs://bucket/root/output/'),
    (CompositeFileConnector(
        name='adls', root_dir='abfss://data@account.dfs.core.windows.net/root'),
     FileOutputConfig(REFERENCE, path='output/'),
     'abfss://data@account.dfs.core.windows.net/root/output/'),
    (AnyURIS3Connector(name='s3'),
     FileOutputConfig(REFERENCE,
                      output_path='s3://other/output/'), 's3://other/output/'),
    (AnyURIGCSConnector(name='gcs'),
     FileOutputConfig(REFERENCE,
                      output_path='gs://other/output/'), 'gs://other/output/'),
    (AnyURIADLSConnector(name='adls'),
     FileOutputConfig(
         REFERENCE,
         output_path='abfss://data@account.dfs.core.windows.net/output/'),
     'abfss://data@account.dfs.core.windows.net/output/'),
])
def test_destination_resolves_for_matching_connector_mode(
        connector, config, expected, monkeypatch):
    monkeypatch.setattr(
        'io_utils.fsutil.s3_filesystem.select_s3_region', lambda path: None)
    assert resolve_file_output_path(config, connector) == expected


@pytest.mark.parametrize(('connector', 'config'), [
    (AnyURIS3Connector(name='s3'), FileOutputConfig(REFERENCE, path='output/')),
    (AnyURIGCSConnector(name='gcs'), FileOutputConfig(REFERENCE,
                                                      path='output/')),
    (CompositeFileConnector(name='s3', root_dir='s3://bucket/root'),
     FileOutputConfig(REFERENCE, output_path='s3://other/output/')),
    (CompositeFileConnector(name='gcs', root_dir='gs://bucket/root'),
     FileOutputConfig(REFERENCE, output_path='gs://other/output/')),
])
def test_destination_rejects_mismatched_connector_mode(connector, config):
    with pytest.raises(ValueError, match='full|relative|outside'):
        resolve_file_output_path(config, connector)


@pytest.mark.parametrize('path', [
    's3://other/root/predictions',
    's3://customer/root-adjacent/predictions',
    's3://customer/escape',
    's3://customer/root/predictions?token=secret',
])
def test_rooted_output_rejects_escape(path):
    with pytest.raises(ValueError, match='outside|query'):
        require_output_path_within_connector(path, 's3://customer/root')
