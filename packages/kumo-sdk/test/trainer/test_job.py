# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest
from kumoapi.jobs import (
    ADLSModelArtifactOutput,
    ADLSPredictionOutput,
    ArtifactExportRequest,
    BatchPredictionOptions,
    BatchPredictionRequest,
    BigQueryPredictionOutput,
    DatabricksModelArtifactOutput,
    DatabricksPredictionOutput,
    GCSModelArtifactOutput,
    GCSPredictionOutput,
    LocalPredictionOutput,
    MetadataField,
    PredictionArtifactType,
    S3ModelArtifactOutput,
    S3PredictionOutput,
    SnowflakeModelArtifactOutput,
    TrainingJobRequest,
    WriteMode,
)
from kumoapi.model_plan import ModelPlan

from kumoai_studio import (
    ADLSConnector,
    BigQueryConnector,
    DatabricksConnector,
    GCSConnector,
    S3Connector,
)
from kumoai_studio.artifact_export import (
    ModelExportConfig,
    ModelRegistryLocation,
)
from kumoai_studio.artifact_export.config import DatabricksOutputConfig
from kumoai_studio.artifact_export.job import export_model
from kumoai_studio.client.jobs import (
    GeneratePredictionTableJobAPI,
    GenerateTrainTableJobAPI,
    TrainingJobAPI,
)
from kumoai_studio.exceptions import HTTPException
from kumoai_studio.trainer.job import (
    ArtifactExportJob,
    BatchPredictionJob,
    BatchPredictionJobResult,
    OutputConfig,
    TrainingJob,
    TrainingJobResult,
)
from kumoai_studio.trainer.trainer import Trainer
from kumoai_studio.trainer.util import build_prediction_output_config


def test_build_adls_prediction_output_config():
    connector = object.__new__(ADLSConnector)
    connector._id = 'connector-adls'
    connector.root_dir = ('abfss://container@account.dfs.core.windows.net/out')
    output = build_prediction_output_config(
        "predictions",
        connector,
        "bp_table",
        [MetadataField.JOB_TIMESTAMP],
        OutputConfig(output_types={"predictions"},
                     output_table_name="bp_table"),
    )

    assert isinstance(output, ADLSPredictionOutput)
    assert output.file_path == "bp_table_predictions"
    assert output.artifact_type == PredictionArtifactType.PREDICTIONS
    assert output.connector_id == 'connector-adls'
    assert output.extra_fields == [MetadataField.JOB_TIMESTAMP]


def test_output_config_does_not_add_a_public_output_path_field():
    assert 'output_path' not in OutputConfig.__dataclass_fields__


def test_build_local_prediction_output_config():
    output = build_prediction_output_config(
        'embeddings',
        None,
        None,
        [],
        OutputConfig(output_types={'embeddings'}),
    )

    assert isinstance(output, LocalPredictionOutput)
    assert output.artifact_type == PredictionArtifactType.EMBEDDINGS


def test_build_databricks_prediction_output_config_preserves_write_mode():
    connector = object.__new__(DatabricksConnector)
    connector._id = 'connector-databricks'
    config = OutputConfig(
        output_types={'predictions'},
        output_connector=connector,
        output_table_name=('schema', 'prediction'),
        connector_specific_config=DatabricksOutputConfig(
            write_mode=WriteMode.APPEND),
    )

    output = build_prediction_output_config(
        'predictions',
        connector,
        'schema__kumo__prediction',
        [MetadataField.JOB_TIMESTAMP],
        config,
    )

    assert isinstance(output, DatabricksPredictionOutput)
    assert output.table_name == 'schema__kumo__prediction_predictions'
    assert output.write_mode == WriteMode.APPEND


def test_databricks_output_config_defaults_to_retry_safe_overwrite():
    assert DatabricksOutputConfig().write_mode == WriteMode.OVERWRITE


@pytest.mark.parametrize(
    'output_config,expected_outputs',
    [
        (None, {PredictionArtifactType.PREDICTIONS}),
        (
            OutputConfig(output_types={'predictions', 'embeddings'}),
            {
                PredictionArtifactType.PREDICTIONS,
                PredictionArtifactType.EMBEDDINGS,
            },
        ),
    ],
)
def test_predict_submits_requested_or_default_local_output(
    output_config,
    expected_outputs,
    mock_backend,
):
    trainer = object.__new__(Trainer)
    trainer._training_job_id = 'training-job'
    trainer._kumo_project_id = 'project-id'
    graph = mock.MagicMock()
    graph.kumo_project_id = 'project-id'
    graph.snapshot.return_value = 'graph-id@version'
    prediction_table = mock.MagicMock()
    prediction_table.kumo_project_id = 'project-id'
    prediction_table.job_id = 'prediction-table-job'
    response = mock.MagicMock()
    response.ok = True
    response.empty.return_value = True
    mock_backend.prediction.maybe_create.return_value = (
        'batch-prediction-job', response)

    job = trainer.predict(
        graph=graph,
        prediction_table=prediction_table,
        output_config=output_config,
        non_blocking=True,
    )

    assert isinstance(job, BatchPredictionJob)
    call_args = mock_backend.prediction.maybe_create.call_args
    request = call_args.args[0]
    assert {output.artifact_type
            for output in request.outputs} == expected_outputs
    assert all(
        isinstance(output, LocalPredictionOutput)
        for output in request.outputs)


def test_fit_uses_active_local_generated_table_snapshot(mock_backend):
    trainer = Trainer(ModelPlan())
    graph = mock.MagicMock()
    graph.id = 'graph-id'
    graph.kumo_project_id = 'project-id'
    graph.snapshot.side_effect = AssertionError('unexpected graph snapshot')
    train_table = mock.MagicMock()
    train_table.job_id = 'train-table-job'
    train_table.kumo_project_id = 'project-id'
    resource = SimpleNamespace(
        config=SimpleNamespace(pquery_id='predictive-query-id',
                               custom_train_table=None),
        custom_train_table_used=False,
    )
    mock_backend.name = 'local'
    mock_backend.train_table.get.return_value = resource
    mock_backend.train_table.snapshot_id.return_value = 'graph-id@data'
    mock_backend.graph.get_snapshot.return_value = SimpleNamespace(
        graph_name='graph-id')
    mock_backend.training.create.return_value = 'training-job'

    job = trainer.fit(graph=graph, train_table=train_table, non_blocking=True)

    assert isinstance(job, TrainingJob)
    request = mock_backend.training.create.call_args.args[0]
    assert request.graph_snapshot_id == 'graph-id@data'
    graph.snapshot.assert_not_called()


def test_fit_restores_graph_when_table_snapshot_is_not_active(mock_backend):
    trainer = Trainer(ModelPlan())
    graph = mock.MagicMock()
    graph.id = 'graph-id'
    graph.kumo_project_id = 'project-id'
    graph.snapshot.return_value = 'graph-id@restored-data'
    train_table = mock.MagicMock()
    train_table.job_id = 'train-table-job'
    train_table.kumo_project_id = 'project-id'
    resource = SimpleNamespace(
        config=SimpleNamespace(pquery_id='predictive-query-id',
                               custom_train_table=None),
        custom_train_table_used=False,
    )
    mock_backend.name = 'local'
    mock_backend.train_table.get.return_value = resource
    mock_backend.train_table.snapshot_id.return_value = 'graph-id@old-data'
    mock_backend.graph.get_snapshot.side_effect = ValueError(
        'snapshot is not active')
    mock_backend.training.create.return_value = 'training-job'

    trainer.fit(graph=graph, train_table=train_table, non_blocking=True)

    request = mock_backend.training.create.call_args.args[0]
    assert request.graph_snapshot_id == 'graph-id@restored-data'
    graph.snapshot.assert_called_once_with(non_blocking=True)


def test_fit_refuses_active_table_snapshot_from_another_graph(mock_backend):
    trainer = Trainer(ModelPlan())
    graph = mock.MagicMock()
    graph.id = 'requested-graph'
    graph.kumo_project_id = 'project-id'
    train_table = mock.MagicMock()
    train_table.job_id = 'train-table-job'
    train_table.kumo_project_id = 'project-id'
    resource = SimpleNamespace(
        config=SimpleNamespace(pquery_id='predictive-query-id',
                               custom_train_table=None),
        custom_train_table_used=False,
    )
    mock_backend.name = 'local'
    mock_backend.train_table.get.return_value = resource
    mock_backend.train_table.snapshot_id.return_value = 'other-graph@data'
    mock_backend.graph.get_snapshot.return_value = SimpleNamespace(
        graph_name='other-graph')

    with pytest.raises(ValueError, match='other-graph.*requested-graph'):
        trainer.fit(graph=graph, train_table=train_table, non_blocking=True)

    mock_backend.training.create.assert_not_called()
    graph.snapshot.assert_not_called()


def test_predict_uses_active_local_generated_table_snapshot(mock_backend):
    trainer = object.__new__(Trainer)
    trainer._training_job_id = 'training-job'
    trainer._kumo_project_id = 'project-id'
    graph = mock.MagicMock()
    graph.id = 'graph-id'
    graph.kumo_project_id = 'project-id'
    graph.snapshot.side_effect = AssertionError('unexpected graph snapshot')
    prediction_table = mock.MagicMock()
    prediction_table.job_id = 'prediction-table-job'
    prediction_table.kumo_project_id = 'project-id'
    response = mock.MagicMock()
    response.ok = True
    response.empty.return_value = True
    mock_backend.name = 'local'
    mock_backend.prediction_table.snapshot_id.return_value = 'graph-id@data'
    mock_backend.graph.get_snapshot.return_value = SimpleNamespace(
        graph_name='graph-id')
    mock_backend.prediction.maybe_create.return_value = (
        'batch-prediction-job', response)

    job = trainer.predict(
        graph=graph,
        prediction_table=prediction_table,
        non_blocking=True,
    )

    assert isinstance(job, BatchPredictionJob)
    request = mock_backend.prediction.maybe_create.call_args.args[0]
    assert request.graph_snapshot_id == 'graph-id@data'
    graph.snapshot.assert_not_called()


@pytest.mark.parametrize(
    'output_name,expected',
    [
        ('deep/many/layers/bp', 'deep/many/layers/bp_predictions'),
        ('s3://output-bucket/root/deep/many/layers/bp',
         'deep/many/layers/bp_predictions'),
    ],
)
def test_build_rooted_s3_prediction_output_uses_relative_path(
        output_name, expected):
    connector = object.__new__(S3Connector)
    connector._id = 'connector-s3'
    connector.any_uri = False
    connector.root_dir = 's3://output-bucket/root'
    output = build_prediction_output_config(
        'predictions',
        connector,
        output_name,
        [],
        OutputConfig(output_types={'predictions'},
                     output_table_name=output_name),
    )

    assert isinstance(output, S3PredictionOutput)
    assert output.connector_id == 'connector-s3'
    assert output.file_path == expected


@pytest.mark.parametrize(
    ('connector_type', 'root_dir', 'output_name', 'artifact', 'wire_type',
     'expected'),
    [
        (GCSConnector, 'gs://bucket/root', 'deep/bp', 'predictions',
         GCSPredictionOutput, 'deep/bp_predictions'),
        (GCSConnector, 'gs://bucket/root', 'gs://bucket/root/deep/bp',
         'predictions', GCSPredictionOutput, 'deep/bp_predictions'),
        (
            ADLSConnector,
            'abfss://curated@lake.dfs.core.windows.net/root',
            ('abfss://curated@lake.dfs.core.windows.net/root/deep/bp'),
            'predictions',
            ADLSPredictionOutput,
            'deep/bp_predictions',
        ),
        (GCSConnector, 'gs://bucket/root', 'deep/bp.csv', 'predictions',
         GCSPredictionOutput, 'deep/bp_predictions.csv'),
        (ADLSConnector, 'abfss://curated@lake.dfs.core.windows.net/root',
         'deep/bp.csv', 'embeddings', ADLSPredictionOutput,
         'deep/bp_embeddings.csv'),
    ],
)
def test_gcs_adls_prediction_output_uses_relative_wire_coordinate(
        connector_type, root_dir, output_name, artifact, wire_type, expected):
    connector = object.__new__(connector_type)
    connector._id = f'connector-{connector_type.__name__.lower()}'
    connector.any_uri = False
    connector.root_dir = root_dir

    output = build_prediction_output_config(
        artifact, connector, output_name, [],
        OutputConfig(output_types={artifact}, output_table_name=output_name))

    assert isinstance(output, wire_type)
    assert output.file_path == expected


@pytest.mark.parametrize(('connector_type', 'output_name', 'destination'), [
    (S3Connector, 's3://output-bucket/deep/output',
     's3://output-bucket/deep/output_predictions'),
    (GCSConnector, 'gs://output-bucket/deep/output',
     'gs://output-bucket/deep/output_predictions'),
    (ADLSConnector, 'abfss://out@lake.dfs.core.windows.net/deep/output',
     'abfss://out@lake.dfs.core.windows.net/deep/output_predictions'),
])
def test_rootless_prediction_output_uses_full_path(connector_type, output_name,
                                                   destination):
    connector = object.__new__(connector_type)
    connector._id = 'connector-any-uri'
    connector.any_uri = True
    connector.root_dir = None

    output = build_prediction_output_config(
        'predictions', connector, output_name, [],
        OutputConfig(output_types={'predictions'},
                     output_table_name=output_name))

    assert cast(Any, output).file_path == destination


@pytest.mark.parametrize('connector_type',
                         [S3Connector, GCSConnector, ADLSConnector])
def test_rootless_prediction_output_requires_full_path(connector_type):
    connector = object.__new__(connector_type)
    connector._id = 'connector-any-uri'
    connector.any_uri = True
    connector.root_dir = None

    with pytest.raises(ValueError, match='URI must use'):
        build_prediction_output_config(
            'predictions', connector, 'relative/output', [],
            OutputConfig(output_types={'predictions'},
                         output_table_name='relative/output'))


def test_rootless_prediction_export_submits_full_path(
        disable_artifact_export_polling):
    connector = object.__new__(S3Connector)
    connector._id = 'connector-any-uri'
    connector._kumo_project_id = 'project-id'
    connector.any_uri = True
    connector.root_dir = None
    job = object.__new__(BatchPredictionJobResult)
    job.job_id = 'batch-prediction-job'
    job._kumo_project_id = 'project-id'
    state = mock.MagicMock()
    state.client.artifact_export_api.create.return_value = 'export-job'

    with mock.patch('kumoai_studio.global_state', state):
        job.export(
            OutputConfig(
                output_types={'predictions'},
                output_connector=connector,
                output_table_name='s3://output-bucket/deep/predictions',
            ))

    request = state.client.artifact_export_api.create.call_args.args[0]
    assert (request.prediction_output.file_path ==
            's3://output-bucket/deep/predictions_predictions')


@pytest.mark.parametrize(
    'request_id',
    [None, 'd4af0106-44dc-4c42-aa53-d622b49161bd'],
)
def test_export(request_id, disable_artifact_export_polling):
    job_id = "bp-job-123"
    connector = object.__new__(BigQueryConnector)
    connector._id = 'connector-bq'
    connector._kumo_project_id = 'project-id'
    output_config = OutputConfig(
        output_types={"predictions"},
        output_connector=connector,
        output_table_name="bp_table",
        output_metadata_fields=[MetadataField.JOB_TIMESTAMP],
    )

    with mock.patch("kumoai_studio.global_state",
                    new=mock.MagicMock()) as mock_global, mock.patch(
                        'kumoai_studio.trainer.job.uuid.uuid4',
                        return_value='8f14e45f-ea5e-4a40-a79b-3f55a4df7c4d',
                    ) as uuid4:
        # Patch global_state as a MagicMock, no autospec
        # Now define any needed chain of attributes
        # so that `global_state.client.artifact_export_api.create` works.
        mock_global.client.artifact_export_api.create.return_value = \
            "export-job-123"

        job_result = BatchPredictionJobResult(job_id, project_id='project-id')
        if request_id is None:
            export_job = job_result.export(output_config, non_blocking=True)
        else:
            export_job = job_result.export(
                output_config,
                non_blocking=True,
                request_id=request_id,
            )

        assert isinstance(export_job, ArtifactExportJob)
        assert export_job.id == "export-job-123"
        mock_global.client.artifact_export_api.create.assert_called_once()
        args, _ = mock_global.client.artifact_export_api.create.call_args
        assert args[0].job_id == 'bp-job-123'
        expected_request_id = ('8f14e45f-ea5e-4a40-a79b-3f55a4df7c4d'
                               if request_id is None else request_id)
        assert args[0].request_id == expected_request_id
        if request_id is None:
            uuid4.assert_called_once_with()
        else:
            uuid4.assert_not_called()
        assert isinstance(args[0].prediction_output, BigQueryPredictionOutput)
        output = args[0].prediction_output
        assert output.write_mode == WriteMode.OVERWRITE
        assert output.connector_id == 'connector-bq'
        assert output.artifact_type == PredictionArtifactType.PREDICTIONS
        assert output.table_name == 'bp_table_predictions'
        assert output.extra_fields[0] == 'JOB_TIMESTAMP'


def test_export_with_multiple_prediction_output():
    job_id = "bp-job-123"
    connector = object.__new__(BigQueryConnector)
    connector._id = 'connector-bq'
    connector._kumo_project_id = 'project-id'
    output_config = OutputConfig(
        output_types={"predictions", "embeddings"},
        output_connector=connector,
        output_table_name="bp_table",
        output_metadata_fields=[MetadataField.JOB_TIMESTAMP],
    )

    with pytest.raises(ValueError) as err:
        with mock.patch("kumoai_studio.global_state",
                        new=mock.MagicMock()) as mock_global:
            # Patch global_state as a MagicMock, no autospec
            # Now define any needed chain of attributes
            # so that `global_state.client.artifact_export_api.create` works.
            mock_global.client.artifact_export_api.create.return_value = \
                "export-job-123"

            job_result = BatchPredictionJobResult(job_id,
                                                  project_id='project-id')
            job_result.export(output_config, non_blocking=True)
    assert str(err.value) == (
        'Export one artifact per export() call. Use separate calls '
        'for predictions and embeddings.')


def test_job_search_snapshots_project_from_each_resource():
    api = mock.MagicMock()
    resource = mock.MagicMock(job_id='training-job',
                              project_id='project-owned')
    api.list.return_value = [resource]

    with mock.patch.object(TrainingJob, '_api', return_value=api):
        jobs = TrainingJob.search_by_tags({'team': 'forecasting'},
                                          project_id='project-owned')

    api.list.assert_called_once_with(limit=10,
                                     additional_tags={'team': 'forecasting'},
                                     project_id='project-owned')
    assert len(jobs) == 1
    assert jobs[0].id == 'training-job'
    assert jobs[0].kumo_project_id == 'project-owned'


def test_training_job_result_load_config(mock_backend):
    job_id = "training-job-456"
    mock_config = TrainingJobRequest(job_tags={}, pquery_id="pquery-123")

    mock_backend.training.get_config.return_value = mock_config
    job = TrainingJob(job_id, project_id='project-id')
    config = job.load_config()

    assert config == mock_config
    mock_backend.training.get_config.assert_called_once_with(
        job_id, project_id='project-id')


def test_batch_prediction_job_load_config(mock_backend):
    job_id = "bp-job-789"
    mock_config = BatchPredictionRequest(
        job_tags={}, model_training_job_id="training-job-456",
        predict_options=BatchPredictionOptions())

    mock_backend.prediction.get_config.return_value = mock_config
    job = BatchPredictionJob(job_id, project_id='project-id')
    config = job.load_config()

    assert config == mock_config
    mock_backend.prediction.get_config.assert_called_once_with(
        job_id, project_id='project-id')


@pytest.mark.parametrize(
    'request_id',
    [None, 'd4af0106-44dc-4c42-aa53-d622b49161bd'],
)
def test_export_model_non_blocking(request_id,
                                   disable_artifact_export_polling):
    connector = _s3_export_connector()
    config = ModelExportConfig(
        training_job_id='training-job-123',
        output_connector=connector,
        output_path='deep/many/layers/output',
    )

    with mock.patch('kumoai_studio.global_state',
                    new=mock.MagicMock()) as mock_global, mock.patch(
                        'kumoai_studio.artifact_export.job.uuid.uuid4',
                        return_value='8f14e45f-ea5e-4a40-a79b-3f55a4df7c4d',
                    ) as uuid4:
        mock_global.client.artifact_export_api.create.return_value = \
            'export-job-789'

        if request_id is None:
            result = export_model(config, non_blocking=True,
                                  project_id='project-id')
        else:
            result = export_model(config, non_blocking=True,
                                  request_id=request_id,
                                  project_id='project-id')

        assert isinstance(result, ArtifactExportJob)
        assert result.id == 'export-job-789'
        mock_global.client.artifact_export_api.create.assert_called_once()
        args, _ = mock_global.client.artifact_export_api.create.call_args
        req = args[0]
        assert isinstance(req, ArtifactExportRequest)
        assert req.job_id == 'training-job-123'
        expected_request_id = ('8f14e45f-ea5e-4a40-a79b-3f55a4df7c4d'
                               if request_id is None else request_id)
        assert req.request_id == expected_request_id
        if request_id is None:
            uuid4.assert_called_once_with()
        else:
            uuid4.assert_not_called()
        assert isinstance(req.model_output.output, S3ModelArtifactOutput)
        assert req.model_output.output.connector_id == 'connector-s3'
        assert req.model_output.output.path == 'deep/many/layers/output'


def _s3_export_connector(*, any_uri=False):
    connector = object.__new__(S3Connector)
    connector._id = 'connector-s3'
    connector._kumo_project_id = 'project-id'
    connector.any_uri = any_uri
    connector.root_dir = None if any_uri else 's3://bucket/root'
    return connector


def _gcs_export_connector(*, any_uri=False):
    connector = object.__new__(GCSConnector)
    connector._id = 'connector-gcs'
    connector._kumo_project_id = 'project-id'
    connector.any_uri = any_uri
    connector.root_dir = None if any_uri else 'gs://bucket/root'
    return connector


def _adls_export_connector(*, any_uri=False):
    connector = object.__new__(ADLSConnector)
    connector._id = 'connector-adls'
    connector._kumo_project_id = 'project-id'
    connector.any_uri = any_uri
    connector.root_dir = (None if any_uri else
                          'abfss://curated@lake.dfs.core.windows.net/root')
    return connector


@pytest.mark.parametrize(
    'connector,path,expected',
    [
        (
            _s3_export_connector(),
            's3://bucket/root/deep/many/layers/model',
            S3ModelArtifactOutput(
                connector_id='connector-s3',
                path='deep/many/layers/model',
            ),
        ),
        (
            _gcs_export_connector(),
            'gs://bucket/root/deep/many/layers/model',
            GCSModelArtifactOutput(
                connector_id='connector-gcs',
                path='deep/many/layers/model',
            ),
        ),
        (
            _adls_export_connector(),
            ('abfss://curated@lake.dfs.core.windows.net/'
             'root/deep/model'),
            ADLSModelArtifactOutput(
                connector_id='connector-adls',
                path='deep/model',
            ),
        ),
    ],
    ids=['s3', 'gcs', 'adls'],
)
def test_model_export_accepts_full_rooted_path(connector, path, expected):
    wire = ModelExportConfig(
        training_job_id='training-job',
        output_connector=connector,
        output_path=path,
    )._to_api_config()

    assert wire.output == expected


@pytest.mark.parametrize('connector,output_path,expected', [
    (
        _s3_export_connector(any_uri=True),
        's3://output-bucket/deep/model',
        S3ModelArtifactOutput(
            connector_id='connector-s3',
            path='s3://output-bucket/deep/model',
        ),
    ),
    (
        _gcs_export_connector(any_uri=True),
        'gs://output-bucket/deep/model',
        GCSModelArtifactOutput(
            connector_id='connector-gcs',
            path='gs://output-bucket/deep/model',
        ),
    ),
    (
        _adls_export_connector(any_uri=True),
        'abfss://models@lake.dfs.core.windows.net/deep/model',
        ADLSModelArtifactOutput(
            connector_id='connector-adls',
            path='abfss://models@lake.dfs.core.windows.net/deep/model',
        ),
    ),
], ids=['s3', 'gcs', 'adls'])
def test_rootless_model_output_uses_full_path(connector, output_path,
                                              expected):
    wire = ModelExportConfig(
        training_job_id='training-job',
        output_connector=connector,
        output_path=output_path,
    )._to_api_config()

    assert wire.output == expected


@pytest.mark.parametrize('connector', [
    _s3_export_connector(any_uri=True),
    _gcs_export_connector(any_uri=True),
    _adls_export_connector(any_uri=True),
], ids=['s3', 'gcs', 'adls'])
def test_rootless_model_output_requires_full_path_before_request(connector):
    state = mock.MagicMock()
    config = ModelExportConfig(
        training_job_id='training-job',
        output_connector=connector,
        output_path='relative/model',
    )
    with mock.patch('kumoai_studio.global_state', state), \
            pytest.raises(ValueError, match='URI must use'):
        export_model(config)
    state.client.artifact_export_api.create.assert_not_called()


def test_export_model_rejects_cross_project_before_network():
    state = mock.MagicMock()
    config = ModelExportConfig(
        training_job_id='training-job',
        output_connector=_s3_export_connector(),
        output_path='deep/model',
    )

    with mock.patch('kumoai_studio.global_state', state), \
            pytest.raises(ValueError, match='belongs to project'):
        export_model(config, project_id='other-project')

    state.client.artifact_export_api.create.assert_not_called()


def test_training_job_api_get_encoders_returns_none_on_404():
    mock_client = mock.MagicMock()
    mock_client._get.return_value = mock.MagicMock(
        status_code=404,
        text='not found',
        ok=False,
    )

    api = TrainingJobAPI(mock_client)

    assert api.get_encoders('training-job-123') is None


def test_training_job_api_get_encoders_raises_non_404_errors():
    mock_client = mock.MagicMock()
    mock_client._get.return_value = mock.MagicMock(
        status_code=500,
        text='internal error',
        ok=False,
    )

    api = TrainingJobAPI(mock_client)

    with pytest.raises(HTTPException, match='internal error'):
        api.get_encoders('training-job-123')


def test_training_holdout_data_uses_only_explicit_project():
    client = mock.MagicMock()
    client._get.return_value.ok = True
    client._get.return_value.text = 'https://safe-holdout-reference'
    api = TrainingJobAPI(client)

    assert api.holdout_data_url(
        'training-job',
        project_id='project-b') == ('https://safe-holdout-reference')
    client._get.assert_called_once_with(
        '/training_jobs/training-job/holdout',
        params={'project_id': 'project-b'},
    )


def test_generate_train_table_data_uses_only_explicit_project():
    client = mock.MagicMock()
    client._get.return_value.ok = True
    api = GenerateTrainTableJobAPI(client)

    with mock.patch('kumoai_studio.client.jobs.parse_response',
                    return_value=['https://table-part']) as parse:
        result = api.get_table_data('train-table-job', project_id='project-b')

    assert result == ['https://table-part']
    client._get.assert_called_once_with(
        '/gentraintable_jobs/train-table-job/table_data',
        params={'project_id': 'project-b'},
    )
    parse.assert_called_once()


def test_generate_prediction_table_data_uses_only_explicit_project():
    client = mock.MagicMock()
    client._get.return_value.ok = True
    api = GeneratePredictionTableJobAPI(client)

    with mock.patch('kumoai_studio.client.jobs.parse_response',
                    return_value=['https://prediction-part']) as parse:
        result = api.get_table_data('prediction-table-job',
                                    project_id='project-b')

    assert result == ['https://prediction-part']
    client._get.assert_called_once_with(
        '/genpredtable_jobs/prediction-table-job/table_data',
        params={'project_id': 'project-b'},
    )
    parse.assert_called_once()


@pytest.mark.parametrize(
    ('api_type', 'method_name', 'legacy_kwargs'),
    [
        (TrainingJobAPI, 'holdout_data_url', {
            'presigned': False
        }),
        (GenerateTrainTableJobAPI, 'get_table_data', {
            'presigned': False
        }),
        (GenerateTrainTableJobAPI, 'get_table_data', {
            'raw_path': True
        }),
        (GeneratePredictionTableJobAPI, 'get_table_data', {
            'presigned': False
        }),
    ],
)
def test_removed_job_data_query_controls_are_not_accepted(
        api_type, method_name, legacy_kwargs):
    api = api_type(mock.MagicMock())

    with pytest.raises(TypeError, match=next(iter(legacy_kwargs))):
        getattr(api, method_name)('job-id', **legacy_kwargs)


def test_training_result_holdout_uses_server_selected_safe_reference(
        mock_backend):
    result = TrainingJobResult('training-job', project_id='project-owned')
    mock_backend.training.holdout_data_url.return_value = (
        'https://safe-holdout-reference')

    assert result.holdout_url() == 'https://safe-holdout-reference'

    mock_backend.training.holdout_data_url.assert_called_once_with(
        'training-job', project_id='project-owned')


def test_generate_train_table_split_masks_uses_explicit_project():
    client = mock.MagicMock()
    client._get.return_value.ok = True
    api = GenerateTrainTableJobAPI(client)

    with mock.patch('kumoai_studio.client.jobs.parse_response',
                    return_value={}) as parse:
        assert api.get_split_masks('train-table-job',
                                   project_id='project-b') == {}

    client._get.assert_called_once_with(
        '/gentraintable_jobs/train-table-job/split_masks',
        params={'project_id': 'project-b'},
    )
    parse.assert_called_once()


# --- registry-addressed model export ---------------------------------------


def _snowflake_export_connector():
    from kumoai_studio.connector.snowflake_connector import SnowflakeConnector

    connector = object.__new__(SnowflakeConnector)
    connector._id = 'connector-sf'
    connector._kumo_project_id = 'project-id'
    return connector


def _databricks_export_connector():
    # yapf: disable
    from kumoai_studio.connector.databricks_connector import (
        DatabricksConnector,
    )

    # yapf: enable

    connector = object.__new__(DatabricksConnector)
    connector._id = 'connector-dbx'
    connector._kumo_project_id = 'project-id'
    return connector


def _registry_backend(supports=True):
    r"""A backend whose export area does or does not claim registry support."""
    backend = mock.MagicMock()
    backend.name = 'fake'
    backend.artifact_export.supports_model_registry = supports
    return backend


@pytest.mark.parametrize('connector,expected_type,expected_id', [
    (_snowflake_export_connector(), SnowflakeModelArtifactOutput,
     'connector-sf'),
    (_databricks_export_connector(), DatabricksModelArtifactOutput,
     'connector-dbx'),
], ids=['snowflake', 'databricks'])
def test_a_registry_export_carries_the_name_and_version(
        connector, expected_type, expected_id):
    wire = ModelExportConfig(
        training_job_id='training-job',
        output_connector=connector,
        output_location=ModelRegistryLocation('CHURN_GNN', version='V3',
                                              comment='q3'),
    )._to_api_config()

    assert wire.output == expected_type(connector_id=expected_id,
                                        model_name='CHURN_GNN',
                                        version_name='V3', comment='q3')


def test_an_unversioned_registry_export_leaves_the_version_to_the_registry():
    wire = ModelExportConfig(
        training_job_id='training-job',
        output_connector=_snowflake_export_connector(),
        output_location=ModelRegistryLocation('CHURN_GNN'),
    )._to_api_config()

    # Only the registry-addressed outputs carry a version; the union's
    # path-addressed members do not, so the type is named before it is read.
    assert isinstance(wire.output, SnowflakeModelArtifactOutput)
    assert wire.output.version_name is None


def test_local_registry_export_uses_the_typed_request_path(
        disable_artifact_export_polling):
    config = ModelExportConfig(
        training_job_id='training-job',
        output_connector=_snowflake_export_connector(),
        output_location=ModelRegistryLocation('CHURN_GNN', version='V3'),
    )
    backend = _registry_backend(supports=True)
    backend.artifact_export.create.return_value = 'export-job-1'

    with mock.patch('kumoai_studio.artifact_export.job.cloud_backend',
                    return_value=backend):
        export_model(
            config,
            non_blocking=True,
            request_id='11111111-1111-1111-1111-111111111111',
            project_id='project-id',
        )

    request = backend.artifact_export.create.call_args.args[0]
    assert request == ArtifactExportRequest(
        request_id='11111111-1111-1111-1111-111111111111',
        job_id='training-job',
        model_output=config._to_api_config(),
    )
    assert backend.artifact_export.create.call_args.kwargs == {
        'project_id': 'project-id'
    }
    backend.artifact_export.publish_model.assert_not_called()


def test_a_registry_connector_refuses_a_path():
    r"""The two are not interchangeable, so the error names the right field."""
    with pytest.raises(ValueError, match='output_location'):
        ModelExportConfig(
            training_job_id='training-job',
            output_connector=_snowflake_export_connector(),
            output_path='models/churn',
        )


def test_a_registry_connector_requires_a_location():
    with pytest.raises(ValueError, match='output_location'):
        ModelExportConfig(
            training_job_id='training-job',
            output_connector=_databricks_export_connector(),
        )


def test_an_object_store_connector_refuses_a_location():
    with pytest.raises(ValueError, match='output_path'):
        ModelExportConfig(
            training_job_id='training-job',
            output_connector=_s3_export_connector(),
            output_location=ModelRegistryLocation('CHURN_GNN'),
        )


def test_a_path_export_is_unchanged_by_the_new_field():
    r"""`output_path` gained a default; positional construction must not move."""
    positional = ModelExportConfig('training-job', _s3_export_connector(),
                                   'deep/model')

    assert positional.output_path == 'deep/model'
    assert positional.output_location is None
    assert not positional.is_registry
    assert positional._to_api_config().output == S3ModelArtifactOutput(
        connector_id='connector-s3', path='deep/model')


@pytest.mark.parametrize('model_name', ['', '  ', ' CHURN '])
def test_a_registry_location_refuses_a_blank_or_padded_name(model_name):
    with pytest.raises(ValueError, match='model_name'):
        ModelRegistryLocation(model_name)


def test_a_registry_location_refuses_a_blank_version():
    with pytest.raises(ValueError, match='version'):
        ModelRegistryLocation('CHURN_GNN', version='')


def test_a_backend_without_a_registry_refuses_before_the_request():
    r"""The Kumo service does not publish to a registry, so the caller is told
    that rather than handed a rejection from inside the request.
    """  # noqa: D209
    config = ModelExportConfig(
        training_job_id='training-job',
        output_connector=_snowflake_export_connector(),
        output_location=ModelRegistryLocation('CHURN_GNN'),
    )
    backend = _registry_backend(supports=False)

    with mock.patch('kumoai_studio.artifact_export.job.cloud_backend',
                    return_value=backend), \
            pytest.raises(NotImplementedError, match='cannot publish a model'):
        export_model(config, project_id='project-id')

    backend.artifact_export.create.assert_not_called()


def test_an_area_that_raises_on_unknown_attributes_does_not_claim_support():
    r"""Several areas turn an unknown attribute into a raiser, and a callable
    is truthy -- so the probe must ask for ``True``, not for truthiness.
    """  # noqa: D209

    class _Raising:
        def __getattr__(self, name):
            def raiser(*args, **kwargs):
                raise NotImplementedError(name)

            return raiser

    backend = mock.MagicMock()
    backend.name = 'raising'
    backend.artifact_export = _Raising()
    config = ModelExportConfig(
        training_job_id='training-job',
        output_connector=_snowflake_export_connector(),
        output_location=ModelRegistryLocation('CHURN_GNN'),
    )

    with mock.patch('kumoai_studio.artifact_export.job.cloud_backend',
                    return_value=backend), \
            pytest.raises(NotImplementedError, match='cannot publish a model'):
        export_model(config, project_id='project-id')


def test_a_path_export_is_not_gated_on_registry_support():
    r"""Every backend serves an object-store model export; only a registry one
    is gated.
    """  # noqa: D209
    config = ModelExportConfig(
        training_job_id='training-job',
        output_connector=_s3_export_connector(),
        output_path='deep/model',
    )
    backend = _registry_backend(supports=False)
    backend.artifact_export.create.return_value = 'export-job-1'

    with mock.patch('kumoai_studio.artifact_export.job.cloud_backend',
                    return_value=backend):
        export_model(config, non_blocking=True, project_id='project-id')

    backend.artifact_export.create.assert_called_once()


@pytest.mark.parametrize('connector_type,root', [
    (S3Connector, 's3://bucket/root'),
    (GCSConnector, 'gs://bucket/root'),
    (ADLSConnector, 'abfss://out@lake.dfs.core.windows.net/root'),
])
@pytest.mark.parametrize('rooted', [False, True])
def test_prediction_exact_paths(connector_type, root, rooted):
    connector = object.__new__(connector_type)
    connector._id = 'output-connector'
    connector.any_uri = not rooted
    connector.root_dir = root if rooted else None
    paths = {
        kind: f'{root}/{kind}/%70roject/'
        for kind in ('predictions', 'embeddings')
    }
    config = OutputConfig(output_connector=connector, output_paths=paths)
    assert config.output_paths == paths
    assert config.output_types == set(paths)
    for kind, path in paths.items():
        output = build_prediction_output_config(kind, connector,
                                                output_config=config)
        assert output.file_path == (f'{kind}/project/'
                                    if rooted else path.replace('%70', 'p'))


def test_exact_path_export_submits_without_suffix(
        disable_artifact_export_polling):
    connector = object.__new__(S3Connector)
    connector._id = 'output-connector'
    connector._kumo_project_id = 'project-id'
    connector.any_uri = True
    connector.root_dir = None
    job = object.__new__(BatchPredictionJobResult)
    job.job_id = 'batch-prediction-job'
    job._kumo_project_id = 'project-id'
    state = mock.MagicMock()
    state.client.artifact_export_api.create.return_value = 'export-job'
    path = 's3://bucket/predictions/project/'
    with mock.patch('kumoai_studio.global_state', state):
        job.export(
            OutputConfig(output_connector=connector,
                         output_paths={'predictions': path}))
    request = state.client.artifact_export_api.create.call_args.args[0]
    assert request.prediction_output.file_path == path


def test_predict_submits_both_exact_paths(mock_backend):
    connector = object.__new__(GCSConnector)
    connector._id = 'output-connector'
    connector._kumo_project_id = 'project-id'
    connector.any_uri = True
    connector.root_dir = None
    paths = {
        kind: f'gs://bucket/{kind}/project/'
        for kind in ('predictions', 'embeddings')
    }
    trainer = object.__new__(Trainer)
    trainer._training_job_id = 'training-job'
    trainer._kumo_project_id = 'project-id'
    graph = mock.MagicMock()
    graph.kumo_project_id = 'project-id'
    graph.snapshot.return_value = 'graph-id@version'
    table = mock.MagicMock()
    table.kumo_project_id = 'project-id'
    table.job_id = 'prediction-table-job'
    response = mock.MagicMock()
    response.ok = True
    response.empty.return_value = True
    mock_backend.prediction.maybe_create.return_value = ('prediction-job',
                                                         response)
    trainer.predict(
        graph=graph, prediction_table=table, non_blocking=True,
        output_config=OutputConfig(output_connector=connector,
                                   output_paths=paths))
    request = mock_backend.prediction.maybe_create.call_args.args[0]
    assert {
        item.artifact_type.value.lower(): item.file_path
        for item in request.outputs
    } == paths


@pytest.mark.parametrize('connector_type,root', [
    (S3Connector, 's3://bucket/root'),
    (GCSConnector, 'gs://bucket/root'),
    (ADLSConnector, 'abfss://out@lake.dfs.core.windows.net/root'),
])
@pytest.mark.parametrize('rooted', [False, True])
def test_exact_paths_reject_same_destination_aliases(connector_type, root,
                                                     rooted):
    connector = object.__new__(connector_type)
    connector.any_uri = not rooted
    connector.root_dir = root if rooted else None
    with pytest.raises(ValueError, match='distinct output paths'):
        OutputConfig(
            output_connector=connector, output_paths={
                'predictions':
                f'{root}/out.parquet',
                'embeddings':
                ('out.parquet' if rooted else f'{root}/%6fut.parquet'),
            })
