# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Requested prediction artifacts must exist before export configuration."""
from types import SimpleNamespace

import pytest
from kumo_core.types import PredictionArtifactType

from kumo.train.artifact_manifest import PredictionSelectionRef
from rest.config.artifact_export import (
    ArtifactExportInput,
    DatabricksManagedVolumeStage,
)
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.prediction_output import (
    DatabricksPredictionOutput,
    PredictionStorageType,
    S3PredictionOutput,
)
from statedb.config.common import ConnectorID, ProjectID

CONNECTOR_ID = ConnectorID('connector_' + '1' * 32)
PROJECT_ID = ProjectID.generate()


def _s3_output(artifact_type: PredictionArtifactType) -> S3PredictionOutput:
    return S3PredictionOutput(
        storage_type=PredictionStorageType.S3,
        connector_id=CONNECTOR_ID,
        file_path=f's3://bucket/{artifact_type.value.lower()}/',
        artifact_type=artifact_type,
    )


def _job_input(outputs: list) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(outputs=outputs),
        output_connectors={
            CONNECTOR_ID:
            ConnectorExecutionRef(project_id=PROJECT_ID,
                                  connector_id=CONNECTOR_ID)
        },
    )


def test_missing_requested_embeddings_raises() -> None:
    job_input = _job_input([
        _s3_output(PredictionArtifactType.PREDICTIONS),
        _s3_output(PredictionArtifactType.EMBEDDINGS),
    ])
    # The producer returned only one of the two requested artifacts.
    artifact_paths = {
        PredictionArtifactType.PREDICTIONS: '/out/processed_predictions',
    }

    with pytest.raises(ValueError, match='missing requested artifact paths'):
        ArtifactExportInput.from_bp_input(job_input, artifact_paths)


def test_embeddings_kept_when_present() -> None:
    """When embeddings WERE produced (two-tower / shallow / hybrid), the
    EMBEDDINGS output stays in the export config -- byte-identical to before.
    """
    job_input = _job_input([
        _s3_output(PredictionArtifactType.PREDICTIONS),
        _s3_output(PredictionArtifactType.EMBEDDINGS),
    ])
    artifact_paths = {
        PredictionArtifactType.PREDICTIONS: '/out/processed_predictions',
        PredictionArtifactType.EMBEDDINGS: '/out/processed_embeddings',
    }

    export_input = ArtifactExportInput.from_bp_input(job_input, artifact_paths)

    assert PredictionArtifactType.PREDICTIONS in export_input.output_config
    assert PredictionArtifactType.EMBEDDINGS in export_input.output_config


def test_predictions_only_request_unaffected() -> None:
    """A predictions-only request is unchanged (no EMBEDDINGS to skip)."""
    job_input = _job_input([_s3_output(PredictionArtifactType.PREDICTIONS)])
    artifact_paths = {
        PredictionArtifactType.PREDICTIONS: '/out/processed_predictions',
    }

    export_input = ArtifactExportInput.from_bp_input(job_input, artifact_paths)

    assert list(
        export_input.output_config) == [PredictionArtifactType.PREDICTIONS]


def test_selected_inventory_stays_a_bounded_manifest_reference() -> None:
    artifact_type = PredictionArtifactType.PREDICTIONS
    job_input = _job_input([_s3_output(artifact_type)])
    selection = PredictionSelectionRef(
        pointer_path='s3://internal/job/_selected_artifacts.json',
        manifest_path='s3://internal/job/_selections/selection/manifest.json',
        manifest_sha256='a' * 64,
        manifest_size_bytes=100,
        job_id='bp-job',
        selection_id='selection',
        selection_attempt_id='temporal-00001',
    )

    export_input = ArtifactExportInput.from_bp_input(
        job_input,
        {},
        prediction_selection=selection,
    )

    assert export_input.prediction_selection == selection
    assert export_input.prediction_artifact_types == (artifact_type, )
    assert export_input.artifact_paths == {}


def test_embeddings_only_request_creates_no_prediction_export() -> None:
    job_input = _job_input([_s3_output(PredictionArtifactType.EMBEDDINGS)])
    artifact_paths = {
        PredictionArtifactType.EMBEDDINGS: '/out/processed_embeddings',
    }

    export_input = ArtifactExportInput.from_bp_input(job_input, artifact_paths)

    assert list(
        export_input.artifact_paths) == [PredictionArtifactType.EMBEDDINGS]
    assert list(
        export_input.output_config) == [PredictionArtifactType.EMBEDDINGS]


def test_embeddings_only_rejects_unrequested_prediction_path() -> None:
    job_input = _job_input([_s3_output(PredictionArtifactType.EMBEDDINGS)])

    with pytest.raises(ValueError, match='unrequested artifacts'):
        ArtifactExportInput.from_bp_input(
            job_input,
            {
                PredictionArtifactType.EMBEDDINGS: '/out/embeddings',
                PredictionArtifactType.PREDICTIONS: '/out/predictions',
            },
        )


def test_no_output_request_fails() -> None:
    with pytest.raises(ValueError, match='at least one requested output'):
        ArtifactExportInput.from_bp_input(_job_input([]), artifact_paths={})


def test_missing_predictions_still_raises() -> None:
    """A requested PREDICTIONS artifact must be present."""
    job_input = _job_input([_s3_output(PredictionArtifactType.PREDICTIONS)])
    # PREDICTIONS requested but absent from artifact_paths.
    with pytest.raises(ValueError, match='missing requested artifact paths'):
        ArtifactExportInput.from_bp_input(job_input, artifact_paths={})


def test_databricks_output_requires_direct_producer_stage() -> None:
    output = DatabricksPredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=str(CONNECTOR_ID),
        table_name='schema__kumo__predictions',
    )
    job_input = _job_input([output])

    with pytest.raises(ValueError, match='producer-uploaded managed-volume'):
        ArtifactExportInput.from_bp_input(job_input, artifact_paths={})

    stage_path = ('/Volumes/catalog/schema/_kumo_staging/artifact-export/job')
    stage = DatabricksManagedVolumeStage(
        path=stage_path,
        files=[f'{stage_path}/part-00000000.parquet'],
    )
    export = ArtifactExportInput.from_bp_input(
        job_input,
        artifact_paths={},
        databricks_stages={0: stage},
    )

    config = export.output_config[PredictionArtifactType.PREDICTIONS][0]
    assert config.managed_volume_stage == stage
    assert export.artifact_paths[
        PredictionArtifactType.PREDICTIONS] == stage_path
