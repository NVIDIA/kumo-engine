# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prediction, training table and model export destinations."""

from dataclasses import dataclass, field
from typing import List, Mapping, Optional, cast

from kumo_core.sources.object_store_sessions import ObjectStoreSessions
from kumoapi.jobs import MetadataField, PredictionArtifactType, WriteMode
from kumoapi.plugin import ModelRegistryLocation, SourceConnector
from kumoapi.prediction_outputs import resolve_prediction_output_paths

from kumo.engine.connector.base import Connector

_FILE_SOURCES = frozenset({'S3', 'GCS', 'ADLS', 'LOCAL_FILE'})


@dataclass(frozen=True)
class PredictionOutputConfig:
    """Export via ``connector``.

    Use ``output_paths`` for exact predictions/embeddings destinations, or
    ``artifact_type`` and ``table_name`` for one legacy destination.
    ``extra_fields`` applies to every selected artifact, only when passed to
    ``Trainer.predict``; ``PredictionResult.export`` rejects it.
    """

    connector: Connector
    artifact_type: Optional[PredictionArtifactType] = None
    table_name: Optional[str] = None
    write_mode: WriteMode = WriteMode.OVERWRITE
    extra_fields: List[MetadataField] = field(default_factory=list)

    output_paths: Optional[Mapping[str, str]] = None

    def __post_init__(self) -> None:
        if self.output_paths is not None:
            if self.connector.source_type not in _FILE_SOURCES:
                raise ValueError(
                    'output_paths requires a file or object-store connector')
            paths = resolve_prediction_output_paths(
                self.output_paths, output_types=self.artifact_type,
                output_table_name=self.table_name)
            sink = self.connector.sink_backend
            if isinstance(sink, ObjectStoreSessions):
                # Reject aliases; retain caller paths.
                resolve_prediction_output_paths({
                    kind:
                    sink._resolved(cast(SourceConnector, self.connector),
                                   path)[1]
                    for kind, path in paths.items()
                })
            object.__setattr__(self, 'output_paths', paths)
        elif self.artifact_type is None or self.table_name is None:
            raise ValueError(
                'Specify output_paths or artifact_type and table_name')

    def destinations(self) -> dict[str, str]:
        if self.output_paths is not None:
            return dict(self.output_paths)
        assert self.artifact_type is not None and self.table_name is not None
        return {self.artifact_type.value.lower(): self.table_name}


@dataclass(frozen=True)
class TrainingTableOutputConfig:
    """Export a training table to ``table_name`` via ``connector``.

    File and object-store destinations are directories.
    """

    connector: Connector
    table_name: str
    write_mode: WriteMode = WriteMode.OVERWRITE

    def __post_init__(self) -> None:
        if (self.connector.source_type in _FILE_SOURCES
                and not self.table_name.endswith('/')):
            object.__setattr__(self, 'table_name', self.table_name + '/')


@dataclass(frozen=True)
class ModelOutputConfig:
    """Publish a model to exactly one ``path`` or registry ``location``."""

    connector: Connector
    path: Optional[str] = None
    location: Optional[ModelRegistryLocation] = None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.location is None):
            raise ValueError("Exactly one of path or location must be set.")
