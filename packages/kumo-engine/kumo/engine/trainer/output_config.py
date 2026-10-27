# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prediction, training table and model export destinations."""

from dataclasses import dataclass, field
from typing import List, Mapping, Optional, cast

from kumo_core.sources.object_store_sessions import ObjectStoreSessions
from kumoapi.jobs import MetadataField, PredictionArtifactType, WriteMode
from kumoapi.plugin import (
    ModelRegistryLocation,
    SourceConnector,
    check_model_alias,
)
from kumoapi.prediction_outputs import resolve_prediction_output_paths

from kumo.engine.connector import (
    file_store,
    is_file_destination,
    model_sink,
)
from kumo.engine.connector.base import Connector


@dataclass(frozen=True)
class PredictionOutputConfig:
    """Export via ``connector``.

    ``path`` names where predictions go: a directory or file on a file
    destination (local, S3, GCS, a volume or stage), else a table. Use
    ``output_paths`` for exact predictions/embeddings destinations, or
    ``artifact_type`` and ``table_name`` for one legacy destination.
    ``extra_fields`` applies to every selected artifact, only when passed to
    ``Trainer.predict``; ``PredictionResult.export`` rejects it.
    A table is replaced by default; ``WriteMode.APPEND`` adds rows with
    ``PREDICTION_ID`` and ``BUNDLE_ID`` columns. A file destination is
    written at exactly ``path``; ``WriteMode.OVERWRITE`` replaces what it
    holds.
    """

    connector: Connector
    artifact_type: Optional[PredictionArtifactType] = None
    table_name: Optional[str] = None
    write_mode: Optional[WriteMode] = None
    extra_fields: List[MetadataField] = field(default_factory=list)

    output_paths: Optional[Mapping[str, str]] = None
    path: Optional[str] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.path is not None:
            if (self.output_paths is not None
                    or self.artifact_type is not None
                    or self.table_name is not None):
                raise ValueError('path replaces output_paths, artifact_type '
                                 'and table_name; pass only one form.')
            if is_file_destination(self.connector, self.path):
                object.__setattr__(self, 'output_paths',
                                   {'predictions': self.path})
            else:
                object.__setattr__(self, 'artifact_type',
                                   PredictionArtifactType.PREDICTIONS)
                object.__setattr__(self, 'table_name', self.path)
        if self.output_paths is not None:
            paths = resolve_prediction_output_paths(
                self.output_paths, output_types=self.artifact_type,
                output_table_name=self.table_name)
            if not all(
                    is_file_destination(self.connector, path)
                    for path in paths.values()):
                raise ValueError(
                    'output_paths requires a file or object-store connector '
                    '(or a volume or stage path it holds)')
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
                'Specify path, output_paths, or artifact_type and table_name')
        if self.write_mode is None and not all(
                is_file_destination(self.connector, path)
                for path in self.destinations().values()):
            object.__setattr__(self, 'write_mode', WriteMode.OVERWRITE)

    def destinations(self) -> dict[str, str]:
        if self.output_paths is not None:
            return dict(self.output_paths)
        assert self.artifact_type is not None and self.table_name is not None
        return {self.artifact_type.value.lower(): self.table_name}

    def writer(self, destination: str):
        sink = model_sink(self.connector, destination)
        if not callable(getattr(sink, 'write_files', None)):
            raise NotImplementedError(
                'Prediction export requires FileArtifactSink.write_files; '
                'this sink only supports in-memory table output')
        return sink

    def _write(self, destination: str, files: List[str], **kwargs) -> str:
        r"""Write ``files`` to ``destination``; ``OVERWRITE`` first empties
        a directory destination unless it holds this export complete."""
        from kumo_core.object_store_sink import holds_export

        store = file_store(self.connector, destination)
        export_id = kwargs.get('export_id')
        if (self.write_mode is WriteMode.OVERWRITE and store is not None
                and (destination.endswith('/') or len(files) > 1)
                and not (export_id and holds_export(*store, export_id))):
            store[0].delete_dir_contents(store[1], missing_dir_ok=True)
        return self.writer(destination).write_files(
            self.connector, destination, files,
            self.write_mode or WriteMode.OVERWRITE, **kwargs)

    def _id_columns(self, destination: str, prediction_id: Optional[str],
                    bundle_id: Optional[str]) -> dict[str, str]:
        if (self.write_mode is not WriteMode.APPEND
                or is_file_destination(self.connector, destination)):
            return {}
        return {
            name: value
            for name, value in (('PREDICTION_ID', prediction_id),
                                ('BUNDLE_ID', bundle_id)) if value
        }


@dataclass(frozen=True)
class TrainingTableOutputConfig:
    """Export a training table to ``table_name`` via ``connector``.

    File and object-store destinations are directories.
    """

    connector: Connector
    table_name: str
    write_mode: WriteMode = WriteMode.OVERWRITE

    def __post_init__(self) -> None:
        if (not self.table_name.endswith('/')
                and is_file_destination(self.connector, self.table_name)):
            object.__setattr__(self, 'table_name', self.table_name + '/')


@dataclass(frozen=True)
class SnapshotOutputConfig:
    """Export a graph snapshot under ``path`` via ``connector``, a file
    destination (`GraphSnapshot.export`)."""

    connector: Connector
    path: str


#: Publication metadata a registry may act on: the serving snapshot, and
#: each serving option under ``kumo.serving.<option>``.
SERVING_SNAPSHOT_KEY = 'kumo.serving_snapshot'
SERVING_OPTION_PREFIX = 'kumo.serving.'
SERVING_OPTIONS = ('cuda_version', 'python_version', 'function_type',
                   'pip_repository')


@dataclass(frozen=True)
class ModelOutputConfig:
    """Publish a model to exactly one ``path`` or registry ``location``.

    For a Snowflake Model Registry ``location``, ``serving_snapshot`` is the
    exported snapshot (an ``@DB.SCHEMA.STAGE/...`` path) the version scores
    against in Snowflake, and ``serving_options`` overrides how it runs:
    ``cuda_version`` (default ``12.6``), ``python_version`` (the registering
    Python, which it must equal), ``function_type`` (``TABLE_FUNCTION``), and
    ``pip_repository``, a Snowflake pip artifact repository the version's
    third-party requirements install from instead of PyPI.
    """

    connector: Connector
    path: Optional[str] = None
    location: Optional[ModelRegistryLocation] = None
    serving_snapshot: Optional[str] = None
    serving_options: Optional[Mapping[str, str]] = None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.location is None):
            raise ValueError("Exactly one of path or location must be set.")
        unknown = sorted(set(self.serving_options or {}) -
                         set(SERVING_OPTIONS))
        if unknown:
            raise ValueError(f"Unknown serving_options {unknown}; expected "
                             f"any of {list(SERVING_OPTIONS)}.")


def check_model_location(config: ModelOutputConfig) -> None:
    """Raises ``ValueError`` unless ``config.location`` holds an alias and
    a model name its registry accepts."""
    location = config.location
    if location is None:
        return
    if location.alias:
        check_model_alias(location.alias)
    check = getattr(config.connector.sink_backend, 'check_model_location',
                    None)
    if callable(check):
        check(config.connector, location)
