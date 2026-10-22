# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import posixpath
import shutil
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Mapping, Optional, Tuple

import pandas as pd
from kumoapi.jobs import WriteMode
from kumoapi.plugin import (
    ModelRegistryLocation,
    PublishedModel,
    SourceConnector,
)
from pyarrow.fs import FileSystem

_CHUNK = 8 * 1024 * 1024
_OBJECT_STORES = frozenset({'s3', 'gcs', 'abfs'})


class ObjectStoreSinkBackend(ABC):
    scheme: str
    provider: str

    @abstractmethod
    def _destination(self, connector: SourceConnector,
                     coordinate: str) -> Tuple[FileSystem, str]:
        """Return the authenticated filesystem and validated bucket/key path."""
        raise NotImplementedError

    def _uri(self, path: str) -> str:
        return f'{self.scheme}://{path}'

    def write_table(
        self,
        connector: SourceConnector,
        table_name: str,
        frame: pd.DataFrame,
        write_mode: WriteMode = WriteMode.OVERWRITE,
        project_id: Optional[str] = None,
        *,
        export_id: Optional[str] = None,
    ) -> str:
        """Write a frame as one object, or parts when the path ends in a slash.

        Streaming output is not transactional: Arrow closes its output stream
        after serialization errors, which can publish partial data.
        """
        del project_id
        if write_mode is not WriteMode.OVERWRITE:
            raise NotImplementedError(
                f"The local backend cannot write {write_mode} to an object "
                f"store yet (artifact_export area): the hosted connector "
                f"has no append for a file destination either; write to a "
                f"new path, or export to a warehouse")

        filesystem, path = self._destination(connector, table_name)
        return self._write(filesystem, frame, path, export_id)

    def write_files(self, connector, table_name, files,
                    write_mode=WriteMode.OVERWRITE, project_id=None, *,
                    export_id=None, extra_columns=None):
        """Export selected Parquet without collecting their rows."""
        del project_id
        if write_mode is not WriteMode.OVERWRITE:
            raise NotImplementedError(
                'Object-store artifact output does not support APPEND')
        from kumo_core.artifact_export import export_selected_files
        from kumo_core.sources import layout
        filesystem, path = self._destination(connector, table_name)
        kind = layout.file_type(path.rstrip('/')) or layout.PARQUET
        storage = _ArrowExportFiles(
            filesystem, extra_columns, upload=partial(self._upload_file,
                                                      filesystem))
        _, output = export_selected_files(files, path, fs=storage,
                                          identity=export_id or str(files),
                                          file_type=kind,
                                          convert_csv=storage.convert_csv)
        return self._uri(output)

    def publish_model(
        self,
        connector: SourceConnector,
        bundle_dir: str,
        *,
        path: Optional[str] = None,
        location: Optional[ModelRegistryLocation] = None,
        metadata: Mapping[str, str] = {},
        metrics: Mapping[str, float] = {},
        project_id: Optional[str] = None,
    ) -> PublishedModel:
        """Upload a bundle, preserving relative paths and existing remote files."""
        del metrics, project_id
        if location is not None:
            raise NotImplementedError(
                f"The local backend does not publish to a {self.provider} "
                f"model registry (artifact_export area): an object store "
                f"has no version catalog; give the export config an "
                f"output_path instead of an output_location")
        if not path:
            raise ValueError(
                f"Publishing a model to {self.provider} needs a destination "
                f"path. Set output_path on the ModelExportConfig.")

        filesystem, destination = self._destination(connector, path)
        destination = destination.rstrip('/')
        root = Path(bundle_dir)
        for source in sorted(item for item in root.rglob('*')
                             if item.is_file()):
            target = f'{destination}/{source.relative_to(root).as_posix()}'
            _prepared(filesystem, posixpath.dirname(target))
            self._upload_file(filesystem, str(source), target)
        return PublishedModel(
            name=posixpath.basename(destination),
            version=destination,
            uri=self._uri(destination),
            metadata=dict(metadata),
        )

    def _upload_file(self, filesystem, source, destination):
        """Providers may transfer an already prepared local file directly."""
        with open(source, 'rb') as reader, \
                filesystem.open_output_stream(destination) as writer:
            shutil.copyfileobj(reader, writer, _CHUNK)

    def _download_file(self, filesystem, source, destination):
        """Providers may transfer a model file without the streaming fallback."""
        with filesystem.open_input_stream(source) as reader, \
                open(destination, 'wb') as writer:
            shutil.copyfileobj(reader, writer, _CHUNK)

    def fetch_model(
        self,
        connector: SourceConnector,
        local_dir: str,
        *,
        path: Optional[str] = None,
        location: Optional[ModelRegistryLocation] = None,
        project_id: Optional[str] = None,
    ) -> PublishedModel:
        """Replace the local directory with the published bundle."""
        del project_id
        from kumo_core.sources.layout import confined
        from pyarrow.fs import FileSelector, FileType

        if location is not None:
            raise NotImplementedError(
                f"The local backend does not import a model by name from "
                f"{self.provider} (artifact_export area): an object store "
                f"has no version catalog; name the path the bundle was "
                f"published to")
        if not path:
            raise ValueError(
                f"Importing a model from {self.provider} needs the path it "
                f"was published to. Reuse the output_path from the export "
                f"config.")

        filesystem, source = self._destination(connector, path)
        source = source.rstrip('/')
        destination = Path(local_dir)
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)

        found = False
        selector = FileSelector(source, recursive=True, allow_not_found=True)
        for info in filesystem.get_file_info(selector):
            if info.type is not FileType.File:
                continue
            found = True
            # Remote keys are untrusted; each relative path must stay inside
            # the local bundle, even when it contains parent components.
            target = destination / confined(info.path[len(source) + 1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            self._download_file(filesystem, info.path, str(target))
        if not found:
            raise ValueError(
                f"Nothing is published at {self.scheme}://{source}. Check the "
                f"path the model was exported to.")
        return PublishedModel(name=posixpath.basename(source), version=source,
                              uri=self._uri(source))

    def _write(self, filesystem: FileSystem, frame: pd.DataFrame, path: str,
               export_id: Optional[str] = None) -> str:
        from kumo_core.artifact_export import EXPORT_MARKER_NAME
        from kumo_core.sources import layout

        partitioned = path.endswith('/')
        kind = layout.file_type(path.rstrip('/')) or layout.PARQUET
        if partitioned:
            directory = path.rstrip('/')
            marker = _marker(export_id)
            occupied = self._occupied(filesystem, directory, marker)
            _prepared(filesystem, directory)
            if marker is not None and not occupied:
                with filesystem.open_output_stream(
                        f'{directory}/{EXPORT_MARKER_NAME}') as writer:
                    writer.write(marker.encode('utf-8'))
            _write_one(filesystem, frame, f'{directory}/part-00000.{kind}',
                       kind)
            with filesystem.open_output_stream(f'{directory}/_SUCCESS'):
                pass
            return self._uri(f'{directory}/')
        if not path.casefold().endswith(f'.{kind}'):
            path += f'.{kind}'
        _prepared(filesystem, posixpath.dirname(path))
        _write_one(filesystem, frame, path, kind)
        return self._uri(path)

    def _occupied(self, filesystem: FileSystem, directory: str,
                  marker: Optional[str]) -> bool:
        from kumo_core.artifact_export import EXPORT_MARKER_NAME
        from pyarrow.fs import FileSelector, FileType

        # A partitioned export can contain only directories here. Writing a
        # new part beside them would combine two exports when read as a dataset.
        # A matching marker permits retries of the same export.
        selector = FileSelector(directory, recursive=False,
                                allow_not_found=True)
        if not filesystem.get_file_info(selector):
            return False
        path = f'{directory}/{EXPORT_MARKER_NAME}'
        if marker is not None and filesystem.get_file_info(
                path).type is FileType.File:
            with filesystem.open_input_stream(path) as reader:
                if reader.readall().decode('utf-8') == marker:
                    return True
        raise ValueError(
            f"Output directory '{self.scheme}://{directory}/' already "
            f"contains data from another export. Choose an empty destination "
            f"path.")


def _write_one(filesystem: FileSystem, frame: pd.DataFrame, path: str,
               kind: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kumo_core.artifact_export import flatten_arrays_for_csv
    from kumo_core.sources import layout
    from pyarrow import csv

    # The frame may have a non-default index; never export it as a data column.
    if kind == layout.CSV:
        table = pa.Table.from_pandas(flatten_arrays_for_csv(frame),
                                     preserve_index=False)
        with filesystem.open_output_stream(path) as writer:
            csv.write_csv(table, writer)
        return
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path,
                   filesystem=filesystem)


def _prepared(filesystem: FileSystem, directory: str) -> None:
    # Object stores need no parents; create_dir would add visible marker objects.
    # Use type_name so Arrow builds without GCS support can still use S3.
    if filesystem.type_name in _OBJECT_STORES:
        return
    from pyarrow.fs import SubTreeFileSystem
    if isinstance(filesystem, SubTreeFileSystem):
        if filesystem.base_fs.type_name in _OBJECT_STORES:
            return
    if directory:
        filesystem.create_dir(directory, recursive=True)


def _marker(export_id: Optional[str]) -> Optional[str]:
    from kumo_core.artifact_export import export_marker

    if not export_id:
        return None
    return export_marker(export_id)


class _ArrowExportFiles:
    """Adapt authenticated Arrow output to the shared hosted export operation."""
    def __init__(self, filesystem, extra_columns, upload=None):
        self.filesystem = filesystem
        self.extra_columns = extra_columns
        self.upload = upload

    def exists(self, path):
        from pyarrow.fs import FileType
        return self.filesystem.get_file_info(path).type is not FileType.NotFound

    def isdir(self, path):
        from pyarrow.fs import FileType
        return self.filesystem.get_file_info(path).type is FileType.Directory

    def listdir(self, path, max_items=None):
        from pyarrow.fs import FileSelector
        entries = self.filesystem.get_file_info(
            FileSelector(path, allow_not_found=True))
        return [entry.path for entry in entries[:max_items]]

    def read_text(self, path):
        with self.filesystem.open_input_stream(path) as reader:
            return reader.readall().decode('utf-8')

    def write_text(self, path, value):
        with self.filesystem.open_output_stream(path) as writer:
            writer.write(value.encode('utf-8'))

    def makedirs(self, path, exist_ok=True):
        _prepared(self.filesystem, path)

    def touch(self, path):
        with self.filesystem.open_output_stream(path):
            pass

    def rm(self, path):
        self.filesystem.delete_file(path)

    def filesize(self, path):
        return self.filesystem.get_file_info(path).size

    def dirsize(self, path):
        from pyarrow.fs import FileSelector, FileType
        return sum(info.size for info in self.filesystem.get_file_info(
            FileSelector(path, recursive=True)) if info.type is FileType.File)

    def copy_file(self, source, destination, overwrite=True):
        from kumo_core.artifact_export import prepared_artifact_file

        # CSV conversion already applied this destination's metadata.
        columns = self.extra_columns if source.endswith('.parquet') else None
        with prepared_artifact_file(source, columns) as prepared:
            if self.upload is not None:
                self.upload(prepared, destination)
                return
            with open(prepared,
                      'rb') as reader, self.filesystem.open_output_stream(
                          destination) as writer:
                shutil.copyfileobj(reader, writer, _CHUNK)

    def copy_files_threaded(self, pairs, overwrite, max_workers):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(
                executor.map(
                    lambda pair: self.copy_file(*pair, overwrite=overwrite),
                    pairs))

    def convert_csv(self, source, destination):
        from kumo_core.artifact_export import (
            parquet_to_csv,
            prepared_artifact_file,
        )
        with prepared_artifact_file(source, self.extra_columns) as prepared:
            parquet_to_csv(prepared, destination)
