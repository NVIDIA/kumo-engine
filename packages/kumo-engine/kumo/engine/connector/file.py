# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rooted filesystem access for the in-process SDK only."""

import errno
import os
import tempfile
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Literal, Union

from kumo_core.artifact_export import ArtifactDestinationConflict
from kumo_core.object_store_sink import ObjectStoreSinkBackend
from kumo_core.sources import layout
from kumo_core.sources.object_store import ObjectStoreSourceReader
from kumo_core.sources.object_store_backend import ObjectStoreSourceBackend
from kumo_core.types.enums import ConnectorType
from kumoapi.plugin import SourceConnector
from kumoapi.source_table import FileType, LocalFileSourceTable
from pyarrow.fs import LocalFileSystem

from kumo.engine._local_schema import FileConnectorConfig
from kumo.engine.connector.base import Connector


def _relative(name: str) -> str:
    if (not isinstance(name, str) or not name or name.startswith('/')
            or any(char in name for char in ('\\', ':', '?', '#', '\x00'))
            or any(part in ('', '.', '..')
                   for part in name.rstrip('/').split('/'))):
        raise ValueError(
            'Use a nonempty relative path below root_dir; '
            'absolute paths, URIs and traversal are not supported.')
    return name.rstrip('/')


def _inside(root: Path, name: str) -> Path:
    path = root / _relative(name)
    for candidate in (path, *path.parents):
        if candidate == root:
            break
        if candidate.is_symlink() and candidate.is_dir():
            raise ValueError(
                'Directory symlinks are not supported by FileConnector.')
    if not path.resolve().is_relative_to(root):
        raise ValueError(
            'Path escapes FileConnector root_dir through a symlink.')
    return path


def _disjoint(first: Path, second: Path) -> None:
    first, second = first.resolve(), second.resolve()
    if first.is_relative_to(second) or second.is_relative_to(first):
        raise ValueError('Source and destination directories must not overlap.')


def _check_tree(root: Path, path: Path) -> None:
    """Check directory links without following them during traversal.

    Directory symlinks are refused, including internal ones, so Arrow's
    recursive listing cannot follow a cycle. Regular file symlinks may point
    within the connector root. Callers must not concurrently replace links.
    """
    if not path.resolve().is_relative_to(root):
        raise ValueError(
            'Path escapes FileConnector root_dir through a symlink.')
    for directory, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            item = Path(directory) / name
            if item.is_symlink() and (name in dirs or
                                      not item.resolve().is_relative_to(root)):
                raise ValueError(
                    'Directory or escaping symlinks are not supported by FileConnector.'
                )


@dataclass
class _FileTableConfig:
    source_table: LocalFileSourceTable


class _FileReader(ObjectStoreSourceReader):
    def resolve(self, table):
        root = Path(self.root)
        name = _relative(table)
        for candidate in (name, name + '.csv', name + '.parquet'):
            _inside(root, candidate)
        _check_tree(root, root / name)
        return super().resolve(name)

    def tables(self):
        _check_tree(Path(self.root), Path(self.root))
        return super().tables()


class _FileSourceBackend(ObjectStoreSourceBackend):
    scheme = 'file'
    mode = None

    def _reader(self, connector: Union[Connector, SourceConnector], table_name):
        _relative(table_name)
        return self._discovery(connector), table_name

    def _discovery(self, connector: Union[Connector, SourceConnector]):
        config = connector.config
        if not isinstance(config, FileConnectorConfig):
            raise TypeError('Local files require a FileConnectorConfig.')
        return _FileReader(LocalFileSystem(), root=config.root_dir,
                           source_type=ConnectorType.FILE)

    def _root_for(self, connector):
        return None

    def get_table_config(self, connector: Union[Connector, SourceConnector],
                         table_name, project_id=None):
        reader, name = self._reader(connector, table_name)
        path, kind = reader.resolve(name)
        return _FileTableConfig(
            LocalFileSourceTable(path=str(Path(path).resolve()),
                                 file_type=FileType(kind.upper())))

    def prepare_ingest(self, connector, table_name, dest_dir, *args, **kwargs):
        _disjoint(Path(connector.config.root_dir), Path(dest_dir))
        return super().prepare_ingest(connector, table_name, dest_dir, *args,
                                      **kwargs)

    def get_download_urls(self, connector: Union[Connector, SourceConnector],
                          table_name, project_id=None):
        reader, name = self._reader(connector, table_name)
        return [Path(path).as_uri() for path in reader.object_paths(name)]


def _not_a_directory(destination: str) -> NotADirectoryError:
    return NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR),
                              destination)


def _probe_directory(root: Path, destination: str) -> None:
    while not root.is_dir():
        if root.exists() or root.is_symlink():
            raise _not_a_directory(destination)
        root = root.parent
    with tempfile.TemporaryDirectory(prefix='.kumo-write-probe-',
                                     dir=root) as probe:
        (Path(probe) / 'probe').write_bytes(b'probe')


class _FileSinkBackend(ObjectStoreSinkBackend):
    scheme = 'file'
    provider = 'local filesystem'

    def _destination(self, connector: Union[Connector, SourceConnector],
                     coordinate):
        config = connector.config
        if not isinstance(config, FileConnectorConfig):
            raise TypeError('Local files require a FileConnectorConfig.')
        root = Path(config.root_dir)
        path = _inside(root, coordinate)
        _check_tree(root, path)
        return LocalFileSystem(), str(path) + ('/' if coordinate.endswith('/')
                                               else '')

    def preflight_write_files(self, connector, table_name, write_mode=None):
        del write_mode
        _, destination = self._destination(connector, table_name)
        path = Path(destination)
        kind = layout.file_type(destination.rstrip('/')) or layout.PARQUET
        target = None
        roots = [path]
        if not destination.endswith('/'):
            target = path
            if not destination.casefold().endswith(f'.{kind}'):
                target = Path(f'{destination}.{kind}')
            _check_tree(Path(connector.config.root_dir), target)
            # An existing target is overwritten in place, which needs write
            # access to the file and not to its directory.
            roots = ([] if target.is_file() else
                     [target.parent]) + ([path] if path.is_dir() else [])
        try:
            if target is None and path.is_file():
                raise _not_a_directory(destination)
            if target is None and path.is_dir() and any(path.iterdir()):
                # Each prediction run exports under a new identity, so
                # publication never accepts an occupied directory.
                raise ArtifactDestinationConflict(
                    f"Output directory '{destination}' already contains data "
                    'from another export. Choose an empty destination path.')
            for root in roots:
                _probe_directory(root, destination)
            if target is not None and target.is_file():
                os.close(os.open(target, os.O_WRONLY))
        except OSError as error:
            raise OSError(error.errno, error.strerror, destination) from error

    def write_table(self, connector, table_name, frame, *args, **kwargs):
        _relative(table_name)
        # Validate the *final* filename too: the shared writer adds .parquet.
        if not table_name.endswith('/') and not layout.file_type(table_name):
            table_name += '.parquet'
        uri = super().write_table(connector, table_name, frame, *args, **kwargs)
        return Path(uri.removeprefix('file://')).as_uri() + (
            '/' if uri.endswith('/') else '')

    def write_files(self, connector, table_name, files, *args, **kwargs):
        # The shared exporter adds a suffix only for a single-file output.
        if len(files) == 1 and not table_name.endswith('/'):
            kind = layout.file_type(table_name) or layout.PARQUET
            if not table_name.casefold().endswith(f'.{kind}'):
                table_name += f'.{kind}'
        return super().write_files(connector, table_name, files, *args,
                                   **kwargs)

    def publish_model(self, connector, bundle_dir, *, path=None, **kwargs):
        source = Path(bundle_dir).resolve()
        if not source.is_dir() or not any(item.is_file()
                                          for item in source.rglob('*')):
            raise ValueError(
                'Publishing requires an existing nonempty bundle directory.')
        if path:
            _, destination = self._destination(connector, path)
            target = Path(destination)
            _disjoint(source, target)
            if target.exists() and (not target.is_dir()
                                    or any(target.iterdir())):
                raise ValueError('Publish to an empty destination directory.')
            # A bundle must be a self-contained independent copy.
            if any(item.is_symlink() for item in source.rglob('*')):
                raise ValueError('Model bundles must not contain symlinks.')
        published = super().publish_model(connector, str(source), path=path,
                                          **kwargs)
        return replace(published, uri=Path(published.version).as_uri())

    def fetch_model(self, connector, local_dir, *, path=None, **kwargs):
        if path:
            _disjoint(Path(self._destination(connector, path)[1]),
                      Path(local_dir))
        return super().fetch_model(connector, local_dir, path=path, **kwargs)


class FileConnector(Connector):
    """Read local CSV/Parquet tables and write local artifacts.

    For usage and path rules, see
    https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/connect-your-data-linux.

    Source versions use paths, sizes and modification times, not content
    hashes: keep sources unchanged during reads, and do not restore
    timestamps after edits. Source identities use canonical paths and are
    not portable across machines.

    Write artifacts to a separate root. Single-file outputs overwrite,
    partitioned outputs need an empty directory, and append is unsupported.
    Model publication copies a bundle to an empty destination and rejects
    symlinks and overlap.
    """
    def __init__(self, *, root_dir: str):
        super().__init__()
        if not root_dir or '://' in str(root_dir):
            raise ValueError(
                'root_dir must be a nonempty filesystem directory, not a URI.')
        root = Path(root_dir).expanduser().resolve()
        if root.exists() and not root.is_dir():
            raise ValueError('root_dir must be a directory.')
        self._config = FileConnectorConfig(str(root))

    @property
    def source_type(self) -> Literal['LOCAL_FILE']:
        return 'LOCAL_FILE'

    @property
    def config(self) -> FileConnectorConfig:
        return self._config

    @cached_property
    def source_backend(self):
        return _FileSourceBackend()

    @cached_property
    def sink_backend(self):
        return _FileSinkBackend()

    def source_table_for(self, table_name: str) -> LocalFileSourceTable:
        return self.source_backend.get_table_config(self,
                                                    table_name).source_table

    def validate_scratch_dir(self, scratch_dir: Path) -> None:
        _disjoint(Path(self.config.root_dir), scratch_dir)
