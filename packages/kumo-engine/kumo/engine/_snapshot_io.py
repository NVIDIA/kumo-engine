# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import logging
import os
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pyarrow.fs import FileSelector, FileSystem, FileType

_FORMAT = 1
_MANIFEST = 'kumo_snapshot.json'
_INDEX_MANIFEST = '_MANIFEST.json'
_SUCCESS = '_SUCCESS'
_RANGE_BYTES = 64 << 20
_CHUNK = 8 << 20
_ATTEMPTS = 3
_KEY = re.compile('[0-9a-f]{64}')
_PROGRESS_SECONDS = 30

logger = logging.getLogger(__name__)


def _location(connector: Any, path: str) -> Tuple[FileSystem, str]:
    from kumo_core.object_store_sink import ObjectStoreSinkBackend

    try:
        sink = connector.sink_backend
    except Exception:
        sink = None
    if not isinstance(sink, ObjectStoreSinkBackend):
        raise NotImplementedError(
            f'{type(connector).__name__} cannot hold a snapshot export; use '
            'an S3Connector, GCSConnector or a FileConnector on a volume.')
    return sink.filesystem(connector, path, parallel_uploads=True)


def _files(filesystem: FileSystem) -> Any:
    from kumo_core.object_store_sink import _ArrowExportFiles

    return _ArrowExportFiles(filesystem, None)


def _check_key(graph_key: Any, data_version: Any, snapshot_id: Any) -> None:
    from kumo.engine.graph import _snapshot_id

    if not (isinstance(graph_key, str) and isinstance(data_version, str)
            and _KEY.fullmatch(graph_key) and _KEY.fullmatch(data_version)
            and snapshot_id == _snapshot_id(graph_key, data_version)):
        raise ValueError(f'{snapshot_id!r} is not a snapshot id.')


class _Client:
    def __init__(self, connector: Any, path: str) -> None:
        self._make = lambda: _location(connector, path)[0]
        self.current = self._make()
        self._lock = threading.Lock()

    def renew(self, stale: FileSystem) -> None:
        with self._lock:
            if self.current is stale:
                self.current = self._make()


class _Progress:
    def __init__(self, action: str, path: str, total: int) -> None:
        self.action, self.path, self.total = action, path, total
        self.moved = 0
        self.started = self.logged = time.monotonic()
        self._lock = threading.Lock()

    def add(self, size: int) -> None:
        with self._lock:
            self.moved += size
            now = time.monotonic()
            if (now - self.logged < _PROGRESS_SECONDS
                    and self.moved < self.total):
                return
            self.logged = now
        logger.info('Snapshot %s of %s: %d of %d bytes, %.2f GB/s',
                    self.action, self.path, self.moved, self.total,
                    self.moved / max(now - self.started, 1e-9) / 1e9)


def _transfer(connector: Any, path: str, items: Sequence[Any],
              sizes: Sequence[int], work: Callable[..., Any],
              action: str) -> List[Any]:
    from kumo_core.sources.object_store import copies
    from kumo_core.sources.parallel import run_in_parallel

    if not items:
        return []
    stop = threading.Event()
    with copies(items) as workers:
        # After `copies`: Arrow's IO pool size sets the connection pool.
        client = _Client(connector, path)
        logger.info('Snapshot %s of %s: %d files, %d bytes, %d workers',
                    action, path, len(items), sum(sizes), workers)
        progress = _Progress(action, path, sum(sizes))

        def run(item: Any, size: int) -> Any:
            result = _retried(client, work, item, stop)
            progress.add(size)
            return result

        return run_in_parallel(
            [partial(run, item, size) for item, size in zip(items, sizes)],
            workers=workers, stop=stop)


def _retried(client: _Client, work: Callable[..., Any], item: Any,
             stop: threading.Event) -> Any:
    attempt = 0
    while True:
        _stopped(stop)
        current = client.current
        try:
            return work(current, item, stop)
        except (FileNotFoundError, PermissionError):
            raise
        except OSError as error:
            if attempt + 1 == _ATTEMPTS:
                raise
            logger.warning('Retrying a snapshot transfer after: %s', error)
            # Arrow keeps the credentials a client was built with.
            client.renew(current)
            stop.wait(2**attempt)
            attempt += 1


def _stopped(stop: threading.Event) -> None:
    from kumo_core.sources.parallel import WorkCancelled

    if stop.is_set():
        raise WorkCancelled('abandoned after another transfer failed')


def _read_json(filesystem: FileSystem, path: str) -> Optional[Any]:
    try:
        return json.loads(_files(filesystem).read_text(path))
    except FileNotFoundError:
        return None


def _open_output(filesystem: FileSystem, path: str) -> Any:
    _files(filesystem).makedirs(path.rsplit('/', 1)[0])
    return filesystem.open_output_stream(path)


def _write(filesystem: FileSystem, path: str, payload: bytes) -> None:
    with _open_output(filesystem, path) as writer:
        writer.write(payload)


def _fingerprint(index_manifest: bytes) -> str:
    return hashlib.sha256(index_manifest).hexdigest()


def export(snapshot: Any, connector: Any, path: str) -> str:
    from diskgraph import snapshot_range_checksums

    from kumo.engine._provenance import environment

    source = Path(snapshot.index_path)
    graph_key, data_version = snapshot.graph_key, snapshot.data_version
    _check_key(graph_key, data_version, snapshot.id)
    location = f"{path.rstrip('/')}/{graph_key}/{data_version[:16]}"
    index_manifest = (source / _INDEX_MANIFEST).read_bytes()
    sizes = {
        entry['rel_path']: int(entry['size_bytes'])
        for entry in json.loads(index_manifest)['files']
    }
    fingerprint = _fingerprint(index_manifest)
    filesystem, root = _location(connector, location)

    def recorded() -> Optional[dict]:
        existing = _read_json(filesystem, f'{root}/{_MANIFEST}')
        found = (existing.get('snapshot_id')
                 if isinstance(existing, dict) else None)
        if existing is not None and found != snapshot.id:
            raise ValueError(f'{location} holds snapshot {found!r}, not '
                             f'{snapshot.id!r}; remove it and re-export.')
        return existing

    def build() -> Optional[str]:
        return (recorded() or {}).get('index_manifest')

    def committed() -> bool:
        return _files(filesystem).exists(f'{root}/index/{_SUCCESS}')

    def present() -> Dict[str, int]:
        return {
            info.path[len(root) + len('/index/'):]: info.size
            for info in filesystem.get_file_info(
                FileSelector(f'{root}/index', recursive=True,
                             allow_not_found=True))
            if info.type is FileType.File
        }

    def upload(client: FileSystem, name: str, stop: threading.Event) -> None:
        size = len(index_manifest) if name == _INDEX_MANIFEST else sizes[name]
        with open(source / name, 'rb', buffering=0) as reader, _open_output(
                client, f'{root}/index/{name}') as writer:
            for offset in range(0, size, _CHUNK):
                _stopped(stop)
                chunk = reader.read(min(_CHUNK, size - offset))
                if len(chunk) != min(_CHUNK, size - offset):
                    raise OSError(f'{source / name} changed while it was '
                                  'exported')
                writer.write(chunk)

    def send(names: List[str], action: str) -> None:
        names.sort(key=lambda name: -sizes.get(name, 0))
        _transfer(connector, location, names,
                  [sizes.get(name, len(index_manifest)) for name in names],
                  upload, action)

    if committed():
        if build() != fingerprint:
            logger.info('Keeping the existing export of %s at %s',
                        snapshot.id, location)
            return location
        uploaded = present()
        send([
            name for name, size in {
                **sizes, _INDEX_MANIFEST: len(index_manifest)
            }.items() if uploaded.get(name) != size
        ], 'repair')
        return location
    existing = recorded()
    if existing is None and filesystem.get_file_info(
            FileSelector(root, allow_not_found=True)):
        raise ValueError(f'{location} is not empty; remove it and re-export.')
    resume = (existing or {}).get('index_manifest') == fingerprint
    recorded_snapshot = snapshot.manifest
    metadata = {
        'format': _FORMAT,
        'snapshot_id': snapshot.id,
        'graph_key': graph_key,
        'data_version': data_version,
        'index_manifest': fingerprint,
        'graph_config': snapshot.graph_config,
        'snapshot': None if recorded_snapshot is None else
        {k: v
         for k, v in recorded_snapshot.items() if k != 'loaded'},
        'environment': environment(),
        'range_bytes': _RANGE_BYTES,
    }
    _write(filesystem, f'{root}/{_MANIFEST}', json.dumps(metadata).encode())
    with ThreadPoolExecutor(1) as pool:
        checksums = pool.submit(snapshot_range_checksums, str(source),
                                _RANGE_BYTES)
        uploaded = present() if resume else {}
        send([
            name for name, size in sizes.items() if uploaded.get(name) != size
        ], 'export')
        metadata['crc32'] = checksums.result()
    filesystem, root = _location(connector, location)
    if build() != fingerprint:
        raise RuntimeError(f'Another build of {snapshot.id} was exported to '
                           f'{location} meanwhile; re-export.')
    if not committed():
        _write(filesystem, f'{root}/{_MANIFEST}', json.dumps(metadata).encode())
        _write(filesystem, f'{root}/index/{_INDEX_MANIFEST}', index_manifest)
        _write(filesystem, f'{root}/index/{_SUCCESS}', fingerprint.encode())
    return location


def _corrupt(path: str, detail: str) -> ValueError:
    return ValueError(f'The snapshot export at {path} is corrupt ({detail}); '
                      'delete it and run GraphSnapshot.export again.')


def read(connector: Any, path: str) -> Tuple[str, dict]:
    filesystem, root = _location(connector, path)
    manifest = _read_json(filesystem, f'{root}/{_MANIFEST}')
    if manifest is None or not _files(filesystem).exists(
            f'{root}/index/{_SUCCESS}'):
        raise FileNotFoundError(f'No complete snapshot export at {path}; '
                                're-run export.')
    if not isinstance(manifest, dict):
        raise _corrupt(path, 'manifest is not a JSON object')
    if manifest.get('format') != _FORMAT:
        warnings.warn(
            f"Snapshot export at {path} has layout format "
            f"{manifest.get('format')!r}; reading it as format {_FORMAT}.",
            UserWarning, stacklevel=3)
    _check_key(manifest.get('graph_key'), manifest.get('data_version'),
               manifest.get('snapshot_id'))
    span, crcs = manifest.get('range_bytes'), manifest.get('crc32')
    if (type(span) is not int or span <= 0 or not isinstance(crcs, dict)
            or not isinstance(manifest.get('index_manifest'), str)
            or not isinstance(manifest.get('graph_config'), dict)
            or not isinstance(manifest.get('snapshot'), (dict, type(None)))):
        raise _corrupt(path, 'malformed manifest')
    return root, manifest


def _download(root: str, staging: Path, client: FileSystem,
              item: Tuple[str, int, int], stop: threading.Event) -> None:
    name, offset, length = item
    # Each open is a HEAD: pyarrow cannot open a file from a known size.
    with client.open_input_file(f'{root}/index/{name}') as reader:
        fd = os.open(staging / name, os.O_WRONLY)
        try:
            for start in range(offset, offset + length, _CHUNK):
                _stopped(stop)
                want = min(_CHUNK, offset + length - start)
                chunk = reader.read_at(want, start)
                if len(chunk) != want:
                    raise ValueError(f'{name} is truncated')
                os.pwrite(fd, chunk, start)
            getattr(os, 'fdatasync', os.fsync)(fd)
        finally:
            os.close(fd)


def _index_files(path: str, root: str, manifest: dict,
                 connector: Any) -> Tuple[bytes, List[Tuple[str, int]]]:
    from kumo_core.sources.layout import confined

    filesystem, _ = _location(connector, path)
    with filesystem.open_input_stream(
            f'{root}/index/{_INDEX_MANIFEST}') as reader:
        index_manifest = reader.readall()
    if _fingerprint(index_manifest) != manifest['index_manifest']:
        raise _corrupt(path, f'{_INDEX_MANIFEST} is not the exported one')
    try:
        files = [(confined(entry['rel_path']), int(entry['size_bytes']))
                 for entry in json.loads(index_manifest)['files']]
    except Exception as error:
        raise _corrupt(path, f'bad {_INDEX_MANIFEST}: {error}') from error
    if len({name for name, _ in files}) != len(files):
        raise _corrupt(path, f'{_INDEX_MANIFEST} lists a file twice')
    return index_manifest, files


def load_into(connector: Any, path: str, root: str, manifest: dict,
              target: Path) -> None:
    from diskgraph import (
        DiskgraphCorruptionError,
        format_versions,
        snapshot_range_checksums,
        verify_snapshot_integrity,
    )
    from kumo_core.artifacts import FilesystemArtifactStore
    from kumo_core.diskgraph.snapshot import (
        DiskgraphIngestResult,
        write_diskgraph_success_marker,
    )
    from kumo_core.sources.transfer import discard_destination

    from kumo.engine._manifest import write_manifest
    from kumo.engine._provenance import environment
    from kumo.engine.graph import SNAPSHOT_MANIFEST_FILE

    span = manifest['range_bytes']
    index_manifest, files = _index_files(path, root, manifest, connector)
    staging = target.with_name(f'{target.name}.loading')
    discard_destination(str(target))
    discard_destination(str(staging))
    try:
        staging.mkdir(parents=True)
        needed = sum(size for _, size in files)
        disk = os.statvfs(staging)
        if disk.f_bavail * disk.f_frsize < needed:
            raise OSError(f'Loading this snapshot needs {needed} free bytes '
                          f'in {staging}.')
        for name, size in files:
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(staging / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            try:
                if size:
                    os.posix_fallocate(fd, 0, size)
            finally:
                os.close(fd)
        ranges = sorted(((name, offset, min(span, size - offset))
                         for name, size in files
                         for offset in range(0, size, span)),
                        key=lambda item: -item[2])
        try:
            _transfer(connector, path, ranges, [item[2] for item in ranges],
                      partial(_download, root, staging), 'load')
        except ValueError as error:
            raise _corrupt(path, str(error)) from error
        (staging / _INDEX_MANIFEST).write_bytes(index_manifest)
        write_diskgraph_success_marker(str(staging),
                                       DiskgraphIngestResult.for_index(target))
        try:
            verify_snapshot_integrity(str(staging))
            loaded = snapshot_range_checksums(str(staging), span)
        except DiskgraphCorruptionError as error:
            raise _corrupt(path, str(error)) from error
        for name, crcs in sorted(loaded.items()):
            expected = manifest['crc32'].get(name)
            if crcs != expected:
                offset = next((index * span for index, crc in enumerate(crcs)
                               if not expected or index >= len(expected)
                               or crc != expected[index]), 0)
                raise _corrupt(
                    path, f'{name} failed its checksum at offset {offset}')
        _check_readable(staging, manifest)
        if manifest.get('snapshot') is not None:
            write_manifest(
                staging / SNAPSHOT_MANIFEST_FILE, {
                    **manifest['snapshot'], 'loaded': {
                        'formats': format_versions(),
                        'environment': environment(),
                    }
                })
        for directory in sorted({(staging / name).parent
                                 for name, _ in files} | {staging}):
            FilesystemArtifactStore._fsync_directory(directory)
        os.rename(staging, target)
        FilesystemArtifactStore._fsync_directory(target.parent)
    except BaseException:
        discard_destination(str(staging))
        raise


def _check_readable(index: Path, manifest: dict) -> None:
    from diskgraph import SamplerEngine, format_versions

    try:
        SamplerEngine(str(index), num_threads=1)
    except Exception as error:
        built = (manifest.get('snapshot') or {}).get('formats')
        raise RuntimeError(
            f'This DiskGraph cannot open the snapshot: built with formats '
            f'{built}, installed are {format_versions()}. Rebuild it with '
            'this kumo-engine or load it with the exporting one.') from error
