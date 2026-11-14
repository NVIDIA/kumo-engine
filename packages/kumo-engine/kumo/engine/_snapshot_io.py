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
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.diskgraph.sampling_artifacts import (
    ArtifactFile,
    relative_artifact_path,
)
from pyarrow.fs import FileSystem

from kumo.engine.connector.base import Connector

_FORMAT = 1
_MANIFEST = 'kumo_snapshot.json'
_INDEX_MANIFEST = '_MANIFEST.json'
_RANGE_BYTES = 64 << 20
_CHUNK = 8 << 20
_ATTEMPTS = 3
_KEY = re.compile('[0-9a-f]{64}')
_GENERATION = re.compile('[0-9a-f]{32}')
_EXPORT_IDENTITY = '.kumo_export'
_PROGRESS_SECONDS = 30
_DONE = '_DONE'
_ABANDONED_SECONDS = 24 * 3600

logger = logging.getLogger(__name__)


def _location(connector: Any, path: str) -> Tuple[FileSystem, str]:
    from kumo.engine.connector import file_store

    store = file_store(connector, path)
    if store is None:
        raise NotImplementedError(
            f'{type(connector).__name__} cannot hold an artifact export; use '
            'an S3, GCS or file connector.')
    return store


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
        logger.info('Artifact %s of %s: %d of %d bytes, %.2f GB/s',
                    self.action, self.path, self.moved, self.total,
                    self.moved / max(now - self.started, 1e-9) / 1e9)


@dataclass(frozen=True)
class _TransferWork:
    action: str
    run: Callable[..., Any]
    uploaded: Callable[[Any], None] | None = None


def _transfer(connector: Any, path: str, items: Sequence[Any],
              sizes: Sequence[int], work: _TransferWork) -> List[Any]:
    from kumo_core.sources.object_store import copies
    from kumo_core.sources.parallel import run_in_parallel

    if not items:
        return []
    stop = threading.Event()
    notification_lock = threading.Lock()
    with copies(items) as workers:
        client = _Client(connector, path)
        logger.info('Artifact %s of %s: %d files, %d bytes, %d workers',
                    work.action, path, len(items), sum(sizes), workers)
        progress = _Progress(work.action, path, sum(sizes))

        def run(item: Any, size: int) -> Any:
            result = _retried(client, work.run, item, stop)
            if work.uploaded is not None:
                with notification_lock:
                    _stopped(stop)
                    work.uploaded(result)
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
            logger.warning('Retrying an artifact transfer after: %s', error)
            client.renew(current)
            stop.wait(2**attempt)
            attempt += 1


def _stopped(stop: threading.Event) -> None:
    from kumo_core.sources.parallel import WorkCancelled

    if stop.is_set():
        raise WorkCancelled('abandoned after another transfer failed')


def upload_files(
    connector: Connector,
    source: Path,
    path: str,
    files: Mapping[str, int],
    on_file_uploaded: Callable[[str, ArtifactFile], None] | None = None,
) -> tuple[ArtifactFile, ...]:
    """Upload files through the MLOps transfer, retry and progress machinery.

    Args:
        connector: Connector owning the destination.
        source: Immutable local root containing the inventory.
        path: Connector-relative destination directory.
        files: Relative file names and expected byte sizes.
        on_file_uploaded: Serialized notification after a successful stream close.
            Callback errors propagate outside the upload retry loop.
    Returns:
        File identities when notifications are requested, otherwise an empty tuple.
        Digests are computed while uploading only when a notification needs them.
    Raises:
        ValueError: If paths escape the source or the inventory changes.
        OSError: If upload fails after retries.
    """
    if not files:
        return ()
    source = source.resolve(strict=True)
    for name, size in files.items():
        relative_artifact_path(name)
        if size < 0:
            raise ValueError('Artifact sizes must be nonnegative')
    _, target = _location(connector, path)

    def upload(client: FileSystem, name: str,
               stop: threading.Event) -> ArtifactFile | None:
        local = source / relative_artifact_path(name)
        before = local.stat()
        if local.is_symlink() or not local.resolve(strict=True).is_relative_to(source):
            raise ValueError('Symbolic-link artifact or source outside its root')
        size = files[name]
        if before.st_size != size:
            raise ValueError('Artifact size differs from its inventory')
        digest = hashlib.sha256() if on_file_uploaded is not None else None
        with local.open('rb', buffering=0) as reader, _open_output(
                client, f'{target}/{name}', size) as writer:
            for offset in range(0, size, _CHUNK):
                _stopped(stop)
                expected = min(_CHUNK, size - offset)
                chunk = reader.read(expected)
                if len(chunk) != expected:
                    raise ValueError('Artifact changed while it was exported')
                writer.write(chunk)
                if digest is not None:
                    digest.update(chunk)
        after = local.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Artifact changed while it was exported')
        return None if digest is None else ArtifactFile(name, size, digest.hexdigest())

    names = sorted(files, key=lambda name: -files[name])
    notify = (None if on_file_uploaded is None else
              lambda entry: on_file_uploaded(target, entry))
    result = _transfer(connector, path, names, [files[name] for name in names],
                       _TransferWork('export', upload, notify))
    return tuple(entry for entry in result if entry is not None)


def _read_json(filesystem: FileSystem, path: str) -> Optional[Any]:
    try:
        return json.loads(_files(filesystem).read_text(path))
    except FileNotFoundError:
        return None


def _read_bytes(filesystem: FileSystem, path: str) -> bytes:
    with filesystem.open_input_stream(path) as reader:
        return reader.readall()


def _listed_sizes(index_manifest: bytes) -> Dict[str, int]:
    return {
        entry['rel_path']: int(entry['size_bytes'])
        for entry in json.loads(index_manifest)['files']
    }


def _open_output(filesystem: FileSystem, path: str, size: int) -> Any:
    from kumo_core.object_store_sink import open_sized

    _files(filesystem).makedirs(path.rsplit('/', 1)[0])
    return open_sized(filesystem, path, size)


def _write(filesystem: FileSystem, path: str, payload: bytes) -> None:
    with _open_output(filesystem, path, len(payload)) as writer:
        writer.write(payload)


def _fingerprint(index_manifest: bytes) -> str:
    return hashlib.sha256(index_manifest).hexdigest()


def _snapshot_manifest(snapshot: Any) -> bytes:
    from kumo.engine._manifest import (
        RECONSTRUCTED,
        SnapshotManifest,
        SnapshotParents,
        manifest_bytes,
    )
    try:
        return (Path(snapshot.index_path) / PROVENANCE_FILE).read_bytes()
    except OSError:
        return manifest_bytes(
            SnapshotManifest(id=snapshot.id,
                             parents=SnapshotParents(source_tables={}),
                             **RECONSTRUCTED))


def export_location(snapshot: Any, path: str) -> str:
    graph_key, data_version = snapshot._graph_key, snapshot._data_version
    _check_key(graph_key, data_version, snapshot.id)
    return f"{path.rstrip('/')}/{graph_key}/{data_version[:16]}"


def store(snapshot: Any, settings: Any,
          on_file_uploaded: Callable[[str, ArtifactFile], None] | None = None,
          on_files_uploaded: Callable[[], None] | None = None) -> None:
    """Store a snapshot using the configured artifact root.

    Args:
        snapshot: Completed immutable graph snapshot.
        settings: MLOps settings supplying its writable artifact root.
        on_file_uploaded: Serialized notification after each successful file close.
        on_files_uploaded: Replica barrier invoked before committing the export.
    Raises:
        RuntimeError: If upload, notification, or the replica barrier fails.
    """
    from kumo.engine._manifest import SnapshotManifest
    from kumo.engine._mlops import snapshot_path

    try:
        settings.locate(Path(snapshot.index_path) / PROVENANCE_FILE,
                        SnapshotManifest, snapshot_path(snapshot))
        connector = settings.root.connector
        path = settings.root.location('snapshots')
        filesystem, root = _location(connector,
                                     export_location(snapshot, path))
        try:
            marker = _read_json(filesystem, f'{root}/{_MANIFEST}')
        except ValueError:
            marker = None
        if on_file_uploaded is not None or not (
                isinstance(marker, dict) and marker.get('format') == _FORMAT
                and marker.get('snapshot_id') == snapshot.id):
            if on_file_uploaded is None:
                export(snapshot, connector, path)
            else:
                export(snapshot, connector, path,
                       on_file_uploaded=on_file_uploaded,
                       on_files_uploaded=on_files_uploaded)
    except Exception as error:
        raise RuntimeError(f"Could not store snapshot {snapshot.id} under the "
                           f"artifact root: {error}") from error


def export(snapshot: Any, connector: Any, path: str, *,
           on_file_uploaded: Callable[[str, ArtifactFile], None] | None = None,
           on_files_uploaded: Callable[[], None] | None = None) -> str:
    """Export a graph, optionally pipelining replica downloads before commit.

    Args:
        snapshot: Completed immutable graph snapshot.
        connector: Connector providing the destination filesystem.
        path: Snapshot parent directory on the connector.
        on_file_uploaded: Receives the storage directory and verified file identity.
            Callback failures propagate without retrying the successful upload.
        on_files_uploaded: Final replica barrier; failure preserves the last export.
    Returns:
        Connector-relative location of the committed snapshot export.
    Raises:
        OSError: If artifact reads or writes fail after retries.
        ValueError: If the snapshot identity or artifact inventory is invalid.
    """
    from diskgraph import snapshot_range_checksums

    from kumo.engine._provenance import environment
    source = Path(snapshot.index_path)
    graph_key, data_version = snapshot._graph_key, snapshot._data_version
    location = export_location(snapshot, path)
    snapshot_manifest = _snapshot_manifest(snapshot)
    if on_file_uploaded is None and _holds_copy(
            connector, location, snapshot.id, source, snapshot_manifest):
        logger.info('%s already holds an identical export of snapshot %s; '
                    'reused it', location, snapshot.id)
        return location
    index_manifest = (source / _INDEX_MANIFEST).read_bytes()
    sizes = _listed_sizes(index_manifest)
    generation = uuid.uuid4().hex
    filesystem, root = _location(connector, location)
    base = f'{root}/{generation}'

    if on_file_uploaded is not None:
        sizes.update({name: (source / name).stat().st_size
                      for name in (_INDEX_MANIFEST, '_SUCCESS', PROVENANCE_FILE)
                      if (source / name).is_file()})
    with ThreadPoolExecutor(1) as pool:
        checksums = pool.submit(snapshot_range_checksums, str(source), _RANGE_BYTES)
        upload_files(connector, source, f'{location}/{generation}/index', sizes,
                     on_file_uploaded)
        crc32 = checksums.result()
    if on_file_uploaded is None:
        _write(filesystem, f'{base}/index/{_INDEX_MANIFEST}', index_manifest)
    _write(filesystem, f'{base}/{PROVENANCE_FILE}', snapshot_manifest)
    marker = {
        'format': _FORMAT,
        'snapshot_id': snapshot.id,
        'generation': generation,
        'graph_key': graph_key,
        'data_version': data_version,
        'index_manifest': _fingerprint(index_manifest),
        'graph_config': snapshot.graph_config,
        'environment': environment().to_dict(),
        'range_bytes': _RANGE_BYTES,
        'crc32': crc32,
    }
    if on_files_uploaded is not None:
        on_files_uploaded()
    _commit(filesystem, root, generation, json.dumps(marker).encode())
    try:
        (source / _EXPORT_IDENTITY).write_text(export_identity(marker))
    except OSError:
        pass
    return location


def _holds_copy(connector: Any, location: str, snapshot_id: str,
                index: Path, snapshot_manifest: bytes) -> bool:
    from kumo_core.object_store_sink import file_sizes

    try:
        root, manifest = read(connector, location)
        if (manifest.get('snapshot_id') != snapshot_id
                or not holds_export(index, manifest)):
            return False
        filesystem, _ = _location(connector, location)
        remote = file_sizes(filesystem, root)
        listed = _listed_sizes((index / _INDEX_MANIFEST).read_bytes())
        if (_DONE not in remote
                or f'index/{_INDEX_MANIFEST}' not in remote) or any(
                    remote.get(f'index/{name}') != size
                    for name, size in listed.items()):
            return False
        return _read_bytes(filesystem,
                           f'{root}/{PROVENANCE_FILE}') == snapshot_manifest
    except Exception:
        return False


def _commit(filesystem: FileSystem, root: str, generation: str,
            marker: bytes) -> None:
    if filesystem.type_name == 'local':
        staged = f'{root}/{generation}/{_MANIFEST}'
        _write(filesystem, staged, marker)
        filesystem.move(staged, f'{root}/{_MANIFEST}')
    else:
        _write(filesystem, f'{root}/{_MANIFEST}', marker)
    _write(filesystem, f'{root}/{generation}/{_DONE}', b'')
    _clean(filesystem, root)


def _clean(filesystem: FileSystem, root: str) -> None:
    from pyarrow.fs import FileSelector, FileType

    try:
        newest: Dict[str, float] = {}
        done = set()
        for info in filesystem.get_file_info(
                FileSelector(root, recursive=True, allow_not_found=True)):
            name, _, rest = info.path[len(root) + 1:].partition('/')
            if (not rest or info.type is not FileType.File
                    or not _GENERATION.fullmatch(name)):
                continue
            mtime = info.mtime.timestamp() if info.mtime else time.time()
            newest[name] = max(newest.get(name, 0.0), mtime)
            if rest == _DONE:
                done.add(name)
        marker = _read_json(filesystem, f'{root}/{_MANIFEST}') or {}
        keep = marker.get('generation')
        stale = time.time() - _ABANDONED_SECONDS
        for name, mtime in newest.items():
            if name != keep and (name in done or mtime < stale):
                filesystem.delete_dir(f'{root}/{name}')
    except Exception as error:
        logger.warning('Could not remove old snapshot exports under %s: %s',
                       root, error)


def _corrupt(path: str, detail: str) -> ValueError:
    return ValueError(f'The snapshot export at {path} is corrupt ({detail}); '
                      'delete it and run GraphSnapshot.export again.')


def read(connector: Any, path: str) -> Tuple[str, dict]:
    filesystem, root = _location(connector, path)
    try:
        manifest = _read_json(filesystem, f'{root}/{_MANIFEST}')
    except ValueError as error:
        raise _corrupt(path, f'unreadable {_MANIFEST}: {error}') from error
    if manifest is None:
        raise FileNotFoundError(
            f'No snapshot export at {root} (no {_MANIFEST}); check the path.')
    if not isinstance(manifest, dict):
        raise _corrupt(path, 'manifest is not a JSON object')
    if manifest.get('format') != _FORMAT:
        raise ValueError(
            f"Snapshot export at {path} has format "
            f"{manifest.get('format')!r}, not {_FORMAT}; load it with the "
            f"kumo-engine that exported it.")
    _check_key(manifest.get('graph_key'), manifest.get('data_version'),
               manifest.get('snapshot_id'))
    span, crcs = manifest.get('range_bytes'), manifest.get('crc32')
    generation = manifest.get('generation')
    if (type(span) is not int or span <= 0 or not isinstance(crcs, dict)
            or not isinstance(manifest.get('index_manifest'), str)
            or not isinstance(manifest.get('graph_config'), dict)
            or not isinstance(generation, str)
            or not _GENERATION.fullmatch(generation)):
        raise _corrupt(path, 'malformed manifest')
    return f'{root}/{generation}', manifest


def export_identity(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(
        [manifest['range_bytes'], manifest['crc32'],
         manifest['index_manifest']], sort_keys=True).encode()).hexdigest()


def holds_export(index: Path, manifest: dict) -> bool:
    from diskgraph import snapshot_range_checksums

    identity = export_identity(manifest)
    try:
        return (index / _EXPORT_IDENTITY).read_text() == identity
    except FileNotFoundError:
        pass
    if snapshot_range_checksums(str(index),
                                manifest['range_bytes']) != manifest['crc32']:
        return False
    (index / _EXPORT_IDENTITY).write_text(identity)
    return True


def _download(root: str, staging: Path, client: FileSystem,
              item: Tuple[str, int, int], stop: threading.Event) -> None:
    name, offset, length = item
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


def _index_files(
        path: str, root: str, manifest: dict, connector: Any
) -> Tuple[bytes, List[Tuple[str, int]], Optional[bytes]]:
    from kumo_core.sources.layout import confined

    filesystem, _ = _location(connector, path)
    index_manifest = _read_bytes(filesystem, f'{root}/index/{_INDEX_MANIFEST}')
    if _fingerprint(index_manifest) != manifest['index_manifest']:
        raise _corrupt(path, f'{_INDEX_MANIFEST} is not the exported one')
    try:
        files = [(confined(entry['rel_path']), int(entry['size_bytes']))
                 for entry in json.loads(index_manifest)['files']]
    except Exception as error:
        raise _corrupt(path, f'bad {_INDEX_MANIFEST}: {error}') from error
    if len({name for name, _ in files}) != len(files):
        raise _corrupt(path, f'{_INDEX_MANIFEST} lists a file twice')
    try:
        snapshot_manifest = _read_bytes(filesystem, f'{root}/{PROVENANCE_FILE}')
    except FileNotFoundError:
        snapshot_manifest = None
    return index_manifest, files, snapshot_manifest


def load_into(connector: Any, path: str, root: str, manifest: dict,
              target: Path) -> None:
    from kumo_core.diskgraph.snapshot import (
        DiskgraphIngestResult,
        write_diskgraph_success_marker,
    )
    from kumo_core.io.filesystem import fsync_directory
    from kumo_core.sources.transfer import discard_destination

    span = manifest['range_bytes']
    index_manifest, files, snapshot_manifest = _index_files(
        path, root, manifest, connector)
    staging = target.with_name(f'{target.name}.loading')
    discard_destination(str(target))
    discard_destination(str(staging))
    try:
        _allocate(staging, files)
        ranges = sorted(((name, offset, min(span, size - offset))
                         for name, size in files
                         for offset in range(0, size, span)),
                        key=lambda item: -item[2])
        try:
            _transfer(connector, path, ranges, [item[2] for item in ranges],
                      _TransferWork('load', partial(_download, root, staging)))
        except ValueError as error:
            raise _corrupt(path, str(error)) from error
        (staging / _INDEX_MANIFEST).write_bytes(index_manifest)
        write_diskgraph_success_marker(str(staging),
                                       DiskgraphIngestResult.for_index(target))
        _verify(path, staging, manifest)
        _check_readable(staging, manifest)
        (staging / _EXPORT_IDENTITY).write_text(export_identity(manifest))
        if snapshot_manifest is not None:
            (staging / PROVENANCE_FILE).write_bytes(snapshot_manifest)
        for directory in sorted({(staging / name).parent
                                 for name, _ in files} | {staging}):
            fsync_directory(directory)
        os.rename(staging, target)
        fsync_directory(target.parent)
    except BaseException:
        discard_destination(str(staging))
        raise


def _allocate(staging: Path, files: List[Tuple[str, int]]) -> None:
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


def _verify(path: str, staging: Path, manifest: dict) -> None:
    from diskgraph import (
        DiskgraphCorruptionError,
        snapshot_range_checksums,
        verify_snapshot_integrity,
    )

    span = manifest['range_bytes']
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
            raise _corrupt(path,
                           f'{name} failed its checksum at offset {offset}')


def _check_readable(index: Path, manifest: dict) -> None:
    from diskgraph import SamplerEngine

    try:
        SamplerEngine(str(index), num_threads=1)
    except Exception as error:
        built = (manifest.get('environment') or {}).get('kumo_engine')
        raise RuntimeError(
            f'This DiskGraph cannot open the snapshot exported by kumo-engine '
            f'{built} ({error}); load it with that kumo-engine.') from error
