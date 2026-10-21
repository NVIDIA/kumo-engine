# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import warnings
import zlib
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pyarrow.fs import FileSelector, FileSystem, FileType

_FORMAT = 1
_MANIFEST = 'kumo_snapshot.json'
_EXPORTING = '_EXPORTING'
_RANGE_BYTES = 64 << 20
_CHUNK = 8 << 20
_INDEX_MANIFEST = '_MANIFEST.json'
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


def _check_key(graph_key: Any, data_version: Any, snapshot_id: Any) -> None:
    if not (isinstance(graph_key, str) and isinstance(data_version, str)
            and _KEY.fullmatch(graph_key) and _KEY.fullmatch(data_version)
            and snapshot_id == f'{graph_key}@{data_version[:16]}'):
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


def _parallel(connector: Any, path: str, items: Sequence[Any],
              sizes: Sequence[int], work: Callable[..., Any],
              action: str) -> List[Any]:
    from kumo_core.sources.object_store import copies

    stop = threading.Event()
    total = sum(sizes)
    with copies(items) as workers, ThreadPoolExecutor(workers) as pool:
        # After `copies`: Arrow's IO pool size sets the connection pool.
        client = _Client(connector, path)
        logger.info('Snapshot %s of %s: %d files, %d bytes, %d workers',
                    action, path, len(items), total, workers)
        started = time.monotonic()
        futures = {
            pool.submit(_retried, client, work, item, stop): size
            for item, size in zip(items, sizes)
        }
        try:
            pending, moved = set(futures), 0
            while pending:
                done, pending = wait(pending, timeout=_PROGRESS_SECONDS,
                                     return_when=FIRST_EXCEPTION)
                for future in done:
                    if future.exception() is not None:
                        raise future.exception()
                    moved += futures[future]
                elapsed = time.monotonic() - started
                logger.info('Snapshot %s of %s: %d of %d bytes, %.2f GB/s',
                            action, path, moved, total,
                            moved / max(elapsed, 1e-9) / 1e9)
            return [future.result() for future in futures]
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise


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


def _stream(source: Path, size: int, stop: threading.Event,
            write: Optional[Callable[[bytes], Any]] = None) -> List[int]:
    crcs: List[int] = []
    with open(source, 'rb', buffering=0) as reader:
        for offset in range(0, size, _CHUNK):
            _stopped(stop)
            chunk = reader.read(min(_CHUNK, size - offset))
            if len(chunk) != min(_CHUNK, size - offset):
                raise OSError(f'{source} changed while it was exported')
            if offset % _RANGE_BYTES == 0:
                crcs.append(0)
            crcs[-1] = zlib.crc32(chunk, crcs[-1])
            if write is not None:
                write(chunk)
    return crcs or [0]


def _read_json(filesystem: FileSystem, path: str) -> Optional[dict]:
    try:
        with filesystem.open_input_stream(path) as reader:
            return json.loads(reader.readall())
    except FileNotFoundError:
        return None


def _open_output(filesystem: FileSystem, path: str) -> Any:
    if filesystem.type_name not in ('s3', 'gcs'):
        filesystem.create_dir(path.rsplit('/', 1)[0], recursive=True)
    return filesystem.open_output_stream(path)


def _write_json(filesystem: FileSystem, path: str, value: dict) -> None:
    with _open_output(filesystem, path) as writer:
        writer.write(json.dumps(value).encode())


def export(snapshot: Any, connector: Any, path: str) -> str:
    from kumo.engine._provenance import environment

    source = Path(snapshot.index_path)
    graph_key, data_version = source.parent.name, source.name
    _check_key(graph_key, data_version, snapshot.id)
    location = f"{path.rstrip('/')}/{graph_key}/{data_version[:16]}"
    index_manifest = (source / _INDEX_MANIFEST).read_bytes()
    files = sorted([(entry['rel_path'], int(entry['size_bytes']))
                    for entry in json.loads(index_manifest)['files']] +
                   [(_INDEX_MANIFEST, len(index_manifest))],
                   key=lambda file: -file[1])
    fingerprint = hashlib.sha256(index_manifest).hexdigest()
    identity = {'snapshot_id': snapshot.id, 'index_manifest': fingerprint}
    filesystem, root = _location(connector, location)

    def committed() -> Optional[dict]:
        existing = _read_json(filesystem, f'{root}/{_MANIFEST}')
        found = (existing.get('snapshot_id')
                 if isinstance(existing, dict) else None)
        if existing is not None and found != snapshot.id:
            raise ValueError(f'{location} holds snapshot {found!r}, not '
                             f'{snapshot.id!r}; remove it and re-export.')
        return existing

    def finished() -> bool:
        existing = committed()
        if existing and existing.get('index_manifest') != fingerprint:
            raise RuntimeError(f'Another build of {snapshot.id} was exported '
                               f'to {location} meanwhile; re-export.')
        return existing is not None

    def present() -> Dict[str, int]:
        return {
            info.path[len(root) + len('/index/'):]: info.size
            for info in filesystem.get_file_info(
                FileSelector(f'{root}/index', recursive=True,
                             allow_not_found=True))
            if info.type is FileType.File
        }

    def upload(client: FileSystem, file: Tuple[str, int],
               stop: threading.Event) -> List[int]:
        name, size = file
        if uploaded.get(name) == size:
            return _stream(source / name, size, stop)
        with _open_output(client, f'{root}/index/{name}') as writer:
            return _stream(source / name, size, stop, writer.write)

    existing = committed()
    if existing is not None and existing.get('index_manifest') != fingerprint:
        logger.info('Keeping the existing export of %s at %s', snapshot.id,
                    location)
        return location
    if existing is not None:
        uploaded = present()
        stale = [file for file in files if uploaded.get(file[0]) != file[1]]
        if stale:
            _parallel(connector, location, stale,
                      [size for _, size in stale], upload, 'repair')
        return location
    marker = _read_json(filesystem, f'{root}/{_EXPORTING}')
    if not isinstance(marker, (dict, type(None))):
        marker = {}
    if marker is None and filesystem.get_file_info(
            FileSelector(root, allow_not_found=True)):
        raise ValueError(f'{location} is not empty; remove it and re-export.')
    if marker is not None and marker.get('snapshot_id') != snapshot.id:
        raise ValueError(f'{location} holds a partial export of another '
                         'snapshot; remove it and re-export.')
    resume = (marker or {}).get('index_manifest') == fingerprint
    _write_json(filesystem, f'{root}/{_EXPORTING}', identity)
    uploaded = present() if resume else {}
    crcs = _parallel(connector, location, files, [size for _, size in files],
                     upload, 'export')
    filesystem, root = _location(connector, location)
    if finished():
        return location
    if (_read_json(filesystem, f'{root}/{_EXPORTING}') != identity
            and not finished()):
        raise RuntimeError(f'Another export to {location} started '
                           'meanwhile; re-export.')
    recorded = snapshot.manifest
    _write_json(
        filesystem, f'{root}/{_MANIFEST}', {
            'format': _FORMAT,
            'snapshot_id': snapshot.id,
            'graph_key': graph_key,
            'data_version': data_version,
            'index_manifest': fingerprint,
            'graph_config': snapshot.graph_config,
            'snapshot': None if recorded is None else
            {k: v for k, v in recorded.items() if k != 'loaded'},
            'environment': environment(),
            'range_bytes': _RANGE_BYTES,
            'files': [{'path': name, 'size': size, 'crc32': file_crcs}
                      for (name, size), file_crcs in zip(files, crcs)],
        })
    try:
        filesystem.delete_file(f'{root}/{_EXPORTING}')
    except FileNotFoundError:
        pass
    return location


def _corrupt(path: str, detail: str) -> ValueError:
    return ValueError(f'The snapshot export at {path} is corrupt ({detail}); '
                      'delete it and run GraphSnapshot.export again.')


def read(connector: Any, path: str) -> Tuple[str, dict]:
    filesystem, root = _location(connector, path)
    manifest = _read_json(filesystem, f'{root}/{_MANIFEST}')
    if manifest is None:
        raise FileNotFoundError(f'No complete snapshot export at {path} '
                                f'({_MANIFEST} is missing); re-run export.')
    if not isinstance(manifest, dict):
        raise _corrupt(path, 'manifest is not a JSON object')
    if manifest.get('format') != _FORMAT:
        warnings.warn(
            f"Snapshot export at {path} has layout format "
            f"{manifest.get('format')!r}; reading it as format {_FORMAT}.",
            UserWarning, stacklevel=3)
    _check_key(manifest.get('graph_key'), manifest.get('data_version'),
               manifest.get('snapshot_id'))
    span, files = manifest.get('range_bytes'), manifest.get('files')
    if (type(span) is not int or span <= 0 or not isinstance(files, list)
            or not isinstance(manifest.get('graph_config'), dict)
            or not isinstance(manifest.get('snapshot'), (dict, type(None)))):
        raise _corrupt(path, 'malformed manifest')
    seen = set()
    for entry in files:
        name, size, crcs = ((entry.get('path'), entry.get('size'),
                             entry.get('crc32'))
                            if isinstance(entry, dict) else (None, None, None))
        if (not isinstance(name, str) or name.startswith('/') or '\\' in name
                or any(part in ('', '.', '..') for part in name.split('/'))
                or name in seen or type(size) is not int or size < 0
                or not isinstance(crcs, list)
                or any(type(crc) is not int for crc in crcs)
                or len(crcs) != max(-(-size // span), 1)):
            raise _corrupt(path, f'bad file entry {name!r}')
        seen.add(name)
    return root, manifest


def _download(root: str, staging: Path, client: FileSystem,
              item: Tuple[str, int, int, int], stop: threading.Event) -> None:
    name, offset, length, expected = item
    crc = 0
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
                crc = zlib.crc32(chunk, crc)
                os.pwrite(fd, chunk, start)
            getattr(os, 'fdatasync', os.fsync)(fd)
        finally:
            os.close(fd)
    if crc != expected:
        raise ValueError(f'{name} failed its checksum at offset {offset}')


def load_into(connector: Any, path: str, root: str, manifest: dict,
              target: Path) -> None:
    from diskgraph import format_versions, verify_snapshot_integrity
    from kumo_core.diskgraph.snapshot import (
        DiskgraphIngestResult,
        write_diskgraph_success_marker,
    )

    from kumo.engine._manifest import write_manifest
    from kumo.engine._provenance import environment
    from kumo.engine.graph import SNAPSHOT_MANIFEST_FILE

    span = manifest['range_bytes']
    files = [(entry['path'], entry['size'], entry['crc32'])
             for entry in manifest['files']]
    staging = target.with_name(f'{target.name}.loading')
    shutil.rmtree(staging, ignore_errors=True)
    try:
        staging.mkdir(parents=True)
        # Any old target stays until the rename.
        needed = sum(size for _, size, _ in files)
        disk = os.statvfs(staging)
        if disk.f_bavail * disk.f_frsize < needed:
            raise OSError(f'Loading this snapshot needs {needed} free bytes '
                          f'in {staging}.')
        for name, size, _ in files:
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(staging / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            try:
                os.ftruncate(fd, size)
            finally:
                os.close(fd)
        ranges = sorted(
            ((name, offset, min(span, size - offset), crcs[index])
             for name, size, crcs in files
             for index, offset in enumerate(range(0, size, span))),
            key=lambda item: -item[2])
        try:
            _parallel(connector, path, ranges, [item[2] for item in ranges],
                      partial(_download, root, staging), 'load')
        except ValueError as error:
            raise _corrupt(path, str(error)) from error
        write_diskgraph_success_marker(str(staging),
                                       DiskgraphIngestResult.for_index(target))
        verify_snapshot_integrity(str(staging))
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
                                 for name, _, _ in files} | {staging}):
            _fsync(directory)
        shutil.rmtree(target, ignore_errors=True)
        os.rename(staging, target)
        _fsync(target.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
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


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
