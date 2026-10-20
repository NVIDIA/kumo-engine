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
_UNSUPPORTED = {
    'SNOWFLAKE': 'Snowflake stages are not supported',
    'DATABRICKS': ('for a Unity Catalog Volume, use an S3Connector or '
                   "GCSConnector on the volume's backing cloud path"),
    'BIGQUERY': 'BigQuery cannot hold files',
}

logger = logging.getLogger(__name__)


def _location(connector: Any, path: str) -> Tuple[FileSystem, str]:
    from kumo_core.object_store_sink import ObjectStoreSinkBackend

    reason = _UNSUPPORTED.get(getattr(connector.source_type, 'name', None))
    sink = None if reason else connector.sink_backend
    if not isinstance(sink, ObjectStoreSinkBackend):
        raise NotImplementedError(
            f'{type(connector).__name__} cannot hold a snapshot export'
            f"{f' ({reason})' if reason else ''}; use an S3Connector, "
            'GCSConnector or FileConnector.')
    return sink.filesystem(connector, path, parallel_uploads=True)


def _check_key(graph_key: Any, data_version: Any, snapshot_id: Any) -> None:
    if not (isinstance(graph_key, str) and isinstance(data_version, str)
            and _KEY.fullmatch(graph_key) and _KEY.fullmatch(data_version)
            and snapshot_id == f'{graph_key}@{data_version[:16]}'):
        raise ValueError(
            f'Snapshot {snapshot_id!r} does not name a graph key and data '
            'version this package produced.')


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
        # Created after `copies` raises Arrow's IO pool, which sizes the
        # client's connection pool.
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
            # Leaving the pool joins the workers, which notice `stop`.
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


def _committed(filesystem: FileSystem, root: str, location: str,
               snapshot_id: str, fingerprint: str) -> bool:
    existing = _read_json(filesystem, f'{root}/{_MANIFEST}')
    if existing is None:
        return False
    if not isinstance(existing, dict):
        existing = {}
    if (existing.get('snapshot_id') == snapshot_id
            and existing.get('index_manifest') == fingerprint):
        return True
    raise ValueError(
        f'{location} already holds an export of snapshot '
        f"{existing.get('snapshot_id')!r} that differs from this one.")


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

    def committed() -> bool:
        return _committed(filesystem, root, location, snapshot.id,
                          fingerprint)

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

    if committed():
        # Re-running repairs files a racing or interrupted export rewrote.
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
        raise ValueError(
            f"{location} holds a partial export of snapshot "
            f"{marker.get('snapshot_id')!r}; remove it and re-export.")
    resume = (marker or {}).get('index_manifest') == fingerprint
    _write_json(filesystem, f'{root}/{_EXPORTING}', identity)
    uploaded = present() if resume else {}
    crcs = _parallel(connector, location, files, [size for _, size in files],
                     upload, 'export')
    filesystem, root = _location(connector, location)
    # An identical export racing this one uploads the same bytes, so
    # whichever finishes first commits and the other finds it committed.
    if committed():
        return location
    if (_read_json(filesystem, f'{root}/{_EXPORTING}') != identity
            and not committed()):
        raise RuntimeError(
            f'Another export to {location} started while this one ran; '
            'remove it and re-export.')
    _write_json(
        filesystem, f'{root}/{_MANIFEST}', {
            'format': _FORMAT,
            'snapshot_id': snapshot.id,
            'graph_key': graph_key,
            'data_version': data_version,
            'index_manifest': fingerprint,
            'graph_config': snapshot.graph_config,
            'snapshot': snapshot.manifest,
            'environment': environment(),
            'range_bytes': _RANGE_BYTES,
            'files': [{
                'path': name,
                'size': size,
                'crc32': file_crcs
            } for (name, size), file_crcs in zip(files, crcs)],
        })
    try:
        filesystem.delete_file(f'{root}/{_EXPORTING}')
    except FileNotFoundError:
        pass
    return location


def _format_note(manifest: Any) -> str:
    if not isinstance(manifest, dict) or manifest.get('format') == _FORMAT:
        return ''
    return (f" (export layout format {manifest.get('format')!r}; this "
            f'engine reads {_FORMAT})')


def _malformed(path: str, detail: str, manifest: Any) -> ValueError:
    return ValueError(f'The snapshot manifest at {path} {detail}; the export '
                      f'is corrupt{_format_note(manifest)}.')


def read(connector: Any, path: str) -> Tuple[str, dict]:
    filesystem, root = _location(connector, path)
    manifest = _read_json(filesystem, f'{root}/{_MANIFEST}')
    if manifest is None:
        raise FileNotFoundError(
            f'No complete snapshot export at {path}: {_MANIFEST} is missing. '
            'Re-run GraphSnapshot.export to finish it.')
    if not isinstance(manifest, dict):
        raise _malformed(path, 'is not a JSON object', manifest)
    if manifest.get('format') != _FORMAT:
        warnings.warn(
            f"Snapshot export at {path} has layout format "
            f"{manifest.get('format')!r}; reading it as format {_FORMAT}.",
            UserWarning, stacklevel=3)
    _check_key(manifest.get('graph_key'), manifest.get('data_version'),
               manifest.get('snapshot_id'))
    span, files = manifest.get('range_bytes'), manifest.get('files')
    if type(span) is not int or span <= 0:
        raise _malformed(path, 'has an invalid range_bytes', manifest)
    if not isinstance(files, list):
        raise _malformed(path, 'has no file list', manifest)
    for key in ('snapshot', 'environment'):
        if not isinstance(manifest.get(key, {}), (dict, type(None))):
            raise _malformed(path, f'has a non-object {key}', manifest)
    if not isinstance(manifest.get('graph_config'), dict):
        raise _malformed(path, 'has no graph_config', manifest)
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
            raise _malformed(
                path, f'lists {name!r} with an invalid or repeated path, size '
                'or checksums', manifest)
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
                    raise ValueError(
                        f'Snapshot file {name} in the export is shorter than '
                        'its manifest records; the export is corrupt.')
                crc = zlib.crc32(chunk, crc)
                os.pwrite(fd, chunk, start)
            # Flush while other ranges download, not all at the end.
            getattr(os, 'fdatasync', os.fsync)(fd)
        finally:
            os.close(fd)
    if crc != expected:
        raise ValueError(
            f'Snapshot file {name} failed its checksum at offset {offset}; '
            'the export is corrupt.')


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
        # Any old target stays until the rename, so the whole size is needed.
        needed = sum(size for _, size, _ in files)
        disk = os.statvfs(staging)
        if disk.f_bavail * disk.f_frsize < needed:
            raise OSError(
                f'Loading this snapshot needs {needed} bytes in {staging}, '
                f'which has {disk.f_bavail * disk.f_frsize} free.')
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
            if _format_note(manifest):
                raise ValueError(f'{error}{_format_note(manifest)}') from error
            raise
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


def _built_with(manifest: Dict[str, Any]) -> Tuple[Any, Any]:
    recorded = manifest.get('snapshot') or {}
    environment = (recorded.get('environment') or manifest.get('environment')
                   or {})
    return recorded.get('formats'), (environment.get('kumo_engine')
                                     if isinstance(environment, dict) else None)


def _check_readable(index: Path, manifest: dict) -> None:
    from diskgraph import SamplerEngine, format_versions

    from kumo.engine._version import __version__

    try:
        SamplerEngine(str(index), num_threads=1)
    except Exception as error:
        formats, engine = _built_with(manifest)
        running = format_versions()
        advice = ('The on-disk formats differ: rebuild the snapshot with '
                  'this kumo-engine, or load it with the exporting one.'
                  if formats is not None and formats != running else
                  'Re-export with a matching kumo-engine.')
        raise RuntimeError(
            "This machine's DiskGraph cannot open the snapshot. It was "
            f"built with DiskGraph formats {formats} by kumo-engine {engine}; "
            f"installed are formats {running} and kumo-engine "
            f"{__version__}{_format_note(manifest)}. {advice}") from error


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
