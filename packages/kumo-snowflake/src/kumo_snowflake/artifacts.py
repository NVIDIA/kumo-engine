# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Durable DiskGraph snapshots in a Snowflake named internal stage.

DiskGraph can only build and verify an index on a local filesystem. A
:class:`StageArtifactLocation` therefore keeps the active copy below
``Storage.scratch`` and translates the snapshot store's tree operations into
Snowflake file operations. The durable address remains the stage URI, so a
second process with an empty scratch directory can restore the same snapshot.

Publication uses a unique temporary prefix and promotes its files with
``COPY FILES``. ``_SUCCESS`` is copied last and is the commit marker: a failed
or interrupted promotion is never considered restorable. Snowflake stages do
not represent empty directories, so a small transport manifest records them;
it is consumed on restore and never reaches DiskGraph's integrity verifier.
"""

import hashlib
import json
import posixpath
import re
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

from kumo_core.diskgraph.snapshot import rebased
# Aliased: three methods below take a `SnapshotStore` parameter named
# `store`, and one name meaning two stores inside one class is how a line
# reaching for the identity store quietly gets the snapshot store instead.
from kumo_core.sources.principals import IdentityRef
from kumo_core.sources.principals import store as identity_store
from kumo_snowflake.session import (
    SnowflakeSession,
    _quoted_literal,
    _require_snowpark,
    _rows,
    _sessions,
    resolved,
)
from kumoapi.plugin import IngestResult, SnapshotStore, _Handle

_Item = TypeVar('_Item')
_Result = TypeVar('_Result')


def _in_parallel(work: Callable[[_Item], _Result], items: Sequence[_Item],
                 workers: int, name: str) -> List[_Result]:
    r"""``pool.map`` re-raises the first failure on iteration, so a caller
    that gets a list back knows every item succeeded. The pool is sized to the
    work so that a two-file tree does not start sixteen threads.
    """
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(items)),
                            thread_name_prefix=name) as pool:
        return list(pool.map(work, items))


_IDENTIFIER = (r'(?:[A-Za-z_][A-Za-z0-9_$]*|"(?:[A-Za-z0-9_$ .-]|"")+")')
_STAGE_LOCATION_BODY = (rf'(?P<stage>@(?P<database>{_IDENTIFIER})\.'
                        rf'(?P<schema>{_IDENTIFIER})\.(?P<name>{_IDENTIFIER}))'
                        r'(?:/(?P<prefix>[A-Za-z0-9_$=+.,/\-]+))?')
_STAGE_LOCATION = re.compile(rf'^{_STAGE_LOCATION_BODY}$')

_TREE_MANIFEST = '.kumo-stage-tree.json'
_SUCCESS = '_SUCCESS'
_COPY_FILES_LIMIT = 1000
# COPY FILES rejects source files larger than 5 GB. PUT accepts larger files,
# so fail before promotion instead of leaving a snapshot that can never be
# committed. Snowflake expresses that limit as 5,368,709,120 bytes.
_COPY_FILES_MAX_BYTES = 5 * 1024**3


@lru_cache(maxsize=2048)
def _split_normalized_stage_location(normalized: str) -> Tuple[str, str, str]:
    r"""Parses one normalized stage location, caching its validated shape."""
    match = _STAGE_LOCATION.fullmatch(normalized)
    if match is None:
        raise ValueError(
            f"Snowflake stage location {normalized!r} must be a fully-qualified "
            "named stage such as '@DB.SCHEMA.KUMO_ARTIFACTS/prefix'. User "
            "and table stages are not supported because COPY FILES cannot "
            "promote snapshots through them.")
    prefix = match.group('prefix') or ''
    if any(part in ('', '.', '..') for part in prefix.split('/') if prefix):
        raise ValueError(
            f'Snowflake stage prefix {prefix!r} must not contain empty, dot, '
            'or parent-directory components.')
    return match.group('stage'), prefix, match.group('name')


def _split_stage_location(location: str) -> Tuple[str, str, str]:
    r"""Returns the stage, normalized prefix, and stage identifier."""
    if not isinstance(location, str):
        raise TypeError(f'a Snowflake stage location must be a string, got '
                        f'{type(location).__name__}')
    return _split_normalized_stage_location(location.rstrip('/'))


def _stage_join(location: str, *parts: str, directory: bool = False) -> str:
    r"""Joins safe relative components onto a validated stage location."""
    stage, prefix, _ = _split_stage_location(location)
    clean = [prefix] if prefix else []
    for part in parts:
        value = str(part).replace('\\', '/').strip('/')
        if not value:
            continue
        if any(piece in ('', '.', '..') for piece in value.split('/')):
            raise ValueError(f'Unsafe Snowflake stage path component {part!r}.')
        clean.append(value)
    suffix = posixpath.join(*clean) if clean else ''
    joined = f'{stage}/{suffix}' if suffix else stage
    return joined + ('/' if directory else '')


def _safe_relative(value: str) -> str:
    r"""Normalizes a stage-relative file name and refuses traversal."""
    relative = value.replace('\\', '/').strip('/')
    if not relative or any(part in ('', '.', '..')
                           for part in relative.split('/')):
        raise ValueError(f'Unsafe Snowflake staged file name {value!r}.')
    return relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_stage_name(name: str, location: str) -> Optional[str]:
    r"""Returns a LIST row's name relative to the location being listed.

    Snowflake may prefix an internal-stage result with the stage name, while
    external storage can return a provider path. Searching for the exact
    caller-supplied prefix handles both shapes without depending on the
    current database or identifier case.
    """
    _, prefix, stage_identifier = _split_stage_location(location)
    value = str(name).replace('\\', '/').rstrip('/')
    stage_prefix = resolved(stage_identifier) + '/'
    has_stage_prefix = (
        value.startswith(stage_prefix) if stage_identifier.startswith('"') else
        value[:len(stage_prefix)].casefold() == stage_prefix.casefold())
    if has_stage_prefix:
        value = value[len(stage_prefix):]
    if prefix:
        if value == prefix:
            return ''
        if value.startswith(prefix + '/'):
            return value[len(prefix) + 1:]
        marker = '/' + prefix + '/'
        offset = value.find(marker)
        if offset >= 0:
            return value[offset + len(marker):]
        return None
    # Every location used by this module has a prefix by the time LIST runs,
    # but retain a useful fallback for a caller listing a bare stage.
    return value.split('/', 1)[1] if '/' in value else value


@dataclass(frozen=True)
class _StageObject:
    relative: str
    size: int


class _StageFiles:
    r"""The file operations injected into ``LocalSnapshotStore``."""
    def __init__(self, session: SnowflakeSession, parallel: int) -> None:
        _require_snowpark(session)
        self._session = session
        self._parallel = parallel

    def _list(self, location: str, *,
              tree: bool = True) -> Dict[str, _StageObject]:
        rows = _rows(self._session,
                     f'LIST {_stage_join(location, directory=tree)}')
        objects: Dict[str, _StageObject] = {}
        for row in rows:
            relative = _relative_stage_name(str(row[0]), location)
            if relative is None:
                continue
            # An exact-file LIST has an empty relative name. Preserve it for
            # `exists`; a tree LIST never produces one.
            key = relative if not relative else _safe_relative(relative)
            if key in objects:
                raise ValueError(
                    f'Snowflake listed {key!r} more than once under '
                    f'{location!r}.')
            objects[key] = _StageObject(key, int(row[1]))
        return objects

    def exists(self, location: str) -> bool:
        r"""Whether exactly :obj:`location`, rather than its prefix, exists."""
        return '' in self._list(location, tree=False)

    def preflight(self, location: str) -> None:
        r"""Checks stage type and every operation snapshot publication uses.

        The capability probe is deliberately operational: grants inherited
        through role hierarchies are difficult to infer correctly from grant
        metadata, while a tiny PUT/COPY/LIST/REMOVE cycle proves the effective
        session can perform the actual workflow. Its UUID-scoped prefix is
        removed in ``finally`` and never overlaps artifact data. A hard
        process termination can leave that disposable prefix behind; it is
        safe to remove because preflight never writes artifact data there.
        """
        match = _STAGE_LOCATION.fullmatch(location)
        assert match is not None
        database_sql = match.group('database')
        schema_sql = match.group('schema')
        stage_name = resolved(match.group('name'))
        database_name = resolved(database_sql)
        schema_name = resolved(schema_sql)
        self._session.sql(
            f'SHOW STAGES IN SCHEMA {database_sql}.{schema_sql}').collect()
        # SHOW output grows over time, so project stable named columns instead
        # of relying on their positional indexes in Snowflake's row shape.
        rows = _rows(
            self._session,
            'SELECT "name", "database_name", "schema_name", "type" '
            'FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))')
        matches = [
            row for row in rows if len(row) == 4 and str(row[0]) == stage_name
            and str(row[1]) == database_name and str(row[2]) == schema_name
        ]
        if not matches:
            raise ValueError(
                f'Snowflake stage {match.group("stage")!r} does not exist or '
                'is not visible to the registered session role.')
        stage_type = str(matches[0][3]).upper()
        if 'INTERNAL' not in stage_type:
            raise ValueError(
                f'Snowflake stage {match.group("stage")!r} is {stage_type!r}; '
                'StageArtifactLocation requires a named internal stage.')

        run_root = _stage_join(location, '.kumo-preflight', uuid.uuid4().hex)
        source = _stage_join(run_root, 'source')
        target = _stage_join(run_root, 'target')
        with tempfile.TemporaryDirectory(prefix='kumo-stage-preflight-') as tmp:
            probe = Path(tmp) / 'probe'
            probe.write_bytes(b'kumo-stage-preflight\n')
            try:
                self._put(probe, source)
                if not self.exists(_stage_join(source, probe.name)):
                    raise RuntimeError(
                        f'PUT to Snowflake stage {location!r} completed but '
                        'the probe was not visible to LIST.')
                self._copy_files(source, target, [probe.name])
                copied = _stage_join(target, probe.name)
                if not self.exists(copied):
                    raise RuntimeError(
                        f'COPY FILES in Snowflake stage {location!r} completed '
                        'but its target was not visible to LIST.')
                downloaded = Path(tmp) / 'downloaded' / probe.name
                self._get(copied, downloaded, probe.stat().st_size)
                if downloaded.read_bytes() != probe.read_bytes():
                    raise RuntimeError(
                        f'GET from Snowflake stage {location!r} returned '
                        'different probe contents.')
            finally:
                self.remove(run_root)

    def copy_tree(self, source: str, target: str) -> None:
        r"""Uploads local-to-stage or downloads stage-to-local."""
        source_is_stage = str(source).startswith('@')
        target_is_stage = str(target).startswith('@')
        if source_is_stage == target_is_stage:
            raise ValueError(
                f'A stage tree copy needs one local and one stage path; got '
                f'{source!r} -> {target!r}.')
        if target_is_stage:
            self._upload_tree(Path(source), target)
        else:
            self._download_tree(source, Path(target))

    def _put(self, source: Path, target_dir: str) -> None:
        results = self._session.file.put(str(source), target_dir, parallel=1,
                                         auto_compress=False,
                                         source_compression='NONE',
                                         overwrite=True)
        if not results:
            raise RuntimeError(
                f'Snowflake reported no result while uploading {source} to '
                f'{target_dir}.')

    def _upload_tree(self, source: Path, target: str) -> None:
        if not source.is_dir():
            raise ValueError(f'No local artifact tree exists at {source}.')
        root = source.resolve()
        files: List[Tuple[str, Path]] = []
        directories: List[str] = []
        for path in sorted(source.rglob('*')):
            relative = path.relative_to(source).as_posix()
            if path.is_dir():
                directories.append(relative)
                continue
            if not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(
                    f'Artifact file {path} resolves outside {source}; stage '
                    'publication does not follow escaping symlinks.')
            files.append((_safe_relative(relative), path))

        with tempfile.TemporaryDirectory(prefix='kumo-stage-manifest-') as tmp:
            manifest = Path(tmp) / _TREE_MANIFEST
            manifest.write_text(
                json.dumps(
                    {
                        'directories': directories,
                        'files': {
                            relative: {
                                'size': path.stat().st_size,
                                'sha256': _sha256(path),
                            }
                            for relative, path in files
                        },
                    }, sort_keys=True) + '\n', encoding='utf-8')
            # Upload order is irrelevant: promotion copies `_SUCCESS` last,
            # and readers treat only that marker as a committed snapshot.
            entries = files + [(_TREE_MANIFEST, manifest)]

            def upload(entry: Tuple[str, Path]) -> None:
                relative, path = entry
                parent = posixpath.dirname(relative)
                self._put(path, _stage_join(target, parent))

            try:
                _in_parallel(upload, entries, self._parallel, 'kumo-stage-put')
                expected = {
                    relative: path.stat().st_size
                    for relative, path in entries
                }
                actual = {
                    name: obj.size
                    for name, obj in self._list(target).items()
                }
                if set(actual) != set(expected):
                    raise RuntimeError(
                        f'Uploading {source} to {target} produced a different '
                        f'file set: expected {sorted(expected)}, got '
                        f'{sorted(actual)}.')
            except Exception:
                self.remove(target)
                raise

    def _get(self, source: str, target: Path,
             expected_size: Optional[int] = None) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        results = self._session.file.get(source, str(target.parent), parallel=1)
        if not results:
            raise RuntimeError(
                f'Snowflake reported no result while downloading {source}.')
        if not target.is_file():
            raise RuntimeError(
                f'Snowflake reported a download for {source}, but no file '
                f'landed at {target}.')
        if (expected_size is not None
                and target.stat().st_size != expected_size):
            raise RuntimeError(
                f'Downloading {source} produced {target.stat().st_size} '
                f'bytes, but the stage reported {expected_size}.')

    def _download_tree(self, source: str, target: Path) -> None:
        objects = self._list(source)
        if not objects:
            raise RuntimeError(
                f'Snowflake stage tree {source!r} was present when restore '
                'started but contains no files.')
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

        def download(item: Tuple[str, _StageObject]) -> None:
            relative, _ = item
            if not relative:
                raise ValueError(
                    f'Cannot restore exact stage object {source!r} as a tree.')
            destination = target / _safe_relative(relative)
            self._get(_stage_join(source, relative), destination)

        try:
            _in_parallel(download, list(objects.items()), self._parallel,
                         'kumo-stage-get')
            manifest = target / _TREE_MANIFEST
            if not manifest.is_file():
                raise RuntimeError(
                    f'Snowflake stage tree {source!r} has no '
                    f'{_TREE_MANIFEST}; it was not published completely by '
                    'StageArtifactLocation.')
            payload = json.loads(manifest.read_text(encoding='utf-8'))
            expected = payload.get('files') or {}
            actual = {
                path.relative_to(target).as_posix(): {
                    'size': path.stat().st_size,
                    'sha256': _sha256(path),
                }
                for path in target.rglob('*')
                if path.is_file() and path != manifest
            }
            if actual != expected:
                changed = sorted(name for name in set(expected) | set(actual)
                                 if expected.get(name) != actual.get(name))
                raise RuntimeError(
                    f'Restoring {source} failed its integrity check for: '
                    f'{changed}.')
            for relative in payload.get('directories') or []:
                (target / _safe_relative(str(relative))).mkdir(
                    parents=True, exist_ok=True)
            manifest.unlink()
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise

    def replace_tree(self, source: str, target: str) -> None:
        r"""Promotes a temporary stage prefix, committing `_SUCCESS` last."""
        if not source.startswith('@') or not target.startswith('@'):
            raise ValueError('Snowflake stage promotion needs two stage paths.')
        if self.exists(_stage_join(target, _SUCCESS)):
            self.remove(source)
            return

        objects = self._list(source)
        if _SUCCESS not in objects:
            raise RuntimeError(
                f'Cannot publish {source!r}: it has no {_SUCCESS} marker.')
        oversized = [
            name for name, obj in objects.items()
            if obj.size > _COPY_FILES_MAX_BYTES
        ]
        if oversized:
            self.remove(source)
            raise ValueError(
                'Snowflake COPY FILES cannot promote files larger than 5 GB; '
                f'{source!r} contains oversized files: {oversized!r}.')
        self.remove(target)
        ordinary = [name for name in objects if name != _SUCCESS]
        try:
            self._copy_files(source, target, ordinary)
            self._copy_files(source, target, [_SUCCESS])
            published = self._list(target)
            expected = {name: obj.size for name, obj in objects.items()}
            actual = {name: obj.size for name, obj in published.items()}
            if actual != expected:
                raise RuntimeError(
                    f'Publishing {target} produced a different file set: '
                    f'expected {sorted(expected)}, got {sorted(actual)}.')
        except Exception:
            # Never remove a tree another publisher committed while this one
            # was failing. Without a marker, the prefix is only our partial
            # copy and is safe to clean up.
            if not self.exists(_stage_join(target, _SUCCESS)):
                self.remove(target)
            raise
        finally:
            self.remove(source)

    def _copy_files(self, source: str, target: str,
                    relatives: List[str]) -> None:
        for start in range(0, len(relatives), _COPY_FILES_LIMIT):
            chunk = relatives[start:start + _COPY_FILES_LIMIT]
            if not chunk:
                continue
            names = ', '.join("'" + _quoted_literal(name) + "'"
                              for name in chunk)
            _rows(
                self._session,
                f'COPY FILES INTO {_stage_join(target, directory=True)} '
                f'FROM {_stage_join(source, directory=True)} FILES = ({names}) '
                'DETAILED_OUTPUT = TRUE')

    def remove(self, location: str) -> None:
        r"""Removes every object under a stage location."""
        _rows(self._session, f'REMOVE {_stage_join(location, directory=True)}')


@dataclass(frozen=True)
class StageArtifactLocation:
    r"""Publishes verified snapshots to a Snowflake named internal stage.

    Args:
        root: Fully-qualified stage URI, optionally with a prefix, such as
            ``@PROD.ML.KUMO_ARTIFACTS/local-backend``. The stage must already
            exist. Its role needs database/schema usage plus stage read/write.
        identity: A name, local identity ID, or
            :class:`~kumoai_studio.identity.Identity` created through
            :func:`kumoai_studio.create_identity`. Its role needs the stage
            privileges below.
        parallel: Concurrent file transfers, from 1 through 99.

    The local working artifact directory is derived from ``Storage.scratch``;
    callers never pass a stage URI to DiskGraph, pandas, Torch, or
    ``kumo_core``.
    """
    root: str
    identity: IdentityRef
    parallel: int = 16

    def __post_init__(self) -> None:
        stage, prefix, _ = _split_stage_location(self.root)
        object.__setattr__(self, 'root',
                           f'{stage}/{prefix}' if prefix else stage)
        if not isinstance(self.parallel, int) or not 1 <= self.parallel <= 99:
            raise ValueError('parallel must be an integer from 1 through 99')

    def local_root(self, scratch: str) -> str:
        r"""Returns a cache keyed by a 12-hex SHA256 truncation of the URI."""
        key = hashlib.sha256(self.root.encode('utf-8')).hexdigest()[:12]
        return str(Path(scratch) / 'artifacts' / f'snowflake-stage-{key}')

    def _session(self) -> SnowflakeSession:
        r"""A session on the identity alone, with no connector target.

        A stage is addressed by its own fully-qualified name, so this
        connection needs no database or schema of its own -- and must not
        borrow a connector's, since a stage location is independent of
        whatever table happened to be read last.
        """
        name = identity_store.resolve(self.identity)
        if name is None:
            known = ', '.join(repr(item) for item in identity_store.names()) or 'none'
            raise ValueError(
                f'Snowflake stage {self.root!r} names identity '
                f'{self.identity!r}, which does not exist in this process. '
                f'Create it with kumoai_studio.create_identity first. '
                f'Existing: {known}.')
        session = _sessions.get(identity_store.get(name), None)
        _require_snowpark(session)
        return session

    def preflight(self) -> None:
        r"""Validates stage type and effective transfer privileges."""
        _StageFiles(self._session(), self.parallel).preflight(self.root)

    def restore(self, store: SnapshotStore[_Handle],
                handle: _Handle) -> Optional[IngestResult]:
        files = _StageFiles(self._session(), self.parallel)
        restored = store.restore(handle, self.root, exists=files.exists,
                                 copy_tree=files.copy_tree)
        if restored is None:
            return None
        return rebased(store, handle)

    def checkpoint(self, store: SnapshotStore[_Handle],
                   handle: _Handle) -> Optional[str]:
        files = _StageFiles(self._session(), self.parallel)
        return store.checkpoint(handle, self.root, exists=files.exists,
                                copy_tree=files.copy_tree,
                                replace_tree=files.replace_tree)

    def durable_path(self, store: SnapshotStore[_Handle],
                     handle: _Handle) -> Optional[str]:
        return store.durable_path(handle, self.root)


__all__ = ['StageArtifactLocation']
