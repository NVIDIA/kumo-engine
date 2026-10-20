# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discover, validate, and content-address the installed Kumo fat wheel."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import re
import urllib.parse
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from ._ml_job_stage import StageStore
from .sql import quote_identifier

logger = logging.getLogger(__name__)

_READ_BUFFER_BYTES = 64 * 1024 * 1024
_REQUIRED_PACKAGES = ('kumoai/', 'kumo_core/', 'kumo_snowflake/')
_REQUIRED_COMPONENTS = frozenset({
    'bbhash',
    'diskgraph',
    'kumo-api',
    'kumo-core',
    'kumo-pql',
    'kumo-snowflake',
    'kumoai',
    'kumoml-enterprise',
    'pyg-lib',
    'torch-geometric',
    'torch-scatter',
})
_ENGINE_PACKAGES = ('kumo/engine/', 'kumo_core/', 'kumo_snowflake/')
_ENGINE_COMPONENTS = frozenset({
    'diskgraph', 'kumo-engine', 'kumo-core', 'kumo-pql',
    'kumo-snowflake', 'torch-geometric',
})
_REQUIRED_RUNTIME_REQUIREMENTS = frozenset({'pytorch-frame==0.3.0'})
_WHEEL_NAME = re.compile(r'[A-Za-z0-9_.+-]+\.whl')


@dataclass(frozen=True)
class InstalledWheel:
    path: Path
    sha256: str
    md5: str
    size: int

    @property
    def name(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class StagedWheel:
    name: str
    sha256: str


def _canonical_distribution_name(name: object) -> str:
    """Apply the normalization used by Python distribution metadata."""
    return re.sub(r'[-_.]+', '-', str(name)).lower()


def _direct_url(
        distribution: importlib.metadata.Distribution) -> dict[str, Any] | None:
    try:
        text = distribution.read_text('direct_url.json')
        return None if text is None else json.loads(text)
    except (OSError, ValueError, TypeError):
        return None


def _archive_hash(payload: dict[str, Any]) -> str | None:
    hashes = payload.get('archive_info', {}).get('hashes', {})
    value = hashes.get('sha256') if isinstance(hashes, dict) else None
    if isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{64}', value):
        return value.lower()
    return None


def discover_installed_wheel() -> InstalledWheel:
    """Find the retained wheel from which the fat bundle was installed.

    Reconstructing a wheel from installed files would be a second build path.
    The supported contract therefore requires pip's PEP 610 provenance to
    point at the original wheel produced by the engine wheel builder.
    """
    candidates: list[tuple[Path, str | None]] = []
    for distribution in importlib.metadata.distributions():
        files = distribution.files or ()
        if not any(str(path) == 'kumo_bundle_manifest.json' for path in files):
            continue
        direct = _direct_url(distribution)
        if direct is None:
            continue
        url = direct.get('url')
        if not isinstance(url, str):
            continue
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != 'file':
            continue
        path = Path(urllib.parse.unquote(parsed.path))
        if path.suffix == '.whl' and path.is_file():
            candidates.append((path, _archive_hash(direct)))
    if len(candidates) != 1:
        raise RuntimeError(
            'submit_ml_job requires Kumo to be installed from one '
            'retained Kumo wheel produced by '
            'packages/kumo-engine/scripts/build_wheel.sh; found '
            f'{len(candidates)} candidate wheels')
    return validate_wheel(*candidates[0])


def _hashes(handle: BinaryIO) -> tuple[str, str, int]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(  # noqa: S324 - Snowflake LIST exposes MD5 metadata.
        usedforsecurity=False)
    size = 0
    while True:
        chunk = handle.read(_READ_BUFFER_BYTES)
        if not chunk:
            break
        size += len(chunk)
        sha256.update(chunk)
        md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest(), size


def validate_wheel(path: Path,
                   expected_sha256: str | None = None) -> InstalledWheel:
    """Validate a readable Kumo fat wheel and its bundled components."""
    path = path.resolve()
    if not path.is_file() or not _WHEEL_NAME.fullmatch(path.name):
        raise RuntimeError(f'{path} is not a readable wheel file')
    try:
        # Match the proven ML-Jobs wheel path: inspect the archive directory
        # and bundle manifest without decompressing every member. A full
        # ``ZipFile.testzip()`` pass performs thousands of high-latency reads
        # on Snowflake's workspace/FUSE mount. The content digest below, plus
        # the digest verification after native stage delivery, binds every
        # byte without that multi-minute validation pass.
        with open(path, 'rb', buffering=_READ_BUFFER_BYTES) as wheel_handle:
            with zipfile.ZipFile(wheel_handle) as archive:
                names = archive.namelist()
                engine_wheel = any(name.startswith('kumo/engine/')
                                   for name in names)
                required_packages = (_ENGINE_PACKAGES if engine_wheel
                                     else _REQUIRED_PACKAGES)
                missing_packages = [
                    prefix for prefix in required_packages
                    if not any(name.startswith(prefix) for name in names)
                ]
                if missing_packages:
                    raise RuntimeError(
                        f'{path.name} is not self-contained; missing '
                        f'{missing_packages!r}')
                manifest = json.loads(archive.read('kumo_bundle_manifest.json'))
    except (OSError, zipfile.BadZipFile, KeyError, ValueError) as error:
        raise RuntimeError(
            f'{path.name} is not a valid Kumo fat wheel: {error}') from error

    components = manifest.get('components') if isinstance(manifest,
                                                          dict) else None
    if not isinstance(components, list):
        raise RuntimeError(f'{path.name} has no valid Kumo bundle manifest')
    component_names = {
        _canonical_distribution_name(component.get('name', ''))
        for component in components if isinstance(component, dict)
    }
    required_components = (_ENGINE_COMPONENTS if engine_wheel
                           else _REQUIRED_COMPONENTS)
    missing_components = sorted(required_components - component_names)
    if missing_components:
        raise RuntimeError(
            f'{path.name} is missing required fat-wheel components '
            f'{missing_components!r}')
    requirements = manifest.get('requires' if engine_wheel
                                else 'requires_at_install')
    if not isinstance(requirements, list) or not all(
            isinstance(requirement, str) for requirement in requirements):
        raise RuntimeError(
            f'{path.name} has no valid install-time requirements')
    declared_requirements = {
        requirement.lower().strip()
        for requirement in requirements
    }
    missing_requirements = ([] if 'pytorch-frame' in component_names else
                            sorted(_REQUIRED_RUNTIME_REQUIREMENTS -
                                   declared_requirements))
    if missing_requirements:
        raise RuntimeError(
            f'{path.name} is missing required runtime dependencies '
            f'{missing_requirements!r}')

    if expected_sha256 is not None:
        # pip already verified and retained this PEP 610 digest when it
        # installed the wheel. Avoid rereading a large workspace/FUSE file
        # here; stage_wheel verifies the server-side object against this
        # digest before it can be used by an ML Job.
        sha256 = expected_sha256.lower()
        md5 = ''
        size = path.stat().st_size
    else:
        with open(path, 'rb', buffering=_READ_BUFFER_BYTES) as handle:
            sha256, md5, size = _hashes(handle)
    return InstalledWheel(path=path, sha256=sha256, md5=md5, size=size)


def _listed_wheel(
    rows: list[dict[str, Any]],
    wheel: InstalledWheel,
) -> dict[str, Any] | None:
    matches = [
        row for row in rows
        if str(row.get('name', '')).rsplit('/', 1)[-1] == wheel.name
    ]
    if len(matches) != 1:
        return None
    row = matches[0]
    try:
        size = int(row.get('size'))
    except (TypeError, ValueError):
        return None
    if size <= 0:
        return None
    return row


def _listing_matches_verified_object(
    rows: list[dict[str, Any]],
    wheel: InstalledWheel,
    stage_md5: object,
    stage_size: object,
) -> bool:
    row = _listed_wheel(rows, wheel)
    expected = str(stage_md5 or '').lower()
    try:
        expected_size = int(stage_size)
        actual_size = int(row.get('size')) if row is not None else -1
    except (TypeError, ValueError):
        return False
    return (row is not None and expected_size > 0
            and actual_size == expected_size and bool(expected)
            and str(row.get('md5', '')).lower() == expected)


def _workspace_path(path: Path) -> bool:
    value = str(path).replace('\\', '/')
    if not value.startswith('/'):
        return False
    mount = os.environ.get('SNOWFLAKE_MOUNT_ROOT')
    if mount:
        root = mount.replace('\\', '/').rstrip('/')
        if value == root or value.startswith(f'{root}/'):
            return True
    return any(part.lower() in {'filesystem', 'user_workspace'}
               for part in value.split('/'))


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _project_sources(store: StageStore,
                     wheel: InstalledWheel) -> tuple[str, ...]:
    """Find immutable Notebook Project directories holding this wheel."""
    database = os.environ.get('NOTEBOOK_PROJECT_DATABASE')
    schema = os.environ.get('NOTEBOOK_PROJECT_SCHEMA')
    main_file = os.environ.get('SNOWFLAKE_NON_INTERACTIVE_NOTEBOOK_NAME')
    if not database or not schema or not main_file:
        return ()
    parts = [part for part in main_file.split('/') if part not in {'', '.'}]
    if not parts or '..' in parts:
        return ()
    relative_dir = '/'.join(parts[:-1])
    qualified_schema = (
        f'{quote_identifier(database)}.{quote_identifier(schema)}')
    try:
        projects = store.session.sql(
            f'SHOW NOTEBOOK PROJECTS IN SCHEMA {qualified_schema}').collect()
    except Exception:
        logger.warning(
            'Could not discover Notebook Project wheel sources; falling '
            'back to verified stage upload',
            exc_info=True,
        )
        return ()
    candidates: list[str] = []
    for project in projects:
        values = _file_values(project)
        name = values.get('name')
        if not isinstance(name, str) or not name:
            continue
        project_name = f'{qualified_schema}.{quote_identifier(name)}'
        try:
            description = store.session.sql(
                f'DESCRIBE NOTEBOOK PROJECT {project_name}').collect()
        except Exception:
            logger.warning('Could not inspect Notebook Project %s',
                           project_name, exc_info=True)
            continue
        version_uri: str | None = None
        for described_row in description:
            described = _file_values(described_row)
            direct = described.get('default_version_location_uri')
            if isinstance(direct, str):
                version_uri = direct
                break
            if str(described.get(
                    'property',
                    '')).lower() == ('default_version_location_uri'):
                value = described.get('value')
                if isinstance(value, str):
                    version_uri = value
                    break
        if not version_uri or not version_uri.startswith(
                'snow://notebook project/'):
            continue
        directory = version_uri.rstrip('/')
        if relative_dir:
            directory += '/' + relative_dir
        source = f'{directory}/{wheel.name}'
        try:
            rows = store.session.sql(f'LIST {_sql_string(source)}').collect()
        except Exception:
            logger.warning('Could not inspect Notebook Project wheel %s',
                           source, exc_info=True)
            continue
        if _listed_wheel([_file_values(row) for row in rows], wheel):
            candidates.append(directory)
    return tuple(candidates)


def _file_values(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        values = value
    else:
        as_dict = getattr(value, 'as_dict', None)
        values = as_dict() if callable(as_dict) else {
            name: getattr(value, name)
            for name in ('name', 'size', 'md5', 'status', 'target')
            if hasattr(value, name)
        }
    return {str(key).lower(): item for key, item in values.items()}


def _stage_sha256(store: StageStore, uri: str) -> str:
    digest = hashlib.sha256()
    stream = store.session.file.get_stream(uri, decompress=False)
    try:
        while chunk := stream.read(_READ_BUFFER_BYTES):
            digest.update(chunk)
    finally:
        close = getattr(stream, 'close', None)
        if callable(close):
            close()
    return digest.hexdigest()


def _remove_uri(store: StageStore, uri: str) -> None:
    try:
        store.session.sql(f'REMOVE {uri}').collect()
    except Exception:
        # Candidate files are never executable and use UUID paths. Cleanup is
        # best effort so a cleanup outage cannot hide the verified fast-path
        # failure and its safe streaming fallback.
        pass


def _copy_project_wheel(
    store: StageStore,
    sources: tuple[str, ...],
    wheel: InstalledWheel,
    directory: str,
) -> bool:
    """Server-copy and cryptographically bind a Notebook Project wheel."""
    target = store.uri(f'{directory}/{wheel.name}')
    for source_dir in sources:
        candidate_relative = f'{directory}/.candidate-{uuid.uuid4().hex}'
        candidate = store.uri(f'{candidate_relative}/{wheel.name}')
        try:
            rows = store.session.sql(
                f'COPY FILES INTO {store.uri(candidate_relative)}/ FROM '
                f'{_sql_string(source_dir + "/")} FILES = '
                f'({_sql_string(wheel.name)}) DETAILED_OUTPUT = TRUE').collect(
                )
            copied = [
                str(_file_values(row).get('file', '')).rsplit('/', 1)[-1]
                for row in rows
            ]
            if copied != [wheel.name]:
                continue
            if _stage_sha256(store, candidate) != wheel.sha256:
                continue
            store.copy_file(f'{candidate_relative}/{wheel.name}', directory)
            if _stage_sha256(store, target) == wheel.sha256:
                return True
            _remove_uri(store, target)
        except Exception:
            continue
        finally:
            _remove_uri(store, candidate)
    return False


def stage_wheel(store: StageStore, wheel: InstalledWheel) -> StagedWheel:
    """Stage ``wheel`` once without first copying a workspace/FUSE file."""
    directory = f'_kumo_ml_jobs/wheels/sha256/{wheel.sha256}'
    relative = f'{directory}/{wheel.name}'
    marker_relative = f'{directory}/wheel.json'
    marker = store.get_json(marker_relative, missing_ok=True)
    cached = (isinstance(marker, dict) and marker.get('sha256') == wheel.sha256
              and marker.get('source_md5') == wheel.md5
              and marker.get('source_size') == wheel.size
              and marker.get('name') == wheel.name
              and _listing_matches_verified_object(store.list(directory), wheel,
                                                   marker.get('stage_md5'),
                                                   marker.get('stage_size')))
    if cached:
        try:
            cached = (_stage_sha256(store, store.uri(relative)) == wheel.sha256)
        except Exception:
            cached = False
    if cached:
        return StagedWheel(wheel.name, wheel.sha256)

    copied = False
    if _workspace_path(wheel.path):
        sources = _project_sources(store, wheel)
        copied = bool(sources) and _copy_project_wheel(store, sources, wheel,
                                                       directory)
    if not copied:
        # A large explicit buffer coalesces the connector's small logical
        # reads into a handful of reads from the Notebook workspace/FUSE
        # mount. It streams straight to the stage and never creates a second
        # local copy; this is also the ordinary non-workspace upload path.
        with open(wheel.path, 'rb', buffering=_READ_BUFFER_BYTES) as handle:
            result = store.session.file.put_stream(
                handle,
                store.uri(relative),
                auto_compress=False,
                overwrite=True,
                parallel=8,
                source_compression='NONE',
            )
        if result is None:
            raise RuntimeError(
                f'Snowflake reported no result while staging {wheel.name!r}')
        values = _file_values(result)
        status = str(values.get('status', '')).upper()
        target = str(values.get('target', '')).rsplit('/', 1)[-1]
        if status and status not in {'UPLOADED', 'SKIPPED'}:
            raise RuntimeError(
                f'Snowflake did not confirm staging {wheel.name!r}: '
                f'{values!r}')
        if target and target != wheel.name:
            raise RuntimeError(
                f'Snowflake staged {target!r}; expected {wheel.name!r}')
        if _stage_sha256(store, store.uri(relative)) != wheel.sha256:
            _remove_uri(store, store.uri(relative))
            raise RuntimeError(
                f'Snowflake staged {wheel.name!r}, but its content did not '
                'match the installed Kumo wheel')
    # LIST treats a fully specified object name as a prefix/pattern on some
    # Snowflake surfaces; a PEP 440 ``+`` in a wheel filename can therefore
    # produce an empty result even after GET verified the object. List the
    # content-addressed directory and select the exact basename ourselves.
    listed = _listed_wheel(store.list(directory), wheel)
    stage_md5 = None if listed is None else listed.get('md5')
    stage_size = None if listed is None else listed.get('size')
    if listed is None or not isinstance(stage_md5, str) or not stage_md5:
        raise RuntimeError(
            f'Snowflake did not list the verified staged wheel {wheel.name!r}')
    store.put_json(
        marker_relative, {
            'schema_version': 1,
            'name': wheel.name,
            'sha256': wheel.sha256,
            'source_md5': wheel.md5,
            'stage_md5': stage_md5.lower(),
            'source_size': wheel.size,
            'stage_size': int(stage_size),
        })
    return StagedWheel(wheel.name, wheel.sha256)
