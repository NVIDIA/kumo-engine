# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable lifecycle primitives (locks, atomic manifests, fail-closed
stale-manifest inspection).

Owns no process/port/cgroup/data cleanup.
"""

import contextlib
import fcntl
import json
import os
import re
import secrets
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence

_MANIFEST_VERSION = 1
_PHASES = frozenset({'planned', 'launching', 'running', 'stopping'})
_CHILD_ID_RE = re.compile(r'^[a-z][a-z0-9-]{0,47}$')


class DevserverLifecycleError(RuntimeError):
    """A lifecycle operation could not safely establish its preconditions."""


class DevserverLifecycleLockError(DevserverLifecycleError):
    """Another coordinator currently owns an advisory lifecycle lock."""


@dataclass(frozen=True)
class DevserverManifest:
    """A manifest describing only resources owned by one coordinator token.

    ``children`` maps a stable child devserver ID to metadata.  If a child
    records resources, they live in ``children[id]["resources"]`` and every
    resource must carry the manifest owner token.  This prevents a later
    recovery operation from treating an unowned resource as disposable.
    """

    phase: str
    owner_token: str
    children: Mapping[str, Mapping[str, Any]]
    version: int = _MANIFEST_VERSION

    def to_dict(self) -> Dict[str, Any]:
        _validate_manifest(self)
        return {
            'version': self.version,
            'phase': self.phase,
            'owner_token': self.owner_token,
            'children': dict(self.children),
        }


def new_owner_token() -> str:
    return secrets.token_urlsafe(32)


def new_manifest(
    children: Mapping[str, Mapping[str, Any]],
    phase: str = 'planned',
    owner_token: Optional[str] = None,
) -> DevserverManifest:
    """Build a validated initial manifest without writing it to disk."""
    manifest = DevserverManifest(phase=phase,
                                 owner_token=owner_token or new_owner_token(),
                                 children=children)
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: DevserverManifest) -> None:
    if manifest.version != _MANIFEST_VERSION:
        raise DevserverLifecycleError(
            f'unsupported devserver manifest version {manifest.version}')
    if manifest.phase not in _PHASES:
        raise DevserverLifecycleError(
            f'manifest phase must be one of {sorted(_PHASES)}')
    if not isinstance(manifest.owner_token, str) or not manifest.owner_token:
        raise DevserverLifecycleError('manifest owner_token must be non-empty')
    if not isinstance(manifest.children, Mapping):
        raise DevserverLifecycleError('manifest children must be an object')

    for child_id, child in manifest.children.items():
        if not isinstance(child_id, str) or not child_id:
            raise DevserverLifecycleError('manifest child IDs must be non-empty')
        if not isinstance(child, Mapping):
            raise DevserverLifecycleError(
                f'manifest child {child_id!r} must be an object')
        resources = child.get('resources', {})
        if not isinstance(resources, Mapping):
            raise DevserverLifecycleError(
                f'manifest child {child_id!r}.resources must be an object')
        for resource_name, resource in resources.items():
            if not isinstance(resource_name, str) or not resource_name:
                raise DevserverLifecycleError(
                    f'manifest child {child_id!r} has invalid resource name')
            if not isinstance(resource, Mapping):
                raise DevserverLifecycleError(
                    f'manifest resource {child_id}/{resource_name} must be an '
                    'object')
            if resource.get('owner_token') != manifest.owner_token:
                raise DevserverLifecycleError(
                    f'manifest resource {child_id}/{resource_name} does not '
                    'carry the manifest owner token')


def write_manifest_atomic(path: Path, manifest: DevserverManifest) -> None:
    """Atomically persist ``manifest`` with file and directory durability.

    The parent directory must already have been created by the coordinator.
    A failed write intentionally leaves its private temporary file rather than
    deleting any path during a safety-sensitive lifecycle operation.
    """
    _validate_manifest(manifest)
    path = Path(path)
    if not path.parent.is_dir():
        raise DevserverLifecycleError(
            f'manifest parent directory does not exist: {path.parent}')
    payload = (json.dumps(manifest.to_dict(), sort_keys=True,
                          separators=(',', ':')) + '\n').encode('utf-8')
    temporary = path.with_name(
        f'.{path.name}.{manifest.owner_token[:16]}.{os.getpid()}.tmp')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, 'wb', closefd=True) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        # Do not remove even a failed temporary manifest here.  It can be
        # inspected during manual recovery, and this module never deletes.
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_manifest(path: Path) -> DevserverManifest:
    """Read and validate a manifest."""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise DevserverLifecycleError(
            f'cannot read valid devserver manifest {path}') from error
    if not isinstance(data, dict):
        raise DevserverLifecycleError('manifest top-level value must be an object')
    try:
        manifest = DevserverManifest(version=data['version'],
                                     phase=data['phase'],
                                     owner_token=data['owner_token'],
                                     children=data['children'])
    except KeyError as error:
        raise DevserverLifecycleError(
            f'manifest is missing required key {error.args[0]!r}') from error
    _validate_manifest(manifest)
    return manifest


@contextlib.contextmanager
def advisory_lock(path: Path) -> Iterator[None]:
    """Hold one non-blocking advisory lock until the context exits."""
    path = Path(path)
    if not path.parent.is_dir():
        raise DevserverLifecycleError(
            f'lock parent directory does not exist: {path.parent}')
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DevserverLifecycleLockError(
                f'lifecycle lock is already held: {path}') from error
        yield
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def coordinator_locks(lock_directory: Path,
                      child_ids: Sequence[str]) -> Iterator[None]:
    """Acquire the global coordinator lock followed by sorted child locks.

    Sorting makes simultaneous operations involving multiple child devservers
    use one lock order.  The global lock serializes lifecycle transitions;
    child locks are retained for future operations that target one child.
    """
    lock_directory = Path(lock_directory)
    if len(child_ids) != len(set(child_ids)):
        raise DevserverLifecycleError('child IDs for lifecycle locks are duplicate')
    if any(not _CHILD_ID_RE.fullmatch(child_id) for child_id in child_ids):
        raise DevserverLifecycleError(
            'child IDs for lifecycle locks must be normalized safe IDs')
    child_paths = [lock_directory / f'child-{child_id}.lock'
                   for child_id in sorted(child_ids)]
    with advisory_lock(lock_directory / 'coordinator.lock'):
        with contextlib.ExitStack() as stack:
            for child_path in child_paths:
                stack.enter_context(advisory_lock(child_path))
            yield


class StaleManifestStatus(str, Enum):
    """Whether recovery has sufficient evidence to proceed to a later action."""

    VERIFIED = 'verified'
    AMBIGUOUS = 'ambiguous'


@dataclass(frozen=True)
class StaleManifestInspection:
    """A decision record; this module never performs recovery actions."""

    status: StaleManifestStatus
    reason: str
    verified_resources: tuple[str, ...] = ()


ResourceVerifier = Callable[[str, Mapping[str, Any], str], bool]


def inspect_stale_manifest(
    manifest: DevserverManifest,
    verifier: Optional[ResourceVerifier] = None,
) -> StaleManifestInspection:
    """Fail closed unless every recorded resource is proven owner-token owned.

    ``verifier`` receives ``(child_id/resource_name, resource, owner_token)``.
    It must independently establish that the live resource belongs to that
    token.  A verifier exception or false result makes the manifest ambiguous.
    """
    try:
        _validate_manifest(manifest)
    except DevserverLifecycleError as error:
        return StaleManifestInspection(StaleManifestStatus.AMBIGUOUS,
                                       f'invalid manifest: {error}')

    resources = []
    for child_id, child in manifest.children.items():
        for resource_name, resource in child.get('resources', {}).items():
            resources.append((f'{child_id}/{resource_name}', resource))
    if resources and verifier is None:
        return StaleManifestInspection(
            StaleManifestStatus.AMBIGUOUS,
            'recorded resources require an owner-token verifier')

    verified = []
    for resource_id, resource in resources:
        try:
            owned = verifier(resource_id, resource, manifest.owner_token)
        except Exception as error:  # verifier is external and must fail closed
            return StaleManifestInspection(
                StaleManifestStatus.AMBIGUOUS,
                f'owner-token verification raised for {resource_id}: {error}',
                tuple(verified))
        if not owned:
            return StaleManifestInspection(
                StaleManifestStatus.AMBIGUOUS,
                f'owner-token verification failed for {resource_id}',
                tuple(verified))
        verified.append(resource_id)
    return StaleManifestInspection(StaleManifestStatus.VERIFIED,
                                   'all recorded resources are owner-token verified',
                                   tuple(verified))
