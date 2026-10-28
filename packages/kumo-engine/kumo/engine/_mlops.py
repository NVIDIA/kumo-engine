# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The MLOps switch: :class:`MLOps` settings and the :func:`mlops` block."""

import contextlib
import contextvars
import os
import posixpath
import re
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    FrozenSet,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Tuple,
    Union,
)

from kumoapi.plugin import ModelRegistryLocation, check_model_alias

from kumo.engine._manifest import KINDS


@dataclass(frozen=True)
class _Root:
    connector: Any
    path: str
    _uri: str

    def location(self, relative: str) -> str:
        r"""``relative`` as a path on :attr:`connector`."""
        base = self.path.rstrip('/')
        return f'{base}/{relative}' if base else relative

    def uri(self, relative: str) -> str:
        r"""``relative`` as a full URI or path, for messages and records."""
        return f"{self._uri.rstrip('/')}/{relative}"


@dataclass(frozen=True)
class MLOps:
    r"""Where artifacts go and what records them, for :func:`mlops` blocks
    and ``mlops=`` arguments.

    Args:
        artifact_root: an ``s3://``, ``gs://``, ``file://`` or local path
            (``/Volumes/...`` on a Databricks cluster), or a ``(connector,
            path)`` pair for a store a URI cannot reach, such as a volume
            through a ``DatabricksConnector`` or a Snowflake stage.
        tracking: the experiment tracker, a
            ``kumo.engine.tracking.Tracking``. Its ``check(registry)``, if
            any, runs here, so a server or registry that is unreachable or
            refuses these credentials fails before any work.
        registry: a ``ModelRegistryLocation`` each fit registers its model
            at, in the tracker's registry: MLflow's (Unity Catalog, SageMaker
            MLflow) for ``MlflowTracking``, the Snowflake Model Registry for
            ``SnowflakeTracking``.
        store: the artifact kinds stored under the root when produced
            (``snapshot``, ``training_table``, ``model``);
            ``True`` for all, ``False`` for none.
        on_error: what a failure to track a run or register a model does:
            ``'warn'`` warns and the call returns; ``'raise'`` raises once
            the call has finished and stored its result, with the failure as
            the cause.

    In ``load_from_connector`` calls, an ``MLOps`` stands in for the
    connector, with paths relative to its artifact root.
    """

    artifact_root: Union[str, Tuple[Any, str]]
    tracking: Any = field(default=None, kw_only=True)
    registry: Any = field(default=None, kw_only=True)
    store: Union[bool, Iterable[str]] = field(default=True, kw_only=True)
    on_error: Literal['warn', 'raise'] = field(default='warn', kw_only=True)
    root: _Root = field(init=False, repr=False, compare=False)
    kinds: FrozenSet[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from kumo.engine.connector import (
            connector_for_root,
            is_file_destination,
        )

        if not self.artifact_root:
            raise ValueError('MLOps needs an artifact_root.')
        if self.on_error not in ('warn', 'raise'):
            raise ValueError(f"on_error must be 'warn' or 'raise', not "
                             f"{self.on_error!r}.")
        if isinstance(self.artifact_root, str):
            if self.artifact_root.startswith('@'):
                raise ValueError(
                    'A stage artifact_root needs its connector: pass '
                    f'(SnowflakeConnector(...), {self.artifact_root!r}).')
            root = _Root(connector_for_root(self.artifact_root), '',
                         self.artifact_root)
        else:
            connector, path = self.artifact_root
            if not is_file_destination(connector, path):
                raise ValueError(
                    f'{type(connector).__name__} cannot hold files at '
                    f'{path!r}; artifact_root needs a file location.')
            _require_within_root(connector, path)
            root = _Root(connector, path, path)
        kinds = (frozenset(KINDS) if self.store is True else
                 frozenset() if self.store is False else frozenset(self.store))
        unknown = sorted(kinds - set(KINDS))
        if unknown:
            raise ValueError(f'Unknown store kinds {unknown}; choose from '
                             f'{list(KINDS)}.')
        if kinds:
            _require_writable(root)
        if self.registry is not None:
            self._check_registry()
        check = getattr(self.tracking, 'check', None)
        if callable(check):
            check(self.registry)
        object.__setattr__(self, 'root', root)
        object.__setattr__(self, 'kinds', kinds)

    def _check_registry(self) -> None:
        from kumo.engine.trainer.output_config import (
            ModelOutputConfig,
            check_model_location,
        )

        registry = self.registry
        if not isinstance(registry, ModelRegistryLocation):
            raise TypeError(
                f'registry takes a ModelRegistryLocation, not '
                f'{type(registry).__name__}; the tracker registers it.')
        if registry.alias is not None:
            check_model_alias(registry.alias)
        if callable(getattr(self.tracking, 'log_model', None)):
            if registry.version is not None:
                raise ValueError('MLflow assigns registered model versions; '
                                 'leave ModelRegistryLocation.version unset.')
            return
        connector = getattr(self.tracking, 'connector', None)
        if connector is None:
            raise ValueError(
                'registry= registers through the tracker: pass '
                'tracking=MlflowTracking(...) or SnowflakeTracking(connector, '
                '...).')
        check_model_location(ModelOutputConfig(connector, location=registry))

    def wants(self, kind: str) -> bool:
        r"""Whether artifacts of ``kind`` are stored under the root."""
        return kind in self.kinds


def _require_within_root(connector: Any, path: str) -> None:
    from kumo_core.sources.object_store_sessions import ObjectStoreSessions

    sink = connector.sink_backend
    if not isinstance(sink, ObjectStoreSessions):
        return
    try:
        sink._resolved(connector, path)
    except ValueError as error:
        target = sink._root_for(connector)
        within = (f'root_dir {sink.scheme}://{target.path}'
                  if target is not None else 'target')
        name = type(connector).__name__
        raise ValueError(
            f'artifact_root {path!r} is outside this {name}\'s {within} '
            f'({error}), so nothing could be stored there. Give the root '
            f'a connector that contains it: ({name}(root_dir={path!r}, '
            f'identity=...), {path!r}).') from error


def _require_writable(root: _Root) -> None:
    from uuid import uuid4

    from kumo.engine.connector import file_store

    store = file_store(root.connector,
                       root.location(f'.kumo-write-check-{uuid4().hex}'))
    if store is None:
        return
    filesystem, probe = store
    try:
        if filesystem.type_name == 'local':
            existing = posixpath.dirname(probe)
            while not os.path.exists(existing):
                existing = posixpath.dirname(existing)
            if not os.access(existing, os.W_OK | os.X_OK):
                raise PermissionError(f'{existing} is not writable')
            return
        try:
            filesystem.open_output_stream(probe).close()
        except FileNotFoundError:  # a store with directories, before first use
            filesystem.create_dir(posixpath.dirname(probe), recursive=True)
            filesystem.open_output_stream(probe).close()
        filesystem.delete_file(probe)
    except Exception as error:
        raise PermissionError(
            f'kumo.MLOps cannot write under artifact_root '
            f'{root.uri("")!r}: {type(error).__name__}: {error}. Grant this '
            f'identity write and delete rights on it, or pick another '
            f'root.') from error


_CURRENT: contextvars.ContextVar[Optional[MLOps]] = contextvars.ContextVar(
    'kumo_mlops', default=None)


@contextlib.contextmanager
def mlops(settings: MLOps) -> Iterator[MLOps]:
    r"""Turns MLOps on for the calls in the block (in this thread or asyncio
    task): each snapshot, training table and model is stored
    under ``settings.artifact_root``. An inner block replaces an outer one;
    an ``mlops=`` argument replaces both for its call. Runs are tracked and
    models registered as ``settings`` says."""
    if not isinstance(settings, MLOps):
        raise TypeError('kumo.mlops() takes a kumo.MLOps')
    token = _CURRENT.set(settings)
    try:
        yield settings
    finally:
        _CURRENT.reset(token)


def resolve(settings: Optional[MLOps] = None) -> Optional[MLOps]:
    r"""The settings a call runs with: its ``mlops=`` argument, else the
    enclosing block's, else None (off)."""
    return settings if settings is not None else _CURRENT.get()


def raise_on_error(settings: Optional[MLOps], error: Optional[BaseException],
                   call: str, stacklevel: int = 2) -> None:
    r"""Reports ``error``, a tracking or registration failure of a finished
    ``call``: raises when ``settings.on_error`` is ``'raise'``, else warns
    unless `warn_failure` already warned about it."""
    if error is None or settings is None:
        return
    message = (f'{call} finished, but MLOps tracking or registration '
               f'failed: {full_error(error)}')
    if settings.on_error == 'raise':
        raise RuntimeError(message) from error
    warn_failure(message, error, stacklevel + 1)


def full_error(error: BaseException) -> str:
    r"""``Type: message``, not shortened."""
    return f'{type(error).__name__}: {error}'


def warn_failure(message: str, error: BaseException, stacklevel: int = 2,
                 also: Optional[BaseException] = None) -> None:
    r"""Warns ``message`` (a :class:`UserWarning`) once per ``error``; it
    also reports ``also``, an earlier error that caused ``error``."""
    if getattr(error, '_kumo_warned', False):
        return
    warnings.warn(message, UserWarning, stacklevel=stacklevel + 1)
    for reported in (error, also):
        try:
            setattr(reported, '_kumo_warned', True)
        except AttributeError:
            pass


def on_root(source: Any, path: str) -> Tuple[Any, str]:
    r"""The connector and path a ``load_from_connector(source, path=...)``
    reads: an `MLOps` ``source`` reads ``path`` under its artifact root."""
    if isinstance(source, MLOps):
        return source.root.connector, source.root.location(path.lstrip('/'))
    return source, path


def snapshot_path(snapshot: Any) -> str:
    r"""Where a snapshot, or a snapshot id, is stored under a root."""
    snapshot_id = snapshot if isinstance(snapshot, str) else snapshot.id
    return 'snapshots/' + snapshot_id.replace('@', '/')


_SNAPSHOT_ID = re.compile(r'[0-9a-f]+@[0-9a-f]+')


def snapshot_uri(snapshot: Any,
                 settings: Optional[MLOps] = None) -> Optional[str]:
    r"""A snapshot to serve against, as a location: a URI or stage path as
    given; a `GraphSnapshot` or snapshot id as stored under the artifact
    root of ``settings`` (else of the enclosing block)."""
    if snapshot is None or (isinstance(snapshot, str)
                            and not _SNAPSHOT_ID.fullmatch(snapshot)):
        return snapshot
    settings = resolve(settings)
    if settings is None:
        raise ValueError(
            'A snapshot given by object or id is read from the artifact '
            'root: call this in a kumo.mlops block, or pass the location '
            'snapshot.export(...) returned.')
    return settings.root.uri(snapshot_path(snapshot))
