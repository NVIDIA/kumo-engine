# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The MLOps switch: :class:`MLOps` settings and the :func:`mlops` block."""

import contextlib
import contextvars
import os
import posixpath
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

from kumo.engine._manifest import KINDS


@dataclass(frozen=True)
class _Root:
    connector: Any
    path: str
    _uri: str

    def location(self, relative: str) -> str:
        base = self.path.rstrip('/')
        return f'{base}/{relative}' if base else relative

    def uri(self, relative: str) -> str:
        return f"{self._uri.rstrip('/')}/{relative}"


@dataclass(frozen=True)
class MLOps:
    r"""MLOps settings.

    Args:
        artifact_root: a URI, local path or ``(connector, path)`` pair.
        store: artifact kinds to store; ``True`` for all, ``False`` none.
        on_error: ``'warn'`` or ``'raise'`` when MLOps fails.
    """

    artifact_root: Union[str, Tuple[Any, str]]
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
        object.__setattr__(self, 'root', root)
        object.__setattr__(self, 'kinds', kinds)

    def wants(self, kind: str) -> bool:
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
            f'({error}); give it a connector that contains it.') from error


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
        except FileNotFoundError:
            filesystem.create_dir(posixpath.dirname(probe), recursive=True)
            filesystem.open_output_stream(probe).close()
        delete = getattr(root.connector.sink_backend, 'delete_file', None)
        if delete is None:
            filesystem.delete_file(probe)
        else:
            delete(root.connector, probe)
    except Exception as error:
        raise PermissionError(
            f'kumo.MLOps cannot write under artifact_root {root.uri("")!r} '
            f'({type(error).__name__}: {error}); grant write and delete '
            f'rights or pick another root.') from error


_CURRENT: contextvars.ContextVar[Optional[MLOps]] = contextvars.ContextVar(
    'kumo_mlops', default=None)


@contextlib.contextmanager
def mlops(settings: MLOps) -> Iterator[MLOps]:
    r"""Turns MLOps on, as ``settings`` says, for the calls in the block."""
    if not isinstance(settings, MLOps):
        raise TypeError('kumo.mlops() takes a kumo.MLOps')
    token = _CURRENT.set(settings)
    try:
        yield settings
    finally:
        _CURRENT.reset(token)


def resolve(settings: Optional[MLOps] = None) -> Optional[MLOps]:
    r"""The ``mlops=`` argument, else the enclosing block's."""
    return settings if settings is not None else _CURRENT.get()


def raise_on_error(settings: Optional[MLOps], error: Optional[BaseException],
                   call: str, stacklevel: int = 2) -> None:
    r"""Raises or warns about ``error``, as ``settings.on_error`` says."""
    if error is None or settings is None:
        return
    message = f'{call} finished, but MLOps failed: {full_error(error)}'
    if settings.on_error == 'raise':
        raise RuntimeError(message) from error
    warn_failure(message, error, stacklevel + 1)


def full_error(error: BaseException) -> str:
    r"""``Type: message``, not shortened."""
    return f'{type(error).__name__}: {error}'


def warn_failure(message: str, error: BaseException, stacklevel: int = 2,
                 also: Optional[BaseException] = None) -> None:
    r"""Warns ``message`` once per ``error`` (and ``also``)."""
    if getattr(error, '_kumo_warned', False):
        return
    warnings.warn(message, UserWarning, stacklevel=stacklevel + 1)
    for reported in (error, also):
        try:
            setattr(reported, '_kumo_warned', True)
        except AttributeError:
            pass


def on_root(source: Any, path: str) -> Tuple[Any, str]:
    r"""``path`` under an `MLOps` root, else ``source`` as given."""
    if isinstance(source, MLOps):
        return source.root.connector, source.root.location(path.lstrip('/'))
    return source, path


def snapshot_path(snapshot: Any) -> str:
    snapshot_id = snapshot if isinstance(snapshot, str) else snapshot.id
    return 'snapshots/' + snapshot_id.replace('@', '/')
