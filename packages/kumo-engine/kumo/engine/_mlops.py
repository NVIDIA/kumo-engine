# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The MLOps switch: :class:`MLOps` settings and the :func:`mlops` block."""

import contextlib
import contextvars
from dataclasses import dataclass, field
from typing import Any, FrozenSet, Iterable, Iterator, Optional, Tuple, Union

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
    r"""Where artifacts go, for :func:`mlops` blocks and ``mlops=``
    arguments.

    Args:
        artifact_root: an ``s3://``, ``gs://``, ``file://`` or local path
            (``/Volumes/...`` on a Databricks cluster), or a ``(connector,
            path)`` pair for a store a URI cannot reach, such as a volume
            through a ``DatabricksConnector`` or a Snowflake stage.
        store: the artifact kinds stored under the root when produced
            (``snapshot``, ``training_table``, ``model``);
            ``True`` for all, ``False`` for none.

    In ``load_from_connector`` calls, an ``MLOps`` stands in for the
    connector, with paths relative to its artifact root.
    """

    artifact_root: Union[str, Tuple[Any, str]]
    store: Union[bool, Iterable[str]] = field(default=True, kw_only=True)
    root: _Root = field(init=False, repr=False, compare=False)
    kinds: FrozenSet[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from kumo.engine.connector import (
            connector_for_root,
            is_file_destination,
        )

        if not self.artifact_root:
            raise ValueError('MLOps needs an artifact_root.')
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
        object.__setattr__(self, 'root', root)
        object.__setattr__(self, 'kinds', kinds)

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


_CURRENT: contextvars.ContextVar[Optional[MLOps]] = contextvars.ContextVar(
    'kumo_mlops', default=None)


@contextlib.contextmanager
def mlops(settings: MLOps) -> Iterator[MLOps]:
    r"""Turns MLOps on for the calls in the block (in this thread or asyncio
    task): each snapshot, training table and model is stored
    under ``settings.artifact_root``. An inner block replaces an outer one;
    an ``mlops=`` argument replaces both for its call."""
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
