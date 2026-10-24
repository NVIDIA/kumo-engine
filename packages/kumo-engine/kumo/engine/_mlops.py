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
    r"""Where artifacts go and what records them, for :func:`mlops` blocks
    and ``mlops=`` arguments.

    Args:
        artifact_root: an ``s3://``, ``gs://``, ``file://`` or local path
            (``/Volumes/...`` on a Databricks cluster), or a ``(connector,
            path)`` pair for a store a URI cannot reach, such as a volume
            through a ``DatabricksConnector`` or a Snowflake stage.
        tracking: a ``kumo.engine.tracking.Tracking`` that records fits and
            prediction runs.
        registry: where fits register models: a ``ModelOutputConfig`` with a
            registry ``location``, or a ``ModelRegistryLocation`` registered
            through the MLflow tracker.
        export: the artifact kinds stored under the root when produced
            (``snapshot``, ``training_table``, ``model``, ``prediction``);
            ``True`` for all, ``False`` for none.
    """

    artifact_root: Union[str, Tuple[Any, str]]
    tracking: Any = None
    registry: Any = None
    export: Union[bool, Iterable[str]] = True
    root: _Root = field(init=False, repr=False, compare=False)
    exports: FrozenSet[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from kumo.engine.connector import (
            connector_for_root,
            is_file_destination,
        )

        if not self.artifact_root:
            raise ValueError('MLOps needs an artifact_root.')
        if isinstance(self.artifact_root, str):
            root = _Root(connector_for_root(self.artifact_root), '',
                         self.artifact_root)
        else:
            connector, path = self.artifact_root
            if not is_file_destination(connector, path):
                raise ValueError(
                    f'{type(connector).__name__} cannot hold files at '
                    f'{path!r}; artifact_root needs a file location.')
            root = _Root(connector, path, path)
        kinds = (frozenset(KINDS) if self.export is True else
                 frozenset() if self.export is False else frozenset(
                     self.export))
        unknown = sorted(kinds - set(KINDS))
        if unknown:
            raise ValueError(f'Unknown export kinds {unknown}; choose from '
                             f'{list(KINDS)}.')
        object.__setattr__(self, 'root', root)
        object.__setattr__(self, 'exports', kinds)

    def wants(self, kind: str) -> bool:
        r"""Whether artifacts of ``kind`` are stored under the root."""
        return kind in self.exports


#: The settings a resolved call runs with.
Settings = MLOps

_CURRENT: contextvars.ContextVar[Optional[MLOps]] = contextvars.ContextVar(
    'kumo_mlops', default=None)


@contextlib.contextmanager
def mlops(settings: MLOps) -> Iterator[MLOps]:
    r"""Turns MLOps on for the calls in the block (in this thread or asyncio
    task): each snapshot, training table, model and prediction run is stored
    under ``settings.artifact_root``, and fits and prediction runs are
    tracked and registered as ``settings`` says. An inner block replaces an
    outer one; an ``mlops=`` argument replaces both for its call."""
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
