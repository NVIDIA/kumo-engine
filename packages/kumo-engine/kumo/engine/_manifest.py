# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared JSON manifest format for engine artifacts."""

import json
import types
import warnings
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from kumo_core.io.filesystem import atomic_write_bytes

SCHEMA_VERSION = 1
_OMIT_NONE = 'omit_none'

R = TypeVar('R', bound='Record')
M = TypeVar('M', bound='Manifest')


def omit_none() -> Any:
    """A field written only when it is not None."""
    return field(default=None, metadata={_OMIT_NONE: True})


@dataclass(frozen=True, kw_only=True)
class Record:
    """A JSON object; keys this version does not know are kept in ``extra``."""

    extra: Mapping[str, Any] = field(default_factory=dict, repr=False,
                                     compare=False)

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.extra)
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name != 'extra' and not (value is None
                                             and item.metadata.get(_OMIT_NONE)):
                data[item.name] = _plain(value)
        return data

    @classmethod
    def from_dict(cls: Type[R], data: Mapping[str, Any]) -> R:
        hints = get_type_hints(cls)
        names = {item.name for item in fields(cls)} - {'extra'}
        return cls(
            **{
                name: _typed(hints[name], data[name])
                for name in names if name in data
            }, extra={
                key: value
                for key, value in data.items() if key not in names
            })


def _plain(value: Any) -> Any:
    if isinstance(value, Record):
        return value.to_dict()
    if isinstance(value, AbcMapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _typed(hint: Any, value: Any) -> Any:
    origin, args = get_origin(hint), get_args(hint)
    if value is None:
        return None
    if origin in (Union, types.UnionType):
        return _typed(next(arg for arg in args if arg is not type(None)), value)
    if isinstance(hint, type) and issubclass(hint, Record):
        return hint.from_dict(value)
    if origin in (dict, AbcMapping) and args:
        return {key: _typed(args[1], item) for key, item in value.items()}
    if origin is list and args:
        return [_typed(args[0], item) for item in value]
    return value


@dataclass(frozen=True, kw_only=True)
class Environment(Record):
    kumo_engine: Optional[str]
    build_commit: Optional[str]
    kumo_core: Optional[str]
    diskgraph: Optional[str]
    diskgraph_build_commit: Optional[str]
    torch: Optional[str]
    python: Optional[str]


@dataclass(frozen=True, kw_only=True)
class SchedulerContext(Record):
    platform: str
    job_id: Optional[str]
    job_run_id: Optional[str]


@dataclass(frozen=True, kw_only=True)
class SourceTableEntry(Record):
    connector: Optional[str]
    location: Optional[str]
    version: Optional[str]
    version_kind: str
    dependencies: Optional[Dict[str, str]] = omit_none()
    normalized_sql_fingerprint: Optional[str] = omit_none()
    sql: Optional[str] = omit_none()
    dialect: Optional[str] = omit_none()
    namespace: Optional[List[Any]] = omit_none()


@dataclass(frozen=True, kw_only=True)
class Manifest(Record):
    kind: str
    schema_version: int = SCHEMA_VERSION
    id: Optional[str]
    created_at: Optional[str]
    environment: Optional[Environment]
    scheduler: Optional[SchedulerContext]
    reconstructed: Optional[bool] = omit_none()


def stamp() -> Dict[str, Any]:
    """Envelope fields for a manifest written now."""
    from kumo.engine._provenance import environment, scheduler_context

    return dict(
        created_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        environment=environment(), scheduler=scheduler_context())


#: Envelope fields for a manifest rebuilt after the fact.
RECONSTRUCTED: Dict[str, Any] = dict(created_at=None, environment=None,
                                     scheduler=None, reconstructed=True)


@dataclass(frozen=True, kw_only=True)
class SnapshotParents(Record):
    source_tables: Dict[str, SourceTableEntry]


@dataclass(frozen=True, kw_only=True)
class SnapshotManifest(Manifest):
    kind: str = 'graph_snapshot'
    parents: SnapshotParents
    record_sources: Optional[bool] = omit_none()


@dataclass(frozen=True, kw_only=True)
class TrainingTableParents(Record):
    snapshot: str
    source_table: Optional[SourceTableEntry] = omit_none()


@dataclass(frozen=True, kw_only=True)
class TrainingTableManifest(Manifest):
    kind: str = 'training_table'
    parents: TrainingTableParents
    query: str
    generation_plan: Dict[str, Any]
    task_type: str
    split_seed: Optional[int]


def write_manifest(path: Path, manifest: Manifest) -> None:
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + '\n'
    atomic_write_bytes(path, payload.encode('utf-8'))


def read_manifest(path: Path, kind: Type[M]) -> Optional[M]:
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        warnings.warn(f"Ignoring unreadable manifest {path}: {error}",
                      UserWarning, stacklevel=3)
        return None
    version = (manifest.get('schema_version')
               if isinstance(manifest, dict) else None)
    if type(version) is not int or not 1 <= version <= SCHEMA_VERSION:
        warnings.warn(
            f"Ignoring {path}: manifest schema_version {version!r} is not "
            f"supported by this kumo-engine (up to {SCHEMA_VERSION}).",
            UserWarning,
            stacklevel=3,
        )
        return None
    try:
        return kind.from_dict(manifest)
    except (AttributeError, TypeError, ValueError) as error:
        warnings.warn(f"Ignoring unreadable manifest {path}: {error}",
                      UserWarning, stacklevel=3)
        return None
