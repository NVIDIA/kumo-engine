# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared JSON manifest format for engine artifacts."""

import json
import types
import warnings
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from kumo_core.io.filesystem import atomic_write_bytes
from kumo_core.summaries import Dataset, Summary, Tag

SCHEMA_VERSION = 1
_OMIT_NONE = 'omit_none'

R = TypeVar('R', bound='Record')
M = TypeVar('M', bound='Manifest')


def omit_none() -> Any:
    """A field written only when it is not None, or read as an explicit
    null (kept in ``extra``)."""
    return field(default=None, metadata={_OMIT_NONE: True})


@dataclass(frozen=True)
class Artifact:
    r"""An artifact a manifest names, stored at ``location``, which tracked
    runs tag as ``tag``. A dataset is data a run reads, or writes when its
    ``context`` is ``output``."""

    kind: str
    id: str
    location: Optional[str]
    is_dataset: bool
    tag: Optional[str] = None
    connector: Optional[str] = None
    version: Optional[str] = None
    version_kind: Optional[str] = None
    context: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class Record:
    """A JSON object; keys this version does not know are kept in ``extra``."""

    extra: Mapping[str, Any] = field(default_factory=dict, repr=False,
                                     compare=False)

    def as_artifact(self, key: str) -> Optional[Artifact]:
        r"""The artifact this record names, if any, as ``key`` in its
        parent."""
        return None

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
        nullable = {
            item.name
            for item in fields(cls) if item.metadata.get(_OMIT_NONE)
        }
        return cls(
            **{
                name: _typed(hints[name], data[name])
                for name in names if name in data
            }, extra={
                key: value
                for key, value in data.items()
                if key not in names or (key in nullable and value is None)
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
    if isinstance(hint, type) and issubclass(hint, Manifest):
        return _parent(hint, value)
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
class SourceTableEntry(Record):
    connector: Optional[str]
    location: Optional[str]
    version: Optional[str]
    version_kind: str
    dependencies: Optional[Dict[str, str]] = omit_none()
    location_fingerprint: Optional[str] = omit_none()
    normalized_sql_fingerprint: Optional[str] = omit_none()
    sql: Optional[str] = omit_none()
    dialect: Optional[str] = omit_none()
    namespace: Optional[List[Any]] = omit_none()

    def as_artifact(self, key: str) -> Artifact:
        return Artifact('source_table', key, self.location, True,
                        connector=self.connector, version=self.version,
                        version_kind=self.version_kind)


#: The artifact kinds, as manifests and the artifact root name them.
KINDS = ('snapshot', 'training_table', 'model')


@dataclass(frozen=True, kw_only=True)
class Manifest(Record):
    """An artifact's record. A stub (a parent with no record of its own)
    sets only ``kind``, ``id`` and ``reconstructed``."""

    kind: str = ''
    schema_version: int = SCHEMA_VERSION
    id: str
    created_at: Optional[str] = None
    environment: Optional[Environment] = None
    reconstructed: Optional[bool] = omit_none()
    location: Optional[str] = omit_none()

    #: Fields left out where this manifest is embedded in a child's.
    EMBED_OMITS = ()
    #: Whether this kind is data a run reads or writes.
    IS_DATASET: ClassVar[bool] = False
    #: The tag of its location in tracked runs.
    URI_TAG: ClassVar[Optional[str]] = None

    def as_artifact(self, key: str) -> Artifact:
        return Artifact(self.kind, self.id, self.location, self.IS_DATASET,
                        self.URI_TAG or f'kumo.{self.kind}_uri',
                        version=self.id, version_kind='exact')


def artifacts(manifest: Manifest) -> List[Artifact]:
    r"""``manifest``'s artifact, then each one it names (parents, their
    source tables, dependencies, outputs), depth first, each once. A named
    manifest that is not a dataset, such as a warm-start model, is listed
    but not opened: its own manifest names what it was made from."""
    found: Dict[Any, Artifact] = {}
    for artifact in _walk(manifest, manifest.kind, manifest):
        found.setdefault((artifact.kind, artifact.id), artifact)
    return list(found.values())


def _walk(value: Any, key: str, root: Manifest) -> Iterator[Artifact]:
    if isinstance(value, Record):
        artifact = value.as_artifact(key)
        if artifact is not None:
            yield artifact
        if isinstance(value, Manifest) and not value.IS_DATASET and \
                value is not root:
            return
        for item in fields(value):
            yield from _walk(getattr(value, item.name), item.name, root)
    elif isinstance(value, AbcMapping):
        for name, item in value.items():
            yield from _walk(item, str(name), root)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item, key, root)


#: Connector kinds of the artifact root's URI schemes.
_SCHEMES = {'s3': 'S3', 'gs': 'GCS', 'file': 'LOCAL_FILE', '': 'LOCAL_FILE'}

Details = Mapping[str, Tuple[Mapping[str, Any], Mapping[str, Any]]]


def artifact_summaries(manifest: Manifest, context: str,
                       settings: Any = None,
                       details: Optional[Details] = None) -> List[Summary]:
    r"""Where ``manifest``'s artifacts are, for its tracked run: each
    artifact's tag (the nearest of each), ``kumo.artifact_root`` under MLOps
    ``settings``, and a `Dataset` per stored dataset, read in ``context``
    or written, with the schema and profile ``details`` gives its id."""
    located = [item for item in artifacts(manifest) if item.location]
    tags: Dict[str, str] = {}
    for item in located:
        if item.tag:
            tags.setdefault(item.tag, str(item.location))
    if settings is not None:
        tags['kumo.artifact_root'] = settings.root.uri('').rstrip('/')
    summaries: List[Summary] = [Tag(name, value)
                                for name, value in tags.items()]
    for item in located:
        if not item.is_dataset:
            continue
        location = str(item.location)
        source = {'connector': item.connector or _connector(location),
                  'location': location, 'version': item.version,
                  'version_kind': item.version_kind}
        schema, profile = (details or {}).get(item.id, ({}, {}))
        summaries.append(Dataset(item.id, source, schema, profile,
                                 item.context or context))
    return summaries


def _connector(location: str) -> Optional[str]:
    from urllib.parse import urlparse

    return None if location.startswith('@') else _SCHEMES.get(
        urlparse(location).scheme)


def stamp() -> Dict[str, Any]:
    """Envelope fields for a manifest written now."""
    from kumo.engine._provenance import environment

    return dict(
        created_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        environment=environment())


#: Envelope fields for a manifest rebuilt after the fact.
RECONSTRUCTED: Dict[str, Any] = dict(created_at=None, environment=None,
                                     reconstructed=True)


def _kinds() -> Dict[str, Type['Manifest']]:
    found: Dict[str, Type[Manifest]] = {}
    pending = list(Manifest.__subclasses__())
    while pending:
        cls = pending.pop()
        found[next(str(item.default) for item in fields(cls)
                   if item.name == 'kind')] = cls
        pending.extend(cls.__subclasses__())
    return found


def stub(kind: str, id: str) -> 'Manifest':
    """The manifest of a ``kind`` artifact known only by its ``id``."""
    return _kinds()[kind](id=id, reconstructed=True)


def embedded(manifest: M) -> M:
    """``manifest`` as a child embeds it: without its statistics and the
    fields that change after it is written."""
    return replace(manifest, **{
        name: None
        for name in manifest.EMBED_OMITS
    })


def _parent(hint: Type[M], value: Any) -> M:
    try:
        return hint.from_dict(value)
    except (AttributeError, TypeError, ValueError, KeyError) as error:
        identifier = value.get('id') if isinstance(value, AbcMapping) else None
        warnings.warn(f"Reading an unreadable parent manifest as a stub: "
                      f"{error}", UserWarning, stacklevel=2)
        return hint(id=str(identifier), reconstructed=True)


@dataclass(frozen=True, kw_only=True)
class SnapshotParents(Record):
    source_tables: Dict[str, SourceTableEntry]


@dataclass(frozen=True, kw_only=True)
class SnapshotManifest(Manifest):
    """What a graph snapshot was built from: each source table's connector,
    location and version, and the engine build that wrote it."""

    kind: str = 'snapshot'
    IS_DATASET: ClassVar[bool] = True
    parents: Optional[SnapshotParents] = None
    record_sources: Optional[bool] = omit_none()


@dataclass(frozen=True, kw_only=True)
class TrainingTableParents(Record):
    snapshot: SnapshotManifest
    source_table: Optional[SourceTableEntry] = omit_none()


@dataclass(frozen=True, kw_only=True)
class TrainingTableManifest(Manifest):
    """How a training table was generated: from which snapshot (embedded)
    and, for caller-supplied rows, which source table."""

    kind: str = 'training_table'
    IS_DATASET: ClassVar[bool] = True
    parents: Optional[TrainingTableParents] = None
    query: Optional[str] = None
    generation_plan: Optional[Dict[str, Any]] = None
    task_type: Optional[str] = None
    split_seed: Optional[int] = None


@dataclass(frozen=True, kw_only=True)
class ModelParents(Record):
    training_table: Optional[TrainingTableManifest] = None
    warm_start: Optional['ModelManifest'] = None
    derived_from: Optional['ModelManifest'] = omit_none()


@dataclass(frozen=True, kw_only=True)
class FitTiming(Record):
    started_at: str
    finished_at: str
    seconds: float


@dataclass(frozen=True, kw_only=True)
class TrackingRef(Record):
    r"""A run in an experiment tracker: its ``platform`` (``mlflow``,
    ``databricks``, ``sagemaker``, ``snowflake``), the platform's own
    ``run_id`` and, where the platform has one, the run's ``url``."""

    platform: str
    run_id: str
    url: Optional[str] = omit_none()


@dataclass(frozen=True, kw_only=True)
class Dependency(Record):
    r"""One thing a model needs to predict: its ``predictor`` entry point,
    a ``package`` requirement, or an ``asset`` (content-addressed by
    ``sha256``, loaded by ``entry_point``, stored at ``location``)."""

    kind: str
    name: str
    version: Optional[str] = omit_none()
    extras: Optional[List[str]] = omit_none()
    asset_kind: Optional[str] = omit_none()
    sha256: Optional[str] = omit_none()
    entry_point: Optional[str] = omit_none()
    packing: Optional[str] = omit_none()
    location: Optional[str] = omit_none()

    def as_artifact(self, key: str) -> Optional[Artifact]:
        kind = _kinds().get(self.kind)
        return None if kind is None else Artifact(
            self.kind, str(self.sha256), self.location, kind.IS_DATASET,
            f'kumo.{self.asset_kind}_uri', version=self.sha256,
            version_kind='exact')


@dataclass(frozen=True, kw_only=True)
class ModelManifest(Manifest):
    """How a model bundle was fitted: from which training table (embedded,
    with its snapshot) and warm-start model, when, and with which fit
    mode. Its ``id`` is the bundle id; ``dependencies`` is what predicting
    needs besides the bundle and a snapshot (none listed: the built-in
    `Trainer` predictor)."""

    kind: str = 'model'
    URI_TAG: ClassVar[Optional[str]] = 'kumo.bundle_uri'
    parents: Optional[ModelParents] = None
    fit: Optional[FitTiming] = None
    fit_mode: Optional[str] = None
    tracking: Optional[TrackingRef] = omit_none()
    dependencies: Optional[List[Dependency]] = omit_none()

    EMBED_OMITS = ('tracking', )


@dataclass(frozen=True, kw_only=True)
class AssetManifest(Manifest):
    """An asset a model depends on, stored once under
    ``<root>/assets/<asset_kind>/<sha256>/``; its ``id`` is the sha256."""

    kind: str = 'asset'
    asset_kind: Optional[str] = None
    name: Optional[str] = None
    version: Optional[str] = None
    format: Optional[str] = None
    source: Optional[str] = None


def manifest_bytes(manifest: Manifest) -> bytes:
    return (json.dumps(manifest.to_dict(), indent=2, sort_keys=True) +
            '\n').encode('utf-8')


def write_manifest(path: Path, manifest: Manifest) -> None:
    atomic_write_bytes(path, manifest_bytes(manifest))


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
