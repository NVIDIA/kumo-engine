# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""What a model needs to predict besides its bundle and a snapshot: its
dependencies, recorded at fit, and the assets among them, resolved at
predict time."""

import hashlib
import importlib
import os
import shutil
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import (
    Any,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from kumo_core.artifacts import PROVENANCE_FILE

from kumo.engine._manifest import (
    AssetManifest,
    Dependency,
    ModelManifest,
    manifest_bytes,
    read_manifest,
    stamp,
)

#: A place assets are stored: a URI or path, or a ``(connector, path)``.
Location = Union[str, Tuple[Any, str]]


_LOCATIONS: ContextVar[Tuple[Location, ...]] = ContextVar(
    'kumo_asset_locations', default=())


@contextmanager
def asset_locations(locations: Sequence[Location]) -> Iterator[None]:
    r"""Asset roots that dependencies resolved in the block look in
    first."""
    token = _LOCATIONS.set((*locations, *_LOCATIONS.get()))
    try:
        yield
    finally:
        _LOCATIONS.reset(token)


class Asset(Protocol):
    r"""A file or directory a model needs at predict time, addressed by its
    content. ``files`` maps names to local files; the class's
    ``load(directory, dependency)`` restores one from a directory holding
    them, and must give the same ``fingerprint()``."""

    asset_kind: str
    name: str
    version: Optional[str]

    def fingerprint(self) -> str:
        ...

    def files(self) -> Mapping[str, Path]:
        ...


class MissingDependencyError(RuntimeError):
    r"""A model's dependency is in none of the places looked in."""


class FileAsset:
    r"""A plain file, or a directory of files."""
    def __init__(self, asset_kind: str, name: str, path: Union[str, Path],
                 version: Optional[str] = None) -> None:
        self.asset_kind, self.name, self.version = asset_kind, name, version
        self.path = Path(path)

    def files(self) -> Mapping[str, Path]:
        if self.path.is_file():
            return {self.path.name: self.path}
        return {
            path.relative_to(self.path).as_posix(): path
            for path in sorted(self.path.rglob('*'))
            if path.is_file() and path.name != PROVENANCE_FILE
        }

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, path in sorted(self.files().items()):
            digest.update(f'{name}\0{path.stat().st_size}\0'.encode())
            with open(path, 'rb') as handle:
                for block in iter(lambda: handle.read(1 << 20), b''):
                    digest.update(block)
        return digest.hexdigest()

    @classmethod
    def load(cls, directory: Path, dependency: Dependency) -> 'FileAsset':
        files = [
            path for path in directory.iterdir()
            if path.is_file() and path.name != PROVENANCE_FILE
        ]
        return cls(str(dependency.asset_kind), dependency.name,
                   files[0] if len(files) == 1 else directory,
                   dependency.version)


def predictor(entry_point: str) -> Dependency:
    return Dependency(kind='predictor', name=entry_point)


def package(name: str, version: str, extras: Sequence[str]) -> Dependency:
    return Dependency(kind='package', name=name, version=version,
                      extras=list(extras))


def entry_point(value: Any) -> str:
    cls = type(value)
    return f'{cls.__module__}:{cls.__qualname__}'


def imported(entry: str) -> Any:
    module, _, name = entry.partition(':')
    value: Any = importlib.import_module(module)
    for part in name.split('.'):
        value = getattr(value, part)
    return value


class Dependencies:
    r"""Records a fit's dependencies (``require``) and resolves a fitted
    model's assets (``resolve``). With MLOps ``settings`` whose export
    includes models, each required asset is stored once under the
    artifact root; ``locations`` are asset roots looked in first."""
    def __init__(self, records: Sequence[Dependency] = (),
                 settings: Any = None,
                 locations: Sequence[Location] = ()) -> None:
        self.records: List[Dependency] = list(records)
        self.errors: List[BaseException] = []
        self._settings = settings
        self._locations = [*locations, *_LOCATIONS.get()]

    @classmethod
    def of(cls, manifest: Optional[ModelManifest], settings: Any = None,
           locations: Sequence[Location] = ()) -> 'Dependencies':
        return cls(getattr(manifest, 'dependencies', None) or (), settings,
                   locations)

    def add(self, dependency: Dependency) -> Dependency:
        self.records.append(dependency)
        return dependency

    def require(self, asset: Asset) -> Dependency:
        sha256 = asset.fingerprint()
        location = None
        settings = self._settings
        if settings is not None and settings.wants('model'):
            try:
                location = _store(asset, sha256, settings)
            except Exception as error:
                self.errors.append(error)
        return self.add(
            Dependency(kind='asset', name=asset.name, version=asset.version,
                       asset_kind=asset.asset_kind, sha256=sha256,
                       entry_point=entry_point(asset), packing='external',
                       location=location))

    def assets(self, asset_kind: str) -> List[Dependency]:
        return [
            record for record in self.records
            if record.kind == 'asset' and record.asset_kind == asset_kind
        ]

    def resolve(self, asset_kind: str, name: str) -> Path:
        r"""A local directory holding the asset, checked against its
        sha256: from the given locations, then where it was stored, then
        the current artifact root."""
        record = next((record for record in self.assets(asset_kind)
                       if record.name == name), None)
        if record is None:
            raise MissingDependencyError(
                f'The model has no {asset_kind} dependency named {name!r}.')
        tried = []
        for connector, path in self.candidates(record):
            tried.append(str(path))
            try:
                directory = _fetch(connector, path, str(record.sha256))
            except (FileNotFoundError, OSError, ValueError):
                continue
            if directory is not None and _verified(directory, record):
                return directory
        raise MissingDependencyError(
            f"The model needs {asset_kind} {name!r} (sha256 "
            f"{record.sha256}), which is in none of "
            f"{tried or 'no configured locations'}. Store it by fitting "
            f"with MLOps on (it is kept under <artifact root>/assets/), "
            f"predict inside a kumo.mlops block whose artifact root holds "
            f"it, or register it where the model looks it up.")

    def candidates(self, record: Dependency) -> List[Tuple[Any, str]]:
        from kumo.engine.connector import connector_for_root

        relative = f'{record.asset_kind}/{record.sha256}'
        found: List[Tuple[Any, str]] = []
        for location in self._locations:
            if isinstance(location, str):
                found.append((connector_for_root(location), relative))
            else:
                connector, path = location
                found.append((connector, f"{path.rstrip('/')}/{relative}"))
        if record.location:
            parent, _, leaf = record.location.rstrip('/').rpartition('/')
            try:
                found.append((connector_for_root(parent or '/'), leaf))
            except ValueError:
                pass
        settings = self._settings
        if settings is not None:
            found.append((settings.root.connector,
                          settings.root.location(f'assets/{relative}')))
        return found


def asset_cache() -> Path:
    base = os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache'
    return Path(base) / 'kumo' / 'assets'


def _store(asset: Asset, sha256: str, settings: Any) -> Optional[str]:
    r"""Writes ``asset`` once to ``<root>/assets/<asset_kind>/<sha256>/``,
    its ``_provenance.json`` last; returns where."""
    from kumo_core.object_store_sink import _prepared

    from kumo.engine.connector import file_store

    relative = f'assets/{asset.asset_kind}/{sha256}'
    try:
        filesystem, root = file_store(settings.root.connector,
                                      settings.root.location(relative))
        root = root.rstrip('/')
        marker = f'{root}/{PROVENANCE_FILE}'
        if filesystem.get_file_info(marker).is_file:
            return settings.root.uri(relative)
        _prepared(filesystem, root)
        for name, path in asset.files().items():
            with open(path, 'rb') as reader, \
                    filesystem.open_output_stream(f'{root}/{name}') as writer:
                shutil.copyfileobj(reader, writer, 1 << 20)
        location = settings.root.uri(relative)
        with filesystem.open_output_stream(marker) as writer:
            writer.write(manifest_bytes(_manifest(asset, sha256, location)))
        return location
    except Exception as error:
        raise RuntimeError(
            f"Could not store {asset.asset_kind} {asset.name!r} under the "
            f"artifact root: {error}") from error


def _manifest(asset: Asset, sha256: str, location: str) -> AssetManifest:
    return AssetManifest(id=sha256, location=location,
                         asset_kind=asset.asset_kind,
                         name=asset.name, version=asset.version,
                         format=entry_point(asset),
                         source=str(getattr(asset, 'path', '') or '') or None,
                         **stamp())


def _fetch(connector: Any, path: str, sha256: str) -> Optional[Path]:
    from pyarrow.fs import FileSelector, FileType, LocalFileSystem

    from kumo.engine.connector import file_store

    store = file_store(connector, path)
    if store is None:
        return None
    filesystem, root = store
    root = root.rstrip('/')
    if filesystem.get_file_info(
            f'{root}/{PROVENANCE_FILE}').type != FileType.File:
        return None
    if isinstance(filesystem, LocalFileSystem):
        return Path(root)
    target = asset_cache() / sha256
    if (target / PROVENANCE_FILE).is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f'.{sha256}-', dir=target.parent))
    try:
        for info in filesystem.get_file_info(FileSelector(root,
                                                          recursive=True)):
            if info.type != FileType.File:
                continue
            local = partial / info.path[len(root) + 1:]
            local.parent.mkdir(parents=True, exist_ok=True)
            with filesystem.open_input_stream(info.path) as reader, \
                    open(local, 'wb') as writer:
                shutil.copyfileobj(reader, writer, 1 << 20)
        try:
            partial.rename(target)
        except OSError:
            if not (target / PROVENANCE_FILE).is_file():
                raise
    finally:
        shutil.rmtree(partial, ignore_errors=True)
    return target


def _verified(directory: Path, record: Dependency) -> bool:
    stored = read_manifest(directory / PROVENANCE_FILE, AssetManifest)
    if stored is None or stored.id != record.sha256:
        return False
    loader = imported(record.entry_point or entry_point(FileAsset))
    return bool(loader.load(directory, record).fingerprint() == record.sha256)


def loaded(directory: Path, record: Dependency) -> Any:
    return imported(record.entry_point or '').load(directory, record)
