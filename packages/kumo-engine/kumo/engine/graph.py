# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""`Graph` (metadata) and `GraphSnapshot` (materialized).

`Graph` describes a `kumoapi.graph.GraphDefinition` (edges) over a tuple of
`Table` objects, referenced directly -- no I/O, no id lookup. Materializing a
`Graph` -- reading each `Table`'s connector, running the result through
`kumo-core` to build column stats and the on-disk index -- is the explicit,
separate `snapshot()` step, mirroring the cloud pipeline's DG-ingest stage.

`snapshot()` ports the real `kumo_core` DiskGraph ingest call
(`kumo_core.diskgraph.ingest.execute_diskgraph_ingest`) used today by the
cloud SDK's `_LocalGraphBackend.create_snapshot`
(`kumoai/backend/local/graph.py:251-455`), reading each table via
`Table.connector.source_backend.prepare_ingest(...)`
(`kumoapi.plugin.SourceBackend`).

The materialized `GraphSnapshot.id` is `f'{graph.id}@{data_version[:16]}'`
-- `graph.id` (`Graph.id`, the schema-only fingerprint) plus a fingerprint
of every table's `prepared.version` -- exactly the composite identity
`kumo_core.diskgraph.snapshot.DiskgraphSnapshotHandle`/`LocalSnapshotStore`
already model (as `graph_key`/`graph_data_version`), and exactly what the
cloud SDK's own `create_snapshot` builds (`graph_key=_fingerprint(
graph_config)`, `graph_data_version=_fingerprint(dict(versions))`,
`kumoai/backend/local/graph.py:366-372,443`). This is not optional:
`Graph.id` alone is schema-only, so keying the output directory on it alone
(an earlier version of this method did) means re-running `snapshot()` after
the *source data* changes silently overwrites the index directory an
already-returned, ostensibly-immutable `GraphSnapshot` still points to -- a
caller holding an old `GraphSnapshot` would have its `index_path` mutated
out from under it. Including `data_version` in the directory key makes two
`snapshot()` calls against different data land in different directories, so
an existing `GraphSnapshot` is never invalidated by a later one.

`LocalSnapshotStore` also gives this a safe, free reuse check
(`check_diskgraph_output`, used the same way its own `list_ready()` does): if
a valid completed build already exists for this exact `(graph.id,
data_version)`, ingest is skipped and the existing index is reused --
reusing was *not* safe under the old schema-only id (two calls could
legitimately mean different data), but is now, since the id changes
whenever the data does.

Two `snapshot()` calls sharing a `graph_key` (`self.id`) -- different
processes, or different threads in this one -- would otherwise race on two
things: staging (both writing into the same `persist_dir`) and, once
staged, the check-then-build gap this reuse check creates (both seeing no
existing build for the same `(graph_key, data_version)` and both running
`execute_diskgraph_ingest` into the same directory). Both are closed by one
`filelock.FileLock`, keyed on `graph_key` and held for the whole staging +
build sequence -- `data_version` isn't known until staging finishes, so it
can't be locked on its own any earlier, and this favors correctness over
letting two different data versions of the same graph build in parallel.
`FileLock`, not `SoftFileLock`: it's a real OS-level advisory lock
(`flock`/`LockFileEx`), so the kernel releases it even if a process
crashes mid-build, unlike a lock-file-existence mutex, which would leave a
stale lock behind. Deliberately not ported from the cloud SDK: its
`Storage`/`ArtifactLocation`-backed durable restore/checkpoint layer (a
remote/durable cache, e.g. a Snowflake stage) -- out of scope for now.
"""

import json
import logging
import threading
import warnings
from dataclasses import dataclass, field, replace
from functools import cached_property, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.pquery import NativeGraph, normalize_graph
from kumoapi.graph import Edge

from kumo.engine._manifest import (
    RECONSTRUCTED,
    SnapshotManifest,
    SnapshotParents,
    SourceTableEntry,
    read_manifest,
    stamp,
    write_manifest,
)
from kumo.engine._provenance import source_table
from kumo.engine._scratch import (
    fingerprint,
    get_scratch_dir,
    snapshot_scratch_dir,
)
from kumo.engine._storage import network_storage
from kumo.engine.connector.base import Connector

if TYPE_CHECKING:
    from kumo.engine.trainer.output_config import SnapshotOutputConfig
from kumo.engine.table import Table

SNAPSHOT_MANIFEST_FILE = PROVENANCE_FILE

_warned_scratch: set[Path] = set()
_warned_backfill: set[Path] = set()
logger = logging.getLogger(__name__)


def _warn_if_network_scratch(root: Path) -> None:
    if root in _warned_scratch:
        return
    storage = network_storage(root)
    if storage is not None:
        _warned_scratch.add(root)
        warnings.warn(
            f"The kumo.engine scratch directory {root} is on {storage}. "
            "Graph sampling and training-table generation read it randomly "
            "and run much slower there; local NVMe is recommended.",
            UserWarning,
            stacklevel=3,
        )


_LOW_FK_MATCH_RATE = 0.5


def _warn_low_fk_match_rates(stats: Dict[str, Any],
                             tables: Dict[str, Any]) -> None:
    edges = stats.get('edge_counts', {})
    failures = stats.get('key_failures_by_context', {})
    for name, table in sorted(tables.items()):
        for fkey in table.get('fkeys', []):
            dst = fkey['dst_table']
            # DiskGraph relation name: src_fkey_to_dst_pkey
            relation = f"{name}_{fkey['name']}_to_{dst}_{fkey['dst_key']}"
            missed = failures.get(f'dst_lookup:{relation}', {})
            matched = edges.get(relation, 0)
            total = (matched + missed.get('lookup_failures', 0) +
                     missed.get('membership_failures', 0))
            if total and matched < _LOW_FK_MATCH_RATE * total:
                warnings.warn(
                    f"Foreign key {name}.{fkey['name']} matched {matched} of "
                    f"{total} non-null rows ({matched / total:.2%}) to "
                    f"{dst}.{fkey['dst_key']}; unmatched rows get no edge. "
                    "Check that both columns hold the same IDs.",
                    UserWarning,
                    stacklevel=3,
                )


def _ingest_stats(path: Any) -> Dict[str, Any]:
    try:
        stats = json.loads(Path(path).read_text())['stats']
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return stats if isinstance(stats, dict) else {}


def _warn_ingest_data_loss(result: Any, graph_config: Dict[str, Any]) -> None:
    from kumo_core.diskgraph.validation import parse_failure_warning

    for table in graph_config.get('tables', []):
        name = table['table_name']
        try:
            path = Path(
                result.kumo_stats_dir) / 'col_stats_min' / f'{name}.json'
            column_stats = json.loads(path.read_text())
        except (AttributeError, OSError, TypeError, ValueError):
            continue
        if not isinstance(column_stats, dict):
            continue
        for key, stat in sorted(column_stats.items()):
            if not isinstance(stat, dict):
                continue
            count = stat.get('value')
            if (key.endswith('__NUM_PARSE_FAILURES__False')
                    and type(count) is int and count > 0):
                column = key.rsplit('__', 3)[0]
                warnings.warn(parse_failure_warning(name, column, count),
                              UserWarning, stacklevel=3)

    stats = _ingest_stats(getattr(result, 'ingest_stats_path', None))
    if not stats:
        return
    tables = {table['table_name']: table for table in graph_config['tables']}
    duplicates: Dict[str, int] = {}
    for context, count in stats.get('pk_duplicate_counts', {}).items():
        table_name = context.split(':', 1)[0]
        duplicates[table_name] = duplicates.get(table_name, 0) + count
    for table_name, count in sorted(duplicates.items()):
        if count and table_name in tables:
            pkey = tables[table_name]['pkey']['name']
            warnings.warn(
                f"Snapshot contains {count} duplicate primary-key row(s) "
                f"for {table_name}.{pkey}; canonical-first key lookup "
                "semantics apply.",
                UserWarning,
                stacklevel=3,
            )
    _warn_low_fk_match_rates(stats, tables)
    failures = stats.get('key_failures_by_context', {})
    source_rows = stats.get('source_rows_by_table', {})
    for table_name, table in sorted(tables.items()):
        pkey = table.get('pkey', {}).get('name')
        counts = failures.get(f'pkey:{table_name}.{pkey}', {})
        count = counts.get('parse_failures', 0)
        if count:
            total = source_rows.get(table_name, 0)
            fraction = f" ({count / total:.1%} of source rows)" if total else ''
            warnings.warn(
                f"Snapshot dropped {count} row(s) with a null or invalid "
                f"primary key in {table_name}.{pkey}{fraction}.",
                UserWarning,
                stacklevel=3,
            )


_LOCATION_KEYS = frozenset({'table_name', 'source_table_name', 'connector_id'})
_SCHEMA_PARTS = {
    'cols': 'columns',
    'fkeys': 'edges',
    'pkey': 'primary key',
    'time_col': 'time column',
    'end_time_col': 'end time column',
}


def _schema_tables(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        table['table_name']: {
            key: (sorted(json.dumps(item, sort_keys=True, default=str)
                         for item in value) if isinstance(value, list) else
                  value)
            for key, value in table.items() if key not in _LOCATION_KEYS
        }
        for table in config.get('tables', [])
    }


def schema_changes(before: Mapping[str, Any],
                   after: Mapping[str, Any]) -> List[str]:
    r"""How two graph configs differ in tables, columns, types, keys and
    edges; empty when only connectors or table locations differ."""
    old, new = _schema_tables(before), _schema_tables(after)
    changes = [
        f"table {name} {'removed' if name in old else 'added'}"
        for name in sorted(old.keys() ^ new.keys())
    ]
    for name in sorted(old.keys() & new.keys()):
        parts = sorted(
            _SCHEMA_PARTS.get(key, key)
            for key in old[name].keys() | new[name].keys()
            if old[name].get(key) != new[name].get(key))
        if parts:
            changes.append(f"{name} {', '.join(parts)}")
    return changes


def report_other_graph(message: str, before: Optional[Mapping[str, Any]],
                       after: Mapping[str, Any], stacklevel: int) -> None:
    if before is None:
        warnings.warn(
            f"{message} The model does not record its training graph, so "
            "what differs is unknown; a different connector or table "
            "location alone also changes the graph id.", UserWarning,
            stacklevel=stacklevel + 1)
        return
    changes = schema_changes(before, after)
    if changes:
        warnings.warn(f"{message} Schema changes: {'; '.join(changes)}.",
                      UserWarning, stacklevel=stacklevel + 1)
    else:
        logger.info('%s The schema is the same; only table connectors or '
                    'locations differ.', message)


def _store() -> Any:
    from kumo_core.diskgraph.snapshot import (
        DiskgraphLocalConfig,
        LocalSnapshotStore,
    )

    return LocalSnapshotStore(
        DiskgraphLocalConfig(data_root=str(get_scratch_dir() / 'snapshots')))


def _lock(graph_key: str) -> Any:
    from filelock import FileLock

    persist_dir = snapshot_scratch_dir(graph_key) / 'source_tables'
    persist_dir.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(f'{persist_dir}.lock')


def _loaded(store: Any, handle: Any) -> Optional[Any]:
    from diskgraph import DiskgraphCorruptionError, read_success_marker
    from kumo_core.diskgraph.snapshot import rebased

    path = store.path(handle)
    if read_success_marker(path) is None:
        return None
    try:
        store.verify(handle)
        return rebased(store, handle)
    except (DiskgraphCorruptionError, ValueError) as error:
        warnings.warn(f"Reloading the damaged graph index at {path}: {error}",
                      UserWarning, stacklevel=3)
        return None


def _snapshot_id(graph_key: str, data_version: str) -> str:
    return f'{graph_key}@{data_version[:16]}'


_COMPARABLE = ('exact', 'timestamp', 'listing')


def _source_change(table: Table, recorded: Optional[SourceTableEntry],
                   record_sources: bool) -> Dict[str, Any]:
    from kumoapi.source_table import CustomSQLSourceTable

    before = None if recorded is None else recorded.version
    current, same_source = None, False
    comparable = (recorded is not None
                  and recorded.version_kind in _COMPARABLE)
    try:
        source = table.definition.source_table
        if not isinstance(source, CustomSQLSourceTable):
            connector = table.connector
            stamp = connector.source_backend.get_data_version(
                connector, table.table_name)
            now = source_table(connector, table.table_name, None, stamp,
                               record_sources=record_sources)
            current = now.version
            comparable = comparable and now.version_kind in _COMPARABLE
            same_source = recorded is not None and (
                recorded.connector, recorded.location,
                recorded.location_fingerprint) == (now.connector, now.location,
                                                   now.location_fingerprint)
    except Exception:
        current = None
    if not (same_source and comparable and before is not None
            and current is not None):
        status = 'unknown'
    else:
        status = 'unchanged' if before == current else 'changed'
    return {'status': status, 'recorded': before, 'current': current}


@dataclass(frozen=True)
class GraphSnapshot:
    r"""The materialized artifact produced by `Graph.snapshot()`: an on-disk
    `kumo-core` index plus column stats, analogous to `IngestResult` in the
    cloud pipeline. Immutable -- re-running `snapshot()` produces a new
    `GraphSnapshot`, never mutates this one.

    Deliberately does not hold a back-reference to the `Graph` it came from
    (that would make this class and `Graph` mutually referential); a caller
    that needs both already holds its own `Graph` object directly.

    Args:
        id: `f'{graph.id}@{data_version[:16]}'` -- `graph.id` is `Graph.id`
            (the schema-only fingerprint the `Graph` this snapshot came from
            was built with); `data_version` fingerprints every table's
            `prepared.version` at ingest time. Distinct from `Graph.id` on
            purpose: two snapshots of the same (unchanged) `Graph` taken at
            different times, over different underlying data, get different
            `id`s and different `index_path`s, so neither ever invalidates
            the other.
        graph_config: the portable graph dict `kumo_core` training/prediction
            consume (what the cloud backend calls `to_portable(definition,
            connector_ids)` and passes as `graph=`/`graph_config=` into
            `TrainingWorkflowRequest.from_spec`/`materialize(...)`). Computed
            once here, at snapshot time, rather than being rebuilt on every
            `Trainer.fit`/`predict` call.
        record_sources: the `record_sources` choice this snapshot was taken
            with; `None` when unknown (built elsewhere or loaded).
    """

    id: str
    index_path: Path
    graph_config: Dict[str, Any]
    record_sources: bool | None = field(default=None, compare=False)

    @property
    def _graph_key(self) -> str:
        """The `Graph.id` this snapshot was built from."""
        return self.id.partition('@')[0]

    @property
    def _data_version(self) -> str:
        """The full fingerprint of the source versions the index was built
        from; `id` carries its first 16 characters."""
        return Path(self.index_path).name

    @cached_property
    def _native_graph(self) -> NativeGraph:
        """`graph_config` in `kumo_core`'s normalized form, as
        `Graph.native_graph`."""
        return normalize_graph(self.graph_config)

    def export(self, output_config: 'SnapshotOutputConfig') -> str:
        r"""Copy the index to `<path>/<graph id>/<data version[:16]>` and
        return that location. An earlier export there stays loadable until
        the new one commits, then is replaced. The connector must hold files
        at the path: an S3, GCS or file connector, or a volume or stage
        whose sink offers a filesystem.
        """
        from kumo.engine._snapshot_io import export

        return export(self, output_config.connector, output_config.path)

    @classmethod
    def load_from_connector(cls, connector: Connector, *, path: str,
                            graph: Optional['Graph'] = None) -> 'GraphSnapshot':
        r"""Copy the snapshot exported at `path` into the local store.

        Args:
            path: the location `export` returned.
            graph: if given, warn when its source tables changed since the
                snapshot was built (see `source_changes`).
        """
        from kumo_core.diskgraph.snapshot import DiskgraphSnapshotHandle

        from kumo.engine._snapshot_io import holds_export, load_into, read

        root, manifest = read(connector, path)
        graph_key = manifest['graph_key']
        if graph is not None and graph.id != graph_key:
            report_other_graph(
                f"Snapshot at {path} was taken of another graph.",
                graph._graph_config, manifest['graph_config'], stacklevel=2)
        store = _store()
        handle = DiskgraphSnapshotHandle(
            graph_key=graph_key, graph_data_version=manifest['data_version'])
        with _lock(graph_key):
            result = _loaded(store, handle)
            if result is not None and not holds_export(
                    Path(result.index_path), manifest):
                result = None
            if result is None:
                try:
                    load_into(connector, path, root, manifest,
                              Path(store.path(handle)))
                except FileNotFoundError:
                    # A newer export of this snapshot committed meanwhile.
                    newer_root, newer = read(connector, path)
                    if (newer_root == root
                            or newer['snapshot_id'] != manifest['snapshot_id']):
                        raise
                    root, manifest = newer_root, newer
                    load_into(connector, path, root, manifest,
                              Path(store.path(handle)))
                result = _loaded(store, handle)
            if result is None:
                raise RuntimeError(f'The snapshot loaded from {path} did not '
                                   'pass its integrity check.')
        snapshot = cls(id=manifest['snapshot_id'],
                       index_path=Path(result.index_path),
                       graph_config=manifest['graph_config'])
        if graph is not None:
            changed = [
                alias
                for alias, change in snapshot.source_changes(graph).items()
                if change['status'] == 'changed'
            ]
            if changed:
                warnings.warn(
                    f"Snapshot at {path} predates source table(s) "
                    f"{', '.join(changed)}; see `snapshot.source_changes`.",
                    UserWarning, stacklevel=2)
        return snapshot

    def source_changes(self, graph: 'Graph') -> Dict[str, Dict[str, Any]]:
        r"""Each of `graph`'s tables' source status since this snapshot:
        `unchanged`, `changed` or `unknown`. Never raises."""
        try:
            manifest = self.manifest
            recorded = ({} if manifest is None or manifest.reconstructed else
                        dict(manifest.parents.source_tables))
            record_sources = (manifest is None
                              or manifest.record_sources is not False)
        except Exception:
            recorded, record_sources = {}, True
        from kumo_core.sources.parallel import run_in_parallel

        tables = graph._named_tables
        changes = run_in_parallel(
            [
                partial(_source_change, table, recorded.get(alias),
                        record_sources)
                for alias, table in tables.items()
            ], workers=max(1, min(len(tables), 16)), stop=threading.Event())
        return dict(zip(tables, changes))

    @property
    def manifest(self) -> SnapshotManifest | None:
        """The snapshot manifest stored next to the index, if any."""
        return read_manifest(Path(self.index_path) / SNAPSHOT_MANIFEST_FILE,
                             SnapshotManifest)


def _write_snapshot_manifest(result: Any, snapshot_id: str,
                             source_tables: Dict[str, SourceTableEntry],
                             record_sources: bool, *, fresh: bool) -> None:
    path = Path(result.index_path) / SNAPSHOT_MANIFEST_FILE
    try:
        if not fresh and path.exists():
            existing = read_manifest(path, SnapshotManifest)
            if existing is None or (existing.record_sources
                                    is not False) == record_sources:
                return
            parents = (SnapshotParents(source_tables=source_tables)
                       if existing.parents is None else replace(
                           existing.parents, source_tables=source_tables))
            manifest = replace(existing, record_sources=record_sources,
                               parents=parents)
        else:
            manifest = SnapshotManifest(
                id=snapshot_id,
                parents=SnapshotParents(source_tables=source_tables),
                record_sources=record_sources,
                **(stamp() if fresh else RECONSTRUCTED))
        write_manifest(path, manifest)
    except Exception as error:
        if fresh or path not in _warned_backfill:
            if not fresh:
                _warned_backfill.add(path)
            warnings.warn(
                f"Could not write the snapshot manifest at {path}: {error}",
                UserWarning,
                stacklevel=4,
            )


@dataclass(frozen=True)
class Graph:
    r"""An immutable, metadata-only description of a graph: a set of `Table`
    objects plus the edges between them. A table mapping names graph aliases;
    a sequence uses each source table name as its graph alias.
    """

    tables: Mapping[str, Table] | Sequence[Table]
    edges: Sequence[Edge] = field(default_factory=tuple)

    @cached_property
    def _named_tables(self) -> Dict[str, Table]:
        if isinstance(self.tables, Mapping):
            return dict(self.tables)
        result = {}
        for table in self.tables:
            if table.table_name in result:
                raise ValueError(
                    f"Duplicate graph table name {table.table_name!r}; "
                    "use an alias mapping to distinguish the sources.")
            result[table.table_name] = table
        return result

    @cached_property
    def id(self) -> str:
        r"""A content-fingerprint id derived from this graph's definition and
        table sources -- the same approach the cloud pipeline uses. Two
        structurally identical `Graph` objects converge on the same id,
        regardless of `self.tables`/`self.edges` construction order (both
        are sorted before hashing). Computed once, on first access, and
        cached (`functools.cached_property`, same mechanism as
        `native_graph`/`_graph_config` below) -- it's a pure function of
        immutable state, so there is nothing to invalidate.

        Schema-only -- it says nothing about the underlying data, which can
        change between two `snapshot()` calls even though this id doesn't.
        `snapshot()` uses this as the `graph_key` half of a `GraphSnapshot`'s
        own (different, data-version-aware) `id`; see that method.

        Each table contributes `(connector.id, table_name, definition)`.
        `connector.id` is itself a fingerprint of `(source_type, config)`
        (`kumo.engine.connector.base.Connector.id`) -- deliberately not the
        connector's `identity`: two connectors that read the same target
        under different credentials still read the same data, so the
        credentials used to get there don't belong in a content fingerprint.
        `definition` is serialized via `kumoapi.json_serde.to_json_dict`,
        the same canonical JSON projection the wire protocol uses, so it's
        stable across process restarts (unlike `repr`, which isn't
        guaranteed to be).
        """
        import json

        from kumoapi.json_serde import to_json_dict

        payload = {
            'tables':
            sorted(
                ({
                    'connector_id':
                    table.connector.id,
                    'table_name':
                    alias,
                    'definition': (to_json_dict(table.definition)
                                   if table.definition is not None else None),
                } for alias, table in self._named_tables.items()),
                key=lambda entry: str(entry['table_name']),
            ),
            'edges':
            sorted(
                ({
                    'src_table': edge.src_table,
                    'fkey': edge.fkey,
                    'dst_table': edge.dst_table,
                } for edge in self.edges),
                key=lambda entry:
                (entry['src_table'], entry['fkey'], entry['dst_table']),
            ),
        }
        canonical = json.dumps(payload, sort_keys=True)
        return fingerprint(canonical.encode())

    @cached_property
    def _graph_config(self) -> Dict[str, Any]:
        r"""The portable graph dict `kumo_core` consumes --
        `kumo.engine._portable.portable_graph_config` over this graph's table
        *schemas* (no ingested index needed). Computed once, on first
        access, and cached (`functools.cached_property` writes directly
        into `self.__dict__`, bypassing `__setattr__`, which is why this
        needs no `object.__setattr__` dance despite `Graph` being frozen --
        same mechanism as `PredictiveQuery._parsed`). A conversion failure
        is *not* cached, so a later access re-converts.

        The one shared computation behind `native_graph` (below),
        `validate()`, and `_snapshot()` -- each used to build its own,
        separately, so `graph.validate()` followed by `graph.snapshot()`
        would otherwise convert the same graph twice.
        """
        from kumo.engine._infer import refuse_unsupported_columns
        from kumo.engine._portable import portable_graph_config

        for table in self._named_tables.values():
            if table.definition is not None:
                refuse_unsupported_columns(table.connector, table.table_name,
                                           table.definition.cols)
        return portable_graph_config(self._named_tables, self.edges,
                                     name=self.id)

    @cached_property
    def native_graph(self) -> NativeGraph:
        r"""This graph, converted to `kumo_core`'s own normalized
        representation -- `self._graph_config` followed by `kumo_core.
        pquery.normalize_graph`. Cached the same way `_graph_config` is.

        `validate()` and `PredictiveQuery._parsed` (`self.graph.
        native_graph`, directly) both consume this rather than each
        building their own.
        """
        return normalize_graph(self._graph_config)

    def validate(self) -> None:
        r"""Validate table schemas, key references and key dtype compatibility.

        Keys must have matching types or allow lossless integer widening,
        following the shared hosted dtype compatibility rules. Invalid
        graphs raise before snapshot ingestion starts, as do columns whose
        source type has no Kumo dtype and that no declared dtype resolved.
        """
        self.native_graph

    def snapshot(self, *, record_sources: bool = True,
                 mlops: Optional[Any] = None) -> GraphSnapshot:
        r"""Materialize this graph: read each table's connector, run the
        result through `kumo-core` to build column stats and the on-disk
        index. The index is stored under a directory keyed by both `self.id`
        and a fingerprint of every table's data version (`data_version`) --
        see the module docstring for why `self.id` alone isn't a safe key,
        and `GraphSnapshot`'s docstring for the resulting `id` shape. If a
        valid build already exists for that exact key, ingest is skipped and
        it's reused as is.

        Args:
            mlops: a `kumo.MLOps`; with MLOps on (this or a `kumo.mlops`
                block), the snapshot is also exported under the artifact
                root's `snapshots/`, unless it is there already.
            record_sources: if False, the snapshot manifest records each
                source as connector and table name only (no bucket, path or
                account) and Custom SQL as a hash only (no SQL text).
        """
        self.validate()
        for table in self._named_tables.values():
            table.connector.validate_scratch_dir(get_scratch_dir())
        _warn_if_network_scratch(get_scratch_dir())

        graph_key = self.id
        persist_dir = snapshot_scratch_dir(graph_key) / 'source_tables'

        # Guards two things, both shared across every `snapshot()` call for
        # this `graph_key` -- different processes, or different threads in
        # this one: staging (into `persist_dir`, in `_snapshot_no_lock`) and
        # the check-then-build gap around the final index (`output_dir`,
        # only known once staging has produced `table_data_versions`, so it
        # can't be locked on its own any earlier). One lock, held for the
        # whole staging + build sequence, keyed on `graph_key` alone rather
        # than `(graph_key, data_version)`: `data_version` isn't known until
        # staging finishes, and correctness (no concurrent writers into the
        # same directory) matters more here than letting two different data
        # versions of the same graph build in parallel. `FileLock`, not
        # `SoftFileLock`: a real OS-level advisory lock (`flock`/
        # `LockFileEx`), released by the kernel even if this process crashes
        # mid-build -- a plain in-memory lock wouldn't coordinate across
        # processes, and a lock-file-existence mutex would leave a stale
        # lock behind after a crash.
        from kumo.engine._mlops import resolve

        settings = resolve(mlops)
        with _lock(graph_key):
            snapshot = self._snapshot_no_lock(graph_key, persist_dir,
                                              record_sources)
        if settings is not None and settings.wants('snapshot'):
            from kumo.engine._snapshot_io import store

            store(snapshot, settings)
        return snapshot

    def _snapshot_no_lock(self, graph_key: str, persist_dir: Path,
                          record_sources: bool = True) -> GraphSnapshot:
        r"""`snapshot()`'s body, run under its `FileLock`. Not meant to be
        called on its own -- `persist_dir`, unlike everything else here,
        isn't a pure function of `self` alone (it's `snapshot()`'s lock key,
        computed once there and threaded through), so calling this directly
        would skip the lock that makes it safe to stage into.
        """
        from kumo_core.diskgraph.ingest import (
            DiskgraphIngestSpec,
            execute_diskgraph_ingest,
        )
        from kumo_core.diskgraph.runner import IngestExecutionConfig
        from kumo_core.diskgraph.snapshot import (
            DiskgraphIngestResult,
            DiskgraphSnapshotHandle,
            check_diskgraph_output,
        )
        from kumo_core.sources.layout import staging_dir
        from kumo_core.sources.transfer import discard_destination
        from kumoapi.source_table import CustomSQLSourceTable

        from kumo.engine.connector.source_table import checked_sql

        graph_config = self._graph_config
        store = _store()

        table_data_versions: Dict[str, str] = {}
        source_paths: Dict[str, str] = {}
        source_tables: Dict[str, SourceTableEntry] = {}

        try:
            for alias, table in self._named_tables.items():
                assert table.definition is not None  # portable_graph_config
                # already required this.
                connector = table.connector
                source_backend = connector.source_backend
                dest_dir = staging_dir(persist_dir, alias)
                dest_dir.mkdir(parents=True, exist_ok=True)

                name, sql = table.table_name, None
                source = table.definition.source_table
                if isinstance(source, CustomSQLSourceTable):
                    checked = checked_sql(connector, source.sql)
                    name, sql = checked.table, checked.sql
                prepared = source_backend.prepare_ingest(
                    connector,
                    name,
                    str(dest_dir),
                    dtypes={
                        col.name: col.dtype
                        for col in table.definition.cols
                    },
                    sql=sql,
                )
                if prepared.credentials is not None:
                    raise NotImplementedError(
                        "TODO(local-sdk): scoped per-table ingest credentials "
                        "(PrepareIngestResult.credentials) aren't wired through "
                        "to kumo-core's ingest yet -- matches the current "
                        "limitation in kumoai/backend/local/graph.py's "
                        "_stage_sources, which raises the same way.")
                table_data_versions[alias] = prepared.version
                source_tables[alias] = source_table(
                    connector, name, sql, prepared.version,
                    record_sources=record_sources)

                paths = prepared.source.get('paths') or []
                if paths:
                    parents = {str(Path(path).parent) for path in paths}
                    if len(parents) != 1:
                        raise NotImplementedError(
                            "TODO(local-sdk): a table staged across multiple "
                            "directories isn't handled here (mirrors "
                            "kumoai/backend/local/graph.py's create_snapshot, "
                            "which also refuses this case).")
                    source_paths[alias] = parents.pop()
                else:
                    source_paths[alias] = str(
                        prepared.source.get('path') or dest_dir)

            data_version = fingerprint(
                json.dumps(table_data_versions, sort_keys=True).encode())
            handle = DiskgraphSnapshotHandle(graph_key=graph_key,
                                             graph_data_version=data_version)
            output_dir = Path(store.path(handle))

            existing = check_diskgraph_output(str(output_dir))
            if existing is not None:
                result = existing
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                # The Rust ingest binary only writes the index's
                # `_MANIFEST.json` (read later by, e.g., kumo_core.diskgraph.
                # stores.diskgraph_index_size_bytes) when given a
                # `success_marker_content` to write atomically on completion.
                # Predicting the result before running (rather than using the
                # real one afterward) is what the cloud SDK's own
                # `create_snapshot` does too -- confirmed by hitting a real
                # `FileNotFoundError` on `_MANIFEST.json` from a real
                # `Trainer.fit()` without this.
                predicted = DiskgraphIngestResult.for_index(output_dir)
                result = execute_diskgraph_ingest(
                    graph_config,
                    DiskgraphIngestSpec(
                        data_persist_dir=str(persist_dir),
                        table_data_versions=table_data_versions,
                        output_dir=str(output_dir),
                        source_paths=source_paths,
                        execution=IngestExecutionConfig(
                            success_marker_content=predicted.to_json()),
                    ),
                )

            snapshot_id = _snapshot_id(graph_key, data_version)
            _write_snapshot_manifest(result, snapshot_id, source_tables,
                                     record_sources, fresh=existing is None)

            _warn_ingest_data_loss(result, graph_config)
            return GraphSnapshot(
                id=snapshot_id,
                index_path=Path(result.index_path),
                graph_config=graph_config,
                record_sources=record_sources,
            )
        finally:
            discard_destination(str(persist_dir))
