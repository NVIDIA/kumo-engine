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
import time
import warnings
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from kumo_core.pquery import NativeGraph, normalize_graph
from kumoapi.graph import Edge

from kumo.engine._scratch import (
    fingerprint,
    get_scratch_dir,
    snapshot_scratch_dir,
)
from kumo.engine._storage import network_storage
from kumo.engine._visualization import SVG
from kumo.engine.table import Table

_warned_scratch: set[Path] = set()


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

    try:
        report = json.loads(Path(result.ingest_stats_path).read_text())
        stats = report['stats']
    except (AttributeError, OSError, ValueError, KeyError, TypeError):
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


@dataclass(frozen=True)
class GraphSnapshot:
    r"""The materialized artifact produced by `Graph.snapshot()`: an on-disk
    `kumo-core` index plus column stats, analogous to `IngestResult` in the
    cloud pipeline. Immutable -- re-running `snapshot()` produces a new
    `GraphSnapshot`, never mutates this one.

    Retains its source graph so `update()` can preserve table declarations.

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
    """

    id: str
    index_path: Path
    graph_config: Dict[str, Any]
    _graph: 'Graph | None' = field(default=None, repr=False, compare=False)

    def update(self, connector=None, *, mode='auto', compact=False,
               reuse_encoders=True) -> 'GraphSnapshot':
        """Return an immutable snapshot using proven appends where possible.

        `mode='append'` rejects historical edits; `mode='full'` rebuilds.
        Omitting `connector` keeps each table's original connector.
        """
        from kumo.engine.incremental import update_snapshot

        return update_snapshot(self, connector, mode=mode, compact=compact,
                               reuse_encoders=reuse_encoders)

    def compact(self) -> 'GraphSnapshot':
        """Return a compacted copy; this snapshot remains immutable."""
        from kumo.engine.incremental import compact_snapshot

        return compact_snapshot(self)

    def validate(self):
        """Scan merged nodes/edges and verify referential integrity."""
        from diskgraph import validate_incremental_snapshot

        return json.loads(validate_incremental_snapshot(str(self.index_path)))

    @property
    def update_history(self):
        """Return lineage records from the original build to this snapshot."""
        records, root = [], self.index_path
        while root is not None:
            path = root / '_incremental/update.json'
            if not path.is_file():
                break
            record = json.loads(path.read_text())
            records.append(record)
            root = Path(record['parent']) if record.get('parent') else None
        return list(reversed(records))

    @property
    def _latest_update(self):
        return json.loads((self.index_path / '_incremental/update.json').read_text())

    @property
    def tables_rebuilt(self) -> list[str]:
        """Tables re-encoded by the latest update."""
        record = self._latest_update
        return [name for name, table in record['tables'].items()
                if table['status'] == 'rebuilt']

    @property
    def rows_appended(self) -> int:
        """Rows appended by the latest update."""
        return sum(table.get('rows_added', 0)
                   for table in self._latest_update['tables'].values()
                   if table['status'] == 'appended')

    @property
    def duration_seconds(self) -> float:
        """Wall time recorded for the latest snapshot operation."""
        return self._latest_update.get('duration_seconds', 0.0)

    @property
    def fragmentation(self):
        """Per-relation delta count divided by the compacted base count."""
        from kumo.engine.incremental import fragmentation

        return fragmentation(self)


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

    def visualize(self, path: str | Path | None = None, *,
                  show_cols: bool = False, show_samples: bool = False,
                  layout: str = 'auto') -> SVG:
        """Display pannable table relationships in notebooks and return SVG.

        Args:
            path: Optional local file, overwritten if it exists. A .pdf suffix
                writes a vector graph and full schema/connection index.
                A .md or .markdown suffix writes a Mermaid relationship
                diagram and complete Markdown schema tables;
                other paths receive UTF-8 SVG. Paths refer to the filesystem
                where Kumo Engine runs.
            show_cols: Add columns up to 10 rows per table (5 above 10 tables).
                Primary keys, foreign keys, and declared time (TS) and end-time
                (TE) columns are always included, even above this limit.
                False shows only these columns.
            show_samples: Show up to six non-null examples per column in notebook
                tooltips, in two columns and up to three rows, selected from
                up to 30 preview rows per table. False (default) reads no
                source data. Previews may
                run warehouse queries and incur costs; returned row limits
                do not bound bytes scanned. Unavailable previews are labeled.
                Samples become part of saved notebook output, never file exports.
            layout: Initial arrangement: auto, standard, or compact.
                Auto chooses from graph structure, independently of viewport
                size. Standard preserves the layered relationship view;
                compact balances crowded groups.
                Notebook views offer a selector to compare arrangements.

        Uses declared schemas. More than 30 tables, self-relations,
        and missing schema/key definitions raise ValueError.
        Display automatically uses environment-provided widget packages in
        detected Snowflake Workspace notebooks and HTML in other hosts.
        The optional as_widget() helper explicitly selects the widget view.
        Full screen is available when the notebook host permits it.
        Dragging, full screen, Fit, and zoom preserve the selected arrangement.
        SVG exports use the initial layout and contain no interactive controls.
        The export control shows Python commands to save PDF and Markdown
        files in the notebook filesystem. PDF exports include all declared
        columns in the index, even when the diagram omits them.
        Markdown diagrams use the Markdown viewer's Mermaid layout rather than
        the notebook arrangement; schema tables remain readable without Mermaid.
        """
        from kumo.engine._visualization import render

        svg = render(self._named_tables, self.edges, show_cols=show_cols,
                     show_samples=show_samples, layout=layout)
        if path is not None:
            destination = Path(path)
            if destination.suffix.lower() == '.pdf':
                destination.write_bytes(svg._pdf(svg._initial_layout))
            elif destination.suffix.lower() in ('.md', '.markdown'):
                destination.write_text(svg._markdown(), encoding='utf-8')
            else:
                destination.write_text(svg, encoding='utf-8')
        return svg

    def snapshot(self, *, enable_updates: bool = False, previous_snapshot: GraphSnapshot | None = None,
                 reuse_encoders: bool = True,
                 force_tables: Sequence[str] = (),
                 force_full_rebuild: bool = False,
                 _deferred_fingerprints: Mapping[str, Any] | None = None) -> GraphSnapshot:
        r"""Materialize this graph: read each table's connector, run the
        result through `kumo-core` to build column stats and the on-disk
        index. The index is stored under a directory keyed by both `self.id`
        and a fingerprint of every table's data version (`data_version`) --
        see the module docstring for why `self.id` alone isn't a safe key,
        and `GraphSnapshot`'s docstring for the resulting `id` shape. If a
        valid build already exists for that exact key, ingest is skipped and
        it's reused as is. For local files, exact content hashes are checked
        before staging, so unchanged snapshots also skip source copying.
        ``enable_updates=True`` retains encoding caches for later updates.
        With ``previous_snapshot``, reuse unchanged encoded tables and edges.
        ``force_tables`` re-encodes named tables; ``reuse_encoders=False`` fits
        fresh encoders for rebuilt tables.
        ``force_full_rebuild=True`` ignores the parent and all cached outputs.
        """
        from filelock import FileLock

        enable_updates = enable_updates or previous_snapshot is not None or bool(force_tables)
        if force_full_rebuild:
            previous_snapshot = None
            force_tables = tuple(self._named_tables)
        self.validate()
        for table in self._named_tables.values():
            table.connector.validate_scratch_dir(get_scratch_dir())
        _warn_if_network_scratch(get_scratch_dir())

        graph_key = self.id
        persist_dir = snapshot_scratch_dir(graph_key) / 'source_tables'
        persist_dir.parent.mkdir(parents=True, exist_ok=True)

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
        with FileLock(f'{persist_dir}.lock'):
            return self._snapshot_no_lock(graph_key, persist_dir,
                                          previous_snapshot, reuse_encoders,
                                          force_tables, _deferred_fingerprints, enable_updates)

    def _snapshot_no_lock(self, graph_key: str, persist_dir: Path,
                          previous_snapshot: GraphSnapshot | None = None,
                          reuse_encoders: bool = True,
                          force_tables: Sequence[str] = (),
                          deferred_fingerprints: Mapping[str, Any] | None = None,
                          enable_updates: bool = False,
                          stage_all: bool = False) \
            -> GraphSnapshot:
        r"""`snapshot()`'s body, run under its `FileLock`. Not meant to be
        called on its own -- `persist_dir`, unlike everything else here,
        isn't a pure function of `self` alone (it's `snapshot()`'s lock key,
        computed once there and threaded through), so calling this directly
        would skip the lock that makes it safe to stage into.
        """
        started = time.perf_counter()
        from kumo_core.diskgraph.ingest import (
            DiskgraphIngestSpec,
            execute_diskgraph_ingest,
        )
        from kumo_core.diskgraph.incremental import fingerprint_files
        from kumo_core.diskgraph.runner import IngestExecutionConfig
        from kumo_core.diskgraph.snapshot import (
            DiskgraphIngestResult,
            DiskgraphLocalConfig,
            DiskgraphSnapshotHandle,
            LocalSnapshotStore,
            check_diskgraph_output,
        )
        from kumo_core.sources.layout import staging_dir
        from kumo_core.sources.transfer import discard_destination
        from kumoapi.source_table import CustomSQLSourceTable

        from kumo.engine.connector.file import FileConnector
        from kumo.engine.connector.source_table import checked_sql

        graph_config = self._graph_config
        store = LocalSnapshotStore(
            DiskgraphLocalConfig(data_root=str(get_scratch_dir() /
                                               'snapshots')))

        table_data_versions: Dict[str, str] = {}
        source_paths: Dict[str, str] = {}
        source_fingerprints = {}
        unknown = set(force_tables) - self._named_tables.keys()
        if unknown:
            raise ValueError(f'Unknown force_tables: {sorted(unknown)}')
        if previous_snapshot is not None:
            if not (previous_snapshot.index_path / '_incremental' /
                    'manifest.json').is_file():
                raise ValueError('Previous snapshot has no incremental cache; '
                                 'create a new full snapshot first.')

        # Local file versions hash the exact source bytes. An unchanged graph
        # can reuse its completed index before copying or decoding any table.
        # Other providers' metadata tokens do not promise exact content identity.
        if all(isinstance(table.connector, FileConnector)
               for table in self._named_tables.values()):
            source_fingerprints = {
                alias: table.connector.source_backend.get_source_fingerprint(
                    table.connector, table.table_name, include_row_groups=enable_updates)
                for alias, table in self._named_tables.items()
            }
            table_data_versions = {name: fp.content_hash
                                   for name, fp in source_fingerprints.items()}
            data_version = fingerprint(
                json.dumps({'tables': table_data_versions,
                            'format': 'incremental-v1' if enable_updates else 'exact-v1'}, sort_keys=True).encode())
            handle = DiskgraphSnapshotHandle(graph_key=graph_key,
                                             graph_data_version=data_version)
            existing = check_diskgraph_output(store.path(handle))
            if existing is not None and previous_snapshot is None and not force_tables:
                _warn_ingest_data_loss(existing, graph_config)
                return GraphSnapshot(f'{graph_key}@{data_version[:16]}',
                                     Path(existing.index_path), graph_config, self)

        if (previous_snapshot is not None and reuse_encoders and not force_tables
                and not deferred_fingerprints and not stage_all and source_fingerprints
                and graph_key != previous_snapshot.id.split('@')[0]):
            from kumo.engine.expansion import try_expansion

            expanded = try_expansion(self, previous_snapshot, source_fingerprints)
            if expanded is not None:
                return expanded

        if (previous_snapshot is not None and reuse_encoders and not force_tables
                and not deferred_fingerprints and not stage_all and source_fingerprints
                and graph_key == previous_snapshot.id.split('@')[0]):
            from kumo.engine.feature_updates import try_feature_update

            updated = try_feature_update(self, previous_snapshot, source_fingerprints)
            if updated is not None:
                return updated

        reusable_versions, skipped = {}, []
        if (previous_snapshot is not None and not stage_all
                and graph_key == previous_snapshot.id.split('@')[0]):
            reusable_versions = json.loads(
                (previous_snapshot.index_path / '_incremental/manifest.json').read_text())['versions']

        try:
            persist_dir.mkdir(parents=True, exist_ok=True)
            for alias, table in self._named_tables.items():
                assert table.definition is not None  # portable_graph_config
                # already required this.
                connector = table.connector
                source_backend = connector.source_backend
                candidate = (deferred_fingerprints or {}).get(alias, source_fingerprints.get(alias))
                if (candidate is not None and alias not in force_tables
                        and reusable_versions.get(alias) == candidate.content_hash):
                    reader, name = source_backend._reader(connector, table.table_name)
                    source_paths[alias] = str(reader.resolve(name)[0])
                    skipped.append(alias)
                    continue
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
                    data_version=table_data_versions.get(alias),
                )
                if prepared.credentials is not None:
                    raise NotImplementedError(
                        "TODO(local-sdk): scoped per-table ingest credentials "
                        "(PrepareIngestResult.credentials) aren't wired through "
                        "to kumo-core's ingest yet -- matches the current "
                        "limitation in kumoai/backend/local/graph.py's "
                        "_stage_sources, which raises the same way.")
                table_data_versions[alias] = prepared.version

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

                if not isinstance(connector, FileConnector):
                    # Warehouse metadata and object mtimes can miss edits.
                    # Hash the exported bytes actually consumed by ingest.
                    source_fingerprints[alias] = fingerprint_files(
                        alias, Path(source_paths[alias]))
                    table_data_versions[alias] = source_fingerprints[alias].content_hash
                elif alias not in source_fingerprints:
                    source_fingerprints[alias] = connector.source_backend.get_source_fingerprint(
                        connector, name, include_row_groups=enable_updates)
                    if source_fingerprints[alias].content_hash != prepared.version:
                        raise RuntimeError(f'Source changed while staging {alias}')

            for alias in skipped:
                table = self._named_tables[alias]
                current = table.connector.source_backend.get_source_fingerprint(
                    table.connector, table.table_name, include_row_groups=False)
                if current.content_hash != source_fingerprints[alias].content_hash:
                    raise RuntimeError(f'Source changed while staging {alias}')

            # Mixed updates rebuild edited tables against cached history;
            # proven additions are applied to the resulting index next.
            for name, old in (deferred_fingerprints or {}).items():
                source_fingerprints[name] = old
                table_data_versions[name] = old.content_hash
            data_version = fingerprint(
                json.dumps({'tables': table_data_versions,
                            'format': 'incremental-v1' if enable_updates else 'exact-v1'}, sort_keys=True).encode())
            if previous_snapshot is not None or force_tables:
                # Encoder lineage affects the bytes even for identical sources.
                # A forced rebuild gets a new location, never its parent's.
                from uuid import uuid4
                data_version = fingerprint(json.dumps({
                    'sources': table_data_versions,
                    'parent': previous_snapshot.id if previous_snapshot else None,
                    'reuse_encoders': reuse_encoders,
                    'force': sorted(force_tables),
                    'run': uuid4().hex if force_tables else None,
                }, sort_keys=True).encode())
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
                reused_config = None
                if skipped:
                    reused_config = json.loads(
                        (previous_snapshot.index_path / 'diskgraph_config.json').read_text())
                    reused_config['tables'] = [table for table in reused_config['tables']
                                               if table['name'] in skipped]
                incremental_config = persist_dir / 'incremental.json'
                if enable_updates:
                    incremental_config.write_text(json.dumps({
                        'parent': str(previous_snapshot.index_path.resolve())
                        if previous_snapshot else None,
                        'reuse_encoders': reuse_encoders,
                        'require_reuse': skipped,
                        'force_tables': list(force_tables),
                        'snapshot_id': f'{graph_key}@{data_version[:16]}',
                        'parent_snapshot_id': previous_snapshot.id if previous_snapshot else None,
                        'source_fingerprints': {
                            name: {**asdict(fp), 'table_name': name}
                            for name, fp in source_fingerprints.items()
                        },
                    }))
                result = execute_diskgraph_ingest(
                    graph_config,
                    DiskgraphIngestSpec(
                        data_persist_dir=str(persist_dir),
                        table_data_versions=table_data_versions,
                        output_dir=str(output_dir),
                        source_paths=source_paths,
                        reused_config=reused_config,
                        execution=IngestExecutionConfig(
                            success_marker_content=predicted.to_json(),
                            extra_args=('--incremental-config',
                                        str(incremental_config)) if enable_updates else ()),
                    ),
                )

                if enable_updates:
                    record_path = output_dir / '_incremental/update.json'
                    record = json.loads(record_path.read_text())
                    record['duration_seconds'] = time.perf_counter() - started
                    native_metadata = json.loads((output_dir / 'index_metadata.json').read_text())
                    record['edges'] = {
                        edge['name']: {'status': 'reused' if edge['name'] in record['reused_edges'] else 'rebuilt',
                                       'delta_entries': 0, 'fragmentation': 0.0}
                        for edge in native_metadata['edge_relations']
                    }
                    record['total_fragmentation'] = 0.0
                    if record['update_chain_length'] >= 30:
                        warnings.warn('Snapshot has at least 30 updates; rebuild or compact it.', UserWarning)
                    record_path.write_text(json.dumps(record))
                    from kumo.engine.feature_updates import retain_sources

                    for name in self._named_tables:
                        source = (previous_snapshot.index_path / '_incremental/sources' / name
                                  if name in skipped else Path(source_paths[name]))
                        paths = [source] if source.is_file() else sorted(source.rglob('*.parquet'))
                        retain_sources(output_dir, name, paths)

            if previous_snapshot is not None and reuse_encoders:
                from kumo_core.diskgraph.incremental import warn_encoder_drift

                for name in self._named_tables:
                    relative = Path('kumo_col_stats/col_stats_full') / f'{name}.json'
                    old_stats = previous_snapshot.index_path / relative
                    new_stats = Path(result.index_path) / relative
                    if old_stats.is_file() and new_stats.is_file():
                        warn_encoder_drift(name, json.loads(old_stats.read_text()),
                                           json.loads(new_stats.read_text()))
            _warn_ingest_data_loss(result, graph_config)
            return GraphSnapshot(
                id=f'{graph_key}@{data_version[:16]}',
                index_path=Path(result.index_path),
                graph_config=graph_config,
                _graph=self,
            )
        except Exception as error:
            if skipped and 'unstaged table requires encoding' in getattr(
                    error, 'private_output_tail', ''):
                discard_destination(str(persist_dir))
                return self._snapshot_no_lock(
                    graph_key, persist_dir, previous_snapshot, reuse_encoders,
                    force_tables, deferred_fingerprints, enable_updates, stage_all=True)
            raise
        finally:
            discard_destination(str(persist_dir))
