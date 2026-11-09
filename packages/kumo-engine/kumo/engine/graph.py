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
import math
import struct
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from functools import cached_property, partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.csv_rules import looks_missing
from kumo_core.pquery import NativeGraph, normalize_graph
from kumoapi.graph import Edge
from kumoapi.typing import (
    Dtype,
    TimestampPrecision,
    validate_timestamp_precision,
)

from kumo.engine._events import Event, EventEmitter
from kumo.engine._infer import _SAMPLE_ROWS
from kumo.engine._manifest import (
    RECONSTRUCTED,
    BuildCost,
    SnapshotManifest,
    SnapshotParents,
    SourceTableEntry,
    read_manifest,
    stamp,
    write_manifest,
)
from kumo.engine._prep_cost import build_cost, start
from kumo.engine._provenance import source_table
from kumo.engine._scratch import (
    fingerprint,
    get_scratch_dir,
    snapshot_scratch_dir,
)
from kumo.engine._storage import network_storage
from kumo.engine._visualization import SVG
from kumo.engine.connector.base import Connector
from kumo.engine.table import Table

if TYPE_CHECKING:
    from kumo.engine.trainer.output_config import SnapshotOutputConfig

logger = logging.getLogger(__name__)
_warned_scratch: set[Path] = set()
_warned_backfill: set[Path] = set()


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


_FLOAT32_EXACT_BITS = 24
_FLOAT32_DTYPES = frozenset({Dtype.float, Dtype.float32})
_FLOAT64_DIGITS = sys.float_info.dig
_FLOAT64_PLACES = math.ceil(
    (sys.float_info.mant_dig - sys.float_info.min_exp) * math.log10(2))
_FLOAT32_SMALLEST = 2.0**-149
_SAMPLE_BUFFER_SIZE = 1 << 20
_NUMERICAL = 'kumo.numerical'


def _read_column_stats(kumo_stats_dir: Any, table: str) -> Dict[str, Any]:
    try:
        path = Path(kumo_stats_dir) / 'col_stats_min' / f'{table}.json'
        column_stats = json.loads(path.read_text())
    except (OSError, TypeError, ValueError):
        return {}
    return column_stats if isinstance(column_stats, dict) else {}


def _by_column(column_stats: Mapping[str, Any],
               stype: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    r"""The ``__False`` statistics of each column, keyed by statistic name;
    ``stype`` keeps only the columns DiskGraph collected under that key."""
    stats: Dict[str, Dict[str, Any]] = {}
    for key, entry in column_stats.items():
        parts = key.rsplit('__', 3)
        if (isinstance(entry, dict) and len(parts) == 4 and parts[3] == 'False'
                and stype in (None, parts[1])):
            stats.setdefault(parts[0], {})[parts[2]] = entry.get('value')
    return stats


def _count(stat: Mapping[str, Any], key: str) -> int:
    value = stat.get(key)
    return value if type(value) is int and value > 0 else 0


def _non_finite(stat: Mapping[str, Any]) -> int:
    fraction = stat.get('INVALID_FRACTION')
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        return 0
    return round(fraction * _count(stat, 'COUNT')) if fraction > 0 else 0


def invalid_value_counts(kumo_stats_dir: Any,
                         tables: Sequence[str]) -> Dict[str, Dict[str, int]]:
    r"""Per table and column, how many values the declared type could not
    use at ingest: values it could not read, stored as missing, and
    non-finite numbers, stored as is but left out of the statistics."""
    counts: Dict[str, Dict[str, int]] = {}
    for table in sorted(tables):
        column_stats = _read_column_stats(kumo_stats_dir, table)
        numerical = _by_column(column_stats, _NUMERICAL)
        for column, stat in sorted(_by_column(column_stats).items()):
            count = (_count(stat, 'NUM_PARSE_FAILURES') +
                     _non_finite(numerical.get(column, {})))
            if count:
                counts.setdefault(table, {})[column] = count
    return counts


def _float32_limit(places: int) -> float:
    return float(2**(_FLOAT32_EXACT_BITS - math.ceil(places * math.log2(10))))


def _places(number: Decimal) -> int:
    r"""Decimal places after the trailing zeros, read off the digits so the
    caller's decimal context cannot round or trap. A step below the smallest
    float64 is zero in every float and so is no step."""
    digits, exponent = number.as_tuple()[1:]
    if not number.is_finite() or not any(digits):
        return 0
    zeros = 0
    for digit in reversed(digits):
        if digit:
            break
        zeros += 1
    places = max(0, -(exponent + zeros))
    return places if places <= _FLOAT64_PLACES else 0


def _float32_zero(number: Decimal) -> bool:
    r"""Whether float32 stores the value as 0, decided by the round trip
    itself, so a value that rounds up to the smallest subnormal does not
    count."""
    value = float(number) if number.is_finite() else 0.0
    return (0 < abs(value) < _FLOAT32_SMALLEST
            and struct.unpack('f', struct.pack('f', value))[0] == 0)


def _float32_infinite(number: Decimal) -> bool:
    r"""Whether float32 stores a finite value as infinity, decided by the
    round trip itself."""
    value = float(number) if number.is_finite() else 0.0
    if not math.isfinite(value):
        return False
    try:
        return math.isinf(struct.unpack('f', struct.pack('f', value))[0])
    except OverflowError:
        return True


def _sampled_precision(values: Any) -> Tuple[int, int, int]:
    r"""The most decimal places among the values, each read as the shortest
    decimal that names it, so a declared scale no value uses does not
    count, and how many of them float32 stores as 0 and as infinity."""
    places = underflows = overflows = 0
    for value in values.to_pylist():
        try:
            number = Decimal(
                value if isinstance(value, Decimal) else str(value).strip())
        except InvalidOperation:
            continue
        places = max(places, _places(number))
        underflows += _float32_zero(number)
        overflows += _float32_infinite(number)
    return places, underflows, overflows


def _reads_as_number(physical: Any) -> bool:
    r"""Whether DiskGraph parses the physical type into a float32 column:
    numbers, decimals and text or bytes it decodes as text. Anything else,
    such as a date, becomes missing with a parse-failure count of its own,
    and the sampler cannot read every value of it either."""
    import pyarrow as pa

    if pa.types.is_dictionary(physical):
        return _reads_as_number(physical.value_type)
    return (pa.types.is_floating(physical) or pa.types.is_integer(physical)
            or pa.types.is_decimal(physical) or pa.types.is_string(physical)
            or pa.types.is_large_string(physical)
            or pa.types.is_binary(physical)
            or pa.types.is_large_binary(physical))


def _narrowed_float_columns(
        alias: str, table: Table, paths: Sequence[str],
        source_path: Any) -> Dict[Tuple[str, str], Tuple[str, int, int, int]]:
    r"""Each float32 column staged from a wider type DiskGraph reads as a
    number, with that type, the decimal places its values are written with
    and how many float32 stores as 0 and as infinity, read from the table's
    first ``_SAMPLE_ROWS`` rows across its parts, all such columns together
    and through a bounded buffer rather than whole column chunks. A backend
    that names a directory instead of its parts has it globbed, only when
    the table has a float32 column; a part of any other kind, such as the
    CSV a custom backend may stage, is left to DiskGraph's own reading."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    assert table.definition is not None
    columns = [
        column.name for column in table.definition.cols
        if column.dtype in _FLOAT32_DTYPES
    ]
    if not columns:
        return {}
    narrowed: Dict[Tuple[str, str], Tuple[str, int, int, int]] = {}
    budget = _SAMPLE_ROWS
    parts = [path for path in paths if Path(path).suffix.lower() == '.parquet']
    if not paths:
        parts = sorted(
            str(part) for part in Path(source_path).rglob('*.parquet'))
    for path in parts:
        parquet = pq.ParquetFile(path, pre_buffer=False,
                                 buffer_size=_SAMPLE_BUFFER_SIZE)
        schema = parquet.schema_arrow
        sampled = []
        for column in columns:
            if column not in schema.names:
                continue
            physical = schema.field(column).type
            if (pa.types.is_float32(physical) or pa.types.is_float16(physical)
                    or not _reads_as_number(physical)):
                continue
            narrowed.setdefault((alias, column), (str(physical), 0, 0, 0))
            if not pa.types.is_integer(physical):
                sampled.append(column)
        if budget <= 0 or not sampled:
            continue
        for batch in parquet.iter_batches(budget, columns=sampled,
                                          use_threads=False):
            for column in sampled:
                physical, places, zeros, infinities = narrowed[alias, column]
                more, below, above = _sampled_precision(batch.column(column))
                narrowed[alias, column] = (physical, max(places, more),
                                           zeros + below, infinities + above)
            budget -= batch.num_rows
            break
    return narrowed


def _warn_missing_like_text(table: str, column: Mapping[str, Any],
                            stat: Mapping[str, Any]) -> None:
    if (column.get('dtype') != 'string'
            or column.get('stype') != 'categorical'):
        return
    counts = stat.get('CATEGORY_COUNTS')
    if not isinstance(counts, list):
        return
    found = {
        entry[0]: entry[1]
        for entry in counts if isinstance(entry, list) and len(entry) == 2
        and looks_missing(entry[0]) and type(entry[1]) is int and entry[1] > 0
    }
    if not found:
        return
    listed = sum(entry[1] for entry in counts if isinstance(entry, list)
                 and len(entry) == 2 and type(entry[1]) is int)
    present = _count(stat, 'COUNT') - _count(stat, 'NUM_NA')
    bound = 'at least ' if listed < present else ''
    tokens = ', '.join(repr(value) for value in sorted(found))
    warnings.warn(
        f"{table}.{column['name']} holds {bound}{sum(found.values())} "
        f"value(s) that look like missing ({tokens}); they are kept as text. "
        "If they mean missing, replace them with nulls at the source; a "
        "number or date column reads them as missing.", UserWarning,
        stacklevel=3)


def _magnitude(stat: Mapping[str, Any]) -> Optional[float]:
    values = [stat.get('MIN'), stat.get('MAX')]
    counts = stat.get('CATEGORY_COUNTS')
    if isinstance(counts, list):
        values += [
            entry[0] for entry in counts if isinstance(entry, list) and entry
        ]
    numbers = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, bool) and math.isfinite(number):
            numbers.append(abs(number))
    return max(numbers, default=None)


def _warn_float32_narrowing(
        table: str, column: Mapping[str, Any], stat: Mapping[str, Any],
        source: Optional[Tuple[str, int, int, int]]) -> None:
    if source is None:
        return
    physical, places, underflows, overflows = source
    if underflows:
        warnings.warn(
            f"{table}.{column['name']} is stored as float32 but comes from "
            f"{physical} and holds {underflows} sampled value(s) too small "
            "for float32, which stores them as 0. Declare dtype=Dtype.float64 "
            "for the column to keep them.", UserWarning, stacklevel=3)
    if overflows:
        warnings.warn(
            f"{table}.{column['name']} is stored as float32 but comes from "
            f"{physical} and holds {overflows} sampled value(s) too large "
            "for float32, which stores them as infinity. Declare "
            "dtype=Dtype.float64 for the column to keep them.", UserWarning,
            stacklevel=3)
    magnitude = _magnitude(stat)
    if magnitude is None:
        return
    limit = _float32_limit(places)
    if magnitude < limit:
        return
    step = ('every whole number'
            if places == 0 else f"steps of {10**-places:g}")
    reaches = (f"{magnitude:,.{places}f}" if places <= _FLOAT64_DIGITS else
               f"{magnitude:.{_FLOAT64_DIGITS}g}")
    warnings.warn(
        f"{table}.{column['name']} is stored as float32 but comes from "
        f"{physical} and reaches {reaches}; from {limit:,.0f} on "
        f"float32 no longer resolves {step}, so values are rounded. Declare "
        "dtype=Dtype.float64 for the column to keep full precision.",
        UserWarning, stacklevel=3)


def _warn_ingest_data_loss(
    result: Any,
    graph_config: Dict[str, Any],
    narrowed: Optional[Mapping[Tuple[str, str], Tuple[str, int, int,
                                                      int]]] = None,
) -> None:
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
        stats = _by_column(column_stats)
        numerical = _by_column(column_stats, _NUMERICAL)
        for column, stat in sorted(numerical.items()):
            count = _non_finite(stat)
            if count:
                warnings.warn(
                    f"{count} non-finite value(s) (inf, -inf or NaN) found "
                    f"in {name}.{column}. They are stored as is and left out "
                    "of the column statistics. Infinity usually comes from a "
                    "division by zero upstream; replace non-finite values at "
                    "the source.", UserWarning, stacklevel=3)
        for column in table.get('cols', []):
            stat = stats.get(column['name'], {})
            _warn_missing_like_text(name, column, stat)
            _warn_float32_narrowing(name, column, stat, (narrowed or {}).get(
                (name, column['name'])))

    try:
        stats = json.loads(Path(result.ingest_stats_path).read_text())['stats']
    except (AttributeError, OSError, ValueError, KeyError, TypeError):
        return
    if not isinstance(stats, dict):
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
    old, new = _schema_tables(before), _schema_tables(after)
    changes = [
        f"table {name} {'removed' if name in old else 'added'}"
        for name in sorted(old.keys() ^ new.keys())
    ]
    for name in sorted(old.keys() & new.keys()):
        parts = sorted(
            _SCHEMA_PARTS.get(key, key) + (_changed_columns(
                old[name][key], new[name][key]) if key == 'cols' else '')
            for key in old[name].keys() | new[name].keys()
            if old[name].get(key) != new[name].get(key))
        if parts:
            changes.append(f"{name} {', '.join(parts)}")
    return changes


def _changed_columns(before: List[str], after: List[str]) -> str:
    old, new = ({
        column['name']: column
        for column in map(json.loads, side or [])
    } for side in (before, after))
    names = sorted(name for name in old.keys() | new.keys()
                   if old.get(name) != new.get(name))
    return f" ({', '.join(names)})" if names else ''


def report_other_graph(message: str, before: Optional[Mapping[str, Any]],
                       after: Mapping[str, Any], stacklevel: int,
                       strict: bool = False) -> None:
    r"""Warn of a snapshot of another graph; strict raises on schema change."""
    changes = None if before is None else schema_changes(before, after)
    if changes and strict:
        raise ValueError(
            f"{message} Schema changes: {'; '.join(changes)}; declare the "
            f"same tables, column types and edges to score with "
            f"on_error='raise'.")
    if changes is None:
        detail = ("The model does not record its training graph, so what "
                  "differs is unknown; a different connector or table "
                  "location alone also changes the graph id.")
    elif changes:
        detail = f"Schema changes: {'; '.join(changes)}."
    else:
        detail = ("The schema is the same; only table connectors or "
                  "locations differ.")
    warnings.warn(f"{message} {detail}", UserWarning,
                  stacklevel=stacklevel + 1)


class _SnapshotEvents(EventEmitter):

    def __init__(self, callbacks: Sequence[Callable[[Event], None]]):
        super().__init__(callbacks)
        self._started = time.monotonic()

    def emit(self, name: Any, **fields: Any) -> None:
        params = {**fields.pop('params', {}),
                  'seconds': round(time.monotonic() - self._started, 1)}
        logger.info('Graph.snapshot: %s %s', name, params)
        super().emit(name, params=params, **fields)


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


def _load_local(connector: Any, path: str, root: str,
                manifest: Dict[str, Any]) -> Any:
    from kumo_core.diskgraph.snapshot import DiskgraphSnapshotHandle

    from kumo.engine._snapshot_io import holds_export, load_into, read

    store = _store()
    handle = DiskgraphSnapshotHandle(
        graph_key=manifest['graph_key'],
        graph_data_version=manifest['data_version'])
    target = Path(store.path(handle))
    with _lock(manifest['graph_key']):
        result = _loaded(store, handle)
        if result is not None and holds_export(Path(result.index_path),
                                               manifest):
            return result
        try:
            load_into(connector, path, root, manifest, target)
        except FileNotFoundError:
            newer_root, newer = read(connector, path)
            if (newer_root == root
                    or newer['snapshot_id'] != manifest['snapshot_id']):
                raise
            load_into(connector, path, newer_root, newer, target)
        result = _loaded(store, handle)
    if result is None:
        raise RuntimeError(f'The snapshot loaded from {path} did not pass '
                           'its integrity check.')
    return result


def _snapshot_id(graph_key: str, data_version: str) -> str:
    return f'{graph_key}@{data_version[:16]}'


_COMPARABLE = frozenset({'exact', 'timestamp', 'listing'})


def _current_source(table: Table,
                    record_sources: bool) -> Optional[SourceTableEntry]:
    from kumoapi.source_table import CustomSQLSourceTable

    if isinstance(table.definition.source_table, CustomSQLSourceTable):
        return None
    connector = table.connector
    try:
        version = connector.source_backend.get_data_version(
            connector, table.table_name)
    except Exception:
        return None
    return source_table(connector, table.table_name, None, version,
                        record_sources=record_sources)


def _source_change(table: Table, recorded: Optional[SourceTableEntry],
                   record_sources: bool) -> Dict[str, Any]:
    now = _current_source(table, record_sources)
    before = None if recorded is None else recorded.version
    current = None if now is None else now.version
    known = (None not in (recorded, now, before, current)
             and {recorded.version_kind, now.version_kind} <= _COMPARABLE
             and _place(recorded) == _place(now))
    status = ('unknown' if not known else
              'unchanged' if before == current else 'changed')
    return {'status': status, 'recorded': before, 'current': current}


def _place(entry: SourceTableEntry) -> tuple:
    return entry.connector, entry.location, entry.location_fingerprint


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
        return self.id.partition('@')[0]

    @property
    def _data_version(self) -> str:
        return Path(self.index_path).name

    @cached_property
    def _native_graph(self) -> NativeGraph:
        return normalize_graph(self.graph_config)

    def export(self, output_config: 'SnapshotOutputConfig') -> str:
        r"""Export the snapshot via ``output_config``; returns its location."""
        from kumo.engine._snapshot_io import export

        return export(self, output_config.connector, output_config.path)

    @classmethod
    def load_from_connector(cls, connector: Connector, *,
                            path: Optional[str] = None,
                            id: Optional[str] = None,
                            graph: Optional['Graph'] = None) -> 'GraphSnapshot':
        r"""Load a snapshot exported at `path`.

        Args:
            connector: a connector, or a `kumo.MLOps` for its artifact root.
            path: the location `export` returned.
            id: a snapshot id under the `kumo.MLOps` root, instead of `path`.
            graph: warn if its sources changed since the snapshot.
        """
        from kumo.engine._mlops import MLOps, on_root, snapshot_path
        from kumo.engine._snapshot_io import read

        if (path is None) == (id is None):
            raise ValueError('Pass exactly one of path or id.')
        if id is not None:
            if not isinstance(connector, MLOps):
                raise TypeError('id= loads from a kumo.MLOps artifact root; '
                                'pass path= with a connector.')
            path = snapshot_path(id)
        connector, path = on_root(connector, path)
        root, manifest = read(connector, path)
        if graph is not None and graph.id != manifest['graph_key']:
            report_other_graph(
                f"Snapshot at {path} was taken of another graph.",
                graph._graph_config, manifest['graph_config'], stacklevel=2)
        result = _load_local(connector, path, root, manifest)
        snapshot = cls(id=manifest['snapshot_id'],
                       index_path=Path(result.index_path),
                       graph_config=manifest['graph_config'])
        if graph is not None:
            snapshot._warn_changed_sources(graph, path)
        return snapshot

    def _warn_changed_sources(self, graph: 'Graph', path: str) -> None:
        changed = [
            alias for alias, change in self.source_changes(graph).items()
            if change['status'] == 'changed'
        ]
        if changed:
            warnings.warn(
                f"Snapshot at {path} predates source table(s) "
                f"{', '.join(changed)}; see `snapshot.source_changes`.",
                UserWarning, stacklevel=3)

    def source_changes(self, graph: 'Graph') -> Dict[str, Dict[str, Any]]:
        r"""Each of `graph`'s tables as `unchanged`, `changed` or `unknown`."""
        from kumo_core.sources.parallel import run_in_parallel

        try:
            manifest = self.manifest
            recorded = ({} if manifest is None or manifest.reconstructed else
                        dict(manifest.parents.source_tables))
            record_sources = (manifest is None
                              or manifest.record_sources is not False)
        except Exception:
            recorded, record_sources = {}, True
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
        """The stored snapshot manifest, if any."""
        return read_manifest(Path(self.index_path) / PROVENANCE_FILE,
                             SnapshotManifest)

    @property
    def invalid_value_counts(self) -> Dict[str, Dict[str, int]]:
        r"""Per table and column, the number of values the declared type
        could not use at ingest: values it could not read, which are stored
        as missing, and non-finite numbers, which are stored as is and left
        out of the statistics. Columns with none are left out."""
        return invalid_value_counts(
            Path(self.index_path) / 'kumo_col_stats',
            [table['table_name'] for table in self.graph_config['tables']])


def _write_snapshot_manifest(result: Any, snapshot_id: str,
                             source_tables: Dict[str, SourceTableEntry],
                             record_sources: bool, *, fresh: bool,
                             build: Optional[BuildCost] = None) -> None:
    path = Path(result.index_path) / PROVENANCE_FILE
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
                record_sources=record_sources, build=build,
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


def index_timestamp_precision(index_path: Path) -> TimestampPrecision:
    r"""The clock an index stores timestamps in (seconds when it records
    none)."""
    path = Path(index_path) / 'index_metadata.json'
    if not path.is_file():
        return 'seconds'
    return json.loads(path.read_text()).get('timestamp_unit', 'seconds')


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

    def snapshot(self,
                 timestamp_precision: TimestampPrecision = 'seconds', *,
                 record_sources: bool = True,
                 mlops: Optional[Any] = None,
                 callbacks: Sequence[Callable[[Event], None]] = ()
                 ) -> GraphSnapshot:
        r"""Materialize this graph: read each table's connector, run the
        result through `kumo-core` to build column stats and the on-disk
        index. The index is stored under a directory keyed by both `self.id`
        and a fingerprint of every table's data version (`data_version`) --
        see the module docstring for why `self.id` alone isn't a safe key,
        and `GraphSnapshot`'s docstring for the resulting `id` shape. If a
        valid build already exists for that exact key, ingest is skipped and
        it's reused as is.

        `timestamp_precision` is the clock the index stores timestamps in:
        `"nanoseconds"`, `"microseconds"`, `"milliseconds"`, `"seconds"` (the
        default), `"hours"` or `"days"`. Window lengths truncate to the clock,
        and a window shorter than one tick is a configuration error. Each
        clock is cached as a separate snapshot.

        Args:
            mlops: also export the snapshot under this `kumo.MLOps` root.
            record_sources: if False, record sources without locations or
                SQL text.
            callbacks: called with each progress `Event`.
        """
        validate_timestamp_precision(timestamp_precision)
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
        events = _SnapshotEvents(callbacks)
        events.emit('snapshot_started', params={
            'graph': graph_key, 'tables': list(self._named_tables)})
        with _lock(graph_key):
            snapshot = self._snapshot_no_lock(graph_key, persist_dir,
                                              timestamp_precision,
                                              record_sources, events,
                                              timed=settings is not None)
        if settings is not None and settings.wants('snapshot'):
            from kumo.engine._mlops import raise_on_error
            from kumo.engine._snapshot_io import store

            try:
                store(snapshot, settings)
            except Exception as error:
                raise_on_error(settings, error, 'Graph.snapshot')
        return snapshot

    def _snapshot_no_lock(
            self, graph_key: str, persist_dir: Path,
            timestamp_precision: TimestampPrecision = 'seconds',
            record_sources: bool = True,
            events: Optional['_SnapshotEvents'] = None,
            *, timed: bool = False) -> GraphSnapshot:
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
        from kumo_core.diskgraph.stores import diskgraph_index_size_bytes
        from kumo_core.sources.layout import staging_dir
        from kumo_core.sources.transfer import discard_destination
        from kumoapi.source_table import CustomSQLSourceTable

        from kumo.engine.connector.source_table import checked_sql

        graph_config = self._graph_config
        store = _store()

        table_data_versions: Dict[str, str] = {}
        source_paths: Dict[str, str] = {}
        source_tables: Dict[str, SourceTableEntry] = {}
        narrowed: Dict[Tuple[str, str], Tuple[str, int, int, int]] = {}

        started = start() if timed else None
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
                if events is not None:
                    events.emit('table_staged', params={'table': alias})
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
                narrowed.update(
                    _narrowed_float_columns(alias, table, paths,
                                            source_paths[alias]))

            version_parts = [
                json.dumps(table_data_versions, sort_keys=True).encode()
            ]
            if timestamp_precision != 'seconds':
                version_parts.append(timestamp_precision.encode())
            data_version = fingerprint(*version_parts)
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
                        timestamp_precision=timestamp_precision,
                        execution=IngestExecutionConfig(
                            success_marker_content=predicted.to_json()),
                    ),
                )

            snapshot_id = _snapshot_id(graph_key, data_version)
            build = None if started is None or existing is not None else (
                build_cost(snapshot_id, started,
                           diskgraph_index_size_bytes(result.index_path)))
            _write_snapshot_manifest(result, snapshot_id, source_tables,
                                     record_sources, fresh=existing is None,
                                     build=build)

            _warn_ingest_data_loss(result, graph_config, narrowed)
            if events is not None:
                events.emit('snapshot_ended', params={
                    'snapshot_id': snapshot_id, 'reused': existing is not None})
            return GraphSnapshot(
                id=snapshot_id,
                index_path=Path(result.index_path),
                graph_config=graph_config,
                record_sources=record_sources,
            )
        finally:
            discard_destination(str(persist_dir))
