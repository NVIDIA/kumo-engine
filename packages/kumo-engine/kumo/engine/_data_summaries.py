# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Summaries of a training table's splits and of a graph snapshot, read
from the statistics their builds wrote (``split_stats.json``,
``col_stats.json``, ``ingest_stats.json``), with no pass over the data."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.instrumentation import format_error
from kumo_core.summaries import Note, Summary, Table, Tag

from kumo.engine._card import entity
from kumo.engine._manifest import Details

logger = logging.getLogger(__name__)

_SPLITS = ('train', 'val', 'test')
_COLUMN_STATS = ('count', 'null_rate', 'num_unique', 'min', 'max', 'mean',
                 'std', 'q25', 'q50', 'q75', 'avg_tokens', 'max_tokens')
#: Column statistics' titles, "≈" for the estimated ones.
_TITLES = {'null_rate': 'null rate', 'num_unique': 'distinct≈',
           'q25': 'q25≈', 'q50': 'q50≈', 'q75': 'q75≈',
           'avg_tokens': 'avg tokens≈', 'max_tokens': 'max tokens≈',
           'top': 'top values≈'}
_TOP_CATEGORIES = 10
_TARGET_BINS = 50  # as DiskGraph's node-regression histogram
_QUARTILES = ('target_min', 'target_percentile_25', 'target_median',
              'target_percentile_75', 'target_max')
_EDGE_STATS = {'FRAC_SRC_NONZERO_DEGREE': 'src_match_frac',
               'FRAC_DST_NONZERO_DEGREE': 'dst_match_frac'}
_HEALTH = ('table or link', 'rows', 'columns', 'dropped rows',
           'duplicate keys', 'rows without time', 'edges',
           'source rows matched', 'target rows matched', 'failed lookups')
# The Kumo platform's data checks.
_BAD_TIME_FRACTION = 0.05
_INVALID_FRACTION = 0.10
_MIN_MATCH_FRACTION = 0.20
_MIN_ROWS = 10


def split_summaries(staging_dir: Path) -> List[Summary]:
    r"""The brief table ``data/splits``: each split's rows, entities (by
    the query's entity table) and label balance, its positive rate or the
    label's mean, minimum and maximum; and a note naming a numerical
    target's bins."""
    stats = _read(staging_dir / 'split_stats.json')
    tables = {split: stats.get(f'{split}_table_stats') for split in _SPLITS}
    tables = {split: table for split, table in tables.items()
              if isinstance(table, dict)}
    targets = {split: table.get('target_distribution') or {}
               for split, table in tables.items()}
    if any(_number(target.get('target_mean')) is not None
           for target in targets.values()):
        columns = ('mean', 'min', 'max')
        label = {name: f'target_{name}' for name in columns}
    elif any(set(target.get('count') or {}) <= {'true', 'false'}
             and (target.get('count') or {}) for target in targets.values()):
        columns, label = ('entities', 'positive rate'), {}
    else:
        columns, label = ('entities',), {}
    rows = tuple(
        (split, _whole(table.get('num_train_rows')), *(
            _value(table, targets[split], name, label) for name in columns))
        for split, table in tables.items())
    query = _read(staging_dir / PROVENANCE_FILE).get('query')
    entities = f"{entity(query) or 'entities'}≈"
    summaries: List[Summary] = []
    if rows:
        summaries.append(Table('data/splits', ('split', 'rows', *(
            entities if name == 'entities' else name for name in columns)),
            rows, brief=True))
    note = _bins_note((targets.get('train') or {}))
    return summaries + ([Note('notes', note)] if note else [])


def _value(table: Mapping[str, Any], target: Mapping[str, Any], name: str,
           label: Mapping[str, str]) -> Optional[float]:
    if name == 'entities':
        return _at_most(_whole(table.get('num_entities')),
                        _whole(table.get('num_train_rows')))
    if name == 'positive rate':
        counts = target.get('count') or {}
        total = sum(counts.values())
        return counts.get('true', 0) / total if total else None
    return _number(target.get(label[name]))


def target_edges(staging_dir: Path) -> Tuple[float, ...]:
    r"""Bin edges for a numerical target from the training split's
    statistics: its histogram's bins, or `_TARGET_BINS` even bins from its
    min to its max, and its quartiles; empty without them."""
    return _target_bins(_train_target(staging_dir))[0]


def _train_target(staging_dir: Path) -> Mapping[str, Any]:
    return (_read(staging_dir / 'split_stats.json').get('train_table_stats')
            or {}).get('target_distribution') or {}


def _target_bins(target: Mapping[str, Any]) -> Tuple[Tuple[float, ...], str]:
    bins = (target.get('histogram') or {}).get('bins') or ()
    source = f"the training target's {max(len(bins) - 1, 0)} histogram bins"
    low, high = (_number(target.get(key))
                 for key in ('target_min', 'target_max'))
    if not bins and low is not None and high is not None and high > low:
        bins = [low + (high - low) * i / _TARGET_BINS
                for i in range(_TARGET_BINS + 1)]
        source = (f"{_TARGET_BINS} even bins from the training target's "
                  f"minimum to its maximum")
    edges = {*bins, *(target[key] for key in _QUARTILES
                      if _number(target.get(key)) is not None)}
    return (tuple(sorted(edges)), source) if len(edges) > 1 else ((), '')


def _bins_note(target: Mapping[str, Any]) -> Optional[str]:
    edges, source = _target_bins(target)
    if not edges:
        return None
    return (f'Target histograms cut at {source} and its quartiles '
            f'({len(edges) + 1} bins).')


def snapshot_summaries(index: Path,
                       graph_config: Mapping[str, Any]) -> List[Summary]:
    r"""``data/health``, a row per table and per link; a ``problems`` note
    per failed Kumo data check, counted by the tag ``kumo.data_warnings``;
    and ``data/columns``, a row per column."""
    ingest = _read(index / 'ingest_stats.json').get('stats') or {}
    columns: Dict[str, Dict[str, Any]] = _read(index / 'col_stats.json')
    relations = _relations(_read(index / 'edge_stats.json'), ingest)
    profiles = _profiles(columns)
    problems = _problems(ingest, columns, relations, graph_config)
    return [
        _health(ingest, columns, relations, graph_config),
        *(Note('problems', problem) for problem in problems),
        *([Tag('kumo.data_warnings', str(len(problems)))] if problems
          else []),
        Table('data/columns',
              ('table', 'column', 'stype',
               *(_TITLES.get(name, name) for name in _COLUMN_STATS),
               _TITLES['top']),
              tuple((table, column, *profile.values())
                    for table, table_profiles in profiles.items()
                    for column, profile in table_profiles.items())),
    ]


def _profiles(columns: Mapping[str, Mapping[str, Any]]
              ) -> Dict[str, Dict[str, Any]]:
    return {
        table: {column: _profile(stats) for column, stats in table_stats.items()}
        for table, table_stats in sorted(columns.items())
    }


def _relations(edge_stats: Mapping[str, Any],
               ingest: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    r"""Each relation's ``count``, ``src_match_frac`` and ``dst_match_frac``
    (the share of source and destination rows with an edge) and
    ``fk_lookup_failures``, under DiskGraph's ``<src>_<fkey>_to_<dst>_<pkey>``
    name."""
    relations: Dict[str, Dict[str, float]] = {
        name: {'count': count}
        for name, count in (ingest.get('edge_counts') or {}).items()
        if not name.endswith('_reverse')
    }
    for key, entry in edge_stats.items():
        parts = key.split('__')
        if len(parts) == 7 and isinstance(entry, dict):
            src, _, dst, _, fkey, pkey, stat = parts
            name = _EDGE_STATS.get(stat)
            relation = relations.get(f'{src}_{fkey}_to_{dst}_{pkey}')
            if name and relation is not None:
                relation[name] = entry.get('value')
    failures = ingest.get('key_failures_by_context') or {}
    for name, relation in relations.items():
        missed = failures.get(f'dst_lookup:{name}') or {}
        relation['fk_lookup_failures'] = (missed.get('lookup_failures', 0) +
                                          missed.get('membership_failures', 0))
    return relations


def _links(table: Mapping[str, Any],
           relations: Mapping[str, Mapping[str, float]]
           ) -> List[Tuple[Mapping[str, Any], Mapping[str, float]]]:
    return [(fkey, relations.get(
        f"{table['table_name']}_{fkey['name']}_to_{fkey['dst_table']}_"
        f"{fkey['dst_key']}") or {}) for fkey in table.get('fkeys') or ()]


def _health(ingest: Mapping[str, Any], columns: Mapping[str, Any],
            relations: Mapping[str, Mapping[str, float]],
            graph_config: Mapping[str, Any]) -> Table:
    r"""A row per table, then per link, with '' in the columns that apply
    only to the other kind."""
    duplicates = _by_table(ingest.get('pk_duplicate_counts'), ':')
    null_time = _by_table(ingest.get('pk_null_timestamp_drops'), ':shard-')
    dropped = ingest.get('dropped_rows_counts') or {}
    nodes = ingest.get('node_counts') or {}
    rows: List[Tuple[Any, ...]] = []
    tables = graph_config.get('tables') or ()
    for table in tables:
        name = table['table_name']
        rows.append((name, nodes.get(name), len(columns.get(name) or {}),
                     dropped.get(name, 0), duplicates.get(name, 0),
                     null_time.get(name, 0) if table.get('time_col') else '',
                     '', '', '', ''))
    for table in tables:
        for fkey, relation in _links(table, relations):
            rows.append((_link(table, fkey), '', '', '', '', '',
                         *(relation.get(key) for key in (
                             'count', 'src_match_frac', 'dst_match_frac',
                             'fk_lookup_failures'))))
    return Table('data/health', _HEALTH, tuple(rows))


def _link(table: Mapping[str, Any], fkey: Mapping[str, Any]) -> str:
    return f"{table['table_name']}.{fkey['name']} → {fkey['dst_table']}"


def _problems(ingest: Mapping[str, Any], columns: Mapping[str, Any],
              relations: Mapping[str, Mapping[str, float]],
              graph_config: Mapping[str, Any]) -> List[str]:
    r"""Each table and link that breaks the Kumo platform's data checks."""
    duplicates = _by_table(ingest.get('pk_duplicate_counts'), ':')
    null_time = _by_table(ingest.get('pk_null_timestamp_drops'), ':shard-')
    sources = ingest.get('source_rows_by_table') or {}
    failures = ingest.get('key_failures_by_context') or {}
    problems: List[str] = []
    for table in graph_config.get('tables') or ():
        name = table['table_name']
        stats = columns.get(name) or {}
        nodes = (ingest.get('node_counts') or {}).get(name)
        rows = sources.get(name) or nodes
        pkey = (table.get('pkey') or {}).get('name')
        bad_pks = (failures.get(f'pkey:{name}.{pkey}')
                   or {}).get('parse_failures', 0)
        time = (table.get('time_col') or {}).get('name')
        time_stats = stats.get(time) or {}
        bad_times = (null_time.get(name, 0) +
                     time_stats.get('null_count', 0) +
                     time_stats.get('parse_fail_count', 0))
        invalid = [column for column, column_stats in stats.items()
                   if column_stats.get('count') and _invalid(column_stats) >=
                   _INVALID_FRACTION * column_stats['count']]
        problems += [
            *([f'`{name}`: {bad_times / rows:.1%} of rows have a missing or '
               f'invalid time']
              if time and rows and bad_times >= _BAD_TIME_FRACTION * rows
              else []),
            *([f'`{name}`: {bad_pks} missing or invalid primary keys']
              if bad_pks else []),
            *([f'`{name}`: {duplicates[name]} duplicate primary keys']
              if duplicates.get(name) else []),
            *(f'`{name}.{column}`: at least {_INVALID_FRACTION:.0%} invalid '
              f'values' for column in sorted(invalid)),
            *([f'`{name}`: only {nodes} rows'] if nodes is not None
              and nodes < _MIN_ROWS else []),
        ]
        for fkey, relation in _links(table, relations):
            for side, table_name in (('src', name),
                                     ('dst', fkey['dst_table'])):
                match = _number(relation.get(f'{side}_match_frac'))
                if match is not None and match < _MIN_MATCH_FRACTION:
                    problems.append(
                        f'link `{_link(table, fkey)}`: {match:.1%} of '
                        f'`{table_name}` rows match')
    return problems


def table_details(index: Path, snapshot_id: Optional[str] = None) -> Details:
    r"""Each source table's schema and profile, its rows and column
    statistics, for its `Dataset`; given ``snapshot_id``, the snapshot's
    own profile too, its tables and rows."""
    rows = (_read(index / 'ingest_stats.json').get('stats')
            or {}).get('node_counts') or {}
    profiles = _profiles(_read(index / 'col_stats.json'))
    tables = _read(index / 'diskgraph_config.json').get('tables') or ()
    snapshot = {} if snapshot_id is None else {snapshot_id: ({}, {
        'tables': len(tables),
        'rows': sum(rows.get(table.get('name'), 0) for table in tables)})}
    return {**snapshot, **{
        table.get('kumo_source_table_name') or table.get('name'): (
            {column['name']: {'stype': column.get('semantic_type'),
                              'dtype': column.get('source_dtype')}
             for column in table.get('columns') or ()},
            {'rows': rows.get(table.get('name')),
             'columns': profiles.get(table.get('name'), {})})
        for table in tables
    }}


def training_table_details(table: Any) -> Details:
    r"""A training table's schema and profile, its rows and columns, from
    its Parquet metadata alone."""
    from pyarrow import parquet

    manifest = table.manifest
    path = Path(table.train_table_path)
    if manifest is None or not path.exists():
        return {}
    parts = sorted(path.rglob('*.parquet')) if path.is_dir() else [path]
    if not parts:
        return {}
    schema = parquet.read_schema(parts[0])
    return {manifest.id: (
        {field.name: {'stype': None, 'dtype': _dtype(field.type)}
         for field in schema},
        {'rows': sum(parquet.ParquetFile(part).metadata.num_rows
                     for part in parts),
         'columns': len(schema)})}


def _dtype(arrow_type: Any) -> str:
    text = str(arrow_type)
    return {'double': 'float64', 'float': 'float32'}.get(text, (
        'time' if text.startswith('timestamp') else
        'date' if text.startswith('date') else text))


def _by_table(counts: Optional[Mapping[str, int]],
              separator: str) -> Dict[str, int]:
    tables: Dict[str, int] = {}
    for key, count in (counts or {}).items():
        table = key.split(separator, 1)[0]
        tables[table] = tables.get(table, 0) + count
    return tables


def _invalid(stats: Mapping[str, Any]) -> int:
    return stats.get('parse_fail_count', 0) + stats.get('invalid_count', 0)


def _profile(stats: Mapping[str, Any]) -> Dict[str, Any]:
    count = _number(stats.get('count'))
    profile: Dict[str, Any] = {'stype': stats.get('semantic_type')}
    for name in _COLUMN_STATS:
        value = (stats.get('null_count', 0) / count if count else None) \
            if name == 'null_rate' else _number(stats.get(name))
        profile[name] = value
    profile['num_unique'] = _at_most(profile['num_unique'], count)
    top = () if stats.get('semantic_type') == 'id' else (
        stats.get('category_counts') or ())
    profile['top'] = ', '.join(f'{value}: {n}'
                               for value, n in top[:_TOP_CATEGORIES]) or None
    return profile


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _at_most(estimate: Optional[float],
             exact: Optional[float]) -> Optional[float]:
    return None if estimate is None else (
        estimate if exact is None else min(estimate, exact))


def _whole(value: Any) -> Optional[int]:
    return None if _number(value) is None else int(value)


def _read(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        logger.warning('Could not read %s for the run summary: %s', path,
                       format_error(error))
        return {}
    return value if isinstance(value, dict) else {}
