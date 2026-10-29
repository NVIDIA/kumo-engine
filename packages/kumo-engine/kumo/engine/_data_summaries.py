# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Summaries of a training table's splits and of a graph snapshot, read
from the statistics their builds wrote (``split_stats.json``,
``col_stats.json``, ``ingest_stats.json``), with no pass over the data."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from kumo_core.instrumentation import format_error
from kumo_core.summaries import Dataset, Scalar, Summary, Table, Tag

logger = logging.getLogger(__name__)

_SPLITS = ('train', 'val', 'test')
_LABEL_STATS = {'target_mean': 'label_mean', 'target_median': 'label_q50',
                'target_min': 'label_min', 'target_max': 'label_max'}
_COLUMN_STATS = ('count', 'null_rate', 'num_unique', 'min', 'max', 'mean',
                 'std', 'q25', 'q50', 'q75', 'avg_tokens', 'max_tokens')
_TOP_CATEGORIES = 10
_EDGE_STATS = {'FRAC_SRC_NONZERO_DEGREE': 'src_match_frac',
               'FRAC_DST_NONZERO_DEGREE': 'dst_match_frac'}
# The Kumo platform's data checks.
_BAD_TIME_FRACTION = 0.05
_INVALID_FRACTION = 0.10
_MIN_MATCH_FRACTION = 0.20
_MIN_ROWS = 10
_TAG_LENGTH = 5000


def split_summaries(staging_dir: Path) -> List[Summary]:
    r"""``data/<split>/rows`` and ``entities`` per split, and its label
    balance: ``positive_rate``, or the label's mean, median, min and max."""
    stats = _read(staging_dir / 'split_stats.json')
    summaries: List[Summary] = []
    for split in _SPLITS:
        table = stats.get(f'{split}_table_stats')
        if not isinstance(table, dict):
            continue
        prefix = f'data/{split}'
        for key, name in (('num_train_rows', 'rows'),
                          ('num_entities', 'entities')):
            if _number(table.get(key)) is not None:
                summaries.append(Scalar(f'{prefix}/{name}', table[key]))
        target = table.get('target_distribution') or {}
        counts = target.get('count') or {}
        if sum(counts.values()):
            summaries.append(Scalar(f'{prefix}/positive_rate',
                                    counts.get('true', 0) /
                                    sum(counts.values())))
        summaries += [
            Scalar(f'{prefix}/{name}', target[key])
            for key, name in _LABEL_STATS.items()
            if _number(target.get(key)) is not None
        ]
    return summaries


def snapshot_summaries(index: Path, graph_config: Mapping[str, Any],
                       context: str) -> List[Summary]:
    r"""The snapshot's tables and relations (`_graph_scalars`), their data
    quality (`_quality`), ``graph/col_stats`` with a row per column, and a
    `Dataset` per source table (its location, version, schema and profile)
    in ``context``, ``training`` or ``scoring``."""
    ingest = _read(index / 'ingest_stats.json').get('stats') or {}
    columns: Dict[str, Dict[str, Any]] = _read(index / 'col_stats.json')
    relations = _relations(_read(index / 'edge_stats.json'), ingest)
    profiles = {
        table: {column: _profile(stats) for column, stats in table_stats.items()}
        for table, table_stats in sorted(columns.items())
    }
    return [
        *_graph_scalars(ingest, relations),
        *_quality(ingest, columns, relations, graph_config),
        Table('graph/col_stats',
              ('table', 'column', 'stype', *_COLUMN_STATS, 'top'),
              tuple((table, column, *profile.values())
                    for table, table_profiles in profiles.items()
                    for column, profile in table_profiles.items())),
        *_datasets(index, ingest, profiles, context),
    ]


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


def _graph_scalars(ingest: Mapping[str, Any],
                   relations: Mapping[str, Mapping[str, float]]
                   ) -> List[Summary]:
    rows = ingest.get('node_counts') or {}
    return [
        *(Scalar(f'graph/rows/{table}', count)
          for table, count in sorted(rows.items())),
        *(Scalar(f'graph/edges/{relation}/{name}', value)
          for relation, stats in sorted(relations.items())
          for name, value in stats.items() if _number(value) is not None),
        Scalar('graph/tables', len(rows)),
        Scalar('graph/edge_types', len(relations)),
    ]


def _quality(ingest: Mapping[str, Any], columns: Mapping[str, Any],
             relations: Mapping[str, Mapping[str, float]],
             graph_config: Mapping[str, Any]) -> List[Summary]:
    r"""``graph/<table>/{duplicate_pks,dropped_rows,null_time_dropped}`` and,
    when any table breaks the Kumo platform's data checks, the tag
    ``kumo.data_warnings`` naming each."""
    duplicates = _by_table(ingest.get('pk_duplicate_counts'), ':')
    null_time = _by_table(ingest.get('pk_null_timestamp_drops'), ':shard-')
    dropped = ingest.get('dropped_rows_counts') or {}
    sources = ingest.get('source_rows_by_table') or {}
    failures = ingest.get('key_failures_by_context') or {}
    summaries: List[Summary] = []
    warnings: List[str] = []
    for table in graph_config.get('tables') or ():
        name = table['table_name']
        counts = {'duplicate_pks': duplicates.get(name, 0),
                  'dropped_rows': dropped.get(name, 0),
                  'null_time_dropped': null_time.get(name, 0)}
        summaries += [Scalar(f'graph/{name}/{key}', value)
                      for key, value in counts.items()]
        stats = columns.get(name) or {}
        nodes = (ingest.get('node_counts') or {}).get(name)
        rows = sources.get(name) or nodes
        pkey = (table.get('pkey') or {}).get('name')
        bad_pks = (failures.get(f'pkey:{name}.{pkey}')
                   or {}).get('parse_failures', 0)
        time = (table.get('time_col') or {}).get('name')
        time_stats = stats.get(time) or {}
        bad_times = (counts['null_time_dropped'] +
                     time_stats.get('null_count', 0) +
                     time_stats.get('parse_fail_count', 0))
        invalid = [column for column, column_stats in stats.items()
                   if column_stats.get('count') and _invalid(column_stats) >=
                   _INVALID_FRACTION * column_stats['count']]
        warnings += [
            *([f'{name}: {bad_times / rows:.1%} missing or invalid time']
              if time and rows and bad_times >= _BAD_TIME_FRACTION * rows
              else []),
            *([f'{name}: {bad_pks} missing or invalid primary keys']
              if bad_pks else []),
            *([f"{name}: {counts['duplicate_pks']} duplicate primary keys"]
              if counts['duplicate_pks'] else []),
            *(f'{name}.{column}: at least {_INVALID_FRACTION:.0%} invalid '
              f'values' for column in sorted(invalid)),
            *([f'{name}: {nodes} rows'] if nodes is not None
              and nodes < _MIN_ROWS else []),
        ]
        for fkey in table.get('fkeys') or ():
            relation = relations.get(
                f"{name}_{fkey['name']}_to_{fkey['dst_table']}_"
                f"{fkey['dst_key']}") or {}
            for side, table_name in (('src', name),
                                     ('dst', fkey['dst_table'])):
                match = _number(relation.get(f'{side}_match_frac'))
                if match is not None and match < _MIN_MATCH_FRACTION:
                    warnings.append(
                        f"{name}.{fkey['name']} -> {fkey['dst_table']}: "
                        f"{match:.1%} of {table_name} rows match")
    if warnings:
        summaries.append(Tag('kumo.data_warnings',
                             '; '.join(warnings)[:_TAG_LENGTH]))
    return summaries


def _datasets(index: Path, ingest: Mapping[str, Any],
              profiles: Mapping[str, Mapping[str, Any]],
              context: str) -> List[Summary]:
    sources = (_read(index / '_provenance.json').get('parents')
               or {}).get('source_tables') or {}
    rows = ingest.get('node_counts') or {}
    datasets: List[Summary] = []
    for table in _read(index / 'diskgraph_config.json').get('tables') or ():
        name = table.get('kumo_source_table_name') or table.get('name')
        if name in sources:
            datasets.append(Dataset(
                name=name, source=sources[name], context=context,
                schema={column['name']: {'stype': column.get('semantic_type'),
                                         'dtype': column.get('source_dtype')}
                        for column in table.get('columns') or ()},
                profile={'rows': rows.get(table.get('name')),
                         'columns': profiles.get(table.get('name'), {})}))
    return datasets


def _by_table(counts: Optional[Mapping[str, int]],
              separator: str) -> Dict[str, int]:
    tables: Dict[str, int] = {}
    for key, count in (counts or {}).items():
        table = key.split(separator, 1)[0]
        tables[table] = tables.get(table, 0) + count
    return tables


def _invalid(stats: Mapping[str, Any]) -> int:
    return stats.get('parse_fail_count', 0) + stats.get('invalid_count', 0)


def table_source(manifest: Any) -> List[Summary]:
    r"""A `Dataset` for a training table read from its own source table,
    which a snapshot's sources do not cover."""
    from kumo.engine._tracking import fit_inputs

    source = fit_inputs(manifest).get('training_table')
    return [] if source is None else [
        Dataset('training_table', source, {}, {}, 'training')]


def _profile(stats: Mapping[str, Any]) -> Dict[str, Any]:
    count = _number(stats.get('count'))
    profile: Dict[str, Any] = {'stype': stats.get('semantic_type')}
    for name in _COLUMN_STATS:
        value = (stats.get('null_count', 0) / count if count else None) \
            if name == 'null_rate' else _number(stats.get(name))
        profile[name] = value
    top = () if stats.get('semantic_type') == 'id' else (
        stats.get('category_counts') or ())
    profile['top'] = ', '.join(f'{value}: {n}'
                               for value, n in top[:_TOP_CATEGORIES]) or None
    return profile


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


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
