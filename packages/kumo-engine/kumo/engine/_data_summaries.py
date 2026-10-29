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
from kumo_core.summaries import Dataset, Scalar, Summary, Table

logger = logging.getLogger(__name__)

_SPLITS = ('train', 'val', 'test')
_LABEL_STATS = {'target_mean': 'label_mean', 'target_median': 'label_q50',
                'target_min': 'label_min', 'target_max': 'label_max'}
_COLUMN_STATS = ('count', 'null_rate', 'num_unique', 'min', 'max', 'mean',
                 'std', 'q25', 'q50', 'q75', 'avg_tokens', 'max_tokens')
_TOP_CATEGORIES = 10


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


def snapshot_summaries(index: Path, context: str) -> List[Summary]:
    r"""``graph/rows/<table>``, ``graph/edges/<edge>``, ``graph/tables`` and
    ``graph/edge_types``; ``graph/col_stats``, a row per column; and a
    `Dataset` per source table (its location, version, schema and profile)
    in ``context``, ``training`` or ``scoring``."""
    ingest = _read(index / 'ingest_stats.json').get('stats') or {}
    columns: Dict[str, Dict[str, Any]] = _read(index / 'col_stats.json')
    sources = (_read(index / '_provenance.json').get('parents')
               or {}).get('source_tables') or {}
    config = _read(index / 'diskgraph_config.json').get('tables') or []
    rows = dict(ingest.get('node_counts') or {})
    edges = {name: count
             for name, count in (ingest.get('edge_counts') or {}).items()
             if not name.endswith('_reverse')}
    summaries: List[Summary] = [
        *(Scalar(f'graph/rows/{table}', count)
          for table, count in sorted(rows.items())),
        *(Scalar(f'graph/edges/{edge}', count)
          for edge, count in sorted(edges.items())),
        Scalar('graph/tables', len(rows)),
        Scalar('graph/edge_types', len(edges)),
    ]
    profiles = {
        table: {column: _profile(stats) for column, stats in table_stats.items()}
        for table, table_stats in sorted(columns.items())
    }
    summaries.append(Table('graph/col_stats',
                           ('table', 'column', 'stype', *_COLUMN_STATS, 'top'),
                           tuple((table, column, *profile.values())
                                 for table, table_profiles in profiles.items()
                                 for column, profile in table_profiles.items())))
    for table in config:
        name = table.get('kumo_source_table_name') or table.get('name')
        if name not in sources:
            continue
        summaries.append(Dataset(
            name=name, source=sources[name], context=context,
            schema={column['name']: {'stype': column.get('semantic_type'),
                                     'dtype': column.get('source_dtype')}
                    for column in table.get('columns') or ()},
            profile={'rows': rows.get(table.get('name')),
                     'columns': profiles.get(table.get('name'), {})}))
    return summaries


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
    top = stats.get('category_counts') or ()
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
