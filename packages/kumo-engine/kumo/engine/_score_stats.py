# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Score statistics of prediction Parquet files, computed by DiskGraph."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from kumo_core.types import TaskType

TOP_SCORE = 'top_score'
TOP_CLASSES = 100
#: The column each task type's scores are summarised by.
SCORE_COLUMNS = {
    TaskType.BINARY_CLASSIFICATION: 'True_PROB',
    TaskType.REGRESSION: 'TARGET_PRED',
    TaskType.FORECASTING: 'TARGET_PRED',
    TaskType.MULTICLASS_CLASSIFICATION: TOP_SCORE,
    TaskType.MULTILABEL_CLASSIFICATION: TOP_SCORE,
    TaskType.MULTILABEL_RANKING: TOP_SCORE,
    TaskType.STATIC_LINK_PREDICTION: TOP_SCORE,
    TaskType.TEMPORAL_LINK_PREDICTION: TOP_SCORE,
}
_CLASS_COLUMNS = {'CLASS', 'SCORE', 'LABEL', 'PREDICTED'}

Paths = Sequence[Union[str, Path]]


def parquet_stats(paths: Paths, columns: Sequence[str], *,
                  max_per: Optional[Sequence[str]] = None
                  ) -> Dict[str, Dict[str, Any]]:
    r"""Summaries of ``columns``: percentiles of numbers, counts of labels."""
    import diskgraph
    from pyarrow import parquet

    schema = parquet.read_schema(paths[0])
    specs = [_spec(name, schema.field(name).type) for name in columns]
    full, _ = diskgraph.compute_col_stats_for_parquet(
        [str(path) for path in paths], json.dumps(specs), '',
        percentiles=True, summary_stats_by_class=False, histograms=False,
        category_counts_limit=TOP_CLASSES,
        max_per=list(max_per or []))
    stats = json.loads(full)
    return {
        spec['name']: _summary(stats, spec['name'], spec['stype'])
        for spec in specs
    }


def top_score_stats(paths: Paths) -> Dict[str, Any]:
    r"""Summary of each prediction's highest ``SCORE``.

    A prediction is a run of consecutive rows with equal non-class columns
    (entity, and anchor when present), in any score order.
    """
    from pyarrow import parquet

    keys = [
        name for name in parquet.read_schema(paths[0]).names
        if name not in _CLASS_COLUMNS and not name.startswith('__')
    ]
    return parquet_stats(paths, ['SCORE'], max_per=keys)['SCORE']


def positive_rate(summary: Dict[str, Any]) -> Optional[float]:
    r"""Share of true/1 values in a label column's summary."""
    counts = summary.get('top') or []
    total = sum(count for _, count in counts)
    positive = sum(count for value, count in counts if value in (True, 1))
    return positive / total if total else None


def _spec(name: str, dtype: Any) -> Dict[str, str]:
    import pyarrow as pa

    if pa.types.is_floating(dtype):
        return {'name': name, 'stype': 'numerical',
                'dtype': 'float32' if pa.types.is_float32(dtype) else
                'float64'}
    if pa.types.is_boolean(dtype):
        return {'name': name, 'stype': 'categorical', 'dtype': 'bool'}
    if pa.types.is_integer(dtype):
        return {'name': name, 'stype': 'categorical', 'dtype': 'int64'}
    return {'name': name, 'stype': 'categorical', 'dtype': 'string'}


def _summary(stats: Dict[str, Any], name: str,
             stype: str) -> Dict[str, Any]:
    def value(stat: str, approx: bool = False) -> Any:
        entry = stats.get(f'{name}__kumo.{stype}__{stat}__{approx}')
        return None if entry is None else entry['value']

    if stype == 'numerical':
        rows = value('COUNT') or 0
        non_finite = round((value('INVALID_FRACTION') or 0) * rows)
        count = rows - (value('NUM_NA') or 0) - non_finite
        return {
            'count': count,
            'mean': value('MEAN') if count else None,
            'min': value('MIN') if count else None,
            'max': value('MAX') if count else None,
            'percentiles': value('PERCENTILES', True) or [],
        }
    return {
        'count': (value('COUNT') or 0) - (value('NUM_NA') or 0),
        'distinct': value('NUM_UNIQUE'),
        'distinct_estimated': bool(value('CATEGORY_COUNTS_CAPPED')),
        'top': value('CATEGORY_COUNTS') or [],
    }
