# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Score counts of a prediction run's output files."""

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from kumo_core.score_counts import Counts
from kumo_core.types import TaskType

_SCORES = {
    TaskType.BINARY_CLASSIFICATION: ('True_PROB', 'True_PROB', 'score', False),
    TaskType.REGRESSION: ('TARGET_PRED', 'TARGET_PRED', 'target', False),
    TaskType.FORECASTING: ('TARGET_PRED', 'TARGET_PRED', 'target', False),
    TaskType.MULTICLASS_CLASSIFICATION: ('SCORE', 'confidence', 'score', True),
    TaskType.MULTILABEL_CLASSIFICATION: ('SCORE', 'top_score', 'top_score',
                                         True),
    TaskType.MULTILABEL_RANKING: ('SCORE', 'top_score', 'top_score', True),
    TaskType.STATIC_LINK_PREDICTION: ('SCORE', 'top_score', 'top_score', True),
    TaskType.TEMPORAL_LINK_PREDICTION: ('SCORE', 'top_score', 'top_score',
                                        True),
}
_CLASS_COLUMNS = {'CLASS', 'SCORE', 'LABEL', 'PREDICTED'}

Paths = Sequence[Union[str, Path]]


def output_counts(paths: Paths, task_type: TaskType,
                  edges: Optional[Mapping[str, Tuple[float, ...]]]
                  ) -> Optional[Counts]:
    r"""Score counts of prediction files; None without bins."""
    import diskgraph
    import pyarrow as pa
    from pyarrow import parquet

    column, name, axis, top = _SCORES.get(task_type, (None,) * 4)
    if not paths or axis is None or axis not in (edges or {}):
        return None
    schema = parquet.read_schema(paths[0])
    specs = [{'name': column, 'stype': 'numerical',
              'dtype': 'float32' if pa.types.is_float32(
                  schema.field(column).type) else 'float64'}]
    binary = task_type is TaskType.BINARY_CLASSIFICATION
    if binary:
        specs.append({'name': 'TARGET_PRED', 'stype': 'categorical',
                      'dtype': 'bool'})
    full, _ = diskgraph.compute_col_stats_for_parquet(
        [str(path) for path in paths], json.dumps(specs), '',
        percentiles=False, summary_stats_by_class=False, histograms=False,
        max_per=[key for key in schema.names
                 if key not in _CLASS_COLUMNS and not key.startswith('__')]
        if top else [],
        fixed_edges={column: list(edges[axis])})
    stats = json.loads(full)
    fixed = stats[f'{column}__kumo.numerical__FIXED_HISTOGRAM__False']['value']
    counts = Counts({axis: tuple(edges[axis])}, {
        name: np.asarray(fixed['counts']),
        f'{name}/sum': np.asarray(float(fixed['sum'])),
    })
    if fixed['min'] is not None:
        counts.extents[name] = (fixed['min'], fixed['max'])
    if binary:
        counts.total('predicted_positive', _positives(stats))
    return counts


def _positives(stats: Dict[str, Any]) -> int:
    entry = stats.get('TARGET_PRED__kumo.categorical__CATEGORY_COUNTS__False')
    return sum(count for value, count in (entry or {}).get('value') or []
               if value in (True, 1))
