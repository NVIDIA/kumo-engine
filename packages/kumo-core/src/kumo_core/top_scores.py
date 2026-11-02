# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Holdout top scores, spilled to Parquet and counted by DiskGraph."""

import json
from typing import Any, Dict, Optional

import numpy as np
from kumo_core.score_counts import Counts

COLUMN = 'top_score'
_PERCENTILES = np.linspace(0, 1, 101).tolist()


class TopScoreSpill:
    r"""Writes each batch's entities and top scores to one Parquet file."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._writer: Any = None

    def write(self, entity: Any, score: Any) -> None:
        import pyarrow as pa
        from pyarrow import parquet

        table = pa.table({'ENTITY': np.asarray(entity, dtype=np.int64),
                          COLUMN: np.asarray(score)})
        if self._writer is None:
            self._writer = parquet.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)

    def close(self) -> Optional[str]:
        r"""The file, or None when nothing was written."""
        if self._writer is None:
            return None
        self._writer.close()
        return self.path


def top_score_counts(path: Optional[str]) -> Counts:
    r"""Top scores on bins cut at their own exact percentiles."""
    values = () if path is None else _stats(
        path, quantiles={COLUMN: _PERCENTILES})['QUANTILES']['values']
    edges = tuple(np.unique(values).tolist())
    if not edges:
        return Counts({COLUMN: ()}, {COLUMN: np.zeros(1, dtype=np.int64)})
    fixed = _stats(path, fixed_edges={COLUMN: list(edges)})['FIXED_HISTOGRAM']
    return Counts({COLUMN: edges}, {COLUMN: np.asarray(fixed['counts'])},
                  {COLUMN: (fixed['min'], fixed['max'])})


def _stats(path: Any, **options: Any) -> Dict[str, Any]:
    import diskgraph
    from pyarrow import parquet

    dtype = str(parquet.read_schema(path).field(COLUMN).type)
    full, _ = diskgraph.compute_col_stats_for_parquet(
        [str(path)], json.dumps([{'name': COLUMN, 'stype': 'numerical',
                                  'dtype': 'float32' if dtype == 'float'
                                  else 'float64'}]), '',
        summary_stats_by_class=False, histograms=False, **options)
    prefix = f'{COLUMN}__kumo.numerical__'
    return {key.removeprefix(prefix).removesuffix('__False'): entry['value']
            for key, entry in json.loads(full).items()
            if key.startswith(prefix)}
