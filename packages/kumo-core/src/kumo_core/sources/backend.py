# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Iterable, Mapping, Sequence

from kumoapi.plugin import PrepareIngestResult
from kumoapi.source_table import SourceColumn, SourceTableDataResponse
from kumoapi.typing import Dtype

from .errors import SourceNotFoundError
from .types import StagedTable


def select_columns(
        table: str, catalog: Mapping[str, Dtype],
        dtypes: Mapping[str, Dtype] | None = None) -> dict[str, Dtype]:
    declared = dict(dtypes or {})
    missing = sorted(set(declared) - set(catalog))
    if missing:
        raise SourceNotFoundError(
            f'Table {table} has no column {", ".join(missing)}. The graph '
            f'declares it, so it cannot be read; correct the table or drop '
            f'the column from the graph.')
    return {
        column: declared.get(column, dtype)
        for column, dtype in catalog.items()
        if not declared or column in declared
    }


def ingest_result(staged: StagedTable) -> PrepareIngestResult:
    return PrepareIngestResult(
        source={
            'type': 'parquet',
            'paths': list(staged.paths)
        },
        source_size=staged.source_size,
        version=staged.version,
        mode_used=staged.mode_used.value,
    )


def source_columns(columns: Iterable[tuple[str, Dtype]]) -> list[SourceColumn]:
    from kumo_core.stype_inference import infer_stype_from_dtype

    return [
        SourceColumn(name=name, dtype=dtype,
                     stype=infer_stype_from_dtype(dtype), is_primary=False)
        for name, dtype in columns
    ]


def sampled_table(table_name: str, columns: Sequence[SourceColumn],
                  rows: Iterable[Any]) -> SourceTableDataResponse:
    import pandas as pd

    frame = pd.DataFrame(list(rows),
                         columns=pd.Index(column.name for column in columns))
    return SourceTableDataResponse(table_name=table_name, cols=list(columns),
                                   sample_rows=frame.to_json(orient='table'))
