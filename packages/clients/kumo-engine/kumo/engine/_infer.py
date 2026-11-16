# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Table schema inference: dtype/stype per column, from source metadata and,
where the dtype alone doesn't settle it, a sample of the actual data.

Ported from `kumoai/backend/local/table.py`'s `_LocalTableBackend.
infer_metadata`/`_stype_for_column`/`_sample`/`_statistics`/
`_column_statistics` in the cloud SDK. Sample decoding and statistics now
have one implementation in `kumo_core`; this adapter calls
`connector.source_backend` directly (no `Backend`/connector-id-lookup
indirection -- `Table.connector` already *is* the live connector) and threads a
Custom SQL table's query to the reader as `sql=` (the reader applies the
sample gating).

Two entry points, split the same way the cloud SDK splits `SourceTable.
column_dict` (source dtype + a pure dtype->stype rule)
from `Table.infer_metadata()` (samples the source to resolve whatever the
cheap pass left ambiguous):

* `infer_source_columns` -- source-schema lookup. Filesystem CSV sources
  read rows to infer dtypes; this is not necessarily a catalog-only call.
  Backs `SourceTable.column_dict` (`kumo.engine.connector.source_table`).
  Returns `kumoapi.
  source_table.SourceColumn`, which -- unlike `kumoapi.table.Column` --
  allows `stype=None`, exactly the "not yet resolved" state a cheap catalog
  read can leave a numeric/string column in.
* `resolve_table_columns` -- takes that cheap mapping and finishes the job:
  primary-key/time-column stype override, then a sampled read
  (`kumoai/backend/local/table.py`'s `_sample`) for anything still
  ambiguous. Backs `Table.from_source_table`, which is where this package's
  cheap/expensive split necessarily ends -- `kumoapi.table.TableDefinition.
  cols` is a `List[Column]`, and `kumoapi.table.Column` (unlike the cloud
  SDK's own mutable wrapper) requires `stype`/`dtype` set, so a `Table`
  cannot hold a partially-resolved schema the way a cloud-SDK `Table` can
  between `from_source_table` and a later `infer_metadata()` call. All the
  sampling this package ever needs, for a given `Table`, therefore has to
  happen inside `from_source_table` itself, not deferred to a later no-arg
  call.
"""

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

from kumo_core.stype_inference import infer_stype_from_name
from kumoapi.plugin import SourceBackend
from kumoapi.source_table import SourceColumn
from kumoapi.table import Column
from kumoapi.typing import Dtype, Stype

from kumo.engine.connector.base import Connector

#: Rows sampled for stats-based stype inference, matching the cloud SDK's
#: `kumoai/backend/local/_common.py`'s `_SAMPLE_ROWS`.
_SAMPLE_ROWS = 10_000


def _stype_from_dtype(dtype: Dtype) -> Optional[Stype]:
    r"""Returns the stype a data type settles on its own, if any -- `None`
    for the numeric and string families, which need column statistics to
    separate (`_stype_from_stats`). Ported from
    `kumoai/backend/local/table.py`'s `_stype_from_dtype`.
    """
    from kumo_core.stype_inference import infer_stype_from_dtype

    return infer_stype_from_dtype(dtype)


def _stype_from_stats(dtype: Dtype, num_unique: int, unique_fraction: float,
                      num_unique_multi: Optional[int] = None,
                      multicategorical_proven: bool = False) -> Optional[Stype]:
    r"""Returns the stype a column's statistics imply. Ported from
    `kumoai/backend/local/table.py`'s `_stype_from_stats`.
    """
    from kumo_core.stype_inference import infer_stype_from_stats

    return infer_stype_from_stats(dtype, num_unique, unique_fraction,
                                  num_unique_multi, multicategorical_proven)


def _stype_for_column(
    dtype: Optional[Dtype],
    stats: Optional[Tuple[int, float]],
    num_unique_multi: Optional[int] = None,
    multicategorical_proven: bool = False,
    *,
    name: str = "",
) -> Optional[Stype]:
    r"""Returns the stype implied by one column's dtype, falling back to its
    sampled statistics. Identifier and numerical names use the shared rules;
    explicit primary-key/time-column roles are applied in
    `resolve_table_columns`.
    Ported from `kumoai/backend/local/table.py`'s `_stype_for_column`.
    """
    if dtype is None:
        return None
    implied = _stype_from_dtype(dtype) or infer_stype_from_name(name, dtype)
    if implied is not None:
        return implied
    if stats is None:
        return None
    return _stype_from_stats(dtype, *stats, num_unique_multi,
                             multicategorical_proven)


def _column_statistics(frame) -> Dict[str, Tuple[int, float]]:
    from kumo_core.stype_inference import column_statistics

    return column_statistics(frame)


def _sample_frame(sample_rows: Optional[str]):
    from kumo_core.sources.preview import sample_dataframe

    return sample_dataframe(sample_rows)


def _sample(
    source_backend: SourceBackend,
    connector: Connector,
    table_name: str,
    sql: Optional[str] = None,
) -> Tuple[Dict[str, Dtype], Dict[str, Tuple[int, float]], Any, bool]:
    r"""Reads a sample and returns per-column dtypes, statistics, the frame, and
    whether the source itself came back with zero rows. Any failure to
    reach the source returns empty mappings rather than raising --
    statistics are an enhancement: a column whose stype follows from its
    dtype must still resolve with no read at all, and only a column that
    genuinely needed the data should fail (the caller's own error then names
    it). Ported from `kumoai/backend/local/table.py`'s `_sample`/
    `_statistics`.
    """
    try:
        response = source_backend.get_sample(connector, table_name,
                                             _SAMPLE_ROWS, sql=sql)
    except Exception:
        return {}, {}, None, False
    dtypes = {col.name: col.dtype for col in response.cols}
    frame = _sample_frame(response.sample_rows)
    is_empty = frame is not None and len(frame) == 0
    return dtypes, ({} if frame is None else _column_statistics(frame)), \
        frame, is_empty


def infer_source_columns(
    connector: Connector,
    table_name: str,
    sql: Optional[str] = None,
) -> Dict[str, SourceColumn]:
    r"""Schema read for `table_name` via `SourceBackend.get_columns`.
    See the module docstring for source-dependent I/O. Infers stype
    from the dtype where that settles it on its own. Backs
    `SourceTable.column_dict` (`kumo.engine.connector.source_table`).

    A column left ambiguous by dtype alone (most numeric/string columns)
    keeps `stype=None` here -- `resolve_table_columns` is what samples to
    settle those.

    Raises:
        ValueError: if the source reports no columns.
    """
    response = connector.source_backend.get_columns(connector, table_name,
                                                    sql=sql)
    if not response.cols:
        raise ValueError(f"Table {table_name!r} has no columns.")
    return {
        col.name:
        SourceColumn(
            name=col.name,
            stype=col.stype or _stype_from_dtype(col.dtype),
            dtype=col.dtype,
            is_primary=col.is_primary,
            multicategorical_delimiter=col.multicategorical_delimiter,
        )
        for col in response.cols
    }


def refuse_unsupported_columns(connector: Connector, table_name: str,
                               columns: List[Column]) -> None:
    r"""Raises for columns whose source type has no Kumo dtype and that no
    declared dtype resolved.
    """
    from kumo_core.sources.object_store_backend import ObjectStoreSourceBackend

    names = [col.name for col in columns if col.dtype == Dtype.unsupported]
    if not names:
        return
    backend = connector.source_backend
    types = (backend.source_types(connector, table_name) if isinstance(
        backend, ObjectStoreSourceBackend) else {})
    listed = ', '.join(
        f"{name!r} ({types[name]})" if name in types else repr(name)
        for name in names)
    raise ValueError(
        f"Table {table_name!r} has columns whose source type Kumo cannot "
        f"read: {listed}. Declare a dtype for them (`dtypes=` or "
        f"`with_column(name, dtype=...)`), leave them out of `column_names`, "
        f"or cast them at the source.")


def resolve_table_columns(
    connector: Connector,
    table_name: str,
    columns: Dict[str, SourceColumn],
    *,
    column_names: Optional[List[str]],
    primary_key: Optional[str],
    time_column: Optional[str],
    end_time_column: Optional[str],
    sql: Optional[str] = None,
    dtypes: Optional[Mapping[str, Dtype]] = None,
) -> List[Column]:
    r"""Finishes what `infer_source_columns`'s cheap pass left open: applies
    the primary-key/time-column stype override, then -- only if something is
    still ambiguous -- samples the source (`_sample`) to resolve it. Backs
    `Table.from_source_table`.

    Args:
        connector: read through `connector.source_backend` if sampling is
            needed.
        table_name: the provider-native table name.
        columns: `SourceTable.column_dict`'s cheap-pass result (or an
            equivalent mapping) -- name to (possibly stype-ambiguous)
            `SourceColumn`.
        column_names: restrict to these columns. `None` keeps every column
            in `columns`.
        primary_key: names the primary-key column. Not auto-detected --
            reasserts `Stype.ID` on it regardless of what the cheap/sampled
            pass implied.
        time_column: names the time column, similarly reasserted as
            `Stype.timestamp`.
        end_time_column: names the end-time column, same as `time_column`.
        sql: a Custom SQL table's query, sampled instead of `table_name`.
        dtypes: declared dtypes that replace the source's, including for
            columns whose source type has no Kumo dtype.

    Raises:
        ValueError: if `column_names` matches nothing, or a selected
            column's dtype/stype still can't be resolved after sampling.
        PermanentSourceError: if the source sample came back with zero
            rows and a column is still ambiguous -- no amount of sampling
            resolves that, and DiskGraph can't ingest an empty table
            anyway, so this names the real constraint instead of the
            generic "cannot be resolved" guidance above.
    """
    selected = [
        col for name, col in columns.items()
        if column_names is None or name in column_names
    ]
    if not selected:
        raise ValueError(f"Table {table_name!r} has no columns to resolve "
                         f"(column_names={column_names!r} matched none of "
                         f"{sorted(columns)!r}).")
    unknown = sorted(set(dtypes or {}) - {col.name for col in selected})
    if unknown:
        raise ValueError(f"Table {table_name!r} has no selected column "
                         f"{', '.join(map(repr, unknown))} to declare a "
                         f"dtype for.")
    selected = [
        replace(col, dtype=dtypes[col.name], stype=None,
                multicategorical_delimiter=None)
        if dtypes and col.name in dtypes else col for col in selected
    ]

    resolved: Dict[str, Tuple[Optional[Dtype], Optional[Stype]]] = {
        col.name:
        (col.dtype, Stype.ID if col.name == primary_key else
         Stype.timestamp if col.name in (time_column, end_time_column) else
         col.stype or _stype_for_column(col.dtype, None, name=col.name))
        for col in selected
    }
    delimiters = {col.name: col.multicategorical_delimiter for col in selected}
    if any(stype is None or (stype == Stype.multicategorical and dtype ==
                             Dtype.string and delimiters[name] is None)
           for name, (dtype, stype) in resolved.items()):
        source_backend = connector.source_backend
        sampled, statistics, frame, is_empty = _sample(source_backend,
                                                       connector, table_name,
                                                       sql)
        for col in selected:
            dtype, stype = resolved[col.name]
            dtype = dtype or sampled.get(col.name)
            num_unique_multi = None
            delimiter = delimiters[col.name]
            if (dtype == Dtype.string and frame is not None
                    and col.name in frame
                    and stype in (None, Stype.multicategorical)):
                from kumo_core.stype_inference import (
                    multicategorical_separator,
                    multicategorical_token_counts,
                )

                values = frame[col.name].dropna().astype('string')
                delimiter = delimiter or multicategorical_separator(values)
                if delimiter is not None:
                    num_unique_multi = len(
                        multicategorical_token_counts(values, delimiter))
            if stype is None:
                stype = _stype_for_column(dtype, statistics.get(col.name),
                                          num_unique_multi, delimiter
                                          is not None, name=col.name)
            if stype == Stype.multicategorical and dtype == Dtype.string:
                delimiters[col.name] = delimiter
            resolved[col.name] = (dtype, stype)
        if is_empty and any(stype is None for _, stype in resolved.values()):
            from kumo_core.sources.errors import PermanentSourceError
            raise PermanentSourceError(
                f"Table {table_name!r} returned no rows. An empty source "
                f"table cannot be ingested; DiskGraph needs at least one "
                f"row to infer an encoding.")

    result: List[Column] = []
    for col in selected:
        dtype, stype = resolved[col.name]
        if dtype is None or stype is None:
            raise ValueError(
                f"Column {col.name!r} in table {table_name!r} cannot be "
                "resolved: it has "
                f"{'no dtype' if dtype is None else f'dtype {dtype}'} and "
                f"no stype, and the source returned no statistics to infer "
                f"one from. Check the column exists and is reachable, or "
                f"supply its stype explicitly (e.g. via `Table.with_column`"
                f" after construction).")
        result.append(
            Column(
                name=col.name,
                stype=stype,
                dtype=dtype,
                multicategorical_delimiter=delimiters[col.name],
            ))
    return result
