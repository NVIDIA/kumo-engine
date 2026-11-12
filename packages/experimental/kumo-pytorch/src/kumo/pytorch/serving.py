# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble serving batches from rows the caller supplies.

Serving here means: load the artifact once, then on each call take the rows the
caller fetched for one or more requests and assemble them, with the requests,
into the batch format the model was trained on. Fetching, looking up or sampling
the rows is the caller's job: serving queries no engine or data store. Whatever
the caller supplies is the whole dataset for that call, read with the in-memory
engine's semantics under the artifact's fitted state, in inference mode. A
standard forward pass over the batch answers the requests.

`ServingAssembler` does everything that depends only on the artifact when it is
built: it checks the contract, rebuilds the fitted plan, and derives
`ServingAssembler.rows_spec`, which says which rows to fetch. Each call to
`ServingAssembler.assemble` only checks the supplied rows, binds them and the
requests, and assembles.

The batches equal what batch inference produces for the same requests over the
same rows. Rows the samples would not read can be left out, and so can tables
and columns inference never reads. A row they would read and the caller leaves
out is absent, as from any dataset: the batch differs and nothing is refused,
because serving cannot tell it from a row that does not exist. Fetching what
``rows_spec`` lists avoids that.

`ServingRowsSpec` and the types it is made of (`TableRows`, `TableRoute`,
`RowsJoin`, `RowLimit`, `ColumnRows`) are public here, for serving code that
drives and checks its own row lookup from the spec.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Mapping, Sequence

import pyarrow as pa

from kumo.pytorch.artifact import Artifact
from kumo.pytorch.assembly import _fingerprint, fit_plan
from kumo.pytorch.contract.batch import Batch
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import SampleRequest
from kumo.pytorch.data.records import Scalar
from kumo.pytorch.dataset import Dataset, InMemoryRows
from kumo.pytorch.engines.in_memory import InMemoryEngine
from kumo.pytorch.engines.in_memory.source import InMemorySource
from kumo.pytorch.preparation import _check, reused_state
from kumo.pytorch.serving_rows import (
    ColumnRole,
    ColumnRows,
    RowLimit,
    RowsJoin,
    ServingRowsSpec,
    TableRoute,
    TableRows,
    dtype_text,
)
from kumo.pytorch.spec import DataLoaderSpec, ValidationError
from kumo.pytorch.task import request_batches

__all__ = [
    "ColumnRole",
    "ColumnRows",
    "RowLimit",
    "RowsJoin",
    "ServingAssembler",
    "ServingRows",
    "ServingRowsSpec",
    "ServingValue",
    "TableRoute",
    "TableRows",
]

ServingValue = Scalar | None
"""One supplied cell: a value of the column's declared type, or ``None``."""

ServingRows = Mapping[str, Sequence[Mapping[str, ServingValue]] | pa.Table]
"""Supplied rows by table name: row mappings or a ``pyarrow.Table`` per table."""

_SEE_SPEC = "ServingAssembler.rows_spec lists what to supply"


class ServingAssembler:
    """Inference batches for one artifact, from rows the caller fetches and passes in.

    Build one at service startup and reuse it for every request::

        server = ServingAssembler(artifact)
        print(server.rows_spec)  # which rows to fetch per request
        batches = server.assemble(rows, requests)  # rows the caller fetched; 1+ requests

    Construction checks the contract against the in-memory engine, checks that
    the fitted state is complete and belongs to this contract, rebuilds the
    fitted plan in inference mode, and derives `rows_spec`. Nothing there
    depends on rows, so nothing is repeated per call.

    :param artifact: an artifact written by `Artifact.save` and read by
        `Artifact.load`, or built by `Task.to_artifact`.
    """

    def __init__(self, artifact: Artifact) -> None:
        engine = InMemoryEngine()
        empty = Dataset({table.name: InMemoryRows(()) for table in artifact.contract.schema.tables})
        contract, issues = _check(artifact.contract, empty, engine)
        if issues:
            raise ValidationError(issues)
        fitted = artifact.fitted_state
        if not fitted.has_targets:
            raise ValueError(
                "serving needs the artifact's complete fitted state, not fitted_state.inputs; "
                "load the training task's artifact with Artifact.load(directory)"
            )
        input_state, target_encoders = reused_state(contract, fitted, None)
        self.contract: DataLoaderSpec = contract
        self.rows_spec: ServingRowsSpec = ServingRowsSpec.from_contract(contract)
        self._plan = fit_plan(
            contract.sample,
            contract.batch,
            (),
            None,
            schema_fingerprint=_fingerprint(contract.schema),
            contract_fingerprint=contract.fingerprint,
            contract=contract,
            input_state=input_state,
            target_encoders=target_encoders,
            inference=True,
        )
        self._declared: dict[str, dict[str, DType]] = {
            table.name: dict(table.columns or {}) for table in contract.schema.tables
        }
        self._keys = {table.name: table.primary_key for table in contract.schema.tables}

    def assemble(
        self,
        rows: ServingRows,
        requests: Sequence[SampleRequest],
        *,
        batch_size: int | None = None,
        seed: int = 0,
    ) -> list[Batch]:
        """Inference batches for ``requests``, built only from the supplied ``rows``.

        Rows are refused, naming the table, row and column, for: a table the
        schema does not declare, a required table left out, a column the table
        does not declare, a required column left out, a value that is not of
        its column's declared type, a NULL or repeated primary key, and a
        request whose entity (or candidate) has no row. A table or column
        ``rows_spec`` marks optional may be left out; it reads as empty or
        ``None``, which inference never reads.

        :param rows: table name to its rows for these requests, as row mappings
            or an Arrow table.
        :param batch_size: samples per batch; defaults to all requests in one, which
            a token budget in the contract can still split.
        :param seed: the seed batch inference used, for seeded graph sampling.
        """

        if not requests:
            raise ValueError("serving needs at least one request")
        tables = self._tables(rows)
        self._require_entities(tables, requests)
        source = InMemorySource(self.contract.schema, tables)
        try:
            return list(
                request_batches(
                    self._plan.with_source(source),
                    tuple(requests),
                    batch_size=len(requests) if batch_size is None else batch_size,
                    epoch=0,
                    seed=seed,
                    shuffle=False,
                    rank=0,
                    world_size=1,
                    drop_last=False,
                )
            )
        finally:
            source.close()

    def _tables(self, rows: ServingRows) -> dict[str, list[dict[str, ServingValue]]]:
        unknown = sorted(set(rows) - set(self._declared))
        if unknown:
            raise ValueError(
                f"rows supplied for {unknown}, which the artifact's schema does not declare"
                f"{_suggest(unknown[0], self._declared)}; {_SEE_SPEC}"
            )
        missing = [name for name in self.rows_spec.required_tables if name not in rows]
        if missing:
            raise ValueError(
                f"no rows supplied for required tables {missing}; pass an empty list when a "
                f"request has none; {_SEE_SPEC}"
            )
        return {
            name: self._table(name, rows[name]) if name in rows else [] for name in self._declared
        }

    def _table(
        self, name: str, supplied: Sequence[Mapping[str, ServingValue]] | pa.Table
    ) -> list[dict[str, ServingValue]]:
        declared = self._declared[name]
        if isinstance(supplied, pa.Table):
            self._check_columns(name, None, supplied.column_names)
            records = self._arrow_rows(name, supplied)
        elif isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
            records = supplied
        else:
            raise TypeError(
                f"rows[{name!r}] must be a sequence of row mappings or a pyarrow.Table, "
                f"not {type(supplied).__name__}"
            )
        key = self._keys[name]
        seen: dict[ServingValue, int] = {}
        result: list[dict[str, ServingValue]] = []
        for index, row in enumerate(records):
            if not isinstance(row, Mapping):
                raise TypeError(f"rows[{name!r}][{index}] must be a mapping of column to value")
            if not isinstance(supplied, pa.Table):
                self._check_columns(name, index, row.keys())
            values: dict[str, ServingValue] = {}
            for column, dtype in declared.items():
                value = row.get(column)
                try:
                    values[column] = dtype.normalize(value)
                except ValueError:
                    raise ValueError(
                        f"rows[{name!r}][{index}][{column!r}]: {value!r} is not a valid "
                        f"{dtype_text(dtype)} value, the column's declared type; {_SEE_SPEC}"
                    ) from None
            identity = values[key]
            if identity is None:
                raise ValueError(
                    f"rows[{name!r}][{index}] has a NULL primary key {key!r}; {_SEE_SPEC}"
                )
            if identity in seen:
                raise ValueError(
                    f"rows[{name!r}][{index}] repeats primary key {key}={identity!r} of "
                    f"rows[{name!r}][{seen[identity]}]; supply each row once; {_SEE_SPEC}"
                )
            seen[identity] = index
            result.append(values)
        return result

    def _check_columns(self, name: str, index: int | None, columns: Iterable[str]) -> None:
        declared = self._declared[name]
        present = list(columns)
        where = f"rows[{name!r}]" if index is None else f"rows[{name!r}][{index}]"
        unknown = [column for column in present if column not in declared]
        if unknown:
            raise ValueError(
                f"{where} has column {unknown[0]!r}, which table {name!r} does not declare"
                f"{_suggest(unknown[0], declared)}; {_SEE_SPEC}"
            )
        table = self.rows_spec.tables[name]
        absent = [
            column
            for column, spec in table.columns.items()
            if spec.required and column not in present
        ]
        if absent:
            raise ValueError(
                f"{where} lacks required columns {absent}; pass None for a missing value; "
                f"{_SEE_SPEC}"
            )

    def _arrow_rows(self, name: str, table: pa.Table) -> list[dict[str, ServingValue]]:
        declared = self._declared[name]
        columns = {}
        for field in table.schema:
            dtype = declared[field.name]
            column = table.column(field.name)
            if pa.types.is_null(field.type):
                column = pa.nulls(len(column), dtype.to_arrow())
            elif not _same_type(_stored(field.type), dtype) and not _widens(
                _stored(field.type), dtype
            ):
                raise ValueError(
                    f"rows[{name!r}] column {field.name!r} is Arrow {field.type}, not "
                    f"the declared {dtype_text(dtype)}; {_SEE_SPEC}"
                )
            columns[field.name] = column
        return pa.table(columns).to_pylist() if columns else [{} for _ in range(len(table))]

    def _require_entities(
        self,
        tables: Mapping[str, list[dict[str, ServingValue]]],
        requests: Sequence[SampleRequest],
    ) -> None:
        sample = self.contract.sample
        present = {name: {row[self._keys[name]] for row in rows} for name, rows in tables.items()}

        def require(table: str, entity_id: object, what: str) -> None:
            key = self._keys[table]
            dtype = self._declared[table][key]
            try:
                identity = dtype.normalize(entity_id)
            except ValueError:
                raise ValueError(
                    f"{what} {entity_id!r} is not a valid {dtype_text(dtype)} {table}.{key}"
                ) from None
            if identity not in present[table]:
                raise ValueError(
                    f"{what} {entity_id!r} has no row in rows[{table!r}] ({key} = "
                    f"{entity_id!r}); {_SEE_SPEC}"
                )

        for request in requests:
            require(
                sample.entity.table,
                request.entity_id,
                f"request {request.sample_id!r}: entity_id",
            )
            for group, candidates in request.candidates.items():
                spec = sample.candidate_sets.get(group)
                if spec is None:
                    continue
                for candidate in candidates:
                    require(
                        spec.entity.table,
                        candidate.entity_id,
                        f"request {request.sample_id!r}: candidate set {group!r} entity_id",
                    )


def _suggest(name: str, options: Iterable[str]) -> str:
    close = difflib.get_close_matches(name, list(options), n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _same_type(arrow_type: pa.DataType, dtype: DType) -> bool:
    try:
        return DType.from_arrow(arrow_type) == dtype
    except ValueError:
        return False


def _stored(arrow_type: pa.DataType) -> pa.DataType:
    return arrow_type.value_type if pa.types.is_dictionary(arrow_type) else arrow_type


def _widens(arrow_type: pa.DataType, dtype: DType) -> bool:
    """Values the row path accepts too: integers for integers and seconds, numbers for floats."""

    if dtype.integer or dtype.name == "timestamp":
        return pa.types.is_integer(arrow_type)
    if dtype.name in {"float32", "float64"}:
        return pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type)
    return False
