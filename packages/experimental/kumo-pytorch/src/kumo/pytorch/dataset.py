# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a schema's tables physically live.

`DataSchema` says what tables *mean* — their columns, keys, relations and
clocks. This module says where their bytes *are*. The two are deliberately
separate, and the separation is load-bearing rather than tidy:

* The contract says nothing about storage; the dataset binds the source kinds
  the engine reads (`Parquet` for the DiskGraph engine, `InMemoryRows` for the
  in-memory engine).
* Physical layout changes far more often than meaning does. A table moving from
  local Parquet to object storage is not a change to the contract.

A `Dataset` names no engine and, apart from in-memory rows, is serializable.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from glob import glob, has_magic
from pathlib import Path
from typing import Any

from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.schema import DataSchema

__all__ = [
    "Dataset",
    "InMemoryRows",
    "Parquet",
    "TableSource",
]


class _MissingParquetColumns(ValueError):
    """Missing logical columns in one physical file."""

    def __init__(self, path: str, columns: list[str]) -> None:
        self.path = path
        self.columns = columns
        super().__init__(f"in {path!r}: missing columns {columns}")


@dataclass(frozen=True)
class Parquet:
    """One logical table stored as Parquet.

    ``path`` may name a file, a directory, or a glob. ``columns`` renames
    physical columns to the logical names the schema uses, for the common case
    where a warehouse column is not called what the model calls it.

    :param path: a file, a directory, or a glob.
    :param columns: physical column name to the logical name the schema uses.
    """

    kind: str = field(default="parquet", init=False)
    path: str = ""
    columns: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("a Parquet source requires a path")
        renames = {str(source): str(target) for source, target in self.columns.items()}
        if len(set(renames.values())) != len(renames):
            raise ValueError(f"Parquet column renames collide: {sorted(renames.values())}")
        object.__setattr__(self, "columns", renames)

    def _bind_columns(
        self,
        selected: Sequence[str] | None = None,
        *,
        file_columns: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, str]:
        """Resolve logical names to physical fields consistently across every file.

        With no selection, discover all logical columns. Renames apply simultaneously;
        ambiguity is refused only for selected columns. Physical field names must be unique.
        """
        import pyarrow.parquet as pq

        binding = None
        for path in _local_parquet_files(self.path) if file_columns is None else file_columns:
            try:
                names = pq.read_schema(path).names if file_columns is None else file_columns[path]
                duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
                if duplicates:
                    raise ValueError(f"duplicate physical column names {duplicates}")
                candidates: dict[str, list[str]] = {}
                for name in names:
                    candidates.setdefault(self.columns.get(name, name), []).append(name)
                if selected is None and binding is not None and set(binding) != set(candidates):
                    raise ValueError(
                        "column names differ across Parquet files; "
                        "select shared columns explicitly with TableSpec.columns"
                    )
                missing = sorted(set(selected) - candidates.keys()) if selected is not None else []
                if missing:
                    raise _MissingParquetColumns(path, missing)
                current = {}
                for logical in candidates if selected is None else selected:
                    physical = candidates.get(logical, [])
                    if len(physical) != 1:
                        raise ValueError(
                            f"ambiguous logical column {logical!r}: physical columns {physical}"
                        )
                    current[logical] = physical[0]
                if binding is not None and current != binding:
                    raise ValueError(
                        f"column binding differs across Parquet files: {current!r} != {binding!r}"
                    )
                binding = current
            except _MissingParquetColumns:
                raise
            except (ValueError, OSError) as error:
                raise ValueError(f"in {path!r}: {error}") from error
        assert binding is not None
        return binding

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form of this storage reference."""
        return {"kind": self.kind, "path": self.path, "columns": dict(self.columns)}


@dataclass(frozen=True)
class InMemoryRows:
    """One logical table held as Python mappings.

    The only source kind that is **not** serializable: the rows are the data.
    Useful for tests, examples and small runs; not a storage format.

    :param rows: the rows themselves, as mappings of column name to value.
    """

    kind: str = field(default="in_memory", init=False)
    rows: Sequence[Mapping[str, Any]] = ()

    def to_dict(self) -> dict[str, Any]:
        """Always refused: in-memory rows are the data, not a reference to it."""
        raise TypeError(
            "in-memory rows cannot be serialized; they are the data, not a reference to it"
        )


TableSource = Parquet | InMemoryRows


@dataclass(frozen=True)
class Dataset:
    """Physical binding for every table a schema declares.

    Validation against the schema happens in `validate_against`, not here, so a
    dataset can be constructed and inspected without a contract in hand.

    A dataset must bind exactly the contract's tables; stored columns are
    checked against their declared dtypes. ``prepare`` has no default engine.

    :param tables: logical table name, as the schema calls it, to where its
        bytes are. No engine is named here: ``DiskGraphEngine`` reads `Parquet`,
        and ``InMemoryEngine`` reads `InMemoryRows`.
    """

    tables: Mapping[str, TableSource]

    def __post_init__(self) -> None:
        if not self.tables:
            raise ValueError("a dataset must bind at least one table")
        bound = dict(self.tables)
        for name, source in bound.items():
            if not name:
                raise ValueError("dataset table names must be non-empty")
            if not isinstance(source, Parquet | InMemoryRows):
                raise TypeError(f"table {name!r} is bound to an unknown source {source!r}")
        object.__setattr__(self, "tables", bound)

    @property
    def kinds(self) -> frozenset[str]:
        """Source kinds present, which an engine checks before it reads them."""

        return frozenset(source.kind for source in self.tables.values())

    def missing_for(self, table_names: Sequence[str]) -> tuple[str, ...]:
        """The names in ``table_names`` this dataset does not bind."""
        return tuple(name for name in table_names if name not in self.tables)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form; refused when a table is `InMemoryRows`."""
        return {name: source.to_dict() for name, source in self.tables.items()}

    def resolve_schema(self, schema: DataSchema) -> DataSchema:
        """Fill omitted types from Parquet metadata, without reading rows."""
        import pyarrow.parquet as pq

        tables = []
        for table in schema.tables:
            if table.columns is not None and all(table.columns.values()):
                tables.append(table)
                continue
            source = self.tables.get(table.name)
            if not isinstance(source, Parquet):
                raise ValueError(
                    f"schema.tables.{table.name}.columns: type inference requires Parquet "
                    "metadata; declare every DType for InMemoryRows"
                )
            try:
                binding = source._bind_columns(
                    None if table.columns is None else tuple(table.columns)
                )
            except (ValueError, OSError) as error:
                raise ValueError(f"bind {table.name} {error}") from error
            columns = dict.fromkeys(binding) if table.columns is None else dict(table.columns)
            for path in _local_parquet_files(source.path):
                try:
                    metadata = pq.read_schema(path)
                    for name, declared in columns.items():
                        actual = DType.from_arrow(metadata.field(binding[name]).type)
                        if declared is not None and actual != declared:
                            raise ValueError(
                                f"declared {declared!r}, physical type resolves to {actual!r}"
                            )
                        columns[name] = actual
                except (KeyError, ValueError, OSError) as error:
                    raise ValueError(f"bind {table.name} in {path!r}: {error}") from error
            tables.append(replace(table, columns=columns))
        return replace(schema, tables=tuple(tables))


def _local_parquet_files(value: str) -> list[str]:
    if "://" in value:
        raise ValueError(f"local Parquet paths are required; got {value!r}")
    path = Path(value).expanduser()
    if has_magic(value):
        candidates = [Path(item) for item in glob(str(path), recursive=True)]
    elif path.is_dir():
        candidates = list(path.rglob("*.parquet"))
    else:
        candidates = [path]
    files = sorted(str(item.resolve()) for item in candidates if item.is_file())
    if not files:
        raise FileNotFoundError(f"no local Parquet files match {value!r}")
    return files
