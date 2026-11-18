# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A table's Parquet files: which they are, whether they agree, and how libcudf reads them."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kumo.pytorch.contract.dtypes import DType
    from kumo.pytorch.contract.schema import DataSchema
    from kumo.pytorch.dataset import Dataset, Parquet

__all__ = ['resolve_sources']


def split_form(dtype: DType) -> DType:
    """The type a split's file holds a `dtype` column in (`test.kumo_pytorch.reference_engine.gpu.split` writes them with libcudf, which
    writes no 256-bit decimal): a ``decimal256`` of precision up to 38 as ``decimal128`` of its precision and
    scale, a wider one as ``bytes`` (its 32 bytes a value, `_wide_decimal`); any other type as itself."""

    from kumo.pytorch.contract.dtypes import DType

    if dtype.name != "decimal256":
        return dtype
    if dtype.precision <= 38:
        return DType("decimal128", precision=dtype.precision, scale=dtype.scale)
    return DType("bytes")


def resolve_sources(schema: DataSchema, dataset: Dataset, *,
                    only: Sequence[str] | None = None) -> dict[str, tuple[Parquet, dict[str, DType]]]:
    """Each table's Parquet source and resolved column types, checked as loading checks them."""

    from kumo.pytorch.dataset import Parquet

    names = [table.name for table in schema.tables] if only is None else list(only)
    unknown = sorted(set(names) - {table.name for table in schema.tables})
    if unknown:
        raise ValueError(f"the schema declares no tables {unknown}")
    missing = dataset.missing_for(names)
    if missing:
        raise ValueError(f"the dataset binds no source for tables {sorted(missing)}")
    for name in names:
        source = dataset.tables[name]
        if not isinstance(source, Parquet):
            raise ValueError(
                f"test.kumo_pytorch.reference_engine.gpu reads Parquet sources only; table {name!r} is bound to "
                f"a {source.kind!r} source"
            )
    resolved = {table.name: table for table in dataset.resolve_schema(_only(schema, names)).tables}
    sources = {}
    for name in names:
        columns = dict(resolved[name].columns)
        sources[name] = (dataset.tables[name], columns)
    return sources


def _only(schema: DataSchema, names: Sequence[str]) -> DataSchema:
    """`schema` with just the tables `names` and the relations among them."""

    from dataclasses import replace

    keep = set(names)
    return replace(
        schema,
        tables=tuple(table for table in schema.tables if table.name in keep),
        relations=tuple(relation for relation in schema.relations
                        if relation.left_table in keep and relation.right_table in keep),
    )


def _physical_names(table: str, source: Parquet, logical: Sequence[str]) -> dict[str, str]:
    """Logical column name to the physical name read, through the source's renames.

    Kumo PyTorch's ``Parquet.columns`` maps a physical name to a logical one. A name
    renamed away is no longer readable under itself, and a rename onto a name
    that is also another physical column's own is refused as ambiguous.
    """

    renames = dict(source.columns)
    onto = {target: physical for physical, target in renames.items()}
    names = {}
    for name in logical:
        if name in onto:
            names[name] = onto[name]
        elif name in renames:
            raise ValueError(
                f"{table}.{name}: the source renames physical column {name!r} "
                f"to {renames[name]!r}, so no column is called {name!r}"
            )
        else:
            names[name] = name
    return names


def _check_files(table: str, files: Sequence[str], physical: Mapping[str, str],
                 columns: Mapping[str, DType], renames: Mapping[str, str], *, split: bool = False) -> None:
    """Each declared column, in every file, with its declared type (a split's own files: or the type a split
    stores it in, `split_form`). Metadata only."""

    import pyarrow.parquet as pq

    from kumo.pytorch.contract.dtypes import DType

    for path in files:
        try:
            fields = {field.name: field.type for field in pq.read_schema(path)}
            for name, dtype in columns.items():
                source_name = physical[name]
                if source_name not in fields:
                    raise ValueError(f"no column {source_name!r}")
                if source_name != name and name in fields and name not in renames:
                    raise ValueError(
                        f"column {source_name!r} is renamed to {name!r}, "
                        f"which is also a column of its own"
                    )
                actual = DType.from_arrow(fields[source_name])
                if actual != dtype and not (split and actual == split_form(dtype)):
                    raise ValueError(f"declared {dtype!r}, physical type resolves to {actual!r}")
        except (ValueError, OSError) as error:
            raise ValueError(f"bind {table} in {path!r}: {error}") from error


def _gds_present() -> bool:
    """Whether this machine can use GPUDirect Storage: its ``nvidia_fs`` kernel module is loaded.

    That module is what lets an SSD write straight into GPU memory. It exists on
    Linux only, so anywhere else this is False.
    """

    return Path("/sys/module/nvidia_fs").is_dir() or Path("/proc/driver/nvidia-fs").exists()


def _compat_mode() -> None:
    """On a machine without GPUDirect Storage, read through the host from the start.

    KvikIO, which libcudf reads files with, starts NVIDIA's cuFile driver on its
    first read in its default mode. Where GPUDirect Storage is present, that is
    the fast path: the SSD writes straight into GPU memory, so the default is
    kept. Where it is absent, starting the driver only adds about 0.8 s to the
    process's first read and writes a ``cufile.log`` into the working directory,
    before reading through the host anyway, so KvikIO is told to read through the
    host directly. Set for the process, before KvikIO's first use; left alone if
    any ``KVIKIO_`` variable is already set.
    """

    if any(name.startswith("KVIKIO_") for name in os.environ):
        return
    if not _gds_present():
        os.environ["KVIKIO_COMPAT_MODE"] = "ON"


def _reader(table: str, source: Parquet, columns: Mapping[str, DType], *,
            split: bool = False) -> tuple[Any, dict[str, str]]:
    """libcudf's reader options for `columns` of `source`, after checking its files (`split`: a split's own).

    Returns the options and each logical column's physical name.
    """

    from test.kumo_pytorch.reference_engine.gpu.data.memory import ensure_pool

    import pylibcudf as plc

    from kumo.pytorch.dataset import \
        _local_parquet_files  # Kumo PyTorch's own file listing

    ensure_pool()  # every table read starts here: GPU memory comes from one pool from now on
    files = _local_parquet_files(source.path)
    physical = _physical_names(table, source, list(columns))
    _check_files(table, files, physical, columns, source.columns, split=split)
    _compat_mode()
    options = (
        plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo(files))
        # No pandas index columns, no silently missing column, files whose schemas
        # differ only in columns not read, text read as text, exact names.
        .use_pandas_metadata(False)
        .ignore_missing_columns(False)
        .allow_mismatched_pq_schemas(True)
        .convert_strings_to_categories(False)
        .case_sensitive_names(True)
        .column_names([physical[name] for name in columns])
        .build()
    )
    return options, physical
