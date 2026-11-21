# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Connector tables as local Parquet for kumo-pytorch (experimental)."""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from kumo.engine._scratch import get_scratch_dir
from kumo.engine.connector._staging import stage_source_table
from kumo.engine.connector.source_table import SourceTable

if TYPE_CHECKING:
    from kumo.pytorch import api


@dataclass(frozen=True)
class StagedTable:
    r"""One table staged by `stage_tables`."""
    #: Directory of the table's Parquet parts.
    path: Path
    #: The connector's data version when staged, if it reports one.
    version: Optional[str]


@dataclass(frozen=True)
class StagedTables:
    r"""Tables staged by one `stage_tables` call, keyed by the names passed
    to it."""
    root: Path
    tables: Mapping[str, StagedTable]

    def source(self, name: str) -> 'api.Parquet':
        r"""Table ``name`` as a kumo-pytorch source, for a `Dataset` or an
        `AnchorTable`."""
        return _api().Parquet(path=str(self.tables[name].path))

    def dataset(self) -> 'api.Dataset':
        r"""Every table as a kumo-pytorch `Dataset`."""
        return _api().Dataset(
            tables={name: self.source(name)
                    for name in self.tables})

    def remove(self) -> None:
        r"""Deletes the staged files."""
        shutil.rmtree(self.root)


def stage_tables(
    tables: Mapping[str, SourceTable],
    *,
    root: Optional[Union[str, Path]] = None,
) -> StagedTables:
    r"""Stages each table through its connector as local Parquet.

    Each call writes a new directory, so tables staged earlier stay as they
    were. Columns have the types the connector stages, which can differ from
    the source's.

    Args:
        tables: the name kumo-pytorch uses for each table, to
            ``connector["TABLE"]`` or ``connector.sql(...)``.
        root: local directory to stage under; defaults to the scratch
            directory.
    """
    _api()  # Fail before staging if kumo-pytorch is missing.
    for name, table in tables.items():
        if not isinstance(table, SourceTable):
            raise TypeError(f'{name!r} must be a connector table, not '
                            f'{type(table).__name__}.')
    parent = Path(root) if root is not None else get_scratch_dir() / 'pytorch'
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='tables-', dir=parent))
    staged = {}
    try:
        for index, (name, table) in enumerate(tables.items()):
            # Numbered directories keep table names out of paths. Passing the
            # catalog's dtypes stops object stores casting stored Parquet
            # types, e.g. decimals.
            path, version = stage_source_table(
                table, directory / f'{index:03d}',
                dtypes={column.name: column.dtype
                        for column in table.columns})
            staged[name] = StagedTable(path, version)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return StagedTables(directory, staged)


def _api() -> Any:
    try:
        from kumo.pytorch import api
    except ImportError as error:
        raise ImportError(
            'Staging tables for kumo-pytorch needs the kumo-pytorch '
            'package.') from error
    return api
