# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""`SourceTable`: a reference to a table living behind a `Connector`.

Mirrors the cloud SDK's `kumoai.connector.SourceTable` (`connector[name]` ->
`SourceTable`), minus what only exists to round-trip a server-side project:
construction here only stores the connector and table name, and there is
no `SQLSourceTable` (Custom SQL tables aren't supported by this package
yet -- see "Status" in the package README).

`column_dict`/`columns` use the source-schema pass described in `kumo.engine._infer`.
A column whose stype the source dtype alone doesn't settle (most
numeric/string columns) comes back with
`stype=None`; resolving that via a sampled read is `Table.
from_source_table`'s job, not this class's -- see `kumo.engine._infer`'s module
docstring for why the split lands there rather than in a later no-arg
`infer_metadata()` call the way it does in the cloud SDK.

Cached (`functools.cached_property`) per `SourceTable` instance, computed
once on first access -- unlike the cloud SDK, which round-trips to the
backend on every access.
"""

from functools import cached_property
from typing import Dict, List

from kumoapi.source_table import SourceColumn

from kumo.engine._local_schema import LocalSourceTableType
from kumo.engine.connector.base import Connector


class SourceTable:
    r"""A reference to `table_name` within `connector`, with its (possibly
    partially-resolved) schema read lazily -- and cached -- from the source
    catalog.

    Constructed via `connector[table_name]` (`Connector.__getitem__`) or
    `connector.table(table_name)`, not directly -- consumed by `Table.
    from_source_table(source_table, ...)`.
    """
    def __init__(self, table_name: str, connector: Connector) -> None:
        self.table_name = table_name
        self.connector = connector

    @cached_property
    def column_dict(self) -> Dict[str, SourceColumn]:
        r"""Every column of this table, name to `kumoapi.source_table.
        SourceColumn` -- dtype from the source schema, stype from the
        dtype where that settles it on its own, `None` otherwise. No
        primary-key/time-column awareness (that's a role
        within a particular `Graph`'s schema, not a property of the source
        table itself) -- both are `Table.from_source_table`'s job.
        """
        from kumo.engine._infer import infer_source_columns

        return infer_source_columns(self.connector, self.table_name)

    @property
    def columns(self) -> List[SourceColumn]:
        return list(self.column_dict.values())

    def to_source_table(self) -> LocalSourceTableType:
        r"""Delegate source-coordinate resolution to
        `Connector.source_table_for` for `TableDefinition.source_table`.
        Resolution may perform source I/O to determine the path and format.
        """
        return self.connector.source_table_for(self.table_name)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(table_name={self.table_name!r}, "
                f"connector={self.connector!r})")
