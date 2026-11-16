# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Portable graph conversion: `kumo.engine.graph.Graph` -> the dict `kumo_core`
training/prediction consume as `graph_config`.

Ported from `kumoai/backend/local/portable.py` in the cloud SDK, adapted to
this package's `Graph` shape: `Graph.edges` are already resolved
`kumoapi.graph.Edge` objects (built directly by the caller), not the
column-group form the cloud SDK's `GraphDefinition.col_groups` holds -- so
the `graph_api.resolve_foreign_keys` step `_to_portable_graph` does isn't
needed here. `kumoapi.graph.Edge`'s fields (`src_table`, `fkey`, `dst_table`)
already exactly match what `kumo_core.portable_graph._project_portable_graph`
expects an edge to look like, confirmed by reading both directly.

`connector_id` per table: originally omitted here on the reasoning that
nothing in this package looks a connector up by id -- every `Connector` is
threaded by direct object reference everywhere else (`Table.connector`,
`PredictionOutputConfig.connector`, ...). That reasoning covered
`_project_portable_graph`/`normalize_graph` correctly (both treat
`connector_id` as fully optional), but missed that `kumo_core.diskgraph.
ingest.execute_diskgraph_ingest`'s own input validation
(`validate_resolved_diskgraph_inputs`) separately requires `connector_id` to
be a non-empty string on every table, regardless of whether anything later
dereferences it -- confirmed by hitting `DiskgraphInputError: graph.
tables[0].connector_id must be a non-empty string` from a real `Graph.
snapshot()` call. So `connector_id` is filled in after all, using
`table.connector.id` (this package's own fingerprinted pseudo-id) -- exactly
the same "must be well-formed and stable, not actually resolvable" shape
requirement `source_table_name` already has, not a real registry lookup.

`source_table_name` per table: `kumo_core.pquery.normalize_graph` requires
this to look like `catalog.schema.table` (exactly three dot-separated,
non-empty parts), but -- confirmed by reading every use of it in
`kumo_core` -- it is never dereferenced to actually read data. It ends up
folded into `kumo_core.types.schema.SourceTableID`/`VersionedSourceTableID`,
used purely as a stable identity/versioning key. The real data read already
happens earlier and separately, via `SourceBackend.prepare_ingest(...)` ->
local Parquet -> `DiskgraphIngestSpec.source_paths` (`Graph.snapshot()`).
So `_source_table_name` below only needs to produce a well-formed, stable
identifier from fields the `Connector` already holds directly -- not to
replicate the cloud SDK's `_source_coordinate`/`_object_store_coordinate`/
`_custom_sql_coordinate` (`portable.py:107-258`), which exist there to drive
an actual `locate`-based connector lookup this package has no equivalent of.
"""

import warnings
from typing import Any, Dict, Mapping, Sequence

from kumoapi.graph import Edge, graph_definition_from_edges
from kumoapi.source_table import (
    ADLSRelativeSource,
    ADLSSourceTable,
    BigQuerySourceTable,
    CustomSQLSourceTable,
    DatabricksSourceTable,
    GCSRelativeSource,
    GCSSourceTable,
    LocalFileSourceTable,
    S3RelativeSource,
    S3SourceTable,
    SnowflakeSourceTable,
)

from kumo.engine._local_schema import GraphDefinition, LocalSourceTableType
from kumo.engine._scratch import fingerprint
from kumo.engine.table import Table


class PortableGraphError(ValueError):
    r"""Raised when a `Graph` can't be converted into `kumo_core`'s portable
    `graph_config` -- e.g. a table has no resolved `definition`, or
    `kumo_core.portable_graph._project_portable_graph` itself rejects the
    result (a missing primary key on an edge's destination table, a dangling
    edge, ...).
    """


def _escape_dots(value: str) -> str:
    r"""`source_table_name` is split on `.` to find its three parts, so a
    literal `.` inside one (an S3 key, a Parquet filename, ...) has to be
    escaped first -- otherwise a path like `events/2026/orders.parquet`
    would look like a fourth part and fail `normalize_graph`'s shape check.
    """
    return value.replace('.', '%2E')


def _source_table_name(source_table: LocalSourceTableType) -> str:
    r"""Builds the `catalog.schema.table`-shaped identifier
    `kumo_core.pquery.normalize_graph` requires. See module docstring for
    why this only needs to be well-formed and stable, not a resolvable
    connection target.
    """
    if isinstance(source_table, LocalFileSourceTable):
        return 'file.local.' + fingerprint(source_table.path.encode())
    if isinstance(source_table, SnowflakeSourceTable):
        return '.'.join(
            _escape_dots(part)
            for part in (source_table.database, source_table.schema_name,
                         source_table.table))
    if isinstance(source_table, BigQuerySourceTable):
        return '.'.join(
            _escape_dots(part)
            for part in (source_table.project_id, source_table.dataset_id,
                         source_table.table_name))
    if isinstance(source_table, DatabricksSourceTable):
        # Already catalog.schema.table by construction
        # (DatabricksConnector.source_table_for).
        return source_table.table
    if isinstance(source_table, S3SourceTable):
        location = (source_table.source.path if isinstance(
            source_table.source, S3RelativeSource) else source_table.source.uri)
        return '.'.join(('s3', _escape_dots(source_table.connector_id[:16]),
                         _escape_dots(location)))
    if isinstance(source_table, ADLSSourceTable):
        location = (source_table.source.path if isinstance(
            source_table.source, ADLSRelativeSource) else
                    source_table.source.uri)
        return '.'.join(('adls', _escape_dots(source_table.connector_id[:16]),
                         _escape_dots(location)))
    if isinstance(source_table, GCSSourceTable):
        location = (source_table.source.path if isinstance(
            source_table.source, GCSRelativeSource) else
                    source_table.source.uri)
        return '.'.join(('gcs', _escape_dots(source_table.connector_id[:16]),
                         _escape_dots(location)))
    if isinstance(source_table, CustomSQLSourceTable):
        return '.'.join(('custom_sql',
                         _escape_dots(source_table.connector_id[:16]),
                         _escape_dots(source_table.table)))
    raise PortableGraphError(
        f"no source_table_name coordinate builder for "
        f"{type(source_table).__name__}.")


def portable_graph_config(
    tables: Mapping[str, Table],
    edges: Sequence[Edge],
    *,
    name: str,
) -> Dict[str, Any]:
    r"""Builds the dict `kumo_core` training/prediction consume as
    `graph_config`, via `kumo_core.portable_graph._project_portable_graph`.

    Args:
        tables: every table in the graph. Each must have a resolved
            `definition` -- there is no schema auto-resolution here (see the
            TODO on `Table.definition` in `kumo/engine/table.py`).
        edges: the graph's edges, passed straight through with no foreign-key
            resolution step (see module docstring).
        name: the graph's name -- `kumo_core` requires a `'name'` key on the
            portable dict; the caller (`Graph._graph_config`) passes its own
            `Graph.id`.
    """
    definitions = {}
    for alias, table in tables.items():
        if table.definition is None:
            raise PortableGraphError(
                f"Table {alias!r} has no resolved `definition`; "
                "call infer_metadata() or provide a definition.")
        definitions[alias] = table.definition
    try:
        definition = graph_definition_from_edges(
            definitions, edges, definition_type=GraphDefinition)
    except ValueError as error:
        raise PortableGraphError(str(error)) from error

    for edge in edges:
        if edge.src_table == edge.dst_table:
            pkey = definition.tables[edge.dst_table].pkey
            warnings.warn(
                f"Self-relation {edge.src_table}.{edge.fkey} -> "
                f"{edge.dst_table}.{pkey} is not supported by Kumo Local "
                "and will be omitted from the snapshot.",
                UserWarning,
                stacklevel=2,
            )

    source_table_names = {
        name: _source_table_name(table.source_table)
        for name, table in definition.tables.items()
    }
    connector_ids = {name: table.connector.id for name, table in tables.items()}

    from kumo_core.portable_graph import _project_portable_graph

    try:
        graph_config = _project_portable_graph(definition, edges,
                                               source_table_names,
                                               connector_ids)
    except ValueError as error:
        raise PortableGraphError(str(error)) from error

    graph_config['name'] = name
    return graph_config
