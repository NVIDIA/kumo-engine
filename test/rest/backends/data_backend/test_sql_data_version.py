# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Change-detection (data_version) unit tests for Custom SQL tables.

These exercise ``get_table_data_version``'s SQL branch with a mocked
referenced-table version resolver, so they never touch a warehouse.
"""
import asyncio

import pytest
from kumoapi.table import Column
from kumoapi.typing import Dtype, Stype

from kumo.connector import (
    BigQueryConnector,
    DatabricksConnector,
    SnowflakeConnector,
)
from rest.backends.data_backend import utils
from rest.config.data_backend import ConnectorSpec
from rest.config.table_v2 import CustomSQLSourceTable, TableDefinition
from statedb.config.table import TableDataVersion

# NOTE(Tier A reconcile): master hardened ConnectorID to require the
# ``connector_<32 hex>`` shape, so the original 'c' fixture no longer validates.
_CID = 'connector_00000000000000000000000000000000'
# A resolved connector spec carries the connector's NATIVE config
# (``ConfigMixin.config()``), exactly as every production construction does
# (``connector_config=<conn>.config()``). That config has no ``type`` field, so
# the warehouse type is derived from the connector class — a bare
# ``{'type': ...}`` dict is not a real spec shape and cannot be built into a
# connector.
SPEC = ConnectorSpec(
    connector_id=_CID,
    connector_config=SnowflakeConnector(name='wh', account='acc',
                                        warehouse='COMPUTE', database='DB',
                                        schema_name='S').config())

# BigQuery and Databricks specs exercise the referenced-table normalization for
# the other two SQL dialects. The version path derives the dialect from the
# spec's connector class (not the ``CustomSQLSourceTable.connector_type`` field),
# so these drive the BigQuery/Databricks branches of the leaf normalization.
_BQ_PROJECT = 'future-pager-502102-v6'
_BQ_DATASET = 'kumo_smoke'
BQ_SPEC = ConnectorSpec(
    connector_id=_CID,
    connector_config=BigQueryConnector(name='wh', project_id=_BQ_PROJECT,
                                       dataset_id=_BQ_DATASET).config())
DBX_SPEC = ConnectorSpec(
    connector_id=_CID, connector_config=DatabricksConnector(
        name='wh', host='https://x.cloud.databricks.com', cluster_id='c',
        warehouse_id='w', catalog='main', schema_name='kumo_smoke').config())


def _sql_table(
    sql='SELECT item_key, total FROM sales', cols=None,
    connector_type='snowflake',
    referenced_tables=('sales', )) -> TableDefinition:
    cols = cols or [
        Column(name='item_key', stype=Stype.ID, dtype=Dtype.string),
        Column(name='total', stype=Stype.numerical,
               dtype=Dtype.float),
    ]
    return TableDefinition(
        cols=cols,
        source_table=CustomSQLSourceTable(
            connector_id=_CID, connector_type=connector_type, sql=sql,
            table='custom_sql_abc', referenced_tables=list(referenced_tables)),
        pkey='item_key',
        time_col=None,
    )


# A wall-clock reading aligned to a max-staleness bucket boundary, so bucket
# arithmetic in the tests is exact. ``_get_sql_table_data_version`` folds an
# aligned ``int(time.time()) // (CUSTOM_SQL_MAX_STALENESS_HOURS * 3600)`` bucket
# into the version, so tests pin ``time.time`` to keep results deterministic and
# to drive rollover explicitly rather than depending on the real clock.
_BUCKET_SECS = utils.CUSTOM_SQL_MAX_STALENESS_HOURS * 3600
_FIXED_NOW = float(_BUCKET_SECS * 100)


def _data_version(table_def, versions, monkeypatch, *, raise_for=(),
                  now=_FIXED_NOW):
    async def fake_gtdv(connector_spec, name, refresh_table_version=False,
                        table_def=None):
        if name in raise_for:
            raise RuntimeError(f"cannot resolve version for {name}")
        return TableDataVersion(versions[name])

    monkeypatch.setattr(utils, 'get_table_data_version', fake_gtdv)
    # Pin the clock so the aligned staleness bucket is fixed within a test
    # (stable inputs → stable version) unless a test advances ``now`` to force a
    # bucket rollover.
    monkeypatch.setattr(utils.time, 'time', lambda: now)
    return asyncio.run(utils._get_sql_table_data_version(SPEC, table_def, True))


def test_data_version_deterministic_and_sensitive(monkeypatch):
    # ``FROM sales`` parses to the ref ``sales``; the resolution path reduces it
    # to the connector-dialect leaf before probing, and Snowflake upper-folds an
    # unquoted identifier, so the version resolver is keyed on ``SALES``.
    versions = {'SALES': 'v1'}
    base = _data_version(_sql_table(), versions, monkeypatch)

    # No-change recompute is a true no-op (identical hash → skip-guard hits).
    assert base == _data_version(_sql_table(), versions, monkeypatch)

    # Formatting-only edit is stable.
    assert base == _data_version(
        _sql_table('select  item_key,\n total\nFROM sales'), versions,
        monkeypatch)

    # Semantic SQL edit changes it.
    assert base != _data_version(
        _sql_table('SELECT item_key, total FROM sales WHERE total > 0'),
        versions, monkeypatch)

    # Referenced-table version change changes it.
    assert base != _data_version(_sql_table(), {'SALES': 'v2'}, monkeypatch)

    # cols/dtype-override change changes it (mandatory — export casts per cols,
    # and the fingerprint path-id excludes cols).
    override = [
        Column(name='item_key', stype=Stype.ID, dtype=Dtype.string),
        Column(name='total', stype=Stype.numerical, dtype=Dtype.int),
    ]
    assert base != _data_version(_sql_table(cols=override), versions,
                                 monkeypatch)


def test_indeterminate_referenced_version_forces_reingest(monkeypatch):
    # When a referenced table's version can't be resolved, each computation
    # gets a fresh sentinel, so the data_version differs every time and can
    # never yield a stale cache hit (AC6.5).
    v1 = _data_version(_sql_table(), {}, monkeypatch, raise_for={'SALES'})
    v2 = _data_version(_sql_table(), {}, monkeypatch, raise_for={'SALES'})
    assert v1 != v2


def test_unparseable_sql_forces_reingest(monkeypatch):
    # If the SQL no longer parses, the dependency set is unknown; each
    # computation gets a fresh sentinel so the version differs every time
    # (never a stale hit) rather than hashing SQL+cols alone.
    table = _sql_table(sql='this is (not sql')
    v1 = _data_version(table, {}, monkeypatch)
    v2 = _data_version(table, {}, monkeypatch)
    assert v1 != v2


def test_referenced_tables_resolved_via_physical_path_not_synthetic(
        monkeypatch):
    # The referenced external tables (not the SQL table's synthetic name) are
    # what get resolved through the physical version path.
    resolved = []

    async def fake_gtdv(connector_spec, name, refresh_table_version=False,
                        table_def=None):
        resolved.append(name)
        return TableDataVersion('phys')

    monkeypatch.setattr(utils, 'get_table_data_version', fake_gtdv)
    asyncio.run(utils._get_sql_table_data_version(SPEC, _sql_table(), True))
    # Only the referenced table (as its connector-dialect leaf, ``SALES`` for
    # Snowflake) is resolved via the connector path; the synthetic name
    # ('custom_sql_abc') is never probed as a physical table.
    assert resolved == ['SALES']


def test_get_table_data_version_dispatches_to_sql_branch(monkeypatch):
    # Exercise the real (decorated) ``get_table_data_version`` entrypoint: a SQL
    # table_def selects the SQL branch. A no-reference query needs no connector,
    # so this covers the dispatch + hashing without a warehouse.
    monkeypatch.setattr(utils.time, 'time', lambda: _FIXED_NOW)
    table = _sql_table(sql='SELECT 1 AS a')
    dv = asyncio.run(
        utils.get_table_data_version(SPEC, table.get_source_table_name(),
                                     table_def=table))
    # Deterministic and equal to computing the content hash directly. The caller
    # passes the aligned staleness bucket via the dedicated ``max_staleness_bucket``
    # field (never the referenced-table map), so the expected hash must supply it
    # the same way (the pure ``custom_sql_data_version`` itself never reads the
    # clock).
    from kumo_core.types import ConnectorType

    from kumo.data.custom_sql import custom_sql_data_version
    bucket = int(_FIXED_NOW) // _BUCKET_SECS
    expected = custom_sql_data_version('SELECT 1 AS a', ConnectorType.SNOWFLAKE,
                                       {},
                                       {c.name: c.dtype
                                        for c in table.cols},
                                       max_staleness_bucket=bucket)
    assert str(dv) == expected


def test_stable_within_bucket_but_rolls_over_at_boundary(monkeypatch):
    # The max-staleness floor with a NON-EMPTY parsed ref set whose extracted
    # version is held STABLE ('sales' -> 'v1') across all three computations.
    # Within one aligned window the bucket is constant, so stable SQL + stable
    # refs hash identically (cache hit, no churn); the moment the clock crosses a
    # CUSTOM_SQL_MAX_STALENESS_HOURS boundary the bucket rolls and the version
    # differs — forcing a re-ingest even though the parsed reference's version
    # never moved. This is the core guarantee: the extracted refs are an
    # incomplete proxy (they can be static while the query's real data drifts via
    # TVFs/read_files/views), so the floor must bound staleness regardless.
    versions = {'SALES': 'v1'}
    # ``_FIXED_NOW`` is aligned to a bucket boundary, so bucket math is exact.
    v_start = _data_version(_sql_table(), versions, monkeypatch, now=_FIXED_NOW)
    v_same_window = _data_version(_sql_table(), versions, monkeypatch,
                                  now=_FIXED_NOW + _BUCKET_SECS - 1)
    assert v_start == v_same_window

    v_next_window = _data_version(_sql_table(), versions, monkeypatch,
                                  now=_FIXED_NOW + _BUCKET_SECS)
    assert v_start != v_next_window


def test_empty_ref_set_bounded_by_staleness_floor(monkeypatch):
    # The degenerate extreme of the incomplete-proxy problem: a query that PARSES
    # but references no external tables (e.g. ``SELECT 1``) extracts an EMPTY set,
    # which used to hash to a constant forever → the table would never re-ingest
    # (silent staleness). The staleness floor now bounds it: stable within a
    # window, but rolls over across a boundary.
    table = _sql_table(sql='SELECT 1 AS a')
    v0 = _data_version(table, {}, monkeypatch, now=_FIXED_NOW)
    v0_same = _data_version(table, {}, monkeypatch,
                            now=_FIXED_NOW + _BUCKET_SECS - 1)
    assert v0 == v0_same

    v1 = _data_version(table, {}, monkeypatch, now=_FIXED_NOW + _BUCKET_SECS)
    assert v0 != v1


def _resolve_and_capture(spec, table_def, versions, monkeypatch, *,
                         now=_FIXED_NOW):
    """Run the version path, returning ``(data_version, [probed names])``.

    Records the exact table name handed to ``get_table_data_version`` so a test
    can assert the connector probe receives the normalized bare leaf (no quoting,
    no qualifier), not the dialect-quoted parse output.
    """
    probed = []

    async def fake_gtdv(connector_spec, name, refresh_table_version=False,
                        table_def=None):
        probed.append(name)
        return TableDataVersion(versions[name])

    monkeypatch.setattr(utils, 'get_table_data_version', fake_gtdv)
    monkeypatch.setattr(utils.time, 'time', lambda: now)
    dv = asyncio.run(utils._get_sql_table_data_version(spec, table_def, True))
    return dv, probed


def test_bigquery_referenced_table_resolves_deterministically(monkeypatch):
    # Regression: a BigQuery Custom SQL table whose referenced table is written
    # fully-qualified with backticks (as ``referenced_tables_or_none`` renders
    # it) used to be handed straight to the connector probe, which rejected the
    # quoted/over-qualified name, threw, and collapsed to a fresh per-call
    # ``indeterminate`` sentinel -> a different data_version every call, so batch
    # prediction asked for a snapshot training never materialized. The
    # resolution path now reduces the ref to its case-preserving bare leaf.
    bq_sql = ('SELECT item_key FROM '
              '`future-pager-502102-v6`.`kumo_smoke`.`ACCOUNT`')
    table = _sql_table(
        sql=bq_sql, connector_type='bigquery', cols=[
            Column(name='item_key', stype=Stype.ID,
                   dtype=Dtype.string)
        ])

    v1, probed1 = _resolve_and_capture(BQ_SPEC, table, {'ACCOUNT': 'phys-1'},
                                       monkeypatch)
    v2, probed2 = _resolve_and_capture(BQ_SPEC, table, {'ACCOUNT': 'phys-1'},
                                       monkeypatch)

    # The connector probe receives the backtick-stripped, case-preserved bare
    # leaf (BigQuery table names are case-sensitive), NOT the quoted 3-part name.
    assert probed1 == ['ACCOUNT']
    # Deterministic across recomputes (train vs. batch-prediction parity).
    assert str(v1) == str(v2)
    # A resolvable ref never produces the fresh-uuid sentinel.
    assert 'indeterminate:' not in str(v1)


def test_databricks_referenced_table_normalized_no_regression(monkeypatch):
    # Databricks resolved deterministically before the fix (its connector probe
    # falls back to a stable hour-bucket rather than propagating). Confirm the
    # leaf normalization keeps it deterministic and strips backticks: the probe
    # gets the lower-folded bare leaf that ``_fully_qualify_table_name`` expects.
    dbx_sql = 'SELECT item_key FROM `main`.`kumo_smoke`.`ACCOUNT`'
    table = _sql_table(
        sql=dbx_sql, connector_type='databricks', cols=[
            Column(name='item_key', stype=Stype.ID,
                   dtype=Dtype.string)
        ])

    v1, probed1 = _resolve_and_capture(DBX_SPEC, table, {'account': 'phys-1'},
                                       monkeypatch)
    v2, _ = _resolve_and_capture(DBX_SPEC, table, {'account': 'phys-1'},
                                 monkeypatch)

    # Databricks folds identifiers to lower case; the leaf is unquoted.
    assert probed1 == ['account']
    assert str(v1) == str(v2)
    assert 'indeterminate:' not in str(v1)


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
