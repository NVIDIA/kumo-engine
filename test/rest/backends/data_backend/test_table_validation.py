# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import dataclasses
import os.path as osp
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

import pytest
from kumo_core.types import ColStatType, EdgeStatType, TableStatType
from kumo_core.types.schema import BaseColumnConfig, VersionedSourceTableID
from kumoapi.typing import Dtype, Stype
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    CancelledError,
)

from common_utils.error import PublicException
from io_utils import fsutil
from kumo.data import get_active_cache
from kumo.data.backend_cache import AsyncDataTaskStatus, TableDataTaskResult
from kumo.data.cache import SourceTableColumn
from kumo.data.common import MaterializationStage, TableRefreshStage
from kumo.data.utils import get_materialize_path
from kumo.stats.stats import Stats
from kumo.validation import ValidationError, ValidationResponse
from kumo.validation.validation import ValidationException
from rest.backends.data_backend.activities import DataBackendActivities
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.validation import (
    ColStatsValidator,
    GraphRefreshValidator,
    GraphTableStatsNotReady,
    IngestionValidator,
)
from rest.backends.data_backend.workflows import table_refresh_workflow as trw
from rest.config.data_backend import (
    ConnectorExecutionRef,
    GraphDataVersionInfo,
    GraphVersion,
)
from rest.config.file_type import FileType
from rest.config.graph_v2 import (
    ColumnKey,
    ColumnKeyGroup,
    GraphDataComputeInput,
    GraphDefinition,
)
from rest.config.table_v2 import Column as RestColumn
from rest.config.table_v2 import (
    CustomSQLSourceTable,
    DataSourceType,
    S3SourceTable,
    TableDefinition,
)
from service_framework.deploy_env_config import DeployEnvConfig
from statedb.config.common import ProjectID, TableIDv2, TableName
from statedb.config.table import TableDataVersion
from temporalio_commons.error_utils import is_error_non_retryable


def test_validate_single_table_col_stats(make_default_table_def_v2):
    backend_cache = get_active_cache()
    table_def = TableDefinition(
        **dataclasses.asdict(make_default_table_def_v2()))
    test_version = '123456'

    backend_cache.add_table_stats(
        table_def.fingerprint(), {
            TableStatType.NUM_ROWS:
            Stats(value=100, approx=False, data_version=test_version)
        })

    base_cols = table_def.to_base_columns()

    # Create fake stat set that will result in warnings and errors:
    # non-unique pkey (and add some stats for other cols as well)
    test_col_stats = {
        base_col: {
            ColStatType.NUM_UNIQUE:
            Stats(value=10, approx=False, data_version=test_version),
            ColStatType.COUNT:
            Stats(value=100, approx=False, data_version=test_version),
        }
        for base_col in base_cols
    }

    # 3rd column, 'f2' is sequence stype, set mismatched lengths:
    test_col_stats[base_cols[2]][ColStatType.SEQUENCE_MIN_LENGTH] = Stats(
        value=10, approx=False, data_version=test_version)
    test_col_stats[base_cols[2]][ColStatType.SEQUENCE_MAX_LENGTH] = Stats(
        value=12, approx=False, data_version=test_version)

    # set 'f3' over invalid threshold
    test_col_stats[base_cols[3]][ColStatType.INVALID_FRACTION] = Stats(
        value=0.2, approx=False, data_version=test_version)

    backend_cache.add_col_stats(table_def.fingerprint(), test_col_stats)

    validator = ColStatsValidator()
    response = validator.post_col_stats(table_def.get_connector_id(), table_def,
                                        test_version)
    # INVALID_FRACTION is now vestigial under DiskGraph (see
    # _VESTIGIAL_FULL_STATS in kumo/graph/column.py), so the
    # "Large number invalid entries" warning path is unreachable.
    assert len(response.warnings) == 1
    assert any(e.title == 'Incomplete table statistics'
               for e in response.errors)
    response.errors = [
        e for e in response.errors if e.title != 'Incomplete table statistics'
    ]
    assert len(response.errors) == 1
    assert response.warnings[0].title == 'Duplicate primary keys'
    assert response.errors[0].title == 'Inconsistent embedding dimensions'


def test_validate_pkey_invalid_and_duplicate_warnings():
    backend_cache = get_active_cache()
    table_def = TableDefinition(
        cols=[RestColumn(name='pkey', stype=Stype.ID, dtype=Dtype.int)],
        source_table=S3SourceTable(
            connector_id=f'connector_{"1" * 32}',
            source={
                'mode': 'relative',
                'path': 'mixed-invalid-and-duplicate-pkeys',
            },
            file_type=FileType.PARQUET,  # pyright: ignore
            data_source_type=DataSourceType.S3,
        ),
        pkey='pkey',
    )
    version = 'mixed-invalid-and-duplicate-pkeys'
    backend_cache.add_table_stats(
        table_def.fingerprint(), {
            TableStatType.NUM_ROWS:
            Stats(value=100, approx=False, data_version=version)
        })
    pkey_col = BaseColumnConfig(name='pkey', stype=Stype.ID, dtype=Dtype.int)
    backend_cache.add_col_stats(
        table_def.fingerprint(), {
            pkey_col: {
                ColStatType.COUNT:
                Stats(value=100, approx=False, data_version=version),
                ColStatType.NUM_UNIQUE:
                Stats(value=98, approx=False, data_version=version),
                ColStatType.NA_FRACTION:
                Stats(value=0.01, approx=False, data_version=version),
                ColStatType.NUM_NA:
                Stats(value=1, approx=False, data_version=version),
            }
        })

    unscoped_response = ColStatsValidator().post_col_stats(
        table_def.get_connector_id(), table_def, version)
    assert not any(warning.title == 'Duplicate primary keys'
                   for warning in unscoped_response.warnings)

    response = ColStatsValidator().post_col_stats(
        table_def.get_connector_id(),
        table_def,
        version,
        pkey_duplicate_count=2,
    )
    warnings = {warning.title: warning for warning in response.warnings}

    assert 'Missing or invalid primary keys' in warnings
    assert warnings['Duplicate primary keys'].message.startswith(
        '2 (2.0%) of the rows')


def test_validate_single_table_dimension(make_default_table_def_v2):
    backend_cache = get_active_cache()
    table_def = TableDefinition(
        **dataclasses.asdict(make_default_table_def_v2()))
    test_version = '123456'

    backend_cache.add_table_stats(
        table_def.fingerprint(), {
            TableStatType.NUM_ROWS:
            Stats(value=100, approx=False, data_version=test_version)
        })

    base_cols = table_def.to_base_columns()

    test_col_stats = {
        base_col: {
            ColStatType.NUM_UNIQUE:
            Stats(value=10, approx=False, data_version=test_version),
            ColStatType.COUNT:
            Stats(value=10, approx=False, data_version=test_version),
        }
        for base_col in base_cols
    }

    # 3rd column, 'f2' is sequence stype, set large lengths:
    test_col_stats[base_cols[2]][ColStatType.SEQUENCE_MIN_LENGTH] = Stats(
        value=5000, approx=False, data_version=test_version)
    test_col_stats[base_cols[2]][ColStatType.SEQUENCE_MAX_LENGTH] = Stats(
        value=5000, approx=False, data_version=test_version)

    backend_cache.add_col_stats(table_def.fingerprint(), test_col_stats)

    validator = ColStatsValidator()
    response = validator.post_col_stats(table_def.get_connector_id(), table_def,
                                        test_version)
    # Test data has 10% valid values per column, triggering invalid entries warning
    # assert len(response.warnings) == 1
    # assert response.warnings[0].title == 'Large number invalid entries'
    assert any(e.title == 'Incomplete table statistics'
               for e in response.errors)
    response.errors = [
        e for e in response.errors if e.title != 'Incomplete table statistics'
    ]
    assert len(response.errors) == 1
    assert response.errors[0].title == 'Too many feature dimensions'


def test_validate_table_ingestion(make_default_table_def_v2):
    backend_cache = get_active_cache()
    api_table = make_default_table_def_v2()
    api_table_date = dataclasses.replace(api_table, time_col='DATE')
    table_def = TableDefinition(**dataclasses.asdict(api_table_date))

    test_version = '123456'

    backend_cache.add_table_stats(
        table_def.fingerprint(), {
            TableStatType.NUM_ROWS:
            Stats(value=0, approx=False, data_version=test_version),
            TableStatType.SOURCE_SIZE:
            Stats(value=0, approx=False, data_version=test_version),
        })

    validator = IngestionValidator(DeployEnvConfig.for_test())

    # case 1: materialization stage not set
    response = validator.post_ingestion(table_def, test_version)
    assert len(response.errors) == 1
    assert response.errors[0].title == 'Table ingestion failed'

    backend_cache.materialization_stage.put(table_def.fingerprint(),
                                            test_version,
                                            MaterializationStage.INGEST,
                                            save=True)
    mat_path = get_materialize_path(table_def.fingerprint(), test_version,
                                    MaterializationStage.INGEST)

    fsutil.makedirs(mat_path, exist_ok=True)
    with open(osp.join(mat_path, 'part-0.parquet'), 'w') as f:
        f.write('')

    # case 2: stage set and dir exists, but no rows
    response = validator.post_ingestion(table_def, test_version)
    assert len(response.errors) == 1
    assert response.errors[0].title == 'Table is empty'

    backend_cache.add_table_stats(
        table_def.fingerprint(), {
            TableStatType.NUM_ROWS:
            Stats(value=100, approx=False, data_version=test_version),
            TableStatType.SOURCE_SIZE:
            Stats(value=100, approx=False, data_version=test_version),
        })

    # case 3: stage set and data exists, and some rows were dropped
    response = validator.post_ingestion(table_def, test_version)
    assert len(response.errors) == 1
    assert response.errors[0].title == 'Ingested table empty'
    assert (
        'Rows may have been dropped due to malformed data (CSV) or invalid timestamps in'
        in response.errors[0].message)

    # case 4: without time_col, table is empty
    table_def_no_time = TableDefinition(**dataclasses.asdict(api_table))
    response = validator.post_ingestion(table_def_no_time, test_version)
    assert len(response.errors) == 1
    assert response.errors[0].title == 'Ingested table empty'
    assert 'has no rows' in response.errors[0].message
    assert 'dropped during ingestion' not in response.errors[0].message


def test_validate_direct_source_without_materialization_stage(
        make_default_table_def_v2):
    table = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    version = 'direct-source'
    get_active_cache().add_table_stats(
        table.fingerprint(), {
            TableStatType.NUM_ROWS:
            Stats(value=100, approx=False, data_version=version),
            TableStatType.SOURCE_SIZE:
            Stats(value=1000, approx=False, data_version=version),
        })

    response = IngestionValidator(DeployEnvConfig.for_test()).post_ingestion(
        table,
        version,
        check_local=False,
        require_materialization_stage=False,
    )

    assert response.ok


def test_graph_source_sizes_accept_file_source_paths(
        make_default_table_def_v2, data_backend_ctx: DataBackendContext):
    table = TableDefinition(**dataclasses.asdict(
        make_default_table_def_v2('fixture-data/connector_123/ACCOUNT')))
    version = 'file-source-path'
    table_id = VersionedSourceTableID(
        connector_name=table.get_connector_id(),
        source_table_name=str(table.get_source_table_name()),
        version=version,
    )
    get_active_cache().add_table_stats(
        table.fingerprint(), {
            TableStatType.NUM_ROWS:
            Stats(value=100, approx=False, data_version=version),
            TableStatType.SOURCE_SIZE:
            Stats(value=4096, approx=False, data_version=version),
        })

    result = DataBackendActivities.validate_graph_table_stats(
        data_backend_ctx,
        {table_id: table},
        TableRefreshStage.INGEST_TABLE,
        connector_specs=[],
        aliases={table_id: [TableName('ACCOUNT'),
                            TableName('ACCOUNT_COPY')]},
    )

    assert result.total_source_size_bytes == 8192
    assert result.source_size_bytes_by_table == {
        'ACCOUNT': 4096,
        'ACCOUNT_COPY': 4096
    }


# test graph-level validations that require tables ingested


def test_graph_post_table_ingest_validate(fake_graph_and_tables_request_v2,
                                          data_backend_ctx: DataBackendContext):
    table_args_list = [
        table_def for _alias, table_def in fake_graph_and_tables_request_v2
    ]
    test_version = '12345'
    versioned_table_ids = [
        VersionedSourceTableID(
            connector_name=create_table.get_connector_id(),
            source_table_name=str(create_table.get_source_table_name()),
            version=test_version) for create_table in table_args_list
    ]
    # populate table sizes as table stats in cache
    backend_cache = get_active_cache()
    for table_args in table_args_list:
        backend_cache.add_table_stats(
            table_args.fingerprint(), {
                TableStatType.SOURCE_SIZE:
                Stats(value=10000, approx=False, data_version=test_version),
                TableStatType.NUM_ROWS:
                Stats(value=10000000, approx=False, data_version=test_version)
            })
        backend_cache.save_col_stats_cache(table_args.fingerprint(),
                                           test_version)
    validator = GraphRefreshValidator(DeployEnvConfig.for_test())
    resp = validator.post_table_ingest(
        ctx=data_backend_ctx,
        versioned_table_dict=dict(zip(versioned_table_ids, table_args_list)),
    )

    assert resp.warnings == []
    assert resp.errors == []

    # Row limits were removed (kumo#30173); the graph-size error now comes from
    # the byte cap only. A per-table SOURCE_SIZE above the 2 TB graph limit
    # trips it regardless of row count.
    for table_args in table_args_list:
        backend_cache.add_table_stats(
            table_args.fingerprint(), {
                TableStatType.SOURCE_SIZE:
                Stats(value=3 *
                      (1024**4), approx=False, data_version=test_version),
                TableStatType.NUM_ROWS:
                Stats(value=10000000, approx=False, data_version=test_version)
            })
        backend_cache.save_col_stats_cache(table_args.fingerprint(),
                                           test_version)
    validator = GraphRefreshValidator(DeployEnvConfig.for_test())
    resp = validator.post_table_ingest(
        ctx=data_backend_ctx,
        versioned_table_dict=dict(zip(versioned_table_ids, table_args_list)))

    assert resp.warnings == []
    assert resp.errors[0].title == 'Graph is too large'


def test_graph_post_table_ingest_retries_missing_exact_stats(
        fake_graph_and_tables_request_v2, data_backend_ctx: DataBackendContext):
    table_args_list = [
        table_def for _alias, table_def in fake_graph_and_tables_request_v2
    ]
    # Use a version dedicated to this test. The backend stats cache is shared
    # across the test session (keyed by connector/source/version/fingerprint),
    # so reusing the '12345' version written by
    # test_graph_post_table_ingest_validate would read those stats back and the
    # "missing stats" path would never be exercised.
    test_version = 'missing_exact_stats_only'
    versioned_table_ids = [
        VersionedSourceTableID(
            connector_name=create_table.get_connector_id(),
            source_table_name=str(create_table.get_source_table_name()),
            version=test_version) for create_table in table_args_list
    ]

    validator = GraphRefreshValidator(DeployEnvConfig.for_test())
    with pytest.raises(GraphTableStatsNotReady,
                       match='Exact table stats are missing'):
        validator.post_table_ingest(
            ctx=data_backend_ctx,
            versioned_table_dict=dict(zip(versioned_table_ids,
                                          table_args_list)),
        )


def test_graph_post_table_ingest_retries_partial_table_stats(
        fake_graph_and_tables_request_v2, data_backend_ctx: DataBackendContext):
    table_args_list = [
        table_def for _alias, table_def in fake_graph_and_tables_request_v2
    ]
    # Dedicated version (see note in the missing-exact-stats test): keep this
    # test independent of stats written by other tests in the shared cache.
    test_version = 'partial_stats_missing_source_size'
    versioned_table_ids = [
        VersionedSourceTableID(
            connector_name=create_table.get_connector_id(),
            source_table_name=str(create_table.get_source_table_name()),
            version=test_version) for create_table in table_args_list
    ]
    backend_cache = get_active_cache()
    for table_args in table_args_list:
        backend_cache.save_table_stats(
            table_args.fingerprint(), test_version, {
                TableStatType.NUM_ROWS:
                Stats(value=100, approx=False, data_version=test_version),
            })

    validator = GraphRefreshValidator(DeployEnvConfig.for_test())
    with pytest.raises(GraphTableStatsNotReady, match='SOURCE_SIZE'):
        validator.post_table_ingest(
            ctx=data_backend_ctx,
            versioned_table_dict=dict(zip(versioned_table_ids,
                                          table_args_list)),
        )


def test_graph_post_table_ingest_tolerates_missing_num_rows(
        fake_graph_and_tables_request_v2, data_backend_ctx: DataBackendContext):
    """NUM_ROWS is optional (kumo#30173): the file/parquet connector persists
    none -- no O(part-count) footer scan -- so a table carrying only
    SOURCE_SIZE must validate rather than block the graph refresh with a
    readiness failure. SOURCE_SIZE stays required (covered above)."""
    table_args_list = [
        table_def for _alias, table_def in fake_graph_and_tables_request_v2
    ]
    # Dedicated version (see note in the missing-exact-stats test).
    test_version = 'partial_stats_missing_num_rows'
    versioned_table_ids = [
        VersionedSourceTableID(
            connector_name=create_table.get_connector_id(),
            source_table_name=str(create_table.get_source_table_name()),
            version=test_version) for create_table in table_args_list
    ]
    # Write only SOURCE_SIZE -- exactly what the file/parquet connector's
    # ingest now persists (no NUM_ROWS).
    backend_cache = get_active_cache()
    for table_args in table_args_list:
        backend_cache.save_table_stats(
            table_args.fingerprint(), test_version, {
                TableStatType.SOURCE_SIZE:
                Stats(value=100, approx=False, data_version=test_version),
            })

    validator = GraphRefreshValidator(DeployEnvConfig.for_test())
    # No readiness raise, and a 100-byte graph clears the byte limit.
    resp = validator.post_table_ingest(
        ctx=data_backend_ctx,
        versioned_table_dict=dict(zip(versioned_table_ids, table_args_list)))
    assert resp.errors == []


def test_graph_post_table_ingest_converges_after_stats_written(
        fake_graph_and_tables_request_v2, data_backend_ctx: DataBackendContext):
    """A readiness failure must clear once the missing stat is written.

    This is what makes the retry meaningful: when validate_graph_table_stats
    retries and stat propagation has caught up, post_table_ingest must stop
    raising and succeed (rather than loop until the schedule-to-close timeout).
    Simulates the missing SOURCE_SIZE appearing on a later retry attempt.
    """
    table_args_list = [
        table_def for _alias, table_def in fake_graph_and_tables_request_v2
    ]
    test_version = 'convergence_after_stats_written'
    versioned_table_ids = [
        VersionedSourceTableID(
            connector_name=create_table.get_connector_id(),
            source_table_name=str(create_table.get_source_table_name()),
            version=test_version) for create_table in table_args_list
    ]
    versioned_table_dict = dict(zip(versioned_table_ids, table_args_list))
    backend_cache = get_active_cache()
    validator = GraphRefreshValidator(DeployEnvConfig.for_test())

    # Attempt 1: only NUM_ROWS present -> readiness failure (would retry).
    for table_args in table_args_list:
        backend_cache.save_table_stats(
            table_args.fingerprint(), test_version, {
                TableStatType.NUM_ROWS:
                Stats(value=100, approx=False, data_version=test_version),
            })
    with pytest.raises(GraphTableStatsNotReady, match='SOURCE_SIZE'):
        validator.post_table_ingest(ctx=data_backend_ctx,
                                    versioned_table_dict=versioned_table_dict)

    # Stat propagation catches up (what a later retry attempt would observe).
    for table_args in table_args_list:
        backend_cache.add_table_stats(
            table_args.fingerprint(), {
                TableStatType.NUM_ROWS:
                Stats(value=100, approx=False, data_version=test_version),
                TableStatType.SOURCE_SIZE:
                Stats(value=100, approx=False, data_version=test_version),
            })

    # Attempt 2: stats are now complete -> validation converges (no raise).
    resp = validator.post_table_ingest(
        ctx=data_backend_ctx, versioned_table_dict=versioned_table_dict)
    assert resp.errors == []


def test_graph_table_stats_not_ready_is_retryable():
    """Regression guard for the whole point of this change.

    Missing/partial table stats during graph refresh are a readiness race, not
    a permanent validation failure, so the validate_graph_table_stats activity
    must RETRY (letting stat propagation converge) instead of hard-failing the
    parent graph-refresh workflow. That only happens because
    GraphTableStatsNotReady is retryable: it is a plain RuntimeError, NOT a
    PublicException (the worker interceptor stamps every PublicException as
    non_retryable), and its type name is not in the default non-retryable set.

    If someone later makes it subclass PublicException or adds it to the
    non-retryable types, the type-based tests above would still pass while the
    fix silently reverts to instant-fail. This test pins the actual behavior.
    """
    assert issubclass(GraphTableStatsNotReady, RuntimeError)
    assert not issubclass(GraphTableStatsNotReady, PublicException)

    # is_error_non_retryable() is the exact predicate the worker interceptor
    # and retry policy rely on. The readiness error must be retryable...
    assert is_error_non_retryable(
        GraphTableStatsNotReady('stats not ready')) is False
    # ...whereas the ValidationException it replaced was non-retryable, which
    # is what made graph refresh fail immediately before this fix.
    assert is_error_non_retryable(
        ValidationException(
            'Required table stats missing (SOURCE_SIZE)')) is True


# test graph-level validations that require col stats


def test_graph_post_col_stats_validate(fake_graph_and_tables_request_v2):
    table_def_pairs = fake_graph_and_tables_request_v2
    test_version = '12345'
    fake_ranges = {
        'TRANS': (datetime(1993, 1, 1), datetime(1998, 10, 31)),
        'LOAN': (datetime(1993, 7, 5), datetime(1998, 12, 31))
    }
    backend_cache = get_active_cache()
    date_col = BaseColumnConfig(name='DATE', stype=Stype.timestamp,
                                dtype=Dtype.date)
    for _alias, table_def in table_def_pairs:
        src_name = str(table_def.get_source_table_name())
        if src_name in fake_ranges.keys():
            table_def.time_col = 'DATE'
            min, max = fake_ranges[src_name]
            fake_stats = {
                date_col: {
                    ColStatType.MIN:
                    Stats(value=min, approx=False, data_version=test_version),
                    ColStatType.MAX:
                    Stats(value=max, approx=False, data_version=test_version)
                }
            }
            backend_cache.add_col_stats(table_def.fingerprint(), fake_stats)
            # need to add table stats so tables don't get ignored by
            # get_many_exact_stats inside validate
            backend_cache.add_table_stats(
                table_def.fingerprint(), {
                    TableStatType.SOURCE_SIZE:
                    Stats(value=1000, approx=False, data_version=test_version),
                    TableStatType.NUM_ROWS:
                    Stats(value=1000, approx=False, data_version=test_version)
                })
            backend_cache.save_col_stats_cache(table_def.fingerprint(),
                                               test_version)
    versioned_table_dict = {}
    aliases = {}
    for alias, table_def in table_def_pairs:
        v_id = VersionedSourceTableID(
            connector_name=table_def.get_connector_id(),
            source_table_name=str(table_def.get_source_table_name()),
            version=test_version,
        )
        versioned_table_dict[v_id] = table_def
        aliases[v_id] = alias

    validator = GraphRefreshValidator(DeployEnvConfig.for_test())
    resp = validator.post_col_stats(versioned_table_dict, aliases=aliases)
    # Now aggregated into a single warning with bullet points
    assert len(resp.warnings) == 1
    assert resp.warnings[0].title == 'Possible Data Missing'
    assert 'When connected tables have large temporal gaps' in resp.warnings[
        0].message
    assert ('TRANS_FIXTURE (1993-01-01 UTC) is 6 months before the earliest '
            'fact in LOAN_FIXTURE (1993-07-05 UTC)' in resp.warnings[0].message)


def test_time_gap_warning_multiple_pairs():
    """Test time gap warning aggregation with multiple table pairs."""
    validator = GraphRefreshValidator(DeployEnvConfig.for_test())

    # Test _get_time_gap_entry returns None for small gaps
    result = validator._get_time_gap_entry(
        "TABLE_A",
        "TABLE_B",
        datetime(2020, 1, 1),
        datetime(2020, 2, 1),  # 1 month gap
        "earliest")
    assert result is None  # Below MAX_ALLOWED_GAP_MONTHS threshold

    # Test _get_time_gap_entry returns entry for large gaps
    result = validator._get_time_gap_entry(
        "TABLE_A",
        "TABLE_B",
        datetime(2020, 1, 1),
        datetime(2020, 9, 1),  # 8 month gap
        "earliest")
    assert result is not None
    assert result == {
        "gap": 8,
        "period": "earliest",
        "earlier_table": "TABLE_A",
        "later_table": "TABLE_B",
        "earlier_date": datetime(2020, 1, 1),
        "later_date": datetime(2020, 9, 1),
    }

    # Test single pair warning with "earliest" period
    warning = validator._create_time_gap_warning([result])
    assert warning.title == "Possible Data Missing"
    assert warning.message == (
        "When connected tables have large temporal gaps in their "
        "records, data may be dropped. Please review your data sources "
        "if this is unexpected:"
        "\n- TABLE_A (2020-01-01 UTC) is 8 months before the earliest "
        "fact in TABLE_B (2020-09-01 UTC)")

    # Test single pair warning with "latest" period
    latest_entry = {
        "gap": 6,
        "period": "latest",
        "earlier_table": "TABLE_X",
        "later_table": "TABLE_Y",
        "earlier_date": datetime(2020, 1, 1),
        "later_date": datetime(2020, 7, 1),
    }
    warning = validator._create_time_gap_warning([latest_entry])
    assert warning.title == "Possible Data Missing"
    assert warning.message == (
        "When connected tables have large temporal gaps in their "
        "records, data may be dropped. Please review your data sources "
        "if this is unexpected:"
        "\n- TABLE_Y (2020-07-01 UTC) is 6 months after the latest "
        "fact in TABLE_X (2020-01-01 UTC)")

    # Test multiple pairs warning lists all gaps as bullet points
    entries = [
        {
            "gap": 8,
            "period": "earliest",
            "earlier_table": "TABLE_A",
            "later_table": "TABLE_B",
            "earlier_date": datetime(2020, 1, 1),
            "later_date": datetime(2020, 9, 1),
        },
        {
            "gap": 6,
            "period": "latest",
            "earlier_table": "TABLE_C",
            "later_table": "TABLE_D",
            "earlier_date": datetime(2020, 3, 1),
            "later_date": datetime(2020, 9, 1),
        },
        {
            "gap": 5,
            "period": "earliest",
            "earlier_table": "TABLE_E",
            "later_table": "TABLE_F",
            "earlier_date": datetime(2020, 2, 1),
            "later_date": datetime(2020, 7, 1),
        },
    ]
    warning = validator._create_time_gap_warning(entries)
    assert warning.title == "Possible Data Missing"
    assert warning.message == (
        "When connected tables have large temporal gaps in their "
        "records, data may be dropped. Please review your data sources "
        "if this is unexpected:"
        "\n- TABLE_A (2020-01-01 UTC) is 8 months before the earliest "
        "fact in TABLE_B (2020-09-01 UTC)"
        "\n- TABLE_D (2020-09-01 UTC) is 6 months after the latest "
        "fact in TABLE_C (2020-03-01 UTC)"
        "\n- TABLE_E (2020-02-01 UTC) is 5 months before the earliest "
        "fact in TABLE_F (2020-07-01 UTC)")


# test graph-level validations that require edge stats


def test_graph_edge_stats_validate(fake_graph_and_tables_request_v2):
    table_def_pairs = fake_graph_and_tables_request_v2
    connector_id = table_def_pairs[0][1].get_connector_id()
    connector_specs = {
        connector_id:
        ConnectorExecutionRef(
            connector_id=connector_id,
            project_id=ProjectID(f'project_{"1" * 32}'),
        )
    }
    input_tables = {}
    table_aliases = {}
    tables_by_name = {}
    for alias, table_def in table_def_pairs:
        table_id = TableIDv2.compose(None, TableName(alias), table_def.fpid)
        input_tables[table_id] = table_def
        table_aliases[table_id] = TableName(alias)
        tables_by_name[TableName(alias)] = table_def

    financial_groups = [
        [
            'DISTRICT_FIXTURE.DISTRICT_ID',
            'ACCOUNT_FIXTURE.DISTRICT_ID',
            'CLIENT_FIXTURE.DISTRICT_ID',
        ],
        ['CLIENT_FIXTURE.CLIENT_ID', 'DISP_FIXTURE.CLIENT_ID'],
        [
            'ACCOUNT_FIXTURE.ACCOUNT_ID',
            'DISP_FIXTURE.ACCOUNT_ID',
            'LOAN_FIXTURE.ACCOUNT_ID',
            'ORDERS_FIXTURE.ACCOUNT_ID',
            'TRANS_FIXTURE.ACCOUNT_ID',
        ],
        ['DISP_FIXTURE.DISP_ID', 'CARD_FIXTURE.DISP_ID'],
    ]
    col_key_groups = [
        ColumnKeyGroup(columns=tuple(
            ColumnKey(table_name=TableName(s.split('.', 1)[0]),
                      col_name=s.split('.', 1)[1]) for s in group))
        for group in financial_groups
    ]

    graph_name = "test_graph"
    graph_input = GraphDataComputeInput(
        graph_display_name=graph_name,
        graph_key=graph_name,
        tables=input_tables,
        base_tables={},
        connectors=connector_specs,
        graph_definition=GraphDefinition(tables=tables_by_name,
                                         col_groups=col_key_groups),
        table_aliases=table_aliases,
    )
    fake_graph_version = '123456'
    graph_version_info = GraphDataVersionInfo(
        graph_data_version=GraphVersion(fake_graph_version),
        table_data_versions={
            TableName(str(table_def.get_source_table_name())):
            TableDataVersion('1234')
            for _alias, table_def in table_def_pairs
        })
    backend_cache = get_active_cache()
    fake_stats = []
    good_edge = {
        EdgeStatType.NUM_SRC_NONZERO_DEGREE:
        Stats(value=100, approx=False, data_version=fake_graph_version),
        EdgeStatType.NUM_DST_NONZERO_DEGREE:
        Stats(value=100, approx=False, data_version=fake_graph_version),
        EdgeStatType.FRAC_SRC_NONZERO_DEGREE:
        Stats(value=1, approx=False, data_version=fake_graph_version),
        EdgeStatType.FRAC_DST_NONZERO_DEGREE:
        Stats(value=1, approx=False, data_version=fake_graph_version),
    }
    fake_stats.append(good_edge)
    bad_edge = copy.deepcopy(good_edge)
    bad_edge[EdgeStatType.NUM_SRC_NONZERO_DEGREE].value = 1
    bad_edge[EdgeStatType.FRAC_SRC_NONZERO_DEGREE].value = 0.1
    fake_stats.append(bad_edge)

    src = SourceTableColumn(connector_id, 'CARD', 'DISP_ID')
    dst = SourceTableColumn(connector_id, 'DISP', 'DISP_ID')
    backend_cache.graph_cache.add_edge_stats(graph_name, src, dst,
                                             fake_stats[0], fake_graph_version)
    src = SourceTableColumn(connector_id, 'DISP', 'CLIENT_ID')
    dst = SourceTableColumn(connector_id, 'CLIENT', 'CLIENT_ID')
    backend_cache.graph_cache.add_edge_stats(graph_name, src, dst,
                                             fake_stats[1], fake_graph_version)
    validator = GraphRefreshValidator(DeployEnvConfig.for_test())
    resp = validator.post_edge_stats(graph_input, graph_version_info)
    assert len(resp.errors) == 0
    assert len(resp.warnings) == 0


def test_validate_col_stats_dropped_rows(make_default_table_def_v2):
    """Test that column stats validation does not trigger when DROPPED_ROWS
    is not available. A separate test in test_dropped_row_stats.py tests
    that DROPPED_ROWS is computed properly."""
    backend_cache = get_active_cache()
    api_table = make_default_table_def_v2()
    # Use first column as time_col to ensure it exists
    first_col_name = api_table.cols[0].name
    table_def = TableDefinition(**dataclasses.asdict(
        dataclasses.replace(api_table, time_col=first_col_name)))
    test_version = '123456'

    # Add table stats (no longer includes DROPPED_ROWS)
    backend_cache.add_table_stats(
        table_def.fingerprint(), {
            TableStatType.NUM_ROWS:
            Stats(value=100, approx=False, data_version=test_version)
        })

    base_cols = table_def.to_base_columns()

    # Create column stats with DROPPED_ROWS for time column
    test_col_stats = {}

    for base_col in base_cols:
        if base_col.name == table_def.time_col:
            test_col_stats[base_col] = {
                ColStatType.DROPPED_ROWS:
                Stats(value=50, approx=False, data_version=test_version),
                ColStatType.COUNT:
                Stats(value=100, approx=False, data_version=test_version),
                ColStatType.NA_FRACTION:
                Stats(value=0.0, approx=False, data_version=test_version),
                ColStatType.INVALID_FRACTION:
                Stats(value=0.0, approx=False, data_version=test_version),
            }
        else:
            test_col_stats[base_col] = {
                ColStatType.COUNT:
                Stats(value=100, approx=False, data_version=test_version),
                ColStatType.NA_FRACTION:
                Stats(value=0.0, approx=False, data_version=test_version),
                ColStatType.INVALID_FRACTION:
                Stats(value=0.0, approx=False, data_version=test_version),
            }

    backend_cache.add_col_stats(table_def.fingerprint(), test_col_stats)

    validator = ColStatsValidator()
    response = validator.post_col_stats('test_connector', table_def,
                                        test_version)

    time_col_warnings = [
        w for w in response.warnings if 'time col' in w.title.lower()
    ]
    assert len(time_col_warnings) == 0, (
        f"Expected no time col warning since DROPPED_ROWS is always computed "
        f"separately and before other column stats, got warnings: "
        f"{[item.title for item in response.warnings]}")


_SQL_CONNECTOR_ID = f'connector_{"5" * 32}'


def _sql_table_def(
    sql_name: str,
    *,
    pkey='pkey',
    cols=None,
    connector_type='snowflake',
) -> TableDefinition:
    r"""A minimal Custom SQL ``TableDefinition`` (``is_sql=True``) whose
    synthetic source name is ``sql_name``."""
    if cols is None:
        cols = [
            RestColumn('pkey', Stype.ID, Dtype.int),
            RestColumn('f1', Stype.numerical, Dtype.float),
        ]
    return TableDefinition(
        cols=cols,
        source_table=CustomSQLSourceTable(connector_id=_SQL_CONNECTOR_ID,
                                          connector_type=connector_type,
                                          sql='SELECT pkey, f1 FROM t',
                                          table=sql_name,
                                          referenced_tables=['t']),
        pkey=pkey,
        time_col=None,
    )


def test_fr15_post_col_stats_empty_sql_result_errors(make_default_table_def_v2):
    """An empty Custom SQL result errors in ``post_col_stats`` (query bug),
    while a raw table with the same zero-row stats stays lenient there."""
    backend_cache = get_active_cache()
    version = 'fr15_colstats_empty'

    sql_def = _sql_table_def('custom_sql_fr15_colstats_empty')
    backend_cache.add_table_stats(sql_def.fingerprint(), {
        TableStatType.NUM_ROWS:
        Stats(value=0, approx=False, data_version=version)
    })
    response = ColStatsValidator().post_col_stats(sql_def.get_connector_id(),
                                                  sql_def, version)
    assert [e.title for e in response.errors] == ['SQL table returned no rows']

    # Raw contrast: the empty-result error is SQL-only.
    raw_def = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    backend_cache.add_table_stats(raw_def.fingerprint(), {
        TableStatType.NUM_ROWS:
        Stats(value=0, approx=False, data_version=version)
    })
    raw_response = ColStatsValidator().post_col_stats(
        raw_def.get_connector_id(), raw_def, version)
    assert not any(e.title == 'SQL table returned no rows'
                   for e in raw_response.errors)


def test_fr15_post_col_stats_null_pkey_sql_errors(make_default_table_def_v2):
    """A null/unparseable SQL primary key errors in ``post_col_stats``; the same
    bad-pkey fraction on a raw table is only a warning."""
    backend_cache = get_active_cache()
    version = 'fr15_colstats_pkey'

    def _bad_pkey_stats(table_def):
        base_cols = table_def.to_base_columns()
        col_stats = {
            base_col: {
                ColStatType.COUNT:
                Stats(value=100, approx=False, data_version=version),
                ColStatType.NUM_NA:
                Stats(value=0, approx=False, data_version=version),
            }
            for base_col in base_cols
        }
        pkey_col = next(c for c in base_cols if c.name == table_def.pkey)
        # Half the primary keys are null -> above MAX_PKEY_BAD_FRACTION.
        col_stats[pkey_col][ColStatType.NA_FRACTION] = Stats(
            value=0.5, approx=False, data_version=version)
        col_stats[pkey_col][ColStatType.NUM_NA] = Stats(value=50, approx=False,
                                                        data_version=version)
        return col_stats

    sql_def = _sql_table_def('custom_sql_fr15_colstats_pkey')
    backend_cache.add_table_stats(
        sql_def.fingerprint(), {
            TableStatType.NUM_ROWS:
            Stats(value=100, approx=False, data_version=version)
        })
    backend_cache.add_col_stats(sql_def.fingerprint(), _bad_pkey_stats(sql_def))
    response = ColStatsValidator().post_col_stats(sql_def.get_connector_id(),
                                                  sql_def, version)
    assert any(e.title == 'Null primary keys in SQL table'
               for e in response.errors)
    assert not any(e.title == 'Incomplete table statistics'
                   for e in response.errors)

    broken = _bad_pkey_stats(sql_def)
    next(stats for col, stats in broken.items()
         if col.name == 'f1')[ColStatType.COUNT] = Stats(
             value=0, approx=False, data_version=version)
    backend_cache.add_col_stats(sql_def.fingerprint(), broken)
    response = ColStatsValidator().post_col_stats(sql_def.get_connector_id(),
                                                  sql_def, version)
    assert not any(e.title == 'Incomplete table statistics'
                   for e in response.errors)

    raw_def = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    backend_cache.add_table_stats(
        raw_def.fingerprint(), {
            TableStatType.NUM_ROWS:
            Stats(value=100, approx=False, data_version=version)
        })
    backend_cache.add_col_stats(raw_def.fingerprint(), _bad_pkey_stats(raw_def))
    raw_response = ColStatsValidator().post_col_stats(
        raw_def.get_connector_id(), raw_def, version)
    assert not any(e.title == 'Null primary keys in SQL table'
                   for e in raw_response.errors)
    assert any(w.title == 'Missing or invalid primary keys'
               for w in raw_response.warnings)


@pytest.mark.asyncio
async def test_standalone_refresh_never_completes_invalid_stats(monkeypatch):
    table = _sql_table_def('custom_sql_incomplete_manifest')
    version = TableDataVersion('incomplete_manifest')
    monkeypatch.setattr(
        trw.workflow, 'info', lambda: SimpleNamespace(
            parent=None, workflow_id='test-workflow', run_id='test-run-id'))
    workflow = trw.TableRefreshWorkflow()
    saved, colstat_stages = [], []

    async def save(result, notify_watcher=True):
        saved.append(copy.deepcopy(result))

    async def ingest():
        return TableDataTaskResult(status=AsyncDataTaskStatus.DONE,
                                   validation=ValidationResponse(),
                                   stage=TableRefreshStage.INGEST_TABLE,
                                   data_version=version)

    async def execute(activity, *args, **kwargs):
        if activity is trw.DataBackendActivities.compute_standalone_table_col_stats:
            return True
        if activity is trw.DataBackendActivities.validate_table_col_stats:
            return ValidationResponse(errors=[
                ValidationError(title='Incomplete table statistics',
                                message='missing')
            ])
        return ValidationResponse()

    async def update():
        return None

    async def save_colstat(stage):
        colstat_stages.append(stage)

    workflow._save_result = save
    workflow._run_ingestion = ingest
    workflow._update_table_timestamp_units = update
    workflow._save_colstat_task_result = save_colstat
    monkeypatch.setattr(trw.workflow, 'execute_activity', execute)
    result = await workflow.run(
        SimpleNamespace(_config=DeployEnvConfig.for_test()),
        trw.TableRefreshWorkflowInput(
            connector_spec=ConnectorExecutionRef(
                connector_id=table.get_connector_id(),
                project_id=f'project_{"0" * 32}', authorization_token='token'),
            table=table, data_version=version))
    assert result.stage is not TableRefreshStage.COMPLETED
    assert not colstat_stages
    assert all(item.stage is not TableRefreshStage.COMPLETED for item in saved)


@pytest.mark.parametrize('public_failure', [True, False],
                         ids=['public', 'internal'])
@pytest.mark.asyncio
async def test_standalone_refresh_returns_failed_col_stats_result(
        monkeypatch, public_failure):
    table = _sql_table_def('custom_sql_wide_integer')
    version = TableDataVersion('wide_integer')
    message = (
        f"Table '{table.get_source_table_name()}' could not be refreshed: "
        "column 'wide_int' contains a scale-zero decimal value outside "
        "Kumo's signed 64-bit integer range")
    internal_message = 'sensitive internal column-statistics failure'
    if public_failure:
        application_error = ApplicationError(
            'activity failed',
            PublicException(message=message, non_retryable=True).to_dict(),
            type='PublicException',
            non_retryable=True,
        )
        expected_message = message
    else:
        application_error = ApplicationError(
            internal_message,
            type='DiskgraphSystemError',
            non_retryable=False,
        )
        expected_message = trw._INTERNAL_COL_STATS_FAILURE_MESSAGE
    activity_error = ActivityError(
        message='activity failed',
        scheduled_event_id=1,
        started_event_id=2,
        identity='',
        activity_type='compute_standalone_table_col_stats',
        activity_id='1',
        retry_state=None,
    )
    activity_error.__cause__ = application_error
    monkeypatch.setattr(
        trw.workflow, 'info', lambda: SimpleNamespace(
            parent=None, workflow_id='test-workflow', run_id='test-run-id'))
    workflow = trw.TableRefreshWorkflow()
    saved = []
    saved_colstat = []

    async def save(result, notify_watcher=True):
        saved.append(copy.deepcopy(result))

    async def ingest():
        return TableDataTaskResult(status=AsyncDataTaskStatus.DONE,
                                   validation=ValidationResponse(),
                                   stage=TableRefreshStage.INGEST_TABLE,
                                   data_version=version)

    async def execute(activity, *args, **kwargs):
        if activity is trw.DataBackendActivities.compute_standalone_table_col_stats:
            raise activity_error
        if activity is trw.DataBackendActivities.save_colstat_task_result:
            saved_colstat.append(copy.deepcopy(kwargs['args'][2]))
            return None
        return ValidationResponse()

    async def update():
        return None

    workflow._save_result = save
    workflow._run_ingestion = ingest
    workflow._update_table_timestamp_units = update
    monkeypatch.setattr(trw.workflow, 'execute_activity', execute)

    result = await workflow.run(
        SimpleNamespace(_config=DeployEnvConfig.for_test()),
        trw.TableRefreshWorkflowInput(
            connector_spec=ConnectorExecutionRef(
                connector_id=table.get_connector_id(),
                project_id=f'project_{"0" * 32}', authorization_token='token'),
            table=table,
            data_version=version,
        ),
    )

    assert result.status is AsyncDataTaskStatus.FAILED
    assert result.stage is TableRefreshStage.COMPUTE_MIN_COL_STATS
    assert result.error == expected_message
    assert saved[-1].status is AsyncDataTaskStatus.FAILED
    assert saved[-1].stage is TableRefreshStage.COMPUTE_MIN_COL_STATS
    assert saved[-1].error == expected_message
    assert saved_colstat == [saved[-1]]
    assert internal_message not in result.error


@pytest.mark.asyncio
async def test_standalone_refresh_propagates_col_stats_cancellation(
        monkeypatch):
    table = _sql_table_def('custom_sql_cancelled_col_stats')
    version = TableDataVersion('cancelled_col_stats')
    activity_error = ActivityError(
        message='activity cancelled',
        scheduled_event_id=1,
        started_event_id=2,
        identity='',
        activity_type='compute_standalone_table_col_stats',
        activity_id='1',
        retry_state=None,
    )
    activity_error.__cause__ = CancelledError('activity cancelled')
    monkeypatch.setattr(
        trw.workflow, 'info', lambda: SimpleNamespace(
            parent=None, workflow_id='test-workflow', run_id='test-run-id'))
    workflow = trw.TableRefreshWorkflow()
    saved = []
    saved_colstat = []

    async def save(result, notify_watcher=True):
        saved.append(copy.deepcopy(result))

    async def ingest():
        return TableDataTaskResult(status=AsyncDataTaskStatus.DONE,
                                   validation=ValidationResponse(),
                                   stage=TableRefreshStage.INGEST_TABLE,
                                   data_version=version)

    async def execute(activity, *args, **kwargs):
        if activity is trw.DataBackendActivities.compute_standalone_table_col_stats:
            raise activity_error
        if activity is trw.DataBackendActivities.save_colstat_task_result:
            saved_colstat.append(copy.deepcopy(kwargs['args'][2]))
            return None
        return ValidationResponse()

    async def update():
        return None

    workflow._save_result = save
    workflow._run_ingestion = ingest
    workflow._update_table_timestamp_units = update
    monkeypatch.setattr(trw.workflow, 'execute_activity', execute)

    with pytest.raises(ActivityError, match='activity cancelled'):
        await workflow.run(
            SimpleNamespace(_config=DeployEnvConfig.for_test()),
            trw.TableRefreshWorkflowInput(
                connector_spec=ConnectorExecutionRef(
                    connector_id=table.get_connector_id(),
                    project_id=f'project_{"0" * 32}',
                    authorization_token='token'),
                table=table,
                data_version=version,
            ),
        )

    assert saved_colstat == []
    assert all(item.status is not AsyncDataTaskStatus.FAILED for item in saved)


@pytest.mark.asyncio
async def test_standalone_refresh_completes_when_stats_intentionally_skipped(
        monkeypatch):
    table = _sql_table_def('custom_sql_direct_without_staged_parquet')
    version = TableDataVersion('direct_without_staged_parquet')
    monkeypatch.setattr(
        trw.workflow, 'info', lambda: SimpleNamespace(
            parent=None, workflow_id='test-workflow', run_id='test-run-id'))
    workflow = trw.TableRefreshWorkflow()
    saved, colstat_stages = [], []

    async def save(result, notify_watcher=True):
        saved.append(copy.deepcopy(result))

    async def ingest():
        return TableDataTaskResult(status=AsyncDataTaskStatus.DONE,
                                   validation=ValidationResponse(),
                                   stage=TableRefreshStage.INGEST_TABLE,
                                   data_version=version)

    async def execute(activity, *args, **kwargs):
        if activity is trw.DataBackendActivities.compute_standalone_table_col_stats:
            return False
        if activity is trw.DataBackendActivities.validate_table_col_stats:
            raise AssertionError('skipped stats must not be validated')
        return ValidationResponse()

    async def update():
        return None

    async def save_colstat(stage):
        colstat_stages.append(stage)

    workflow._save_result = save
    workflow._run_ingestion = ingest
    workflow._update_table_timestamp_units = update
    workflow._save_colstat_task_result = save_colstat
    monkeypatch.setattr(trw.workflow, 'execute_activity', execute)
    result = await workflow.run(
        SimpleNamespace(_config=DeployEnvConfig.for_test()),
        trw.TableRefreshWorkflowInput(
            connector_spec=ConnectorExecutionRef(
                connector_id=table.get_connector_id(),
                project_id=f'project_{"0" * 32}', authorization_token='token'),
            table=table, data_version=version))

    assert result.stage is TableRefreshStage.COMPLETED
    assert not result.min_col_stats_computed
    assert not result.full_col_stats_computed
    assert not colstat_stages
    assert saved[-1].stage is TableRefreshStage.COMPLETED
    assert saved[-1].col_stats_computed is False


def test_fr15_post_ingestion_empty_sql_result_errors():
    """An empty Custom SQL result errors in ``post_ingestion`` with the
    query-specific message rather than the raw ``Table is empty``."""
    backend_cache = get_active_cache()
    version = 'fr15_ingest_empty'
    sql_def = _sql_table_def('custom_sql_fr15_ingest_empty')

    backend_cache.materialization_stage.put(sql_def.fingerprint(), version,
                                            MaterializationStage.INGEST,
                                            save=True)
    backend_cache.add_table_stats(
        sql_def.fingerprint(), {
            TableStatType.NUM_ROWS:
            Stats(value=0, approx=False, data_version=version),
            TableStatType.SOURCE_SIZE:
            Stats(value=0, approx=False, data_version=version),
        })

    # ``check_local=False`` isolates the zero-row source signal from the local
    # parquet ``is_empty`` recovery sniff (the raw "Ingested table empty" path).
    response = IngestionValidator(DeployEnvConfig.for_test()).post_ingestion(
        sql_def, version, check_local=False)
    assert any(e.title == 'SQL table returned no rows' for e in response.errors)
    assert not any(e.title == 'Table is empty' for e in response.errors)


def test_databricks_custom_sql_stats_drive_graph_limits(monkeypatch):
    """A Databricks query now stages to parquet like every other warehouse's,
    so its stats are exact at ingest and the byte limit runs on them here --
    it no longer has to be deferred until DiskGraph published a count."""
    version = 'databricks_exact_graph'
    sql_def = _sql_table_def('databricks_exact_graph',
                             connector_type='databricks')
    versioned_id = VersionedSourceTableID(
        connector_name=sql_def.get_connector_id(),
        source_table_name=str(sql_def.get_source_table_name()),
        version=version,
    )
    exact = SimpleNamespace(
        table_stats={
            TableStatType.NUM_ROWS:
            Stats(value=1000, approx=False, data_version=version),
            TableStatType.SOURCE_SIZE:
            Stats(value=4096, approx=False, data_version=version),
        })
    monkeypatch.setattr(get_active_cache(), 'get_many_min_exact_stats',
                        lambda *_args, **_kwargs: [exact])
    graph_calls = []

    validator = GraphRefreshValidator(DeployEnvConfig.for_test())
    validator._limits = SimpleNamespace(validate_graph=lambda **kwargs: (
        graph_calls.append(kwargs), ValidationResponse())[1])

    response = validator.post_table_ingest(
        ctx=mock.Mock(),
        versioned_table_dict={versioned_id: sql_def},
    )

    assert not response.errors
    assert [call['total_size_bytes'] for call in graph_calls] == [4096]


def test_fr15_post_table_ingest_empty_sql_result_errors(
        data_backend_ctx: DataBackendContext):
    """The graph-level total-rows gate errors when a Custom SQL table in the
    graph returned no rows."""
    backend_cache = get_active_cache()
    version = 'fr15_graph_empty'
    sql_def = _sql_table_def('custom_sql_fr15_graph_empty')
    versioned_id = VersionedSourceTableID(
        connector_name=sql_def.get_connector_id(),
        source_table_name=str(sql_def.get_source_table_name()), version=version)

    backend_cache.add_table_stats(
        sql_def.fingerprint(), {
            TableStatType.SOURCE_SIZE:
            Stats(value=10000, approx=False, data_version=version),
            TableStatType.NUM_ROWS:
            Stats(value=0, approx=False, data_version=version),
        })
    backend_cache.save_col_stats_cache(sql_def.fingerprint(), version)

    resp = GraphRefreshValidator(DeployEnvConfig.for_test()).post_table_ingest(
        ctx=data_backend_ctx,
        versioned_table_dict={versioned_id: sql_def},
    )
    assert any(e.title == 'SQL table returned no rows' for e in resp.errors)


def test_fr15_sql_edge_health_gates():
    """An edge touching a Custom SQL table errors on zero matches and warns on a
    low match rate; a raw-only edge is untouched by the FR15 edge gates."""
    connector_id = _SQL_CONNECTOR_ID
    graph_version = 'fr15_edge_v1'

    sql_def = _sql_table_def('custom_sql_fr15_edge',
                             cols=[RestColumn('id', Stype.ID,
                                              Dtype.int)], pkey='id')

    # A raw S3-backed sibling table so the edge is SQL-to-raw.
    raw_def = TableDefinition(
        cols=[RestColumn('id', Stype.ID, Dtype.int)],
        source_table=S3SourceTable(
            connector_id=connector_id,
            source={
                'mode': 'relative',
                'path': 'RAW_EDGE'
            },
            file_type=FileType.PARQUET,  # pyright: ignore
            data_source_type=DataSourceType.S3,
        ),
        pkey=None,
    )

    sql_name = TableName('SQL_FIXTURE')
    raw_name = TableName('RAW_FIXTURE')
    tables_by_name = {sql_name: sql_def, raw_name: raw_def}
    table_aliases = {
        TableIDv2.compose(None, sql_name, sql_def.fpid): sql_name,
        TableIDv2.compose(None, raw_name, raw_def.fpid): raw_name,
    }
    input_tables = {
        tid: tables_by_name[name]
        for tid, name in table_aliases.items()
    }
    col_group = ColumnKeyGroup(columns=(
        ColumnKey(table_name=sql_name, col_name='id'),
        ColumnKey(table_name=raw_name, col_name='id'),
    ))
    graph_name = 'fr15_edge_graph'
    graph_input = GraphDataComputeInput(
        graph_display_name=graph_name,
        graph_key=graph_name,
        tables=input_tables,
        base_tables={},
        connectors={
            connector_id:
            ConnectorExecutionRef(connector_id=connector_id,
                                  project_id=ProjectID(f'project_{"1" * 32}'))
        },
        graph_definition=GraphDefinition(tables=tables_by_name,
                                         col_groups=[col_group]),
        table_aliases=table_aliases,
    )
    version_info = GraphDataVersionInfo(
        graph_data_version=GraphVersion(graph_version), table_data_versions={
            TableName(str(sql_def.get_source_table_name())):
            TableDataVersion('1'),
            TableName(str(raw_def.get_source_table_name())):
            TableDataVersion('1'),
        })

    # src is the SQL table's pkey ('id'); dst is the raw table's 'id'.
    src = SourceTableColumn(connector_id, str(sql_def.get_source_table_name()),
                            'id')
    dst = SourceTableColumn(connector_id, str(raw_def.get_source_table_name()),
                            'id')
    backend_cache = get_active_cache()

    # Zero matching rows -> error naming the SQL join column.
    backend_cache.graph_cache.add_edge_stats(
        graph_name, src, dst, {
            EdgeStatType.EDGE_COUNT:
            Stats(value=0, approx=False, data_version=graph_version)
        }, graph_version)
    resp = GraphRefreshValidator(DeployEnvConfig.for_test()).post_edge_stats(
        graph_input, version_info)
    assert any(e.title == 'SQL table edge matches no rows' for e in resp.errors)

    # Non-zero but low match rate (10% < 20% threshold) -> warning.
    low_version = 'fr15_edge_v2'
    version_info_low = dataclasses.replace(
        version_info, graph_data_version=GraphVersion(low_version))
    backend_cache.graph_cache.add_edge_stats(
        graph_name, src, dst, {
            EdgeStatType.EDGE_COUNT:
            Stats(value=5, approx=False, data_version=low_version),
            EdgeStatType.FRAC_SRC_NONZERO_DEGREE:
            Stats(value=0.1, approx=False, data_version=low_version),
            EdgeStatType.FRAC_DST_NONZERO_DEGREE:
            Stats(value=0.1, approx=False, data_version=low_version),
        }, low_version)
    resp_low = GraphRefreshValidator(
        DeployEnvConfig.for_test()).post_edge_stats(graph_input,
                                                    version_info_low)
    assert any(w.title == 'Low SQL table edge match rate'
               for w in resp_low.warnings)


def test_graph_source_sizes_reject_an_unusable_source_size(
        make_default_table_def_v2, data_backend_ctx: DataBackendContext):
    """A zero SOURCE_SIZE fails, unless the table has its own error to report.

    ``pool/service.py`` returns early on ``estimate_bytes <= 0``, so a total
    of 0 would skip disk admission for the whole graph in silence. A direct
    Databricks estimate can produce this value before ingest.
    """
    version = 'unusable-source-size'

    def register(table, source_size=0, num_rows=100):
        table_id = VersionedSourceTableID(
            connector_name=table.get_connector_id(),
            source_table_name=str(table.get_source_table_name()),
            version=version,
        )
        get_active_cache().add_table_stats(
            table.fingerprint(), {
                TableStatType.NUM_ROWS:
                Stats(value=num_rows, approx=False, data_version=version),
                TableStatType.SOURCE_SIZE:
                Stats(value=source_size, approx=False, data_version=version),
            })
        return table_id

    table = TableDefinition(**dataclasses.asdict(
        make_default_table_def_v2('fixture-data/connector_123/TRANSACTION')))
    table_id = register(table)
    with pytest.raises(ValidationException, match='TRANSACTION') as excinfo:
        DataBackendActivities.validate_graph_table_stats(
            data_backend_ctx, {table_id: table}, TableRefreshStage.INGEST_TABLE,
            connector_specs=[], aliases={table_id: [TableName('T')]})
    # Permanent, so VALIDATE_STATS_RETRY_POLICY must not spend its ten
    # attempts on it the way it does for the readiness race above.
    assert is_error_non_retryable(excinfo.value) is True

    # A table that failed validation on its own terms reports that error, but
    # no partial source-size map escapes from the otherwise valid tables.
    positive = TableDefinition(**dataclasses.asdict(
        make_default_table_def_v2('fixture-data/connector_123/ACCOUNT')))
    zero = _sql_table_def('custom_sql_zero_size')
    positive_id = register(positive, source_size=4096)
    zero_id = register(zero, num_rows=0)
    result = DataBackendActivities.validate_graph_table_stats(
        data_backend_ctx, {
            positive_id: positive,
            zero_id: zero
        }, TableRefreshStage.INGEST_TABLE, connector_specs=[], aliases={
            positive_id: [TableName('POSITIVE')],
            zero_id: [TableName('ZERO')],
        })
    assert any(e.title == 'SQL table returned no rows'
               for e in result.validation.errors)
    assert result.total_source_size_bytes == 0
    assert result.source_size_bytes_by_table == {}


@pytest.mark.parametrize('count', [0, 1, 100])
def test_column_parse_warning_uses_exact_count(make_default_table_def_v2,
                                               count):
    table = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    cache = get_active_cache()
    version = 'parse-warning'
    cache.add_table_stats(
        table.fingerprint(), {
            TableStatType.NUM_ROWS:
            Stats(value=100, approx=False, data_version=version)
        })
    col = table.to_base_columns()[3]
    cache.add_col_stats(
        table.fingerprint(), {
            col: {
                kind: Stats(value=value, approx=False, data_version=version)
                for kind, value in {
                    ColStatType.COUNT: 100,
                    ColStatType.NUM_PARSE_FAILURES: count,
                    ColStatType.NUM_NA: count,
                    ColStatType.NA_FRACTION: count / 100,
                }.items()
            }
        })
    result = ColStatsValidator().post_col_stats(table.get_connector_id(), table,
                                                version, alias='events')
    assert [warning.message for warning in result.warnings] == ([
        f'{count} invalid values found in events.f3. Invalid values are parsed '
        'as null. Please correct and retrain if needed.'
    ] if count else [])
