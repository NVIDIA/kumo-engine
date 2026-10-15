# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the v2-native validators in ``rest.driver.table_logic``.

These functions (``_validate_*_v2``) operate on ``TableDefinition``
directly, without a v1 conversion step. They mirror the legacy v1
``_validate_*`` family in message format.

They raise ``ValidationException`` (an allowlisted ``PublicException``
subclass) for user-input validation errors rather than a bare
``ValueError``: these validators run inside data-backend single activities,
and ``_raise_safe_single_activity_failure`` masks any non-allowlisted
exception type as the generic "Data backend operation failed. Please retry
or contact Kumo support." Raising the allowlisted typed exception is what
lets the hand-written, actionable message reach the caller with a 422
instead of being replaced by that generic 500 (kumo-ai/kumo#29794).

Notable v1/v2 difference: the v2 versions drop the ``if not args.is_view:``
guard in ``_validate_pkey_and_time_col`` because ``TableDefinition`` does
not model views.
"""
from test.rest.config import test_table as table_test_fixtures
# Reuse the per-source-type TableDefinition factories already defined in
# `test/rest/config/test_table.py` (and the v1 round-trip helper) so we
# don't rebuild source-table fixtures here. Module-private (`_`-prefixed)
# is fine — Python doesn't enforce import access, and pytest collects both
# modules anyway.
from types import SimpleNamespace
from typing import Callable, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kumo_core.types.schema import BaseColumnConfig
from kumoapi import table as table_api
from kumoapi.typing import Dtype, Stype

from common_utils.error import PublicException
from kumo.config.graph import TableMetadataConfig
from kumo.connector.connector import Connector
from kumo.connector.sql_connector import SQLConnector
from kumo.validation import ValidationError, ValidationException
from rest.backends.data_backend import data_backend
from rest.config.table_v2 import TableDefinition
from rest.driver import table_logic as table_logic_module
from rest.driver.error_constants import (
    INVALID_PRIMARY_KEY_TITLE,
    INVALID_TIME_COLUMN_TITLE,
    TABLE_READ_ERROR_MESSAGE,
    TABLE_READ_PROVIDER_ERROR,
)
from rest.driver.table_logic import (
    _resolve_table_metadata_with_connector,
    _table_read_error_v2,
    _validate_col_names_v2,
    _validate_cols_are_unique_v2,
    _validate_pkey_and_time_col_v2,
    _validate_table_metadata_v2,
    _validate_table_permissions_v2,
    _validate_table_size_v2,
)
from rest.utils.routers import ValidationHTTPException
from statedb.config.table import (
    PKEY_IS_ID,
    PKEY_PRESENT_IN_COLS,
    TIME_COL_NEQ_PKEY,
    TIME_COL_PRESENT_IN_COLS,
)
from statedb.utils.errors import TableInvalidColumnNameError

# Source-type factory parametrization shared across V2-vs-V1 contract tests.
ALL_SOURCE_FACTORIES = [
    table_test_fixtures._s3_named,
    table_test_fixtures._gcs_named,
    table_test_fixtures._snowflake,
    table_test_fixtures._databricks,
    table_test_fixtures._bigquery,
]

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _col(name: str, stype: Stype = Stype.numerical,
         dtype: Dtype = Dtype.float64) -> table_api.Column:
    return table_api.Column(name=name, stype=stype, dtype=dtype)


def _make_definition(
    cols: List[table_api.Column],
    pkey: Optional[str] = None,
    time_col: Optional[str] = None,
    end_time_col: Optional[str] = None,
    factory: Callable[[], TableDefinition] = table_test_fixtures._s3_named,
) -> TableDefinition:
    """Construct a v2 TableDefinition with caller-controlled cols for
    branch-specific validator tests. The source_table comes from the
    shared factory (``_s3_named`` by default) so we don't rebuild
    ``S3SourceTable`` instances here.
    """
    base = factory()
    return TableDefinition(
        source_table=base.source_table,
        cols=tuple(cols),
        pkey=pkey,
        time_col=time_col,
        end_time_col=end_time_col,
    )


def _source_col(name: str, stype: Stype = Stype.numerical,
                dtype: Dtype = Dtype.float64) -> BaseColumnConfig:
    return BaseColumnConfig(name=name, stype=stype, dtype=dtype)


def _source_metadata_config(
        cols: List[BaseColumnConfig]) -> TableMetadataConfig:
    """Minimal source-table metadata for ``_validate_table_metadata_v2``."""
    return TableMetadataConfig(
        table_name="customers",
        data_source="",
        source_name="customers",
        cols=cols,
    )


# ---------------------------------------------------------------------------
# _validate_col_names_v2
# ---------------------------------------------------------------------------


class TestValidateColNamesV2:
    def test_valid_names_pass(self):
        d = _make_definition(
            cols=[_col("id", Stype.ID, Dtype.int64),
                  _col("score")])
        _validate_col_names_v2(d)  # no raise

    @pytest.mark.parametrize(
        "reserved", ["ENTITY", "TARGET", "ASSUMPTION", "ENTITY_PRECOMPUTE"])
    def test_reserved_name_as_pkey_raises(self, reserved):
        d = _make_definition(
            cols=[_col(reserved, Stype.ID, Dtype.int64),
                  _col("v")],
            pkey=reserved,
        )
        with pytest.raises(TableInvalidColumnNameError):
            _validate_col_names_v2(d)

    def test_reserved_name_as_non_pkey_passes(self):
        d = _make_definition(
            cols=[
                _col("real_id", Stype.ID, Dtype.int64),
                _col("ENTITY", Stype.numerical, Dtype.float64),
            ],
            pkey="real_id",
        )
        _validate_col_names_v2(d)  # no raise — not the pkey

    def test_invalid_sql_identifier_raises(self):
        # Spaces are not legal SQL identifiers (validate_sql_id rejects).
        d = _make_definition(cols=[_col("bad name")])
        with pytest.raises(Exception):
            _validate_col_names_v2(d)


# ---------------------------------------------------------------------------
# _validate_cols_are_unique_v2
# ---------------------------------------------------------------------------


class TestValidateColsAreUniqueV2:
    def test_unique_passes(self):
        d = _make_definition(cols=[_col("a"), _col("b")])
        _validate_cols_are_unique_v2(d)

    def test_duplicate_raises_with_table_label(self):
        d = _make_definition(cols=[_col("dup"), _col("dup")])
        # The label in the v2 error message is the source-side table name
        # (v2 has no per-graph alias on the TableDefinition); pull it from
        # the same definition for the assertion so the test is independent
        # of which factory we used.
        expected_label = str(d.get_source_table_name())
        with pytest.raises(ValidationException) as exc:
            _validate_cols_are_unique_v2(d)
        # Message format mirrors v1 — uses '<source_name>.<col_name>'
        assert "duplicate col names" in str(exc.value)
        assert f"{expected_label}.dup" in str(exc.value)

    def test_case_different_names_pass(self):
        # v1 comparison is case-sensitive; v2 inherits the same behavior.
        d = _make_definition(cols=[_col("ID"), _col("id")])
        _validate_cols_are_unique_v2(d)

    def test_empty_cols_passes(self):
        # No cols -> no duplicates possible. (TableDefinition pydantic
        # validators may reject empty cols — this is exercised via a
        # try/except so the test passes whether or not construction
        # succeeds.)
        try:
            d = _make_definition(cols=[])
        except Exception:
            pytest.skip("empty cols rejected at construction time")
        _validate_cols_are_unique_v2(d)


# ---------------------------------------------------------------------------
# _validate_pkey_and_time_col_v2
# ---------------------------------------------------------------------------


class TestValidatePkeyAndTimeColV2:
    def test_valid_pkey_and_time_col_pass(self):
        d = _make_definition(
            cols=[
                _col("id", Stype.ID, Dtype.int64),
                _col("ts", Stype.timestamp, Dtype.time),
                _col("v"),
            ],
            pkey="id",
            time_col="ts",
        )
        _validate_pkey_and_time_col_v2(d)

    def test_pkey_none_skips(self):
        d = _make_definition(cols=[_col("v")], pkey=None)
        _validate_pkey_and_time_col_v2(d)

    def test_pkey_not_in_cols_raises_pkey_present_in_cols(self):
        d = _make_definition(cols=[_col("v")], pkey="missing")
        with pytest.raises(ValidationException) as exc:
            _validate_pkey_and_time_col_v2(d)
        assert exc.value.as_error() == ValidationError(
            INVALID_PRIMARY_KEY_TITLE, PKEY_PRESENT_IN_COLS)

    def test_pkey_wrong_stype_raises_pkey_is_id(self):
        d = _make_definition(
            cols=[_col("p", Stype.categorical, Dtype.string)],
            pkey="p",
        )
        with pytest.raises(ValidationException) as exc:
            _validate_pkey_and_time_col_v2(d)
        assert exc.value.as_error() == ValidationError(
            INVALID_PRIMARY_KEY_TITLE, PKEY_IS_ID)

    def test_time_col_not_in_cols_raises(self):
        d = _make_definition(
            cols=[_col("id", Stype.ID, Dtype.int64)],
            pkey="id",
            time_col="missing",
        )
        with pytest.raises(ValidationException) as exc:
            _validate_pkey_and_time_col_v2(d)
        assert exc.value.as_error() == ValidationError(
            INVALID_TIME_COLUMN_TITLE, TIME_COL_PRESENT_IN_COLS)

    def test_time_col_equals_pkey_raises_time_col_neq_pkey(self):
        d = _make_definition(
            cols=[_col("id", Stype.ID, Dtype.int64)],
            pkey="id",
            time_col="id",
        )
        with pytest.raises(ValidationException) as exc:
            _validate_pkey_and_time_col_v2(d)
        assert exc.value.as_error() == ValidationError(
            INVALID_TIME_COLUMN_TITLE, TIME_COL_NEQ_PKEY)

    def test_time_col_wrong_stype_raises(self):
        d = _make_definition(
            cols=[
                _col("id", Stype.ID, Dtype.int64),
                _col("ts", Stype.categorical, Dtype.string),
            ],
            pkey="id",
            time_col="ts",
        )
        with pytest.raises(ValidationException) as exc:
            _validate_pkey_and_time_col_v2(d)
        msg = str(exc.value)
        assert "Time Column" in msg
        assert "timestamp" in msg

    def test_end_time_col_validated_same_as_time_col(self):
        d = _make_definition(
            cols=[
                _col("id", Stype.ID, Dtype.int64),
                _col("ts", Stype.timestamp, Dtype.time),
            ],
            pkey="id",
            time_col="ts",
            end_time_col="missing",
        )
        with pytest.raises(ValidationException) as exc:
            _validate_pkey_and_time_col_v2(d)
        assert exc.value.as_error() == ValidationError(
            INVALID_TIME_COLUMN_TITLE, TIME_COL_PRESENT_IN_COLS)

    def test_both_time_cols_valid_pass(self):
        d = _make_definition(
            cols=[
                _col("id", Stype.ID, Dtype.int64),
                _col("start", Stype.timestamp, Dtype.time),
                _col("end", Stype.timestamp, Dtype.time),
            ],
            pkey="id",
            time_col="start",
            end_time_col="end",
        )
        _validate_pkey_and_time_col_v2(d)

    def test_no_view_guard_runs_unconditionally(self):
        # v1 has `if not args.is_view:` to skip pkey/time-col validation
        # for views. v2 doesn't model views — pkey validation always runs.
        # Confirm by passing a missing pkey: v2 must raise (no skip).
        d = _make_definition(cols=[_col("v")], pkey="missing")
        with pytest.raises(ValidationException):
            _validate_pkey_and_time_col_v2(d)


# ---------------------------------------------------------------------------
# _validate_table_size_v2
# ---------------------------------------------------------------------------


class TestValidateTableSizeV2:
    def _make_connector(self, name: str = "S3", source_size: int = 0,
                        not_implemented: bool = False) -> MagicMock:
        connector = MagicMock(spec=Connector)
        # `_validate_table_size_v2` derives `connector_type` from the class
        # name minus 'Connector', so name the mock class accordingly.
        connector.__class__.__name__ = f"{name}Connector"
        connector.name = "test_conn"
        if not_implemented:
            connector._get_selected_table_source_size.side_effect = \
                NotImplementedError()
        else:
            connector._get_selected_table_source_size.return_value = \
                source_size
        return connector

    def test_no_limit_for_connector_type_skips(self):
        d = _make_definition(cols=[_col("v")])
        connector = self._make_connector("S3", source_size=10**12)
        # No limits at all -> no validation runs.
        _validate_table_size_v2.sync(d, connector, {})

    def test_size_under_limit_passes(self):
        d = _make_definition(cols=[_col("v")])
        connector = self._make_connector("S3", source_size=100)
        _validate_table_size_v2.sync(d, connector, {"s3": 1000})

    def test_size_at_boundary_passes(self):
        d = _make_definition(cols=[_col("v")])
        connector = self._make_connector("S3", source_size=1000)
        # Boundary: condition is strict `>`, so equal passes.
        _validate_table_size_v2.sync(d, connector, {"s3": 1000})

    def test_size_over_limit_raises(self):
        d = _make_definition(cols=[_col("v")])
        connector = self._make_connector("S3", source_size=2000)
        with pytest.raises(ValidationException):
            _validate_table_size_v2.sync(d, connector, {"s3": 1000})

    def test_not_implemented_treated_as_zero(self):
        d = _make_definition(cols=[_col("v")])
        connector = self._make_connector("S3", not_implemented=True)
        # NotImplementedError -> size 0 -> passes any positive limit.
        _validate_table_size_v2.sync(d, connector, {"s3": 1})


# ---------------------------------------------------------------------------
# _validate_table_permissions_v2
# ---------------------------------------------------------------------------


class TestValidateTablePermissionsV2:
    def test_non_sql_connector_skipped(self):
        d = _make_definition(cols=[_col("v")])
        connector = MagicMock(spec=Connector)  # not SQLConnector
        # No raise expected — function returns silently for non-SQL.
        _validate_table_permissions_v2.sync(d, connector)

    def test_sql_connector_ok_passes(self):
        d = _make_definition(cols=[_col("v")])
        connector = MagicMock(spec=SQLConnector)
        connector.check_table_read_permissions.return_value = None
        _validate_table_permissions_v2.sync(d, connector)
        connector.check_table_read_permissions.assert_called_once()

    def test_sql_connector_not_implemented_skipped(self):
        d = _make_definition(cols=[_col("v")])
        connector = MagicMock(spec=SQLConnector)
        connector.check_table_read_permissions.side_effect = \
            NotImplementedError()
        # NotImplementedError swallowed; function returns silently.
        _validate_table_permissions_v2.sync(d, connector)


# ---------------------------------------------------------------------------
# _table_read_error_v2
# ---------------------------------------------------------------------------


class TestTableReadErrorV2:
    """``_table_read_error_v2`` produces a ``ValidationHTTPException`` whose
    formatted message is built from the ``TableDefinition`` source/connector
    fields via the shared ``TABLE_READ_ERROR_MESSAGE`` template.
    """
    @pytest.mark.parametrize('factory', ALL_SOURCE_FACTORIES)
    def test_message_for_all_source_types(self, factory):
        table_def = factory()
        err = FileNotFoundError("source missing")

        v2_exc = _table_read_error_v2(table_def, err)

        assert isinstance(v2_exc, ValidationHTTPException)
        expected_v2 = TABLE_READ_ERROR_MESSAGE.format(
            str(table_def.get_source_table_name()),
            table_def.get_connector_id(),
            TABLE_READ_PROVIDER_ERROR,
        )
        assert v2_exc.detail == expected_v2

    def test_redacts_underlying_provider_error(self, caplog):
        table_def = table_test_fixtures._s3_named()
        provider_secret = "known-provider-error-secret"
        err = ValueError(provider_secret)
        v2_exc = _table_read_error_v2(table_def, err)
        assert isinstance(v2_exc, ValidationHTTPException)
        assert provider_secret not in v2_exc.detail
        assert provider_secret not in caplog.text


# ---------------------------------------------------------------------------
# Regression (kumo-ai/kumo#29794): user-input validation errors must raise an
# allowlisted PublicException so the data-backend single-activity wrapper
# forwards the actionable message instead of masking it as a generic 500.
# ---------------------------------------------------------------------------


class TestValidatorsRaiseAllowlistedPublicException:
    """``_raise_safe_single_activity_failure`` only forwards an activity
    error's real message when the raised type's name is in
    ``_SINGLE_ACTIVITY_PUBLIC_ERROR_TYPES``; any other type (notably a bare
    ``ValueError``) is masked as "Data backend operation failed. Please retry
    or contact Kumo support." These validators carry hand-written,
    credential-free, user-actionable messages, so a regression that reverts
    them to ``ValueError`` would silently re-mask the message. Guard the type
    at the raise sites of both single activities: the ``_validate_*_v2``
    family and ``_resolve_table_metadata_with_connector``.
    """
    def _assert_allowlisted(self, exc: Exception) -> None:
        # A bare ValueError would be masked by the single-activity wrapper.
        assert not isinstance(exc, ValueError)
        assert isinstance(exc, PublicException)
        assert type(exc).__name__ in (
            data_backend._SINGLE_ACTIVITY_PUBLIC_ERROR_TYPES)

    def test_duplicate_cols_error_is_allowlisted(self):
        d = _make_definition(cols=[_col("dup"), _col("dup")])
        with pytest.raises(ValidationException) as exc:
            _validate_cols_are_unique_v2(d)
        self._assert_allowlisted(exc.value)
        assert "duplicate col names" in str(exc.value)

    def test_pkey_not_in_cols_error_is_allowlisted(self):
        d = _make_definition(cols=[_col("v")], pkey="missing")
        with pytest.raises(ValidationException) as exc:
            _validate_pkey_and_time_col_v2(d)
        self._assert_allowlisted(exc.value)
        # The actionable message survives (not the generic masked one).
        assert exc.value.as_error() == ValidationError(
            INVALID_PRIMARY_KEY_TITLE, PKEY_PRESENT_IN_COLS)

    def test_pkey_wrong_stype_error_is_allowlisted(self):
        d = _make_definition(
            cols=[_col("p", Stype.categorical, Dtype.string)],
            pkey="p",
        )
        with pytest.raises(ValidationException) as exc:
            _validate_pkey_and_time_col_v2(d)
        self._assert_allowlisted(exc.value)
        assert exc.value.as_error() == ValidationError(
            INVALID_PRIMARY_KEY_TITLE, PKEY_IS_ID)

    def test_time_col_wrong_stype_error_is_allowlisted(self):
        d = _make_definition(
            cols=[
                _col("id", Stype.ID, Dtype.int64),
                _col("ts", Stype.categorical, Dtype.string),
            ],
            pkey="id",
            time_col="ts",
        )
        with pytest.raises(ValidationException) as exc:
            _validate_pkey_and_time_col_v2(d)
        self._assert_allowlisted(exc.value)

    def test_metadata_unspecified_dtype_error_is_allowlisted(self):
        d = _make_definition(cols=[_col("a")])
        # Construction-time validation already requires a dtype, so bypass
        # it to reach the validator's defensive branch.
        object.__setattr__(d.cols[0], "dtype", None)
        metadata = _source_metadata_config([_source_col("a")])
        with pytest.raises(ValidationException) as exc:
            _validate_table_metadata_v2.sync(d, MagicMock(), metadata)
        self._assert_allowlisted(exc.value)
        assert "has non-specified data type" in str(exc.value)

    def test_metadata_column_not_in_source_error_is_allowlisted(self):
        d = _make_definition(cols=[_col("ghost")])
        metadata = _source_metadata_config([_source_col("other")])
        with pytest.raises(ValidationException) as exc:
            _validate_table_metadata_v2.sync(d, MagicMock(), metadata)
        self._assert_allowlisted(exc.value)
        assert "does not exist in table" in str(exc.value)

    def test_metadata_dtype_mismatch_error_is_allowlisted(self):
        d = _make_definition(cols=[_col("a", Stype.numerical, Dtype.int64)])
        metadata = _source_metadata_config(
            [_source_col("a", Stype.categorical, Dtype.string)])
        with pytest.raises(ValidationException) as exc:
            _validate_table_metadata_v2.sync(d, MagicMock(), metadata)
        self._assert_allowlisted(exc.value)
        assert "was selected to be Int64 but should be String" in str(exc.value)

    def test_metadata_selected_unsupported_columns_are_actionable(self):
        d = _make_definition(
            cols=[
                _col("id", Stype.ID, Dtype.int64),
                _col("payload", Stype.unsupported, Dtype.unsupported),
                _col("shape", Stype.unsupported, Dtype.unsupported),
            ], pkey="id")
        metadata = _source_metadata_config([
            _source_col("id", Stype.ID, Dtype.int64),
            _source_col("payload", Stype.unsupported,
                        Dtype.unsupported),
            _source_col("shape", Stype.unsupported,
                        Dtype.unsupported),
        ])

        with pytest.raises(ValidationException) as exc:
            _validate_table_metadata_v2.sync(d, MagicMock(), metadata)

        self._assert_allowlisted(exc.value)
        error = exc.value.as_error()
        assert error.title == "Unsupported column type"
        assert "'payload', 'shape'" in error.message
        assert "Exclude them from the Kumo table definition" in error.message

    def test_metadata_declared_dtype_resolves_unsupported_source_column(self):
        declared = [
            _col("id", Stype.ID, Dtype.int64),
            _col("payload", Stype.numerical, Dtype.int64),
        ]
        metadata = _source_metadata_config([
            _source_col("id", Stype.ID, Dtype.int64),
            _source_col("payload", Stype.unsupported, Dtype.unsupported),
            _source_col("shape", Stype.unsupported, Dtype.unsupported),
        ])
        connector = MagicMock()
        connector.name = "fixture"

        undeclared = _make_definition(
            cols=declared +
            [_col("shape", Stype.unsupported, Dtype.unsupported)], pkey="id")
        with pytest.raises(ValidationException) as exc:
            _validate_table_metadata_v2.sync(undeclared, connector, metadata)
        assert "Column 'shape' in table" in exc.value.as_error().message

        _validate_table_metadata_v2.sync(
            _make_definition(cols=declared, pkey="id"), connector, metadata)

    def test_metadata_unselected_unsupported_source_column_is_ignored(self):
        d = _make_definition(cols=[_col("id", Stype.ID, Dtype.int64)],
                             pkey="id")
        metadata = _source_metadata_config([
            _source_col("id", Stype.ID, Dtype.int64),
            _source_col("payload", Stype.unsupported,
                        Dtype.unsupported),
        ])
        connector = MagicMock()
        connector.name = "fixture"

        _validate_table_metadata_v2.sync(d, connector, metadata)

    def test_metadata_tail_value_error_is_actionable(self):
        d = _make_definition(
            cols=[
                _col("id", Stype.ID, Dtype.int64),
                _col("not_a_time", Stype.categorical, Dtype.string),
            ],
            pkey="id",
            time_col="not_a_time",
        )
        metadata = _source_metadata_config([
            _source_col("id", Stype.ID, Dtype.int64),
            _source_col("not_a_time", Stype.categorical,
                        Dtype.string),
        ])
        connector = MagicMock()
        connector.name = "fixture"

        with pytest.raises(ValidationException) as exc:
            _validate_table_metadata_v2.sync(d, connector, metadata)

        self._assert_allowlisted(exc.value)
        error = exc.value.as_error()
        assert error.title == "Invalid table definition"
        assert "not_a_time" in error.message
        assert "Must be either" in error.message

    @pytest.mark.asyncio
    async def test_infer_metadata_unknown_column_error_is_allowlisted(self):
        request = table_api.TableMetadataRequest(
            source_table=table_test_fixtures._s3_named().source_table,
            cols=[
                table_api.ColumnMetadataRequest(
                    name="ghost",
                    stype=Stype.numerical,
                    dtype=Dtype.float64,
                ),
            ],
            pkey=None,
        )
        source_metadata = MagicMock()
        source_metadata.all_cols.return_value = [
            SimpleNamespace(name="other"),
        ]
        with patch.object(
                table_logic_module,
                "_get_source_table_metadata",
                new=SimpleNamespace(coroutine=AsyncMock(
                    return_value=source_metadata)),
        ):
            with pytest.raises(ValidationException) as exc:
                await _resolve_table_metadata_with_connector(
                    request, MagicMock())
        self._assert_allowlisted(exc.value)
        assert "Column ghost does not exist in table" in str(exc.value)

    @pytest.mark.asyncio
    async def test_infer_metadata_missing_source_error_is_allowlisted(self):
        request = table_api.TableMetadataRequest(
            source_table=table_test_fixtures._s3_named().source_table,
            cols=[
                table_api.ColumnMetadataRequest(
                    name="a",
                    stype=Stype.numerical,
                    dtype=Dtype.float64,
                ),
            ],
            pkey=None,
        )
        source_metadata = MagicMock()
        source_metadata.all_cols.return_value = [SimpleNamespace(name="a")]
        with patch.object(
                table_logic_module,
                "_get_source_table_metadata",
                new=SimpleNamespace(coroutine=AsyncMock(
                    return_value=source_metadata)),
        ), patch.object(
                table_logic_module.Table,
                "_infer_metadata_blocking",
                new=SimpleNamespace(coroutine=AsyncMock(
                    side_effect=FileNotFoundError("missing"))),
        ), patch.object(
                table_logic_module.Table,
                "_infer_dtypes_from_stypes_blocking",
                new=SimpleNamespace(coroutine=AsyncMock()),
        ):
            with pytest.raises(ValidationException) as exc:
                await _resolve_table_metadata_with_connector(
                    request, MagicMock())
        self._assert_allowlisted(exc.value)
        assert "Specified source table could not be found" in str(exc.value)
