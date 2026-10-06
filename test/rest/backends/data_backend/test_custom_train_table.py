# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the supplied-training-table validator.

These exercise ``_validate_supplied_train_table`` directly with small fixture
parquets in ``tmp_path`` so we can assert on the ``ValidationResponse``
without Temporal activity wrapping.

Every fixture names its columns the way DiskGraph reads them --
``ENTITY1``, ``TARGET1``, ``TIMESTAMP`` -- because that is now the only
naming either backend accepts.
"""
import math
import os
import sys
from typing import Optional
from unittest import mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.types import ConnectorType, TaskType
from kumoapi.pquery import QueryType
from kumoapi.source_table import (
    CustomSQLSourceTable,
    DatabricksSourceTable,
    FileType,
    S3SourceTable,
    SnowflakeSourceTable,
)

from rest.backends.data_backend.activities import custom_train_table
from rest.backends.data_backend.activities.custom_train_table import (
    _validate_supplied_train_table,
    missing_source_column_errors,
)
from statedb.utils.errors import UnsupportedInDiskgraphError


def _write_parquet(path: str, columns: dict) -> None:
    """Write ``columns`` (name -> pyarrow Array/ChunkedArray/list) as a
    parquet directory at ``path``."""
    os.makedirs(path, exist_ok=True)
    table = pa.table(columns)
    pq.write_table(table, os.path.join(path, "part-0.parquet"))


def _custom_source(name: str) -> S3SourceTable:
    return S3SourceTable(
        connector_id='connector_00000000000000000000000000000000',
        source={
            'mode': 'relative',
            'path': name
        },
        file_type=FileType.PARQUET,
    )


def _generated(tmp_path, **overrides) -> str:
    """A three-row generated table, named as DiskGraph reads it."""
    columns = {
        "ENTITY1": pa.array([1, 2, 3], type=pa.int64()),
        "TARGET1": pa.array([0.1, 0.2, 0.3], type=pa.float64()),
    }
    columns.update(overrides)
    path = str(tmp_path / "generated")
    _write_parquet(path, columns)
    return path


def _supplied(tmp_path, **overrides) -> str:
    """The same three rows, supplied, with whatever this test changes."""
    columns = {
        "ENTITY1": pa.array([1, 2, 3], type=pa.int64()),
        "TARGET1": pa.array([0.1, 0.2, 0.3], type=pa.float64()),
    }
    columns.update(overrides)
    columns = {
        name: value
        for name, value in columns.items() if value is not None
    }
    path = str(tmp_path / "supplied")
    _write_parquet(path, columns)
    return path


def _titles(response) -> list:
    return [error.title for error in response.errors]


# Names ######################################################################


def test_a_singular_entity_column_is_refused_by_the_name_it_needs(tmp_path):
    """The one behaviour change this validator exists to make: ``ENTITY`` is
    not a spelling of ``ENTITY1``, on any connector."""
    generated = _generated(tmp_path)
    supplied = str(tmp_path / "supplied")
    _write_parquet(
        supplied, {
            "ENTITY": pa.array([1, 2, 3], type=pa.int64()),
            "TARGET1": pa.array([0.1, 0.2, 0.3], type=pa.float64()),
        })

    resp = _validate_supplied_train_table(generated, supplied, None,
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Required column not found" in _titles(resp)
    assert "missing ENTITY1" in resp.errors[0].message


def test_a_singular_target_column_is_refused_the_same_way(tmp_path):
    """``TARGET`` is a feature column that happens to be spelled like one, so
    the table reads as having lost the target generation wrote.

    Refused by the contract's own required-column check rather than by
    comparison with the generated table, so a registered table with no
    generated one behind it is refused the same way.
    """
    generated = _generated(tmp_path)
    supplied = str(tmp_path / "supplied")
    _write_parquet(
        supplied, {
            "ENTITY1": pa.array([1, 2, 3], type=pa.int64()),
            "TARGET": pa.array([0.1, 0.2, 0.3], type=pa.float64()),
        })

    resp = _validate_supplied_train_table(generated, supplied, None,
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Required column not found" in _titles(resp)
    assert "TARGET1" in resp.errors[0].message


def test_a_singular_target_is_refused_for_a_link_task_by_name(tmp_path):
    """A link task looks ``TARGET1`` up directly, so it is named as required
    rather than only missed against the generated table."""
    list_t = pa.list_(pa.float64())
    generated = _generated(tmp_path, TARGET1=pa.array([[0.1], [0.2], [0.3]],
                                                      type=list_t))
    supplied = str(tmp_path / "supplied")
    _write_parquet(
        supplied, {
            "ENTITY1": pa.array([1, 2, 3], type=pa.int64()),
            "TARGET": pa.array([[0.1], [0.2], [0.3]], type=list_t),
        })

    resp = _validate_supplied_train_table(generated, supplied, None,
                                          TaskType.TEMPORAL_LINK_PREDICTION)

    assert not resp.ok
    assert "missing TARGET1" in resp.errors[0].message


def test_the_source_check_refuses_a_singular_name_before_the_export(tmp_path):
    """The same rule, on a warehouse's column names, so a misnamed table is
    refused before anybody pays to export it."""
    generated = _generated(tmp_path)

    errors = missing_source_column_errors(['ENTITY', 'TARGET1'], generated,
                                          None)

    assert [error.title for error in errors] == ['Required column not found']
    assert 'missing ENTITY1' in errors[0].message


def test_the_source_check_refuses_a_missing_split_column(tmp_path):
    """``split_col_name`` has to reach the source check the same way
    ``weight_col_name`` does, or a misnamed split column would only be
    caught after the export."""
    generated = _generated(tmp_path)

    errors = missing_source_column_errors(['ENTITY1', 'TARGET1'], generated,
                                          None, 'SPLIT')

    assert [error.title for error in errors] == ['Column not found']
    assert 'SPLIT' in errors[0].message


def test_the_source_check_lets_the_anchor_be_synthesized(tmp_path):
    """A static query's anchor is a sentinel ingest writes, so a source is
    not expected to carry one yet."""
    generated = _generated(
        tmp_path, TIMESTAMP=pa.array([1, 2, 3], type=pa.timestamp("ns")))

    assert missing_source_column_errors(['ENTITY1', 'TARGET1'], generated,
                                        None) == []


def test_weight_column_missing(tmp_path):
    generated = _generated(tmp_path)
    supplied = _supplied(tmp_path)

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Column not found" in _titles(resp)


def test_a_table_naming_no_weight_column_passes_every_other_check(tmp_path):
    """A complete table supplied as it stands, carrying no weights.

    Nothing about it is weighted, so nothing weight-shaped is checked -- and
    everything else is, unchanged.
    """
    resp = _validate_supplied_train_table(_generated(tmp_path),
                                          _supplied(tmp_path), None,
                                          TaskType.REGRESSION)

    assert resp.ok, f"unexpected errors: {resp.errors}"


def test_an_unweighted_table_is_still_gated_on_its_row_count(tmp_path):
    """The parity checks do not read weights, so naming none keeps them."""
    generated = _generated(tmp_path)  # 3 rows
    supplied = _supplied(tmp_path, ENTITY1=pa.array([1, 2], type=pa.int64()),
                         TARGET1=pa.array([0.1, 0.2], type=pa.float64()))

    resp = _validate_supplied_train_table(generated, supplied, None,
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Row count mismatch" in _titles(resp)


def test_an_unweighted_table_is_still_gated_on_its_column_types(tmp_path):
    generated = _generated(tmp_path)
    supplied = _supplied(tmp_path, ENTITY1=pa.array(["1", "2", "3"],
                                                    type=pa.string()))

    resp = _validate_supplied_train_table(generated, supplied, None,
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Column type mismatch" in _titles(resp)


def test_an_unweighted_table_is_still_gated_on_its_columns(tmp_path):
    generated = _generated(tmp_path, extra=pa.array(["a", "b", "c"],
                                                    type=pa.string()))
    supplied = _supplied(tmp_path)

    resp = _validate_supplied_train_table(generated, supplied, None,
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Column missing in the supplied table" in _titles(resp)


# Types ######################################################################


@pytest.mark.parametrize(("column", "value"), [
    ("ENTITY1", pa.array([True, False, True], type=pa.bool_())),
    ("ENTITY1", pa.array([1, 2, 3], type=pa.int32())),
    ("TARGET1", pa.array([b"a", b"b", b"c"], type=pa.binary())),
    ("TIMESTAMP", pa.array([1, 2, 3], type=pa.date32())),
])
def test_a_column_diskgraph_cannot_read_is_refused_by_type(
        tmp_path, column, value):
    """DiskGraph refuses none of these itself: an entity key it cannot read
    resolves to no node, an anchor it cannot read reads as null, and the run
    reports success over whatever survived."""
    generated = _generated(tmp_path, **{column: value})
    supplied = _supplied(tmp_path, **{column: value})

    resp = _validate_supplied_train_table(generated, supplied, None,
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Column type cannot be read" in _titles(resp)
    assert column in resp.errors[0].message


def test_a_boolean_entity_is_refused_where_a_boolean_target_is_not(tmp_path):
    """`cell_to_value` reads a boolean and `value_to_pk_key` does not, so an
    entity key is narrower than a target."""
    generated = _generated(
        tmp_path, TARGET1=pa.array([True, False, True], type=pa.bool_()))
    supplied = _supplied(tmp_path, TARGET1=pa.array([True, False, True],
                                                    type=pa.bool_()))

    resp = _validate_supplied_train_table(generated, supplied, None,
                                          TaskType.REGRESSION)

    assert resp.ok, f"unexpected errors: {resp.errors}"


# Weights ####################################################################


def test_nulls_in_weights_error(tmp_path):
    generated = _generated(tmp_path)
    supplied = _supplied(tmp_path, w=pa.array([1.0, None, 2.0],
                                              type=pa.float64()))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Weight column contains N/A" in _titles(resp)


def test_inf_in_weights_error(tmp_path):
    generated = _generated(tmp_path)
    supplied = _supplied(tmp_path, w=pa.array([1.0, math.inf, 2.0],
                                              type=pa.float64()))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Weight column contains infinite values" in _titles(resp)


def test_all_non_positive_error(tmp_path):
    generated = _generated(tmp_path)
    supplied = _supplied(tmp_path, w=pa.array([0.0, -1.0, -2.0],
                                              type=pa.float64()))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Weight column contains no positive values" in _titles(resp)


def test_negative_weights_non_temporal_lp_error(tmp_path):
    generated = _generated(tmp_path)
    supplied = _supplied(tmp_path, w=pa.array([1.0, -1.0, 2.0],
                                              type=pa.float64()))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Weight column contains negative values" in _titles(resp)


def test_negative_weights_temporal_lp_no_error(tmp_path):
    list_t = pa.list_(pa.float64())
    targets = pa.array([[0.1], [0.2, 0.3], [0.4]], type=list_t)
    generated = _generated(tmp_path, TARGET1=targets)
    supplied = _supplied(tmp_path, TARGET1=targets,
                         w=pa.array([[1.0], [-1.0, 2.0], [3.0]], type=list_t))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.TEMPORAL_LINK_PREDICTION)

    assert all("negative" not in (error.title or "")
               for error in resp.errors), f"unexpected errors: {resp.errors}"


def test_int_weights_happy_path(tmp_path):
    """An integer weight skips the inf check and still passes."""
    generated = _generated(tmp_path)
    supplied = _supplied(tmp_path, w=pa.array([1, 2, 3], type=pa.int64()))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.REGRESSION)

    assert resp.ok, f"unexpected errors: {resp.errors}"


def test_a_weight_type_diskgraph_cannot_read_is_refused(tmp_path):
    generated = _generated(tmp_path)
    supplied = _supplied(tmp_path, w=pa.array(["1", "2", "3"],
                                              type=pa.string()))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Column type cannot be read" in _titles(resp)


# Array targets ##############################################################


def test_array_target_scalar_weight_mismatch(tmp_path):
    list_t = pa.list_(pa.float64())
    targets = pa.array([[0.1], [0.2, 0.3], [0.4]], type=list_t)
    generated = _generated(tmp_path, TARGET1=targets)
    supplied = _supplied(tmp_path, TARGET1=targets,
                         w=pa.array([1.0, 2.0, 3.0], type=pa.float64()))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.TEMPORAL_LINK_PREDICTION)

    assert not resp.ok
    assert "Weight column type mismatch for array target" in _titles(resp)


def test_scalar_target_array_weight_mismatch(tmp_path):
    list_t = pa.list_(pa.float64())
    generated = _generated(tmp_path)
    supplied = _supplied(tmp_path, w=pa.array([[1.0], [2.0], [3.0]],
                                              type=list_t))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Weight column type mismatch for array target" in _titles(resp)


def test_array_target_array_weight_happy_path(tmp_path):
    list_t = pa.list_(pa.float64())
    targets = pa.array([[0.1], [0.2, 0.3], [0.4]], type=list_t)
    generated = _generated(tmp_path, TARGET1=targets)
    supplied = _supplied(tmp_path, TARGET1=targets,
                         w=pa.array([[1.0], [2.0, 4.0], [3.0]], type=list_t))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.TEMPORAL_LINK_PREDICTION)

    assert resp.ok, f"unexpected errors: {resp.errors}"


def test_array_weight_explodes_for_inf_check(tmp_path):
    """Inf inside an exploded list weight is still found."""
    list_t = pa.list_(pa.float64())
    targets = pa.array([[0.1], [0.2, 0.3], [0.4]], type=list_t)
    generated = _generated(tmp_path, TARGET1=targets)
    supplied = _supplied(
        tmp_path, TARGET1=targets, w=pa.array([[1.0], [2.0, math.inf], [3.0]],
                                              type=list_t))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.TEMPORAL_LINK_PREDICTION)

    assert not resp.ok
    assert "Weight column contains infinite values" in _titles(resp)


def test_a_per_element_weight_must_be_as_long_as_its_target(tmp_path):
    list_t = pa.list_(pa.float64())
    targets = pa.array([[0.1], [0.2, 0.3], [0.4]], type=list_t)
    generated = _generated(tmp_path, TARGET1=targets)
    supplied = _supplied(tmp_path, TARGET1=targets,
                         w=pa.array([[1.0], [2.0], [3.0]], type=list_t))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.TEMPORAL_LINK_PREDICTION)

    assert not resp.ok
    assert "Weight column length does not match TARGET1" in _titles(resp)


# Contents ###################################################################


def test_a_null_entity_key_is_refused_rather_than_skipped(tmp_path):
    """DiskGraph skips such a row and reports success."""
    generated = _generated(tmp_path)
    supplied = _supplied(tmp_path, ENTITY1=pa.array([1, None, 3],
                                                    type=pa.int64()))

    resp = _validate_supplied_train_table(generated, supplied, None,
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "ENTITY1 contains N/A" in _titles(resp)


def test_a_null_target_is_refused_rather_than_relabelled(tmp_path):
    generated = _generated(tmp_path)
    supplied = _supplied(tmp_path, TARGET1=pa.array([0.1, None, 0.3],
                                                    type=pa.float64()))

    resp = _validate_supplied_train_table(generated, supplied, None,
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "TARGET1 contains N/A" in _titles(resp)


def test_an_empty_table_is_refused(tmp_path):
    generated = _generated(tmp_path)
    supplied = _supplied(tmp_path, ENTITY1=pa.array([], type=pa.int64()),
                         TARGET1=pa.array([], type=pa.float64()))

    resp = _validate_supplied_train_table(generated, supplied, None,
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Supplied table is empty" in _titles(resp)


# Correspondence #############################################################


def test_dtype_mismatch_is_error(tmp_path):
    generated = _generated(tmp_path)
    # A type DiskGraph reads perfectly well, and not the one generation
    # wrote: exact Arrow parity is its own rule.
    supplied = _supplied(tmp_path, TARGET1=pa.array([1, 2, 3], type=pa.int64()),
                         w=pa.array([1.0, 2.0, 3.0], type=pa.float64()))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Column type mismatch" in _titles(resp)


def test_generated_column_missing_in_supplied_is_error(tmp_path):
    generated = _generated(tmp_path, extra=pa.array(["a", "b", "c"],
                                                    type=pa.string()))
    supplied = _supplied(tmp_path, w=pa.array([1.0, 2.0, 3.0],
                                              type=pa.float64()))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Column missing in the supplied table" in _titles(resp)


@pytest.mark.parametrize(
    "supplied_timestamp",
    [
        pa.array([1, 2, 3], type=pa.timestamp("us", tz="Etc/UTC")),
        pa.array([1, 2, 3], type=pa.int64()),
    ],
)
def test_happy_path(tmp_path, supplied_timestamp):
    generated = _generated(
        tmp_path, TIMESTAMP=pa.array([1, 2, 3], type=pa.timestamp("ns")))
    supplied = _supplied(tmp_path, TIMESTAMP=supplied_timestamp,
                         w=pa.array([1.0, 2.0, 3.0], type=pa.float64()))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.REGRESSION)

    assert resp.ok, f"unexpected errors: {resp.errors}"
    assert resp.warnings == []


def test_row_count_mismatch_error(tmp_path):
    generated = _generated(tmp_path)  # 3 rows
    supplied = _supplied(tmp_path, ENTITY1=pa.array([1, 2], type=pa.int64()),
                         TARGET1=pa.array([0.1, 0.2], type=pa.float64()),
                         w=pa.array([1.0, 2.0], type=pa.float64()))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert "Row count mismatch" in _titles(resp)


def test_row_count_mismatch_can_be_explicitly_allowed(tmp_path, monkeypatch):
    from rest.backends.data_backend.activities import custom_train_table as m

    monkeypatch.setattr(
        m.CUSTOM_TT_ALLOW_ROW_COUNT_MISMATCH,
        'parse_value',
        lambda: True,
    )
    generated = _generated(tmp_path)  # 3 rows
    supplied = _supplied(tmp_path, ENTITY1=pa.array([1, 2], type=pa.int64()),
                         TARGET1=pa.array([0.1, 0.2], type=pa.float64()),
                         w=pa.array([1.0, 2.0], type=pa.float64()))

    resp = _validate_supplied_train_table(generated, supplied, "w",
                                          TaskType.REGRESSION)

    assert resp.ok, f"unexpected errors: {resp.errors}"
    assert [warning.title
            for warning in resp.warnings] == ['Row count mismatch allowed']


def test_an_integer_anchor_in_the_wrong_unit_is_caught_by_its_span(tmp_path):
    """An integer anchor carries no unit and is read as nanoseconds, so one
    written in seconds dates every row wrongly without failing."""
    nanos = [
        1_700_000_000_000_000_000, 1_700_000_001_000_000_000,
        1_700_000_002_000_000_000
    ]
    generated = _generated(tmp_path,
                           TIMESTAMP=pa.array(nanos, type=pa.timestamp("ns")))
    supplied = _supplied(
        tmp_path,
        TIMESTAMP=pa.array([value // 1_000_000_000 for value in nanos],
                           type=pa.int64()))

    resp = _validate_supplied_train_table(generated, supplied, None,
                                          TaskType.REGRESSION)

    assert not resp.ok
    assert any('looks like seconds' in title for title in _titles(resp))


# Cost #######################################################################


def test_validate_streams_only_the_columns_the_rules_read(tmp_path):
    """OOM regression guard: the validator must never materialize the full
    supplied parquet via ``ParquetDataset.read()`` / ``pq.read_table()``.
    Pins that a footer-only ``ParquetFile`` open and a *projected*
    ``ParquetFile.iter_batches`` are used, and ``ParquetDataset`` is not.

    The projection is the columns a row's fate turns on, and no other: a
    supplied table may carry any number of feature columns beside them and
    none is decoded here.
    """
    from rest.backends.data_backend.activities import custom_train_table as m

    generated = _generated(tmp_path)
    supplied = _supplied(tmp_path, w=pa.array([1.0, 2.0, 3.0],
                                              type=pa.float64()),
                         feature=pa.array(["a", "b", "c"]))

    def _boom_dataset(*args, **kwargs):
        raise AssertionError(
            "validator must not materialize parquet via ParquetDataset")

    def _boom_read_table(*args, **kwargs):
        raise AssertionError("validator must not read a table whole")

    real_parquet_file = pq.ParquetFile
    opens = []
    iter_batches_calls = []

    class _SpyParquetFile(real_parquet_file):
        def __init__(self, *args, **kwargs):
            opens.append(args)
            super().__init__(*args, **kwargs)

        def iter_batches(self, *args, **kwargs):
            iter_batches_calls.append((args, kwargs))
            return super().iter_batches(*args, **kwargs)

    with mock.patch.object(m.pq, 'ParquetDataset', _boom_dataset), \
            mock.patch.object(m.pq, 'read_table', _boom_read_table), \
            mock.patch.object(m.pq, 'ParquetFile', _SpyParquetFile):
        resp = _validate_supplied_train_table(generated, supplied, "w",
                                              TaskType.REGRESSION)

    assert resp.ok, f"unexpected errors: {resp.errors}"
    # Footer-only opens happened (once per table = 2 opens min).
    assert len(opens) >= 2, (
        f"expected footer-only ParquetFile opens, got {len(opens)}")
    assert iter_batches_calls, (
        "expected ParquetFile.iter_batches to be used for streaming")
    for args, kwargs in iter_batches_calls:
        assert kwargs.get('columns') == [
            'ENTITY1', 'TARGET1', 'w'
        ], (f"iter_batches must project only the columns the rules read, "
            f"got columns={kwargs.get('columns')}")


# Sanity: silence unused-import warnings for helpers referenced via
# closures or skipped pathways.
_ = Optional

# ---------------------------------------------------------------------------
# ingest_custom_train_table
# ---------------------------------------------------------------------------

_activity_module = sys.modules[
    'rest.backends.data_backend.activities.custom_train_table']


def _make_parquet_connector(tmp_path, table_name: str,
                            df: pd.DataFrame) -> tuple:
    """Write ``df`` as ``<tmp_path>/<table_name>.parquet`` and return
    ``(ParquetConnector, source_path)``.

    The connector is set up exactly the way the production code paths
    create one: a root_dir + a logical table_name. ``_get_path`` will
    return the on-disk parquet file path for that table.
    """
    from kumo.connector.parquet_connector import ParquetConnector
    src = tmp_path / f'{table_name}.parquet'
    df.to_parquet(src)
    connector = ParquetConnector(name='test', root_dir=str(tmp_path))
    return connector, str(src)


def test_ingest_custom_train_table_returns_dir_for_file_connector(tmp_path):
    """FileConnector branch: ingest must produce a *directory* at
    ``custom_train_table_path`` containing the customer's parquet data.

    Downstream ``materialize_training_table`` requires a directory of
    parquet parts. A single-file source must become a Parquet part inside that directory."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'TARGET': [0.1, 0.2, 0.3],
        'WEIGHT': [1.0, 2.0, 3.0],
    })
    connector, _ = _make_parquet_connector(tmp_path, 'custom_tt', df)

    spec = mock.Mock()
    spec.source_table = _custom_source('custom_tt')
    spec.connector_spec = mock.Mock()

    dest = str(tmp_path / 'dest_dir')

    with mock.patch.object(_activity_module, 'get_connector_from_spec',
                           return_value=connector):
        result_path, response = custom_train_table.ingest_custom_train_table(
            custom_train_table_spec=spec,
            custom_train_table_path=dest,
            deploy_config=mock.Mock(),
            id='test-job',
            task_type=TaskType.REGRESSION,
            query_type=QueryType.TEMPORAL,
            list_distinct_target_is_array=False,
        )

    assert result_path == dest
    assert response.ok, f"unexpected errors: {response.errors}"
    assert os.path.isdir(dest)
    assert any(name.endswith('.parquet') for name in os.listdir(dest))
    out = pd.read_parquet(dest)
    pd.testing.assert_frame_equal(
        out.sort_values('id').reset_index(drop=True),
        df.sort_values('id').reset_index(drop=True))


def test_ingest_custom_train_table_connector_error_returns_validation_error(
        tmp_path, caplog):
    """If ``_get_path`` raises on a FileConnector (e.g. mis-configured
    root_dir / missing file in connector internals), the activity must
    surface a ValidationError on the response rather than propagating
    the raw exception."""
    df = pd.DataFrame({'id': [1, 2, 3], 'TARGET': [0.1, 0.2, 0.3]})
    connector, _ = _make_parquet_connector(tmp_path, 'custom_tt', df)
    # Force the FileConnector path-resolution to raise. This simulates
    # downstream connector internals (e.g. listing a missing remote dir)
    # raising under the FileConnector branch.
    provider_secret = 'custom-table-provider-secret-must-not-escape'
    with mock.patch.object(connector, '_get_path',
                           side_effect=RuntimeError(provider_secret)):
        spec = mock.Mock()
        spec.source_table = _custom_source('custom_tt')
        spec.connector_spec = mock.Mock()
        dest = str(tmp_path / 'unused_dest_dir')

        with mock.patch.object(_activity_module, 'get_connector_from_spec',
                               return_value=connector):
            result_path, response = custom_train_table.ingest_custom_train_table(
                custom_train_table_spec=spec,
                custom_train_table_path=dest,
                deploy_config=mock.Mock(),
                id='test-job',
                task_type=TaskType.REGRESSION,
                query_type=QueryType.TEMPORAL,
                list_distinct_target_is_array=False,
            )

    # On the error path the activity returns ``custom_train_table_path``
    # unchanged (matches the warehouse-unload error branch).
    assert result_path == dest
    assert not response.ok
    assert any(e.title == "Failed to ingest custom training table"
               for e in response.errors)
    rendered_errors = ' '.join(e.message for e in response.errors)
    assert provider_secret not in rendered_errors
    assert provider_secret not in caplog.text


def test_custom_train_table_connector_resolution_error_is_sanitized(
        tmp_path, caplog):
    provider_secret = 'custom-table-resolution-token-must-not-escape'
    spec = mock.Mock()
    spec.source_table = _custom_source('custom_tt')
    spec.connector_spec = mock.Mock()
    dest = str(tmp_path / 'unused_dest_dir')

    with mock.patch.object(
            _activity_module,
            'get_connector_from_spec',
            side_effect=RuntimeError(provider_secret),
    ):
        result_path, response = custom_train_table.ingest_custom_train_table(
            custom_train_table_spec=spec,
            custom_train_table_path=dest,
            deploy_config=mock.Mock(),
            id='test-job',
            task_type=TaskType.REGRESSION,
            query_type=QueryType.TEMPORAL,
            list_distinct_target_is_array=False,
        )

    assert result_path == dest
    assert [error.message for error in response.errors
            ] == [custom_train_table.SAFE_TABLE_INGESTION_ERROR_MESSAGE]
    assert provider_secret not in caplog.text


def test_ingest_custom_train_table_warehouse_branch(tmp_path):
    """Warehouse-connector branch: ``export_table_to_blob_storage`` must
    be called with the table_name and destination path; the dest must be
    pre-cleaned before the export; and the synthesize helper must run
    AFTER the export (using the threaded task-shape kwargs).
    """
    dest = str(tmp_path / 'wh_dest_dir')
    # Drop a stale file inside dest to verify pre-clean.
    os.makedirs(dest, exist_ok=True)
    stale = os.path.join(dest, 'stale.parquet')
    pq.write_table(pa.table({'id': pa.array([99], type=pa.int64())}), stale)
    assert os.path.exists(stale)

    # Fake warehouse-style connector: NOT a FileConnector, but exposes
    # ``export_table_to_blob_storage``. Calls are recorded so we can
    # assert sequencing.
    calls: list = []

    class _FakeWarehouseConnector:
        def open(self):
            calls.append(('open', ))

        def export_table_to_blob_storage(self, table_name: str,
                                         out_path: str) -> None:
            calls.append(('export', table_name, out_path))
            # Write a part to the dest so the synthesize step has
            # something to inspect.
            os.makedirs(out_path, exist_ok=True)
            pq.write_table(
                pa.table({
                    'ENTITY': pa.array([0, 1, 2], type=pa.int64()),
                    'TARGET': pa.array([1, 2, 3], type=pa.int64()),
                }),
                os.path.join(out_path, 'part-00000.parquet'),
            )

    connector = _FakeWarehouseConnector()
    spec = mock.Mock()
    spec.source_table = _custom_source('wh_table')
    spec.connector_spec = mock.Mock()

    with mock.patch.object(_activity_module, 'get_connector_from_spec',
                           return_value=connector):
        # Use a task-shape that triggers TIMESTAMP synthesis (static
        # node-pred, no entity_time_col) so the post-export step runs.
        result_path, response = custom_train_table.ingest_custom_train_table(
            custom_train_table_spec=spec,
            custom_train_table_path=dest,
            deploy_config=mock.Mock(),
            id='test-job',
            task_type=TaskType.REGRESSION,
            query_type=QueryType.STATIC,
            list_distinct_target_is_array=False,
            entity_time_col=None,
        )

    assert result_path == dest
    assert response.ok, f"unexpected errors: {response.errors}"

    # Pre-clean must have removed the stale file before export ran.
    assert not os.path.exists(stale), (
        "warehouse branch did not pre-clean stale parquet before export")

    # Call order: open() then export_table_to_blob_storage().
    assert calls[0] == ('open', )
    assert calls[1] == ('export', 'wh_table', dest)

    # Synthesize must have run AFTER export — the part now has a
    # TIMESTAMP column appended.
    out = pd.read_parquet(dest)
    assert 'TIMESTAMP' in out.columns
    # And it must be zero-filled (sentinel for static node-pred).
    assert (out['TIMESTAMP'] == 0).all()


def test_custom_train_table_snowflake_uses_shared_reader(tmp_path):
    from kumoapi.typing import Dtype

    from kumo.connector.snowflake_connector import SnowflakeConnector
    connector = SnowflakeConnector('snowflake', 'account', 'warehouse',
                                   'database', 'schema')
    connector._context = mock.Mock()
    cursor = mock.Mock()
    source = pa.table({
        'ENTITY': pa.array([1, 2], type=pa.int64()),
        'TARGET': pa.array([None, 3], type=pa.int64()),
    })
    connector._context.cursor.return_value = cursor
    connector.get_query_metadata = mock.Mock(return_value=mock.Mock(cols=[]))
    for name in source.column_names:
        column = mock.Mock(dtype=Dtype.int64)
        column.name = name
        connector.get_query_metadata.return_value.cols.append(column)

    def prepare(reader, table, dest, **kwargs):
        from kumo_core.sources.types import ReadMode, StagedTable
        os.makedirs(dest, exist_ok=True)
        path = os.path.join(dest, 'part-00000.parquet')
        pq.write_table(source, path)
        return StagedTable(directory=dest, paths=(path, ),
                           version='custom-source',
                           source_size=os.path.getsize(path), num_rows=2,
                           mode_used=ReadMode.STAGED)

    connector.export_table_to_blob_storage = mock.Mock(
        side_effect=AssertionError('warehouse COPY must not run'))
    spec = mock.Mock(
        source_table=SnowflakeSourceTable(
            snowflake_connector_id=(
                'connector_00000000000000000000000000000000'),
            database='DATABASE',
            schema_name='SCHEMA',
            table='CUSTOM_TT',
        ),
        connector_spec=mock.Mock(),
    )
    destination = str(tmp_path / 'snowflake')

    with mock.patch.object(_activity_module, 'get_connector_from_spec',
                           return_value=connector), mock.patch(
                               'kumo_snowflake.SnowflakeSourceReader.prepare',
                               autospec=True, side_effect=prepare) as acquire:
        result_path, response = custom_train_table.ingest_custom_train_table(
            custom_train_table_spec=spec,
            custom_train_table_path=destination,
            deploy_config=mock.Mock(),
            id='test-job',
            task_type=TaskType.REGRESSION,
            query_type=QueryType.TEMPORAL,
            list_distinct_target_is_array=False,
        )

    assert response.ok
    assert result_path == destination
    connector.export_table_to_blob_storage.assert_not_called()
    assert acquire.call_args.kwargs['sql'] == (
        'SELECT * FROM "DATABASE"."SCHEMA"."CUSTOM_TT"')
    result = pq.read_table(os.path.join(destination, 'part-00000.parquet'))
    assert result.schema == source.schema
    assert result.to_pydict() == source.to_pydict()


def _write_one_row(_name, path, **_kwargs):
    """A stand-in export that lands what a real one would. A mock that writes
    nothing leaves an empty destination, which the emptiness check correctly
    reports as an empty source."""
    os.makedirs(path, exist_ok=True)
    pq.write_table(pa.table({'ENTITY': pa.array([1], type=pa.int64())}),
                   os.path.join(path, 'part-00000.parquet'))


@pytest.mark.parametrize('custom_sql', [False, True])
def test_custom_train_table_databricks_is_staged_like_any_warehouse(
        tmp_path, custom_sql):
    """Databricks takes the same warehouse branch as Snowflake and BigQuery:
    the connector exports to Kumo-owned parquet and the engine is handed a
    path, never a source spec or a credential file."""
    from kumo.connector.databricks_connector import DatabricksConnector
    connector = DatabricksConnector(
        'databricks',
        host='https://workspace.example.com',
        cluster_id='',
        warehouse_id='warehouse',
        catalog='catalog',
        schema_name='schema',
    )
    connector_id = 'connector_00000000000000000000000000000000'
    if custom_sql:
        source_table = CustomSQLSourceTable(
            connector_id=connector_id,
            connector_type=ConnectorType.DATABRICKS,
            table='custom_tt',
            sql='SELECT * FROM orders;',
        )
    else:
        source_table = DatabricksSourceTable(
            databricks_connector_id=connector_id,
            table='schema__kumo__custom_tt',
        )
    spec = mock.Mock(source_table=source_table, connector_spec=mock.Mock())
    destination = str(tmp_path / 'databricks')

    with (
            mock.patch.object(_activity_module, 'get_connector_from_spec',
                              return_value=connector),
            mock.patch.object(connector, 'open'),
            mock.patch.object(connector, 'close'),
            mock.patch.object(connector, 'export_table_read_only',
                              side_effect=_write_one_row) as export,
            mock.patch.object(_activity_module,
                              '_synthesize_sentinel_timestamp'),
            mock.patch('kumo.data.custom_sql.validate_connector_sql_scope'),
    ):
        path, response = custom_train_table.ingest_custom_train_table(
            custom_train_table_spec=spec,
            custom_train_table_path=destination,
            deploy_config=mock.Mock(),
            id='test-job',
            task_type=TaskType.REGRESSION,
            query_type=QueryType.TEMPORAL,
            list_distinct_target_is_array=False,
        )

    assert response.ok, [error.message for error in response.errors]
    assert path == destination
    assert export.call_args.args == (source_table.table, destination)
    if custom_sql:
        # The trailing terminator is stripped so the query embeds cleanly in
        # the connector's `FROM (<sql>)` position.
        assert export.call_args.kwargs == {'sql': 'SELECT * FROM orders'}
    else:
        assert export.call_args.kwargs == {}


def test_ingest_custom_train_table_unsupported_connector_raises(tmp_path):
    """Connectors that are neither FileConnector nor expose
    ``export_table_to_blob_storage`` must raise
    ``UnsupportedInDiskgraphError``."""
    from kumo.connector.pandas_connector import PandasConnector
    from statedb.utils.errors import UnsupportedInDiskgraphError

    df = pd.DataFrame({'id': [1, 2, 3]})
    connector = PandasConnector(df_dict={'custom_tt': df})
    # Sanity-check the contract assumption: a PandasConnector is neither
    # a FileConnector nor exposes ``export_table_to_blob_storage``.
    from kumo.connector.file_connector import FileConnector as _FC
    assert not isinstance(connector, _FC)
    assert not hasattr(connector, 'export_table_to_blob_storage')

    spec = mock.Mock()
    spec.source_table = _custom_source('custom_tt')
    spec.connector_spec = mock.Mock()
    dest = str(tmp_path / 'unused_dest_dir')

    with mock.patch.object(_activity_module, 'get_connector_from_spec',
                           return_value=connector):
        with pytest.raises(UnsupportedInDiskgraphError):
            custom_train_table.ingest_custom_train_table(
                custom_train_table_spec=spec,
                custom_train_table_path=dest,
                deploy_config=mock.Mock(),
                id='test-job',
                task_type=TaskType.REGRESSION,
                query_type=QueryType.TEMPORAL,
                list_distinct_target_is_array=False,
            )


def test_ingest_custom_train_table_idempotent_on_retry(tmp_path):
    """Temporal-retry safety: a second invocation against the same
    ``custom_train_table_path`` must succeed (not raise ``FileExistsError``)
    and produce identical content. Exercises both the pre-clean of stale
    files in dest and the ``overwrite=True`` on ``copy_file`` for the
    single-file FileConnector branch."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'TARGET': [0.1, 0.2, 0.3],
        'WEIGHT': [1.0, 2.0, 3.0],
    })
    connector, _ = _make_parquet_connector(tmp_path, 'custom_tt', df)

    spec = mock.Mock()
    spec.source_table = _custom_source('custom_tt')
    spec.connector_spec = mock.Mock()
    dest = str(tmp_path / 'dest_dir')

    with mock.patch.object(_activity_module, 'get_connector_from_spec',
                           return_value=connector):
        # First attempt — populates dest.
        result1, resp1 = custom_train_table.ingest_custom_train_table(
            custom_train_table_spec=spec,
            custom_train_table_path=dest,
            deploy_config=mock.Mock(),
            id='test-job',
            task_type=TaskType.REGRESSION,
            query_type=QueryType.TEMPORAL,
            list_distinct_target_is_array=False,
        )
        assert resp1.ok, f"first call errors: {resp1.errors}"
        first_out = pd.read_parquet(dest)

        # Drop a stale parquet under dest that the second attempt must
        # pre-clean. This simulates a partial-failure state from a prior
        # Temporal attempt.
        pq.write_table(
            pa.table({
                'id': pa.array([99], type=pa.int64()),
                'STALE_COL': pa.array(['x'], type=pa.string()),
            }),
            os.path.join(dest, 'part-stale.parquet'),
        )

        # Second attempt — must succeed (not raise FileExistsError, and
        # the stale parquet must be pre-cleaned so the dest matches the
        # first attempt's output exactly).
        result2, resp2 = custom_train_table.ingest_custom_train_table(
            custom_train_table_spec=spec,
            custom_train_table_path=dest,
            deploy_config=mock.Mock(),
            id='test-job',
            task_type=TaskType.REGRESSION,
            query_type=QueryType.TEMPORAL,
            list_distinct_target_is_array=False,
        )

    assert result1 == result2 == dest
    assert resp2.ok, f"second call errors: {resp2.errors}"
    second_out = pd.read_parquet(dest)
    pd.testing.assert_frame_equal(
        first_out.sort_values('id').reset_index(drop=True),
        second_out.sort_values('id').reset_index(drop=True))
    # Stale file must be gone after the pre-clean.
    assert not os.path.exists(os.path.join(dest, 'part-stale.parquet'))


# ---------------------------------------------------------------------------
# _synthesize_sentinel_timestamp
# ---------------------------------------------------------------------------


def _write_synth_parquet(path: str, columns: dict) -> None:
    """Write ``columns`` to ``path`` as a single parquet file."""
    pq.write_table(pa.table(columns), path)


def _schema_names(parquet_path: str) -> list:
    return pq.read_schema(parquet_path).names


def test_synthesize_sentinel_timestamp_temporal_is_noop(tmp_path):
    """Temporal queries already carry real ``TIMESTAMP``; synthesis must
    not touch the file even if for some reason TIMESTAMP is missing."""
    parquet_dir = str(tmp_path / "tt")
    os.makedirs(parquet_dir)
    part = os.path.join(parquet_dir, "part-00000.parquet")
    _write_synth_parquet(
        part, {
            "ENTITY": pa.array([0, 1, 2], type=pa.int64()),
            "TARGET": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        })
    original_bytes = open(part, "rb").read()

    custom_train_table._synthesize_sentinel_timestamp(
        parquet_dir,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
        list_distinct_target_is_array=False,
        entity_time_col=None,
    )

    # Temporal → no-op: file must be byte-identical.
    assert open(part, "rb").read() == original_bytes
    assert "TIMESTAMP" not in _schema_names(part)


def test_synthesize_sentinel_timestamp_static_node_pred_no_time_synth(tmp_path):
    """Static node-pred without entity_time_col → synthesize zero-filled
    int64 ``TIMESTAMP``."""
    parquet_dir = str(tmp_path / "tt")
    os.makedirs(parquet_dir)
    part = os.path.join(parquet_dir, "part-00000.parquet")
    _write_synth_parquet(
        part, {
            "ENTITY": pa.array([10, 20, 30], type=pa.int64()),
            "TARGET": pa.array([0.5, 0.6, 0.7], type=pa.float64()),
        })

    custom_train_table._synthesize_sentinel_timestamp(
        parquet_dir,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.STATIC,
        list_distinct_target_is_array=False,
        entity_time_col=None,
    )

    schema = pq.read_schema(part)
    assert "TIMESTAMP" in schema.names
    assert schema.field("TIMESTAMP").type == pa.int64()
    table = pq.read_table(part)
    assert table.column("TIMESTAMP").to_pylist() == [0, 0, 0]
    # Other columns must be preserved.
    assert table.column("ENTITY").to_pylist() == [10, 20, 30]


def test_synthesize_sentinel_timestamp_static_lp_array_synth(tmp_path):
    """Static LP with ``list_distinct_target_is_array=True`` → synthesize
    zero-filled ``TIMESTAMP`` (matches DG's static-LP-array emission)."""
    parquet_dir = str(tmp_path / "tt")
    os.makedirs(parquet_dir)
    part = os.path.join(parquet_dir, "part-00000.parquet")
    _write_synth_parquet(
        part, {
            "ENTITY":
            pa.array([0, 1, 2], type=pa.int64()),
            "TARGET":
            pa.array([["a", "b"], ["c"], ["d", "e"]], type=pa.list_(
                pa.string())),
        })

    custom_train_table._synthesize_sentinel_timestamp(
        parquet_dir,
        task_type=TaskType.STATIC_LINK_PREDICTION,
        query_type=QueryType.STATIC,
        list_distinct_target_is_array=True,
        entity_time_col=None,
    )

    table = pq.read_table(part)
    assert "TIMESTAMP" in table.schema.names
    assert table.schema.field("TIMESTAMP").type == pa.int64()
    assert table.column("TIMESTAMP").to_pylist() == [0, 0, 0]


def test_synthesize_sentinel_timestamp_already_present_is_noop(tmp_path):
    """Idempotent re-ingest: if a part-*.parquet already has
    ``TIMESTAMP`` (e.g. customer's source includes it, or a previous
    Temporal attempt synthesized it), the helper must skip without
    rewriting the file."""
    parquet_dir = str(tmp_path / "tt")
    os.makedirs(parquet_dir)
    part = os.path.join(parquet_dir, "part-00000.parquet")
    _write_synth_parquet(
        part, {
            "ENTITY": pa.array([0, 1], type=pa.int64()),
            "TARGET": pa.array([0.1, 0.2], type=pa.float64()),
            "TIMESTAMP": pa.array([42, 43], type=pa.int64()),
        })
    original_bytes = open(part, "rb").read()

    custom_train_table._synthesize_sentinel_timestamp(
        parquet_dir,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.STATIC,
        list_distinct_target_is_array=False,
        entity_time_col=None,
    )

    # Byte-identical: no rewrite happened. Crucially, the existing
    # TIMESTAMP values (42, 43) are NOT overwritten by zeros.
    assert open(part, "rb").read() == original_bytes
    assert pq.read_table(part).column("TIMESTAMP").to_pylist() == [42, 43]


def test_synthesize_sentinel_timestamp_multi_part_directory(tmp_path):
    """Multi-part directory layout (Spark/Hive style): synthesize must
    process every ``part-*.parquet`` independently, skipping files that
    already have ``TIMESTAMP`` and rewriting only those that don't."""
    parquet_dir = str(tmp_path / "tt")
    os.makedirs(parquet_dir)
    part0 = os.path.join(parquet_dir, "part-00000.parquet")
    part1 = os.path.join(parquet_dir, "part-00001.parquet")
    part2 = os.path.join(parquet_dir, "part-00002.parquet")
    _write_synth_parquet(
        part0, {
            "ENTITY": pa.array([0, 1], type=pa.int64()),
            "TARGET": pa.array([0.1, 0.2], type=pa.float64()),
        })
    _write_synth_parquet(
        part1, {
            "ENTITY": pa.array([2, 3], type=pa.int64()),
            "TARGET": pa.array([0.3, 0.4], type=pa.float64()),
        })
    # Pre-populate part2 with TIMESTAMP — must remain untouched.
    _write_synth_parquet(
        part2, {
            "ENTITY": pa.array([4, 5], type=pa.int64()),
            "TARGET": pa.array([0.5, 0.6], type=pa.float64()),
            "TIMESTAMP": pa.array([100, 200], type=pa.int64()),
        })
    part2_bytes = open(part2, "rb").read()

    custom_train_table._synthesize_sentinel_timestamp(
        parquet_dir,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.STATIC,
        list_distinct_target_is_array=False,
        entity_time_col=None,
    )

    assert pq.read_table(part0).column("TIMESTAMP").to_pylist() == [0, 0]
    assert pq.read_table(part1).column("TIMESTAMP").to_pylist() == [0, 0]
    # part2 was not rewritten — original values preserved byte-for-byte.
    assert open(part2, "rb").read() == part2_bytes
    assert pq.read_table(part2).column("TIMESTAMP").to_pylist() == [100, 200]


def test_synthesize_sentinel_timestamp_atomicity_on_failure(tmp_path):
    """If the parquet write to the tmp path raises mid-stream, the
    original parquet at ``parquet_path`` MUST be byte-identical and no
    stale ``.tmp`` file is left behind."""
    from rest.backends.data_backend.activities import \
        custom_train_table as _ct_mod
    parquet_dir = str(tmp_path / "tt")
    os.makedirs(parquet_dir)
    part = os.path.join(parquet_dir, "part-00000.parquet")
    _write_synth_parquet(
        part, {
            "ENTITY": pa.array([0, 1, 2], type=pa.int64()),
            "TARGET": pa.array([0.1, 0.2, 0.3], type=pa.float64()),
        })
    original_bytes = open(part, "rb").read()

    # The streaming helper writes batches through pq.ParquetWriter — patch
    # write_table on the writer instance so we exercise the cleanup branch
    # without depending on which top-level pyarrow API got called.
    real_writer_cls = _ct_mod.pq.ParquetWriter

    class _BoomWriter:
        def __init__(self, dest, *a, **kw):
            # Drop a partial tmp so we can verify the cleanup branch.
            if isinstance(dest, str) and dest.endswith(".tmp"):
                with open(dest, "wb") as fh:
                    fh.write(b"PARTIAL")
            self._dest = dest

        def write_table(self, *a, **kw):
            raise IOError("simulated mid-write crash")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    assert real_writer_cls is not None  # silence unused-binding lint
    with mock.patch.object(_ct_mod.pq, "ParquetWriter", _BoomWriter):
        with pytest.raises(IOError, match="simulated mid-write crash"):
            custom_train_table._synthesize_sentinel_timestamp(
                parquet_dir,
                task_type=TaskType.REGRESSION,
                query_type=QueryType.STATIC,
                list_distinct_target_is_array=False,
                entity_time_col=None,
            )

    # Original parquet untouched.
    assert open(part, "rb").read() == original_bytes
    # Stale tmp must have been cleaned up by the helper.
    assert not os.path.exists(part + ".tmp")


def test_synthesize_sentinel_timestamp_static_lp_non_array_is_noop(tmp_path):
    """Deferred-case decision pin: static LP non-array (DG would emit
    ``PSEUDO_TIMESTAMP``, a different literal name) MUST be a no-op
    under approach B's current scope. The customer's parquet will lack
    ``PSEUDO_TIMESTAMP`` and downstream consumers will fail naturally
    — we explicitly do not synthesize ``PSEUDO_TIMESTAMP`` here."""
    parquet_dir = str(tmp_path / "tt")
    os.makedirs(parquet_dir)
    part = os.path.join(parquet_dir, "part-00000.parquet")
    _write_synth_parquet(
        part, {
            "ENTITY": pa.array([0, 1, 2], type=pa.int64()),
            "TARGET": pa.array([0, 1, 2], type=pa.int64()),
        })
    original_bytes = open(part, "rb").read()

    custom_train_table._synthesize_sentinel_timestamp(
        parquet_dir,
        task_type=TaskType.STATIC_LINK_PREDICTION,
        query_type=QueryType.STATIC,
        list_distinct_target_is_array=False,
        entity_time_col=None,
    )

    # No-op: file must be byte-identical, no TIMESTAMP and no
    # PSEUDO_TIMESTAMP appended.
    assert open(part, "rb").read() == original_bytes
    schema_names = _schema_names(part)
    assert "TIMESTAMP" not in schema_names
    assert "PSEUDO_TIMESTAMP" not in schema_names


def test_synthesize_sentinel_timestamp_static_np_with_entity_time_is_noop(
        tmp_path):
    """Static node-pred WITH ``entity_time_col`` set: the customer
    supplies real per-row time via the entity column; synthesis must
    not overwrite that with zeros."""
    parquet_dir = str(tmp_path / "tt")
    os.makedirs(parquet_dir)
    part = os.path.join(parquet_dir, "part-00000.parquet")
    _write_synth_parquet(
        part, {
            "ENTITY": pa.array([0, 1, 2], type=pa.int64()),
            "TARGET": pa.array([0.1, 0.2, 0.3], type=pa.float64()),
        })
    original_bytes = open(part, "rb").read()

    custom_train_table._synthesize_sentinel_timestamp(
        parquet_dir,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.STATIC,
        list_distinct_target_is_array=False,
        entity_time_col="event_ts",
    )

    # No-op: file untouched, TIMESTAMP not synthesized.
    assert open(part, "rb").read() == original_bytes
    assert "TIMESTAMP" not in _schema_names(part)


def test_filesystem_and_path_routes_s3_through_arrow_cloud_filesystem(
        monkeypatch):
    # S3 / GCS return the credentialed choke-point filesystem plus a
    # scheme-stripped path; the tuple shape callers expect is preserved.
    from rest.backends.data_backend.activities import custom_train_table as ctt

    seen = {}
    sentinel_fs = object()

    def fake_acf(path):
        seen['path'] = path
        return sentinel_fs

    monkeypatch.setattr(ctt, 'arrow_cloud_filesystem', fake_acf)

    fs, bare = ctt._filesystem_and_path('s3://bucket/key/f.parquet')
    assert fs is sentinel_fs
    assert bare == 'bucket/key/f.parquet'
    assert seen['path'] == 's3://bucket/key/f.parquet'

    fs_g, bare_g = ctt._filesystem_and_path('gs://bkt/k/f.parquet')
    assert fs_g is sentinel_fs
    assert bare_g == 'bkt/k/f.parquet'


def test_filesystem_and_path_local_returns_none(monkeypatch):
    from rest.backends.data_backend.activities import custom_train_table as ctt

    def boom(path):
        raise AssertionError('local paths must not use the cloud choke point')

    monkeypatch.setattr(ctt, 'arrow_cloud_filesystem', boom)
    fs, bare = ctt._filesystem_and_path('/tmp/x.parquet')
    assert fs is None
    assert bare == '/tmp/x.parquet'


def test_filesystem_and_path_adls_unchanged(monkeypatch):
    # ADLS keeps using FileSystem.from_uri, not the S3 / GCS choke point.
    from rest.backends.data_backend.activities import custom_train_table as ctt

    def boom(path):
        raise AssertionError('adls must keep using FileSystem.from_uri')

    class _FakeFS:
        @staticmethod
        def from_uri(path):
            return ('adls-fs', 'container/key')

    monkeypatch.setattr(ctt, 'arrow_cloud_filesystem', boom)
    monkeypatch.setattr(ctt, 'FileSystem', _FakeFS)
    fs, bare = ctt._filesystem_and_path(
        'abfss://c@acct.dfs.core.windows.net/key')
    assert fs == 'adls-fs'
    assert bare == 'container/key'


# Custom SQL against an exporter that cannot take a query ######################
#
# The cell nothing else covers. Every other Custom SQL case here uses
# Databricks -- the one connector whose exporter gained a ``sql`` parameter --
# and every non-Databricks case uses a source shape with no ``sql`` attribute,
# so the branch that forwards the kwarg never meets a callee that would reject
# it. Both connectors below are real, not mocks: a ``MagicMock`` accepts any
# keyword, which is exactly how forwarding ``sql`` to a signature without it
# would go unnoticed.


def _custom_sql_source(connector_id: str) -> CustomSQLSourceTable:
    return CustomSQLSourceTable(
        connector_id=connector_id,
        sql='SELECT ENTITY, TARGET FROM orders',
        table='custom_sql_tt',
    )


def test_custom_sql_on_snowflake_is_refused_not_forwarded(tmp_path):
    """Snowflake's ``export_table_read_only(source_name, destination, cols)``
    has no ``sql``. Forwarding one raises ``TypeError`` inside the broad
    ``except Exception``, which reports it as the generic sanitized ingest
    failure and tells the user nothing."""
    from kumo.connector.snowflake_connector import SnowflakeConnector
    connector = SnowflakeConnector('snowflake', 'account', 'warehouse',
                                   'database', 'schema')
    connector._context = mock.Mock()
    # The capability this test exists to protect. Asserted on the declared
    # flag, not on `inspect.signature`: `@blocking_method()` erases every
    # export to `(*args, **kwargs)`, so the signature is identical for
    # Snowflake and Databricks and cannot tell them apart.
    assert connector.exports_custom_sql is False

    spec = mock.Mock(
        source_table=_custom_sql_source(
            'connector_00000000000000000000000000000000'),
        connector_spec=mock.Mock(),
    )

    with mock.patch.object(_activity_module, 'get_connector_from_spec',
                           return_value=connector):
        with pytest.raises(UnsupportedInDiskgraphError,
                           match='SnowflakeConnector cannot export a query'):
            custom_train_table.ingest_custom_train_table(
                custom_train_table_spec=spec,
                custom_train_table_path=str(tmp_path / 'snowflake'),
                deploy_config=mock.Mock(),
                id='test-job',
                task_type=TaskType.REGRESSION,
                query_type=QueryType.TEMPORAL,
                list_distinct_target_is_array=False,
            )


def test_custom_sql_on_bigquery_is_refused_not_forwarded(tmp_path):
    """BigQuery reaches the same branch through ``export_table_to_blob_storage
    (table_name, dest_path, columns)``, which also has no ``sql``."""
    from kumo.connector.bigquery_connector import BigQueryConnector
    connector = BigQueryConnector('bigquery', 'project', 'dataset')
    connector._context = mock.Mock()
    assert not hasattr(connector, 'export_table_read_only')
    assert connector.exports_custom_sql is False

    spec = mock.Mock(
        source_table=_custom_sql_source(
            'connector_00000000000000000000000000000001'),
        connector_spec=mock.Mock(),
    )

    with mock.patch.object(_activity_module, 'get_connector_from_spec',
                           return_value=connector):
        with pytest.raises(UnsupportedInDiskgraphError,
                           match='BigQueryConnector cannot export a query'):
            custom_train_table.ingest_custom_train_table(
                custom_train_table_spec=spec,
                custom_train_table_path=str(tmp_path / 'bigquery'),
                deploy_config=mock.Mock(),
                id='test-job',
                task_type=TaskType.REGRESSION,
                query_type=QueryType.TEMPORAL,
                list_distinct_target_is_array=False,
            )


@pytest.mark.parametrize('has_file', [False, True])
def test_an_empty_custom_training_table_is_named_not_mismatched(
        tmp_path, has_file):
    """The training twin of `EmptyPredictionTableError`. An empty source
    otherwise reaches `_validate_weighted_tt_pyarrow` as a row-count
    mismatch, and says nothing at all once
    `CUSTOM_TT_ALLOW_ROW_COUNT_MISMATCH` is set."""
    from kumo.connector.snowflake_connector import SnowflakeConnector
    connector = SnowflakeConnector('snowflake', 'account', 'warehouse',
                                   'database', 'schema')
    connector._context = mock.Mock()
    destination = str(tmp_path / 'empty')

    def write_schema_only_parquet(_name, path, **_kwargs):
        if not has_file:
            from kumo.connector.connector import ExportMetrics
            return ExportMetrics(num_rows=0, source_size=0)
        # What a streamed read of a zero-row result actually lands: one
        # part with the schema and no rows.
        os.makedirs(path, exist_ok=True)
        pq.write_table(
            pa.table({
                'ENTITY': pa.array([], type=pa.int64()),
                'TARGET': pa.array([], type=pa.int64()),
            }), os.path.join(path, 'part-00000.parquet'))

    spec = mock.Mock(
        source_table=SnowflakeSourceTable(
            snowflake_connector_id=(
                'connector_00000000000000000000000000000000'),
            database='DATABASE',
            schema_name='SCHEMA',
            table='CUSTOM_TT',
        ),
        connector_spec=mock.Mock(),
    )

    with (
            mock.patch.object(_activity_module, 'get_connector_from_spec',
                              return_value=connector),
            mock.patch.object(connector, 'open'),
            mock.patch.object(connector, 'close'),
            mock.patch.object(connector, 'export_table_read_only',
                              side_effect=write_schema_only_parquet),
    ):
        _path, response = custom_train_table.ingest_custom_train_table(
            custom_train_table_spec=spec,
            custom_train_table_path=destination,
            deploy_config=mock.Mock(),
            id='test-job',
            task_type=TaskType.REGRESSION,
            query_type=QueryType.TEMPORAL,
            list_distinct_target_is_array=False,
        )

    assert not response.ok
    assert response.errors[0].title == 'Custom training table is empty'
    assert 'CUSTOM_TT' in response.errors[0].message
    # Not the generic sanitized text: an empty result is the user's, and
    # nothing about it needs hiding.
    assert (response.errors[0].message
            != custom_train_table.SAFE_TABLE_INGESTION_ERROR_MESSAGE)


# The two the hosted side used to be alone in skipping #######################


def test_a_split_column_the_supplied_table_lacks_is_refused_here_too(tmp_path):
    """Before this, the hosted presence check was never given the split
    column: a source lacking it was accepted and ``build_split_spec`` then
    resolved a ``ManualSplit`` on a column that is not there."""
    resp = _validate_supplied_train_table(_generated(tmp_path),
                                          _supplied(tmp_path), None,
                                          TaskType.REGRESSION, 'SPLIT')

    assert _titles(resp) == ['Column not found']
    assert "'SPLIT'" in resp.errors[0].message


def test_a_split_column_outside_the_encoding_is_refused_here_too(tmp_path):
    """The values, not only the name and the type: the hosted side read no
    split column at all, so a column of 'train'/'val' put every row in no
    split and the run reported success."""
    supplied = _supplied(
        tmp_path, SPLIT=pa.array(['train', 'val', 'test'], type=pa.string()))

    resp = _validate_supplied_train_table(_generated(tmp_path), supplied, None,
                                          TaskType.REGRESSION, 'SPLIT')

    assert _titles(resp) == ['Split column is not the encoding that is read']


def test_a_dictionary_encoded_column_is_gated_on_what_the_file_stores(tmp_path):
    """A connector export may land a dictionary-encoded ENTITY1 that pyarrow
    reports as ``int64``. Read as reported it passes every gate, matches the
    generated ``int64`` exactly, and DiskGraph then drops every row."""
    supplied = str(tmp_path / 'supplied')
    _write_parquet(
        supplied, {
            'ENTITY1': pa.array([1, 2, 3], type=pa.int64()).dictionary_encode(),
            'TARGET1': pa.array([0.1, 0.2, 0.3], type=pa.float64()),
        })

    resp = _validate_supplied_train_table(_generated(tmp_path), supplied, None,
                                          TaskType.REGRESSION)

    assert _titles(resp) == ['Column type cannot be read']
    assert 'ENTITY1' in resp.errors[0].message


def test_the_generated_anchor_is_only_read_when_there_is_one_to_compare(
        tmp_path, monkeypatch):
    """Only an integer anchor is compared against it, so a table without one
    does not walk every row-group footer of the generated table over the
    object store."""
    from rest.backends.data_backend.activities import custom_train_table as ctt

    def boom(files):
        raise AssertionError('the generated anchor was read with nothing to '
                             'compare it against')

    monkeypatch.setattr(ctt, '_anchor_span_ns', boom)

    resp = _validate_supplied_train_table(_generated(tmp_path),
                                          _supplied(tmp_path), None,
                                          TaskType.REGRESSION)

    assert resp.ok, f'unexpected errors: {resp.errors}'


def test_generated_parts_with_no_anchor_have_an_unknown_span(tmp_path):
    generated = tmp_path / 'generated-parts'
    generated.mkdir()
    nanos = [
        1_700_000_000_000_000_000, 1_700_000_001_000_000_000,
        1_700_000_002_000_000_000
    ]
    pq.write_table(
        pa.table({
            'ENTITY1': [1, 2],
            'TIMESTAMP': pa.array(nanos[:2], type=pa.timestamp('ns')),
            'TARGET1': [0.1, 0.2],
        }), generated / 'part-00000.parquet')
    pq.write_table(pa.table({
        'ENTITY1': [3],
        'TARGET1': [0.3],
    }), generated / 'part-00001.parquet')
    supplied = _supplied(
        tmp_path,
        TIMESTAMP=pa.array([value // 1_000_000_000 for value in nanos],
                           type=pa.int64()))

    response = _validate_supplied_train_table(str(generated), supplied, None,
                                              TaskType.REGRESSION)

    assert response.ok


@pytest.mark.parametrize(('second', 'title'), [
    ({
        'ENTITY1': pa.array([4], pa.int64())
    }, 'Column missing in a part of the supplied table'),
    ({
        'ENTITY1': pa.array(['4'], pa.string()),
        'TARGET1': pa.array([0.4], pa.float64()),
    }, 'Column type differs between parts'),
])
def test_every_part_is_gated_and_not_only_the_first(tmp_path, second, title):
    """The schema is read off part zero and the rows off every part, and a
    projected read of a part that lacks a requested column omits it silently
    rather than failing -- so a well-formed part zero would otherwise carry
    the rest of the table past the whole check.

    Row-count-neutral on purpose: part-1 adds a 4th row, so a generated table
    of 3 would fail on row count first and never reach the schema gate. Here
    the generated table has 4 rows too, so the only thing that can fail is
    what part-1 diverges on.
    """
    generated = _generated(tmp_path, ENTITY1=pa.array([1, 2, 3, 4], pa.int64()),
                           TARGET1=pa.array([0.1, 0.2, 0.3, 0.4], pa.float64()))
    supplied = _supplied(tmp_path)
    pq.write_table(pa.table(second), os.path.join(supplied, 'part-1.parquet'))

    resp = _validate_supplied_train_table(generated, supplied, None,
                                          TaskType.REGRESSION)

    assert _titles(resp) == [title]


def test_a_registration_is_gated_with_no_generated_table_to_compare(tmp_path):
    """A registered table stands in for nothing, so the correspondence rules
    are skipped and every other one still runs. Whether the anchor is real is
    the caller's to say: reading it off the supplied table is circular, since
    it decides whether that table needs one."""
    supplied = _supplied(tmp_path)

    resp = _validate_supplied_train_table(None, supplied, None,
                                          TaskType.REGRESSION, anchored=False)
    assert resp.ok, f'unexpected errors: {resp.errors}'

    resp = _validate_supplied_train_table(None, supplied, None,
                                          TaskType.REGRESSION)
    assert _titles(resp) == ['Required column not found']
