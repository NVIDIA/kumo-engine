# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import os.path as osp
import pickle
import threading
from typing import List, Optional

import pandas as pd
import pytest
from kumoapi.typing import Dtype, Stype
from pandas.testing import assert_frame_equal

from common_utils.error import PublicException
from kumo.config.graph import TableMetadataConfig
from kumo.connector.connector import (
    Connector,
    SourceTableNotFoundError,
    SourceTableUnavailableError,
)
from kumo.connector.csv_connector import CSVConnector
from kumo.connector.databricks_connector import DatabricksConnector
from kumo.connector.snowflake_connector import SnowflakeConnector
from kumo.connector.utils import cast_df_to_dtypes
from kumo.testing import withDatabricksCredentials, withSnowflakeCredentials
from service_framework.deploy_env import DATABRICKS_HOST

num_open_calls = 0
num_close_calls = 0
num_read_sample_calls = 0
num_read_calls = 0
num_stats_calls = 0


@pytest.fixture(autouse=True)
def reset_my_connector_call_counters():
    global num_open_calls, num_close_calls
    global num_read_sample_calls, num_read_calls, num_stats_calls
    num_open_calls = 0
    num_close_calls = 0
    num_read_sample_calls = 0
    num_read_calls = 0
    num_stats_calls = 0


class MyConnector(Connector):
    def _open(self, credentials):
        global num_open_calls
        num_open_calls += 1
        return True

    def _get_connector_type(self) -> str:
        return "my_connector"

    def _close(self, context):
        global num_close_calls
        num_close_calls += 1

    def _read_table(self, context, table_name, *args, **kwargs):
        global num_read_calls
        num_read_calls += 1

    def _read_table_chunk(self, context, table_name, *args, **kwargs):
        global num_read_calls
        num_read_calls += 1

    def _read_table_sample(self, context, table_name, *args, **kwargs):
        global num_read_sample_calls
        num_read_sample_calls += 1
        return None

    def list_tables(self, schema_name: Optional[str] = None):
        return ['table1', 'table2']

    def _get_table_metadata(
        self,
        table_names: List[str],
        skip_cache: bool = False,
        skip_inferral: bool = False,
        schema_name: Optional[str] = None,
        ignore_errors: bool = False,
    ):
        if table_names is None:
            table_names = self.list_tables()
        return [
            TableMetadataConfig(table_name=table_name, source_name=table_name,
                                data_source=self.name)
            for table_name in table_names
        ]

    def _get_table_stats(self, table_config, stats):
        global num_stats_calls
        num_stats_calls += 1

    def _get_table_source_size(self, table_name: str,
                               data_version: str | None = None):
        pass

    def _get_table_num_rows(self, table_name: str,
                            data_version: str | None = None):
        pass


@pytest.mark.parametrize(
    'error_type', [SourceTableNotFoundError, SourceTableUnavailableError])
def test_source_table_error_roundtrip(error_type):
    original = error_type('missing_table', 'warehouse')

    payload = original.to_dict()
    restored = PublicException.from_dict(payload)

    assert payload['kwargs'] == {
        'table_name': 'missing_table',
        'connector_name': 'warehouse',
    }
    assert type(restored) is error_type
    assert restored.message == original.message


@pytest.mark.parametrize(
    'error_type', [SourceTableNotFoundError, SourceTableUnavailableError])
@pytest.mark.parametrize(
    ('skip_cache', 'skip_inferral'),
    [(True, False), (False, True)],
)
def test_ignore_errors_handles_metadata_fetch_failures(monkeypatch, error_type,
                                                       skip_cache,
                                                       skip_inferral):
    connector = MyConnector(name='my_connector')

    def fail_metadata(*args, **kwargs):
        raise error_type('missing_table', connector.name)

    monkeypatch.setattr(connector, '_get_table_metadata', fail_metadata)

    assert connector.get_table_metadata(
        ['missing_table'],
        skip_cache=skip_cache,
        skip_inferral=skip_inferral,
        ignore_errors=True,
    ) == []


def test_ignore_errors_does_not_hide_unrelated_metadata_failure(monkeypatch):
    connector = MyConnector(name='my_connector')

    def fail_metadata(*args, **kwargs):
        raise RuntimeError('metadata service failed')

    monkeypatch.setattr(connector, '_get_table_metadata', fail_metadata)

    with pytest.raises(RuntimeError, match='metadata service failed'):
        connector.get_table_metadata(
            ['table1'],
            skip_cache=True,
            ignore_errors=True,
        )


def test_default_metadata_fetch_only_ignores_expected_table_errors(monkeypatch):
    connector = MyConnector(name='my_connector')
    errors = iter([
        SourceTableNotFoundError('missing_table', connector.name),
        RuntimeError('metadata service failed'),
    ])

    def fail_metadata(*args, **kwargs):
        raise next(errors)

    monkeypatch.setattr(connector, '_get_table_cols_metadata', fail_metadata)

    assert Connector._get_table_metadata(
        connector,
        ['missing_table'],
        ignore_errors=True,
    ) == []
    with pytest.raises(RuntimeError, match='metadata service failed'):
        Connector._get_table_metadata(
            connector,
            ['broken_table'],
            ignore_errors=True,
        )


def test_connector_mock():
    connector = MyConnector(name='my_connector')

    connector.read_table('dummy')
    connector.read_table('dummy', num_rows=10)
    assert len(connector.get_table_metadata()) == 2
    assert connector.name == 'my_connector'
    del connector

    assert num_open_calls == 1
    assert num_read_calls == 2
    assert num_close_calls == 1

    with MyConnector(name='my_connector') as connector:
        connector.read_table('dummy')
        connector.read_table('dummy', num_rows=10)

    assert num_open_calls == 2
    assert num_read_calls == 4
    assert num_close_calls == 2

    @MyConnector(name='my_connector')
    def read_table(connector):
        connector.read_table('dummy')
        connector.read_table('dummy', num_rows=10)

    read_table()  # pylint: disable=no-value-for-parameter

    assert num_open_calls == 3
    assert num_read_calls == 6
    assert num_close_calls == 3


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_csv_and_snowflake_connector():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'csv_data', 'FINANCIAL')
    connector1 = CSVConnector('csv', root_dir)

    connector2 = SnowflakeConnector('snowflake', account='xva19026',
                                    warehouse='WH_XS', database='KUMO',
                                    schema_name='FINANCIAL')

    pkey = 'LOAN_ID'
    cols = [
        ('ACCOUNT_ID', Stype.ID, Dtype.int),
        ('DATE', Stype.timestamp, Dtype.date),
        ('AMOUNT', Stype.numerical, Dtype.float),
        ('PAYMENTS', Stype.numerical, Dtype.float),
        ('STATUS', Stype.categorical, Dtype.string),
    ]

    df1 = connector1.read_table('LOAN', pkey, cols).sort_index()
    df2 = connector2.read_table('LOAN', pkey, cols).sort_index()
    # pandas-2 infers a narrower int dtype for the pkey index from one source
    # (int16) than the other (int64). check_dtype=False only relaxes column
    # dtypes, so also relax the index dtype to compare values, not dtypes.
    pd.testing.assert_frame_equal(df1, df2, check_dtype=False,
                                  check_index_type=False)


@withDatabricksCredentials
def test_databricks_connector():
    connector = DatabricksConnector(
        'databricks',
        host=DATABRICKS_HOST.parse_value(),
        cluster_id='0626-191119-75431625',
        warehouse_id='82d859e4a6b283a0',
        catalog='hive_mapped_uc',
        schema_name='hm',
    )

    pkey = 'customer_id'

    df = connector.read_table('hm.customers', pkey=pkey, num_rows=100)
    assert len(df) == 100
    for col in [
            'FN', 'Active', 'club_member_status', 'fashion_news_frequency',
            'age', 'postal_code'
    ]:
        assert col in df
    assert df.index.name == 'customer_id'


def test_cast_df_to_dtypes_temporal_cast_is_best_effort():
    input_df = pd.DataFrame({
        'string_date': [
            '2000-09-01 00:00:00',
            pd.NA,
            'not-a-date',
        ],
    })

    got_df = cast_df_to_dtypes(
        input_df,
        {'string_date': Dtype.date},
        # Temporal casts use DiskGraph's best-effort contract and do not
        # consult the numeric-cast error threshold.
        error_threshold=0.0,
    )

    expect_df = pd.DataFrame({
        'string_date': [
            datetime.datetime(2000, 9, 1),
            pd.NaT,
            pd.NaT,
        ],
    })
    assert_frame_equal(got_df, expect_df)


def test_dont_drop_all_na():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'csv_data', 'HOME_CREDIT_DEFAULT_RISK')
    connector = CSVConnector('csv', root_dir, na_values='?')
    table_config = connector.get_table_metadata(['BUREAU'])[0]
    assert 'AMT_ANNUITY' in [col.name for col in table_config.cols]


class _WeakrefableSession:
    r"""Dummy context object that supports weakref. Needed to exercise
    the ``weakref.finalize(self._context, ...)`` branch in ``open()``.
    """


class _RaceConnector(MyConnector):
    r"""Like MyConnector but ``_open`` returns a weakref-able object so
    ``weakref.finalize(context, ...)`` is actually invoked.
    """
    def _open(self, credentials):
        return _WeakrefableSession()


class _RetryConnector(MyConnector):
    """Stateful connector for open-validation and cleanup fault injection."""
    def __init__(self, *, validation_errors=None, close_errors=None):
        super().__init__(name='retry_test')
        self.open_calls = 0
        self.close_calls = 0
        self.validation_calls = 0
        self.validation_errors = list(validation_errors or [])
        self.close_errors = list(close_errors or [])

    def _open(self, credentials):
        del credentials
        self.open_calls += 1
        return _WeakrefableSession()

    def _close(self, context):
        assert isinstance(context, _WeakrefableSession)
        self.close_calls += 1
        if self.close_errors:
            error = self.close_errors.pop(0)
            if error is not None:
                raise error

    def validate_permissions(self):
        self.validation_calls += 1
        if self.validation_errors:
            error = self.validation_errors.pop(0)
            if error is not None:
                raise error


def test_connector_transient_validation_failure_reopens_fresh_session():
    transient_error = RuntimeError('transient provider failure')
    connector = _RetryConnector(validation_errors=[transient_error, None])

    with pytest.raises(RuntimeError) as caught:
        connector.open()

    assert caught.value is transient_error
    assert not connector.is_connected
    assert connector.open_calls == 1
    assert connector.validation_calls == 1
    assert connector.close_calls == 1

    # An explicit outer retry gets a fresh context. Repeated open after the
    # successful retry remains idempotent and does not revalidate or reconnect.
    connector.open()
    connector.open()
    assert connector.is_connected
    assert connector.open_calls == 2
    assert connector.validation_calls == 2

    connector.close()
    connector.close()
    assert not connector.is_connected
    assert connector.close_calls == 2


def test_connector_validation_cleanup_failure_preserves_primary_error(caplog):
    primary_error = RuntimeError('primary-provider-secret')
    connector = _RetryConnector(
        validation_errors=[primary_error],
        close_errors=[RuntimeError('cleanup-provider-secret')],
    )

    with pytest.raises(RuntimeError) as caught:
        connector.open()

    assert caught.value is primary_error
    assert not connector.is_connected
    assert connector.close_calls == 1
    assert 'validation_error_type=RuntimeError' in caplog.text
    assert 'cleanup_error_type=RuntimeError' in caplog.text
    assert 'provider-secret' not in caplog.text


def test_connector_close_failure_detaches_broken_session_before_reopen():
    cleanup_error = RuntimeError('provider close failed')
    connector = _RetryConnector(close_errors=[cleanup_error])
    connector.open()

    with pytest.raises(RuntimeError) as caught:
        connector.close()

    assert caught.value is cleanup_error
    assert not connector.is_connected
    assert connector.close_calls == 1
    # Repeated cleanup is idempotent, while a later use creates a fresh
    # session instead of reusing the one whose close failed.
    connector.close()
    connector.open()
    assert connector.is_connected
    assert connector.open_calls == 2


def test_connector_open_close_concurrent_no_attribute_error():
    r"""Regression test for a race in ``Connector.open()`` / ``close()``
    where two threads alternating ``open`` / ``close`` could see
    ``self._context`` deleted mid-open and raise::

        AttributeError: '...Connector' object has no attribute '_context'

    on the ``weakref.finalize(self._context, self.close)`` line. The
    base class now serializes ``_context`` mutations with an ``RLock``
    and registers the finalizer against a local variable (not
    ``self._context``), so neither thread can observe a torn state.
    """
    connector = _RaceConnector(name='race_test')
    iterations = 200
    errors: List[Exception] = []
    barrier = threading.Barrier(2)

    def open_loop() -> None:
        for _ in range(iterations):
            barrier.wait()
            try:
                connector.open()
            except Exception as e:
                errors.append(e)

    def close_loop() -> None:
        for _ in range(iterations):
            barrier.wait()
            try:
                connector.close()
            except Exception as e:
                errors.append(e)

    t_open = threading.Thread(target=open_loop)
    t_close = threading.Thread(target=close_loop)
    t_open.start()
    t_close.start()
    t_open.join(timeout=30)
    t_close.join(timeout=30)

    assert not t_open.is_alive() and not t_close.is_alive(), (
        'open/close threads deadlocked')
    assert not errors, f'Concurrent open/close raised: {errors}'


def test_connector_is_picklable():
    r"""Regression test for the interaction between the open/close race
    fix and ``__getstate__``.

    The base class stores an ``RLock`` on ``self._open_lock``.
    ``_thread.RLock`` is not picklable, so ``__getstate__`` must strip
    it from the state, and ``__setstate__`` must re-create it.
    """
    connector = MyConnector(name='pickle_test')
    revived = pickle.loads(pickle.dumps(connector))
    # Revived connector has a working lock and can complete open/close.
    revived.open()
    assert revived.is_connected
    revived.close()
    assert not revived.is_connected


def test_connector_del_on_partially_initialized_object_does_not_raise():
    r"""Regression test for ``close()`` being called on a half-built
    connector.

    If a subclass's ``__init__`` raises before ``super().__init__``
    runs, the object exists but has no ``_open_lock``. ``__del__`` still
    fires at GC and calls ``close()``, which must not raise — pre-PR
    ``close()`` short-circuited on ``is_connected``; we preserve that.
    """
    class _BrokenInit(MyConnector):
        def __init__(self, name: str):
            # Deliberately skip super().__init__ so `_open_lock` is
            # never assigned. This simulates a subclass whose own
            # __init__ raises before calling super().__init__.
            self._name = name

    obj = _BrokenInit(name='broken')
    # Should not raise.
    obj.close()


def test_sql_connector_is_picklable():
    r"""Regression test: ``SQLConnector.__getstate__`` shadowed the base
    ``Connector.__getstate__`` without stripping ``_open_lock`` (a
    ``threading.RLock`` added by the open/close race fix). Pickling any
    SQL-flavored connector (Snowflake, Databricks, BigQuery)
    therefore raised::

        TypeError: cannot pickle '_thread.RLock' object

    inside ``_ForkingPickler.dumps`` whenever a connector was crossed
    over a multiprocessing queue — e.g. during prediction-table
    generation (see kumo-ai/kumo#28760).
    """
    connector = SnowflakeConnector('snowflake', account='example-account',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='FINANCIAL')
    revived = pickle.loads(pickle.dumps(connector))
    # Revived instance has a fresh, usable RLock — i.e. ``__setstate__``
    # actually re-created it via ``super().__setstate__``.
    assert revived._open_lock.acquire(blocking=False)
    revived._open_lock.release()
