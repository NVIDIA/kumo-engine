# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import logging
import os

import pytest
from pytest import LogCaptureFixture

from io_utils import fsutil
from io_utils import logging as kumo_logging
from monitoring import tracing
from monitoring.tracing import TraceOptions

TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), ".test_files")
CARRIER = None


@pytest.fixture(autouse=True, scope='function')
def setup_and_teardown_test():
    kumo_logging.initialize_logging(
        log_to_file=False,
        logging_level_overrides={'monitoring': logging.DEBUG})
    fsutil.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    yield

    fsutil.rm(TEST_OUTPUT_DIR, ignore_errors=True)
    tracing.disable_tracing()
    tracing.shutdown_tracing()


def test_tracing_simple():
    """Test a simple tracing example with three nested functions."""
    tracing.initialize_tracing(name="test_tracing", output_dir=TEST_OUTPUT_DIR)
    tracing.enable_tracing()

    tracing_dir = tracing.get_output_dir()
    file = os.path.join(tracing_dir, "tracing-test_tracing.txt")

    # Execution tree:
    #
    # tts_func_one(a=1, b=2)
    #     -> tts_func_two(c=1, d=4)
    #         -> tts_func_three(e=1, f=6)
    #
    tts_func_one(a=1, b=2)

    with open(file, "r", encoding="utf-8") as f:
        json_out = [json.loads(x) for x in f.readlines()]

    keyed_by_name = {item["name"]: item for item in json_out}

    assert keyed_by_name["test.utils.test_tracing.tts_func_one"][
        "parent_id"] is None
    expected_attributes = {
        "params.a": 1,
        "params.b": 2,
        "return": -20,
    }
    assert keyed_by_name["test.utils.test_tracing.tts_func_one"][
        "attributes"] == expected_attributes

    assert (keyed_by_name["test.utils.test_tracing.tts_func_two"]["parent_id"]
            == keyed_by_name["test.utils.test_tracing.tts_func_one"]["context"]
            ["span_id"])
    expected_attributes = {
        "params.c": 1,
        "params.d": 4,
        "return": -20,
    }
    assert keyed_by_name["test.utils.test_tracing.tts_func_two"][
        "attributes"] == expected_attributes

    assert (
        keyed_by_name["test.utils.test_tracing.tts_func_three"]["parent_id"] ==
        keyed_by_name["test.utils.test_tracing.tts_func_two"]["context"]
        ["span_id"])
    expected_attributes = {
        "params.e": 1,
        "params.f": 6,
        "return": -5,
    }
    assert keyed_by_name["test.utils.test_tracing.tts_func_three"][
        "attributes"] == expected_attributes


def test_tracing_exception_handling():
    """Tests that exceptions are handled gracefully & stored in trace logs."""

    tracing.initialize_tracing(name="test_tracing", output_dir=TEST_OUTPUT_DIR)
    tracing.enable_tracing()

    tracing_dir = tracing.get_output_dir()
    file = os.path.join(tracing_dir, "tracing-test_tracing.txt")

    try:
        func_exception()
    except CustomException:
        pass

    with open(file, "r", encoding="utf-8") as f:
        json_trace = json.loads(f.readline())

    assert json_trace['status']['status_code'] == 'ERROR'
    assert json_trace['events'][0]['name'] == 'exception'


def test_provider_boundaries_keep_span_export_secret_free(caplog):
    """Remote fsutil and connector spans retain timing but not exceptions."""
    from unittest.mock import Mock

    from io_utils.fsutil import impl_adls, impl_gcs, impl_s3, impl_ucv
    from kumo.connector.blocking import blocking_method
    from kumo.connector.blocking import trace as connector_trace
    from kumo.connector.snowflake_connector import SnowflakeConnector

    tracing.initialize_tracing(name='provider_safe', output_dir=TEST_OUTPUT_DIR)
    tracing.enable_tracing()
    caplog.set_level(logging.INFO)
    canary = 'provider-body password=otel-secret-canary'

    for module in (impl_s3, impl_gcs, impl_ucv, impl_adls):
        options = module.TraceOptions()
        assert options.record_exceptions is False

        @tracing.trace(options)
        def fail_remote_provider():
            raise RuntimeError(canary)

        with pytest.raises(RuntimeError):
            fail_remote_provider()

    @connector_trace()
    def fail_connector_trace():
        raise RuntimeError(canary)

    class FailingConnectorMethod:
        @blocking_method()
        def fail(self):
            raise RuntimeError(canary)

    with pytest.raises(RuntimeError):
        fail_connector_trace()
    with pytest.raises(RuntimeError):
        FailingConnectorMethod().fail()

    sql_secret = 'snowflake-inline-credential-secret-canary'
    query = (
        "COPY INTO 's3://safe/output' FROM safe_table "
        "CREDENTIALS = ("
        f"AWS_KEY_ID = 'access' AWS_SECRET_KEY = '{sql_secret}' "
        "AWS_TOKEN = 'token')")
    context = Mock()
    context.cursor.return_value.execute.side_effect = RuntimeError(canary)
    connector = SnowflakeConnector('snowflake', 'account', 'warehouse',
                                   'database', 'schema')
    with pytest.raises(RuntimeError):
        connector.execute_meta_query(context, query)

    tracing.shutdown_tracing()
    trace_file = os.path.join(tracing.get_output_dir(),
                              'tracing-provider_safe.txt')
    with open(trace_file, 'r', encoding='utf-8') as file:
        exported = file.read()

    assert canary not in exported
    assert sql_secret not in exported
    assert query not in exported
    assert canary not in caplog.text
    assert sql_secret not in caplog.text
    assert query not in caplog.text
    assert 'COPY INTO' not in caplog.text
    assert 'CREDENTIALS' not in caplog.text
    spans = [json.loads(line) for line in exported.splitlines()]
    assert spans
    assert all(not any(event.get('name') == 'exception'
                       for event in span.get('events', [])) for span in spans)


def test_tracing_span_passing():
    """Tests that spans can successfully be passed to decorated functions."""

    tracing.initialize_tracing(name="test_tracing", output_dir=TEST_OUTPUT_DIR)
    tracing.enable_tracing()

    tracing_dir = tracing.get_output_dir()
    file = os.path.join(tracing_dir, "tracing-test_tracing.txt")

    ttsp_func(a=1, b=2)

    with open(file, "r", encoding="utf-8") as f:
        json_trace = json.loads(f.readline())

    assert json_trace['attributes'] == {
        'params.a': 1,
        'params.b': 2,
        'a + b': 3,
        'return': 3
    }


def test_tracing_initialize_and_enable_tracing(caplog: LogCaptureFixture):
    """Tests whether initializing and enabling tracing work as indended."""

    # Independent test: enable without initializing
    tracing.enable_tracing()
    with caplog.at_level(logging.DEBUG):
        func_no_children(a=1, b=2)
        assert "Tracing is not enabled or initialized" in caplog.text
    tracing.disable_tracing()

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        func_no_children(a=1, b=2)
        assert "Tracing is not enabled or initialized" in caplog.text

    # Post-initialization tests
    tracing.initialize_tracing(name="test_tracing", output_dir=TEST_OUTPUT_DIR)
    tracing_dir = tracing.get_output_dir()
    file = os.path.join(tracing_dir, "tracing-test_tracing.txt")

    # 1) pre-enabling
    assert tracing.is_tracing_initialized()
    assert not tracing.is_tracing_enabled()

    func_no_children(a=1, b=2)
    with open(file, "r", encoding="utf-8") as f:
        assert f.readline() == ""

    # 2) post-enabling
    tracing.enable_tracing()

    func_no_children(a=1, b=2)
    with open(file, "r", encoding="utf-8") as f:
        assert f.readline() != ""

    assert tracing.is_tracing_enabled()


def test_tracing_shutdown_tracing():
    """
    Tests whether tracing.shutdown_tracing() works as intended
    by starting two tracing sessions writing to different files.
    """
    tracing.enable_tracing()

    tracing.initialize_tracing(
        name="test_tracing",
        span_exporters=[tracing.SpanExporterType.FILE],
        output_dir=TEST_OUTPUT_DIR,
    )

    func_no_children(a=1, b=2)

    tracing.shutdown_tracing()

    # Initialize tracing for all examples
    tracing.initialize_tracing(
        name="test_tracing_2",
        span_exporters=[tracing.SpanExporterType.FILE],
        output_dir=TEST_OUTPUT_DIR,
    )

    func_no_children(a=3, b=4)

    tracing_dir = tracing.get_output_dir()
    file_one = os.path.join(tracing_dir, "tracing-test_tracing.txt")
    file_two = os.path.join(tracing_dir, "tracing-test_tracing_2.txt")

    with open(file_one, "r", encoding="utf-8") as f:
        json_one = json.loads(f.readline())
    with open(file_two, "r", encoding="utf-8") as f:
        json_two = json.loads(f.readline())

    assert json_one["attributes"]["params.a"] == 1
    assert json_one["attributes"]["params.b"] == 2
    assert json_two["attributes"]["params.a"] == 3
    assert json_two["attributes"]["params.b"] == 4


def test_tracing_span_propagation():
    """Tests span propagation between two functions with a common carrier."""

    tracing.initialize_tracing(name="test_tracing_sp",
                               output_dir=TEST_OUTPUT_DIR)
    tracing.enable_tracing()

    tracing_dir = tracing.get_output_dir()
    file = os.path.join(tracing_dir, "tracing-test_tracing_sp.txt")

    # client performs function, obtains CARRIER (here, stored globally)
    ttsp_func_client(a=1, b=2)

    # server (somehow, e.g. via HTTP GET request storing data) obtains
    # CARRIER from client and continues in the same span context
    ttsp_func_server(c=2, d=3, carrier=CARRIER)

    json_out = None
    with open(file, "r", encoding="utf-8") as f:
        json_out = [json.loads(x) for x in f.readlines()]

    keyed_by_name = {item["name"]: item for item in json_out}

    assert keyed_by_name['test.utils.test_tracing.ttsp_func_client'][
        'context']['trace_id'] == keyed_by_name[
            'test.utils.test_tracing.ttsp_func_server']['context']['trace_id']

    assert keyed_by_name['test.utils.test_tracing.ttsp_func_client'][
        'parent_id'] is None
    assert keyed_by_name['test.utils.test_tracing.ttsp_func_server'][
        'parent_id'] == keyed_by_name[
            'test.utils.test_tracing.ttsp_func_client']['context']['span_id']


# NOTE: Unable to test Jaeger (as it requires the Jaeger backend to be
# collecting data on a specified port). This test must be executed last,
# as adding a span exporter is global (per-process) and so adding CONSOLE
# before other initializations will use CONSOLE everywhere else.
def test_tracing_span_exporters():
    # Inject a buffer as the CONSOLE exporter's destination so the exported
    # spans can be asserted on directly, instead of capturing the process's
    # real stdout file descriptor.
    buffer = io.StringIO()
    tracing.initialize_tracing(
        name="test_tracing",
        span_exporters=[
            tracing.SpanExporterType.CONSOLE,
            tracing.SpanExporterType.FILE,
        ],
        output_dir=TEST_OUTPUT_DIR,
        out=buffer,
    )
    tracing.enable_tracing()

    func_no_children(a=1, b=2)

    assert "func_no_children" in buffer.getvalue()


# FUNCTIONS

# test_tracing_simple


@tracing.trace(TraceOptions(trace_params=True, trace_return_value=True))
def tts_func_one(a: int, b: int) -> int:
    return tts_func_two(a, b + 2)


@tracing.trace(TraceOptions(trace_params=True, trace_return_value=True))
def tts_func_two(c: int, d: int) -> int:
    return tts_func_three(c, d + 2) * d


@tracing.trace(TraceOptions(trace_params=True, trace_return_value=True))
def tts_func_three(e: int, f: int) -> int:
    return e - f


# test_tracing_span_passing


@tracing.trace(TraceOptions(trace_params=True, trace_return_value=True))
def ttsp_func(a: int, b: int, **kwargs) -> int:
    tracing.write_attributes_to_current_span({'a + b': a + b})
    return a + b


# test_tracing_span_propagation


@tracing.trace(
    TraceOptions(trace_params=True, trace_return_value=True,
                 propagate_span=tracing.SpanPropagatorType.TEXTMAP))
def ttsp_func_client(a: int, b: int, **kwargs):
    global CARRIER
    CARRIER = kwargs['span_carrier']
    return a + b


@tracing.trace(
    TraceOptions(
        trace_params=True, trace_return_value=True,
        use_propagated_span=(tracing.SpanPropagatorType.TEXTMAP, None)))
def ttsp_func_server(c: int, d: int, **kwargs):
    return c + d


# general functions


@tracing.trace(TraceOptions(trace_params=True, trace_return_value=True))
def func_no_children(a: int, b: int):
    return a + b


class CustomException(Exception):
    pass


@tracing.trace()
def func_exception():
    raise CustomException("This is an exception.")
