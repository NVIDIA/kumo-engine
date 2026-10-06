# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from time import sleep
from typing import Dict, List, Optional

import pytest
from opentelemetry import baggage
from opentelemetry.trace import SpanKind

import monitoring.tracing
from io_utils import kumo_dataclass
from io_utils.json import load_json
from monitoring.settings import MonitoringSettings
from rest.utils import tracing

_RESOURCE_NAME = 'kumo-pytest'


@pytest.fixture(autouse=True)
def _isolate_otel_context():
    """Restore the ambient otel context after every test.

    ``set_context_from_carrier`` attaches a context and never detaches
    (fine for its worker-process callers, fatal for test isolation): after
    ``test_context_carrier_load`` every later span in the process inherits
    a REMOTE parent, so ``RequestTraceAnalyzer`` skips its analysis
    (``span.parent`` set) and ``test_request_analyzer`` fails — but only
    in pytest-split shards where both tests land together, so it surfaces
    as an order-dependent flake whenever the shard boundaries move.
    """
    from opentelemetry import context as otel_context
    token = otel_context.attach(otel_context.get_current())
    yield
    otel_context.detach(token)


@kumo_dataclass
class TestingSpan:
    name: str
    trace_id: str
    instance_id: str
    version: str
    service_name: str

    @classmethod
    def from_json(cls, data: Dict) -> 'TestingSpan':
        name = data.get('name', '')

        attributes = data.get('attributes', {})
        instance_id = attributes.get('instance_id', '')
        version = attributes.get('version', '')

        resource = data.get('resource', {})
        service_name = resource.get('attributes', {}).get('service.name', '')

        context = data.get('context', {})
        trace_id = context.get('trace_id', '')

        return cls(name, trace_id, instance_id, version, service_name)


def get_test_spans(trace_id: Optional[str] = None) -> List[TestingSpan]:
    trace_data = load_json(tracing.get_test_filepath(), Dict[str, List])
    spans = [TestingSpan.from_json(span) for span in trace_data['traces']]
    if trace_id is not None:
        return [span for span in spans if span.trace_id == trace_id]
    return spans


def get_test_trace_id(parent_span_name: str) -> str:
    trace_id = None
    for span in get_test_spans():
        if span.name == f'KUMO-{parent_span_name}':
            trace_id = span.trace_id
    assert trace_id is not None
    return trace_id


def get_test_span(
    span_name: str,
    trace_id: Optional[str] = None,
) -> Optional[TestingSpan]:
    for span in get_test_spans(trace_id):
        if span.name == f'KUMO-{span_name}':
            if trace_id is not None:
                assert span.trace_id == trace_id
            return span
    return None


@pytest.fixture(scope="module")
def init_tracing():
    assert not tracing.is_initialized()
    trace_analyzer_config = tracing.RequestTraceAnalyzerConfig(
        request_latency_threshold=0.0,
        duplicate_calls=2,
        duration_frac=0.1,
        repeated_calls=3,
        merged_duration_frac=0.5,
    )
    tracing.init_tracing(_RESOURCE_NAME, MonitoringSettings(),
                         trace_analyzer_config)
    yield
    monitoring.tracing.disable_tracing()


def test_tracing_initialized(init_tracing: None, ):
    assert tracing.is_initialized()


def test_simple_span(init_tracing: None, ):
    with tracing.start_span('simple_span'):
        print('Hello, World!')

    span = get_test_span('simple_span')
    assert span is not None
    assert span.instance_id != 'None'
    assert span.version != 'None'

    assert span.service_name == _RESOURCE_NAME


def test_parent_child_span(init_tracing: None, ):
    with tracing.start_span('parent_span'):
        print('Hello, World!')
        with tracing.start_span('child_span'):
            print('hello world')

    with tracing.start_span('sibling_span'):
        print('Unrelated')

    trace_id = get_test_trace_id('parent_span')

    assert len(get_test_spans(trace_id)) == 2  # Includes parent span
    assert get_test_span('child_span', trace_id) is not None


def test_context_carrier_basic(init_tracing: None, ):
    with tracing.start_span('save_context'):
        tracing.add_metadata_to_current_context('testkey', 'testvalue')
        carrier = tracing.get_context_carrier()
        assert 'testkey=testvalue' in carrier.get('baggage', '')

    print(carrier, get_test_trace_id('save_context'))


def test_context_carrier_load(init_tracing: None, ):
    # Generated from a call to `test_context_carrier_basic``
    remote_trace_id_hex = "0x67a3ca472727261c87affebb255d70d8"
    carrier = {
        'traceparent':
        '00-67a3ca472727261c87affebb255d70d8-d73daf410d50eac0-01',
        'baggage':
        ('instance_id=i-039063efeeff7bde8,'
         'version=1.6.1.dev71%2Bgd0c61d0fb.d20230130,testkey=testvalue')
    }

    with tracing.start_span('load_context'):
        # Read the vanilla local context
        prev_context = tracing.get_context()
        assert baggage.get_baggage('testkey', context=prev_context) is None
        local_trace_id = tracing.get_current_span().get_span_context().trace_id

        # Load context from carrier
        tracing.set_context_from_carrier(carrier)

        # Check new context
        post_context = tracing.get_context()
        assert baggage.get_baggage('testkey',
                                   context=post_context) == 'testvalue'
        remote_trace_id = tracing.get_current_span().get_span_context().trace_id
        assert hex(remote_trace_id) == remote_trace_id_hex
        assert remote_trace_id != local_trace_id


def test_async_span(init_tracing: None, ):
    async_span = tracing.start_async_span('simple_async_span')

    assert get_test_span('simple_async_span') is None

    async_span.end()

    span = get_test_span('simple_async_span')
    assert span is not None

    assert span.instance_id != 'None'
    assert span.version != 'None'

    assert span.service_name == _RESOURCE_NAME


def test_request_analyzer(init_tracing: None, caplog: pytest.LogCaptureFixture):
    @monitoring.tracing.trace(
        monitoring.tracing.TraceOptions(trace_params=True))
    def foo(param1: int, param2: str):
        sleep(0.1)

    with caplog.at_level(logging.WARNING):
        with tracing.start_span('dummy_api_request',
                                kind=SpanKind.SERVER) as span:
            span.set_attribute('http.route', '/v0/api/`{id`}')
            foo(1, 'abc')
            foo(1, 'abc')
            foo(0, 'x')
            foo(0, 'y')
            foo(0, 'z')

    assert ('Too many duplicate calls of '
            'test.rest.test_rest_tracing.test_request_analyzer.<locals>.foo'
            '(param1=1,param2=abc)') in caplog.text
    assert ('Too many repeated calls of '
            'test.rest.test_rest_tracing.test_request_analyzer.<locals>.foo'
            ) in caplog.text

    monitoring.tracing.disable_tracing()


def test_included_router_partial_match_does_not_500(init_tracing: None) -> None:
    """starlette 1.x `_IncludedRouter` regression guard.

    A request that PARTIAL-matches an included router (here: wrong method on
    an include_router route) crashes otel-fastapi's unguarded
    `_get_route_details` with AttributeError and 500s every such request
    (live incident: all /v0/auth/* requests). `connect_fastapi` installs a
    guard; this pins it.
    """
    from fastapi import APIRouter, FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    router = APIRouter(prefix='/v0/auth')

    @router.post('/login')
    def login() -> dict:
        return {'ok': True}

    app.include_router(router)
    tracing.connect_fastapi(app, tracing.get_current_provider())

    client = TestClient(app, raise_server_exceptions=False)
    # Method mismatch => Match.PARTIAL on the _IncludedRouter entry.
    assert client.get('/v0/auth/login').status_code == 405
    assert client.post('/v0/auth/login').status_code == 200
