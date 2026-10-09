# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Sequence, Tuple

import cachetools
from fastapi import FastAPI
from kumo_core.version import get_kumo_version
from opentelemetry import baggage, context, trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc import trace_exporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Span, SpanContext

import monitoring.tracing
from io_utils import fsutil, logging
from io_utils.json import from_json, to_json
from monitoring.settings import MonitoringSettings
from service_framework.deploy_env import DeployStage
from service_framework.deploy_env_config import deploy_env

LOCAL_JAEGER_HOSTNAME = "localhost"

# Will be set to Jaeger endpoint in Grafana Cloud when available
REMOTE_JAEGER_HOSTNAME = "localhost"

INIT_METADATA_KEYS = ['instance_id', 'version']

ROOT_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__))).split('kumo')[0]
TRACING_TESTDIR = os.path.join(ROOT_DIR, 'kumo', 'test_results', 'traces')

log = logging.getLogger(__name__)
r"""
  _____       _ _   _       _ _          _   _
 |_   _|     (_| | (_)     | (_)        | | (_)
   | |  _ __  _| |_ _  __ _| |_ ______ _| |_ _  ___  _ __
   | | | '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \
  _| |_| | | | | |_| | (_| | | |/ | (_| | |_| | (_) | | | |
 |_____|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|


 This section contains functions that help to initialize tracing in each
 process that needs it.
"""


class FileSpanExporter(SpanExporter):
    """Implementation of :class:`SpanExporter` that writes spans to a
    file.

    This should be used for Kumo unit tests or in the case that we want to
    store spans to a file somewhere.
    """
    def __init__(
        self,
        filepath: str,
    ):
        self._filepath = filepath
        self._lock = threading.Lock()

        log.info('Initializing FileSpanExporter with filepath %s', filepath)
        base_dir = os.path.dirname(self._filepath)
        if base_dir != '':
            fsutil.makedirs(base_dir, exist_ok=True)

        # Initialize the empty file
        self._save_json({"traces": []})

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            file_data = self._load_json()
            for span in spans:
                file_data['traces'].append(json.loads(span.to_json()))
            with self._lock:
                self._save_json(file_data)
            return SpanExportResult.SUCCESS
        except Exception as e:
            log.error(e)
            return SpanExportResult.FAILURE

    # Because save_json(), load_json() and fsutil implementations are all
    # enabled for tracing, hence we cannot directly invoke those due to
    # infinite recursion.
    def _load_json(self):
        with open(self._filepath, "r", encoding="utf-8") as f:
            return from_json(f.read(), dict[str, list])

    def _save_json(self, data):
        with open(self._filepath, "w", encoding="utf-8") as f:
            f.write(to_json(data))

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        del timeout_millis
        return True


def _get_span_interval(span: ReadableSpan) -> Tuple[int, int]:
    if not span.start_time or not span.end_time:
        return (0, 0)

    return (span.start_time, max(span.start_time, span.end_time))


def _get_parameterized_key(span: ReadableSpan) -> Optional[str]:
    params = {}
    attrs = span.attributes or {}
    for k, v in attrs.items():
        if k.startswith('params.'):
            params[k[len('params.'):]] = str(v)
    if not params:
        return None
    params_str = ','.join([f'{k}={v}' for k, v in sorted(params.items())])
    return f'{span.name}({params_str})'


@dataclass
class RequestTraceAnalyzerConfig:
    # TODO(siyang): parse it from env variables?

    # Minimum total request latency to activate trace analysis.
    request_latency_threshold: float = 1.0

    # Request paths (prefix match) excluded from trace analysis.
    excluded_urls: Sequence[str] = ('/v0/admin', '/v0/fixture',
                                    '/v0/authentication', '/v0/log')

    # We use the following heuristics to detect potential optimization
    # opportunities via caching, batching and parallelizing. The basic idea is
    # to aggregate/summarize all spans belonging to the same @traced function
    # (either with or without parameters) during the request execution, then:
    #
    # - For parameterized func spans ("track_params" enabled in trace
    # options), trigger warning if we observe several duplicate calls:
    duplicate_calls: int = 3
    # AND each span takes a significant fraction of total latency duration
    duration_frac: float = 0.1
    #
    # - For un-parameterized func spans (ie., spans belonging to the same func
    # name), trigger warning if there are more than:
    repeated_calls: int = 4
    # AND
    # merged total duration is greater than significant fraction of the total
    # request latency (which suggests it's a major factor of slowness):
    merged_duration_frac: float = 0.3


# Convert nanosecond time interval tuple to duration seconds (float)
def _get_duration_sec(ns_time_interval: Tuple[int, int]) -> float:
    return (ns_time_interval[1] - ns_time_interval[0]) / 1e9


# Utility class to aggregate a group of spans.
class SpanGroup:
    def __init__(self) -> None:
        self.intervals: List[Tuple[int, int]] = []

    def add_span(self, span: ReadableSpan):
        if span.start_time is None or span.end_time is None:
            # Should this never happen???
            log.warning('Span has None start time or end time: %s', span)
        else:
            self.intervals.append(_get_span_interval(span))

    @property
    def summary(self) -> str:
        """For logging warning message."""
        return (
            'Span group has {count} spans, total exec time '
            '{exec_time:.3f}s, merged duration {merged_duration:.3f}s').format(
                count=len(self.intervals),
                exec_time=sum([_get_duration_sec(i) for i in self.intervals]),
                merged_duration=self.merged_duration)

    @property
    def merged_duration(self) -> float:
        """
        Total duration of merged intervals (which could be overlapping).
        See https://leetcode.com/problems/merge-intervals
        """
        if len(self.intervals) == 1:
            return _get_duration_sec(self.intervals[0])

        self.intervals.sort(key=lambda x: x[0])
        total_duration = 0.0
        current_interval: Tuple[int, int] = self.intervals[0]
        for i in range(1, len(self.intervals)):
            next_interval = self.intervals[i]
            if next_interval[0] >= current_interval[1]:
                # start a new interval:
                total_duration += _get_duration_sec(current_interval)
                current_interval = next_interval
            else:
                # extend the end of current interval
                current_interval = (current_interval[0], next_interval[1])
        return total_duration + _get_duration_sec(current_interval)


class RequestTraceAnalyzer(SpanProcessor):
    """
    In-house span processor for inspecting and analyzing all spans
    associated with the same REST API request (same trace id).
    """
    def __init__(self, config: RequestTraceAnalyzerConfig) -> None:
        # Keep track of all spans in TTL cache to prevent accidental memory
        # leak. Upon completion of a request's trace, we'll delete the spans
        # associated with the trace id.
        #
        # Key: trace_id (int); value: List[ReadableSpan]
        self._trace_id_spans = cachetools.TTLCache(maxsize=1024, ttl=10 * 60)
        self._config = config

    def on_end(self, span: "ReadableSpan") -> None:
        span_ctx = span.context or get_span_context()
        trace_id = span_ctx.trace_id

        if trace_id in self._trace_id_spans:
            spans: List[ReadableSpan] = self._trace_id_spans[trace_id]
            spans.append(span)
        else:
            self._trace_id_spans[trace_id] = [span]

        if span.parent or span.kind != trace.SpanKind.SERVER:
            # Only analyze server-started top-level span per REST API request.
            return

        all_spans = self._trace_id_spans.pop(trace_id)
        if span.attributes is None:
            return  # skip span without attributes

        http_route = span.attributes.get('http.route')
        if http_route is None:
            return

        assert isinstance(http_route, str)
        for excluded_url in self._config.excluded_urls:
            if http_route.startswith(excluded_url):
                return

        request_interval = _get_span_interval(span)
        request_latency = (request_interval[1] - request_interval[0]) / 1e9
        if request_latency < self._config.request_latency_threshold:
            return

        log.info('Analyzing request: %s, latency: %.3fs', span.name,
                 request_latency)
        # Analyze all child spans.
        group_by_param_key: Dict[str, SpanGroup] = {}
        group_by_span_name: Dict[str, SpanGroup] = {}
        for span in all_spans:
            if '-coroutine(' in span.name:
                continue  # Skip any span for "sync-to-async" func wrapper
            group_by_span_name.setdefault(span.name, SpanGroup()).add_span(span)
            param_key = _get_parameterized_key(span)
            if param_key:
                group_by_param_key.setdefault(param_key,
                                              SpanGroup()).add_span(span)

        warning_messages = []
        # Check spans grouped by parameterized name
        min_duration = self._config.duration_frac * request_latency
        for key, group in group_by_param_key.items():
            duplicate_calls = 0
            for interval in group.intervals:
                if _get_duration_sec(interval) > min_duration:
                    duplicate_calls += 1
            if duplicate_calls >= self._config.duplicate_calls:
                warning_messages.append(
                    f'Too many duplicate calls of {key}: {group.summary}')

        # Check spans grouped by (un-parameterized) span name
        min_duration = self._config.merged_duration_frac * request_latency
        for name, group in group_by_span_name.items():
            if (len(group.intervals) >= self._config.repeated_calls
                    and group.merged_duration >= min_duration):
                warning_messages.append(
                    f'Too many repeated calls of {name}: {group.summary}')

        if warning_messages:
            log.warning('Detected problems in request %s trace_id=%s:\n%s',
                        span.name, trace.format_trace_id(trace_id),
                        '\n'.join(warning_messages))

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        del timeout_millis
        return True


def init_tracing(
    service_name: str,
    settings: MonitoringSettings,
    trace_analyzer_config:
    RequestTraceAnalyzerConfig = RequestTraceAnalyzerConfig(),
) -> trace.TracerProvider:
    """
    This function should be called when a process/service (such as
    kumo-rest, kumo-temporalworker-mlworkflow, etc) starts. This sets up the
    tracer provider to export data to the right location (Jaeger for now)
    """
    # Checks if the tracer is uninitialized (ProxyTracerProvider)
    if is_initialized():
        if deploy_env().deploy_mode == DeployStage.TEST:
            raise ValueError('Cannot initialize multiple providers in TEST')
        return get_current_provider()

    # Mark tracing as initialized for backwards compatibility
    monitoring.tracing.enable_tracing()
    monitoring.tracing.initialize_tracing(service_name, [], False)

    resource = Resource(attributes={"service.name": service_name})

    # set the tracer provider
    provider = TracerProvider(resource=resource)

    # Log spans to console in tests
    if deploy_env().deploy_mode == DeployStage.TEST:
        test_path = get_test_filepath()
        span_processor = SimpleSpanProcessor(FileSpanExporter(test_path))
    else:
        host = settings.jaeger_hostname
        port = settings.jaeger_port
        oltp_exporter = trace_exporter.OTLPSpanExporter(
            endpoint=f'http://{host}:{port}', insecure=True)
        span_processor = BatchSpanProcessor(oltp_exporter)

    provider.add_span_processor(span_processor=span_processor)
    provider.add_span_processor(RequestTraceAnalyzer(trace_analyzer_config))

    trace.set_tracer_provider(provider)

    # Add baggage (metadata) to context
    add_init_metadata_to_current_context()

    log.info('Started OpenTelemetry tracing at %s:%d (Jaeger) [%s]',
             settings.jaeger_hostname, settings.jaeger_port, service_name)

    return provider


def connect_fastapi(
    app: FastAPI,
    provider: trace.TracerProvider,
):
    """
    This function enables automatic telemetry collection using
    FastAPI middleware. This allows us to collect very basic traces when an
    endpoint is hit.
    """
    def server_request_hook(span: Span, scope: dict):
        # This is the top-level span for all REST API calls
        # Add baggage (metadata) to context
        add_init_metadata_to_current_context()

        if span and span.is_recording():
            add_all_metadata_to_span(get_context(), span)
        else:
            log.debug('server_request_hook - Span is not recording')

    def client_request_hook(span: Span, scope: dict, message: dict):
        if span and span.is_recording():
            span.set_attribute("path", scope.get('path', ''))
            add_all_metadata_to_span(get_context(), span)
        else:
            log.debug('client_request_hool - Span is not recording')

    def client_response_hook(span: Span, scope: dict, message: dict):
        if span and span.is_recording():
            span.set_attribute("body", message.get('body', ''))
            add_all_metadata_to_span(get_context(), span)
        else:
            log.debug('client_response_hook - Span is not recording')

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=provider,
        client_request_hook=client_request_hook,
        client_response_hook=client_response_hook,
        server_request_hook=server_request_hook,
    )


def is_initialized() -> bool:
    """Returns whether tracing is already initialized in the calling process"""
    return not isinstance(get_current_provider(), trace.ProxyTracerProvider)


r"""
   _____            _            _
  / ____|          | |          | |
 | |     ___  _ __ | |_ _____  _| |_
 | |    / _ \| '_ \| __/ _ \ \/ | __|
 | |___| (_) | | | | ||  __/>  <| |_
  \_____\___/|_| |_|\__\___/_/\_\\__|

  _____                                   _   _
 |  __ \                                 | | (_)
 | |__) _ __ ___  _ __   __ _  __ _  __ _| |_ _  ___  _ __
 |  ___| '__/ _ \| '_ \ / _` |/ _` |/ _` | __| |/ _ \| '_ \
 | |   | | | (_) | |_) | (_| | (_| | (_| | |_| | (_) | | | |
 |_|   |_|  \___/| .__/ \__,_|\__, |\__,_|\__|_|\___/|_| |_|
                 | |           __/ |
                 |_|          |___/

 This section contains functions that help to propagate context.

 Distributed tracing is all about Spans. Spans are objects that record the
 start and end of a discrete operation. This is why we usually use a context.
 When we enter the context, the span starts, and when we exit it the span ends.

 At any time, if a span starts, OpenTelemetry links it as a child span of the
 most recently started span that hasn't ended, which is called the parent span.
 This allows us to track a hierarchy of "operations" that is useful for
 observability.

 However, this only works when the spans are created in the same process. If
 you need to jump to a different process, remote server, etc, then you need to
 pass along the span hierarchy present at the time of the remote call. (Note
 we're using "remote" here even in the case of a different process.) In order
 to efficiently and robustly pass along the Context, which is a python object,
 we can serialize it into a context "carrier" which will "carry" the context
 of the span hierarchy through a remote call.

 This section contains helper functions meant to help propagate or carry this
 context.
"""


def get_current_provider() -> trace.TracerProvider:
    """Get the currently active global trace provider."""
    return trace.get_tracer_provider()


def get_span_context() -> SpanContext:
    """Get the currently active SpanContext (lowest on the hierarchy)"""
    return trace.get_current_span().get_span_context()


def get_current_span() -> Span:
    """Get the currently active Span (lowest on the hierarchy)"""
    return trace.get_current_span()


def get_context() -> Context:
    """Get the currently active context (lowest on the hierarchy)"""
    return context.get_current()


def get_context_carrier() -> Dict:
    """
    Get the current context in "carrier" form (Dict). If a remote call is being
    made, context should be propagated using this context carrier.
    """
    carrier = {}
    inject(carrier)
    carrier['name'] = 'OpenTelemetryContextCarrier'
    return carrier


def get_context_from_carrier(carrier: Dict) -> Context:
    """
    Extracts the Context object from a carrier dictionary. This should be
    called inside the remote call.
    """
    context = extract(carrier)
    return context


def set_context_from_carrier(carrier: Dict):
    """
    Extracts the Context object from a carrier dictionary and sets it to be
    the current context. This should be done when a process is receiving a
    carrier object from its caller.
    """
    ctx = get_context_from_carrier(carrier)
    context.attach(ctx)


def add_all_metadata_to_span(ctx: Context, span: Span):
    """
    Adds all metadata in context (in baggage) as span attributes to
    see in Jaeger
    """
    for key in INIT_METADATA_KEYS:
        value = baggage.get_baggage(key, ctx)
        span.set_attribute(key, str(value))


def add_metadata_to_current_context(key: str, value: str):
    """Add metadata by key to the current context"""
    context.attach(baggage.set_baggage(key, value))


def add_init_metadata_to_current_context():
    """Add common metadata to the current context"""
    add_metadata_to_current_context("instance_id", deploy_env().instance_id)
    add_metadata_to_current_context("version", get_kumo_version())


r"""
   _____
 / ____|
 | (___  _ __   __ _ _ __  ___
  \___ \| '_ \ / _` | '_ \/ __|
  ____) | |_) | (_| | | | \__ \
 |_____/| .__/ \__,_|_| |_|___/
        | |
        |_|

 This section contains context manager functions that start new spans
 (locally or remote)
"""


@contextmanager
def start_span(
    name: str,
    link: Optional[trace.Link] = None,
    ctx: Optional[Context] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
) -> Generator[Span, None, None]:
    """
    This function starts a local span. Optionally you can pass in a link or a
    ctx to link remote spans.
    """
    provider = trace.get_tracer_provider()
    links = [link] if link is not None else None

    ctx = ctx if ctx is not None else context.get_current()

    with provider.get_tracer(__name__).start_as_current_span(
            f"KUMO-{name}", links=links, context=ctx, kind=kind) as span:
        try:
            add_all_metadata_to_span(ctx, span)
            yield span
        finally:
            pass


def start_async_span(
    name: str,
    link: Optional[trace.Link] = None,
    ctx: Optional[Context] = None,
) -> Span:
    """
    This function starts a local async span. Optionally you can pass in a
    link or a ctx to link remote spans.
    """
    provider = trace.get_tracer_provider()
    links = [link] if link is not None else None

    ctx = ctx if ctx is not None else context.get_current()

    span = provider.get_tracer(__name__).start_span(f"KUMO-{name}", links=links,
                                                    context=ctx)
    add_all_metadata_to_span(ctx, span)
    return span


# Testing
def get_test_filepath() -> str:
    """Get the filepath where spans are written to in unit tests"""
    return os.path.join(TRACING_TESTDIR, 'pytest.txt')
