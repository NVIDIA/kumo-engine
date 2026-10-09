# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import opentelemetry.context
import opentelemetry.trace
import pandas as pd
from asgiref.sync import SyncToAsync
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)
from opentelemetry.util._once import Once

logger = logging.getLogger(__name__)

TRACING_DIR = os.path.expanduser("~/.monitoring/tracing")
TRACING_INITIALIZED = False
TRACING_ENABLED = False


class SpanExporterType(Enum):
    """
    Defines enabled SpanExporters, configurable with keyword arguments in
    initialize_tracing. This class is exposed to the user (instead of the
    SpanExporter implementations) so that the user does not need to be
    aware of specific implementation details (and the API is cleaner
    as a result).

    Supported kwargs:
        CONSOLE:
            No keyword arguments supported.
        FILE:
            output_dir (str): the absolute path of the directory to
                write trace logs to. Default: /home/ubuntu/.monitoring/tracing.
        JAEGER:
            jaeger_agent_host_name (str): the agent hostname of the
                jaeger exporter. Default: localhost.
            jaeger_port (int): the port number of the jaeger instance.
                Default: 4317.
    """

    CONSOLE = 1
    FILE = 2
    JAEGER = 3


def _format_span(span: ReadableSpan) -> str:
    return span.to_json(indent=None) + os.linesep  # type: ignore


def _get_span_processor_from_type(
    name: str,
    type: SpanExporterType,
    output_dir: str = TRACING_DIR,
    jaeger_agent_host_name: str = 'localhost',
    jaeger_port: int = 4317,
    out: Optional[IO] = None,
) -> SpanProcessor:
    """Obtains a span processor from a SpanExporterType.

    Args:
        out: destination stream for the CONSOLE exporter. Defaults to the
            process stdout; tests inject a buffer so the exported spans can be
            asserted on without capturing the real file descriptor.
    """
    global TRACING_DIR

    if type == SpanExporterType.CONSOLE:
        return SimpleSpanProcessor(
            ConsoleSpanExporter(out=out if out is not None else sys.stdout))

    if type == SpanExporterType.FILE:
        TRACING_DIR = output_dir
        os.makedirs(TRACING_DIR, exist_ok=True)

        return SimpleSpanProcessor(
            ConsoleSpanExporter(
                # TODO(manan): does this leak memory?
                out=open(
                    TRACING_DIR + f'/tracing-{name}.txt',
                    "w",
                    encoding="utf-8",
                ),
                formatter=_format_span,
            ))

    if type == SpanExporterType.JAEGER:
        oltp_exporter = OTLPSpanExporter(
            endpoint=f'http://{jaeger_agent_host_name}:{jaeger_port}')

        return BatchSpanProcessor(oltp_exporter)

    raise NotImplementedError(f'{type} is not supported.')


class SpanPropagatorType(Enum):
    """
    Defines enabled SpanPropagators, configurable with keyword arguments in
    propagate_span and retrieve_span.
    """

    TEXTMAP = 1
    JAEGER = 2  # TODO(manan): implement Jaeger propagation.


def _get_span_propagator_from_type(type: SpanPropagatorType):
    """Obtains a span propagator from a SpanPropagatorType."""
    if type == SpanPropagatorType.TEXTMAP:
        return TraceContextTextMapPropagator()

    return None


def initialize_tracing(
    name: str,
    span_exporters: List[SpanExporterType] = [SpanExporterType.FILE],
    initialize_tracer_provider: bool = True,
    **span_exporter_kwargs,
) -> None:
    """
    Initializes a tracing session by setting the tracer provider and span
    exporters.

    TODO(manan): Read from configuration file?

    Args:
        name (str): the name of the trace
        span_exporters (List[SpanExporter]): the list of SpanExporters to
            use for this trace.
        initialize_tracer_provider (bool): whether to initialize the tracer
            provider; this can only be done once per execution context.
            Default: True.
        **span_exporter_kwargs (Dict[str, Any]): a list of arguments to be
            passed to the relevant SpanExporters. Default values are provided
            for all supported SpanExporters.
    """
    global TRACING_INITIALIZED
    if TRACING_INITIALIZED:
        logger.warning("Tracing has already been initialized.")
        return
    TRACING_INITIALIZED = True

    # Set the tracer_provider: only allowed once per execution context. If
    # tracing is initialized multiple times, the first TracerProvider will be
    # used and a warning will be printed (this is OK).
    if initialize_tracer_provider:
        if SpanExporterType.JAEGER not in span_exporters:
            provider = TracerProvider()
        else:
            provider = TracerProvider(
                resource=Resource.create({SERVICE_NAME: name}))
        opentelemetry.trace.set_tracer_provider(provider)

        # Add span processors based on the provided span exporters.
        for span_exporter_type in span_exporters:
            provider.add_span_processor(
                _get_span_processor_from_type(name, span_exporter_type,
                                              **span_exporter_kwargs))


def shutdown_tracing() -> None:
    """
    Shuts down a tracing session. Note: this is done automatically on process
    exit, the only reason for a user to manually shut down tracing is if
    multiple unique tracing instances are required in a single process (this
    is rarely the case).
    """
    global TRACING_INITIALIZED
    tracer_provider = opentelemetry.trace.get_tracer_provider()
    tracer_provider.shutdown()  # type: ignore
    TRACING_INITIALIZED = False
    # We have to reset this to allow setting trace provider again (in test)
    opentelemetry.trace._TRACER_PROVIDER_SET_ONCE = Once()


def enable_tracing() -> None:
    """Enables tracing."""
    global TRACING_ENABLED
    TRACING_ENABLED = True


def disable_tracing() -> None:
    """Disables tracing."""
    global TRACING_ENABLED
    TRACING_ENABLED = False


def is_tracing_enabled() -> bool:
    """Checks whether tracing is enabled."""
    return TRACING_ENABLED


def is_tracing_initialized() -> bool:
    """Checks whether tracing has been initialized."""
    return TRACING_INITIALIZED


def get_output_dir() -> str:
    """
    Returns the directory at which traces are stored, if
    SpanExporterType.FILE is enabled.
    """
    return TRACING_DIR


@dataclass
class TraceOptions:
    span_name: str = ""
    span_name_suffix: str = ""

    # Whether/which parameter values to track in the trace span:
    # if true/false, all or none parameter values will be included in span's
    # attributes; if this is a list, only specified param keys(names) will be
    # included ("self" is always excluded).
    trace_params: Union[bool, Sequence[str]] = False

    trace_return_value: bool = False
    flatten_attributes: bool = True
    # External provider exceptions may contain credentials, signed requests,
    # or response payloads. Such call sites can retain span timing/status
    # without exporting the exception message or stacktrace.
    record_exceptions: bool = True
    # TODO: deprecate below options?
    propagate_span: Optional[SpanPropagatorType] = None
    use_propagated_span: Optional[Tuple[SpanPropagatorType, Any]] = None

    def get_span_name(self, fn: Callable):
        if self.span_name:
            return self.span_name
        return f'{fn.__module__}.{fn.__qualname__}{self.span_name_suffix}'


@contextmanager
def _trace(function: Callable, options: TraceOptions, *args: Any,
           kwargs: Dict[str, Any]):
    # no-op if tracing is disabled or not initialized
    if not is_tracing_enabled() or not is_tracing_initialized():
        logger.debug('Tracing is not enabled or initialized')
        yield None
        return

    try:
        # TODO(manan): how inefficient is this? Is there a way to cache the
        # tracer?
        tracer = opentelemetry.trace.get_tracer(__name__)

        # check if the function accepts kwargs if any of these options are
        # enabled
        if (options.propagate_span or options.use_propagated_span):
            arg_spec = inspect.getfullargspec(function)
            if not arg_spec.varkw:
                raise Exception("Keyword arguments expected but not provided.")

        # span name defaults to function name if not provided
        _span_name = options.get_span_name(function)

        # deal with the new context if a propagated span context is being
        # used
        ctx = None
        if options.use_propagated_span:
            carrier = options.use_propagated_span[1]
            if options.use_propagated_span[1] is None:
                carrier = kwargs["carrier"]
            ctx = parse_propagated_span(
                options.use_propagated_span[0],
                carrier,
            )
        else:
            ctx = opentelemetry.context.get_current()

        # Create new span, start it:
        _span = tracer.start_as_current_span(
            _span_name,
            context=ctx,
            record_exception=options.record_exceptions,
            set_status_on_exception=options.record_exceptions,
        )
    except Exception as e:
        logger.warning(f"Capturing span failed with exception {e}")
        yield None
        return

    with _span as span:
        # TODO(manan): this code better not raise...
        if options.trace_params:
            parameter_values = _params_to_dict(
                function,
                options.trace_params,
                *args,
                **kwargs,
            )
            _store_dict_in_span(parameter_values, span,
                                options.flatten_attributes)

        if options.propagate_span is not None:
            kwargs["span_carrier"] = do_propagate_span(options.propagate_span)

        yield span


def trace(options: TraceOptions = TraceOptions()) -> Callable:
    """
    Defines a tracing decorator that enables tracing on its decorated function,
    provided that tracing has been enabled and initialized (no-op if either has
    not been done).

    The design of this function is motivated by the philosophy that users of a
    tracing library should have to do little to no work in order to enable
    tracing on a function and observe metrics and logs they care about.
    In order to facilitate this, most users will only have to decorate a
    function with @trace(...) and the rest will be handled for them.

    For those that want to inject custom attributes into the trace, they may
    use `write_attributes_to_current_span()` to define custom attributes within
    the function.

    For those that want to propagate a span to another process that uses the
    propagated span context, they may make use of the propagate_span and
    use_propagated_span arguments (as documented below). By default, parameter
    values and returned values are traced (and flattened), although this
    behavior is customizable as well.

    TODO(manan): perhaps instead of propagate and use_propagated, send and
    receive?

    Arguments:
        span_name (str): The name of the span used to log tracing metrics for
            the decorated function. Default: the function name.
        propagate_span (Optional[SpanPropagatorType]): Whether to propagate the
            span using the span propagator associated with the provided
            SpanPropagatorType.  Propagation allows for spans to be used across
            process boundaries. The carrier for the propagated span context is
            stored in kwargs['span_carrier'], so enabling this option requires
            the decorated function to accept keyword arguments. Default: None,
            which disables span propagation.
        use_propagated_span (Optional[Tuple[SpanPropagatorType, Any]]): Whether
            to use a propagated span, defined by a SpanPropagatorType and a
            carrier for the propagated span context. If the carrier for the
            propagated span context is None, it is expected to be passed as a
            keyword argument to the function named carrier. Using a propagated
            span allows a function to start its trace as a child of the
            propagated context, keeping traces consistent across process
            boundaries. A common scenario for this is a client/server request,
            where the client passes a propagted span to the server and the
            server uses the propagated span to keep traces in sync. Default:
            None, which does not use a propagated span.
        trace_parameter_values (bool): Whether to include parameter values as
            attributes in the trace. Default: True.
        trace_return_value (bool): Whether to include return value(s) as
            attributes in the trace. Default: True.
        flatten_attributes (bool): Whether to flatten attributes (parameter or
            return values, if enabled) in the trace. Default: True.

    Examples:
        TODO(manan): Some basic examples here.
    """
    def _inner_trace(function: Callable) -> Callable:
        @wraps(function)
        def _trace_sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _trace(function, options, *args, kwargs=kwargs) as span:
                ret = function(*args, **kwargs)
                if options.trace_return_value and span:
                    _store_dict_in_span({"return": ret}, span,
                                        options.flatten_attributes)
                return ret

        @wraps(function)
        async def _trace_async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _trace(function, options, *args, kwargs=kwargs) as span:
                ret = await function(*args, **kwargs)
                if options.trace_return_value and span:
                    _store_dict_in_span({"return": ret}, span,
                                        options.flatten_attributes)
                return ret

        if inspect.iscoroutinefunction(function) or isinstance(
                function, SyncToAsync):
            return _trace_async_wrapper
        else:
            return _trace_sync_wrapper

    return _inner_trace


def write_attributes_to_current_span(attributes: Dict[Any, Any]) -> None:
    """
    Writes custom attributes (key/value pairs) to the current span.
    """
    span = opentelemetry.trace.get_current_span()
    _store_dict_in_span(attributes, span, False)


def do_propagate_span(span_propagator_type: SpanPropagatorType) -> Any:
    """
    Propagates the current span context with the defined span_propagator_type.
    Useful for enabling tracing across process boundaries.

    Arguments:
        span_propagator_type (SpanPropagatorType): The type of propagator to
        use.

    Returns:
        The carrier containing information about the propagated span.
    """
    propagator = _get_span_propagator_from_type(span_propagator_type)
    if propagator is None:
        raise Exception(
            f"Propagator type {span_propagator_type} is not supported")

    carrier = {}
    propagator.inject(carrier=carrier)
    return carrier


def parse_propagated_span(span_propagator_type: SpanPropagatorType,
                          carrier: Any) -> Any:
    """
    Returns the context of a propagated span (extracted from carrier) for
    continued tracing across process boundaries.

    Arguments:
        span_propagator_type (SpanPropagatorType): The type of propagator used
            when injecting into carrier.
        carrier (Any): The carrier containing the span context for extraction.

    Returns:
        The SpanContext obtained from the propagated span.
    """
    propagator = _get_span_propagator_from_type(span_propagator_type)
    if propagator is None:
        raise Exception(
            f"Propagator type {span_propagator_type} is not supported")

    return propagator.extract(carrier=carrier)


# UTILITY FUNCTIONS


def _serialize_dict(d: Dict[Any, Any]) -> Dict[Any, Any]:
    """Serializes a dictionary."""
    return json.loads(json.dumps(d, default=str))


def _flatten_dict(d: Dict[Any, Any], sep="_"):
    """Flattens a dictionary, joining parent/child keys with `sep`."""
    flattened = pd.json_normalize(d, sep=sep).to_dict(orient="records")
    return flattened[0] if len(flattened) > 0 else {}


def _params_to_dict(
    func: Callable,
    params_to_track: Union[bool, Sequence[str]],
    *args: Any,
    **kwargs: Any,
) -> Dict[Any, Any]:
    """Returns parameters of function `func` as a dictionary."""
    try:
        bound_arguments = inspect.signature(func).bind(*args, **kwargs)
    except TypeError:
        return {}
    bound_arguments.apply_defaults()

    def _track_key(key: str):
        if key == 'self':
            return False

        if isinstance(params_to_track, bool):
            return params_to_track

        return key in params_to_track

    return {
        f'params.{key}': value
        for key, value in bound_arguments.arguments.items() if _track_key(key)
    }


def _store_dict_in_span(d: Dict[Any, Any], span: Any, flatten=True):
    """
    Stores a dictionary in a span (as attributes), optionally flattening it.
    """
    if flatten:
        d = _flatten_dict(d)
    d = {k: v if v is not None else 'None' for k, v in d.items()}
    span.set_attributes(_serialize_dict(d))
