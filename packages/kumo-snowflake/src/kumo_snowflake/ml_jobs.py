# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit user-function submission to Snowflake ML Jobs.

This is intentionally not a Kumo backend. The submitted function owns the
runtime API it invokes inside Snowflake; this package only submits and observes
that function.
"""

from __future__ import annotations

import dis
import inspect
import logging
import math
import time
import types
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Callable, Mapping, Sequence, cast

from kumoapi.common import JobStatus
from kumoapi.jobs import (
    AutoTrainerProgress,
    JobEventLogEntry,
    JobStatusReport,
    PredictionProgress,
)
from kumoapi.json_serde import from_json
from kumoapi.plugin import PublishedModel

from ._ml_job_runtime import execute_user_function
from ._ml_job_stage import StageStore, row_dict, validate_stage_name
from ._ml_job_wheel import discover_installed_wheel, stage_wheel
from .ml_job_contracts import (
    _TRANSPORT_KEY,
    SnowflakeMLJobConfigurationError,
    SnowflakeMLJobDiagnostic,
    SnowflakeMLJobError,
    SnowflakeMLJobFailed,
    SnowflakeMLJobPermissionError,
    SnowflakeMLJobPreflightError,
    SnowflakeMLJobProgress,
    SnowflakeMLJobResult,
    SnowflakeMLJobStatus,
    SnowflakeMLJobSubmissionError,
    SnowflakeMLJobTransientError,
    _parse_transport_node,
    _published_model_from_payload,
    _published_model_payload,
    _transport_node,
)
from .sql import (
    declared_qualified_name,
    qualified_name,
    resolve_qualified_name,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_JOB_PREFIX = 'KUMO_SDK_'
_RESERVED_ATTRIBUTE_PREFIX = 'kumo.'
_STATUS_MAP = {
    'BLOCKED': SnowflakeMLJobStatus.QUEUED,
    'PENDING': SnowflakeMLJobStatus.QUEUED,
    'QUEUED': SnowflakeMLJobStatus.QUEUED,
    'RUNNING': SnowflakeMLJobStatus.RUNNING,
    'DONE': SnowflakeMLJobStatus.SUCCEEDED,
    'SUCCEEDED': SnowflakeMLJobStatus.SUCCEEDED,
    'FAILED': SnowflakeMLJobStatus.FAILED,
    'FAILED_WITH_ERROR': SnowflakeMLJobStatus.FAILED,
    'INTERNAL_ERROR': SnowflakeMLJobStatus.FAILED,
    'DISCONNECTED': SnowflakeMLJobStatus.FAILED,
    'ABORTING': SnowflakeMLJobStatus.CANCELLING,
    'CANCELLING': SnowflakeMLJobStatus.CANCELLING,
    'ABORTED': SnowflakeMLJobStatus.CANCELLED,
    'CANCELLED': SnowflakeMLJobStatus.CANCELLED,
    'DELETED': SnowflakeMLJobStatus.DELETED,
}
_TRANSIENT_CONNECTOR_ERRORS = frozenset({
    'BadGatewayError',
    'GatewayTimeoutError',
    'InterfaceError',
    'InternalServerError',
    'OperationalError',
    'OtherHTTPRetryableError',
    'RequestExceedMaxRetryError',
    'RequestTimeoutError',
    'ServiceUnavailableError',
})


def _load_jobs_api() -> Any:
    try:
        from snowflake.ml import jobs
    except ImportError as error:
        raise SnowflakeMLJobConfigurationError(
            'Snowflake ML Jobs requires snowflake-ml-python >= 1.51,<2; install '
            'the kumo-engine[snowflake] runtime') from error
    return jobs


def _active_session() -> Any:
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except Exception as error:
        raise SnowflakeMLJobConfigurationError(
            'No active Snowpark session is available; pass session= at '
            'submission or reattachment time') from error


def _exception_chain(error: BaseException) -> list[BaseException]:
    pending = [error]
    found: list[BaseException] = []
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        found.append(candidate)
        # Implicit context can describe an unrelated error being handled.
        for attribute in ('__cause__', 'conn_error'):
            nested = getattr(candidate, attribute, None)
            if isinstance(nested, BaseException):
                pending.append(nested)
    return found


def _exception_kind(error: BaseException) -> str:
    candidates = _exception_chain(error)
    sql_exception_names = {
        'SnowparkSQLException',
        'DatabaseError',
        'ProgrammingError',
        'OperationalError',
        'SnowparkClientException',
    }
    # Authorization failures must never become retryable because a wrapper
    # also carries a transport error.
    if any(
            isinstance(candidate, PermissionError)
            or str(getattr(candidate, 'sqlstate', '') or '') == '42501' or (
                type(candidate).__name__ in sql_exception_names and any(
                    fragment in str(candidate).lower()
                    for fragment in ('insufficient privilege', 'access denied',
                                     'not authorized')))
            for candidate in candidates):
        return 'permission'
    if any(
            isinstance(candidate, (ConnectionError, TimeoutError))
            or type(candidate).__name__ in _TRANSIENT_CONNECTOR_ERRORS
            or str(getattr(candidate, 'sqlstate', '') or '').startswith('08')
            for candidate in candidates):
        return 'transient'
    if any(
            type(candidate).__name__ in sql_exception_names
            for candidate in candidates):
        return 'sql'
    return 'unexpected'


def _preflight_error(operation: str,
                     error: BaseException) -> SnowflakeMLJobPreflightError:
    kind = _exception_kind(error)
    if kind == 'transient':
        return SnowflakeMLJobTransientError(
            f'{operation} failed transiently; retry: {error}', retryable=True)
    if kind == 'permission':
        return SnowflakeMLJobPermissionError(
            f'{operation} was not authorized by Snowflake: {error}',
            retryable=False,
        )
    if kind == 'sql':
        return SnowflakeMLJobPreflightError(
            f'{operation} was rejected by Snowflake: {error}',
            retryable=False,
        )
    return SnowflakeMLJobPreflightError(
        f'{operation} failed unexpectedly: {error}', retryable=False)


def _diagnostic(error: BaseException) -> SnowflakeMLJobDiagnostic:
    candidates = _exception_chain(error)

    def field(name: str) -> Any:
        for candidate in reversed(candidates):
            value = getattr(candidate, name, None)
            if value is not None:
                return value
        return None

    code = field('errno')
    if not isinstance(code, int) or isinstance(code, bool):
        code = None

    def text_field(name: str) -> str | None:
        value = field(name)
        if value is None:
            return None
        try:
            return str(value)
        except Exception:
            return None

    return SnowflakeMLJobDiagnostic(
        type=type(candidates[-1]).__name__,
        message=str(error),
        code=code,
        sqlstate=text_field('sqlstate'),
        sfqid=text_field('sfqid'),
    )


def _validate_callable(function: Callable[..., Any]) -> types.FunctionType:
    _validate_function(function, set())
    return cast(types.FunctionType, function)


def _referenced_globals(function: types.FunctionType) -> dict[str, Any]:
    """Include globals read by nested functions created in the entrypoint."""
    names: set[str] = set()

    def visit(code: types.CodeType) -> None:
        names.update(instruction.argval
                     for instruction in dis.get_instructions(code)
                     if instruction.opname in {'LOAD_GLOBAL', 'LOAD_NAME'})
        for value in code.co_consts:
            if isinstance(value, types.CodeType):
                visit(value)

    visit(function.__code__)
    return {
        name: function.__globals__[name]
        for name in names if name in function.__globals__
    }


def _detach_callable(function: types.FunctionType) -> types.FunctionType:
    """Let Snowflake CloudPickle ship validated user functions by value.

    Detached functions are not their module's registered objects, so CloudPickle
    transports their code instead of requiring the original module remotely.
    Shared globals and memoized helpers preserve aliases and recursion without
    modifying the user's namespace or CloudPickle's process-wide registry.
    """
    functions: dict[types.FunctionType, types.FunctionType] = {}
    namespaces: dict[int, dict[str, Any]] = {}

    def clone(original: types.FunctionType) -> types.FunctionType:
        if original in functions:
            return functions[original]
        namespace = namespaces.setdefault(id(original.__globals__),
                                          original.__globals__.copy())
        detached = types.FunctionType(original.__code__, namespace,
                                      original.__name__, original.__defaults__,
                                      original.__closure__)
        functions[original] = detached
        detached.__module__ = original.__module__
        detached.__qualname__ = original.__qualname__
        detached.__doc__ = original.__doc__
        detached.__annotations__ = original.__annotations__.copy()
        detached.__kwdefaults__ = (None if original.__kwdefaults__ is None else
                                   original.__kwdefaults__.copy())
        detached.__dict__.update(original.__dict__)
        if hasattr(original, '__type_params__'):
            detached.__type_params__ = original.__type_params__
        for name, value in _referenced_globals(original).items():
            if (isinstance(value, types.FunctionType)
                    and value.__module__ in {original.__module__, '__main__'}):
                namespace[name] = clone(value)
        return detached

    return clone(function)


def _validate_function(function: Callable[..., Any], seen: set[int]) -> None:
    """Reject live state reachable through a CloudPickled function graph."""
    if id(function) in seen:
        return
    seen.add(id(function))
    if not isinstance(function, types.FunctionType):
        raise SnowflakeMLJobConfigurationError(
            'function must be a Python function, not a bound method, partial, '
            'callable object, or backend instance')
    if function.__name__ == '<lambda>' or '<locals>' in function.__qualname__:
        raise SnowflakeMLJobConfigurationError(
            'function must be defined at module or notebook top level; '
            'lambdas and nested functions are not durable entry points')
    closure = inspect.getclosurevars(function)
    if closure.nonlocals:
        raise SnowflakeMLJobConfigurationError(
            'function captures nonlocal values; pass JSON-safe values through '
            'args=/kwargs= instead and construct sessions/backends inside the '
            'submitted function')
    for name, value in _referenced_globals(function).items():
        if isinstance(value, types.ModuleType):
            continue
        if isinstance(value, types.FunctionType):
            # Same-module helpers travel by value with the entrypoint, so
            # inspect their reachable globals recursively. Other imported
            # functions remain references to remote runtime dependencies.
            if value.__module__ in {function.__module__, '__main__'}:
                _validate_function(value, seen)
            continue
        if isinstance(value, type):
            if value.__module__ in {function.__module__, '__main__'}:
                raise SnowflakeMLJobConfigurationError(
                    f'function references user-defined type {name!r}; import '
                    'the type inside the submitted function or pass JSON-safe '
                    'constructor arguments instead')
            continue
        try:
            _json_input(value, f'function global {name!r}')
        except SnowflakeMLJobConfigurationError as error:
            raise SnowflakeMLJobConfigurationError(
                f'function references unsafe global {name!r} of type '
                f'{type(value).__name__}; import modules and construct '
                'sessions, credentials, backends, locks, and open files '
                'inside the submitted function') from error
    signature = inspect.signature(function)
    # Signature binding proves supplied arguments fit without executing code.
    # Actual values are separately restricted to JSON, preventing a session or
    # lock from entering Snowflake's CloudPickle argument stream.
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL
           for parameter in signature.parameters.values()):
        raise SnowflakeMLJobConfigurationError(
            'variadic positional parameters are unsupported; use explicit '
            'parameters with args=/kwargs=')
    for parameter in signature.parameters.values():
        if parameter.default is inspect.Parameter.empty:
            continue
        try:
            _json_input(parameter.default,
                        f'default for parameter {parameter.name!r}')
        except SnowflakeMLJobConfigurationError as error:
            raise SnowflakeMLJobConfigurationError(
                f'function parameter {parameter.name!r} captures an unsafe '
                'default; pass the value explicitly as JSON-safe metadata') \
                from error
    for name, value in vars(function).items():
        try:
            _json_input(value, f'function attribute {name!r}')
        except SnowflakeMLJobConfigurationError as error:
            raise SnowflakeMLJobConfigurationError(
                f'function attribute {name!r} contains unsafe live state') \
                from error


def _json_input(value: Any, path: str) -> Any:
    if isinstance(value, PublishedModel):
        return _transport_node('published_model',
                               _published_model_payload(value))
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SnowflakeMLJobConfigurationError(
                f'{path} contains a non-finite float')
        return value
    if isinstance(value, (list, tuple)):
        return [
            _json_input(item, f'{path}[{index}]')
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SnowflakeMLJobConfigurationError(
                    f'{path} contains non-string key {key!r}')
            result[key] = _json_input(item, f'{path}.{key}')
        if _TRANSPORT_KEY in result:
            return _transport_node('mapping', result)
        return result
    raise SnowflakeMLJobConfigurationError(
        f'{path} contains unsupported {type(value).__name__}; only JSON-safe '
        'arguments may cross the ML Job boundary')


def _normalize_status(raw: Any) -> SnowflakeMLJobStatus:
    value = getattr(raw, 'value', raw)
    normalized = _STATUS_MAP.get(str(value).upper())
    if normalized is None:
        raise SnowflakeMLJobError(
            f'Snowflake returned unrecognized ML Job status {raw!r}; '
            'upgrade kumo-snowflake before polling again')
    return normalized


def _require_gpu_pool(session: Any, compute_pool: str) -> None:
    try:
        expected, = resolve_qualified_name(compute_pool, parts=1)
    except ValueError as error:
        raise SnowflakeMLJobConfigurationError(
            f'invalid compute_pool identifier {compute_pool!r}') from error
    literal = expected.replace("'", "''")
    rows = session.sql(f"SHOW COMPUTE POOLS LIKE '{literal}'").collect()
    matching = [
        row_dict(row) for row in rows
        if str(row_dict(row).get('name', '')) == expected
    ]
    if len(matching) != 1:
        raise SnowflakeMLJobConfigurationError(
            f'compute pool {compute_pool!r} was not found or is not visible '
            'to the active Snowflake role')
    family = str(matching[0].get('instance_family', '')).upper()
    if not family.startswith('GPU_'):
        raise SnowflakeMLJobConfigurationError(
            f'compute pool {compute_pool!r} is not GPU-backed '
            f'(instance family {family or "unknown"})')


def _validate_tag_names(session: Any, tags: Mapping[str,
                                                    str]) -> dict[str, str]:
    canonical: dict[str, str] = {}
    for name, value in tags.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise SnowflakeMLJobConfigurationError(
                'tags must map fully-qualified Snowflake tag names to strings')
        if name.lower().startswith(_RESERVED_ATTRIBUTE_PREFIX):
            raise SnowflakeMLJobConfigurationError(
                f'tag {name!r} uses the reserved Kumo attribution namespace')
        try:
            database, schema, tag = resolve_qualified_name(name, parts=3)
        except ValueError as error:
            raise SnowflakeMLJobConfigurationError(
                f'tag {name!r} must be a DB.SCHEMA.TAG identifier') from error
        rendered = declared_qualified_name(name, parts=3)
        schema_name = qualified_name(database, schema)
        rows = session.sql(f'SHOW TAGS IN SCHEMA {schema_name}').collect()
        visible = [
            row_dict(row) for row in rows
            if str(row_dict(row).get('name', '')) == tag
        ]
        if len(visible) != 1:
            raise SnowflakeMLJobConfigurationError(
                f'Snowflake tag {name!r} does not exist or is not visible to '
                'the active role')
        canonical[rendered] = value
    return canonical


def _set_tags(session: Any, service_id: str, tags: Mapping[str, str]) -> None:
    if not tags:
        return
    try:
        service_name = declared_qualified_name(service_id, parts=(1, 2, 3))
    except ValueError as error:
        raise SnowflakeMLJobSubmissionError(
            f'Snowflake returned unsafe service ID {service_id!r}',
            retryable=False,
            service_id=service_id,
        ) from error

    def literal(value: str) -> str:
        # Snowflake interprets backslash escapes inside single-quoted strings.
        escaped = value.replace('\\', '\\\\').replace("'", "''")
        return f"'{escaped}'"

    assignments = ', '.join(f'{name} = {literal(value)}'
                            for name, value in tags.items())
    session.sql(f'ALTER SERVICE {service_name} SET TAG {assignments}').collect()


def _locations(execution_id: str) -> dict[str, str]:
    root = f'_kumo_ml_jobs/executions/{execution_id}'
    return {
        'manifest': f'{root}/manifest.json',
        'progress': f'{root}/progress.json',
        'result': f'{root}/result.json',
    }


def _preflight_stage(store: StageStore, execution_id: str) -> None:
    """Exercise the exact stage operations used by submitted jobs."""
    probe_root = f'_kumo_ml_jobs/preflight/{execution_id}'
    primary: BaseException | None = None
    try:
        store.put_json(f'{probe_root}/source.json', {'ok': True})
        store.copy_file(f'{probe_root}/source.json', f'{probe_root}/copy')
        copied = store.get_json(f'{probe_root}/copy/source.json')
        if copied != {'ok': True}:
            raise RuntimeError('stage copy/read preflight returned wrong data')
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            store.remove(probe_root)
        except Exception:
            if primary is None:
                raise
            logger.warning(
                'Could not clean Snowflake ML Job preflight objects below %s',
                store.uri(probe_root),
                exc_info=True,
            )


def _service_index(service_id: str) -> str:
    digest = sha256(service_id.encode('utf-8')).hexdigest()
    return f'_kumo_ml_jobs/services/{digest}.json'


def _parse_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError('expected an ISO datetime string')
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('expected a timezone-aware ISO datetime string')
    return parsed


def _required_string(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise SnowflakeMLJobConfigurationError(
            f'Snowflake ML Job metadata has no valid {name!r}')
    return value


def _decode_result_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_result_value(item) for item in value]
    if isinstance(value, Mapping):
        try:
            node = _parse_transport_node(value)
        except ValueError as error:
            raise SnowflakeMLJobError(str(error)) from error
        if node is not None:
            kind, payload = node
            if kind == 'mapping':
                return {
                    str(key): _decode_result_value(item)
                    for key, item in payload.items()
                }
            if kind == 'published_model':
                return _published_model_from_payload(payload)
            raise SnowflakeMLJobError(
                f'unsupported Kumo result transport {kind!r}')
        else:
            return {
                str(key): _decode_result_value(item)
                for key, item in value.items()
            }
    return value


def _parse_diagnostic(value: Any) -> SnowflakeMLJobDiagnostic | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SnowflakeMLJobError('diagnostic must be an object or null')

    def optional_text(name: str) -> str | None:
        field = value.get(name)
        if field is not None and not isinstance(field, str):
            raise SnowflakeMLJobError(
                f'diagnostic {name!r} must be a string or null')
        return field

    code = value.get('code')
    if code is not None and (not isinstance(code, int)
                             or isinstance(code, bool)):
        raise SnowflakeMLJobError('diagnostic code must be an integer or null')
    error_type = value.get('type', 'RemoteError')
    message = value.get('message', '')
    if not isinstance(error_type, str) or not isinstance(message, str):
        raise SnowflakeMLJobError('diagnostic type and message must be strings')
    return SnowflakeMLJobDiagnostic(
        type=error_type,
        message=message,
        code=code,
        sqlstate=optional_text('sqlstate'),
        sfqid=optional_text('sfqid'),
        traceback=optional_text('traceback'),
    )


class SnowflakeMLJob:
    """Reconnectable, deliberately non-thread-safe ML Job observer."""
    def __init__(
        self,
        service_id: str,
        *,
        stage_name: str,
        execution_id: str,
        progress_location: str,
        result_location: str,
        session: Any,
        jobs_api: Any | None = None,
        handle: Any | None = None,
    ) -> None:
        self.service_id = service_id
        self.execution_id = execution_id
        self.stage_name = validate_stage_name(stage_name)
        self.progress_location = progress_location
        self.result_location = result_location
        self._jobs_api = jobs_api
        self._handle = handle
        self._session: Any | None = session
        self._closed = False

    def _store(self) -> StageStore:
        if self._closed:
            raise RuntimeError('SnowflakeMLJob handle is closed')
        if self._session is None:
            raise RuntimeError('SnowflakeMLJob handle is closed')
        return StageStore(self._session, self.stage_name)

    def _remote(self) -> Any:
        if self._closed:
            raise RuntimeError('SnowflakeMLJob handle is closed')
        if self._session is None:
            raise RuntimeError('SnowflakeMLJob handle is closed')
        if self._handle is None:
            jobs_api = self._jobs_api or _load_jobs_api()
            self._handle = jobs_api.get_job(self.service_id,
                                            session=self._session)
        return self._handle

    def _service_status(self) -> SnowflakeMLJobStatus:
        handle = self._remote()
        raw = getattr(handle, 'status', None)
        raw = raw() if callable(raw) else raw
        return _normalize_status(raw)

    def status(self) -> SnowflakeMLJobStatus:
        """Return normalized Snowflake service status without collecting result."""
        try:
            durable = self._store().get_json(self.result_location,
                                             missing_ok=True)
        except Exception:
            durable = None
            logger.warning(
                'Could not read durable status for %s; querying the service',
                self.service_id,
                exc_info=True,
            )
        if isinstance(durable, Mapping):
            return self._result_from_payload(durable).status
        try:
            return self._service_status()
        except Exception as service_error:
            try:
                durable = self._store().get_json(self.result_location,
                                                 missing_ok=True)
            except Exception:
                logger.warning(
                    'Durable status fallback also failed for %s',
                    self.service_id,
                    exc_info=True,
                )
                raise service_error
            if isinstance(durable, Mapping):
                return self._result_from_payload(durable).status
            raise

    def progress(self) -> SnowflakeMLJobProgress:
        """Return the latest native Kumo progress and stage events."""
        store = self._store()
        try:
            payload = store.get_json(self.progress_location, missing_ok=True)
        except Exception:
            payload = None
            logger.warning('Could not read progress for %s; querying status',
                           self.service_id, exc_info=True)
        try:
            durable = store.get_json(self.result_location, missing_ok=True)
        except Exception:
            durable = None
            logger.warning('Could not read terminal progress for %s',
                           self.service_id, exc_info=True)
        durable_result: SnowflakeMLJobResult | None = None
        if isinstance(durable, Mapping):
            durable_result = self._result_from_payload(durable)
            status = durable_result.status
        else:
            try:
                status = self._service_status()
            except Exception:
                # Recheck durable state if service history has disappeared
                # or its status endpoint is temporarily unavailable.
                status = self.status()
        now = datetime.now(timezone.utc)
        terminal_at = (durable_result.finished_at if durable_result is not None
                       and durable_result.finished_at is not None else now)
        if not isinstance(payload, Mapping):
            job_status = _job_status(status)
            event = JobEventLogEntry(
                stage_name=status.value.title(),
                last_updated_at=terminal_at if job_status.is_terminal else now,
            )
            return SnowflakeMLJobProgress(
                execution_id=self.execution_id,
                status_report=JobStatusReport(
                    status=job_status,
                    tracking_url='',
                    start_time=now,
                    end_time=terminal_at if job_status.is_terminal else None,
                    event_log=[event],
                ),
                updated_at=now,
            )
        if payload.get('schema_version') != _SCHEMA_VERSION:
            raise SnowflakeMLJobError('unsupported progress payload version')
        report = from_json(payload['status_report'], JobStatusReport)
        terminal_status = _job_status(status)
        if terminal_status.is_terminal and report.status != terminal_status:
            events = list(report.event_log)
            stage = terminal_status.value.title()
            if not events or events[-1].stage_name != stage:
                events.append(
                    JobEventLogEntry(stage_name=stage,
                                     last_updated_at=terminal_at))
            report = JobStatusReport(
                status=terminal_status,
                tracking_url='',
                start_time=report.start_time,
                end_time=terminal_at,
                event_log=events,
                validation_response=report.validation_response,
            )
        return SnowflakeMLJobProgress(
            execution_id=self.execution_id,
            status_report=report,
            updated_at=_parse_datetime(payload['updated_at']),
            operation_kind=payload.get('operation_kind'),
            operation_id=payload.get('operation_id'),
            training=(None if payload.get('training') is None else from_json(
                payload['training'], AutoTrainerProgress)),
            prediction=(None if payload.get('prediction') is None else
                        from_json(payload['prediction'], PredictionProgress)),
            current_metrics=payload.get('current_metrics') or None,
            best_metrics=payload.get('best_metrics') or None,
        )

    def logs(self, limit: int = -1) -> str:
        """Return ordinary user logs from the Snowflake ML Job head node."""
        if limit == 0 or limit < -1:
            raise ValueError('limit must be -1 or a positive integer')
        handle = self._remote()
        return str(handle.get_logs(limit=limit, verbose=False))

    def cancel(self) -> None:
        """Request cancellation of the Snowflake ML Job service."""
        handle = self._remote()
        handle.cancel()

    def result(self, timeout_s: float | None = None) -> SnowflakeMLJobResult:
        """Return durable terminal state without misclassifying client errors."""
        if timeout_s is not None and timeout_s < 0:
            raise ValueError('timeout_s must be non-negative or None')
        store = self._store()
        durable = store.get_json(self.result_location, missing_ok=True)
        if isinstance(durable, Mapping):
            return self._result_from_payload(durable)

        handle = self._remote()
        timeout = -1 if timeout_s is None else timeout_s
        try:
            value = handle.result(timeout=timeout)
        except TimeoutError:
            raise
        except Exception as error:
            try:
                status = self.status()
            except Exception:
                raise error
            if status in {
                    SnowflakeMLJobStatus.FAILED,
                    SnowflakeMLJobStatus.CANCELLED,
                    SnowflakeMLJobStatus.DELETED,
            }:
                durable = store.get_json(self.result_location, missing_ok=True)
                if isinstance(durable, Mapping):
                    return self._result_from_payload(durable)
                return SnowflakeMLJobResult(
                    execution_id=self.execution_id,
                    service_id=self.service_id,
                    status=status,
                    diagnostic=_diagnostic(error),
                    finished_at=datetime.now(timezone.utc),
                )
            # RUNNING/QUEUED or a transient read after SUCCEEDED is not a
            # remote failure. Let the client exception remain retryable.
            raise
        if not isinstance(value, Mapping):
            raise SnowflakeMLJobError(
                'Snowflake returned an invalid Kumo terminal envelope')
        return self._result_from_payload(value)

    def _result_from_payload(
            self, payload: Mapping[str, Any]) -> SnowflakeMLJobResult:
        if payload.get('schema_version') != _SCHEMA_VERSION:
            raise SnowflakeMLJobError('unsupported terminal result version')
        if payload.get('execution_id') != self.execution_id:
            raise SnowflakeMLJobError(
                'terminal result belongs to a different Kumo execution')
        status = _normalize_status(payload.get('status'))
        if not status.terminal:
            raise SnowflakeMLJobError(
                'terminal result contains a non-terminal status')
        value = _decode_result_value(payload.get('value'))
        published_model_payload = payload.get('published_model')
        if published_model_payload is None:
            published_model = None
        elif isinstance(published_model_payload, Mapping):
            published_model = _published_model_from_payload(
                published_model_payload)
        else:
            raise SnowflakeMLJobError('published_model must be an object')
        progress_payload = payload.get('progress')
        progress_kind = payload.get('progress_kind')
        if progress_payload is None:
            progress = None
        elif progress_kind == 'training':
            progress = from_json(progress_payload, AutoTrainerProgress)
        elif progress_kind == 'prediction':
            progress = from_json(progress_payload, PredictionProgress)
        else:
            raise SnowflakeMLJobError(
                'terminal result has progress without a supported kind')
        status_report_payload = payload.get('status_report')
        status_report = (None if status_report_payload is None else from_json(
            status_report_payload, JobStatusReport))
        if (status_report is not None
                and status_report.status != _job_status(status)):
            raise SnowflakeMLJobError(
                'terminal result status_report disagrees with its status')
        return SnowflakeMLJobResult(
            execution_id=self.execution_id,
            service_id=self.service_id,
            status=status,
            value=value,
            published_model=published_model,
            progress=progress,
            status_report=status_report,
            diagnostic=_parse_diagnostic(payload.get('diagnostic')),
            finished_at=_parse_datetime(payload['finished_at']),
        )

    def attach(self, interval_s: float = 4.0) -> SnowflakeMLJobResult:
        """Render stages and intermediate Kumo progress until termination."""
        if interval_s <= 0:
            raise ValueError('interval_s must be positive')
        from tqdm.auto import tqdm

        print(f'Attaching to Snowflake ML Job {self.service_id}.')
        current_stage: str | None = None
        bar: Any | None = None
        try:
            while True:
                try:
                    progress = self.progress()
                    events = progress.status_report.event_log
                    stage = (events[-1].stage_name if events else
                             progress.status_report.status.value.title())
                    if stage != current_stage:
                        if bar is not None:
                            bar.close()
                            bar = None
                        print(f'Current stage: {stage}')
                        current_stage = stage

                    completed, total, detail = _progress_units(progress)
                    if total > 0:
                        if bar is None:
                            bar = tqdm(total=total, desc=stage,
                                       dynamic_ncols=True)
                        bar.total = total
                        bar.update(max(0, completed - bar.n))
                        if detail:
                            bar.set_postfix_str(detail)
                        bar.refresh()
                    if progress.status_report.status.is_terminal:
                        return self.result()
                except Exception as error:
                    if _exception_kind(error) != 'transient':
                        raise
                    logger.warning(
                        'Temporary error observing %s; retrying in %.1fs: %s',
                        self.service_id, interval_s, error)
                time.sleep(interval_s)
        finally:
            if bar is not None:
                bar.close()

    def close(self) -> None:
        """Release references held by this observer.

        Snowpark sessions supplied explicitly or resolved as active sessions
        are borrowed and are never closed by the job handle.
        """
        if self._closed:
            return
        self._handle = None
        self._session = None
        self._closed = True


def _job_status(status: SnowflakeMLJobStatus) -> JobStatus:
    return {
        SnowflakeMLJobStatus.QUEUED: JobStatus.QUEUED,
        SnowflakeMLJobStatus.RUNNING: JobStatus.RUNNING,
        SnowflakeMLJobStatus.CANCELLING: JobStatus.RUNNING,
        SnowflakeMLJobStatus.SUCCEEDED: JobStatus.DONE,
        SnowflakeMLJobStatus.FAILED: JobStatus.FAILED,
        SnowflakeMLJobStatus.CANCELLED: JobStatus.CANCELLED,
        SnowflakeMLJobStatus.DELETED: JobStatus.FAILED,
    }[status]


def _progress_units(
        progress: SnowflakeMLJobProgress) -> tuple[float, float, str]:
    training = progress.training
    if training is not None:
        total = float(training.estimated_training_time or training.total_trials)
        completed = (float(training.elapsed_training_time)
                     if training.estimated_training_time else float(
                         training.completed_trials))
        detail = (
            f'{training.completed_trials}/{training.total_trials} trials, '
            f'{training.elapsed_training_time:.0f}s elapsed')
        if progress.current_metrics:
            detail += ', ' + ', '.join(
                f'{name}={value:.4g}'
                for name, value in sorted(progress.current_metrics.items()))
        return completed, total, detail
    prediction = progress.prediction
    if prediction is not None:
        return (
            float(prediction.completed_iterations),
            float(prediction.total_iterations),
            f'{prediction.completed_iterations}/{prediction.total_iterations} iterations',
        )
    return 0.0, 0.0, ''


def _cancel_preserving_primary(handle: Any, primary: BaseException) -> None:
    try:
        handle.cancel()
    except Exception as cancel_error:
        logger.error(
            'Cancelling an incompletely attached ML Job also failed: %s',
            cancel_error)
        raise SnowflakeMLJobSubmissionError(
            f'ML Job attachment failed ({primary}) and cancellation also '
            f'failed ({cancel_error})',
            retryable=False,
            service_id=str(getattr(handle, 'id', '') or '') or None,
        ) from primary


def submit_ml_job(
    function: Callable[..., Any],
    *,
    compute_pool: str,
    stage_name: str,
    runtime_environment: str | None = None,
    tags: Mapping[str, str] | None = None,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    session: Any | None = None,
    target_instances: int = 1,
    min_instances: int | None = None,
    query_warehouse: str | None = None,
    external_access_integrations: Sequence[str] | None = None,
    artifact_repositories: Sequence[str] | None = None,
) -> SnowflakeMLJob:
    """Submit a user-owned top-level function as one Snowflake ML Job.

    Function dispatch uses Snowflake ML Jobs' own CloudPickle transport.  Kumo
    deliberately accepts only top-level functions and JSON-safe arguments (or
    a :class:`~kumoapi.plugin.PublishedModel` returned by an earlier submitted
    job);
    construct runtime clients, sessions, locks, credentials, and open files
    inside ``function`` rather than capturing or passing them.
    """
    function = _validate_callable(function)
    submitted_args = list(args)
    submitted_kwargs = dict(kwargs or {})
    try:
        inspect.signature(function).bind(*submitted_args, **submitted_kwargs)
    except TypeError as error:
        raise SnowflakeMLJobConfigurationError(
            f'args/kwargs do not match {function.__qualname__}: {error}'
        ) from error
    safe_args = _json_input(submitted_args, 'args')
    safe_kwargs = _json_input(submitted_kwargs, 'kwargs')
    if target_instances < 1:
        raise SnowflakeMLJobConfigurationError(
            'target_instances must be positive')
    if min_instances is not None and not 1 <= min_instances <= target_instances:
        raise SnowflakeMLJobConfigurationError(
            'min_instances must be between 1 and target_instances')
    if runtime_environment is not None and (not isinstance(
            runtime_environment, str) or not runtime_environment.strip()):
        raise SnowflakeMLJobConfigurationError(
            'runtime_environment must be a non-empty string when supplied')
    stage_name = validate_stage_name(stage_name)
    submit_session = session if session is not None else _active_session()
    if submit_session is None:
        raise SnowflakeMLJobConfigurationError(
            'Snowflake session resolution returned no session')

    execution_id = uuid.uuid4().hex
    definition_name = f'{_JOB_PREFIX}{execution_id[:20].upper()}'
    locations = _locations(execution_id)
    store = StageStore(submit_session, stage_name)
    jobs_api = _load_jobs_api()
    try:
        _require_gpu_pool(submit_session, compute_pool)
        _preflight_stage(store, execution_id)
        try:
            wheel = discover_installed_wheel()
        except (OSError, RuntimeError, ValueError) as error:
            raise SnowflakeMLJobConfigurationError(
                f'Kumo fat-wheel discovery failed: {error}') from error
        staged = stage_wheel(store, wheel)
        validated_tags = _validate_tag_names(submit_session, tags or {})
        options: dict[str, Any] = {
            'stage_name': stage_name,
            'target_instances': target_instances,
            'min_instances': min_instances or target_instances,
            'pip_requirements': [f'./{staged.name}'],
            'session': submit_session,
            'name': definition_name,
            'env_vars': {
                'KUMO_SUBMITTED_BY': 'kumo-sdk',
                'KUMO_EXECUTION_ID': execution_id,
            },
        }
        optional = {
            'runtime_environment':
            runtime_environment,
            'query_warehouse':
            query_warehouse,
            'external_access_integrations':
            (list(external_access_integrations)
             if external_access_integrations is not None else None),
            'artifact_repositories':
            (list(artifact_repositories)
             if artifact_repositories is not None else None),
        }
        options.update({
            key: value
            for key, value in optional.items() if value is not None
        })
        definition = jobs_api.remote(compute_pool,
                                     **options)(execute_user_function)
    except SnowflakeMLJobConfigurationError:
        raise
    except (TypeError, ValueError) as error:
        raise SnowflakeMLJobConfigurationError(str(error)) from error
    except Exception as error:
        raise _preflight_error('Snowflake ML Job preflight', error) from error

    manifest = {
        'schema_version': _SCHEMA_VERSION,
        'execution_id': execution_id,
        'service_id': None,
        'submission_state': 'submitting',
        'stage_name': stage_name,
        'progress_location': locations['progress'],
        'result_location': locations['result'],
        'callable': {
            'module': function.__module__,
            'qualname': function.__qualname__,
        },
        'definition_name': definition_name,
        'attribution': {
            'kumo.submitted_by': 'kumo-sdk',
            'kumo.execution_id': execution_id,
        },
        'tags': dict(validated_tags),
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    dispatch_started = False
    try:
        store.put_json(locations['manifest'], manifest)
        # Function dispatch writes its app below this predictable definition
        # directory. Seed the verified wheel server-side; the runtime installs
        # exactly this one relative requirement.
        store.copy_file(
            f'_kumo_ml_jobs/wheels/sha256/{staged.sha256}/{staged.name}',
            f'{definition_name}/app',
        )
        config = {
            'schema_version': _SCHEMA_VERSION,
            'execution_id': execution_id,
            'stage_name': stage_name,
            'progress_location': locations['progress'],
            'result_location': locations['result'],
        }
        detached = _detach_callable(function)
        dispatch_started = True
        handle = definition(detached, safe_args, safe_kwargs, config)
    except Exception as error:
        state = 'dispatch_unknown' if dispatch_started else 'failed'
        failed = {**manifest, 'submission_state': state, 'error': str(error)}
        try:
            store.put_json(locations['manifest'], failed)
        except Exception:
            logger.exception('Could not persist failed submission manifest')
        if not dispatch_started and isinstance(error, SnowflakeMLJobError):
            raise
        if dispatch_started:
            message = (
                f'Snowflake ML Job dispatch outcome is unknown for definition '
                f'{definition_name!r}, execution {execution_id!r}. Inspect '
                f'Snowflake services and {stage_name}/{locations["manifest"]} '
                f'before submitting again; a GPU job may already be running: '
                f'{error}')
        else:
            message = f'Snowflake ML Job preparation failed: {error}'
        raise SnowflakeMLJobSubmissionError(
            message,
            retryable=(not dispatch_started
                       and _exception_kind(error) == 'transient'),
        ) from error

    service_id = str(getattr(handle, 'id', '') or '')
    if not service_id:
        _cancel_preserving_primary(
            handle, RuntimeError('Snowflake returned no ML Job service ID'))
        raise SnowflakeMLJobSubmissionError(
            'Snowflake returned an ML Job without a service ID',
            retryable=False,
        )
    submitted = {
        **manifest, 'service_id': service_id,
        'submission_state': 'submitted'
    }
    try:
        _set_tags(submit_session, service_id, validated_tags)
        store.put_json(locations['manifest'], submitted)
        store.put_json(_service_index(service_id), submitted)
    except Exception as error:
        failed = {
            **submitted, 'submission_state': 'failed',
            'error': str(error)
        }
        try:
            store.put_json(locations['manifest'], failed)
            store.put_json(_service_index(service_id), failed)
        except Exception:
            logger.exception('Could not preserve failed ML Job attachment')
        _cancel_preserving_primary(handle, error)
        raise SnowflakeMLJobSubmissionError(
            f'Snowflake ML Job {service_id!r} was cancelled because applying '
            f'tags or persisting reattachment metadata failed: {error}',
            retryable=False,
            service_id=service_id,
        ) from error

    job = SnowflakeMLJob(
        service_id,
        stage_name=stage_name,
        execution_id=execution_id,
        progress_location=locations['progress'],
        result_location=locations['result'],
        session=submit_session,
        jobs_api=jobs_api,
        handle=handle,
    )
    return job


def get_ml_job(
    service_id: str,
    *,
    stage_name: str,
    session: Any | None = None,
) -> SnowflakeMLJob:
    """Reattach to a prior Kumo-submitted Snowflake ML Job."""
    stage_name = validate_stage_name(stage_name)
    lookup_session = session if session is not None else _active_session()
    store = StageStore(lookup_session, stage_name)
    try:
        manifest = store.get_json(_service_index(service_id))
    except Exception as error:
        raise _preflight_error('Snowflake ML Job reattachment',
                               error) from error
    if not isinstance(manifest, Mapping):
        raise SnowflakeMLJobConfigurationError(
            f'No Kumo reattachment metadata exists for {service_id!r} under '
            f'{stage_name}')
    if manifest.get('schema_version') != _SCHEMA_VERSION:
        raise SnowflakeMLJobConfigurationError(
            'unsupported Snowflake ML Job manifest version')
    if manifest.get('service_id') != service_id:
        raise SnowflakeMLJobConfigurationError(
            'reattachment metadata belongs to a different Snowflake service')
    execution_id = _required_string(manifest, 'execution_id')
    progress_location = _required_string(manifest, 'progress_location')
    result_location = _required_string(manifest, 'result_location')
    return SnowflakeMLJob(
        service_id,
        stage_name=stage_name,
        execution_id=execution_id,
        progress_location=progress_location,
        result_location=result_location,
        session=lookup_session,
    )


__all__ = [
    'SnowflakeMLJob',
    'SnowflakeMLJobConfigurationError',
    'SnowflakeMLJobDiagnostic',
    'SnowflakeMLJobError',
    'SnowflakeMLJobFailed',
    'SnowflakeMLJobPermissionError',
    'SnowflakeMLJobPreflightError',
    'SnowflakeMLJobProgress',
    'SnowflakeMLJobResult',
    'SnowflakeMLJobStatus',
    'SnowflakeMLJobSubmissionError',
    'SnowflakeMLJobTransientError',
    'get_ml_job',
    'submit_ml_job',
]
