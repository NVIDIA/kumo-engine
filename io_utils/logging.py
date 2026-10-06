# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Kumo's internal wrapper of the standard Python logging library.

Description
-----------
This library defines logging methods to be used throughout Kumo. There
are a few important details that a user should be aware of before working with
logging utilities.

LoggingMode. The types of logging modes are defined in `class LoggingMode`,
which specifies the level at which logs will be output both to the console
and to the specified rotating log file. This mode is intended to be set once
in `initialize_logging(mode: LoggingMode)` in a runner script.

getLogger(name: str). A thin wrapper over logging.getLogger(name: str) that
simply returns a logger. This is intended to be called at the top of python
modules as `logger = logging.getLogger(__name__)`; logger can then be used
throughout the module.

Notes
-----
DO NOT construct log messages eagerly; that is, use printf-style message format
strings instead of f-strings when passing log strings to loggers. This is
because f-strings are initialized eagerly and incur the cost of initialization
_regardless of whether logs are printed_, whereas printf-style formats are
initialized lazily and therefore do not incur this cost if logs aren't printed.
"""
import atexit
import json
import logging
import logging.config
import logging.handlers
import multiprocessing
import multiprocessing.util
import os
import os.path as osp
import queue
import sys
import threading
import time
import traceback
import warnings
from copy import copy
from dataclasses import asdict
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    TypeAlias,
    Union,
    cast,
)

from logging_loki.emitter import BasicAuth

from io_utils import kumo_dataclass

CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG

Logger: TypeAlias = logging.Logger

LOGGING_DIR = ""
LOGGING_FILE_NAME = 'kumo.log'
LOGGING_INITIALIZED = False
LOG_GROUP_NAME = ""
LOG_STREAM_NAME = ""
_PYTEST_ENABLE_FSUTIL_DEBUG_LOGS_ENV = 'KUMO_PYTEST_ENABLE_FSUTIL_DEBUG_LOGS'


def _optional_bool_env(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if not raw_value:
        return default
    normalized = raw_value.strip().lower()
    if normalized in ('1', 'enabled', 'true', 'yes', 'on'):
        return True
    if normalized in ('0', 'disabled', 'false', 'no', 'off'):
        return False
    raise ValueError(f'Unable to parse bool value from {raw_value}')


_LOKI_EXPORT = _optional_bool_env('LOKI_EXPORT')
_LOKI_URL = os.environ.get('LOKI_URL', '')
_LOKI_USER = os.environ.get('LOKI_USER', '')
_LOKI_PASSWORD = os.environ.get('LOKI_PASSWORD', '')
# The tags are used to filter logs on grafana.
# It should be specified as a serialized dict object.
# An example to set this: {"cluster": "saas"}.
# There's no limit on how many tags we can specify, so we can
# filter logs by combination of different tags on grafana.
_LOKI_TAGS = os.environ.get('LOKI_TAGS', '')


def warn_or_raise(msg: str, raise_on_error: bool = True):
    if raise_on_error:
        raise ValueError(msg)
    else:
        warnings.warn(msg, stacklevel=2)


def _get_logging_dir() -> str:
    return LOGGING_DIR


def _set_logging_dir(dir: str) -> None:
    global LOGGING_DIR
    LOGGING_DIR = dir


def _set_logging_file_name(file_name: str) -> None:
    global LOGGING_FILE_NAME
    LOGGING_FILE_NAME = file_name


def _set_log_group_name(group_name: str):
    global LOG_GROUP_NAME
    LOG_GROUP_NAME = group_name


def _set_log_stream_name(stream_name: str):
    global LOG_STREAM_NAME
    LOG_STREAM_NAME = stream_name


class LoggingMode(Enum):
    """
    Defines logging modes: either debug, development or production.

    TODO(manan, dulloor): discuss potential of adding finer-grained verbosity
    to other modes (e.g. LoggingMode.VERBOSE_L1, etc. that all allow for more
    controlled outputs).)
    """
    DEBUG = 0
    DEVELOPMENT = 1
    PRODUCTION = 2

    @classmethod
    def to_level_str(cls, mode: "LoggingMode"):
        """Converts a LoggingMode to a logging level string."""
        return {
            LoggingMode.DEBUG: 'DEBUG',
            LoggingMode.DEVELOPMENT: 'INFO',
            LoggingMode.PRODUCTION: 'INFO',
        }[mode]


class MultiProcessingLog(logging.Handler):
    """Write records from multiple processes through one file handler."""
    class _Listener(logging.handlers.QueueListener):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stopped_cleanly = True

        def dequeue(self, block):
            try:
                return super().dequeue(block)
            except (EOFError, OSError):
                return cast(logging.LogRecord, None)

        def stop(self) -> None:
            thread = self._thread
            if thread is None:
                self.stopped_cleanly = True
                return
            try:
                self.enqueue_sentinel()
            except (OSError, ValueError):
                pass
            thread.join(_LISTENER_JOIN_TIMEOUT_S)
            self._thread = None
            self.stopped_cleanly = not thread.is_alive()

    def __init__(self, path, mode, maxsize, rotate):
        super().__init__()
        self._record_queue = multiprocessing.Queue(-1)
        self._owner_pid = os.getpid()
        self._file_handler = RotatingFileHandler(path, mode, maxsize, rotate)
        self._listener = self._Listener(self._record_queue, self._file_handler,
                                        respect_handler_level=True)
        self._listener.start()
        # Drain records before multiprocessing tears down the queue at exit.
        self._exit_finalizer = multiprocessing.util.Finalize(
            None, self.close, exitpriority=11)

    def setFormatter(self, fmt):
        super().setFormatter(fmt)
        self._file_handler.setFormatter(fmt)

    def prepare(self, record):
        prepared = copy(record)
        prepared.msg = prepared.getMessage()
        prepared.args = None
        if prepared.exc_info and not prepared.exc_text:
            prepared.exc_text = logging.Formatter().formatException(
                prepared.exc_info)
        prepared.exc_info = None
        return prepared

    def emit(self, record):
        if getattr(self, '_closed', False):
            return
        try:
            self._record_queue.put_nowait(self.prepare(record))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:  # pylint: disable=broad-except
            self.handleError(record)

    def close(self):
        if getattr(self, '_closed', False):
            return
        self._closed = True
        self._exit_finalizer.cancel()
        if os.getpid() == self._owner_pid:
            self._listener.stop()
            if self._listener.stopped_cleanly:
                self._file_handler.close()
            try:
                self._record_queue.close()
            except (OSError, ValueError):
                pass
            self._record_queue.cancel_join_thread()
        else:
            try:
                self._record_queue.close()
                self._record_queue.join_thread()
            except (OSError, ValueError):
                pass
        super().close()


@kumo_dataclass
class LoggingFormatterConfig:
    """
    (Incomplete) configuration for a logging formatter.

    Specification:
        https://docs.python.org/3/library/logging.html#formatter-objects
    """
    format: Optional[str] = None
    cls: Optional[str] = None


@kumo_dataclass
class LoggingHandlerConfig:
    """
    (Incomplete) configuration for a logging handler.

    Specification:
        https://docs.python.org/3/library/logging.html#handler-objects
    """

    # Required arguments
    cls: str  # NOTE: this is the class keyword in python logging
    level: str
    formatter: str

    # Optional file handler arguments (None values are removed prior to
    # dict creation)
    stream: Optional[str] = None
    path: Optional[str] = None
    mode: Optional[str] = None
    maxsize: Optional[int] = None
    rotate: Optional[int] = None

    # Optional cloud watch handler arguments
    log_group_name: Optional[str] = None
    log_stream_name: Optional[str] = None
    create_log_group: Optional[bool] = None
    send_interval: Optional[int] = None
    max_batch_size: Optional[int] = None
    max_message_size: Optional[int] = None

    # Optional loki handler arguments
    url: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    auth: Optional[BasicAuth] = None
    version: Optional[str] = None

    def __post_init__(self):
        if (self.stream is None and self.path is None
                and self.log_group_name is None and self.url is None):
            raise Exception(
                "Neither an output stream nor an output path nor a log "
                "group nor an export url have been set.")


# Union of LoggingMode enum; could also be int value or name of
# logging.<Level> where <Level> can be DEBUG,INFO,WARNING,ERROR,...etc
LoggingLevel = Union[LoggingMode, int, str]


@kumo_dataclass
class LoggerConfig:
    """
    (Incomplete) configuration for a logger object.

    Specification:
        https://docs.python.org/3/library/logging.html#logger-objects
    """
    level: str
    handlers: List[str]
    propagate: bool

    def override_level(self, level: LoggingLevel):
        if isinstance(level, int):
            self.level = logging._levelToName[level]
        elif isinstance(level, str):
            assert level in logging._nameToLevel
            self.level = level
        elif isinstance(level, LoggingMode):
            self.level = LoggingMode.to_level_str(level)


@kumo_dataclass
class LoggingConfig:
    """
    (Incomplete) configuration for a logger configuration. A configuration
    must specify formatters, handlers (that use the formatters), and loggers
    (that use the handlers).

    Specification:
        https://docs.python.org/3/library/logging.config.html
    """
    formatters: Dict[str, LoggingFormatterConfig]
    handlers: Dict[str, LoggingHandlerConfig]
    loggers: Dict[str, LoggerConfig]
    version: int = 1
    disable_existing_loggers: bool = False

    @classmethod
    def logging_dict_factory(cls, data):
        """Helper method for to_dict."""
        no_none = {k: v for k, v in data if v is not None}
        if 'cls' in no_none:
            no_none['class'] = no_none.pop('cls')

        return no_none

    def to_dict(self) -> Dict:
        """
        Converts a LoggingConfig object to a dictionary suitable for ingestion
        via logging.config.dictConfig(...). Namely, removes None values and
        renames cls -> class.
        """
        return asdict(self, dict_factory=LoggingConfig.logging_dict_factory)


def _get_output_format() -> str:
    """Resolve desired output format from DeployEnvConfig when available."""
    try:
        from service_framework.deploy_env_config import deploy_env
        return deploy_env().logging_config.output_format
    except Exception:  # pylint: disable=broad-except
        return 'text'


def _get_config(mode: LoggingMode, log_to_console: bool, log_to_file: bool,
                log_to_cloud_watch: bool, log_process_id: bool,
                logging_level_overrides: Mapping[str, LoggingLevel]):
    """
    Obtains a logging configuration from a LoggingMode.

    TODO(manan): Don't print line number if we're logging INFO only.
    """
    level = LoggingMode.to_level_str(mode)
    handler_names = []
    if log_to_file:
        handler_names.append('file_handler')
    if log_to_console:
        handler_names += ['console', 'error_console']
    if log_to_cloud_watch:
        handler_names += ['cloud_watch']
    if _LOKI_EXPORT:
        handler_names += ['loki']
    assert handler_names
    logger_config = LoggerConfig(
        level=level,
        handlers=handler_names,
        propagate=True,
    )
    # Define handlers that dictate where logs are printed and how messages
    # are handled
    output_format = _get_output_format()
    output_format_value = getattr(output_format, 'value', output_format)
    use_json_format = output_format_value == 'json'
    standard_formatter_name = 'json' if use_json_format else 'standard'
    error_formatter_name = 'json' if use_json_format else 'error'

    handlers = {
        'console':
        LoggingHandlerConfig(cls='logging.StreamHandler', level=level,
                             formatter=standard_formatter_name,
                             stream='ext://sys.stdout'),
        'error_console':
        LoggingHandlerConfig(cls='logging.StreamHandler', level='ERROR',
                             formatter=error_formatter_name,
                             stream='ext://sys.stderr'),
        'file_handler':
        LoggingHandlerConfig(
            cls='io_utils.logging.MultiProcessingLog',
            level=level,
            formatter=standard_formatter_name,
            path=osp.join(_get_logging_dir(), LOGGING_FILE_NAME),
            mode='a',
            maxsize=10485760,
            rotate=20,
        )
    }
    if log_to_cloud_watch:
        handlers['cloud_watch'] = LoggingHandlerConfig(
            cls='watchtower.CloudWatchLogHandler',
            level=level,
            formatter=standard_formatter_name,
            log_group_name=LOG_GROUP_NAME,
            log_stream_name=LOG_STREAM_NAME,
            create_log_group=False,
            send_interval=30,
            max_batch_size=3145728,
            max_message_size=1048576,
        )
    if _LOKI_EXPORT:
        if not _LOKI_URL:
            raise ValueError('LOKI_URL must be set when LOKI_EXPORT is enabled')
        tags = json.loads(_LOKI_TAGS)
        snowflake_account = os.environ.get('SNOWFLAKE_ACCOUNT')
        if snowflake_account:
            snowflake_database = os.environ.get('SNOWFLAKE_DATABASE')
            snowflake_schema = os.environ.get('SNOWFLAKE_SCHEMA')
            tags['snowflake_account'] = (
                f'{snowflake_account}_{snowflake_database}_'
                f'{snowflake_schema}')
        handlers['loki'] = LoggingHandlerConfig(
            cls='logging_loki.LokiHandler',
            level=level,
            formatter=standard_formatter_name,
            url=_LOKI_URL,
            tags=tags,
            auth=(_LOKI_USER, _LOKI_PASSWORD),
            version="1",
        )

    handlers = {k: v for k, v in handlers.items() if k in handler_names}

    if not log_process_id:
        standard_format = ('%(asctime)s - %(levelname)-8s '
                           '[%(funcName)s:%(filename)s:%(lineno)d] '
                           '%(message)s')
    else:
        standard_format = ('%(asctime)s - %(levelname)-8s '
                           '<PID %(process)d::%(processName)s>'
                           '[%(funcName)s:%(filename)s:%(lineno)d] '
                           '%(message)s')

    # Define loggers for top-level kumo code paths and for __main__
    loggers = {
        'async_utils': copy(logger_config),
        'io_utils': copy(logger_config),
        'kumo': copy(logger_config),
        'diskgraph': copy(logger_config),
        'diskgraph_pybind': copy(logger_config),
        'diskgraph_ingest': copy(logger_config),
        'diskgraph_engine': copy(logger_config),
        'diskgraph_train_table': copy(logger_config),
        'service_framework': copy(logger_config),
        'temporalio': copy(logger_config),
        'temporalio_commons': copy(logger_config),
        'monitoring': copy(logger_config),
        'pool': copy(logger_config),
        'rest': copy(logger_config),
        'kumo_core': copy(logger_config),
        'tools': copy(logger_config),
        '__main__': copy(logger_config),
    }
    # apply logging level override
    for logger_name, level in logging_level_overrides.items():
        logger_config = loggers.setdefault(logger_name, copy(logger_config))
        logger_config.override_level(level)

    if use_json_format:
        formatters = {
            'json':
            LoggingFormatterConfig(cls='io_utils.logging_json.JsonLogFormatter')
        }
    else:
        # Define a standard formatter and a custom formatter for more
        # information (e.g. PID, processName, funcName) in the case that an
        # error is logged.
        formatters = {
            'standard':
            LoggingFormatterConfig(format=standard_format),
            'error':
            LoggingFormatterConfig(
                format=('%(asctime)s - %(levelname)-8s '
                        '<PID %(process)d::%(processName)s>'
                        '%(name)s.%(funcName)s(): %(message)s'))
        }

    return LoggingConfig(formatters=formatters, handlers=handlers,
                         loggers=loggers)


def _env_var_enabled(name: str) -> bool:
    return os.environ.get(name, '').lower() in {'1', 'true', 'yes', 'on'}


def _with_pytest_logging_level_overrides(
        logging_level_overrides: Mapping[str, LoggingLevel]):
    overrides = dict(logging_level_overrides)
    if (_PYTEST_ENABLE_FSUTIL_DEBUG_LOGS_ENV in os.environ
            and not _env_var_enabled(_PYTEST_ENABLE_FSUTIL_DEBUG_LOGS_ENV)):
        overrides['io_utils.fsutil'] = logging.INFO
    return overrides


def getLogger(name: str) -> logging.Logger:  # pylint: disable
    """Wrapper for logging.getLogger(name)"""
    return logging.getLogger(name)


def initialize_logging(
    mode: LoggingMode = LoggingMode.DEVELOPMENT,
    logging_dir: Optional[str] = None, logging_file_name: Optional[str] = None,
    *, log_to_console: bool = True, log_to_file: bool = True,
    log_to_cloud_watch: bool = False, log_group_name: Optional[str] = None,
    log_stream_name: Optional[str] = None, aws_region: Optional[str] = None,
    log_process_id: bool = False,
    logging_level_overrides: Mapping[str, LoggingLevel] = {
        'monitoring': logging.WARNING
    }):
    """
    Initializes logging by defining the requisite logging handlers and
    other specifications as in the config YAML. The user just
    needs to use the python `logging` module as normal.

    TODO(manan): Currently, every time initialize_logging is called, logs
    are written to the same directory. Perhaps it's worth separating them
    based on __name__? Ideally, this won't matter down the line once we
    clearly define services and associated names.

    Args:
       mode (LoggingMode): the mode to log in.
       logging_dir (str): the path of the log file to write to.
       logging_file_name (str): the name of the log file to write to.
       log_to_console (bool): whether to log to console. (default: :obj:`True`)
       log_to_cloud_watch (bool): whether to log to AWS cloud watch. This is
            set for EMR Spark driver python log. (default: :obj:`False`)
       log_group_name (str): log group name for cloud watch
       log_stream_name (str): log stream name for cloud watch
       log_process_id (bool): Whether to log process id and name in standard
            log format.
        logging_level_overrides (Dict[logger name, logging level]): logging
        level setting per logger name to override the overall `mode`. The value
        can be either LoggingMode enum, int/str value of logging.<LoggingLevel>
    """
    global LOGGING_INITIALIZED

    # Remove all handlers associated with the root logger object. This is
    # important as code that calls Kumo code may have set their own loggers,
    # which we explicitly want to avoid in order to avoid unexpected
    # interactions with Kumo logging.
    #
    # For example, when Livy calls Kumo code in the EMR driver node, it sets
    # a basic logging configuration before Kumo code is called (which sets
    # its own logging configuration). This basic configuration defines a
    # stderr output logger that, if not removed, will result in Kumo code
    # overfilling the stderr pipe buffer, resulting in a deadlock when
    # producing Kumo logs.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if logging_dir is not None:
        _set_logging_dir(logging_dir)

    if logging_file_name is not None:
        _set_logging_file_name(logging_file_name)

    if log_to_cloud_watch:
        # Set boto3 default region to allow proper initialization of boto3
        # client for cloud watch log handler
        import boto3
        if aws_region is None:
            aws_region = os.environ.get('AWS_REGION', 'us-west-2')
        boto3.setup_default_session(region_name=aws_region)
        assert log_group_name is not None
        assert log_stream_name is not None
        _set_log_group_name(log_group_name)
        _set_log_stream_name(log_stream_name)

    config = _get_config(
        mode, log_to_console, log_to_file, log_to_cloud_watch, log_process_id,
        _with_pytest_logging_level_overrides(logging_level_overrides))
    # Restore direct handlers first: dictConfig does not reconfigure a logger
    # this call omits, which would leave it holding a dead queue.
    drain_logs()
    logging.config.dictConfig(config.to_dict())
    _route_handlers_through_queue(config.loggers.keys())
    # Disable root logger which otherwise would pollute logging.
    logging.getLogger().addHandler(logging.NullHandler())

    LOGGING_INITIALIZED = True

    logging.getLogger(__name__).info(
        f'Logging (mode={mode}) successfully initialized at '
        f'{osp.join(LOGGING_DIR, LOGGING_FILE_NAME)}, '
        f'log_to_file={log_to_file}, log_process_id={log_process_id}')


_LISTENER_JOIN_TIMEOUT_S = 5.0
_FORK_HANDLER_LOCK_TIMEOUT_S = 5.0
_DROP_REPORT_INTERVAL_S = 60.0
# ~10k records is a few MB: deep enough to ride out a slow sink, shallow
# enough that a wedged one cannot grow into an OOM.
_QUEUE_MAX_RECORDS = 10000

_QUEUE_LISTENER: Optional['_RecordQueueListener'] = None
_QUEUE_LISTENER_LOCK = threading.RLock()
_ROUTED_HANDLERS: Dict[str, List[logging.Handler]] = {}
# Unsafe-fork children intentionally keep abandoned handlers alive without
# closing them: even finalizing an inherited poisoned stream can deadlock.
_ABANDONED_FORK_HANDLERS: List[logging.Handler] = []
_DROPPED_RECORDS = 0
_REPORTED_DROPS = 0


class _AtForkState(threading.local):
    def __init__(self) -> None:
        self.acquired_handlers: List[logging.Handler] = []
        self.routing_lock_acquired = False
        self.handlers_quiesced = False
        self.failure_reason: Optional[bytes] = None


_AT_FORK_STATE = _AtForkState()


class _RecordQueueHandler(logging.handlers.QueueHandler):
    def handle(self, record: logging.LogRecord) -> bool:
        # Deliberately not `Handler.handle`: its lock can be orphaned by the
        # very asynchronous exception this indirection exists to survive, and
        # `SimpleQueue.put_nowait` needs no lock of ours.
        rv = self.filter(record)
        if isinstance(rv, logging.LogRecord):
            record = rv
        if rv:
            self.emit(record)
        return bool(rv)

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        # Resolve args so a caller mutating them cannot change what is
        # written, but do not render: the handlers behind the listener have
        # their own formatters.
        record = copy(record)
        record.msg = record.getMessage()
        record.args = None
        return record

    def enqueue(self, record: logging.LogRecord) -> None:
        global _DROPPED_RECORDS
        record_queue: 'queue.SimpleQueue' = self.queue  # type: ignore[assignment]
        if record_queue.qsize() >= _QUEUE_MAX_RECORDS:
            _DROPPED_RECORDS += 1
            return
        record_queue.put_nowait(record)


class _RecordQueueListener(logging.handlers.QueueListener):
    def __init__(self, record_queue, *handlers: logging.Handler) -> None:
        super().__init__(record_queue, *handlers, respect_handler_level=True)
        self._next_drop_report = 0.0

    def handle(self, record: logging.LogRecord) -> None:
        self._report_dropped()
        self._handle(record)

    def _handle(self, record: logging.LogRecord) -> None:
        record = self.prepare(record)
        for handler in self.handlers:
            if record.levelno < handler.level:
                continue
            try:
                handler.handle(record)
            except Exception:  # pylint: disable=broad-except
                # One raising handler must not cost the others this record.
                traceback.print_exc(file=sys.stderr)

    def _report_dropped(self) -> None:
        global _REPORTED_DROPS
        dropped = _DROPPED_RECORDS - _REPORTED_DROPS
        if dropped <= 0:
            return
        now = time.monotonic()
        if now < self._next_drop_report:
            return
        self._next_drop_report = now + _DROP_REPORT_INTERVAL_S
        _REPORTED_DROPS += dropped
        self._handle(
            logging.LogRecord(__name__, logging.WARNING, __file__, 0,
                              'logging queue full: dropped %d records',
                              (dropped, ), None))

    def stop(self) -> None:
        if self._thread is not None:
            self.enqueue_sentinel()
            self._thread.join(_LISTENER_JOIN_TIMEOUT_S)
            self._thread = None


def _route_handlers_through_queue(logger_names: Iterable[str]) -> None:
    """Put every configured handler behind one queue and one writer thread.

    ``StreamHandler.emit`` writes and flushes while holding the handler lock,
    so a slow reader of that stream blocks every thread that wants to log.
    Long enough and Temporal calls a workflow task deadlocked and raises into
    the thread, which can leave the lock held forever (kumo-ai/kumo#30925).
    """
    global _QUEUE_LISTENER
    with _QUEUE_LISTENER_LOCK:
        _ROUTED_HANDLERS.clear()
        targets: List[logging.Handler] = []
        seen: set = set()
        for name in logger_names:
            handlers = [
                handler for handler in logging.getLogger(name).handlers
                if not isinstance(handler, (logging.NullHandler,
                                            _RecordQueueHandler))
            ]
            if not handlers:
                continue
            _ROUTED_HANDLERS[name] = handlers
            for handler in handlers:
                if id(handler) not in seen:
                    seen.add(id(handler))
                    targets.append(handler)
        if not targets:
            return

        record_queue: queue.SimpleQueue = queue.SimpleQueue()
        _QUEUE_LISTENER = _RecordQueueListener(record_queue, *targets)
        _QUEUE_LISTENER.start()

        queue_handler = _RecordQueueHandler(record_queue)
        for name in _ROUTED_HANDLERS:
            logger = logging.getLogger(name)
            logger.handlers = [
                handler for handler in logger.handlers
                if isinstance(handler, logging.NullHandler)
            ]
            logger.addHandler(queue_handler)


def _restore_direct_handlers() -> None:
    with _QUEUE_LISTENER_LOCK:
        for name, handlers in _ROUTED_HANDLERS.items():
            logger = logging.getLogger(name)
            logger.handlers = [
                handler for handler in logger.handlers
                if not isinstance(handler, _RecordQueueHandler)
            ] + handlers
        _ROUTED_HANDLERS.clear()


def _stop_queue_listener() -> None:
    global _QUEUE_LISTENER
    with _QUEUE_LISTENER_LOCK:
        if _QUEUE_LISTENER is not None:
            _QUEUE_LISTENER.stop()
            _QUEUE_LISTENER = None


def drain_logs() -> None:
    """Deliver queued records and log directly from here on.

    For shutdown paths that end in ``os._exit``, which runs no atexit hook.
    """
    with _QUEUE_LISTENER_LOCK:
        _stop_queue_listener()
        _restore_direct_handlers()


def _acquire_routed_handler_locks_before_fork() -> None:
    """Keep the listener out of handler streams while they are inherited.

    Python resets ``Handler.lock`` in a child, but it does not reset internal
    locks in arbitrary handler streams. If the listener is writing during the
    fork, restoring that handler can therefore deadlock on its first flush.
    Holding every handler lock across the fork makes the inherited streams
    quiescent; the stdlib logging at-fork callback then unlocks the child-side
    handler copies before our child callback restores them. At-fork callback
    exceptions cannot abort a fork, so a bounded acquisition failure is
    recorded instead. The child callback then discards every inherited route
    rather than risk touching a poisoned stream.
    """
    state = _AT_FORK_STATE
    state.acquired_handlers = []
    state.routing_lock_acquired = False
    state.handlers_quiesced = False
    state.failure_reason = None
    deadline = time.monotonic() + _FORK_HANDLER_LOCK_TIMEOUT_S
    if not _QUEUE_LISTENER_LOCK.acquire(timeout=_FORK_HANDLER_LOCK_TIMEOUT_S):
        state.failure_reason = b'logging routing lock timed out'
        return
    state.routing_lock_acquired = True
    try:
        seen: set = set()
        for handlers in _ROUTED_HANDLERS.values():
            for handler in handlers:
                if id(handler) in seen:
                    continue
                seen.add(id(handler))
                handler_lock = handler.lock
                if (handler_lock is None or not handler_lock.acquire(
                        timeout=max(0.0, deadline - time.monotonic()))):
                    state.failure_reason = b'logging handler lock timed out'
                    return
                state.acquired_handlers.append(handler)
        state.handlers_quiesced = True
    except BaseException:
        # CPython reports and ignores every exception from at-fork callbacks,
        # including BaseException subclasses, then proceeds with the fork.
        # Preserve the safe degraded-child behavior even for those failures.
        state.failure_reason = b'logging handler quiesce raised unexpectedly'


def _release_routed_handler_locks_after_fork_parent() -> None:
    state = _AT_FORK_STATE
    try:
        for handler in reversed(state.acquired_handlers):
            handler.release()
    finally:
        state.acquired_handlers = []
        if state.routing_lock_acquired:
            _QUEUE_LISTENER_LOCK.release()
        state.routing_lock_acquired = False
        state.handlers_quiesced = False
        state.failure_reason = None


def _discard_inherited_routed_handlers_after_unsafe_fork() -> None:
    """Make inherited routed handlers unreachable without invoking them."""
    # This deliberately drops every route, including handlers acquired before
    # a later one timed out. Rebuilding partial, possibly shared handler sets
    # in this rare failure path risks attaching the wrong stream; preserve
    # child liveness over incomplete logging instead.
    abandoned = {
        id(handler): handler
        for handlers in _ROUTED_HANDLERS.values()
        for handler in handlers
    }
    handler_ids = set(abandoned)
    _ABANDONED_FORK_HANDLERS.extend(abandoned.values())
    for name in _ROUTED_HANDLERS:
        logger = logging.getLogger(name)
        logger.handlers = [
            handler for handler in logger.handlers
            if (not isinstance(handler, _RecordQueueHandler)
                and id(handler) not in handler_ids)
        ]
    _ROUTED_HANDLERS.clear()

    # logging.shutdown() flushes every weakly registered handler, even one no
    # longer attached to a logger. Do not let normal child shutdown revisit a
    # stream that was deliberately abandoned as potentially poisoned.
    logging._handlerList[:] = [  # type: ignore[attr-defined]
        handler_ref
        for handler_ref in logging._handlerList  # type: ignore[attr-defined]
        if handler_ref() is None or id(handler_ref()) not in handler_ids
    ]


def _restore_direct_handlers_after_fork() -> None:
    global _QUEUE_LISTENER, _QUEUE_LISTENER_LOCK
    state = _AT_FORK_STATE
    # logging's earlier at-fork callback has reinitialized every Handler lock.
    # Do not release the copies acquired in the parent: they are already
    # unlocked in the child.
    handlers_quiesced = state.handlers_quiesced
    state.acquired_handlers = []
    state.routing_lock_acquired = False
    state.handlers_quiesced = False
    state.failure_reason = None
    _QUEUE_LISTENER_LOCK = threading.RLock()
    # The listener thread does not survive the fork, and children here exit
    # via os._exit, so a replacement thread would just swallow their records.
    _QUEUE_LISTENER = None
    if not handlers_quiesced:
        # Even os.write can block on a full stderr pipe; do not emit here.
        _discard_inherited_routed_handlers_after_unsafe_fork()
        return
    # Routes can remain after the listener has been stopped independently.
    # Restore based on the routes themselves so they never retain an orphaned
    # queue handler in the child.
    _restore_direct_handlers()


# Not `_stop_queue_listener`: later atexit hooks and `logging.shutdown` still
# log, and a queue nobody reads loses their records.
atexit.register(drain_logs)
if hasattr(os, 'register_at_fork'):
    os.register_at_fork(
        before=_acquire_routed_handler_locks_before_fork,
        after_in_parent=_release_routed_handler_locks_after_fork_parent,
        after_in_child=_restore_direct_handlers_after_fork,
    )


def is_logging_initialized() -> bool:
    return LOGGING_INITIALIZED
