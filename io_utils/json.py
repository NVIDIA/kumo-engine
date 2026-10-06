# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import gzip
import json
import logging
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import is_dataclass
from datetime import date, datetime, time
from functools import lru_cache, partial
from json.decoder import JSONDecodeError
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union

from asgiref.sync import SyncToAsync
from kumoapi.json_serde import dataclass_to_dict  # type: ignore[attr-defined]
from pydantic import SecretStr, TypeAdapter
from pydantic_core import to_jsonable_python

from async_utils.sync import blocking_func
from io_utils import fsutil
from io_utils.encryption import EncryptedStr
from io_utils.fsutil.uri import LocalURI, ParseURI
from io_utils.retry import retry
from monitoring.tracing import TraceOptions, trace

# Dedicated thread pool for JSON I/O operations to avoid blocking the
# asyncio default executor which can cause timeouts under contention.
JSON_IO_THREAD_POOL = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix='json_io_',
)

BYTECODE_PREFIX = '__KUMO_BYTECODE_PREFIX__'
EDGE_JOIN_KEY = '__KUMO_EDGE_KEY__'

T = TypeVar('T')

logger = logging.getLogger(__name__)

# Json only supports a limited set of types, possibly with reduced precision.
# This data type should be used in data structures that are intended to save
# python primitives to JSON, in order to avoid any unexpected type-casting
# surprises at runtime.
DateType = Union[datetime, date, time]
JsonSafePrimative = Union[float, DateType, str, None]

# Maximum integer that can be safely represented as a float64 without losing
# precision
# https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/MIN_SAFE_INTEGER
MAX_SAFE_INT = 9007199254740991
MIN_SAFE_INT = -MAX_SAFE_INT


def json_safe_primitive(val: Any) -> JsonSafePrimative:
    if val is None:
        return None
    # Coerce numpy scalar types to Python natives before isinstance
    # checks. `numpy.int64` is NOT a subclass of `int`, and `numpy.bool_` is
    # NOT a subclass of `bool` (only `numpy.float64` happens to subclass
    # `float`). The DiskGraph backend hands back pandas/numpy-typed scalars
    # that reach this helper unchanged, so we must normalize them here.
    # Lazy-import numpy to avoid a hard dependency in `io_utils`; if numpy
    # is unavailable, fall through to the legacy path.
    try:
        import numpy as np
        if isinstance(val, np.generic):
            # `.item()` returns the canonical Python scalar (int/float/bool/
            # str/bytes/complex/None). Re-enter the helper so the precision
            # and NaN/Inf guards below still apply.
            return json_safe_primitive(val.item())
    except ImportError:
        pass
    if isinstance(val, int):
        if val <= MAX_SAFE_INT and val >= MIN_SAFE_INT:
            return val
        # Cast to string to avoid loss of precision
        return str(val)
    if isinstance(val, float):
        return json_safe_float(val)
    if (isinstance(val, datetime) or isinstance(val, time)
            or isinstance(val, date)):
        return val
    # Note: feel free to add more data types, as long as they are supported by
    # the Pydantic serializer
    raise RuntimeError(f"Could not cast value `{val}` with type "
                       f"`{type(val)}` into a json-safe primative")


def json_safe_float(val: Any) -> Optional[float]:
    # JSON does not support NaN or Inf
    float_val = float(val)
    if math.isnan(float_val) or math.isinf(float_val):
        return None
    return float_val


def json_safe_float_array(arr: Sequence[Any]) -> List[Optional[float]]:
    return [json_safe_float(x) for x in arr]


custom_encoders = {EncryptedStr: lambda v: v.as_encrypted()}


def custom_pydantic_encoder(type_encoders: Dict[type, Any], obj: Any,
                            insecure: bool = False) -> Any:
    """v2 port of the removed ``pydantic.json.custom_pydantic_encoder``.

    Walks ``type(obj).__mro__`` looking for a matching encoder in
    ``type_encoders`` (so subclasses are matched, e.g. ``EncryptedStr`` ->
    ``SecretStr`` is *not* matched here -- only exact-or-subclass of a
    registered key). Falls back to pydantic-core's default jsonable
    serialization (the v2 equivalent of v1's ``pydantic_encoder``), which
    handles BaseModel, Enum, datetime/date/time, etc.

    ``insecure`` is threaded into pydantic's serialization *context* so that
    nested ``@field_serializer`` hooks (e.g. ``UsernamePassword``'s
    secret-revealing serializer, which keys off ``info.context['insecure']``)
    honor it. When ``obj`` is a pydantic model/dataclass that *contains*
    ``SecretStr`` fields, the bare-``SecretStr`` branch in
    ``trusted_custom_pydantic_encoder`` is never reached (the whole model is
    handed to the ``default`` callback as one object), so the context is the
    only channel that can reach those nested serializers. With
    ``insecure=False`` no context is passed, leaving pydantic's default
    masking (``**********``) intact.
    """
    for base in type(obj).__mro__:
        encoder = type_encoders.get(base)
        if encoder is not None:
            return encoder(obj)
    # `serialize_unknown=True` mirrors v1's permissive default encoder, which
    # would str()-ify unknown types rather than raise inside json.dumps.
    # Only pass a context when explicitly insecure so that the default path
    # never weakens secret masking.
    context = {'insecure': True} if insecure else None
    return to_jsonable_python(obj, serialize_unknown=True, context=context)


def trusted_custom_pydantic_encoder(obj, insecure: bool):
    if isinstance(obj, SecretStr):
        if insecure:
            logger.warning('Revealing SecretStr in JSON output!')
            return obj.get_secret_value()
        if not isinstance(obj, EncryptedStr):
            raise ValueError(
                'Secure serialization of SecretStr is not allowed, it must be '
                'converted to EncryptedStr.')

    return custom_pydantic_encoder(custom_encoders, obj, insecure=insecure)


def to_json(pydantic_obj: Any, insecure=False) -> str:
    """Encodes a pydantic object into JSON.

    The `insecure` flag should only be used by trusted internal code where the
    output of the JSON is not accessible to any users and `SecretStr`s are
    hidden in some other fashion.
        - Only used by rest/routers/fixtures.py which encrypts credentials
    """
    pydantic_obj = escape_dataclass(pydantic_obj)
    return json.dumps(
        pydantic_obj,
        default=partial(trusted_custom_pydantic_encoder, insecure=insecure),
        allow_nan=True,
        indent=2,
    )


def to_json_dict(pydantic_obj: Any, insecure=False) -> Dict[str, Any]:
    return json.loads(to_json(pydantic_obj, insecure))


def from_json(data: str, cls: Type[T]) -> T:
    """Decodes a JSON string into a pydantic object.
        If cls is None, returns a dict.
    """
    try:
        dicts = json.loads(data)
    except JSONDecodeError as e:
        # If decode fails, try again with characters from eof trimmed
        try:
            dicts = json.loads(data[:e.pos])
        except JSONDecodeError:
            # If recovery fails, raise the original exception
            logger.exception(f"failed to decode {data=}", stack_info=False)
            raise e
        rogue_chars = data[e.pos:]
        logger.warning(f"recovered from json decode error "
                       f"by trimming {len(data) - e.pos} characters: "
                       f"{rogue_chars=}")
    return from_json_dict(dicts, cls)


@lru_cache(maxsize=None)
def _adapter(cls: Type[T]) -> TypeAdapter:
    r"""Cache the ``TypeAdapter`` per ``cls``.

    Building a ``TypeAdapter`` compiles a pydantic-core schema, which is
    expensive; the result is immutable and safe to reuse across calls, so
    cache it keyed on the type rather than rebuilding it on every decode.
    """
    return TypeAdapter(cls)


def from_json_dict(data: Dict[str, Any], cls: Type[T]) -> T:
    """Decodes a python dict into a pydantic object.
        If cls is None, returns a dict.
    """
    data = restore_escaped_json_fields(data)
    return _adapter(cls).validate_python(data)


def fast_from_json(data: bytes, cls: Type[T]) -> T:
    # TODO(zack): replace from_json with this function. Main blocker is
    # restore_escaped_json_field, which makes the function 3x slower
    dicts = json.loads(data)
    return _adapter(cls).validate_python(dicts)


def restore_escaped_json_fields(conf: Any) -> Any:
    r"""Restore escaped JSON entries, i.e. bytecode, encoded by
    `escape_dataclass`"""
    was_str = False
    if isinstance(conf, str):
        was_str = True
        conf = {0: conf}
    was_list = False
    if isinstance(conf, List):
        conf = {i: conf[i] for i in range(len(conf))}
        was_list = True
    assert isinstance(conf, Dict)
    for key in list(conf.keys()):
        if isinstance(key, str) and (EDGE_JOIN_KEY in key):
            new_key = tuple(key.split(EDGE_JOIN_KEY))
            conf[new_key] = conf.pop(key)
    for key in conf.keys():
        if isinstance(conf[key], str):
            if conf[key][:len(BYTECODE_PREFIX)] == BYTECODE_PREFIX:
                conf[key] = base64.b64decode(conf[key][len(BYTECODE_PREFIX):])
            elif EDGE_JOIN_KEY in conf[key]:
                conf[key] = tuple(conf[key].split(EDGE_JOIN_KEY))
        if isinstance(conf[key], Dict) or isinstance(conf[key], List):
            conf[key] = restore_escaped_json_fields(conf[key])
    if was_list:
        return list(conf.values())
    if was_str:
        return conf[0]
    return conf


@blocking_func(JSON_IO_THREAD_POOL,
               tracing_options=TraceOptions(trace_params=['path']))
def read_json(path: str, compressed=False) -> Any:
    try:
        return fsutil.read_text(path, compression=compressed)
    except Exception as e:
        raise RuntimeError(f"Failed to read data from file: {path}.") from e


@blocking_func(JSON_IO_THREAD_POOL,
               tracing_options=TraceOptions(trace_params=['path']))
def save_json(pydantic_obj: Any, path: str, compressed=False,
              insecure=False) -> None:
    try:
        new_contents = to_json(pydantic_obj, insecure)
        base_dir = os.path.dirname(path)
        if base_dir != '':
            fsutil.makedirs(base_dir, exist_ok=True)
        fsutil.write_text(path, new_contents, compression=compressed)
    except Exception as e:
        raise RuntimeError(
            f"Failed to save class with schema: {type(pydantic_obj)}"
            f" to {path}.") from e


@blocking_func(JSON_IO_THREAD_POOL,
               tracing_options=TraceOptions(trace_params=['path']))
def save_json_atomic(pydantic_obj: Any, path: str, compressed=False,
                     insecure=False) -> None:
    """Publish a complete JSON document without exposing partial contents.

    Local files use a unique, mode-``0600`` sibling followed by
    :func:`os.replace`, which is atomic for cross-process readers. Supported
    S3/GCS publish the already-serialized payload with their complete-object
    upload API at the final key.
    mapped, or one Snowflake PUT. ADLS uploads a unique sibling and performs a
    conditional HNS rename. UCV intentionally fails closed because its current
    overwrite API does not prove atomic visibility.
    """
    try:
        new_contents = to_json(pydantic_obj, insecure)
        scheme = fsutil.get_scheme(path)
        base_dir = os.path.dirname(path)
        if base_dir != '':
            fsutil.makedirs(base_dir, exist_ok=True)

        if scheme == 'local':
            parsed = ParseURI(path)
            assert isinstance(parsed, LocalURI)
            _atomic_local_write_text(parsed.path, new_contents,
                                     compressed=compressed)
            return

        if scheme in {'s3', 'gs'}:
            # S3 put_object and GCS upload_from_string commit complete object
            # generations at the final provider key without a remote temp.
            fsutil.write_text(path, new_contents, compression=compressed)
            return

        if scheme in {'abfs', 'abfss'}:
            fsutil.atomic_write_text(path, new_contents,
                                     compression=compressed)
            return

        # The Databricks Files API does not document atomic overwrite
        # visibility. Fail closed until it has a proven server-side primitive.
        raise ValueError(
            f"Atomic JSON publication is unsupported for scheme {scheme!r}")
    except Exception:
        # This crosses activity boundaries for persisted inputs. Do not attach
        # a cause containing serialized data, credentials, or temporary paths.
        raise RuntimeError(
            f"Failed to atomically save class with schema: "
            f"{type(pydantic_obj)}.") from None


def _atomic_local_write_text(path: str, data: str, *, compressed: bool) -> None:
    """Publish text atomically to a local path.

    The complete payload is written to a unique sibling created with mode
    ``0600``. ``os.replace`` then makes either the previous or the complete new
    document visible to cross-process readers; no process-local lock is
    required. Remote paths intentionally keep their provider's complete-object
    write in :func:`save_json` and never create a second remote object.
    """
    if os.path.islink(path):
        raise RuntimeError('Atomic JSON target must not be a symbolic link.')
    parent = os.path.dirname(path) or '.'
    fd, temp_path = tempfile.mkstemp(prefix='.kumo-json-', suffix='.tmp',
                                     dir=parent)
    try:
        payload = data.encode('utf-8')
        if compressed:
            payload = gzip.compress(payload)
        with os.fdopen(fd, 'wb') as file:
            fd = -1
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    except Exception:
        raise RuntimeError('Atomic local JSON publication failed.') from None
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass


@trace(TraceOptions(trace_params=True))
def load_json(path: str, cls: Type[T], compressed=False, default=None,
              raise_on_missing=True) -> T:
    if not raise_on_missing and not fsutil.exists(path):
        return default  # type: ignore

    # NOTE: This is to mitigate some transient errors observed during JSON loading
    # as mentioned in https://github.com/kumo-ai/kumo/issues/4942.
    @retry(exceptions=RuntimeError, tries=9, delay=1, backoff=2, logger=logger)
    def _read_file() -> str | None:
        try:
            return fsutil.read_text(path, compression=compressed)
        except FileNotFoundError:
            if raise_on_missing:
                raise
            return None
        except Exception as e:
            raise RuntimeError(
                f"Failed to read json file {path}, ex: {e}.") from e

    # if raise_on_missing and FileNotFound, will raise through here
    data: str | None = _read_file()

    def _match_parse_ex(e: Exception):
        return isinstance(e, JSONDecodeError)

    def _handle_parse_ex(e: Exception):
        nonlocal data
        assert isinstance(e, JSONDecodeError)
        logger.info("re-reading json file before re-parse")
        data = _read_file()

    # if parse fails, re-read and re-parse
    @retry(tries=3, callbacks={
        _match_parse_ex: _handle_parse_ex,
    }, logger=logger)
    def _parse_json():
        if data is None:
            # FileNotFound during re-read and raise_on_missing=False
            return None
        try:
            return from_json(data, cls)
        except Exception as e:
            raise ValueError(
                f'Failed to parse data ({len(data)} bytes) read from {path} to '
                f'schema {cls}, ex: {e}') from e

    obj = _parse_json()
    if obj is None:
        # FileNotFound during re-read and raise_on_missing=False
        return default  # type: ignore
    return obj


@blocking_func(JSON_IO_THREAD_POOL,
               tracing_options=TraceOptions(trace_params=True))
def load_many_json(paths: List[str], cls: Type[T],
                   compressed=False) -> List[Optional[T]]:
    files = fsutil.get_many_if_present(paths)

    # Decompress and parse the result as json
    if compressed is True:
        return [
            fast_from_json(gzip.decompress(file), cls) if file else None
            for file in files
        ]
    else:
        # Parse the result as json
        return [fast_from_json(file, cls) if file else None for file in files]


async def load_json_async(path: str, cls: Type[T], compressed=False,
                          default=None, raise_on_missing=False) -> T:
    return (await SyncToAsync(load_json, thread_sensitive=False,
                              executor=JSON_IO_THREAD_POOL)
            (path, cls, compressed, default, raise_on_missing))  # type: ignore


class ToJsonMixin:
    def json(self) -> Dict[str, str]:
        return to_json_dict(self)


def escape_dataclass(conf: Any) -> Any:
    r"""Escape JSON control characters and bytecode in fields of a dataclass
    object"""
    assert is_dataclass(conf) or isinstance(conf, Dict) or isinstance(
        conf, List) or isinstance(conf, str)
    if is_dataclass(conf):
        conf = dataclass_to_dict(conf)
    was_list = False
    if isinstance(conf, List):
        conf = {i: conf[i] for i in range(len(conf))}
        was_list = True
    was_str = False
    if isinstance(conf, str):
        was_str = True
        conf = {0: conf}

    # N.B. Lazy import of torch_geometric
    # try-except allows using this utility without torch installed
    try:
        from torch_geometric.typing import EdgeTypeStr
        for key in list(conf.keys()):
            if isinstance(key, EdgeTypeStr):
                new_key = EDGE_JOIN_KEY.join(key.edge_type)
                conf[new_key] = conf.pop(key)
        for key in conf.keys():
            if isinstance(conf[key], EdgeTypeStr):
                conf[key] = EDGE_JOIN_KEY.join(conf[key].edge_type)
    except ImportError:
        pass

    for key in conf.keys():
        if isinstance(conf[key], Dict) or isinstance(
                conf[key], List) or is_dataclass(conf[key]):
            conf[key] = escape_dataclass(conf[key])
        if isinstance(conf[key], bytes):
            conf[key] = BYTECODE_PREFIX + base64.b64encode(
                conf[key]).decode('ascii')

    if was_list:
        return list(conf.values())
    if was_str:
        return conf[0]
    return conf
