# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import functools
import inspect
import itertools
import json
import logging
from collections import defaultdict
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Sequence,
    TypeVar,
)

from aiocache import Cache
from asgiref.sync import SyncToAsync
from pydantic_core import to_jsonable_python
from typing_extensions import ParamSpec

from service_framework.deploy_stage import _DEPLOY_STAGE, DeployStage
from service_framework.env_var import EnvVar

log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class EventLoopBlockedError(RuntimeError):
    ...


AsyncFunc = Callable[P, Awaitable[R]]

TaskCacheKey = Any
# Dict/cache of inflight async Tasks that are actually running.
inflight_tasks: Dict[TaskCacheKey, asyncio.Task] = dict()
# Async futures that are just awaiting for the actual task (under the same
# TaskCacheKey) to finish.
inflight_futures: Dict[TaskCacheKey, List[asyncio.Future]] = defaultdict(list)


def _inflight_task_done_cb(key: TaskCacheKey, task: asyncio.Task):
    assert task.done()
    ex = task.exception()
    for fut in inflight_futures[key]:
        if ex is None:
            fut.set_result(task.result())
        else:
            fut.set_exception(ex)
    inflight_futures.pop(key)
    inflight_tasks.pop(key)


# Make a parameter value hashable (so that it can be used as part of cache key)
def _hashable_value(fn: str, value: Any) -> Hashable:
    if isinstance(value, Hashable):
        if dataclasses.is_dataclass(value):
            # Even if a dataclass is "hashable", i.e., marked as
            # unsafe_hash=True or frozen=True, it might not be actually
            # hashable due to an unhashable sub-field.
            try:
                return hash(value)
            except:  # noqa: E722
                log.warning('Dataclass type %s is unhashable, fn=%s',
                            type(value), fn)
                # fallback to dumping the dataclass as json
        else:
            return value

    try:
        return json.dumps(
            value, default=functools.partial(to_jsonable_python,
                                             serialize_unknown=True),
            allow_nan=True)
    except:  # noqa: E722
        pass

    # json dumps cannot handle non-plain key object even if key is hashable
    try:
        from temporalio_commons.dataclass_converter import KumoPayloadConverter
        return KumoPayloadConverter().to_payloads([value])[0].data
    except:  # noqa: E722
        log.warning('Unable to serialize %s, fn=%s', type(value), fn)
        if isinstance(value, dict) and len(value) > 0:
            k, v = next(iter(value.items()))
            log.warning('Dict key type=%s, value type=%s', type(k), type(v))
        elif isinstance(value, (list, set)) and len(value) > 0:
            e = next(iter(value))
            log.warning('Container element type=%s', type(e))

    return id(value)  # Memory address of the object instance.


# Given a function, recursively unwrap and flatten the closure (captured
# variables) into a list of hashable values.
# This is needed for nested/inner functions that capture variables from
# enclosing scope.
def _unwrap_closure(fn: Callable) -> List[Hashable]:
    if isinstance(fn, SyncToAsync):
        return _unwrap_closure(fn.func)
    ret = []
    if not hasattr(fn, '__closure__'):
        log.warning('%s (type=%s) has no __closure__ attr', fn, type(fn))
        return [id(fn)]
    if fn.__closure__:
        for cell in fn.__closure__:
            if isinstance(cell.cell_contents, Callable):
                ret.extend(_unwrap_closure(cell.cell_contents))
            else:
                ret.append(_hashable_value(fn.__qualname__, cell.cell_contents))
    return ret


def reuse_inflight_future(
    *,
    param_keys: Optional[Sequence[str]] = None,
    instance_hash: Optional[str] = None,
) -> Callable[[AsyncFunc], AsyncFunc]:
    def _inner(fn: AsyncFunc) -> AsyncFunc:
        @functools.wraps(fn)
        async def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                bound_arguments = inspect.signature(fn).bind(*args, **kwargs)
            except TypeError:
                log.warning('Failed to bound args for %s', fn.__qualname__)
                return await fn(*args, **kwargs)

            bound_arguments.apply_defaults()

            cache_key_kwargs = {
                k: _hashable_value(fn.__qualname__, v)
                for k, v in bound_arguments.arguments.items()
                if param_keys is None or k in param_keys
            }
            # When caching an instance method invocation, we need to include
            # the instance hash (serialized state, if not, id of the instance)
            if instance_hash:
                cache_key_kwargs['self'] = instance_hash

            cache_key_args = [fn.__qualname__] + _unwrap_closure(fn)

            # Check if serialization is too expensive:
            total_size = 0
            for arg in itertools.chain(cache_key_args,
                                       cache_key_kwargs.values()):
                if isinstance(arg, (str, bytes)):
                    total_size += len(arg)
            if total_size > 10 << 20:  # 10MB threshold
                msg = 'Serialized args are too large in {}: {} bytes'.format(
                    fn.__qualname__, total_size)
                if _DEPLOY_STAGE.parse_value() != DeployStage.PROD:
                    raise RuntimeError(msg)
                else:
                    log.critical(msg)

            key = functools._make_key(tuple(cache_key_args), cache_key_kwargs,
                                      True)
            inflight_task = inflight_tasks.get(key)
            if inflight_task is None:
                inflight_task = asyncio.ensure_future(fn(*args, **kwargs))
                inflight_tasks[key] = inflight_task
                inflight_task.add_done_callback(
                    functools.partial(_inflight_task_done_cb, key))
                return await inflight_task  # type: ignore
            else:
                # TODO(siyang): Downgrade to DEBUG if too spammy.
                log.info('Async task for %s is already inflight, awaiting...',
                         fn.__qualname__)
                fut = asyncio.Future()
                inflight_futures[key].append(fut)
                return await fut  # type: ignore

        return _wrapper

    return _inner


_api_cache_store = Cache(Cache.MEMORY, namespace='api')

API_CACHE_TTL = EnvVar(
    'API_CACHE_DEFAULT_TTL', type=int, required=False, default_value=5,
    description='Default cache TTL seconds for REST read-only APIs.')

API_CACHE_ENABLED_ATTR = '__API_CACHE_ENABLED'


def fastapi_cached(
    *,
    param_keys: Optional[Sequence[str]] = None,
    ttl_sec: Optional[int] = None,
    exclude_ctx: bool = True,
) -> Callable[[AsyncFunc], AsyncFunc]:
    """
    Decorator for enabling caching with FastAPI api route(handler) function,
    intended for read-only APIs to return cached resource object.

    This decorator augments API route function signature by adding two caching
    query parameters: "skip_cache" (bool) and "cache_ttl"(optional int).

    Using `aiocache` lib under the hood, this decorator comes with some
    customization:
    1) custom key builder that hashes either all or selected param keys;
    2) support "skip_cache" and "cache_ttl" query parameters, i.e., API request
    can specify "skip_cache=true" to disable caching to "force refresh"; and
    specify "cache_ttl=<int>" to override default cache ttl.

    Args:
        param_keys (Optional[Sequence[str]], optional): Parameters that should
        be included to construct cache key, if not specified, all parameters
        (except "self") will be used.

        ttl_sec (int, optional): Optional cached value TTL per call. If None,
        default to "API_CACHE_DEFAULT_TTL" env var value.

        exclude_ctx (bool): Whether to exclude "ctx" parameter for request
        context (i.e., user authorization args) when building cache key.

    Returns a function decorator for FastAPI route/handler coroutine function.
    """
    def _build_cache_key(fn: AsyncFunc, *args, **kwargs):
        bound_arguments = inspect.signature(fn).bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        cache_key_kwargs = {
            k: _hashable_value(fn.__qualname__, v)
            for k, v in bound_arguments.arguments.items()
            if k != 'self' and (k != 'ctx' or not exclude_ctx) and (
                param_keys is None or k in param_keys)
        }
        key = functools._make_key((fn.__qualname__, ), cache_key_kwargs, True)
        return f'{fn.__qualname__}:{key.__hash__()}'

    def _decorator(fn: AsyncFunc) -> AsyncFunc:
        @functools.wraps(fn)
        async def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            skip_cache = kwargs.pop('skip_cache', False)
            cache_ttl = kwargs.pop('cache_ttl', None)

            coro = reuse_inflight_future(param_keys=param_keys)(fn)

            if skip_cache:
                return await coro(*args, **kwargs)

            if cache_ttl is not None:
                cache_ttl = int(cache_ttl)  # type: ignore
            else:
                cache_ttl = ttl_sec or API_CACHE_TTL.parse_value()
            if cache_ttl <= 0:
                return await coro(*args, **kwargs)

            cache = _api_cache_store
            k = _build_cache_key(fn, *args, **kwargs)
            v = await cache.get(k)
            if v is not None:
                return v

            v = await coro(*args, **kwargs)
            await cache.set(k, v, ttl=cache_ttl)
            return v

        # Make sure we only apply this decorator once.
        if getattr(fn, API_CACHE_ENABLED_ATTR, False):
            log.warning('API handler %s is alreadyed cache-enabled',
                        fn.__qualname__)
            return fn  # prevent applying caching wrapper multiple times
        setattr(_wrapper, API_CACHE_ENABLED_ATTR, True)

        # Augment API route function signature to inject "skip_cache" and
        # "cache_ttl" query parameters for all API routes that enable caching.
        # This is necessary as FastAPI inspects the function signature to
        # find out about declared query parameters.
        fn_sig = inspect.signature(fn)
        params = list(fn_sig.parameters.values())
        params.append(
            inspect.Parameter('skip_cache',
                              inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              default=False, annotation=bool))
        params.append(
            inspect.Parameter('cache_ttl',
                              inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              default=None, annotation=Optional[int]))
        setattr(_wrapper, '__signature__', fn_sig.replace(parameters=params))

        return _wrapper

    return _decorator
