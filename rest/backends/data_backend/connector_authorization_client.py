# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import os
import threading
import weakref
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

from temporalio.client import StartWorkflowInput

from rest.config.data_backend import (
    ActivityAuthorizationRef,
    ConnectorExecutionRef,
)
from rest.settings import get_db_storage_root
from service_framework._raw_deploy_config import SqliteMetadataStorageConfig
from service_framework.deploy_env_config import DeployEnvConfig
from statedb.config.timestamp import now_utc
from statedb.db import (
    StateDB,
    StateServices,
    default_sqlite_path,
    open_state_db_from_deploy_config,
)

_DEFAULT_BOUND_CAPABILITY_TTL = timedelta(days=7)
_CAPABILITY_EXPIRY_GRACE = timedelta(minutes=5)


@dataclasses.dataclass(frozen=True)
class _CachedAuthorizationState:
    state_db: StateDB
    services: StateServices


_AUTHORIZATION_STATE_LOCK = threading.Lock()
_AUTHORIZATION_STATE_PID = os.getpid()
_AUTHORIZATION_STATE_CACHE: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[str, _CachedAuthorizationState],
] = weakref.WeakKeyDictionary()
_AUTHORIZATION_STATE_OPENING: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[str, concurrent.futures.Future[_CachedAuthorizationState]],
] = weakref.WeakKeyDictionary()


def _reset_authorization_state_after_fork() -> None:
    """Replace inherited locks and loop-owned connections in a child."""
    global _AUTHORIZATION_STATE_LOCK
    global _AUTHORIZATION_STATE_PID
    global _AUTHORIZATION_STATE_CACHE
    global _AUTHORIZATION_STATE_OPENING
    # A fork can occur while another parent thread holds the guard.  A PID
    # check performed *after* acquiring that inherited lock would deadlock the
    # child forever, so the at-fork hook must replace the guard itself.
    _AUTHORIZATION_STATE_LOCK = threading.Lock()
    _AUTHORIZATION_STATE_PID = os.getpid()
    _AUTHORIZATION_STATE_CACHE = weakref.WeakKeyDictionary()
    _AUTHORIZATION_STATE_OPENING = weakref.WeakKeyDictionary()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_reset_authorization_state_after_fork)


def _sqlite_authorization_state_key(config: DeployEnvConfig) -> str | None:
    if not isinstance(config.metadata_storage, SqliteMetadataStorageConfig):
        return None
    return default_sqlite_path(get_db_storage_root())


def _discard_inherited_authorization_state_locked() -> None:
    global _AUTHORIZATION_STATE_PID
    pid = os.getpid()
    if pid == _AUTHORIZATION_STATE_PID:
        return
    # Connections and their worker threads cannot cross a fork boundary.
    # Discard inherited references without trying to drive parent-owned loops.
    _AUTHORIZATION_STATE_CACHE.clear()
    _AUTHORIZATION_STATE_OPENING.clear()
    _AUTHORIZATION_STATE_PID = pid


async def _open_authorization_state(
    config: DeployEnvConfig, ) -> tuple[StateDB, StateServices]:
    cache_key = _sqlite_authorization_state_key(config)
    if cache_key is None:
        state_db = await open_state_db_from_deploy_config(
            config,
            state_root=get_db_storage_root(),
        )
        return state_db, StateServices(state_db)

    loop = asyncio.get_running_loop()
    with _AUTHORIZATION_STATE_LOCK:
        _discard_inherited_authorization_state_locked()
        loop_cache = _AUTHORIZATION_STATE_CACHE.setdefault(loop, {})
        cached = loop_cache.get(cache_key)
        if cached is not None:
            return cached.state_db, cached.services
        loop_opening = _AUTHORIZATION_STATE_OPENING.setdefault(loop, {})
        opening = loop_opening.get(cache_key)
        owns_open = opening is None
        if opening is None:
            opening = concurrent.futures.Future()
            loop_opening[cache_key] = opening

    if not owns_open:
        cached = await asyncio.shield(asyncio.wrap_future(opening))
        return cached.state_db, cached.services

    try:
        state_db = await open_state_db_from_deploy_config(
            config,
            state_root=get_db_storage_root(),
        )
        cached = _CachedAuthorizationState(
            state_db=state_db,
            services=StateServices(state_db),
        )
    except BaseException as error:
        with _AUTHORIZATION_STATE_LOCK:
            loop_opening = _AUTHORIZATION_STATE_OPENING.get(loop)
            if loop_opening is not None:
                loop_opening.pop(cache_key, None)
                if not loop_opening:
                    _AUTHORIZATION_STATE_OPENING.pop(loop, None)
            opening.set_exception(error)
        raise
    with _AUTHORIZATION_STATE_LOCK:
        loop_opening = _AUTHORIZATION_STATE_OPENING.get(loop)
        if loop_opening is not None:
            loop_opening.pop(cache_key, None)
            if not loop_opening:
                _AUTHORIZATION_STATE_OPENING.pop(loop, None)
        _AUTHORIZATION_STATE_CACHE.setdefault(loop, {})[cache_key] = cached
        opening.set_result(cached)
    return cached.state_db, cached.services


def _is_cached_authorization_state(state_db: StateDB | None) -> bool:
    if state_db is None:
        return False
    with _AUTHORIZATION_STATE_LOCK:
        _discard_inherited_authorization_state_locked()
        return any(cached.state_db is state_db
                   for loop_cache in _AUTHORIZATION_STATE_CACHE.values()
                   for cached in loop_cache.values())


async def close_cached_authorization_state_for_current_loop() -> None:
    """Close reusable SQLite state owned by the current event loop."""
    loop = asyncio.get_running_loop()
    with _AUTHORIZATION_STATE_LOCK:
        _discard_inherited_authorization_state_locked()
        cached = tuple(_AUTHORIZATION_STATE_CACHE.pop(loop, {}).values())
        _AUTHORIZATION_STATE_OPENING.pop(loop, None)
    for entry in cached:
        await entry.state_db.close()


async def close_cached_authorization_states() -> None:
    """Close all loop-local SQLite state from a quiescent process owner."""
    with _AUTHORIZATION_STATE_LOCK:
        _discard_inherited_authorization_state_locked()
        cached = tuple(entry
                       for loop_cache in _AUTHORIZATION_STATE_CACHE.values()
                       for entry in loop_cache.values())
        _AUTHORIZATION_STATE_CACHE.clear()
        _AUTHORIZATION_STATE_OPENING.clear()
    for entry in cached:
        await entry.state_db.close()


def _authorization_tokens(value: Any, seen: set[int]) -> set[str]:
    if isinstance(value, (ConnectorExecutionRef, ActivityAuthorizationRef)):
        return {value.authorization_token}
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return set()
    value_id = id(value)
    if value_id in seen:
        return set()
    seen.add(value_id)
    if dataclasses.is_dataclass(value):
        tokens: set[str] = set()
        for field in dataclasses.fields(value):
            tokens.update(
                _authorization_tokens(getattr(value, field.name), seen))
        return tokens
    if isinstance(value, Mapping):
        tokens = set()
        for key, item in value.items():
            tokens.update(_authorization_tokens(key, seen))
            tokens.update(_authorization_tokens(item, seen))
        return tokens
    if isinstance(value, (list, tuple, set, frozenset)):
        tokens = set()
        for item in value:
            tokens.update(_authorization_tokens(item, seen))
        return tokens
    return set()


def connector_execution_references(value: Any) -> list[ConnectorExecutionRef]:
    """Return unique connector references in deterministic token order."""
    by_token: dict[str, ConnectorExecutionRef] = {}

    def visit(item: Any, seen: set[int]) -> None:
        if isinstance(item, ConnectorExecutionRef):
            by_token[item.authorization_token] = item
            return
        if item is None or isinstance(item, (str, bytes, int, float, bool)):
            return
        item_id = id(item)
        if item_id in seen:
            return
        seen.add(item_id)
        if dataclasses.is_dataclass(item):
            for field in dataclasses.fields(item):
                visit(getattr(item, field.name), seen)
        elif isinstance(item, Mapping):
            for key, value in item.items():
                visit(key, seen)
                visit(value, seen)
        elif isinstance(item, (list, tuple, set, frozenset)):
            for value in item:
                visit(value, seen)

    visit(value, set())
    return [by_token[token] for token in sorted(by_token)]


def replace_connector_authorization_tokens(
    value: Any,
    replacements: Mapping[str, str],
) -> Any:
    """Pure deterministic replacement used before a child workflow start."""
    if isinstance(value, ConnectorExecutionRef):
        token = replacements.get(value.authorization_token)
        return (dataclasses.replace(value, authorization_token=token)
                if token is not None else value)
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value):
        changed = False
        changes = {}
        for field in dataclasses.fields(value):
            old_field_value = getattr(value, field.name)
            new_field_value = replace_connector_authorization_tokens(
                old_field_value, replacements)
            changes[field.name] = new_field_value
            changed = changed or new_field_value is not old_field_value

        extra = getattr(value, '__pydantic_extra__', None)
        extra_changed = False
        new_extra = extra
        if extra:
            new_extra = {}
            for key, item in extra.items():
                new_key = replace_connector_authorization_tokens(
                    key, replacements)
                new_item = replace_connector_authorization_tokens(
                    item, replacements)
                new_extra[new_key] = new_item
                extra_changed = (extra_changed or new_key is not key
                                 or new_item is not item)

        if not changed and not extra_changed:
            return value

        replaced = dataclasses.replace(value, **changes)
        if extra is not None:
            object.__setattr__(replaced, '__pydantic_extra__', new_extra)
        return replaced
    if isinstance(value, dict):
        changed = False
        result = {}
        for key, item in value.items():
            new_key = replace_connector_authorization_tokens(key, replacements)
            new_item = replace_connector_authorization_tokens(
                item, replacements)
            result[new_key] = new_item
            changed = changed or new_key is not key or new_item is not item
        return result if changed else value
    if isinstance(value, list):
        result = [
            replace_connector_authorization_tokens(item, replacements)
            for item in value
        ]
        return result if any(new is not old
                             for new, old in zip(result, value)) else value
    if isinstance(value, tuple):
        result = tuple(
            replace_connector_authorization_tokens(item, replacements)
            for item in value)
        return result if any(new is not old
                             for new, old in zip(result, value)) else value
    if isinstance(value, set):
        changed = False
        result = set()
        for item in value:
            new_item = replace_connector_authorization_tokens(
                item, replacements)
            result.add(new_item)
            changed = changed or new_item is not item
        return result if changed else value
    if isinstance(value, frozenset):
        changed = False
        result = set()
        for item in value:
            new_item = replace_connector_authorization_tokens(
                item, replacements)
            result.add(new_item)
            changed = changed or new_item is not item
        return frozenset(result) if changed else value
    return value


async def bind_connector_authorizations_for_start(
    input: StartWorkflowInput,
    config: DeployEnvConfig,
) -> tuple[StateDB | None, StateServices | None, tuple[str, ...]]:
    """Bind every opaque input capability to the exact accepted start ID."""
    tokens = tuple(
        sorted(filter(None, _authorization_tokens(input.args, set()))))
    if not tokens:
        return None, None, ()
    state_db, services = await _open_authorization_state(config)
    lifetime = (input.execution_timeout or input.run_timeout
                or _DEFAULT_BOUND_CAPABILITY_TTL)
    expires_at = now_utc() + lifetime + _CAPABILITY_EXPIRY_GRACE
    deployment_id = (f'{config.instance_id}:'
                     f'{config.client_id}')
    newly_bound: list[str] = []
    try:
        for token in tokens:
            _, bound = await services.connector_authorizations.bind_workflow(
                token,
                deployment_id=deployment_id,
                temporal_namespace=config.temporal_namespace,
                workflow_id=input.id,
                expires_at=expires_at,
            )
            if bound:
                newly_bound.append(token)
    except BaseException:
        for token in newly_bound:
            await services.connector_authorizations.revoke(token)
        if not _is_cached_authorization_state(state_db):
            await state_db.close()
        raise
    return state_db, services, tuple(newly_bound)


async def revoke_bound_connector_authorizations(
    state_db: StateDB | None,
    services: StateServices | None,
    tokens: tuple[str, ...],
) -> None:
    try:
        if services is not None:
            for token in tokens:
                await services.connector_authorizations.revoke(token)
    finally:
        if (state_db is not None
                and not _is_cached_authorization_state(state_db)):
            await state_db.close()


async def close_connector_authorization_state(state_db: StateDB | None) -> None:
    if (state_db is not None and not _is_cached_authorization_state(state_db)):
        await state_db.close()


async def revoke_connector_authorizations_for_workflow(
    config: DeployEnvConfig,
    workflow_id: str,
) -> int:
    """Revoke exact workflow authority after cancel/terminate is accepted."""
    state_db, services = await _open_authorization_state(config)
    try:
        return await services.connector_authorizations.revoke_workflow(
            workflow_id=workflow_id,
            deployment_id=(f'{config.instance_id}:'
                           f'{config.client_id}'),
            temporal_namespace=config.temporal_namespace,
        )
    finally:
        if not _is_cached_authorization_state(state_db):
            await state_db.close()
