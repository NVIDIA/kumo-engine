# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import hmac
import json
import secrets
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Iterable

from io_utils import logging
from service_framework.env_var import EnvVar
from statedb.config.common import (
    ConnectorAuthorizationID,
    ConnectorAuthorizationUseID,
)
from statedb.config.timestamp import (
    get_current_timestamps,
    now_utc,
    update_timestamps,
)
from statedb.db.auth import AuthPrincipal
from statedb.db.repositories import (
    JsonRepository,
    _in_predicate,
    _json_value,
    _placeholder,
)
from statedb.db.row_parsing import row_str
from statedb.db.runtime import StateDB, StateTx
from statedb.rest.connector_authorization import (
    ConnectorAuthorizationDBEntry,
    ConnectorAuthorizationResourceKind,
    ConnectorAuthorizationUseDBEntry,
    ConnectorCapabilityAction,
)

_UNBOUND_AUTHORIZATION_TTL = timedelta(minutes=5)
_DEFAULT_INACTIVE_RETENTION = timedelta(days=7)
_DEFAULT_AUTHORIZATION_GC_BATCH_SIZE = 100
_DEFAULT_AUTHORIZATION_USE_GC_BATCH_SIZE = 1000
_DEFAULT_AUTHORIZATION_GC_MIN_INTERVAL = timedelta(minutes=5)
_DEFAULT_AUTHORIZATION_GC_MAINTENANCE_INTERVAL = timedelta(minutes=1)
_DEFAULT_AUTHORIZATION_GC_BATCHES_PER_TICK = 4
_CONNECTOR_AUTHORIZATION_GC_LOCK_ID = 0x4341504743000001

log = logging.getLogger(__name__)

_EXPIRED_RETENTION_SECONDS = EnvVar(
    'CONNECTOR_AUTHORIZATION_EXPIRED_RETENTION_SECONDS',
    int,
    required=False,
    default_value=int(_DEFAULT_INACTIVE_RETENTION.total_seconds()),
)
_REVOKED_RETENTION_SECONDS = EnvVar(
    'CONNECTOR_AUTHORIZATION_REVOKED_RETENTION_SECONDS',
    int,
    required=False,
    default_value=int(_DEFAULT_INACTIVE_RETENTION.total_seconds()),
)
_AUTHORIZATION_GC_BATCH_SIZE = EnvVar(
    'CONNECTOR_AUTHORIZATION_GC_BATCH_SIZE',
    int,
    required=False,
    default_value=_DEFAULT_AUTHORIZATION_GC_BATCH_SIZE,
)
_AUTHORIZATION_USE_GC_BATCH_SIZE = EnvVar(
    'CONNECTOR_AUTHORIZATION_USE_GC_BATCH_SIZE',
    int,
    required=False,
    default_value=_DEFAULT_AUTHORIZATION_USE_GC_BATCH_SIZE,
)
_AUTHORIZATION_GC_MIN_INTERVAL_SECONDS = EnvVar(
    'CONNECTOR_AUTHORIZATION_GC_MIN_INTERVAL_SECONDS',
    int,
    required=False,
    default_value=int(_DEFAULT_AUTHORIZATION_GC_MIN_INTERVAL.total_seconds()),
)
_AUTHORIZATION_GC_MAINTENANCE_INTERVAL_SECONDS = EnvVar(
    'CONNECTOR_AUTHORIZATION_GC_MAINTENANCE_INTERVAL_SECONDS',
    int,
    required=False,
    default_value=int(
        _DEFAULT_AUTHORIZATION_GC_MAINTENANCE_INTERVAL.total_seconds()),
)
_AUTHORIZATION_GC_BATCHES_PER_TICK = EnvVar(
    'CONNECTOR_AUTHORIZATION_GC_BATCHES_PER_TICK',
    int,
    required=False,
    default_value=_DEFAULT_AUTHORIZATION_GC_BATCHES_PER_TICK,
)


class ConnectorAuthorizationError(PermissionError):
    """A workflow capability was absent, stale, or used out of scope."""


@dataclasses.dataclass(frozen=True)
class ConnectorAuthorizationRetentionPolicy:
    """Bounded retention for inactive capability audit records.

    Expiry and revocation are independent terminal clocks. By default, Kumo
    retains either kind of inactive authorization for seven days, then removes
    its activity-use records before removing the authorization. The environment
    variables declared above let deployments increase that audit window and
    tune bounded cleanup work without exposing capability or principal data.
    """

    expired_retention: timedelta = _DEFAULT_INACTIVE_RETENTION
    revoked_retention: timedelta = _DEFAULT_INACTIVE_RETENTION
    authorization_batch_size: int = _DEFAULT_AUTHORIZATION_GC_BATCH_SIZE
    use_batch_size: int = _DEFAULT_AUTHORIZATION_USE_GC_BATCH_SIZE
    opportunistic_min_interval: timedelta = (
        _DEFAULT_AUTHORIZATION_GC_MIN_INTERVAL)
    maintenance_interval: timedelta = (
        _DEFAULT_AUTHORIZATION_GC_MAINTENANCE_INTERVAL)
    maintenance_batches_per_tick: int = (
        _DEFAULT_AUTHORIZATION_GC_BATCHES_PER_TICK)

    def __post_init__(self) -> None:
        if self.expired_retention < timedelta(0):
            raise ValueError('Expired authorization retention cannot be negative')
        if self.revoked_retention < timedelta(0):
            raise ValueError('Revoked authorization retention cannot be negative')
        if self.authorization_batch_size < 1:
            raise ValueError('Authorization cleanup batch size must be positive')
        if self.use_batch_size < 1:
            raise ValueError('Authorization-use cleanup batch size must be positive')
        if self.opportunistic_min_interval < timedelta(0):
            raise ValueError('Authorization cleanup interval cannot be negative')
        if self.maintenance_interval <= timedelta(0):
            raise ValueError(
                'Authorization maintenance interval must be positive')
        if self.maintenance_batches_per_tick < 1:
            raise ValueError(
                'Authorization maintenance batches per tick must be positive')

    @classmethod
    def from_env(cls) -> 'ConnectorAuthorizationRetentionPolicy':
        return cls(
            expired_retention=timedelta(
                seconds=_EXPIRED_RETENTION_SECONDS.parse_value()),
            revoked_retention=timedelta(
                seconds=_REVOKED_RETENTION_SECONDS.parse_value()),
            authorization_batch_size=_AUTHORIZATION_GC_BATCH_SIZE.parse_value(),
            use_batch_size=_AUTHORIZATION_USE_GC_BATCH_SIZE.parse_value(),
            opportunistic_min_interval=timedelta(
                seconds=_AUTHORIZATION_GC_MIN_INTERVAL_SECONDS.parse_value()),
            maintenance_interval=timedelta(
                seconds=(
                    _AUTHORIZATION_GC_MAINTENANCE_INTERVAL_SECONDS.parse_value()
                )),
            maintenance_batches_per_tick=(
                _AUTHORIZATION_GC_BATCHES_PER_TICK.parse_value()),
        )


@dataclasses.dataclass(frozen=True)
class ConnectorAuthorizationCleanupResult:
    """Counts-only cleanup result; never carries capability audit contents."""

    authorizations_deleted: int = 0
    uses_deleted: int = 0


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def canonical_request_digest(value: Any) -> str:
    """Hash a stable, credential-free authorization request shape."""
    def normalize(item: Any) -> Any:
        if dataclasses.is_dataclass(item):
            return {
                field.name: normalize(getattr(item, field.name))
                for field in dataclasses.fields(item)
            }
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, dict):
            return {
                str(key): normalize(val)
                for key, val in sorted(item.items(),
                                       key=lambda pair: str(pair[0]))
            }
        if isinstance(item, (list, tuple)):
            return [normalize(element) for element in item]
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(
            f'Unsupported authorization digest value {type(item).__name__}')

    encoded = json.dumps(normalize(value), sort_keys=True,
                         separators=(',', ':'),
                         ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


class ConnectorAuthorizationService:
    """Issue and consume opaque, workflow-scoped connector capabilities."""
    def __init__(
        self,
        state_db: StateDB,
        *,
        retention_policy: ConnectorAuthorizationRetentionPolicy | None = None,
    ) -> None:
        self._state_db = state_db
        self._authorizations = JsonRepository('connector_authorizations',
                                              ConnectorAuthorizationDBEntry)
        self._uses = JsonRepository('connector_authorization_uses',
                                    ConnectorAuthorizationUseDBEntry)
        self._retention_policy = (
            retention_policy or ConnectorAuthorizationRetentionPolicy.from_env())
        self._cleanup_rate_lock = threading.Lock()
        self._next_cleanup_monotonic = 0.0
        self._authorization_writes_since_cleanup = 0
        self._use_writes_since_cleanup = 0
        self._maintenance_stop: asyncio.Event | None = None
        self._maintenance_task: asyncio.Task[None] | None = None

    async def start_maintenance(self) -> None:
        """Start this process's bounded periodic cleanup owner.

        Every REST replica may run this loop. PostgreSQL uses a non-blocking
        transaction advisory lock so only one replica does work per tick;
        SQLite write transactions are already serialized. A one-batch startup
        cleanup is a fallback before idle periodic draining begins.
        """
        if self._maintenance_task is not None:
            if not self._maintenance_task.done():
                return
            await self._maintenance_task
        self._maintenance_stop = asyncio.Event()
        try:
            result = await self.purge_stale()
        except Exception as error:
            log.warning(
                'Connector authorization startup cleanup failed '
                '(error_type=%s)',
                type(error).__name__,
            )
        else:
            _log_cleanup_counts('startup', result)
        stop = self._maintenance_stop
        self._maintenance_task = asyncio.create_task(
            self._run_maintenance(stop),
            name='connector-authorization-maintenance',
        )

    async def stop_maintenance(self) -> None:
        """Stop periodic cleanup and wait until its current bounded tick ends."""
        stop = self._maintenance_stop
        task = self._maintenance_task
        self._maintenance_stop = None
        self._maintenance_task = None
        if stop is None or task is None:
            return
        stop.set()
        await task

    async def run_maintenance_tick(
        self,
        *,
        now: datetime | None = None,
    ) -> ConnectorAuthorizationCleanupResult:
        """Drain at most the configured number of bounded cleanup batches."""
        total = ConnectorAuthorizationCleanupResult()
        for _ in range(self._retention_policy.maintenance_batches_per_tick):
            result = await self.purge_stale(now=now)
            total = ConnectorAuthorizationCleanupResult(
                authorizations_deleted=(total.authorizations_deleted
                                        + result.authorizations_deleted),
                uses_deleted=total.uses_deleted + result.uses_deleted,
            )
            if not result.authorizations_deleted and not result.uses_deleted:
                break
        return total

    async def _run_maintenance(self, stop: asyncio.Event) -> None:
        interval = self._retention_policy.maintenance_interval.total_seconds()
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except TimeoutError:
                try:
                    result = await self.run_maintenance_tick()
                except Exception as error:
                    # A transient database failure must not permanently stop
                    # retention enforcement. Never log capability contents.
                    log.warning(
                        'Connector authorization maintenance failed '
                        '(error_type=%s)',
                        type(error).__name__,
                    )
                else:
                    _log_cleanup_counts('periodic', result)

    async def purge_stale(
        self,
        *,
        now: datetime | None = None,
    ) -> ConnectorAuthorizationCleanupResult:
        """Delete one bounded batch of retained inactive capabilities.

        Callers may invoke this operation repeatedly to drain a backlog. Each
        call bounds authorization candidates and dependent use-row deletions
        separately, and never returns identifiers, tokens, or principal data.
        """
        async with self._state_db.write_tx() as tx:
            return await self._purge_stale_in_tx(tx, now=now or now_utc())

    async def _maybe_purge_stale_in_tx(
        self,
        tx: StateTx,
        *,
        authorization_writes: int = 0,
        use_writes: int = 0,
    ) -> None:
        monotonic_now = time.monotonic()
        with self._cleanup_rate_lock:
            self._authorization_writes_since_cleanup += authorization_writes
            self._use_writes_since_cleanup += use_writes
            volume_due = (
                self._authorization_writes_since_cleanup
                >= self._retention_policy.authorization_batch_size
                or self._use_writes_since_cleanup
                >= self._retention_policy.use_batch_size)
            if (not volume_due
                    and monotonic_now < self._next_cleanup_monotonic):
                return
            self._next_cleanup_monotonic = (
                monotonic_now
                + self._retention_policy.opportunistic_min_interval.total_seconds()
            )
            self._authorization_writes_since_cleanup = 0
            self._use_writes_since_cleanup = 0
        try:
            await self._purge_stale_in_tx(tx, now=now_utc())
        except BaseException:
            # Permit the next issuance to retry immediately if the transaction
            # is rolled back. No capability details are logged here.
            with self._cleanup_rate_lock:
                self._next_cleanup_monotonic = 0.0
            raise

    async def _purge_stale_in_tx(
        self,
        tx: StateTx,
        *,
        now: datetime,
    ) -> ConnectorAuthorizationCleanupResult:
        if not await _try_acquire_cleanup_lock(tx):
            return ConnectorAuthorizationCleanupResult()
        policy = self._retention_policy
        expired_cutoff = now - policy.expired_retention
        revoked_cutoff = now - policy.revoked_retention
        candidate_ids = await _select_stale_authorization_ids(
            tx,
            expired_cutoff=expired_cutoff,
            revoked_cutoff=revoked_cutoff,
            limit=policy.authorization_batch_size,
        )
        if not candidate_ids:
            return ConnectorAuthorizationCleanupResult()

        use_ids = await _select_authorization_use_ids(
            tx,
            authorization_ids=candidate_ids,
            limit=policy.use_batch_size,
        )
        uses_deleted = await _delete_ids(
            tx, 'connector_authorization_uses', use_ids)
        authorizations_deleted = await _delete_stale_authorizations_without_uses(
            tx,
            authorization_ids=candidate_ids,
            expired_cutoff=expired_cutoff,
            revoked_cutoff=revoked_cutoff,
        )
        return ConnectorAuthorizationCleanupResult(
            authorizations_deleted=authorizations_deleted,
            uses_deleted=uses_deleted,
        )

    async def issue(
        self,
        *,
        principal: AuthPrincipal,
        auth_backend: str,
        resource_kind: ConnectorAuthorizationResourceKind,
        resource_id: str,
        project_id: str,
        request_digest: str,
        allowed_actions: Iterable[ConnectorCapabilityAction],
        deployment_id: str,
        temporal_namespace: str,
    ) -> tuple[ConnectorAuthorizationDBEntry, str]:
        async with self._state_db.write_tx() as tx:
            return await self.issue_in_tx(
                tx,
                principal=principal,
                auth_backend=auth_backend,
                resource_kind=resource_kind,
                resource_id=resource_id,
                project_id=project_id,
                request_digest=request_digest,
                allowed_actions=allowed_actions,
                deployment_id=deployment_id,
                temporal_namespace=temporal_namespace,
            )

    async def issue_in_tx(
        self,
        tx: StateTx,
        *,
        principal: AuthPrincipal,
        auth_backend: str,
        resource_kind: ConnectorAuthorizationResourceKind,
        resource_id: str,
        project_id: str,
        request_digest: str,
        allowed_actions: Iterable[ConnectorCapabilityAction],
        deployment_id: str,
        temporal_namespace: str,
        parent_authorization_id: ConnectorAuthorizationID | None = None,
        expires_at: datetime | None = None,
    ) -> tuple[ConnectorAuthorizationDBEntry, str]:
        return await self._issue_in_tx(
            tx,
            principal_id=str(principal.user_id),
            principal_name=principal.user_name,
            auth_backend=auth_backend,
            resource_kind=resource_kind,
            resource_id=resource_id,
            project_id=project_id,
            request_digest=request_digest,
            allowed_actions=allowed_actions,
            deployment_id=deployment_id,
            temporal_namespace=temporal_namespace,
            parent_authorization_id=parent_authorization_id,
            expires_at=expires_at,
        )

    async def _issue_in_tx(
        self,
        tx: StateTx,
        *,
        principal_id: str,
        principal_name: str,
        auth_backend: str,
        resource_kind: ConnectorAuthorizationResourceKind,
        resource_id: str,
        project_id: str,
        request_digest: str,
        allowed_actions: Iterable[ConnectorCapabilityAction],
        deployment_id: str,
        temporal_namespace: str,
        parent_authorization_id: ConnectorAuthorizationID | None = None,
        expires_at: datetime | None = None,
    ) -> tuple[ConnectorAuthorizationDBEntry, str]:
        actions = tuple(
            sorted(set(allowed_actions), key=lambda item: item.value))
        if not actions:
            raise ValueError('Connector authorization requires an action')
        await self._maybe_purge_stale_in_tx(tx, authorization_writes=1)
        token = secrets.token_urlsafe(32)
        entry = ConnectorAuthorizationDBEntry(
            id=ConnectorAuthorizationID.generate(),
            token_hash=_token_hash(token),
            principal_id=principal_id,
            principal_name=principal_name,
            auth_backend=auth_backend,
            resource_kind=resource_kind,
            resource_id=resource_id,
            project_id=project_id,
            request_digest=request_digest,
            allowed_actions=actions,
            deployment_id=deployment_id,
            temporal_namespace=temporal_namespace,
            workflow_id='',
            workflow_run_id='',
            expires_at=(expires_at or now_utc() + _UNBOUND_AUTHORIZATION_TTL),
            parent_authorization_id=parent_authorization_id,
            timestamps=get_current_timestamps(),
        )
        await self._authorizations.upsert(tx, str(entry.id), entry)
        return entry, token

    async def bind_workflow(
        self,
        token: str,
        *,
        deployment_id: str,
        temporal_namespace: str,
        workflow_id: str,
        expires_at: datetime,
    ) -> tuple[ConnectorAuthorizationDBEntry, bool]:
        if not workflow_id:
            raise ValueError('workflow_id is required')
        async with self._state_db.write_tx() as tx:
            entry = await self._get_by_token_in_tx(tx, token)
            self._require_live(entry)
            if (entry.deployment_id != deployment_id
                    or entry.temporal_namespace != temporal_namespace):
                raise ConnectorAuthorizationError(
                    'Connector authorization deployment mismatch')
            if entry.workflow_id and entry.workflow_id != workflow_id:
                raise ConnectorAuthorizationError(
                    'Connector authorization workflow mismatch')
            if entry.workflow_id:
                return entry, False
            entry.workflow_id = workflow_id
            entry.expires_at = expires_at
            entry.timestamps = update_timestamps(entry.timestamps)
            await self._authorizations.upsert(tx, str(entry.id), entry)
            return entry, True

    async def revoke(self, token: str) -> bool:
        async with self._state_db.write_tx() as tx:
            try:
                entry = await self._get_by_token_in_tx(tx, token)
            except ConnectorAuthorizationError:
                return False
            if entry.revoked_at is None:
                entry.revoked_at = now_utc()
                entry.timestamps = update_timestamps(entry.timestamps)
                await self._authorizations.upsert(tx, str(entry.id), entry)
            return True

    async def revoke_workflow(
        self,
        *,
        workflow_id: str,
        deployment_id: str,
        temporal_namespace: str,
    ) -> int:
        """Revoke live capabilities after an accepted cancel/terminate."""
        if not workflow_id:
            raise ValueError('workflow_id is required')
        revoked = 0
        async with self._state_db.write_tx() as tx:
            rows = await self._authorizations.list_by_field(
                tx, 'workflow_id', workflow_id)
            for row in rows:
                entry = row.payload
                if (entry.deployment_id != deployment_id
                        or entry.temporal_namespace != temporal_namespace
                        or entry.revoked_at is not None):
                    continue
                entry.revoked_at = now_utc()
                entry.timestamps = update_timestamps(entry.timestamps)
                await self._authorizations.upsert(tx, row.id, entry)
                revoked += 1
        return revoked

    async def authorize_activity(
        self,
        token: str,
        *,
        deployment_id: str,
        temporal_namespace: str,
        workflow_id: str,
        workflow_run_id: str,
        activity_id: str,
        action: ConnectorCapabilityAction,
        resource_kind: ConnectorAuthorizationResourceKind,
        resource_id: str,
        project_id: str = '',
        request_digest: str = '',
    ) -> ConnectorAuthorizationDBEntry:
        if not workflow_id or not workflow_run_id or not activity_id:
            raise ConnectorAuthorizationError(
                'Connector authorization requires Temporal provenance')
        async with self._state_db.write_tx() as tx:
            entry = await self._get_by_token_in_tx(tx, token)
            self._require_live(entry)
            if (entry.deployment_id != deployment_id
                    or entry.temporal_namespace != temporal_namespace
                    or entry.workflow_id != workflow_id):
                raise ConnectorAuthorizationError(
                    'Connector authorization workflow provenance mismatch')
            if (entry.resource_kind != resource_kind
                    or entry.resource_id != resource_id
                    or entry.project_id != project_id
                    or entry.request_digest != request_digest):
                raise ConnectorAuthorizationError(
                    'Connector authorization resource scope mismatch')
            if action not in entry.allowed_actions:
                raise ConnectorAuthorizationError(
                    'Connector authorization action is not allowed')
            if (entry.workflow_run_id
                    and entry.workflow_run_id != workflow_run_id):
                raise ConnectorAuthorizationError(
                    'Connector authorization run mismatch')
            if not entry.workflow_run_id:
                entry.workflow_run_id = workflow_run_id
                entry.timestamps = update_timestamps(entry.timestamps)
                await self._authorizations.upsert(tx, str(entry.id), entry)

            existing = await self._uses.find_one_by_fields(
                tx, {
                    'authorization_id': str(entry.id),
                    'workflow_run_id': workflow_run_id,
                    'activity_id': activity_id,
                    'action': action,
                })
            if existing is None:
                use = ConnectorAuthorizationUseDBEntry(
                    id=ConnectorAuthorizationUseID.generate(),
                    authorization_id=entry.id,
                    workflow_id=workflow_id,
                    workflow_run_id=workflow_run_id,
                    activity_id=activity_id,
                    action=action,
                    timestamps=get_current_timestamps(),
                )
                await self._uses.upsert(tx, str(use.id), use)
                await self._maybe_purge_stale_in_tx(tx, use_writes=1)
            elif (existing.payload.workflow_id != workflow_id
                  or existing.payload.authorization_id != entry.id):
                raise ConnectorAuthorizationError(
                    'Connector authorization retry identity mismatch')
            return entry

    async def delegate_to_child(
        self,
        parent: ConnectorAuthorizationDBEntry,
        *,
        child_workflow_id: str,
        child_actions: Iterable[ConnectorCapabilityAction],
    ) -> tuple[ConnectorAuthorizationDBEntry, str]:
        """Mint a child token after the parent delegation use is recorded."""
        requested = set(child_actions)
        if not child_workflow_id or not requested:
            raise ConnectorAuthorizationError(
                'Child workflow authorization scope is required')
        if ConnectorCapabilityAction.DELEGATE_CHILD not in parent.allowed_actions:
            raise ConnectorAuthorizationError(
                'Parent authorization cannot delegate')
        if not requested.issubset(set(parent.allowed_actions)):
            raise ConnectorAuthorizationError(
                'Child authorization exceeds its parent action scope')
        async with self._state_db.write_tx() as tx:
            row = await self._authorizations.get(tx, str(parent.id))
            if row is None:
                raise ConnectorAuthorizationError(
                    'Parent authorization is invalid')
            current = row.payload
            self._require_live(current)
            if (current.token_hash != parent.token_hash
                    or current.workflow_id != parent.workflow_id
                    or current.workflow_run_id != parent.workflow_run_id):
                raise ConnectorAuthorizationError(
                    'Parent authorization changed during delegation')
            if (ConnectorCapabilityAction.DELEGATE_CHILD
                    not in current.allowed_actions
                    or not requested.issubset(set(current.allowed_actions))):
                raise ConnectorAuthorizationError(
                    'Child authorization exceeds its stored parent scope')
            child, token = await self._issue_in_tx(
                tx,
                principal_id=current.principal_id,
                principal_name=current.principal_name,
                auth_backend=current.auth_backend,
                resource_kind=current.resource_kind,
                resource_id=current.resource_id,
                project_id=current.project_id,
                request_digest=current.request_digest,
                allowed_actions=requested,
                deployment_id=current.deployment_id,
                temporal_namespace=current.temporal_namespace,
                parent_authorization_id=current.id,
                expires_at=current.expires_at,
            )
            child.workflow_id = child_workflow_id
            child.timestamps = update_timestamps(child.timestamps)
            await self._authorizations.upsert(tx, str(child.id), child)
            return child, token

    async def _get_by_token_in_tx(
        self,
        tx: StateTx,
        token: str,
    ) -> ConnectorAuthorizationDBEntry:
        if not token:
            raise ConnectorAuthorizationError(
                'Connector authorization is required')
        expected = _token_hash(token)
        row = await self._authorizations.find_one_by_field(
            tx, 'token_hash', expected)
        if row is None or not hmac.compare_digest(row.payload.token_hash,
                                                  expected):
            raise ConnectorAuthorizationError(
                'Connector authorization is invalid')
        return row.payload

    @staticmethod
    def _require_live(entry: ConnectorAuthorizationDBEntry) -> None:
        if entry.revoked_at is not None or entry.expires_at <= now_utc():
            raise ConnectorAuthorizationError(
                'Connector authorization is no longer active')


def _stale_authorization_predicate(
    tx: StateTx,
    args: list[object],
    *,
    expired_cutoff: datetime,
    revoked_cutoff: datetime,
) -> str:
    args.extend((expired_cutoff.isoformat(), revoked_cutoff.isoformat()))
    expires_at = _json_value(tx, 'expires_at')
    revoked_at = _json_value(tx, 'revoked_at')
    return (
        f'({expires_at} <= {_placeholder(tx, len(args) - 1)} '
        f'OR ({revoked_at} IS NOT NULL '
        f'AND {revoked_at} <= {_placeholder(tx, len(args))}))')


async def _try_acquire_cleanup_lock(tx: StateTx) -> bool:
    if tx.backend != 'postgres':
        return True
    row = await tx.fetchrow(
        'SELECT pg_try_advisory_xact_lock($1)',
        _CONNECTOR_AUTHORIZATION_GC_LOCK_ID,
    )
    return bool(row[0]) if row is not None else False


def _log_cleanup_counts(
    trigger: str,
    result: ConnectorAuthorizationCleanupResult,
) -> None:
    if result.authorizations_deleted or result.uses_deleted:
        log.info(
            'Connector authorization cleanup completed '
            '(trigger=%s authorizations_deleted=%d uses_deleted=%d)',
            trigger,
            result.authorizations_deleted,
            result.uses_deleted,
        )


async def _select_stale_authorization_ids(
    tx: StateTx,
    *,
    expired_cutoff: datetime,
    revoked_cutoff: datetime,
    limit: int,
) -> list[str]:
    args: list[object] = []
    stale = _stale_authorization_predicate(
        tx,
        args,
        expired_cutoff=expired_cutoff,
        revoked_cutoff=revoked_cutoff,
    )
    args.append(limit)
    rows = await tx.fetch(
        f"""
        SELECT id
        FROM connector_authorizations
        WHERE {stale}
        ORDER BY id
        LIMIT {_placeholder(tx, len(args))}
        """,
        *args,
    )
    return [row_str(row, 'id', 0) for row in rows]


async def _select_authorization_use_ids(
    tx: StateTx,
    *,
    authorization_ids: list[str],
    limit: int,
) -> list[str]:
    if not authorization_ids:
        return []
    args: list[object] = []
    authorization_predicate = _in_predicate(
        tx,
        _json_value(tx, 'authorization_id'),
        authorization_ids,
        args,
    )
    args.append(limit)
    rows = await tx.fetch(
        f"""
        SELECT id
        FROM connector_authorization_uses
        WHERE {authorization_predicate}
        ORDER BY id
        LIMIT {_placeholder(tx, len(args))}
        """,
        *args,
    )
    return [row_str(row, 'id', 0) for row in rows]


async def _delete_ids(
    tx: StateTx,
    table_name: str,
    row_ids: list[str],
) -> int:
    if not row_ids:
        return 0
    args: list[object] = []
    id_predicate = _in_predicate(tx, 'id', row_ids, args)
    return await _execute_count(
        tx,
        f'DELETE FROM {table_name} WHERE {id_predicate}',
        args,
    )


async def _delete_stale_authorizations_without_uses(
    tx: StateTx,
    *,
    authorization_ids: list[str],
    expired_cutoff: datetime,
    revoked_cutoff: datetime,
) -> int:
    if not authorization_ids:
        return 0
    args: list[object] = []
    stale = _stale_authorization_predicate(
        tx,
        args,
        expired_cutoff=expired_cutoff,
        revoked_cutoff=revoked_cutoff,
    )
    id_predicate = _in_predicate(tx, 'id', authorization_ids, args)
    use_authorization_id = _json_value(tx, 'authorization_id')
    return await _execute_count(
        tx,
        f"""
        DELETE FROM connector_authorizations
        WHERE {stale}
          AND {id_predicate}
          AND NOT EXISTS (
              SELECT 1
              FROM connector_authorization_uses
              WHERE {use_authorization_id} = connector_authorizations.id
          )
        """,
        args,
    )


async def _execute_count(
    tx: StateTx,
    sql: str,
    args: list[object],
) -> int:
    result = await tx.execute(sql, *args)
    if tx.backend == 'postgres':
        try:
            return int(str(result).split()[-1])
        except (IndexError, ValueError):
            return 0
    row = await tx.fetchrow('SELECT changes()')
    return int(row[0] if row is not None else 0)
