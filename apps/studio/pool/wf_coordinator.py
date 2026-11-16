# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow-side coordinator: the single integration point between
Kumo Temporal workflows and the diskgraph pool.

``snapshot_placement_session_ctx`` acquires a placement and releases it on
exit. ``run_pooled_activity`` additionally handles activity dispatch and
worker-loss recovery. Both paths are idempotent on ``(snapshot_id, job_id)``.
"""
import json
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncIterator, Callable, Iterator, List, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    TimeoutError,
    TimeoutType,
)

from common_utils.error import PublicException
from io_utils import logging
from pool import timeouts
from pool.activities import (
    PERMANENT_SNAPSHOT_MISS_ERROR_TYPE,
    get_pool_stats_activity,
    place_snapshot_activity,
    release_snapshot_activity,
)
from pool.dataclasses import (
    PlacementRequest,
    PlacementResult,
    PoolStatsFilter,
    ResourceHint,
    SnapshotActivity,
    SnapshotID,
    WorkerStatus,
)
from pool.task_queues import ACTIVITY_TQ_PREFIX
from temporalio_commons.error_utils import is_cancelled_exception

log = logging.getLogger(__name__)

# Centralized schedule_to_start budget for pooled activities dispatched by
# ``run_pooled_activity``: how long the activity task may sit on the pool
# worker's queue before a worker picks it up. Generous because a freshly
# minted durable worker (Karpenter cold start + image pull + daemon boot)
# can take minutes to start polling its queue. Set once here so the six call
# sites don't each repeat it.
_POOLED_ACTIVITY_SCHEDULE_TO_START = timedelta(minutes=30)

# How long the runner waits between health re-polls while the placed worker
# is STALE, and the ceiling on that wait before giving up on the worker and
# re-placing on a fresh one. 240s comfortably spans the STALE→DEAD window
# (WORKER_DEAD_THRESHOLD_SECS = 300s, minus the 60s STALE entry) so a worker
# that is going to die is observed DEAD before this ceiling.
_STALE_REPOLL_INTERVAL = timedelta(seconds=30)
_STALE_WAIT_CEILING = timedelta(seconds=240)


def task_queue_kwargs(task_queue: str) -> dict:
    """``workflow.execute_activity`` kwargs for the pool-yielded queue."""
    return {'task_queue': task_queue}


# Retry for transient placement failures (Karpenter cold start, ECR
# throttling, K8s API blips). This is ALSO the STALE-wait mechanism: a STALE
# home makes ``route()`` raise a *retryable* ``HomeStaleError``, so Temporal
# keeps retrying the placement until the home settles — recovers to READY
# (``route`` co-locates) or ages to DEAD (``route`` mints fresh). We
# deliberately set NO ``maximum_attempts``: the only bound is
# ``schedule_to_close_timeout = PLACE_SCHEDULE_TO_CLOSE`` (45 min). That
# decouples the retry budget from the FSM's STALE→DEAD window
# (``WORKER_DEAD_THRESHOLD_SECS - WORKER_STALE_THRESHOLD_SECS``); previously
# the 8-attempt budget (~255s) sat only ~15s above that window, so bumping
# ``WORKER_DEAD_THRESHOLD_SECS`` (e.g. for BYOC) could silently starve the
# STALE wait and leak ``HomeStaleError`` into ``run_pooled_activity``. The
# STALE→DEAD transition (≤ ``WORKER_DEAD_THRESHOLD_SECS``, default 300s) is
# the real terminator, far under 45 min. Deterministic failures (missing
# IRSA, invalid spec, RBAC denial) still fail fast — Temporal never retries
# the ``ApplicationError(non_retryable=True)`` ``place_snapshot_activity``
# raises for them, regardless of this policy.
_PLACE_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=60),
)

# Release runs in finally — bounded retry so cleanup can't mask the
# original exit exception. Lease TTL / sweep mop up zombies.
_RELEASE_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=5,
)

# Read-only health probe; cheap, idempotent. A short bounded retry so a
# transient controller blip doesn't abort the runner's re-route decision.
_POOL_STATS_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=15),
    maximum_attempts=4,
)
_POOL_STATS_START_TO_CLOSE = timedelta(seconds=30)
_POOL_STATS_SCHEDULE_TO_CLOSE = timedelta(minutes=2)

_POOLED_ACTIVITY_NO_POLLER_MESSAGE = (
    'The DiskGraph compute task timed out before a worker picked it up. Retry '
    'the job. If the problem continues, contact Kumo support.')


def _iter_error_chain(err: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    stack: List[Optional[BaseException]] = [err]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack.append(getattr(current, 'cause', None))
        stack.append(getattr(current, '__cause__', None))


def _is_schedule_to_start_timeout(err: BaseException) -> bool:
    """Whether ``err`` contains a Temporal ScheduleToStart timeout."""
    return any(
        isinstance(current, TimeoutError)
        and getattr(current, 'type', None) == TimeoutType.SCHEDULE_TO_START
        for current in _iter_error_chain(err))


def _is_permanent_snapshot_miss(err: ActivityError) -> bool:
    """Return True when the activity failed on a consumer-mode snapshot miss.

    This is a placement failure, not a healthy-worker activity failure: the
    current worker cannot satisfy the snapshot without the controller running a
    fresh daemon ensure_snapshot placement.
    """
    return any(
        getattr(cause, 'type', None) == PERMANENT_SNAPSHOT_MISS_ERROR_TYPE
        for cause in _iter_error_chain(err))


def _build_placement_request(
    snapshot_id: SnapshotID,
    activity: SnapshotActivity,
    resource_hint: Optional[ResourceHint],
    shard_index: Optional[int],
    source_size_bytes: int,
) -> PlacementRequest:
    """Build the ``PlacementRequest`` for a pool placement. ``job_id`` is the
    parent workflow_id (suffixed per shard for sharded fan-out) so sibling
    shards hold distinct leases under one workflow. Shared by
    placement contexts and ``run_pooled_activity`` so the lease identity
    is computed one way."""
    workflow_id = workflow.info().workflow_id
    job_id = (f'{workflow_id}#shard-{shard_index}'
              if shard_index is not None else workflow_id)
    return PlacementRequest(
        snapshot_id=snapshot_id,
        job_id=job_id,
        workflow_id=workflow_id,
        activity=activity,
        resource_hint=resource_hint or ResourceHint(),
        source_size_bytes=source_size_bytes,
    )


def _stamp_placement_memo(result: PlacementResult, activity: SnapshotActivity,
                          shard_index: Optional[int]) -> None:
    """Stamp strategy-provided observability metadata onto the workflow memo
    so the Temporal Cloud UI links to the placed pod's logs. See the inline
    rationale in ``snapshot_placement_session_ctx`` for the JSON encoding."""
    if not result.debug_info:
        return
    memo_key = f'pool_{activity.value}'
    if shard_index is not None:
        memo_key = f'{memo_key}_shard_{shard_index}'
    workflow.upsert_memo({memo_key: json.dumps(result.debug_info)})


# Actionable, user-facing message when a placement runs out of its retry
# budget with no more-specific cause. See :func:`_place_snapshot`.
_POOL_PLACEMENT_TIMEOUT_MESSAGE = (
    'Your job could not acquire a diskgraph pool worker within the allowed '
    'time. The pool may be unavailable or at capacity - for example the pool '
    'controller is unavailable or restarting (a crash-looping controller, '
    'e.g. because a required GPU pool could not be validated at startup), or '
    'no worker could be provisioned. Please retry; if the problem persists, '
    'contact Kumo support or your cluster administrator to check the pool '
    'controller (kumo-poolcontroller) health.')


def _classify_placement_failure(err: BaseException) -> tuple[bool, bool]:
    """Walk ``err``'s cause chain once and report ``(has_timeout,
    has_specific_public_exception)``.

    Follows BOTH temporalio's ``.cause`` and Python's ``__cause__`` so a link
    is never missed regardless of how the wrapper set it. ``has_timeout`` is
    True if any link is a Temporal ``TimeoutError`` (the placement ran out of
    budget). ``has_specific_public_exception`` is True if any link is an
    ``ApplicationError`` carrying a decodable ``PublicException`` detail - i.e.
    an allowlisted placement error that ``place_snapshot_activity`` already
    mapped. We check for that detail directly rather than via
    ``extract_public_exception``, which SYNTHESIZES a
    generic (non-None) PublicException for any bare TimeoutError and so cannot
    distinguish a mapped message from the fallback.
    """
    has_timeout = False
    has_public = False
    for cur in _iter_error_chain(err):
        if isinstance(cur, TimeoutError):
            has_timeout = True
        if isinstance(cur, ApplicationError):
            details = getattr(cur, 'details', None)
            if details:
                try:
                    if PublicException.from_dict(details[0]) is not None:
                        has_public = True
                except Exception:
                    pass
    return has_timeout, has_public


async def _place_snapshot(request: PlacementRequest) -> PlacementResult:
    """Dispatch ``place_snapshot_activity`` with the standard place timeouts +
    retry policy. The place retry policy already covers the transient STALE
    home window: route() raises a (retryable) HomeStaleError, so the place is
    retried until the home recovers (READY) or ages to DEAD (re-route mints a
    fresh worker)."""
    try:
        return await workflow.execute_activity(
            place_snapshot_activity,
            request,
            start_to_close_timeout=timeouts.PLACE_START_TO_CLOSE,
            schedule_to_close_timeout=timeouts.PLACE_SCHEDULE_TO_CLOSE,
            retry_policy=_PLACE_RETRY_POLICY,
        )
    except (ActivityError, TimeoutError) as err:
        # A placement that exhausts its schedule_to_close budget surfaces here
        # as a timeout. When the pool controller is unavailable / crash-looping
        # (kumo-ai/kumo#29865) the controller never runs, so it raises no
        # ProvisionTimeoutError / ProvisionTerminalError to carry a
        # PublicException, AND the wrapper activity is killed by its own
        # start_to_close before place_snapshot's inner AutoRPC can surface a
        # mappable error (the RPC execution_timeout is inflated to the full
        # placement budget, > start_to_close). Left unhandled the job then
        # degrades to the generic "your data is too large" timeout message,
        # hiding the real cause and every remedy. Translate ONLY a
        # budget-exhausted timeout that carries no already-mapped placement
        # PublicException; preserve a specific one when it IS on the chain (a
        # non-retryable ProvisionTerminalError always is; a ProvisionTimeoutError
        # that only surfaces via the timeout is preserved best-effort, and the
        # generic message below is a strict superset otherwise), and leave
        # non-timeout failures untouched. Terminal: the placement retry budget
        # is already spent, so an outer retry would only repeat it.
        has_timeout, has_public = _classify_placement_failure(err)
        if has_timeout and not has_public:
            raise ApplicationError(
                f'diskgraph placement did not complete for '
                f'snapshot={request.snapshot_id} job={request.job_id}',
                PublicException(
                    message=_POOL_PLACEMENT_TIMEOUT_MESSAGE).to_dict(),
                type='PoolPlacementUnavailableError',
                non_retryable=True,
            ) from err
        raise


async def _release_snapshot(request: PlacementRequest) -> bool:
    """Release ``request`` idempotently and report whether it completed.

    Failures are logged rather than raised so cleanup cannot mask the original
    workflow failure. The controller lease TTL and dangling-lease sweep recover
    a zombie lease if release never lands.
    """
    try:
        await workflow.execute_activity(
            release_snapshot_activity,
            args=[request.snapshot_id, request.job_id],
            start_to_close_timeout=timeouts.RELEASE_START_TO_CLOSE,
            schedule_to_close_timeout=timeouts.RELEASE_SCHEDULE_TO_CLOSE,
            retry_policy=_RELEASE_RETRY_POLICY,
        )
        return True
    except Exception:
        workflow.logger.exception(
            'release_snapshot_activity failed for snapshot=%s job=%s — '
            'controller lease TTL will reap the zombie', request.snapshot_id,
            request.job_id)
        return False


async def _poll_worker_status(pool_worker_uid: str) -> Optional[WorkerStatus]:
    """Return the controller's current health for ``pool_worker_uid`` (or
    ``None`` if the controller no longer knows the worker — treated as gone).
    Read-only; uses the get_pool_stats probe filtered to the one worker."""
    stats = await workflow.execute_activity(
        get_pool_stats_activity,
        PoolStatsFilter(pool_worker_uids=[pool_worker_uid]),
        start_to_close_timeout=_POOL_STATS_START_TO_CLOSE,
        schedule_to_close_timeout=_POOL_STATS_SCHEDULE_TO_CLOSE,
        retry_policy=_POOL_STATS_RETRY_POLICY,
    )
    for worker in stats.durable_workers:
        if worker.pool_worker_uid == pool_worker_uid:
            return worker.health
    return None


async def run_pooled_activity(
    snapshot_id: SnapshotID,
    activity: SnapshotActivity,
    activity_fn: Callable,
    *,
    args: List[Any],
    schedule_to_close_timeout: timedelta,
    heartbeat_timeout: timedelta,
    retry_policy: Optional[RetryPolicy] = None,
    resource_hint: Optional[ResourceHint] = None,
    shard_index: Optional[int] = None,
    source_size_bytes: int = 0,
    max_attempts: int = 3,
) -> Any:
    """Place a snapshot, dispatch ``activity_fn`` on the placed worker, and —
    on failure — replace an exhausted activity-scoped pod, or poll a
    long-lived worker's health to decide whether to wait, retry, or re-place.

    This is the node-death-resilient placement-and-dispatch path. It is
    Temporal workflow code and therefore strictly
    deterministic: it uses ``workflow.execute_activity`` / ``workflow.sleep``
    only (never wall-clock or ``asyncio.sleep``).

    Each of ``max_attempts`` iterations dispatches ``activity_fn`` once.
    ``schedule_to_close_timeout`` applies independently to each dispatch, so
    the coordinator's theoretical activity-execution bound is
    ``max_attempts * schedule_to_close_timeout``:

    1. (if no live placement) place via ``place_snapshot_activity`` →
       ``task_queue`` + the placed ``pool_worker_uid``. The place retry policy
       absorbs a transient STALE home; a HomeStaleError that escapes it
       surfaces here and is retried as a fresh placement on the next iteration.
    2. dispatch ``activity_fn`` on ``task_queue`` with a centralized
       ``schedule_to_start_timeout`` of 30 min.
    3. on success → release the lease, return the activity result.
    4. on ``ActivityError`` → if it is a cancellation (the user cancelled
       the job), re-raise immediately so cancellation finishes instead of
       re-dispatching a second train run on the still-healthy worker
       (kumo#30191); on a snapshot miss or ScheduleToStart timeout, release
       and re-place. An activity-scoped Kubernetes pod exits after its one
       activity and its Job retries are exhausted before the error reaches
       this workflow, so release and re-place it even when the colocated
       daemon is healthy. Otherwise poll the placed worker's health:

       * STALE → ``workflow.sleep`` and re-poll, bounded to
         ``_STALE_WAIT_CEILING`` from STALE entry; if it recovers to READY,
         re-dispatch on the SAME placement (no re-place); if it ages to
         DEAD/gone, drop the placement so the next iteration re-places.
       * READY → the failure was not worker-death; re-dispatch on the same
         placement.
       * DEAD / DRAINING / gone (or an overflow eph placement) → drop the
         placement so the next iteration re-places on a fresh worker.

    The lease is always released (in ``finally``) when a placement is dropped
    or the runner exits, so a re-place never leaks the prior lease.
    """
    request = _build_placement_request(
        snapshot_id, activity, resource_hint, shard_index,
        source_size_bytes)

    last_error: Optional[BaseException] = None
    # Live placement carried across iterations: re-dispatch reuses it; a
    # DEAD/gone worker drops it (None) so the next iteration re-places.
    task_queue: Optional[str] = None
    pool_worker_uid: Optional[str] = None
    placed = False
    placement_count = 0
    try:
        for _attempt in range(max_attempts):
            if not placed:
                result = await _place_snapshot(request)
                placement_count += 1
                _stamp_placement_memo(result, activity, shard_index)
                task_queue = result.task_queue
                pool_worker_uid = result.pool_worker_uid
                placed = True
                workflow.logger.info(
                    'run_pooled_activity placed: snapshot=%s job=%s tq=%s '
                    'status=%s worker=%s', request.snapshot_id, request.job_id,
                    task_queue, result.status.value, pool_worker_uid)
            try:
                return await workflow.execute_activity(
                    activity_fn,
                    args=args,
                    task_queue=task_queue,
                    schedule_to_close_timeout=schedule_to_close_timeout,
                    schedule_to_start_timeout=(
                        _POOLED_ACTIVITY_SCHEDULE_TO_START),
                    heartbeat_timeout=heartbeat_timeout,
                    retry_policy=retry_policy,
                )
            except ActivityError as err:
                last_error = err
                if is_cancelled_exception(err):
                    # A cancelled job is not a worker failure; re-dispatching
                    # would start a second GPU-consuming train run after the
                    # API reported CANCELLED (kumo#30191). Re-raise; the
                    # ``finally`` still frees the lease.
                    workflow.logger.info(
                        'run_pooled_activity cancelled; releasing placement '
                        'without re-dispatch (snapshot=%s job=%s).',
                        request.snapshot_id, request.job_id)
                    raise
                if _is_permanent_snapshot_miss(err):
                    workflow.logger.warning(
                        'run_pooled_activity snapshot miss on placed worker; '
                        're-placing (snapshot=%s job=%s).', request.snapshot_id,
                        request.job_id)
                elif _is_schedule_to_start_timeout(err):
                    # Daemon health cannot detect an activity poller lost after
                    # placement, so delete this Job before replacing it.
                    workflow.logger.warning(
                        'run_pooled_activity lost activity poller for tq=%s; '
                        're-placing (snapshot=%s job=%s).', task_queue,
                        request.snapshot_id, request.job_id)
                elif (task_queue is not None
                      and task_queue.startswith(ACTIVITY_TQ_PREFIX)):
                    # The activity-scoped worker deliberately exits after one
                    # terminal activity. Kubernetes has already exhausted the
                    # Job's pod retries by the time Temporal surfaces this
                    # ActivityError. The durable daemon can still report
                    # READY, but it does not poll this activity queue; retrying
                    # the same queue would therefore wait for the 30-minute
                    # ScheduleToStart timeout.
                    workflow.logger.warning(
                        'run_pooled_activity activity-scoped worker exhausted '
                        'for tq=%s; re-placing (snapshot=%s job=%s).',
                        task_queue, request.snapshot_id, request.job_id)
                elif pool_worker_uid is None:
                    # Overflow (eph) placement — no durable worker to poll.
                    # The eph pod is throwaway; drop + re-place from scratch.
                    workflow.logger.warning(
                        'run_pooled_activity overflow activity failed for '
                        'snapshot=%s job=%s; re-placing. err=%s',
                        request.snapshot_id, request.job_id, err)
                else:
                    decision = await _await_worker_recovery(pool_worker_uid)
                    if decision == 'retry':
                        workflow.logger.info(
                            'run_pooled_activity worker %s healthy after '
                            'failure; re-dispatching on same placement '
                            '(snapshot=%s job=%s).', pool_worker_uid,
                            request.snapshot_id, request.job_id)
                        continue
                    workflow.logger.warning(
                        'run_pooled_activity worker %s is DEAD/gone after '
                        'failure; re-placing (snapshot=%s job=%s).',
                        pool_worker_uid, request.snapshot_id, request.job_id)

                if not await _release_snapshot(request):
                    break
                placed = False
    finally:
        # Release the final live placement (the success path returns from
        # inside the loop, so its release runs here; a dropped placement was
        # already released above and left placed=False).
        if placed:
            await _release_snapshot(request)

    assert last_error is not None
    if _is_schedule_to_start_timeout(last_error):
        raise ApplicationError(
            f'DiskGraph compute activity for snapshot={request.snapshot_id} '
            f'job={request.job_id} lost its poller after {placement_count} '
            f'placement{"s" if placement_count != 1 else ""}',
            PublicException(
                message=_POOLED_ACTIVITY_NO_POLLER_MESSAGE).to_dict(),
            type='PooledActivityNoPollerError',
            non_retryable=True,
        ) from last_error
    raise last_error


async def _await_worker_recovery(pool_worker_uid: str) -> str:
    """Poll ``pool_worker_uid`` health and return the runner's next action:

    * ``'retry'`` — worker is READY (recovered or never died); re-dispatch on
      the same placement.
    * ``'replace'`` — worker is DEAD / DRAINING / gone, or stayed STALE past
      ``_STALE_WAIT_CEILING``; re-place on a fresh worker.

    While STALE, sleeps ``_STALE_REPOLL_INTERVAL`` and re-polls, bounded to
    ``_STALE_WAIT_CEILING`` from the first STALE observation. Deterministic:
    ``workflow.sleep`` only.
    """
    waited = timedelta()
    while True:
        status = await _poll_worker_status(pool_worker_uid)
        if status == WorkerStatus.READY:
            return 'retry'
        if status != WorkerStatus.STALE:
            # DEAD, DRAINING, or gone (None) — re-place on a fresh worker.
            return 'replace'
        if waited >= _STALE_WAIT_CEILING:
            return 'replace'
        await workflow.sleep(_STALE_REPOLL_INTERVAL)
        waited += _STALE_REPOLL_INTERVAL


@asynccontextmanager
async def snapshot_placement_session_ctx(
    snapshot_id: SnapshotID,
    activity: SnapshotActivity,
    resource_hint: ResourceHint | None = None,
    shard_index: int | None = None,
    source_size_bytes: int = 0,
) -> AsyncIterator[PlacementResult]:
    """Acquire a pool snapshot lease for the ``async with`` block and
    release it on exit; yields the complete placement so callers that need
    capacity metadata can inspect it.
    """
    request = _build_placement_request(
        snapshot_id, activity, resource_hint, shard_index,
        source_size_bytes)
    # place_snapshot is long-running on cold-start (controller waits for
    # K8s daemon registration). start_to_close must accommodate the
    # AutoRPC rpc_timeout configured in pool.client; schedule_to_close
    # is the retry budget. See :mod:`pool.timeouts`.
    result: PlacementResult = await _place_snapshot(request)
    workflow.logger.info(
        'snapshot_placement_session_ctx acquired: '
        'snapshot=%s job=%s tq=%s status=%s',
        request.snapshot_id, request.job_id, result.task_queue,
        result.status.value)
    # Stamp strategy-provided observability metadata (pod_name,
    # grafana_logs_url, ...) onto the workflow memo so anyone landing
    # on the Temporal Cloud URL can jump to the pod's logs. Keyed by
    # activity + shard_index so each parallel branch of a sharded
    # batch-prediction workflow records its own pod without
    # overwriting siblings; non-sharded callers fall through to a
    # plain ``pool_<ACTIVITY>`` key. Serialize to JSON string so the
    # Temporal Cloud UI shows it as readable text — KumoPayloadConverter
    # would otherwise encode the dict as a pydantic-class protobuf blob
    # that the UI can't decode.
    _stamp_placement_memo(result, activity, shard_index)
    try:
        yield result
    finally:
        await _release_snapshot(request)
