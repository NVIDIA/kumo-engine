# Event-Loop Watchdog — Operator / Agent Guide

A daemon thread installed by the worker and REST entry points listed below detects when their asyncio event loop is blocked past a threshold and dumps diagnostics.

If you're reading this because you saw a `STALL DETECTED` log line — jump to **What to do when it fires**.

## What it detects

The asyncio event loop running on the main thread of a Kumo Python process — `temporalio_commons.worker_main`, `pool.worker_main`, `pool.controller_main`, `rest.main`, `rest.public.main` — being unresponsive for more than `EVENT_LOOP_WATCHDOG_STALL_THRESHOLD_SEC` (default **2.0 s**).

Typical causes:
- Sync I/O accidentally called from an `async def` body (a common cause of worker stalls).
- Long CPU-bound work on the event-loop thread without `await asyncio.to_thread(...)`.
- An imported library doing blocking work during a hot path.

## Signal in logs

Search the selected workload’s logs for:

```
event-loop watchdog: STALL DETECTED
```

A full match looks like:

```
ERROR async_utils.event_loop_watchdog: event-loop watchdog: STALL DETECTED — asyncio loop blocked 2.31s (threshold 2.00s). event_id=20260608T154532Z-a1b2c3d4 full_dump=/tmp/event-loop-stall-20260608T154532Z-a1b2c3d4.txt Deduped summary follows:
=== EVENT LOOP THREAD 'MainThread' (id=12345) ===
  File "...", line N, in foo
    ...
=== 17 other thread(s), 4 unique stack signature(s) ===
[12x] 'data-backend-worker-X' (id=...), 'data-backend-worker-Y' (id=...), 'data-backend-worker-Z' (id=...), +9 more
  File "...", line N, in queue.get
    ...
```

Two log lines per stall **episode**, not per check interval:
- One `ERROR ... STALL DETECTED ...` when the threshold is first exceeded.
- One `WARNING ... stall resolved (last measured gap X.XX s, ...)` when the loop starts ticking again.

## What to do when it fires

1. **Read the deduped summary inline in the log line.** The event-loop thread's stack is shown first (verbose); other threads are grouped by stack signature. The loop-thread frame is the smoking gun — it shows where the loop was when blocked.

2. **Retrieve the full dump file if you need it.** The log line includes `full_dump=/tmp/event-loop-stall-<event_id>.txt`. This file has every thread's stack individually (no dedup). To retrieve:

   ```bash
   POD=<pod-name-from-log-labels>
   NS=<namespace>
   EVENT_ID=<from-log-message>
   kubectl --context=<cluster> -n "$NS" exec "$POD" -- cat \
     /tmp/event-loop-stall-${EVENT_ID}.txt
   ```

   The watchdog does not restart the process. The default `/tmp` dump can be lost when the container is replaced or the file is removed, so retrieve it promptly. The original log line also contains the deduped stack summary.

3. **Identify the blocking call site.** In the EVENT LOOP THREAD section, walk upward from the most recent frame (bottom of the stack listing) and find the first frame inside kumo code that isn't a kumo framework wrapper. That's the call site to investigate.

4. **Cross-reference with the data plane.** If the worker also processes Temporal activities, check `temporal task-queue describe --task-queue <queue> --task-queue-type activity` from the same window — a drop to 0 pollers is evidence to correlate with the stall, not proof that the stall caused it. Use the Temporal endpoint and namespace for the selected deployment.

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `EVENT_LOOP_WATCHDOG_ENABLED` | `true` | Master kill switch. Set to `false` to disable the watchdog entirely (e.g., during a known-bad rollout). |
| `EVENT_LOOP_WATCHDOG_STALL_THRESHOLD_SEC` | `2.0` | How long the loop must be unresponsive before a stall is declared. Below 1s risks false positives from GC pauses; above 5s misses the value of firing before Temporal's 10 s SDK cliff. |
| `EVENT_LOOP_WATCHDOG_PROBE_INTERVAL_SEC` | `0.5` | How often the loop bumps the tick counter. Use a positive interval shorter than the threshold; the watchdog does not validate the ratio at startup. |
| `EVENT_LOOP_WATCHDOG_CHECK_INTERVAL_SEC` | `0.25` | How often the watchdog thread evaluates. Smaller narrows the detection edge window. |
| `EVENT_LOOP_WATCHDOG_DUMP_DIR` | `/tmp` | Where the full per-stall stack-dump file is written. Override to a host-mounted persistent volume if you want dumps to survive pod restart. Path must be a local filesystem path (no s3://). |

## Investigation pattern (for agents)

When asked to debug a worker freeze:

1. Read supplied logs, or use [check-logs](../.agents/skills/check-logs/SKILL.md)
   for the selected deployment and time range. Filter for
   `event-loop watchdog: STALL DETECTED`. A match confirms a detected event-loop
   stall; use its deduped summary to investigate the blocking call.

2. If you need richer thread context (e.g., to see what other threads were doing during the stall), retrieve the full dump file from the pod's `/tmp` per the snippet above. Time-bounded — pod restart loses the file.

3. After identifying the blocking call site in the EVENT LOOP THREAD frame, the fix pattern is:
   - Wrap it with `await asyncio.to_thread(...)` if it's truly blocking.
   - Use the `.coroutine()` accessor if the function is decorated with `@blocking_func` / `@blocking_method` and you're calling it as a regular function.
   - For Databricks / SQL connector calls in particular: confirm the call goes through an offloaded path.

## Related tooling

The watchdog is one layer of a broader detection stack:

| Tool | Scope | Threshold | Attribution |
|---|---|---|---|
| `@measure_blocking_code` / `BLOCKING_WARN_THRESHOLD_SEC` | Per-function (opt-in) | sub-second | Function name |
| **This watchdog** | Process-wide | multi-second | Stack trace |

They're complementary — use `@measure_blocking_code` for known-suspect code paths and this watchdog for the "unknown unknown" trigger that you didn't think to instrument.

## Source

- Module: `async_utils/event_loop_watchdog.py`
- Tests: `test/async_utils/test_event_loop_watchdog.py`
- Wired into: `temporalio_commons/worker_main.py`, `pool/worker_main.py`, `pool/controller_main.py`, `rest/app.py` (`startup_event`), `rest/public/app.py` (`startup_event`).
