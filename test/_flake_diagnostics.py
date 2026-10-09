# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-failure diagnostics for flaky tests.

Activated from test/conftest.py's pytest_runtest_makereport hook on
any test failure (including pytest-timeout hits).

What we capture:
  1. Subprocess stacks for every descendant pid of the pytest process,
     via py-spy. The pytest-timeout faulthandler dump only covers the
     main pytest process; the interesting hangs tend to wedge a spawn-
     worker subprocess whose stacks we otherwise never see.
  2. For Temporal-using tests (those with a temporal_client fixture):
     full Temporal state — per-workflow summary plus the complete event
     history for every workflow in the test namespace at the moment of
     failure. This is what tells us which activity was in-flight, which
     signals were in the queue, and where cancellation landed.

All output goes to $ARTIFACTS_DIR/<nodeid>/. The CI workflow uploads
that directory on failure.

This module is best-effort — anything we do here is wrapped so that
collection failures don't mask or replace the original test error.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Optional

# Imported lazily inside the helpers so this module remains importable
# in environments that don't have every optional dep installed.


def _sanitize_path_component(s: str) -> str:
    """Make a string safe for use as a filesystem path component.

    GitHub Actions' upload-artifact rejects paths containing any of
    " : < > | * ? \\r \\n (even on Linux runners, to keep archives
    round-trippable on Windows). Temporal workflow IDs routinely
    contain colons and @, so we have to strip them before using them
    as filenames.
    """
    disallowed = '":<>|*?\r\n/\\'
    return "".join("_" if c in disallowed else c for c in s)


def _artifacts_dir_for(nodeid: str) -> pathlib.Path:
    root = pathlib.Path(
        os.environ.get("ARTIFACTS_DIR", "/tmp/kumo-test-artifacts"))
    safe = _sanitize_path_component(
        nodeid.replace("::", "__").replace("[", "_").replace("]", "_"))
    out = root / safe
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_on_failure(nodeid: str, temporal_client: Optional[Any]) -> None:
    """Run every diagnostic dump. Never raises.

    Called from the conftest pytest_runtest_makereport hook on a failed
    test. The temporal_client argument is the live fixture from the
    failed test (or None if the test didn't request one).
    """
    try:
        out = _artifacts_dir_for(nodeid)
    except Exception as e:
        print(f"[flake-diag] could not create artifact dir: {e!r}",
              file=sys.stderr)
        return

    print(f"[flake-diag] writing diagnostics to {out}", file=sys.stderr)

    # 1. py-spy stacks first: cheapest, most informative for a hang, and
    #    we want them before any teardown disturbs the subprocess tree.
    with contextlib.suppress(Exception):
        _dump_subprocess_stacks(out)

    # 2. Temporal workflow state — async, so bounded by a wall clock.
    if temporal_client is not None:
        with contextlib.suppress(Exception):
            _dump_temporal_state_sync(out, temporal_client)


def _dump_subprocess_stacks(out: pathlib.Path) -> None:
    pyspy = shutil.which("py-spy")
    if not pyspy:
        (out / "py-spy-NOT-INSTALLED.txt").write_text(
            "py-spy not on PATH; skipping subprocess stack dumps.\n"
            "Install with: uv pip install py-spy\n")
        return

    try:
        import psutil
    except ImportError:
        (out / "psutil-NOT-INSTALLED.txt").write_text(
            "psutil not available; skipping subprocess stack dumps.\n")
        return

    stacks_dir = out / "stacks"
    stacks_dir.mkdir(exist_ok=True)

    pytest_pid = os.getpid()
    try:
        me = psutil.Process(pytest_pid)
        targets = [me] + me.children(recursive=True)
    except psutil.Error as e:
        (stacks_dir / "psutil-error.txt").write_text(repr(e) + "\n")
        return

    index_lines = [f"pytest_pid={pytest_pid}", ""]
    for p in targets:
        try:
            name = p.name()
            pid = p.pid
            cmdline = " ".join(p.cmdline())
        except psutil.Error:
            continue
        index_lines.append(f"pid={pid} name={name} cmd={cmdline}")

        out_file = stacks_dir / f"pid{pid}_{_sanitize_path_component(name)}.txt"
        try:
            # py-spy dump on a healthy process returns in <1s; on a deeply
            # wedged process it might block on ptrace. Cap wall time.
            res = subprocess.run(
                [pyspy, "dump", "--pid", str(pid)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            out_file.write_text(
                f"pid={pid} name={name}\n"
                f"cmd={cmdline}\n"
                f"py-spy exit={res.returncode}\n"
                f"----- stdout -----\n{res.stdout}\n"
                f"----- stderr -----\n{res.stderr}\n")
        except subprocess.TimeoutExpired:
            out_file.write_text(
                f"pid={pid} name={name}\npy-spy TIMED OUT after 30s\n")
        except Exception as e:
            out_file.write_text(
                f"pid={pid} name={name}\npy-spy FAILED: {e!r}\n")

    (stacks_dir / "INDEX.txt").write_text("\n".join(index_lines) + "\n")


def _dump_temporal_state_sync(out: pathlib.Path, tc: Any) -> None:
    """Run _dump_temporal_state in a fresh event loop with a wall-clock cap."""
    # We may be called from a sync pytest hook running on the main thread
    # that just had its test-level event loop closed. A fresh loop is safe.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            asyncio.wait_for(_dump_temporal_state(out, tc), timeout=90))
    except asyncio.TimeoutError:
        (out / "temporal-DUMP-TIMED-OUT.txt").write_text(
            "Temporal dump exceeded 90s wall clock; giving up.\n")
    except Exception as e:
        (out / "temporal-DUMP-ERROR.txt").write_text(
            f"Temporal dump raised: {e!r}\n")
    finally:
        try:
            loop.close()
        except Exception:
            pass


async def _dump_temporal_state(out: pathlib.Path, tc: Any) -> None:
    from google.protobuf.json_format import MessageToDict

    client = await tc.get()
    namespace = tc.config.temporal_namespace

    wf_dir = out / "workflows"
    wf_dir.mkdir(exist_ok=True)

    summaries = []
    count = 0
    # list_workflows enumerates ALL workflows in the namespace regardless
    # of status — that's what we want at failure time.
    async for desc in client.list_workflows():
        count += 1
        summary = {
            "workflow_id":
            desc.id,
            "run_id":
            desc.run_id,
            "workflow_type":
            getattr(desc, "workflow_type", None),
            "status":
            str(desc.status) if desc.status is not None else None,
            "task_queue":
            getattr(desc, "task_queue", None),
            "start_time":
            desc.start_time.isoformat() if desc.start_time else None,
            "close_time":
            desc.close_time.isoformat() if desc.close_time else None,
            "execution_time":
            desc.execution_time.isoformat() if desc.execution_time else None,
        }
        summaries.append(summary)

        # Full history: the gold. Every activity schedule/start/complete,
        # every signal, every cancel — all here.
        safe_id = _sanitize_path_component(desc.id)
        safe_run = _sanitize_path_component(desc.run_id or "")
        try:
            handle = client.get_workflow_handle(desc.id, run_id=desc.run_id)
            events = []
            async for event in handle.fetch_history_events():
                events.append(MessageToDict(event))
            (wf_dir / f"{safe_id}__{safe_run}.history.json").write_text(
                json.dumps(events, indent=2, default=str))
        except Exception as e:
            (wf_dir /
             f"{safe_id}__{safe_run}.history-ERROR.txt").write_text(
                 f"Failed to fetch history: {e!r}\n")

    (out / "workflow_summaries.json").write_text(
        json.dumps(
            {
                "namespace": namespace,
                "count": count,
                "workflows": summaries,
            },
            indent=2))


async def count_non_terminal_workflows(tc: Any) -> int:
    """How many workflows on this test's task queues are still non-terminal.

    We match the filter convention of temporalio_commons.terminate_open_workflows:
    only consider workflows whose task queue ends with the test's configured
    suffix, so we don't count persistent/auxiliary workflows on task queues
    that belong to other test sessions or to the Temporal server itself.
    """
    from temporalio.api.enums.v1 import WorkflowExecutionStatus

    client = await tc.get()
    terminal = {
        WorkflowExecutionStatus.WORKFLOW_EXECUTION_STATUS_COMPLETED,
        WorkflowExecutionStatus.WORKFLOW_EXECUTION_STATUS_FAILED,
        WorkflowExecutionStatus.WORKFLOW_EXECUTION_STATUS_CANCELED,
        WorkflowExecutionStatus.WORKFLOW_EXECUTION_STATUS_TERMINATED,
        WorkflowExecutionStatus.WORKFLOW_EXECUTION_STATUS_TIMED_OUT,
    }
    task_queue_suffix = tc.config.task_queue("") or ""
    count = 0
    async for desc in client.list_workflows():
        if desc.status in terminal:
            continue
        tq = getattr(desc, "task_queue", None) or ""
        if task_queue_suffix and not tq.endswith(task_queue_suffix):
            continue
        count += 1
    return count
