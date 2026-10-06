# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused StateDB pytest runner for agent validation loops.

Usage:
    .venv/bin/python -m tools.statedb_pytest \
        test/statedb/db/test_runtime.py::test_state_db_sqlite_write_and_read_tx

Run this from the repo root, outside restricted sandboxes. The runner is for
narrow StateDB slices only; it adds a pytest timeout so fixture/test hangs fail
non-zero, then exits with os._exit after pytest returns so interpreter shutdown
threads cannot leave a completed run stuck.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import signal
import sys
from pathlib import Path

DEFAULT_SMOKE_TEST = (
    "test/statedb/db/test_runtime.py::test_state_db_sqlite_write_and_read_tx"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run focused StateDB pytest selectors with hard-exit behavior.",
        epilog=(
            "Examples:\n"
            "  .venv/bin/python -m tools.statedb_pytest\n"
            "  .venv/bin/python -m tools.statedb_pytest "
            "test/statedb/db/test_schema_bootstrap.py::"
            "test_bootstrap_sqlite_creates_schema_for_empty_target\n"
            "  .venv/bin/python -m tools.statedb_pytest -- -k sqlite test/statedb\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--include-parent-conftest",
        action="store_true",
        help="load parent conftests such as test/conftest.py",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="per-test pytest-timeout value; use 0 to disable",
    )
    parser.add_argument(
        "--wall-timeout-seconds",
        type=int,
        default=120,
        help="whole-process timeout that dumps stacks and exits 124; use 0 to disable",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="pytest selectors/options; prefix with -- when passing pytest flags",
    )
    args = parser.parse_args(argv)
    if args.pytest_args[:1] == ["--"]:
        args.pytest_args = args.pytest_args[1:]
    return args


def _set_env_defaults() -> None:
    repo_root = Path.cwd()
    os.environ.setdefault("MPLCONFIGDIR", str(repo_root / ".tmp" / "mpl"))
    os.environ.setdefault("KUMO_PYTEST_ENABLE_FSUTIL_DEBUG_LOGS", "0")


def _install_wall_timeout(seconds: int) -> None:
    if seconds <= 0:
        return

    def _handle_timeout(_signum: int, _frame: object) -> None:
        print(
            f"\nStateDB pytest wall timeout after {seconds}s; dumping stacks.",
            file=sys.stderr,
            flush=True,
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(124)

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)


def _pytest_args(args: argparse.Namespace) -> list[str]:
    pytest_args: list[str] = []
    if not args.include_parent_conftest:
        pytest_args.append("--confcutdir=test/statedb")
    if args.timeout_seconds > 0:
        pytest_args.extend([
            f"--timeout={args.timeout_seconds}",
            "--timeout-method=thread",
        ])
    pytest_args.extend(args.pytest_args or [DEFAULT_SMOKE_TEST])
    return pytest_args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _set_env_defaults()
    _install_wall_timeout(args.wall_timeout_seconds)

    import pytest

    code = pytest.main(_pytest_args(args))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(code))


if __name__ == "__main__":
    main()
