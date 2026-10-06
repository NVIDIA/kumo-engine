#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Literal, cast

Phase = Literal[
    "black-box",
    "sdk-sqlite",
    "sdk-postgres",
    "unit-static",
    "playwright",
]


@dataclass(frozen=True)
class GateStep:
    phase: Phase
    name: str
    command: list[str]
    env: dict[str, str]
    note: str
    completed_gate: bool = False


def _env_value(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _pytest_xdist_available() -> bool:
    return importlib.util.find_spec("xdist") is not None


def _black_box_command() -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
    ]
    workers = os.environ.get("KUMO_FINAL_GATE_BLACK_BOX_WORKERS")
    if workers is not None:
        command.extend(["-n", workers])
    elif _pytest_xdist_available():
        command.extend(["-n", "auto"])
    command.append("test/rest/integ/black_box")
    return command


def _black_box_note() -> str:
    if "KUMO_FINAL_GATE_BLACK_BOX_WORKERS" in os.environ:
        worker_note = (
            "the pytest worker count is explicitly controlled by "
            "KUMO_FINAL_GATE_BLACK_BOX_WORKERS."
        )
    elif _pytest_xdist_available():
        worker_note = "pytest-xdist is installed, so this defaults to -n auto."
    else:
        worker_note = (
            "pytest-xdist is not installed, so this defaults to serial pytest; "
            "set KUMO_FINAL_GATE_BLACK_BOX_WORKERS after installing xdist to "
            "force parallel execution."
        )
    return (
        "First gate. Requires an already running suitable test stack; "
        f"{worker_note}"
    )


def _dataset_version(storage_mode: str) -> str:
    return _env_value(
        f"KUMO_FINAL_GATE_{storage_mode.upper()}_DATASET_VERSION",
        _env_value("KUMO_FINAL_GATE_DATASET_VERSION", "<dataset-version>"),
    )


def _sdk_step(storage_mode: Literal["sqlite", "postgres"]) -> GateStep:
    upper = storage_mode.upper()
    stack = _env_value(f"KUMO_FINAL_GATE_{upper}_STACK", storage_mode)
    url = _env_value(
        f"KUMO_FINAL_GATE_{upper}_URL",
        f"http://localhost:<{storage_mode}-port>/api",
    )
    api_key = _env_value(f"KUMO_FINAL_GATE_{upper}_API_KEY", "<api-key>")
    storage_root = _env_value(
        f"KUMO_FINAL_GATE_{upper}_STORAGE_ROOT",
        f"<{storage_mode}-storage-root>",
    )
    output_root = _env_value(
        f"KUMO_FINAL_GATE_{upper}_OUTPUT_ROOT",
        f"<{storage_mode}-output-root>",
    )
    local_output = _env_truthy(f"KUMO_FINAL_GATE_{upper}_LOCAL_OUTPUT")
    training_job_id = os.environ.get(f"KUMO_FINAL_GATE_{upper}_TRAINING_JOB_ID")
    postgres_mode = storage_mode if storage_mode == "sqlite" else "bundled"
    command = [
        sys.executable,
        "-m",
        "tools.e2e.harnesses.success.customer_churn_binary_implicit",
        "--url",
        url,
        "--api-key",
        api_key,
        "--dataset-version",
        _dataset_version(storage_mode),
        "--storage-root",
        storage_root,
        "--variant",
        _env_value(f"KUMO_FINAL_GATE_{upper}_VARIANT", "parquet_typed"),
        "--stop-after",
        "bp",
    ]
    if not local_output:
        command.extend(["--output-root", output_root])
    if training_job_id:
        command.extend(["--training-job-id", training_job_id])
    if local_output:
        command.append("--local-output")
    return GateStep(
        phase=cast(Phase, f"sdk-{storage_mode}"),
        name=f"{storage_mode} SDK API-key train and prediction e2e",
        command=command,
        env={
            "KUMO_POSTGRES_MODE": _env_value(
                f"KUMO_FINAL_GATE_{upper}_POSTGRES_MODE",
                postgres_mode,
            ),
            "STACK": stack,
        },
        note=(
            "Uses the SDK-worker-owned churn harness; run only when that "
            "worker has not already supplied an equivalent completed log."
        ),
        completed_gate=True,
    )


def gate_steps() -> list[GateStep]:
    return [
        GateStep(
            phase="black-box",
            name="black-box REST flows",
            command=_black_box_command(),
            env={},
            note=_black_box_note(),
            completed_gate=True,
        ),
        _sdk_step("sqlite"),
        _sdk_step("postgres"),
        GateStep(
            phase="unit-static",
            name="migrated StateDB impacted unit tests",
            command=[
                sys.executable,
                "-m",
                "pytest",
                "test/statedb",
                "test/rest/public/test_public_api_persist_dir.py",
                "test/rest/public/test_public_api_projects.py",
                "test/rest/public/test_public_api_groups.py",
                "test/rest/public/test_public_api_data_source.py",
                "test/rest/public/test_public_api_graph.py",
                "test/rest/public/test_public_gen_train_table_router.py",
                "test/tools/sdk/test_demo_jobs_to_sdk.py",
                "tools/tests/test_db_metadata_cutover.py",
            ],
            env={},
            note=(
                "Represents the migrated Phase 1/StateDB unit coverage. "
                "Postgres-backed StateDB tests use STATEDB_POSTGRES_DSN or "
                "start a temporary postgres:16-alpine container."
            ),
            completed_gate=True,
        ),
        GateStep(
            phase="unit-static",
            name="migrated StateDB static checks",
            command=[
                "pants",
                "lint",
                "statedb::",
                "rest/public::",
                "tools::",
                "test/statedb::",
                "test/rest/public::",
                "test/tools::",
            ],
            env={},
            note=(
                "Static check lane for the migrated API/StateDB surface. "
                "Narrow further only when a worker owns a smaller impacted "
                "set and records it in handoff."
            ),
            completed_gate=True,
        ),
        GateStep(
            phase="playwright",
            name="UI Playwright local stack",
            command=[
                "bash",
                "scripts/run_playwright_local_stack.sh",
                "--reuse-stack",
                "--skip-build",
            ],
            env={},
            note=(
                "Last gate by policy. Drop --reuse-stack/--skip-build when "
                "the final runner owns stack startup from scratch."
            ),
            completed_gate=True,
        ),
    ]


def _selected_steps(phases: set[Phase] | None) -> list[GateStep]:
    steps = gate_steps()
    if phases is None:
        return steps
    return [step for step in steps if step.phase in phases]


def _shell_join(step: GateStep) -> str:
    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(step.env.items())
    )
    command = " ".join(shlex.quote(part) for part in step.command)
    if env_prefix:
        return f"{env_prefix} {command}"
    return command


def _print_text(steps: list[GateStep]) -> None:
    for index, step in enumerate(steps, start=1):
        print(f"{index}. [{step.phase}] {step.name}")
        print(f"   command: {_shell_join(step)}")
        print(f"   note: {step.note}")


def _run_steps(steps: list[GateStep]) -> int:
    for index, step in enumerate(steps, start=1):
        if any(part.startswith("<") and part.endswith(">")
               for part in step.command):
            raise RuntimeError(
                f"{step.name} still has placeholder arguments. Print the "
                "manifest and set the KUMO_FINAL_GATE_* environment overrides "
                "before executing this phase."
            )
        print(f"==> [{index}/{len(steps)}] {step.name}", flush=True)
        env = os.environ.copy()
        env.update(step.env)
        subprocess.run(step.command, env=env, check=True)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print or run the StateDB final-gate order. The default is a "
            "dry-run manifest so long e2e work is not duplicated accidentally."
        )
    )
    parser.add_argument(
        "--phase",
        action="append",
        choices=[
            "black-box",
            "sdk-sqlite",
            "sdk-postgres",
            "unit-static",
            "playwright",
        ],
        help="Limit output/execution to a phase. May be repeated.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "shell"],
        default="text",
        help="Manifest output format for dry-run mode.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the selected gates instead of printing the manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    phases = set(args.phase) if args.phase else None
    steps = _selected_steps(phases)

    if args.execute:
        return _run_steps(steps)
    if args.format == "json":
        print(json.dumps([asdict(step) for step in steps], indent=2))
    elif args.format == "shell":
        for step in steps:
            print(_shell_join(step))
    else:
        _print_text(steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
