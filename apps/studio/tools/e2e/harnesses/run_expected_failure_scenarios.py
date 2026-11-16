#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import subprocess
import sys

from tools.e2e.lib.config import default_storage_root

EXPECTED_FAILURE_SCENARIOS = [
    "tools.e2e.harnesses.expected_failure.invalid_task_regression_on_categorical",
    "tools.e2e.harnesses.expected_failure.invalid_schema_missing_destination",
    "tools.e2e.harnesses.expected_failure.invalid_data_type_struct_feature",
    "tools.e2e.harnesses.expected_failure.invalid_distribution_no_null_implicit",
]


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run SDK scenarios that should fail cleanly.")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--storage-root", default=default_storage_root())
    parser.add_argument("--variant", default="parquet_typed")
    parser.add_argument("--stop-after", default="validate")
    return parser.parse_known_args()


def main() -> int:
    args, extra = _parse_args()
    for module in EXPECTED_FAILURE_SCENARIOS:
        cmd = [
            sys.executable,
            "-m",
            module,
            "--dataset-version",
            args.dataset_version,
            "--storage-root",
            args.storage_root,
            "--variant",
            args.variant,
            "--stop-after",
            args.stop_after,
            *extra,
        ]
        print(f"[sdk-expected-failure] running {module}", flush=True)
        completed = subprocess.run(cmd, check=False)
        if completed.returncode == 0:
            raise RuntimeError(f"Expected {module} to fail, but it succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
