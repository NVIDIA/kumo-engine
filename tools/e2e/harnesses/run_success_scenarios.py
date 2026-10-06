#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import subprocess
import sys

from tools.e2e.lib.config import default_storage_root

SUCCESS_SCENARIOS = [
    "tools.e2e.harnesses.success.customer_churn_binary_implicit",
    "tools.e2e.harnesses.success.customer_loyalty_multiclass_explicit",
    "tools.e2e.harnesses.success.customer_spend_regression",
    "tools.e2e.harnesses.success.marketplace_static_link_prediction",
    "tools.e2e.harnesses.success.account_next_balance_regression",
    "tools.e2e.harnesses.success.reserved_name_output_collision",
]


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run SDK success scenarios.")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--storage-root", default=default_storage_root())
    parser.add_argument("--variant", default="parquet_typed")
    parser.add_argument("--stop-after", default="validate")
    return parser.parse_known_args()


def main() -> int:
    args, extra = _parse_args()
    for module in SUCCESS_SCENARIOS:
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
        print(f"[sdk-success] running {module}", flush=True)
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
