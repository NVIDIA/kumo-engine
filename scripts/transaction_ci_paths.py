# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detect Blueprint changes without skipping the required CI status."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

POLICY_PATHS = frozenset({
    ".github/workflows/transaction-blueprint.yml",
    ".github/workflows/linting.yml",
    "scripts/transaction_ci_paths.py",
    "scripts/check_license_headers.py",
    "scripts/license_header_policy.json",
    "scripts/check_python_license_metadata.py",
    "setup.cfg",
    "test/scripts/test_transaction_ci_paths.py",
    "test/scripts/test_check_license_headers.py",
    "test/scripts/test_python_license_metadata.py",
})


def package_changed(root: Path, base: str, head: str) -> bool:
    for revision in (base, head):
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("expected full Git commit SHA")
    subprocess.run(["git", "cat-file", "-e", f"{head}^{{commit}}"], cwd=root,
                   check=True, capture_output=True)
    if base == "0" * 40:
        return True
    paths = subprocess.check_output([
        "git",
        "diff",
        "--no-renames",
        "--name-only",
        "-z",
        f"{base}...{head}",
        "--",
    ], cwd=root).decode().split("\0")
    return any(
        path.startswith("packages/experimental/kumo-transaction/")
        or path in POLICY_PATHS for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    args = parser.parse_args()
    changed = package_changed(Path.cwd(), args.base, args.head)
    print(f"changed={str(changed).lower()}")


if __name__ == "__main__":
    main()
