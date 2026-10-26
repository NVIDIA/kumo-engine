# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate Cargo workspace license declarations against repository policy."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__:
    from .check_license_headers import load_policy, matching_exclusions
else:
    from check_license_headers import load_policy, matching_exclusions


def validate_workspace(root: Path, metadata: dict[str, Any],
                       policy: dict[str, Any]) -> list[str]:
    errors = []
    members = set(metadata["workspace_members"])
    for package in metadata["packages"]:
        if package["id"] not in members:
            continue
        manifest = Path(package["manifest_path"]).resolve().relative_to(
            root).as_posix()
        expected = policy["license_identifier"]
        for exclusion in matching_exclusions(manifest, policy):
            if exclusion["kind"] == "vendored":
                expected = exclusion["license"]
                break
        if package["license"] != expected:
            errors.append(
                f"{manifest}: expected license {expected!r}, found {package['license']!r}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",
                        type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest-path", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        policy = load_policy(root / "scripts/license_header_policy.json")
        result = subprocess.run(
            [
                "cargo", "metadata", "--locked", "--offline", "--format-version",
                "1", "--no-deps", "--manifest-path",
                str(root / args.manifest_path)
            ],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        errors = validate_workspace(root, json.loads(result.stdout), policy)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Cargo license metadata check failed: {error}", file=sys.stderr)
        return 1
    if errors:
        print("Cargo license metadata check failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print("Cargo workspace license declarations match repository policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
