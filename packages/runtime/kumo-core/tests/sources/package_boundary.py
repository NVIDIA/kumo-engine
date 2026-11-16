# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a connector distribution is allowed to import, checked once for all.

A connector wheel is installed into a customer's own environment, so every
module it reaches is a dependency that customer never asked to resolve: the
platform it was cut out of, another vendor's driver, or a training stack.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

PLATFORM_ROOTS = frozenset({
    "fastapi",
    "io_utils",
    "kubernetes",
    "kumo",
    # Both workflow hosts are forbidden for the same reason the platform is,
    # and the direction is what makes it worth saying: a connector
    # distribution is *loaded by* kumoai-studio or kumo-engine as a plugin,
    # so importing either host back would make the dependency circular and
    # would put a workflow API into the platform's install. Contracts a plugin
    # does implement against live in `kumoapi.plugin`. kumo-engine's own
    # `kumo.engine` submodule is already covered by the "kumo" root above.
    "kumoai_studio",
    "pool",
    "rest",
    "service_framework",
    "sqlalchemy",
    "sqlmodel",
    "statedb",
    "temporalio",
    "temporalio_commons",
})
VENDOR_ROOTS = frozenset(
    {"azure", "boto3", "botocore", "databricks", "google", "snowflake"})
PROVIDER_ROOTS = frozenset({
    "kumo_bigquery",
    "kumo_databricks",
    "kumo_gcs",
    "kumo_adls",
    "kumo_s3",
    "kumo_snowflake",
})


def assert_imports_nothing_foreign(source_root: Path, *,
                                   vendor: str | tuple[str, ...]) -> None:
    owned = {vendor} if isinstance(vendor, str) else set(vendor)
    forbidden = (PLATFORM_ROOTS | (VENDOR_ROOTS - owned) |
                 (PROVIDER_ROOTS - {source_root.name}))
    violations = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".", 1)[0] in forbidden:
                    violations.append(f"{path.name}:{node.lineno}: {name}")
    assert violations == [], "\n".join(violations)


def assert_top_level_import_is_cheap(
        package_root: Path,
        package: str,
        reader: str,
        *,
        unloaded: tuple[str, ...] = (),
) -> None:
    """Run out of process against a bare interpreter, because anything already
    imported by the test session would hide the very import being refused."""
    checks = "\n".join(f"assert {name!r} not in sys.modules"
                       for name in ("torch", "torch_geometric") + unloaded)
    script = (f"import sys\n"
              f"sys.path[:0] = sys.argv[1:]\n"
              f"import {package}\n"
              f"assert {package}.{reader} is not None\n"
              f"{checks}\n")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(package_root / "src"),
            str(package_root.parent / "kumo-core" / "src"),
            str(package_root.parent / "kumo-api"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
