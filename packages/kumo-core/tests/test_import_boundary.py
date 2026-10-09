# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_IMPORT_ROOTS = frozenset({
    "databricks",
    "fastapi",
    "io_utils",
    "kubernetes",
    "kumo",
    "pool",
    "rest",
    "service_framework",
    "snowflake",
    "sqlalchemy",
    "sqlmodel",
    "statedb",
    "temporalio",
    "temporalio_commons",
})


def test_core_source_has_no_platform_imports() -> None:
    source_root = Path(__file__).parents[1] / "src" / "kumo_core"
    violations: list[str] = []

    for source_file in source_root.rglob("*.py"):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), source_file)
        for node in ast.walk(tree):
            imported_modules: list[str]
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules = [node.module]
            else:
                continue
            for module in imported_modules:
                if module.split(".", maxsplit=1)[0] in FORBIDDEN_IMPORT_ROOTS:
                    violations.append(
                        f"{source_file.relative_to(source_root)}:{node.lineno}: {module}"
                    )

    assert violations == []
