# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
from pathlib import Path


FORBIDDEN_IMPORT_ROOTS = {
    "kumo",
    "kumo" + "ml",
    "rest",
    "rfm_rest",
    "temporalio_commons",
}


def test_kumo_core_has_no_monolith_or_legacy_imports() -> None:
    source_root = Path(__file__).parents[1] / "src" / "kumo_core"
    violations: list[str] = []

    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.partition(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                roots = {(node.module or "").partition(".")[0]}
            else:
                continue
            forbidden = roots & FORBIDDEN_IMPORT_ROOTS
            if forbidden:
                violations.append(
                    f"{path.relative_to(source_root)}:{node.lineno}: "
                    f"{', '.join(sorted(forbidden))}"
                )

    assert not violations, "Forbidden package-boundary imports:\n" + "\n".join(
        violations
    )
