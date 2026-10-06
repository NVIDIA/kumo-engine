# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A workflow may only take its clock from a replay-safe source.

``statedb.config.timestamp.now_utc`` returns ``workflow.now()`` inside a
workflow and the wall clock outside it. Every other ``now_utc`` in the tree is
the wall clock, so a workflow that imports one stamps a different value on
replay than it did on the original execution.

Nothing else catches this. The worker runs ``UnsandboxedWorkflowRunner``
(``temporalio_commons/worker_registry.py``), so the sandbox that would reject a
wall-clock call in workflow code is not in the loop.
"""
import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]

# Modules whose ``now_utc`` is the wall clock. Safe in an activity, wrong in a
# workflow.
WALL_CLOCK_SOURCES = {
    'kumo_core.utils.time',
    'common_utils.time',
}
SAFE_SOURCES = {
    'statedb.config.timestamp',
    'statedb.config.common',
}

SKIP_DIRS = {'.git', '.venv', 'node_modules', 'build', 'dist', '3rdparty'}

# Production workflows only. A test defines throwaway workflows that are never
# replayed, and some deliberately want the wall clock -- ``test_heartbeat_utils``
# measures elapsed real time.
SKIP_ROOTS = {'test', 'tests'}


def _python_files():
    for path in REPO.rglob('*.py'):
        parts = path.relative_to(REPO).parts
        if not SKIP_DIRS.isdisjoint(parts) or parts[0] in SKIP_ROOTS:
            continue
        yield path


def _defines_a_workflow(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        for decorator in getattr(node, 'decorator_list', []):
            # ``@workflow.defn`` and ``@workflow.defn(...)``
            target = (decorator.func
                      if isinstance(decorator, ast.Call) else decorator)
            if (isinstance(target, ast.Attribute) and target.attr == 'defn'
                    and isinstance(target.value, ast.Name)
                    and target.value.id == 'workflow'):
                return True
    return False


def _now_utc_imports(tree: ast.Module) -> list[str]:
    return [
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        and any(alias.name == 'now_utc' for alias in node.names)
    ]


def test_workflows_take_their_clock_from_a_replay_safe_source():
    offenders = []
    workflow_modules = 0
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except (SyntaxError, UnicodeDecodeError):
            continue
        if not _defines_a_workflow(tree):
            continue
        workflow_modules += 1
        for module in _now_utc_imports(tree):
            if module in WALL_CLOCK_SOURCES:
                offenders.append(
                    f'{path.relative_to(REPO)} imports now_utc from {module}')

    # Guard the guard: if the decorator shape changes and nothing matches, this
    # test would pass while checking nothing.
    assert workflow_modules > 10, (
        f'only found {workflow_modules} @workflow.defn modules; the detector '
        'is probably broken rather than the tree being clean')
    assert not offenders, (
        'workflow modules must import now_utc from one of '
        f'{sorted(SAFE_SOURCES)}:\n  ' + '\n  '.join(offenders))
