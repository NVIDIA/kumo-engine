# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rules for running the artifact as a Snowflake ML Job (`hosts/snowflake-coco/ml_job.md`).

- Shared session. Job connectors use the launcher's Snowpark session.
- Runtime pin. When `runtime_environment` is omitted, the job runs on Snowflake's default runtime,
  which moves with Snowflake releases and has carried a torch build outside Kumo's range. Every
  submission passes the host's `KUMO_JOB_RUNTIME`, which the adapter defines as '2.9.0'.
"""
import ast
import re
from pathlib import Path

import pytest

SKILLS = Path(__file__).resolve().parents[1] / 'skills'
HOST = SKILLS / 'hosts' / 'snowflake-coco'
ML_JOB = HOST / 'ml_job.md'
ADAPTER = HOST / 'adapter.md'
PINNED = '2.9.0'

FENCE = re.compile(r'^[ \t]*```([^\n]*)\n(.*?)^[ \t]*```', re.DOTALL | re.MULTILINE)
SUBMITTERS = {'submit_file', 'submit_directory', 'submit_from_stage', 'register'}


def submissions(body: str) -> list[ast.Call]:
    calls = []
    for node in ast.walk(ast.parse(body)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in SUBMITTERS:
                if node.func.attr != 'register' or ast.unparse(node.func.value) == 'MLJobDefinition':
                    calls.append(node)
    return calls


def job_blocks() -> list[str]:
    return [body for _, body in FENCE.findall(ML_JOB.read_text())
            if 'submit_' in body or 'MLJobDefinition.register' in body]


def test_job_connectors_share_the_launcher_session() -> None:
    blocks = [body for _, body in FENCE.findall(ML_JOB.read_text())
              if 'SnowflakeConnector' in body]
    assert blocks
    for body in blocks:
        calls = [node for node in ast.walk(ast.parse(body))
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and ast.unparse(node.func.value) == 'kumo.SnowflakeConnector']
        assert calls
        for call in calls:
            assert call.func.attr == 'from_session', ast.unparse(call)
            assert [ast.unparse(arg) for arg in call.args] == ['session']


def test_submissions_are_found() -> None:
    assert len([c for body in job_blocks() for c in submissions(body)]) >= 2


@pytest.mark.parametrize('body', job_blocks())
def test_every_submission_pins_runtime_and_passes_the_session(body: str) -> None:
    for call in submissions(body):
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert kw.get('runtime_environment') == 'KUMO_JOB_RUNTIME', ast.unparse(call)
        assert kw.get('session') == 'session', ast.unparse(call)
        assert kw.get(None) == 'KUMO_JOB_REQUIREMENTS', ast.unparse(call)


def test_runtime_value_matches_the_adapter() -> None:
    assigned = {node.value.value
                for body in job_blocks() for node in ast.walk(ast.parse(body))
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and any(isinstance(t, ast.Name) and t.id == 'KUMO_JOB_RUNTIME'
                        for t in node.targets)}
    assert assigned == {PINNED}
    row = next(line for line in ADAPTER.read_text().splitlines()
               if line.startswith('| `KUMO_JOB_RUNTIME`'))
    assert f"`'{PINNED}'`" in row


def test_no_skill_text_accepts_the_default_runtime() -> None:
    text = '\n'.join(p.read_text() for p in sorted(SKILLS.rglob('*.md')))
    assert not re.search(r'runtime_environment\s*=\s*None', text)
    assert 'Never the default runtime' in ML_JOB.read_text()
