# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rules for running the artifact as a Snowflake ML Job (`hosts/snowflake-coco/ml_job.md`).

- Session order. Kumo's Snowflake connector opens its own Snowpark session; a job's launcher
  already holds one. Once both exist, `get_active_session()` raises `SnowparkSessionException`
  (1409). Every code block that takes the session must take it once, before any connector.
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
TAKE_SESSION = re.compile(r'\bget_active_session\(\)|\bSession\.builder\b')
CONNECTOR = re.compile(r'\bSnowflakeConnector\b|\bfrom_spcs_env\(|\bfrom_connection_toml\(')
SUBMITTERS = {'submit_file', 'submit_directory', 'submit_from_stage', 'register'}


def python_blocks() -> list[tuple[str, str]]:
    blocks = []
    for path in sorted(SKILLS.rglob('*.md')):
        for lang, body in FENCE.findall(path.read_text()):
            if lang.strip() == 'python':
                blocks.append((str(path.relative_to(SKILLS)), body))
    return blocks


def code_lines(body: str) -> str:
    """The block without comments and import lines, so only calls are matched."""
    lines = (line.split('#')[0] for line in body.splitlines())
    return '\n'.join(line for line in lines if not re.match(r'\s*(from|import)\s', line))


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


SESSION_BLOCKS = [(where, body) for where, body in python_blocks()
                  if TAKE_SESSION.search(code_lines(body)) and CONNECTOR.search(code_lines(body))]


def test_session_blocks_are_found() -> None:
    assert any(where.endswith('ml_job.md') for where, _ in SESSION_BLOCKS), SESSION_BLOCKS


@pytest.mark.parametrize('where,body', SESSION_BLOCKS, ids=[w for w, _ in SESSION_BLOCKS])
def test_session_is_taken_once_before_any_connector(where: str, body: str) -> None:
    code = code_lines(body)
    takes = [m.start() for m in TAKE_SESSION.finditer(code)]
    assert len(takes) == 1, f'{where}: the session is taken {len(takes)} times'
    assert takes[0] < CONNECTOR.search(code).start(), \
        f'{where}: a connector is built before the session is taken'


def test_ml_job_states_the_session_rule() -> None:
    text = ' '.join(ML_JOB.read_text().split())
    assert '**Take the session once, at the top, before any Kumo connector.**' in text
    assert 'SnowparkSessionException (1409)' in text


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
