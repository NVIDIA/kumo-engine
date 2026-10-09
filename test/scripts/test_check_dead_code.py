# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def project(tmp_path: Path) -> Iterator[Path]:
    # Vulture matches exclusions against absolute paths; pytest's test_* parent
    # directory would exclude the entire fixture rather than just test files.
    with tempfile.TemporaryDirectory(prefix='vulture-',
                                     dir=tmp_path.parent) as directory:
        root = Path(directory)
        shutil.copy(REPO_ROOT / 'pyproject.toml', root / 'pyproject.toml')
        yield root


def scan(project: Path) -> subprocess.CompletedProcess[str]:
    pytest.importorskip('vulture', reason='Run with the documented Vulture tools')
    return subprocess.run([sys.executable, '-m', 'vulture'], cwd=project,
                          capture_output=True, text=True, check=False)


def test_full_tree_sees_unchanged_callers_and_reports_last_use_removal(
        project: Path) -> None:
    (project / 'lib.py').write_text('def helper():\n    return 1\n')
    (project / 'api.py').write_text('from lib import helper\n')
    consumer = project / 'consumer.py'
    consumer.write_text('from api import helper\nhelper()\n')

    assert scan(project).returncode == 0

    # Only the consumer changes; the newly unused import is in untouched api.py.
    consumer.write_text('print("done")\n')
    result = scan(project)
    assert result.returncode == 3
    assert "api.py:1: unused import 'helper'" in result.stdout


def test_config_excludes_tests_but_reports_source_imports(
        project: Path) -> None:
    tests = project / 'test'
    tests.mkdir()
    (tests / 'test_fixture.py').write_text('import decimal\n')
    benchmark = project / 'packages/diskgraph/docs/col-stats-bench'
    benchmark.mkdir(parents=True)
    (benchmark / 'generate.py').write_text(
        'def generate(total_devices):\n    pass\n')
    (project / 'packages/diskgraph/source.py').write_text('import decimal\n')
    assert scan(project).returncode == 0

    (project / 'source.py').write_text('import decimal\n')
    result = scan(project)
    assert result.returncode == 3
    assert "source.py:1: unused import 'decimal'" in result.stdout
    assert 'test_fixture.py' not in result.stdout


def test_documented_exceptions_do_not_hide_other_findings(
        project: Path) -> None:
    (project / 'interface.py').write_text(
        'from typing import Protocol, Sequence  # noqa: F401\n'
        '\n'
        'class Client(Protocol):\n'
        '    # Match the implementation keyword names.\n'
        '    def send(self, Payload):  # noqa: F841\n'
        '        ...\n'
        '\n'
        'def callback(value: "Sequence"):\n'
        '    return len(value)\n')
    assert scan(project).returncode == 0

    (project / 'unused.py').write_text('import decimal\n')
    result = scan(project)
    assert result.returncode == 3
    assert "unused.py:1: unused import 'decimal'" in result.stdout
    assert 'interface.py' not in result.stdout


@pytest.fixture
def wrapper(project: Path) -> Path:
    scripts = project / 'scripts'
    scripts.mkdir()
    target = scripts / 'check_dead_code.sh'
    shutil.copy(REPO_ROOT / 'scripts/check_dead_code.sh', target)
    return target


@pytest.mark.parametrize('status', [0, 3])
def test_wrapper_pins_vulture_runs_from_root_and_preserves_status(
        project: Path, wrapper: Path, status: int) -> None:
    tools = project / 'tools'
    tools.mkdir()
    mise = tools / 'mise'
    mise.write_text('#!/bin/sh\npwd\nprintf "%s\\n" "$@"\n'
                    f'exit {status}\n')
    mise.chmod(0o755)
    env = {**os.environ, 'PATH': f'{tools}:{os.environ["PATH"]}'}

    result = subprocess.run(['bash', str(wrapper)], cwd=tools, env=env,
                            capture_output=True, text=True, check=False)

    assert result.returncode == status
    assert result.stdout.splitlines() == [
        str(project), 'exec', '--', 'uvx', '--from', 'vulture==2.14', 'vulture'
    ]


def test_wrapper_rejects_obsolete_range_arguments(wrapper: Path) -> None:
    result = subprocess.run(
        ['bash', str(wrapper), '--diff', 'base', 'head'], capture_output=True,
        text=True, check=False)

    assert result.returncode == 2
    assert 'full configured Python source tree' in result.stderr
