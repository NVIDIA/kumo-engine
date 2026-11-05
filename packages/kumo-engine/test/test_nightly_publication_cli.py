# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference creation must identify durable proposal storage before any work."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / 'scripts/nightly_check.py'
SPEC = importlib.util.spec_from_file_location('publication_cli', PATH)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def test_reference_cli_requires_durable_destination_before_work(monkeypatch):
    calls = []
    monkeypatch.setattr(check, 'run', lambda args: calls.append(args) or True)
    arguments = [
        'run', '--configuration', 'inputs.json', '--output-dir', 'out',
        '--wheel', 'reference.whl', '--mode', 'reference', '--scale', '100M',
        '--case', 'binary-normal', '--input-root', 'data', '--scratch',
        'scratch', '--cgroup-root', 'groups', '--time-budget-seconds', '60'
    ]
    with pytest.raises(SystemExit):
        check.main(arguments)
    assert calls == []
    assert check.main(
        arguments +
        ['--reference-output-s3', 's3://test-bucket/proposals']) == 0
    assert calls[0].reference_output_s3 == 's3://test-bucket/proposals'
    assert calls[0].require_local_ssd is True
