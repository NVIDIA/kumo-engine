# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from test.kumo_pytorch.benchmarking.suite import run_suite

import pytest


def test_suite_retains_each_case_and_does_not_pass_partial_evidence(tmp_path):
    cases = [
        {
            "name": "numeric",
            "adapter": "test.kumo_pytorch.checks.test_benchmark:tabular",
            "config": {},
        },
        {"name": "failed", "adapter": "missing_benchmark:factory", "config": {}},
    ]
    result = run_suite(cases, tmp_path / "suite", repetitions=2, epochs=2, timeout=20)
    assert [row["status"] for row in result["cases"]] == ["incomplete", "failed"]
    assert result["status"] == "failed"
    assert json.loads((tmp_path / "suite/results.json").read_text()) == result
    assert "numeric/report.md" in (tmp_path / "suite/report.md").read_text()


@pytest.mark.parametrize("names", [[], ["../outside"], ["repeat", "repeat"]])
def test_invalid_case_names_fail_before_creating_output(tmp_path, names):
    with pytest.raises(ValueError):
        run_suite(
            [{"name": name, "adapter": "unused", "config": {}} for name in names],
            tmp_path / "suite",
        )
    assert not (tmp_path / "suite").exists()
