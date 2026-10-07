# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from setup import parse_project_version


def test_parse_kumo_api_project_version():
    project_file = (Path(__file__).parents[1] / "packages" / "kumo-api" /
                    "pyproject.toml")
    assert parse_project_version(project_file) == "3.0.0"


def test_linting_retains_required_setup_dependencies_check():
    workflow_file = (Path(__file__).parents[1] / ".github" / "workflows" /
                     "linting.yml")
    assert "\n  setup_dependencies:\n" in workflow_file.read_text()
