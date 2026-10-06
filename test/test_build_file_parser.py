# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from setup import parse_project_version, parse_version_constants


def test_parse_constants_basic(tmp_path: Path):
    # create a mock BUILD file
    build_content = """
KUMO_API_VERSION = 'v0.34.0'
# some comment
OTHER_VAR = "should be ignored"
KUMO_EXTRA_VERSION = "1.2.3"  # trailing comma and comment
"""
    build_file = tmp_path / "BUILD"
    build_file.write_text(build_content)

    # call the parser
    constants = parse_version_constants(build_file)

    # check only uppercase constants are captured
    expected = {
        "KUMO_API_VERSION": "v0.34.0",
        "KUMO_EXTRA_VERSION": "1.2.3",
    }
    assert constants == expected


def test_parse_constants_empty(tmp_path: Path):
    build_file = tmp_path / "BUILD"
    build_file.write_text("")
    constants = parse_version_constants(build_file)
    assert constants == {}


def test_parse_constants_invalid_lines(tmp_path: Path):
    build_content = """
lower_case = "ignored"
KUMO_VALID = "ok"
another = 123
"""
    build_file = tmp_path / "BUILD"
    build_file.write_text(build_content)

    constants = parse_version_constants(build_file)
    assert constants == {"KUMO_VALID": "ok"}


def test_parse_3rdparty_build():
    constants = parse_version_constants()
    assert "KUMO_API_VERSION" not in constants


def test_parse_kumo_api_project_version():
    project_file = (Path(__file__).parents[1] / "packages" / "kumo-api" /
                    "pyproject.toml")
    assert parse_project_version(project_file) == "3.0.0"


def test_linting_retains_required_setup_dependencies_check():
    workflow_file = (Path(__file__).parents[1] / ".github" / "workflows" /
                     "linting.yml")
    assert "\n  setup_dependencies:\n" in workflow_file.read_text()
