# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import subprocess

from kumo_core.version import _is_git_repository, get_kumo_version

import kumo.home


def test_is_git_repository(tmp_path: pathlib.Path):
    root_dir = tmp_path.resolve().as_posix()
    assert not _is_git_repository(root_dir)


def test_get_kumo_version():
    # kumo is a namespace package (no __init__.py, so no __file__); resolve
    # the install dir from a real submodule instead.
    kumo_install_path = kumo.home.__file__
    kumo_install_path = kumo_install_path[:kumo_install_path.rindex('/')]
    expected_version_dev_branch = subprocess.check_output(
        ['git', '-C', kumo_install_path, 'rev-parse', '--abbrev-ref',
         'HEAD']).decode("utf-8").strip("\n")
    expected_version_master = subprocess.check_output(
        ['git', '-C', kumo_install_path, 'rev-parse',
         'HEAD']).decode("utf-8").strip("\n")
    assert (get_kumo_version() == expected_version_dev_branch
            or get_kumo_version() == expected_version_master)
