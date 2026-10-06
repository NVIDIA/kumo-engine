# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Scoped to the release-script tests only: test_task_config_validator.py in
# this directory predates any Pants target here and is left unmanaged.
python_tests(
    name="test_release_scripts",
    sources=[
        "test_build_kumo_images.py",
        "test_check_updated_release_branches.py",
        "test_make_kumo3_hotfix_release.py",
        "test_release_v3_guards.py",
        "test_start_kumo3_release.py",
    ],
    dependencies=[
        "//scripts/release:kumo3_release_script",
        "//scripts/release:kumo3_hotfix_release_script",
        "//scripts/release:release_branch_sweep_script",
        "//scripts/release:build_images_script",
    ],
)

python_test_utils(
    name="test_utils",
    sources=["conftest.py"],
)
