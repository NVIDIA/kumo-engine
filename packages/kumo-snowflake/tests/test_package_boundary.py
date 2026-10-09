# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from sources.package_boundary import (
    assert_imports_nothing_foreign,
    assert_top_level_import_is_cheap,
)

ROOT = Path(__file__).parents[1]


def test_the_distribution_imports_no_platform_or_foreign_vendor_module():
    assert_imports_nothing_foreign(ROOT / "src" / "kumo_snowflake",
                                   vendor="snowflake")


def test_a_top_level_import_does_not_load_the_training_stack():
    assert_top_level_import_is_cheap(ROOT, "kumo_snowflake",
                                     "SnowflakeSourceReader")
