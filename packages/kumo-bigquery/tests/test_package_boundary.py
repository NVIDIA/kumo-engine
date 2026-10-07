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
    assert_imports_nothing_foreign(ROOT / "src" / "kumo_bigquery",
                                   vendor="google")


def test_a_top_level_import_loads_neither_training_nor_the_plugin_half():
    # ``bigquery_storage`` alone among the drivers: the platform reads through
    # the REST client and never loads it, so deferring it here still saves
    # something. The rest it imports itself, at ``bigquery_connector.py``.
    assert_top_level_import_is_cheap(
        ROOT, "kumo_bigquery", "BigQuerySourceReader",
        unloaded=("google.cloud.bigquery_storage",
                  "kumo_bigquery.session", "kumo_bigquery.backend",
                  "kumo_bigquery.sink"))
