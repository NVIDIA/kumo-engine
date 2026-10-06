# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import kumo_databricks
import pytest
from sources.package_boundary import (
    assert_imports_nothing_foreign,
    assert_top_level_import_is_cheap,
)

ROOT = Path(__file__).parents[1]


def test_the_distribution_imports_no_platform_or_foreign_vendor_module():
    """The provider depends on contracts, not either Kumo workflow host."""
    assert_imports_nothing_foreign(ROOT / "src" / "kumo_databricks",
                                   vendor="databricks")


def test_the_boundary_rule_rejects_a_sibling_provider(tmp_path):
    source_root = tmp_path / "kumo_databricks"
    source_root.mkdir()
    (source_root / "leak.py").write_text(
        "from kumo_snowflake import SnowflakeSourceReader\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="kumo_snowflake"):
        assert_imports_nothing_foreign(source_root, vendor="databricks")


def test_only_the_provider_contract_is_public():
    assert importlib.util.find_spec("kumo_databricks_connector") is None
    assert kumo_databricks.DatabricksSourceReader is not None
    for workflow_name in (
            "NativeDatabase",
            "PredictionRun",
            "TrainingRun",
            "TrainingRunSpec",
            "fit",
    ):
        assert not hasattr(kumo_databricks, workflow_name)


def test_a_top_level_import_loads_neither_training_nor_the_plugin_half():
    # The SDK plugin half lives here too, and `session` imports the driver at
    # module scope, so the modules themselves are named: nothing but resolving
    # one of the lazy exports may load them. The driver is not named -- the
    # platform imports it at `databricks_connector.py` module scope anyway.
    assert_top_level_import_is_cheap(
        ROOT, "kumo_databricks", "DatabricksSourceReader",
        unloaded=("kumo_databricks.session", "kumo_databricks.backend",
                  "kumo_databricks.sink"))
