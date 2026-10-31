# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from sources.package_boundary import (
    assert_imports_nothing_foreign,
    assert_top_level_import_is_cheap,
)

ROOT = Path(__file__).parents[1]


def test_provider_import_boundary():
    assert_imports_nothing_foreign(ROOT / 'src/kumo_adls', vendor='azure')


def test_top_level_import_is_cheap():
    assert_top_level_import_is_cheap(
        ROOT, 'kumo_adls', 'ADLS_NAMESPACE',
        unloaded=('kumo_adls.backend', 'azure.identity',
                  'azure.storage.filedatalake'))
