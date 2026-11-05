# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).parents[1] / 'pyproject.toml'


def test_the_databricks_extra_reads_unity_catalog_model_storage():
    extras = tomllib.loads(
        PYPROJECT.read_text())['project']['optional-dependencies']
    names = {re.split(r'[<>=!~ \[;]', r, 1)[0] for r in extras['databricks']}
    assert {'boto3', 'azure-storage-file-datalake',
            'google-cloud-storage'} <= names
