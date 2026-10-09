# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

from setuptools import setup

kumo_api_release_ver = '3.0.0'
api_pyproject = Path(__file__).parent.parent / 'kumo-api' / 'pyproject.toml'
api_ver = kumo_api_release_ver
if api_pyproject.exists():
    match = re.search(
        r'^\[project\]\s*$.*?^version\s*=\s*[\'\"]([^\'\"]+)[\'\"]',
        api_pyproject.read_text(encoding='utf-8'),
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f'No [project].version found in {api_pyproject}')
    api_ver = match.group(1)

# The API and SDK change atomically in the monorepo. Wheels retain a portable,
# ordinary versioned dependency; development and CI install packages/kumo-api
# first so the matching colocated source satisfies it without a Git reference.
kumo_api = f'kumo-api=={api_ver}'

setup(
    install_requires=[
        "fsspec>=2025.3.0",
        "pandas",
        "pyarrow>=8.0.0",
        "requests>=2.28.2",
        "urllib3",
        "plotly",
        "typing_extensions>=4.5.0",  # for override
        kumo_api,
        "tqdm>=4.66.0",
        "aiohttp>=3.10.0",
        "pydantic>=1.10.21",
    ], )
