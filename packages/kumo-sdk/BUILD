# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_sources(
    name="kumo_sdk",
    sources=["kumoai_studio/**/*.py"],
    dependencies=["//packages/kumo-api/kumoapi"],
    # The package's own CI runs `ty` with the compatibility overrides in
    # pyproject.toml; do not also subject it to the monorepo's Pyright policy.
    skip_pyright=True,
)
