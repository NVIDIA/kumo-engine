# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_sources(
    name="download",
    sources=["__init__.py", "auth.py", "download.py", "session.py"],
    overrides={
        # Private helpers are consumed by SDK plugins outside this target.
        "session.py": {"skip_pyright": True},
    },
    dependencies=[
        "//3rdparty:gcp_storage",
        "//3rdparty:google_auth",
        "//3rdparty:pyarrow",
        "//packages/kumo-core/src/kumo_core",
    ],
)
