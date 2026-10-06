# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_sources(
    name="download",
    sources=["__init__.py", "download.py", "errors.py", "session.py"],
    overrides={
        # Pants' PyArrow stubs do not export S3FileSystem from pyarrow.fs.
        "session.py": {"skip_pyright": True},
    },
    dependencies=[
        "//3rdparty:boto_s3",
        "//3rdparty:botocore",
        "//3rdparty:pyarrow",
        "//packages/kumo-core/src/kumo_core",
    ],
)
