# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_sources(
    name="s3",
    sources=["_skip.py", "s3.py"],
    dependencies=[
        "//3rdparty:boto_s3",
        "//3rdparty:moto",
        "//io_utils/fsutil:fsutil",
        "//io_utils/fsutil:impl_s3",
    ],
)
