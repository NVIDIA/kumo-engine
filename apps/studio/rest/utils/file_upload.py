# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io_utils import fsutil


def supports_backend_file_upload(storage_root: str) -> bool:
    """Return whether Kumo can upload browser-selected files to storage."""
    return fsutil.supports_multipart_upload(storage_root)
