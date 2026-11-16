# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from urllib.parse import quote


def safe_filename(name: str, max_length: int = 255) -> str:
    r"""Make a string suitable for the use of a filename.
    """
    safe_name = quote(name, safe='')
    # Replace leading/trailing spaces and truncate to max_length:
    return safe_name.strip()[:max_length]
