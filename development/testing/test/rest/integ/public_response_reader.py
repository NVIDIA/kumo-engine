# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlsplit
from urllib.request import urlopen

from io_utils import fsutil


def public_parquet_refs(reference: str) -> list[str]:
    """Expand a local public directory without probing signed HTTP URLs."""
    if urlsplit(reference).scheme in {'http', 'https'}:
        return [reference]
    if fsutil.isdir(reference):
        refs = fsutil.listdir(reference, suffix='.parquet')
        assert refs, 'Public data directory contained no parquet files'
        return refs
    return [reference]


def flatten_public_parquet_refs(references: Iterable[str]) -> list[str]:
    """Expand every public parquet reference without dropping partitions."""
    return [
        parquet_ref for reference in references
        for parquet_ref in public_parquet_refs(reference)
    ]


def read_public_response_bytes(reference: str) -> bytes:
    """Read a public response reference through its supported transport."""
    if urlsplit(reference).scheme in {'http', 'https'}:
        with urlopen(reference, timeout=60) as response:  # nosec B310
            return response.read()
    return fsutil.read_binary(reference)
