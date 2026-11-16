# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hosted URI adapter for common local artifact selection."""
from functools import partial

from kumo_core import artifact_staging as _shared

from io_utils.fsutil.uri import LocalURI, ParseURI


def _resolve(path):
    parsed = ParseURI(path)
    if not isinstance(parsed, LocalURI):
        raise ValueError('Artifact path is not on the local filesystem.')
    return parsed.path


is_local_artifact_path = partial(_shared.is_local_artifact_path,
                                 _resolve=_resolve)
selected_local_parquet_files = partial(_shared.selected_local_parquet_files,
                                       _resolve=_resolve)
local_parquet_files = partial(_shared.local_parquet_files, _resolve=_resolve)
