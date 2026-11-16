# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable CSV artifact writing."""

from __future__ import annotations

from typing import Any

import pandas as pd
from kumo_core.io.filesystem import exists, makedirs, open_file


def save_csv(value: Any, path: str, overwrite: bool = False) -> None:
    if not overwrite and exists(path):
        raise FileExistsError(path)
    if isinstance(value, pd.DataFrame):
        frame = value
    else:
        rows = value if isinstance(value, (tuple, list)) else [value]
        frame = pd.DataFrame(rows)
    parent = path.rsplit("/", maxsplit=1)[0] if "/" in path else ""
    if parent:
        makedirs(parent, exist_ok=True)
    with open_file(path, "wt", encoding="utf-8") as target:
        frame.to_csv(target, index=False, float_format="%.6f")
