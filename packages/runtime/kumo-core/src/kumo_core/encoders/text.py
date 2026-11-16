# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re

import pandas as pd


PUNCTUATION = re.compile(r"[\'\"\.,\(\)\!\?\;\:]")
MULTISPACE = re.compile(r"\s+")


def normalize_text(ser: pd.Series) -> pd.Series:
    r"""Basic normalization for a series of text.

    Normalization includes lowercasing and adjusting spaces.
    Returns a list of words after splitting on whitespace.

    No-op if series is already normalized.
    """
    if len(ser) == 0 or pd.api.types.is_list_like(ser.iloc[0]):
        return ser

    def normalize_fn(line: str) -> list[str]:
        line = PUNCTUATION.sub(" ", line)
        line = re.sub(r"<br\s*/?>", " ", line)  # Handle <br /> or <br>
        line = MULTISPACE.sub(" ", line)
        return line.split()

    ser = ser.fillna('').astype(str).str.lower()
    ser = ser.map(normalize_fn)
    return ser
