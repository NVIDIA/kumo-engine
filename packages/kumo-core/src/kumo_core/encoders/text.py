# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

PUNCTUATION = re.compile(r"[\'\"\.,\(\)\!\?\;\:]")
MULTISPACE = re.compile(r"\s+")


def normalize_text_arrow(array: pa.Array) -> pa.Array:
    """Normalize Arrow text into lists of words without pandas conversion.

    Args:
        array: Raw strings or already-tokenized lists of words.

    Returns:
        Lists of lowercase words, with null strings represented as empty text.
        Already-tokenized lists retain their values and casing.

    Raises:
        ArrowNotImplementedError: If values cannot be converted to text.
    """
    array = pc.dictionary_decode(array)
    if (
        pa.types.is_list(array.type)
        or pa.types.is_large_list(array.type)
        or pa.types.is_fixed_size_list(array.type)
    ):
        return array
    raw = pc.fill_null(pc.cast(array, pa.string()), "")
    text = pc.ascii_lower(raw)
    non_ascii = pc.invert(pc.string_is_ascii(raw))
    if pc.any(non_ascii).as_py():
        # Arrow's simple Unicode casing omits Python's contextual/expanded forms.
        replacements = pa.array(
            [value.lower() for value in pc.filter(raw, non_ascii).to_pylist()]
        )
        text = pc.replace_with_mask(text, non_ascii, replacements)
    text = pc.replace_substring_regex(
        text, pattern=PUNCTUATION.pattern, replacement=" "
    )
    whitespace = r"[\s\x{001c}-\x{001f}\x{0085}\p{Z}]"
    text = pc.replace_substring_regex(
        text, pattern=rf"<br{whitespace}*/?>", replacement=" "
    )
    text = pc.replace_substring_regex(text, pattern=whitespace + "+", replacement=" ")
    return pc.split_pattern(text, pattern=" ")


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

    ser = ser.fillna("").astype(str).str.lower()
    ser = ser.map(normalize_fn)
    return ser
