# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from typing import Any

import pandas as pd
from kumo_core.csv_rules import (
    DENIED,
    SNIFF_CHARS,
    UNSNIFFABLE,
    NoValidSeparatorFoundError,
    normalize_newlines,
    separator_with_reason,
    validate_separator,
)

from io_utils import fsutil, logging

logger = logging.getLogger(__name__)

_ERR_MSG = "Could not auto-detect a delimiter"


def save_csv(cfg: Any, path: str, overwrite: bool = False):
    if isinstance(cfg, pd.DataFrame):
        df = cfg
    else:
        if not isinstance(cfg, (tuple, list)):
            cfg = [cfg]
        df = pd.DataFrame(cfg)
    if fsutil.is_local(path):
        df.to_csv(path, index=False, float_format='%.6f')
    else:
        with tempfile.NamedTemporaryFile() as f:
            df.to_csv(f.name, index=False, float_format='%.6f')
            fsutil.copy_file(f.name, path, overwrite=overwrite)


def detect_csv_sep(path: str) -> str:
    r"""Detects the separator in a CSV file using the python standard library.
    The algorithm assumes that the file roughly adheres to the CSV file
    specification, so it may not be ideal for multicategorical separator
    inference.

    There are a few additions compared to the standard library algorithm:

    - We will prefer a greater number of delimiters, such as the `|` character
    - We disallow certain characters, such as alphanumeric characters
    - If we cannot find a delimiter or find delimiter that is in the deny
      list, we will default to `,`.
    - If the file is empty, we will throw no valid separator found error.
    - After we detect the delimiter, we will load with pandas to ensure that
      the top content in the file is not malformed.
    """
    # TODO(daniel) migrate off get_handle
    with fsutil.get_handle(path, 'r') as f:
        text = normalize_newlines(f.read(SNIFF_CHARS))

    delimiter, reason = separator_with_reason(text)
    if reason == UNSNIFFABLE:
        logger.warning(f'No separator found when loading {path}, likely '
                       f'because the table contains only a single column '
                       f'defaulting to ","')
    elif reason == DENIED and delimiter is not None:
        logger.warning('Found a delimiter in the deny list. Default to ","')
    if delimiter is None:
        # No valid separator since file is empty.
        raise NoValidSeparatorFoundError(_ERR_MSG)

    # Read the csv file and ensure it is not malformed.
    validate_separator(text, delimiter)

    return delimiter
