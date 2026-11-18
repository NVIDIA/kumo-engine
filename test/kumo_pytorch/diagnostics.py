# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-difference evidence without changing the acceptance comparisons."""

from itertools import zip_longest


def mismatch(actual, expected, rows, field, *, precision="exact"):
    absent = object()
    for position, (left, right) in enumerate(
        zip_longest(actual, expected, fillvalue=absent)
    ):
        if left != right:
            row = rows[position] if position < len(rows) else {}
            left = "<absent>" if left is absent else left
            right = "<absent>" if right is absent else right
            return (
                f"{field}: first mismatch at batch position {position}, "
                f"source anchor position {row.get('anchor_row_position', '<absent>')}, "
                f"sample_id={row.get('sample_id', '<absent>')!r}: "
                f"expected={right!r}, actual={left!r}; precision={precision}"
            )
    return f"{field}: no differing value; actual_count={len(actual)}, expected_count={len(expected)}; precision={precision}"
