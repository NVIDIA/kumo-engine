# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column-name diagnostics shared by contracts and physical bindings."""

from collections.abc import Collection
from difflib import get_close_matches


def _column_name_suggestion(missing: Collection[str], available: Collection[str]) -> str | None:
    case_matches = set()
    close_matches = set()
    for name in sorted(missing):
        matches = [item for item in available if item.casefold() == name.casefold()]
        if matches:
            case_matches.update(matches)
        else:
            close_matches.update(get_close_matches(name, sorted(available)))
    hints = []
    if case_matches:
        stored = ", ".join(repr(item) for item in sorted(case_matches))
        hints.append(f"stored as {stored}; column names are case-sensitive")
    if close_matches:
        hints.append(f"did you mean {', '.join(repr(item) for item in sorted(close_matches))}?")
    return "; ".join(hints) or None
