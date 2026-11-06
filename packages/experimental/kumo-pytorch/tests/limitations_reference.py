# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the DiskGraph list in ``docs/limitations.md`` from `DiskGraphEngine.unsupported`.

``python -m tests.limitations_reference --write`` regenerates it;
``python -m tests.limitations_reference`` exits nonzero when it is stale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kumo.pytorch.api import DiskGraphEngine

LIMITATIONS = Path(__file__).resolve().parents[1] / "docs" / "limitations.md"
BEGIN = "<!-- generated from DiskGraphEngine.unsupported by tests/limitations_reference.py -->"
END = "<!-- end of generated list -->"


def render() -> str:
    """The generated block: one row per entry, with its code, what's refused and the next step."""

    rows = [
        "| Code | Refused | Refused at | Instead |",
        "|---|---|---|---|",
        *(
            f"| `{entry.code}` | {entry.description} | `{entry.refused_at}` | "
            f"{entry.next_step[0].upper()}{entry.next_step[1:]}. |"
            for entry in DiskGraphEngine.unsupported.values()
        ),
    ]
    return "\n".join((BEGIN, *rows, END))


def generate(text: str) -> str:
    """``text`` with its generated block replaced by the current one."""

    start, stop = text.index(BEGIN), text.index(END) + len(END)
    return text[:start] + render() + text[stop:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the list")
    args = parser.parse_args(argv)
    text = LIMITATIONS.read_text()
    current = generate(text)
    if args.write:
        LIMITATIONS.write_text(current)
        return 0
    if current != text:
        print("docs/limitations.md is stale; run: python -m tests.limitations_reference --write")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
