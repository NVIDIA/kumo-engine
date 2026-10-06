#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Set the version on the first line of ``kumoai_studio/_version.py``.

The release workflows used to rewrite that file with ``echo ... > _version.py``,
which truncated it and dropped the SPDX header OSRB requires. Replacing only the
first line keeps the header, and keeps the ``exec(input())`` readers in those
same workflows working -- they read line 1 and nothing else.
"""
import sys
from pathlib import Path

VERSION_FILE = Path(
    __file__).resolve().parent.parent / 'kumoai_studio' / '_version.py'


def main() -> int:
    if len(sys.argv) != 2:
        print(f'usage: {Path(sys.argv[0]).name} <version>', file=sys.stderr)
        return 2

    lines = VERSION_FILE.read_text(encoding='utf-8').split('\n')
    lines[0] = f"__version__ = '{sys.argv[1]}'"
    VERSION_FILE.write_text('\n'.join(lines), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
