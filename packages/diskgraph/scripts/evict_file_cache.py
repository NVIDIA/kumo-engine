#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evict clean page-cache pages for files under a path.

This is deliberately narrower than Linux's global drop_caches control: it uses
POSIX_FADV_DONTNEED on each regular file and therefore does not disturb
unrelated workloads on the host. Files are opened read-only and never changed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def regular_files(path: Path):
    if path.is_file():
        yield path
        return
    for root, _dirs, names in os.walk(path):
        root_path = Path(root)
        for name in names:
            candidate = root_path / name
            if candidate.is_file() and not candidate.is_symlink():
                yield candidate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evict only the clean page-cache pages belonging to PATH"
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        parser.error("Python/os on this platform does not expose POSIX_FADV_DONTNEED")
    if not args.path.exists():
        parser.error(f"path does not exist: {args.path}")

    files = 0
    bytes_advised = 0
    errors = 0
    for path in regular_files(args.path):
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            try:
                size = os.fstat(fd).st_size
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            files += 1
            bytes_advised += size
        except OSError as error:
            errors += 1
            print(f"warning: {path}: {error}", file=os.sys.stderr)

    print(
        f"advised DONTNEED for {files} files "
        f"({bytes_advised / (1 << 30):.2f} GiB); errors={errors}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
