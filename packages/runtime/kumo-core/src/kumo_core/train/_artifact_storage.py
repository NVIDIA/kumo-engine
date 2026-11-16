# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage operations used by prediction manifests.

Hosted callers supply their existing provider-aware fsutil and atomic JSON
writer. Standalone callers only need ordinary local files.
"""

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from kumo_core.io.filesystem import atomic_write_bytes, fsync_directory


class _LocalFiles:
    @staticmethod
    def filesize(path):
        try:
            return Path(path).stat().st_size
        except FileNotFoundError:
            return None

    @staticmethod
    def read_binary(path):
        return Path(path).read_bytes()

    @staticmethod
    def get_binary_handle(path, mode):
        return open(path, mode)

    @staticmethod
    def exists(path):
        return Path(path).exists()

    @staticmethod
    def listdir(path):
        return [str(child) for child in sorted(Path(path).iterdir())]

    @staticmethod
    def rm(path, *, ignore_errors):
        shutil.rmtree(path, ignore_errors=ignore_errors)

    @staticmethod
    def write_text_if_absent(path, text):
        # Publish a complete sibling with a no-replace link. O_EXCL on the
        # destination itself would expose partially written JSON to readers.
        with _temporary_text(path, text) as temporary:
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
        fsync_directory(Path(path).parent)
        return True


@contextmanager
def _temporary_text(path, text):
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=parent, prefix='.manifest-')
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        yield temporary
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_json_atomic(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, json.dumps(value, indent=2).encode('utf-8'))


@dataclass(frozen=True)
class _ManifestStorage:
    fs: Any
    save_json_atomic: Callable


_LOCAL_STORAGE = _ManifestStorage(_LocalFiles, _save_json_atomic)
