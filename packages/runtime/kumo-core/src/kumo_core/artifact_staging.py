# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse


def _local_path(path):
    parsed = urlparse(path)
    if parsed.scheme:
        raise ValueError('Artifact path is not on the local filesystem.')
    return parsed.path


def is_local_artifact_path(path: str, *, _resolve=_local_path) -> bool:
    """Return whether ``path`` names the worker's local filesystem.

    A provider stage that happens to be locally mounted is deliberately not
    local for this purpose: warehouse connectors must resolve it as a stage
    reference rather than upload it through a worker-local path.
    """
    try:
        _resolve(path)
        return True
    except ValueError:
        return False


def selected_local_parquet_files(paths: Sequence[str], *,
                                 _resolve=_local_path) -> list[Path]:
    """Validate an explicit local selection under ``local_parquet_files``' rules.

    A batch-prediction selection spans sibling attempt directories, so it
    cannot be expressed as one root scan. Each named file is therefore held to
    the same symlink, regular-file, suffix and control-character rules that
    scanning a single root would have applied.
    """
    files: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        candidate = Path(_resolve(path)).expanduser()
        if candidate.is_symlink():
            raise ValueError('Local artifact paths may not be symlinks.')
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                'Local artifact path does not exist.') from None
        if not resolved.is_file():
            raise ValueError('Selected local artifact must be a regular file.')
        if resolved.suffix.casefold() != '.parquet':
            raise ValueError(
                'Local artifact file must use the .parquet suffix.')
        if any(character in str(resolved)
               for character in ('\x00', '\n', '\r')):
            raise ValueError(
                'Local artifact path contains a control character.')
        if resolved in seen:
            # Two spellings of one file ('shard.parquet' and
            # './shard.parquet') survive the caller's string-level uniqueness
            # check and collapse only here. Uploading the survivor twice would
            # put it under two stage prefixes and load its rows twice.
            raise ValueError(
                'Local artifact selection names the same file twice.')
        seen.add(resolved)
        files.append(resolved)
    if not files:
        raise ValueError('Local artifact selection is empty.')
    return files


def local_parquet_files(path: str, *, _resolve=_local_path) -> list[Path]:
    """Resolve one local Parquet file or a local Parquet dataset safely.

    Artifact post-processing normally produces a flat directory, but accepting
    nested datasets here costs little and avoids relying on that implementation
    detail.  Symlinks are rejected so staging cannot escape the artifact root.
    Non-Parquet sidecar files are ignored.
    """
    unresolved_root = Path(_resolve(path)).expanduser()
    if unresolved_root.is_symlink():
        raise ValueError('Local artifact paths may not be symlinks.')
    try:
        root = unresolved_root.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError('Local artifact path does not exist.') from None

    if any(character in str(root) for character in ('\x00', '\n', '\r')):
        raise ValueError('Local artifact path contains a control character.')

    if root.is_file():
        if root.suffix.casefold() != '.parquet':
            raise ValueError(
                'Local artifact file must use the .parquet suffix.')
        return [root]
    if not root.is_dir():
        raise ValueError('Local artifact path must be a file or directory.')

    files: list[Path] = []
    for candidate in sorted(root.rglob('*')):
        if candidate.is_symlink():
            raise ValueError(
                'Local artifact datasets may not contain symlinks.')
        if not candidate.is_file() or candidate.suffix.casefold() != '.parquet':
            continue
        resolved = candidate.resolve(strict=True)
        if any(character in str(resolved)
               for character in ('\x00', '\n', '\r')):
            raise ValueError(
                'Local artifact path contains a control character.')
        try:
            resolved.relative_to(root)
        except ValueError:
            raise ValueError(
                'Local artifact file resolves outside the artifact directory.'
            ) from None
        files.append(resolved)

    if not files:
        raise ValueError('Local artifact dataset contains no Parquet files.')
    return files
