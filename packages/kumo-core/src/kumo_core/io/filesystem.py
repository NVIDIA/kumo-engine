# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Narrow fsspec filesystem surface used by portable training."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import fsspec
from fsspec.implementations.local import LocalFileSystem

# gcsfs >=2026.3 otherwise maps gs:// to ExtendedGcsFileSystem, whose per-bucket
# gRPC layout lookup stalls when it fails. Kumo uses no zonal/HNS buckets.
os.environ.setdefault("GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT", "false")

_fsspec_fallback: Callable[[str], str] | None = None


def register_fsspec_fallback(adapter: Callable[[str], str]) -> None:
    """Register a platform adapter for protocols unavailable to fsspec."""
    global _fsspec_fallback
    _fsspec_fallback = adapter


def _resolve(path: str) -> tuple[Any, str]:
    return fsspec.core.url_to_fs(fsspec_mangle(path))


def exists(path: str) -> bool:
    fs, fs_path = _resolve(path)
    return bool(fs.exists(fs_path))


def makedirs(path: str, exist_ok: bool = False) -> None:
    fs, fs_path = _resolve(path)
    fs.makedirs(fs_path, exist_ok=exist_ok)


def rm(path: str, ignore_errors: bool = False) -> None:
    fs, fs_path = _resolve(path)
    try:
        fs.rm(fs_path, recursive=True)
    except FileNotFoundError:
        if not ignore_errors:
            raise


def copy_file(src: str, dst: str, overwrite: bool = False) -> None:
    if not overwrite and exists(dst):
        raise FileExistsError(dst)
    parent = dst.rsplit("/", maxsplit=1)[0] if "/" in dst else ""
    if parent:
        makedirs(parent, exist_ok=True)
    with open_file(src, "rb") as source, open_file(dst, "wb") as target:
        shutil.copyfileobj(source, target)


def copy(src: str, dst: str, overwrite: bool = False) -> None:
    src_fs, src_path = _resolve(src)
    if not src_fs.isdir(src_path):
        copy_file(src, dst, overwrite=overwrite)
        return
    for source_path in src_fs.find(src_path):
        relative = PurePosixPath(source_path).relative_to(src_path)
        source_uri = src_fs.unstrip_protocol(source_path)
        target_uri = f"{dst.rstrip('/')}/{relative.as_posix()}"
        copy_file(source_uri, target_uri, overwrite=overwrite)


def is_local(path: str) -> bool:
    protocol = fsspec.utils.get_protocol(path)
    return protocol in {"file", "local"}


def local_path(path: str) -> Path:
    """Resolve a local fsspec URI to its operating-system path."""
    fs, fs_path = _resolve(path)
    if not isinstance(fs, LocalFileSystem):
        raise ValueError(f"path is not on a local filesystem: {path}")
    return Path(fs_path)


def listdir(path: str) -> list[str]:
    fs, fs_path = _resolve(path)
    return [fs.unstrip_protocol(item) for item in fs.ls(fs_path, detail=False)]


def torch_save(value: Any, path: str, **kwargs: Any) -> None:
    import torch

    parent = path.rsplit("/", maxsplit=1)[0] if "/" in path else ""
    if parent:
        makedirs(parent, exist_ok=True)
    with open_file(path, "wb") as target:
        torch.save(value, target, **kwargs)


def open_file(path: str, mode: str, **kwargs: Any) -> Any:
    return fsspec.open(fsspec_mangle(path), mode, **kwargs)


def fsspec_mangle(path: str) -> str:
    """Return a path consumable by fsspec in portable and monolith runtimes.

    ``kumo-core`` uses native fsspec implementations when they are installed.
    A platform runtime may register a compatibility adapter for providers it
    supports without adding that platform as a dependency of ``kumo-core``.
    """
    protocol = fsspec.utils.get_protocol(path)
    if _fsspec_fallback is not None and protocol in {"s3", "s3a"}:
        return _fsspec_fallback(path)
    try:
        fsspec.get_filesystem_class(protocol)
        return path
    except (ImportError, ValueError):
        return _fsspec_fallback(path) if _fsspec_fallback else path
