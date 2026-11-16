# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class UploadedFile:
    local_path: Path
    remote_path: str
    size_bytes: int
    sha256: str


def is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://")


def is_adls_uri(uri: str) -> bool:
    return uri.startswith(("abfss://", "abfs://"))


def is_local_uri(uri: str) -> bool:
    return "://" not in uri or uri.startswith("file://")


def local_path_from_uri(uri: str) -> Path:
    if uri.startswith("file://"):
        return Path(uri[len("file://"):])
    return Path(uri)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def join_uri(root: str, *parts: str) -> str:
    clean_root = root.rstrip("/")
    clean_parts = [part.strip("/") for part in parts if part]
    if not clean_parts:
        return clean_root
    return clean_root + "/" + "/".join(clean_parts)


def dirname_uri(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[0]


def upload_directory(local_dir: Path, remote_root: str,
                     overwrite: bool) -> list[UploadedFile]:
    if is_s3_uri(remote_root):
        return _upload_directory_to_s3(local_dir, remote_root, overwrite)
    if is_adls_uri(remote_root):
        return _upload_directory_to_adls(local_dir, remote_root, overwrite)
    if is_local_uri(remote_root):
        return _copy_directory_local(local_dir, local_path_from_uri(remote_root),
                                     overwrite)
    raise ValueError(f"Unsupported storage root for generator upload: {remote_root}")


def upload_file(local_path: Path, remote_path: str) -> UploadedFile:
    if is_s3_uri(remote_path):
        return _upload_file_to_s3(local_path, remote_path)
    if is_adls_uri(remote_path):
        return _upload_file_to_adls(local_path, remote_path)
    if is_local_uri(remote_path):
        return _copy_file_local(local_path, local_path_from_uri(remote_path))
    raise ValueError(f"Unsupported storage path for generator upload: {remote_path}")


def download_file(uri: str, dest: Path) -> None:
    if is_s3_uri(uri):
        _download_file_from_s3(uri, dest)
    elif is_adls_uri(uri):
        _download_file_from_adls(uri, dest)
    elif is_local_uri(uri):
        shutil.copy2(local_path_from_uri(uri), dest)
    else:
        raise ValueError(f"Unsupported storage path for download: {uri}")


def _copy_file_local(local_path: Path, dest_path: Path) -> UploadedFile:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, dest_path)
    return UploadedFile(
        local_path=local_path,
        remote_path=str(dest_path),
        size_bytes=local_path.stat().st_size,
        sha256=file_sha256(local_path),
    )


def _copy_directory_local(local_dir: Path, dest_dir: Path,
                          overwrite: bool) -> list[UploadedFile]:
    if dest_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{dest_dir} already exists; pass --overwrite")
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    uploaded: list[UploadedFile] = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir)
        out = dest_dir / rel
        uploaded.append(_copy_file_local(path, out))
    return uploaded


def _upload_file_to_s3(local_path: Path, remote_path: str) -> UploadedFile:
    import boto3

    parsed = urlparse(remote_path)
    bucket = parsed.netloc
    key = parsed.path.strip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {remote_path}")
    boto3.client("s3").upload_file(str(local_path), bucket, key)
    return UploadedFile(
        local_path=local_path,
        remote_path=f"s3://{bucket}/{key}",
        size_bytes=local_path.stat().st_size,
        sha256=file_sha256(local_path),
    )


def _download_file_from_s3(uri: str, dest: Path) -> None:
    import boto3

    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.strip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(dest))


def _upload_file_to_adls(local_path: Path, remote_path: str) -> UploadedFile:
    from io_utils import fsutil

    fsutil.makedirs(dirname_uri(remote_path), exist_ok=True)
    fsutil.copy_file(str(local_path),
                     remote_path,
                     overwrite=True,
                     raise_on_exists=False)
    return UploadedFile(
        local_path=local_path,
        remote_path=remote_path,
        size_bytes=local_path.stat().st_size,
        sha256=file_sha256(local_path),
    )


def _download_file_from_adls(uri: str, dest: Path) -> None:
    from io_utils import fsutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    fsutil.copy_file(uri, str(dest), overwrite=True, raise_on_exists=False)


def _upload_directory_to_adls(local_dir: Path, remote_root: str,
                              overwrite: bool) -> list[UploadedFile]:
    from io_utils import fsutil

    if fsutil.exists(remote_root):
        if not overwrite:
            raise FileExistsError(f"{remote_root} already exists; pass --overwrite")
        fsutil.rm(remote_root, ignore_errors=True)
    fsutil.makedirs(remote_root, exist_ok=True)

    uploaded: list[UploadedFile] = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        uploaded.append(_upload_file_to_adls(path, join_uri(remote_root, rel)))
    return uploaded


def _upload_directory_to_s3(local_dir: Path, remote_root: str,
                            overwrite: bool) -> list[UploadedFile]:
    import boto3

    parsed = urlparse(remote_root)
    bucket = parsed.netloc
    prefix = parsed.path.strip("/")
    if not bucket:
        raise ValueError(f"Invalid S3 URI: {remote_root}")

    client = boto3.client("s3")
    if not overwrite:
        existing = client.list_objects_v2(Bucket=bucket,
                                          Prefix=prefix.rstrip("/") + "/",
                                          MaxKeys=1)
        if existing.get("KeyCount", 0):
            raise FileExistsError(f"{remote_root} already exists; pass --overwrite")

    uploaded: list[UploadedFile] = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        key = "/".join(part for part in (prefix, rel) if part)
        remote_path = f"s3://{bucket}/{key}"
        client.upload_file(str(path), bucket, key)
        uploaded.append(
            UploadedFile(
                local_path=path,
                remote_path=remote_path,
                size_bytes=path.stat().st_size,
                sha256=file_sha256(path),
            ))
    return uploaded


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def default_tmp_root() -> Path:
    return Path(os.environ.get("KUMO_E2E_LOCAL_BUILD_ROOT",
                               "/tmp/kumo-e2e-build"))
