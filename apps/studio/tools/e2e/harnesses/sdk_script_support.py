# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
from pathlib import Path

from tools.e2e.lib.config import default_storage_root, storage_backend

DEFAULT_URL = "http://localhost:8888/api"
DEFAULT_API_KEY = "test:DISABLED"


def parse_args() -> argparse.Namespace:
    default_root = default_storage_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=os.getenv("KUMO_API_ENDPOINT") or os.getenv("KUMO_API_URL")
        or DEFAULT_URL,
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("KUMO_API_KEY") or os.getenv("KUMO_TEST_API_KEY")
        or DEFAULT_API_KEY,
    )
    parser.add_argument("--storage-root", default=default_root)
    parser.add_argument(
        "--output-root",
        default=os.getenv("KUMO_E2E_OUTPUT_ROOT")
        or default_root.rstrip("/") + "/outputs",
    )
    parser.add_argument("--suite-name", default="kumo-sdk")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument(
        "--variant",
        choices=("parquet_typed", "string_encoded"),
        default="parquet_typed",
    )
    parser.add_argument(
        "--string-physical-format",
        choices=("csv", "parquet"),
        default="csv",
    )
    parser.add_argument(
        "--stop-after",
        choices=("validate", "save", "train", "bp"),
        default="validate",
    )
    parser.add_argument("--training-job-id")
    parser.add_argument(
        "--local-output",
        action="store_true",
        help="Use SDK local-download batch prediction output instead of writing to an output connector.",
    )
    return parser.parse_args()


def dataset_root(args: argparse.Namespace, dataset: str) -> str:
    root = (
        f"{args.storage_root.rstrip('/')}/{args.suite_name}/{dataset}/"
        f"{args.dataset_version}/{args.variant}")
    if args.variant == "string_encoded":
        root = f"{root}/{args.string_physical_format}"
    return root


def join_uri(root: str, *parts: str) -> str:
    return root.rstrip("/") + "/" + "/".join(
        part.strip("/") for part in parts if part)


def storage_connector(root: str):
    import kumoai_studio as kumo

    backend = storage_backend(root)
    if backend == "adls":
        return kumo.ADLSConnector(root_dir=root)
    if backend == "s3":
        return kumo.S3Connector(root_dir=root)
    if backend == "gcs":
        return kumo.GCSConnector(root_dir=root)
    if backend == "local":
        return kumo.S3Connector(root_dir=root)
    raise ValueError(f"Unsupported SDK connector root: {root}")


def output_exists(output_root: str, output_table: str) -> bool:
    if storage_backend(output_root) == "local":
        return all(
            Path(output_root, f"{output_table}_{kind}").exists()
            for kind in ("predictions", "embeddings"))

    from io_utils import fsutil

    for kind in ("predictions", "embeddings"):
        path = join_uri(output_root, f"{output_table}_{kind}")
        if not fsutil.exists(path) or not bool(fsutil.listdir(path, max_items=1)):
            return False
    return True


def download_file(uri: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if uri.startswith("s3://"):
        from urllib.parse import urlparse

        import boto3

        parsed = urlparse(uri)
        boto3.client("s3").download_file(
            parsed.netloc, parsed.path.lstrip("/"), str(destination))
        return

    if uri.startswith(("abfss://", "abfs://", "gs://")):
        from io_utils import fsutil

        fsutil.copy_file(uri,
                         str(destination),
                         overwrite=True,
                         raise_on_exists=False)
        return

    destination.write_bytes(Path(uri.removeprefix("file://")).read_bytes())
