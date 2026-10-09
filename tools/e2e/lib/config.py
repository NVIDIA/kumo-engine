# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from urllib.error import URLError
from urllib.request import Request, urlopen

STORAGE_SUFFIX = "test/on_demand_e2e/generated_data"
DEFAULT_ADLS_ROOT = (
    "abfss://test-container@kumotestdataplane.dfs.core.windows.net/"
    f"devserver2/{STORAGE_SUFFIX}")
DEFAULT_S3_ROOT = (
    "s3://kumocloud-test-bucket/devserver2/test/on_demand_e2e/generated_data")
DEFAULT_GCS_ROOT = "gs://kumo-test-bucket/devserver2/test/on_demand_e2e/generated_data"


def default_storage_root() -> str:
    for key in (
            "KUMO_E2E_STORAGE_ROOT",
            "KUMO_E2E_ADLS_STORAGE_ROOT",
            "KUMO_E2E_S3_STORAGE_ROOT",
            "KUMO_E2E_GCS_STORAGE_ROOT",
    ):
        value = os.getenv(key)
        if value:
            return value.rstrip("/")

    cloud = detect_cloud_from_metadata()
    if cloud == "azure":
        return DEFAULT_ADLS_ROOT
    if cloud == "aws":
        return DEFAULT_S3_ROOT
    if cloud == "gcp":
        return DEFAULT_GCS_ROOT

    return DEFAULT_ADLS_ROOT


def detect_cloud_from_metadata(timeout: float = 0.25) -> str | None:
    if _metadata_request(
            "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
            headers={"Metadata": "true"},
            timeout=timeout,
    ):
        return "azure"
    if _aws_metadata_available(timeout):
        return "aws"
    if _metadata_request(
            "http://metadata.google.internal/computeMetadata/v1/instance/id",
            headers={"Metadata-Flavor": "Google"},
            timeout=timeout,
    ):
        return "gcp"
    return None


def _metadata_request(url: str,
                      *,
                      headers: dict[str, str] | None = None,
                      method: str = "GET",
                      timeout: float = 0.25) -> bool:
    request = Request(url, headers=headers or {}, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (OSError, TimeoutError, URLError):
        return False


def _aws_metadata_available(timeout: float) -> bool:
    token_request = Request(
        "http://169.254.169.254/latest/api/token",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        method="PUT",
    )
    try:
        with urlopen(token_request, timeout=timeout) as response:
            token = response.read().decode("utf-8")
    except (OSError, TimeoutError, URLError):
        token = ""

    headers = {"X-aws-ec2-metadata-token": token} if token else {}
    return _metadata_request(
        "http://169.254.169.254/latest/meta-data/instance-id",
        headers=headers,
        timeout=timeout,
    )


def storage_backend(root: str) -> str:
    if root.startswith(("abfss://", "abfs://")):
        return "adls"
    if root.startswith("s3://"):
        return "s3"
    if root.startswith("gs://"):
        return "gcs"
    return "local"
