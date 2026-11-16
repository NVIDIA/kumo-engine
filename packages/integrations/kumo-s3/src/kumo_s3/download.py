# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from boto3.s3.transfer import TransferConfig
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

logger = logging.getLogger(__name__)


class DownloadClient(Protocol):
    # These keyword names must match boto3's structural interface.
    def download_file(self, Bucket: str, Key: str, Filename: str, *,  # noqa: F841
                      Config: TransferConfig | None = None) -> None:
        ...


def download_file(bucket: str, key: str, target: str, *,
                  client: Callable[[], DownloadClient],
                  unsigned_client: Callable[[], DownloadClient] | None = None,
                  transfer_config: TransferConfig | None = None) -> None:
    options = {} if transfer_config is None else {'Config': transfer_config}
    try:
        client().download_file(bucket, key, target, **options)
    except (NoCredentialsError, PartialCredentialsError):
        if unsigned_client is None:
            raise
        logger.info(
            'No AWS credentials found; retrying S3 download anonymously')
        unsigned_client().download_file(bucket, key, target, **options)
