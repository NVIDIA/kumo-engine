# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
from contextlib import suppress
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.config import Config
from kumo_core.sources.errors import ReconnectRequiredError
from kumo_core.sources.object_store_access import ObjectStoreAccess
from kumo_s3.download import download_file
from kumo_s3.session import anonymous_filesystem


def source_access(
    bucket: str,
    *,
    session: Any,
    endpoint_url: str | None = None,
    default_region: str | None = None,
    options: Mapping[str, Any] | None = None,
    allow_anonymous: bool = False,
    client_config: Config | None = None,
    verify: bool | str | None = None,
) -> ObjectStoreAccess:
    import kumo_s3

    options = {
        key: value
        for key, value in (options or {}).items() if key not in
        {'access_key', 'secret_key', 'session_token', 'region', 'anonymous'}
    }
    if endpoint_url:
        options.setdefault('endpoint_override', endpoint_url)
        options.setdefault('force_virtual_addressing', True)
    endpoint = options.get('endpoint_override')
    if endpoint:
        endpoint = str(endpoint)
        if '://' not in endpoint:
            endpoint = f"{options.get('scheme', 'https')}://{endpoint}"
    filesystem = kumo_s3.filesystem(bucket, session=session,
                                    endpoint_url=endpoint,
                                    default_region=default_region,
                                    options=options)
    anonymous = filesystem is None
    if anonymous:
        if not allow_anonymous:
            raise ReconnectRequiredError(
                'The selected S3 identity has no usable credentials.',
                table=bucket)
        filesystem = anonymous_filesystem(
            bucket, region=default_region or session.region_name,
            options=options)
    region = filesystem.region
    local = threading.local()
    client_lock = threading.Lock()
    transfer = TransferConfig(use_threads=False,
                              preferred_transfer_client='classic')

    def client(unsigned: bool):
        name = 'unsigned' if unsigned else 'signed'
        existing = getattr(local, name, None)
        if existing is None:
            config = client_config or Config()
            if unsigned:
                config = config.merge(Config(signature_version=UNSIGNED))
            if client_config is None:
                addressing = None
                if 'force_virtual_addressing' in options:
                    addressing = ('virtual'
                                  if options['force_virtual_addressing'] else
                                  'path')
                elif endpoint:
                    addressing = 'virtual'
                if addressing is not None:
                    config = config.merge(
                        Config(s3={'addressing_style': addressing}))
            with client_lock:
                existing = session.client('s3', region_name=region,
                                          endpoint_url=endpoint, config=config,
                                          verify=verify)
            setattr(local, name, existing)
        return existing

    def fetch(path: str, target: str) -> None:
        source_bucket, separator, key = path.partition('/')
        if source_bucket != bucket or not separator or not key:
            raise ValueError('S3 source path is outside the selected bucket.')
        with NamedTemporaryFile(dir=os.path.dirname(os.path.abspath(target)),
                                prefix='.kumo-s3-', delete=False) as temporary:
            staging = temporary.name
        try:
            with kumo_s3.translated('Downloading S3 source', table=key):
                download_file(
                    bucket, key, staging, client=lambda: client(anonymous),
                    unsigned_client=(lambda: client(True)) if allow_anonymous
                    and not anonymous else None, transfer_config=transfer)
            os.replace(staging, target)
        finally:
            with suppress(OSError):
                os.unlink(staging)

    return ObjectStoreAccess(filesystem, fetch)
