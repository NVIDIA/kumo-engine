# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping

from fastapi import Request


async def preserves_s3_ca_certificate(request: Request) -> bool:
    """Return whether an S3 access-key rotation omitted its write-only CA."""
    payload = await request.json()
    if not isinstance(payload, Mapping):
        return False
    credentials = payload.get('credentials')
    return (isinstance(credentials, Mapping)
            and credentials.get('kind') == 's3_access_key'
            and 'ca_certificate' not in credentials)
