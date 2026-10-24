# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
from collections.abc import Callable
from pathlib import Path

import boto3
import pytest
from moto import mock_s3

from io_utils.fsutil import s3_filesystem


def _s3_credentials_configured() -> bool:
    """Return whether an explicit, locally detectable AWS auth is present.

    Test decorators execute while pytest imports modules, so they must not
    contact S3 or trigger botocore's IMDS/container credential discovery.
    Invalid, expired, or insufficient credentials are exercised by the
    integration test itself.
    """
    access_key = os.getenv('AWS_ACCESS_KEY_ID') or os.getenv('AWS_ACCESS_KEY')
    secret_key = (os.getenv('AWS_SECRET_ACCESS_KEY')
                  or os.getenv('AWS_SECRET_KEY'))
    if access_key and secret_key:
        return True

    if os.getenv('AWS_ROLE_ARN') and os.getenv('AWS_WEB_IDENTITY_TOKEN_FILE'):
        return True

    if any(
            os.getenv(name) for name in (
                'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI',
                'AWS_CONTAINER_CREDENTIALS_FULL_URI',
            )):
        return True

    credentials_file = os.getenv('AWS_SHARED_CREDENTIALS_FILE')
    if credentials_file:
        return Path(credentials_file).expanduser().is_file()

    return (Path.home() / '.aws' / 'credentials').is_file()


def withS3Credentials(func: Callable | None = None):
    credentials_missing = not _s3_credentials_configured()
    reason = "No AWS credentials found for S3"
    marker = pytest.mark.skipif(credentials_missing, reason=reason,
                                kumo_resource='aws')
    from kumo.testing._skip import requires_resource
    if func is None:
        return marker
    func = requires_resource('aws', func)
    if inspect.isgeneratorfunction(func):
        from kumo.testing._skip import skip_if
        return skip_if(credentials_missing, reason, func)
    return marker(func)


def withMockedS3Filesystem(func: Callable):
    def wrapper():
        # Save AWS environment variables
        access_id = os.environ.get('AWS_ACCESS_KEY_ID')
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        old_boto = s3_filesystem.threadsafe_boto
        old_ctx = s3_filesystem.s3_ctx

        try:
            with mock_s3():
                s3 = boto3.client('s3', region_name='us-east-1')
                s3_filesystem.threadsafe_boto = \
                    s3_filesystem.ThreadLocalBotoSession()
                s3_filesystem.s3_ctx = s3_filesystem.S3FileSystem.for_test()
                func(s3)
        finally:
            # Reset the environment variables back
            if access_id:
                os.environ['AWS_ACCESS_KEY_ID'] = access_id
            if secret_key:
                os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
            s3_filesystem.threadsafe_boto = old_boto
            s3_filesystem.s3_ctx = old_ctx

    return wrapper
