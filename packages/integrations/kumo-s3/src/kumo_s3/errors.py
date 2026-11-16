# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from kumo_core.sources.errors import (
    PermanentSourceError,
    ReconnectRequiredError,
    SourceAccessDeniedError,
    SourceError,
    SourceNotFoundError,
    TransientSourceError,
)


@contextmanager
def translated(what: str, *, table: str | None = None) -> Iterator[None]:
    from botocore.exceptions import (
        BotoCoreError,
        ClientError,
        ConnectionError,
        HTTPClientError,
        NoCredentialsError,
        PartialCredentialsError,
        ProfileNotFound,
        TokenRetrievalError,
    )

    lapsed = (ProfileNotFound, NoCredentialsError, PartialCredentialsError,
              TokenRetrievalError)
    try:
        yield
    except SourceError:
        raise
    except lapsed as error:
        raise ReconnectRequiredError(f"{what}: {error}", table=table) from None
    except ClientError as error:
        code = error.response.get('Error', {}).get('Code', '')
        if code in {
                'AuthFailure', 'ExpiredToken', 'ExpiredTokenException',
                'InvalidAccessKeyId', 'InvalidClientTokenId', 'InvalidToken',
                'SignatureDoesNotMatch', 'TokenRefreshRequired',
                'UnrecognizedClientException'
        }:
            error_type = ReconnectRequiredError
        elif code in {'AccessDenied', 'AllAccessDisabled', 'Forbidden'}:
            error_type = SourceAccessDeniedError
        elif code in {
                'SlowDown', 'Throttling', 'ThrottlingException',
                'RequestLimitExceeded', 'RequestTimeout', 'InternalError',
                'ServiceUnavailable'
        }:
            error_type = TransientSourceError
        elif code in {'NoSuchKey', 'NoSuchBucket', 'NotFound'}:
            error_type = SourceNotFoundError
        else:
            status = error.response.get('ResponseMetadata',
                                        {}).get('HTTPStatusCode', code)
            error_type = {
                '401': ReconnectRequiredError,
                '403': SourceAccessDeniedError,
                '404': SourceNotFoundError,
                '429': TransientSourceError,
                '500': TransientSourceError,
                '502': TransientSourceError,
                '503': TransientSourceError,
                '504': TransientSourceError,
            }.get(str(status), PermanentSourceError)
        raise error_type(f"{what}: {error}", table=table) from None
    except (ConnectionError, HTTPClientError) as error:
        raise TransientSourceError(f"{what}: {error}", table=table) from None
    except BotoCoreError as error:
        raise PermanentSourceError(f"{what}: {error}", table=table) from None
