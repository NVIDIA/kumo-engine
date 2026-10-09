# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import Iterator

from google.api_core import exceptions
from google.auth.exceptions import RefreshError, TransportError
from google.resumable_media.common import DataCorruption
from kumo_core.sources.errors import (
    PermanentSourceError,
    ReconnectRequiredError,
    SourceAccessDeniedError,
    SourceNotFoundError,
    TransientSourceError,
)
from kumo_core.sources.object_store import classify_arrow_error


@contextmanager
def translated(source: str) -> Iterator[None]:
    try:
        yield
    except (exceptions.Unauthorized, RefreshError):
        raise ReconnectRequiredError('GCS authentication failed',
                                     table=source) from None
    except exceptions.Forbidden:
        raise SourceAccessDeniedError('GCS access was denied',
                                      table=source) from None
    except exceptions.NotFound:
        raise SourceNotFoundError('GCS object was not found',
                                  table=source) from None
    except ConnectionError as error:
        classified = classify_arrow_error(source, 'object', error.__cause__
                                          or error)
        error_type = type(classified)
        if error_type is PermanentSourceError:
            error_type = TransientSourceError
        raise error_type('GCS read failed', table=source) from None
    except (exceptions.ServerError, exceptions.TooManyRequests,
            exceptions.RetryError, TransportError, DataCorruption):
        raise TransientSourceError('GCS read failed; retry the read',
                                   table=source) from None
    except exceptions.GoogleAPICallError:
        raise PermanentSourceError('GCS request failed', table=source) from None
