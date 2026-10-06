# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from kumo_core.sources import diagnostics
from kumo_core.sources.diagnostics import Rule
from kumo_core.sources.errors import (
    PermanentSourceError,
    ReconnectRequiredError,
    SourceAccessDeniedError,
    SourceError,
    SourceNotFoundError,
    TransientSourceError,
)


def _reason(error: BaseException) -> str | None:
    direct = getattr(error, "reason", None)
    if direct:
        return str(direct)
    for detail in getattr(error, "errors", ()) or ():
        if isinstance(detail, dict) and detail.get("reason"):
            return str(detail["reason"])
    return None


def summary(error: BaseException) -> str:
    return diagnostics.safe_summary(error, code=getattr(error, "code", None),
                                    reason=_reason(error))


def _forbidden(error: BaseException) -> type[SourceError]:
    if _reason(error) == "responseTooLarge":
        return PermanentSourceError
    return SourceAccessDeniedError


def rules() -> Sequence[Rule]:
    """A corrupted download and a transport failure both descend from
    ``Exception`` alone, so neither is classified unless it is named."""
    from google.api_core import exceptions
    from google.auth.exceptions import RefreshError, TransportError
    from google.resumable_media.common import DataCorruption

    return (
        ((exceptions.Unauthorized, RefreshError), ReconnectRequiredError),
        (exceptions.Forbidden, _forbidden),
        ((exceptions.ServerError, exceptions.TooManyRequests,
          exceptions.RetryError, DataCorruption, TransportError, OSError),
         TransientSourceError),
        (exceptions.GoogleAPICallError, PermanentSourceError),
    )


@contextmanager
def resolving(table: str) -> Iterator[None]:
    """Scoped to the lookup rather than added to :func:`rules`, because every
    object Google's clients reach for raises the same ``NotFound`` -- a GCS
    object a staged read listed moments earlier is a failed download, not a
    table that does not exist. Raising here also puts this *before*
    ``(GoogleAPICallError, PermanentSourceError)``, which would otherwise
    report a wrong name as a warehouse failure.
    """
    from google.api_core import exceptions

    try:
        yield
    except exceptions.NotFound as error:
        raise SourceNotFoundError(f"bigquery has no {table}: {summary(error)}",
                                  table=table) from None


translated = diagnostics.source_translator("bigquery", rules=rules,
                                           summarize=summary)
