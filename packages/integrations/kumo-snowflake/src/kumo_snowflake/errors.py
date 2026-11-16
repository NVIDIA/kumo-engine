# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Sequence

from kumo_core.sources import diagnostics
from kumo_core.sources.diagnostics import Rule
from kumo_core.sources.errors import (
    PermanentSourceError,
    ReconnectRequiredError,
    SourceAccessDeniedError,
    SourceError,
    TransientSourceError,
)

LAPSED_CREDENTIAL_SQLSTATE = "28"
LAPSED_CREDENTIAL_ERRNO = 390114
ACCESS_DENIED_SQLSTATE = "42501"
INTERNAL_STAGE_UNLOAD_DENIED_ERRNO = 3191
INTERNAL_STAGE_UNLOAD_DENIED_SQLSTATE = "42000"
INTERNAL_STAGE_UNLOAD_DENIED_MESSAGE = (
    "Snowflake account policy PREVENT_UNLOAD_TO_INTERNAL_STAGES blocks "
    "the required internal-stage unload")


def summary(error: BaseException) -> str:
    errno = getattr(error, "errno", None)
    sqlstate = str(getattr(error, "sqlstate", "") or "")
    safe = diagnostics.safe_summary(error, errno=errno, sqlstate=sqlstate,
                                    query=getattr(error, "sfqid", None))
    if (errno == INTERNAL_STAGE_UNLOAD_DENIED_ERRNO
            and sqlstate == INTERNAL_STAGE_UNLOAD_DENIED_SQLSTATE):
        return f"{safe}: {INTERNAL_STAGE_UNLOAD_DENIED_MESSAGE}"
    return safe


def is_lapsed_credential(error: BaseException) -> bool:
    """Whether reconnecting can recover a Snowflake authentication error."""
    return (str(getattr(error, "sqlstate", "")
                or "").startswith(LAPSED_CREDENTIAL_SQLSTATE)
            or getattr(error, "errno", None) == LAPSED_CREDENTIAL_ERRNO)


def rules() -> Sequence[Rule]:
    """Without the closing ``Error`` clause a ``NotSupportedError`` would
    escape unclassified."""
    from snowflake.connector.errors import (
        BadGatewayError,
        Error,
        GatewayTimeoutError,
        InterfaceError,
        InternalServerError,
        OperationalError,
        OtherHTTPRetryableError,
        ProgrammingError,
        RefreshTokenError,
        RequestExceedMaxRetryError,
        RequestTimeoutError,
        ServiceUnavailableError,
    )

    retryable = (OperationalError, InterfaceError, ServiceUnavailableError,
                 InternalServerError, BadGatewayError, GatewayTimeoutError,
                 RequestTimeoutError, OtherHTTPRetryableError,
                 RequestExceedMaxRetryError, OSError)
    return (
        (RefreshTokenError, ReconnectRequiredError),
        (retryable, TransientSourceError),
        (ProgrammingError, _by_sqlstate),
        (Error, PermanentSourceError),
    )


translated = diagnostics.source_translator("snowflake", rules=rules,
                                           summarize=summary)


def _by_sqlstate(error: BaseException) -> type[SourceError]:
    """Snowflake collapses a missing object and an ungranted one into one
    object-resolution code, so only ``42501`` names a grant truly missing."""
    sqlstate = str(getattr(error, "sqlstate", "") or "")
    if is_lapsed_credential(error):
        return ReconnectRequiredError
    if sqlstate == ACCESS_DENIED_SQLSTATE:
        return SourceAccessDeniedError
    return PermanentSourceError
