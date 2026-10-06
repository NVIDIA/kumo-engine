# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import Sequence

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

ERROR_CODE = re.compile(r"^\[([A-Z][A-Z0-9_]*(?:\.[A-Z][A-Z0-9_]*)*)\](?:\s|$)")
REAUTHORIZE = ("invalid access token", "token expired", "expired token",
               "token is expired", "unauthenticated")
DENIED = ("PERMISSION_DENIED", "INSUFFICIENT_PRIVILEGES",
          "INSUFFICIENT_PERMISSIONS")
NOT_FOUND_CODE = "TABLE_OR_VIEW_NOT_FOUND"


class ArtifactValidationError(ValueError):
    """A refusal this package makes on its own inputs, before a single
    statement runs.

    A distinct type rather than a bare :class:`ValueError` so that a caller
    translating this refusal into its own vocabulary -- the hosted connector
    restates it as a sanitised ``ConnectorError`` -- can catch *this* and only
    this. The transport callbacks the caller supplies do real provider and
    filesystem work, and any of them can raise a plain
    ``ValueError`` for its own reasons that have nothing to do with this
    package's validation; catching the base class would relabel that failure
    -- which may carry SQL text, a local path, or an endpoint -- as one of
    this package's own refusals.

    Still a :class:`ValueError` so that a caller which only wants "the write
    was refused" needs no import.
    """


def summary(error: BaseException) -> str:
    return diagnostics.safe_summary(error, code=server_error_code(error))


def rules() -> Sequence[Rule]:
    """The SDK raises subclasses of :class:`OSError`, so they are matched
    before the network errors that class otherwise stands for."""
    from databricks.sdk.errors import (
        DatabricksError,
        PermissionDenied,
        Unauthenticated,
    )
    from databricks.sql.exc import Error, OperationalError

    return (
        (Unauthenticated, ReconnectRequiredError),
        (PermissionDenied, SourceAccessDeniedError),
        (DatabricksError, PermanentSourceError),
        (OperationalError, TransientSourceError),
        (Error, _lapsed_or_permanent),
        (OSError, TransientSourceError),
    )


translated = diagnostics.source_translator("databricks", rules=rules,
                                           summarize=summary)


def server_error_code(error: BaseException) -> str | None:
    from databricks.sql.exc import ServerOperationError

    if not isinstance(error, ServerOperationError):
        return None
    context = getattr(error, "context", None)
    if isinstance(context, dict):
        code = context.get("error_code")
        if isinstance(code, str):
            return code
    message = getattr(error, "message", None)
    if not isinstance(message, str):
        return None
    match = ERROR_CODE.match(message)
    return match.group(1) if match else None


def _lapsed_or_permanent(error: BaseException) -> type[SourceError]:
    """The driver's text is read to classify, never to report."""
    text = str(error).lower()
    if any(phrase in text for phrase in REAUTHORIZE):
        return ReconnectRequiredError
    if _is_denial(error):
        return SourceAccessDeniedError
    if is_not_found(error):
        return SourceNotFoundError
    return PermanentSourceError


def is_not_found(error: BaseException) -> bool:
    """The code alone, never the prose beside it: Databricks interpolates the
    real name into that prose, the same sentence fronts a missing schema,
    catalog and function too, and the code is the one part it does not
    localize.
    """
    code = server_error_code(error)
    if code is None:
        return False
    return code == NOT_FOUND_CODE or code.startswith(f"{NOT_FOUND_CODE}.")


def _is_denial(error: BaseException) -> bool:
    return any(marker in f"{server_error_code(error)} {error}".upper()
               for marker in DENIED)
