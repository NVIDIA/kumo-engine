# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Platform representations of portable DiskGraph ingest errors."""

from kumo_core.diskgraph.runner import \
    DiskgraphCancelledError as CoreDiskgraphCancelledError
from kumo_core.diskgraph.runner import \
    DiskgraphDataplaneNoSpaceError as CoreDiskgraphDataplaneNoSpaceError
from kumo_core.diskgraph.runner import DiskgraphIngestError
from kumo_core.diskgraph.runner import \
    DiskgraphNoSpaceError as CoreDiskgraphNoSpaceError
from kumo_core.diskgraph.runner import DiskgraphRetryExhaustedError
from kumo_core.diskgraph.runner import \
    NonRetryableDiskgraphError as CoreNonRetryableDiskgraphError

from common_utils.error import PublicException


class NonRetryableDiskgraphError(PublicException):
    """Terminal DiskGraph ingest failure."""


class DiskgraphCancelledError(PublicException):
    """DiskGraph ingest was cancelled and may resume on a retry."""

    non_retryable = False


class DiskgraphDataplaneNoSpaceError(NonRetryableDiskgraphError):
    """The separately configured dataplane mirror is full."""


class DiskgraphNoSpaceError(PublicException):
    """The pool worker's local SSD is full and may need re-placement."""

    non_retryable = False


def as_platform_diskgraph_exception(
    error: DiskgraphIngestError, ) -> PublicException:
    """Preserve the platform error contract for a portable ingest error."""
    if isinstance(error, CoreDiskgraphDataplaneNoSpaceError):
        error_type = DiskgraphDataplaneNoSpaceError
    elif isinstance(error, CoreDiskgraphCancelledError):
        error_type = DiskgraphCancelledError
    elif isinstance(error, CoreDiskgraphNoSpaceError):
        error_type = DiskgraphNoSpaceError
    elif isinstance(error, CoreNonRetryableDiskgraphError):
        error_type = NonRetryableDiskgraphError
    elif isinstance(error, DiskgraphRetryExhaustedError):
        error_type = NonRetryableDiskgraphError
    else:
        error_type = NonRetryableDiskgraphError
    platform_error = error_type(
        message=error.message,
        non_retryable=error.non_retryable,
    )
    # These fields are deliberately absent from PublicException.to_dict(), so
    # the untrusted subprocess tail never enters Temporal history or the API.
    # Keep them on the in-process exception for failure correlation instead of
    # silently dropping them at this adapter. They are not persisted here.
    platform_error.private_output_tail = error.private_output_tail
    platform_error.output_fingerprint = error.output_fingerprint
    return platform_error


__all__ = [
    "DiskgraphCancelledError",
    "DiskgraphDataplaneNoSpaceError",
    "DiskgraphNoSpaceError",
    "NonRetryableDiskgraphError",
    "as_platform_diskgraph_exception",
]
