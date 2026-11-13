# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    HttpResponseError,
    ResourceModifiedError,
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceResponseError,
)
from kumo_core.sources.errors import (
    PermanentSourceError,
    ReconnectRequiredError,
    SourceAccessDeniedError,
    TransientSourceError,
)


@contextmanager
def translated():
    """Classify Azure failures without exposing request metadata or tokens."""
    try:
        yield
    except MemoryError as error:
        error.add_note(
            'If this occurred during a data transfer, lowering '
            'KUMO_ADLS_TRANSFER_WORKERS and KUMO_SOURCE_COPY_WORKERS may help. '
            'Set them before importing the provider (restart an existing '
            'process). These limits do not cap total memory or address all '
            'allocation failures; projected reads also buffer their requested '
            'range.')
        raise
    except ClientAuthenticationError:
        raise ReconnectRequiredError('ADLS authentication failed') from None
    except ResourceNotFoundError:
        raise FileNotFoundError('ADLS path was not found') from None
    except ResourceModifiedError:
        raise TransientSourceError(
            'ADLS object changed; retry the read') from None
    except (ServiceRequestError, ServiceResponseError):
        raise TransientSourceError(
            'ADLS transport failed; retry the request') from None
    except HttpResponseError as error:
        status = error.status_code
        if status == 401:
            kind = ReconnectRequiredError
        elif status == 403:
            kind = SourceAccessDeniedError
        elif status in (408, 412, 429) or (status is not None
                                           and status >= 500):
            kind = TransientSourceError
        else:
            kind = PermanentSourceError
        raise kind('ADLS request failed') from None
    except AzureError:
        raise PermanentSourceError('ADLS request failed') from None
