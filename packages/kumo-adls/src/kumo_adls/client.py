# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credential-bound SDK clients; authentication selection belongs to callers."""

from urllib.parse import urlsplit

from azure.core.pipeline.transport import RequestsTransport
from azure.storage.filedatalake import DataLakeServiceClient
from kumoapi.adls import (
    normalize_adls_endpoint_suffix,
    normalize_adls_storage_account,
)
from requests import Session
from requests.adapters import HTTPAdapter

from .clouds import CLOUDS
from .transfer import (
    BULK_DOWNLOAD_CHUNK_BYTES,
    CONNECTIONS,
    INITIAL_GET_BYTES,
    NETWORK_BLOCK_BYTES,
)


def create_client(
        credential, account: str,
        suffix: str = 'dfs.core.windows.net') -> DataLakeServiceClient:
    """Create a caller-owned client/transport, borrowing the supplied credential.

    Closing this client closes its transport, not the credential. No ambient
    identity is selected; callers must supply a credential, including for SAS.
    """
    account = normalize_adls_storage_account(account)
    suffix = normalize_adls_endpoint_suffix(suffix)
    if credential is None:
        raise TypeError(
            'ADLS client construction requires a resolved credential')
    audience = next(values[1] for values in CLOUDS.values()
                    if values[0] == suffix)
    transport_session = Session()
    try:
        adapter = HTTPAdapter(pool_connections=CONNECTIONS,
                              pool_maxsize=CONNECTIONS, pool_block=True)
        transport_session.mount('https://', adapter)
        transport_session.mount('http://', adapter)
        transport = RequestsTransport(
            session=transport_session, session_owner=True,
            connection_data_block_size=NETWORK_BLOCK_BYTES)
    except BaseException:
        transport_session.close()
        raise
    try:
        return DataLakeServiceClient(
            account_url=f'https://{account}.{suffix}', credential=credential,
            audience=audience, max_single_get_size=INITIAL_GET_BYTES,
            max_chunk_get_size=BULK_DOWNLOAD_CHUNK_BYTES,
            connection_data_block_size=NETWORK_BLOCK_BYTES, transport=transport)
    except BaseException:
        transport.close()
        raise


def validate_client(client, account: str, suffix: str) -> tuple[str, str]:
    """Validate a borrowed service client without I/O or credential disclosure."""
    account = normalize_adls_storage_account(account)
    suffix = normalize_adls_endpoint_suffix(suffix)
    message = 'ADLS client does not match the requested account and HTTPS endpoint'
    url = getattr(client, 'url', None)
    if not isinstance(url, str) or any(
            ord(c) <= 32 or ord(c) == 127 for c in url):
        raise ValueError(message)
    try:
        parsed = urlsplit(url)
    except ValueError:
        raise ValueError(message) from None
    # Query parameters can hold SAS credentials; neither inspect nor echo them.
    if (parsed.scheme != 'https'
            or parsed.netloc.casefold() != f'{account}.{suffix}'
            or parsed.path not in ('', '/') or parsed.fragment
            or getattr(client, 'account_name', None) != account):
        raise ValueError(message)
    return account, suffix
