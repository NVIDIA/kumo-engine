# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
import threading

from azure.core.pipeline.transport import RequestsTransport
from azure.identity import (
    AzureAuthorityHosts,
    DefaultAzureCredential,
    WorkloadIdentityCredential,
)
from azure.storage.filedatalake import DataLakeServiceClient
from kumo_core.sources.principals import Principal
from kumoapi.data_source import ADLSIdentityConfig, AzureCloud
from requests import Session
from requests.adapters import HTTPAdapter

from .errors import translated
from .transfer import (
    BULK_DOWNLOAD_CHUNK_BYTES,
    CONNECTIONS,
    INITIAL_GET_BYTES,
    NETWORK_BLOCK_BYTES,
)

# Storage uses the same generic token audience in every cloud.
# DFS endpoints and Entra authorities remain cloud-specific.
CLOUDS = {
    AzureCloud.PUBLIC: ('dfs.core.windows.net', 'https://storage.azure.com',
                        AzureAuthorityHosts.AZURE_PUBLIC_CLOUD),
    AzureCloud.US_GOVERNMENT:
    ('dfs.core.usgovcloudapi.net', 'https://storage.azure.com',
     AzureAuthorityHosts.AZURE_GOVERNMENT),
    AzureCloud.CHINA: ('dfs.core.chinacloudapi.cn', 'https://storage.azure.com',
                       AzureAuthorityHosts.AZURE_CHINA),
}


def credential_for(principal: Principal | None, endpoint_suffix: str):
    cloud = next(
        (cloud
         for cloud, values in CLOUDS.items() if values[0] == endpoint_suffix),
        None)
    if cloud is None:
        raise ValueError('Unsupported ADLS endpoint suffix')
    _, audience, authority = CLOUDS[cloud]
    if principal is None:
        return DefaultAzureCredential(authority=authority), audience
    config = principal.config
    if not isinstance(config, ADLSIdentityConfig):
        raise ValueError('The selected identity is not an ADLS identity')
    if config.cloud != cloud:
        raise ValueError(
            'ADLS endpoint does not match the selected identity cloud')
    # Explicit identities must never fall back to another ambient principal.
    return WorkloadIdentityCredential(tenant_id=config.tenant_id,
                                      client_id=config.client_id,
                                      authority=authority), audience


_LOCK = threading.RLock()
_SESSIONS = {}


def session_for(principal: Principal | None, account: str, suffix: str):
    # Config participates in the key so a rotated identity cannot reuse a
    # client signed as its predecessor. Tokens refresh inside Azure Identity.
    key = (principal.name if principal else None,
           repr(principal.config) if principal else None, account, suffix)
    with _LOCK:
        if key not in _SESSIONS:
            with translated():
                credential, audience = credential_for(principal, suffix)
                transport_session = Session()
                adapter = HTTPAdapter(pool_connections=CONNECTIONS,
                                      pool_maxsize=CONNECTIONS, pool_block=True)
                transport_session.mount('https://', adapter)
                transport_session.mount('http://', adapter)
                transport = RequestsTransport(
                    session=transport_session, session_owner=True,
                    connection_data_block_size=NETWORK_BLOCK_BYTES)
                try:
                    client = DataLakeServiceClient(
                        account_url=f'https://{account}.{suffix}',
                        credential=credential, audience=audience,
                        max_single_get_size=INITIAL_GET_BYTES,
                        max_chunk_get_size=BULK_DOWNLOAD_CHUNK_BYTES,
                        connection_data_block_size=NETWORK_BLOCK_BYTES,
                        transport=transport)
                except Exception:
                    transport.close()
                    credential.close()
                    raise
                _SESSIONS[key] = client, credential
        return _SESSIONS[key][0]


def close(identity_name: str | None = None) -> None:
    """Drop a principal's clients, or every session during process shutdown."""
    with _LOCK:
        keys = [
            key for key in _SESSIONS
            if identity_name is None or key[0] == identity_name
        ]
        sessions = [_SESSIONS.pop(key) for key in keys]
    for client, credential in sessions:
        for resource in (client, credential):
            try:
                resource.close()
            except Exception:
                # Teardown must not prevent closing the remaining sessions
                # or mask the caller's original authentication failure.
                pass


atexit.register(close)


def probe(principal: Principal) -> None:
    """Validate token acquisition; this does not attest storage permissions."""
    config = principal.config
    if not isinstance(config, ADLSIdentityConfig):
        raise ValueError('The selected identity is not an ADLS identity')
    with translated():
        credential, audience = credential_for(principal,
                                              CLOUDS[config.cloud][0])
        with credential:
            credential.get_token(f'{audience}/.default')
