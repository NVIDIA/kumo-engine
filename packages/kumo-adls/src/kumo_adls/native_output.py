# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import threading
from urllib.parse import urlencode, urlsplit

from kumoapi.adls import (
    normalize_adls_endpoint_suffix,
    normalize_adls_filesystem,
    normalize_adls_storage_account,
)
from kumoapi.data_source import ADLSIdentityConfig
from pyarrow.fs import AzureFileSystem, FileSystem, FSSpecHandler

from .session import CLOUDS


def filesystem_for(principal, account, suffix, container):
    """Bind native output without changing or substituting an identity."""
    account = normalize_adls_storage_account(account)
    suffix = normalize_adls_endpoint_suffix(suffix)
    container = normalize_adls_filesystem(container)
    cloud, (_, _, authority) = next((cloud, values)
                                    for cloud, values in CLOUDS.items()
                                    if values[0] == suffix)
    if principal is not None:
        config = principal.config
        if not isinstance(config, ADLSIdentityConfig):
            raise ValueError('The selected identity is not an ADLS identity')
        if config.cloud != cloud:
            raise ValueError(
                'ADLS endpoint does not match the selected identity cloud')
        if (os.environ.get('AZURE_TENANT_ID') != config.tenant_id
                or os.environ.get('AZURE_CLIENT_ID') != config.client_id or
                not os.environ.get('AZURE_FEDERATED_TOKEN_FILE', '').strip()):
            raise NotImplementedError(
                'Native Arrow ADLS writes require AZURE_TENANT_ID and '
                'AZURE_CLIENT_ID to match the selected workload identity, '
                'and AZURE_FEDERATED_TOKEN_FILE to name its projected token. '
                'Configure these in the process environment before writing. '
                'Per-connector workload identities that differ from the '
                'process identity remain supported for reads only.')

    host = os.environ.get('AZURE_AUTHORITY_HOST',
                          'https://login.microsoftonline.com')
    parsed = urlsplit(host)
    if (parsed.scheme != 'https' or parsed.netloc.casefold() != authority
            or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
        raise ValueError(
            'Native Arrow ADLS writes require AZURE_AUTHORITY_HOST=' +
            f'https://{authority} for the selected cloud')
    options = dict(
        blob_storage_authority='.' + suffix.replace('dfs.', 'blob.', 1),
        dfs_storage_authority='.' + suffix)
    if principal is None:
        return AzureFileSystem(account_name=account, **options)

    # Arrow's workload credential reads environment configuration at creation;
    # unlike DefaultAzureCredential it cannot fall back to another principal.
    # The short URI form keeps account/container correct in sovereign clouds.
    query = urlencode(dict(credential_kind='workload_identity', **options))
    filesystem, _ = FileSystem.from_uri(
        f'abfss://{account}@{container}/?{query}')
    return filesystem


class NativeOutputHandler(FSSpecHandler):
    """Keep SDK metadata/reads and return native Arrow output streams."""
    def __init__(self, filesystem, output_filesystem=None, *,
                 output_factory=None):
        super().__init__(filesystem)
        self.output_filesystem = output_filesystem
        self.output_factory = output_factory
        self._output_lock = threading.Lock()

    def _output(self):
        with self._output_lock:
            if self.output_filesystem is None:
                if self.output_factory is None:
                    raise TypeError('Native output needs a filesystem factory')
                self.output_filesystem = self.output_factory()
        return self.output_filesystem

    def create_dir(self, path, recursive):
        self.fs._address(path)
        # Shared sinks prepare parent directories before opening output.
        # Reject unsupported credentials before that SDK mutation as well.
        self._output()
        super().create_dir(path, recursive)

    def open_output_stream(self, path, metadata):
        # Validate the virtual authority before dropping its account qualifier.
        self.fs._address(path)
        authority, _, key = path.strip('/').partition('/')
        container, _, _ = authority.partition('@')
        return self._output().open_output_stream(f'{container}/{key}',
                                                 metadata=metadata)
