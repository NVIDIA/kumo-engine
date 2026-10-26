# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from urllib.parse import quote, unquote, urlsplit

from kumo_core.sources.object_store_backend import ObjectStoreSourceBackend
from kumo_core.sources.object_store_sessions import (
    ObjectStorePaths,
    ObjectStoreRoot,
    ObjectStoreSessions,
)
from kumo_core.sources.principals import IdentityRef
from kumo_core.sources.types import ReadMode
from kumoapi.adls import (
    normalize_adls_root_uri,
    normalize_adls_source_relative_path,
    normalize_adls_source_uri,
)
from kumoapi.data_source import (
    ADLSConnectorResourceConfig,
    ADLSRootedConnectorTarget,
    ConnectorType,
    DataSourceType,
)
from kumoapi.source_table import (
    ADLSRelativeSource,
    ADLSSourceTable,
    ADLSURISource,
)

from . import ADLS_NAMESPACE
from .reader import ADLSSourceReader
from .session import session_for
from .source_access import source_access


def canonical_uri(path: str) -> str:
    """Serialize an account-qualified file or directory path."""
    authority, _, key = path.partition('/')
    return normalize_adls_root_uri(
        f'abfss://{authority}/{quote(key, safe="/-._~!$&()*+,;=:@")}')


class ADLSSessions(ObjectStoreSessions):
    namespace = ADLS_NAMESPACE
    rooted = ADLSRootedConnectorTarget
    resource_config = ADLSConnectorResourceConfig
    paths = ObjectStorePaths(
        scheme='abfss',
        provider='ADLS',
        source_uri=normalize_adls_source_uri,
        destination_uri=normalize_adls_source_uri,
        destination_path=normalize_adls_source_relative_path,
    )

    def _root_for(self, connector):
        target = self._target_config(connector).target
        if not isinstance(target, self.rooted):
            return None
        authority = (f'{target.filesystem}@{target.storage_account}.'
                     f'{target.endpoint_suffix}')
        return ObjectStoreRoot(authority, target.prefix)

    @staticmethod
    def _contained_key(root, uri, *, purpose):
        parsed = urlsplit(normalize_adls_source_uri(uri))
        key = unquote(parsed.path).lstrip('/')
        prefix = root.prefix.rstrip('/')
        outside_prefix = (bool(prefix) and key != prefix
                          and not key.startswith(prefix + '/'))
        if parsed.netloc != root.bucket or outside_prefix:
            raise ValueError(f'ADLS {purpose} is outside the connector root')
        return key

    def _relative_coordinate(self, root, coordinate, *, purpose):
        if '://' in coordinate:
            key = self._contained_key(root, coordinate, purpose=purpose)
            prefix = root.prefix.rstrip('/')
            coordinate = key[len(prefix):].lstrip('/') if prefix else key
        return normalize_adls_source_relative_path(coordinate)

    def _placement(self, connector, table_name):
        root = self._root_for(connector)
        if root is None:
            return super()._placement(connector, table_name)
        table_name = self._relative_coordinate(root, table_name,
                                               purpose='source')
        return root.bucket, root.path, table_name, None

    def _source_access(self, connector, bucket, region):
        del region
        _, host = bucket.split('@', 1)
        account, suffix = host.split('.', 1)
        client = session_for(self._principal(connector), account, suffix)
        return source_access(client, account, suffix)

    def _over(self, connector, bucket, root, region):
        access = self._source_access(connector, bucket, region)
        return ADLSSourceReader(access.filesystem, root=root,
                                source_type=ConnectorType.ADLS,
                                fetch=access.fetch,
                                object_identity=access.object_identity)

    def _filesystem(self, connector, bucket, region):
        return self._source_access(connector, bucket, region).filesystem

    def _resolved(self, connector, coordinate):
        root = self._root_for(connector)
        if root is None:
            raise NotImplementedError(
                'Rootless ADLS connectors are source-only')
        relative = self._relative_coordinate(root, coordinate,
                                             purpose='destination')
        key = '/'.join(part for part in (root.prefix, relative) if part)
        return root.bucket, f'{root.bucket}/{key}', None


class ADLSSourceBackend(ADLSSessions, ObjectStoreSourceBackend):
    source_type = DataSourceType.ADLS
    relative_source = ADLSRelativeSource
    uri_source = ADLSURISource
    source_table = ADLSSourceTable

    def __init__(self, identity: Optional[IdentityRef] = None,
                 storage_namespace: Optional[str] = None,
                 mode: Optional[ReadMode] = None):
        super().__init__(identity, storage_namespace)
        self.mode = mode

    def get_download_urls(self, connector, table_name, project_id=None):
        del project_id
        reader, name = self._reader(connector, table_name)
        return [canonical_uri(path) for path in reader.object_paths(name)]
