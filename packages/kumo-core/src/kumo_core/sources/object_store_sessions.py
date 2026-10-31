# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import posixpath
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple
from urllib.parse import quote, unquote, urlsplit

from kumoapi.plugin import SourceConnector
from pyarrow.fs import FileSystem

from .layout import within_root
from .object_store import ObjectStoreSourceReader
from .object_store_access import ObjectStoreAccess
from .principals import IdentityRef, Principal, resolve

_KEY_SAFE = "/-._~!$&'()*+,;=:@"


@dataclass(frozen=True)
class ObjectStoreRoot:
    bucket: str
    prefix: str = ''
    region: Optional[str] = None

    @property
    def path(self) -> str:
        return f'{self.bucket}/{self.prefix}' if self.prefix else self.bucket


@dataclass(frozen=True)
class ObjectStorePaths:
    scheme: str
    provider: str
    source_uri: Callable[[str], str]
    destination_uri: Callable[[str], str]
    destination_path: Callable[[str], str]


class ObjectStoreSessions(ABC):
    paths: ObjectStorePaths
    namespace: str
    rooted: type[Any]
    resource_config: type[Any]

    def __init__(self, identity: Optional[IdentityRef] = None,
                 storage_namespace: Optional[str] = None) -> None:
        self._identity = identity
        self.storage_namespace = storage_namespace or self.namespace

    @property
    def scheme(self) -> str:
        return self.paths.scheme

    @property
    def provider(self) -> str:
        return self.paths.provider

    def _target_config(self, connector: SourceConnector) -> Any:
        config = connector.config
        if not isinstance(config, self.resource_config):
            raise ValueError(
                f"Connector {connector.id!r} points at a "
                f"{config.type.value} target, so it cannot be read as "
                f"{self.provider}.")
        return config

    def _root_for(self,
                  connector: SourceConnector) -> Optional[ObjectStoreRoot]:
        target = self._target_config(connector).target
        if not isinstance(target, self.rooted):
            return None
        return ObjectStoreRoot(target.bucket, target.prefix,
                               getattr(target, 'region', None))

    @abstractmethod
    def _filesystem(self, connector: SourceConnector, bucket: str,
                    region: Optional[str]) -> FileSystem:
        raise NotImplementedError

    def _source_access(self, connector: SourceConnector, bucket: str,
                       region: Optional[str]) -> ObjectStoreAccess:
        return ObjectStoreAccess(self._filesystem(connector, bucket, region))

    def _reader(self, connector: SourceConnector,
                table_name: str) -> Tuple[ObjectStoreSourceReader, str]:
        bucket, root, name, region = self._placement(connector, table_name)
        return self._over(connector, bucket, root, region), name

    def _discovery(self, connector: SourceConnector) -> ObjectStoreSourceReader:
        root = self._root_for(connector)
        if root is None:
            raise NotImplementedError(
                f"Any-URI {self.provider} connectors cannot list tables; they "
                f"have no discovery root. Pass a full '{self.scheme}://' URI "
                f"to table(), or bind a rooted connector with "
                f"{self.provider}Connector(root_dir='{self.scheme}://...').")
        return self._over(connector, root.bucket, root.path, root.region)

    def _over(self, connector: SourceConnector, bucket: str, root: str,
              region: Optional[str]) -> ObjectStoreSourceReader:
        access = self._source_access(connector, bucket, region)
        return ObjectStoreSourceReader(access.filesystem, root=root,
                                       fetch=access.fetch,
                                       object_identity=access.object_identity)

    def table_location(self, connector: SourceConnector,
                       table_name: str) -> str:
        """Full ``scheme://bucket/key`` of ``table_name``, without credentials."""
        root = self._root_for(connector)
        if root is None:
            return self.paths.source_uri(table_name).rstrip('/')
        key = quote(posixpath.join(root.prefix, table_name), safe=_KEY_SAFE)
        return f'{self.scheme}://{root.bucket}/{key}'.rstrip('/')

    def _placement(self, connector: SourceConnector,
                   table_name: str) -> Tuple[str, str, str, Optional[str]]:
        root = self._root_for(connector)
        if root is not None:
            return root.bucket, root.path, table_name, root.region
        try:
            canonical = self.paths.source_uri(table_name)
        except ValueError as error:
            # The raw URI may contain credentials; report only the normalizer's error.
            raise ValueError(
                f"Connector {connector.id!r} is any-URI, so every table "
                f"is named by a full '{self.scheme}://bucket/key' URI: "
                f"{error}") from error
        parsed = urlsplit(canonical)
        return (parsed.netloc, parsed.netloc, unquote(parsed.path).lstrip('/'),
                None)

    def _resolved(self, connector: SourceConnector,
                  coordinate: str) -> Tuple[str, str, Optional[str]]:
        target = self._root_for(connector)
        root = f'{self.scheme}://{target.path}' if target is not None else ''
        # Decode before fencing: a/%2e%2e/x and a/../x normalize to the same
        # destination under our URI policy.
        # Checking only the encoded spelling would allow an escape from root.
        parsed = urlsplit(coordinate)
        if parsed.scheme or parsed.netloc:
            parsed = urlsplit(self.paths.destination_uri(coordinate))
            key = unquote(parsed.path).lstrip('/')
        elif target is None:
            raise ValueError('Rootless connectors require a full destination')
        else:
            relative = self.paths.destination_path(coordinate)
            parsed = urlsplit(root)
            key = posixpath.join(parsed.path.lstrip('/'), relative)
        if target is not None:
            within_root(f'{self.scheme}://{parsed.netloc}/{key}', root)
        return (parsed.netloc, f'{parsed.netloc}/{key}',
                target.region if target is not None else None)

    def _destination(self, connector: SourceConnector,
                     coordinate: str) -> Tuple[FileSystem, str]:
        bucket, path, region = self._resolved(connector, coordinate)
        return self._filesystem(connector, bucket, region), path

    def _principal(self, connector: SourceConnector) -> Optional[Principal]:
        return resolve(
            (getattr(connector, 'identity', None),
             getattr(connector, 'identity_id', None), self._identity),
            connector=connector,
            config_example=f'{self.provider}IdentityConfig(...)',
            fallback=('Leaving the identity unset is also supported, and '
                      'signs with the ambient credential chain.'),
            optional=True,
        )
