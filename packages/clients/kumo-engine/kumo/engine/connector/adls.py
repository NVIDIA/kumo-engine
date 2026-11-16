# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local Azure Data Lake Storage Gen2 connector."""

from functools import cached_property
from typing import Optional
from urllib.parse import unquote, urlsplit

from kumoapi.adls import normalize_adls_root_uri
from kumoapi.data_source import (
    ADLSAnyURIConnectorTarget,
    ADLSConnectorResourceConfig,
    ADLSRootedConnectorTarget,
    DataSourceType,
)

from kumo.engine.connector._extras import import_provider
from kumo.engine.connector.base import Connector
from kumo.engine.connector.identity import ADLSIdentity


class ADLSConnector(Connector):
    """Read ADLS Gen2 files and write output beneath a configured root.

    Install ``kumo-engine[adls]``. Without an identity, Azure's ambient
    credential chain signs requests. An explicit identity uses workload
    federation. Rootless connectors accept full ``abfss://`` source URIs and
    cannot discover tables or write output.

    Args:
        root_dir: Optional filesystem/prefix URI, for example
            ``abfss://curated@companylake.dfs.core.windows.net/orders/``.
        identity: An ADLS workload identity; None selects ambient credentials.
    """
    def __init__(self, *, root_dir: Optional[str] = None,
                 identity: Optional[ADLSIdentity] = None) -> None:
        super().__init__(identity=identity)
        if root_dir is None:
            target = ADLSAnyURIConnectorTarget()
        else:
            parsed = urlsplit(normalize_adls_root_uri(root_dir))
            filesystem, host = parsed.netloc.split('@', 1)
            storage_account, suffix = host.split('.', 1)
            target = ADLSRootedConnectorTarget(
                storage_account=storage_account, filesystem=filesystem,
                prefix=unquote(parsed.path).strip('/'), endpoint_suffix=suffix)
        self._config = ADLSConnectorResourceConfig(target=target)
        self.target = target
        self.any_uri = isinstance(target, ADLSAnyURIConnectorTarget)
        self.storage_account = getattr(target, 'storage_account', None)
        self.filesystem = getattr(target, 'filesystem', None)
        self.prefix = getattr(target, 'prefix', None)
        self.endpoint_suffix = getattr(target, 'endpoint_suffix', None)

    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.ADLS

    @property
    def config(self) -> ADLSConnectorResourceConfig:
        return self._config

    @cached_property
    def source_backend(self):
        backend = import_provider('kumo_adls.backend', 'adls').ADLSSourceBackend
        return backend(identity=self.identity)

    @cached_property
    def sink_backend(self):
        backend = import_provider('kumo_adls.sink', 'adls').ADLSSinkBackend
        return backend(identity=self.identity)

    def source_table_for(self, table_name: str):
        return self.source_backend.get_table_config(self,
                                                    table_name).source_table

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}(storage_account={self.storage_account!r}, '
            f'filesystem={self.filesystem!r}, prefix={self.prefix!r})')
