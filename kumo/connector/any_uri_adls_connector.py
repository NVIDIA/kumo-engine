# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.utils.registry import register
from kumoapi.adls import normalize_adls_source_uri

from io_utils.fsutil.uri import ADLSURI, ParseURI
from kumo.config import ConnectorConfig
from kumo.connector.any_uri_file_connector import AnyURIFileConnector
from kumo.utils import to_dataclass


class AnyURIADLSConnector(AnyURIFileConnector):
    def __init__(self, name: str):
        super().__init__(name, normalizer=self._normalize_source_uri,
                         provider_name='ADLS')

    def _normalize_source_uri(self, table_name: str) -> str:
        uri = normalize_adls_source_uri(table_name)
        expected = (self._credentials or {}).get('endpoint_suffix')
        parsed = ParseURI(uri)
        if not isinstance(parsed, ADLSURI):
            raise ValueError('Custom ADLS source requires an ADLS URI')
        if expected and parsed.endpoint_suffix != expected:
            raise ValueError(
                'ADLS source endpoint does not match the selected identity '
                'cloud')
        return uri


AnyURIADLSConnectorConfig = to_dataclass(AnyURIADLSConnector,
                                         base_cls=ConnectorConfig)
register(AnyURIADLSConnector, AnyURIADLSConnectorConfig, group='connector')
