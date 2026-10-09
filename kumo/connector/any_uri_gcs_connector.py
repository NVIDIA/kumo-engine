# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.utils.registry import register
from kumoapi.gcs import normalize_gcs_source_uri

from kumo.config import ConnectorConfig
from kumo.connector.any_uri_file_connector import AnyURIFileConnector
from kumo.utils import to_dataclass


class AnyURIGCSConnector(AnyURIFileConnector):
    def __init__(self, name: str):
        super().__init__(name, normalizer=normalize_gcs_source_uri,
                         provider_name='GCS')


AnyURIGCSConnectorConfig = to_dataclass(AnyURIGCSConnector,
                                        base_cls=ConnectorConfig)
register(AnyURIGCSConnector, AnyURIGCSConnectorConfig, group='connector')
