# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.utils.registry import register
from kumoapi.s3 import normalize_s3_source_uri

from kumo.config import ConnectorConfig
from kumo.connector.any_uri_file_connector import AnyURIFileConnector
from kumo.utils import to_dataclass


class AnyURIS3Connector(AnyURIFileConnector):
    def __init__(self, name: str):
        super().__init__(name, normalizer=normalize_s3_source_uri,
                         provider_name='S3')


AnyURIS3ConnectorConfig = to_dataclass(AnyURIS3Connector,
                                       base_cls=ConnectorConfig)
register(AnyURIS3Connector, AnyURIS3ConnectorConfig, group='connector')
