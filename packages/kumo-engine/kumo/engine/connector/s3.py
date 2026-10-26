# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""`S3Connector`: a connector with no server-side registry state. See
`kumo.engine.connector.base`.

TODO(local-sdk): `root_dir` is parsed with a bare `urlsplit` and does not yet
reject ambiguous/percent-encoded path separators.
"""

from functools import cached_property
from typing import Optional
from urllib.parse import urlsplit

from kumoapi.data_source import (
    DataSourceType,
    S3AnyURIConnectorTarget,
    S3ConnectorResourceConfig,
    S3ConnectorTarget,
    S3RootedConnectorTarget,
)
from kumoapi.source_table import S3SourceTable

from kumo.engine.connector._extras import import_provider
from kumo.engine.connector.base import Connector
from kumo.engine.connector.identity import S3Identity


def _parse_s3_root_dir(root_dir: str,
                       region: Optional[str]) -> S3ConnectorTarget:
    parsed = urlsplit(root_dir)
    if parsed.scheme != 's3':
        raise ValueError(f"root_dir must be an s3:// URI, got {root_dir!r}.")
    prefix = parsed.path.strip('/')
    return S3RootedConnectorTarget(bucket=parsed.netloc, prefix=prefix,
                                   region=region)


class S3Connector(Connector):
    r"""Connect directly to Amazon S3 or an S3-compatible service.

    Reads and writes through `kumo_s3.S3SourceBackend`/`S3SinkBackend` --
    install the `s3` extra of this package to use it.

    Two target modes are available:

    * Set `root_dir` to constrain the connector to one bucket and optional
      prefix. Table names are relative paths below the root.
    * Leave `root_dir=None` for rootless mode: every read/write uses a full
      `s3://` URI; discovery remains unavailable.

    Args:
        root_dir: Optional S3 bucket/prefix root, e.g.
            `"s3://company-data/events/"`. `None` selects any-URI mode.
        identity: An `S3Identity`, passed directly. `None` selects
            `kumo_s3`'s ambient AWS credential chain (instance profile,
            `aws configure`, environment) -- there is no server-side
            workload identity in this package.
        region: Optional bucket region for rooted mode. Invalid in any-URI
            mode.
    """
    def __init__(
        self,
        *,
        root_dir: Optional[str] = None,
        identity: Optional[S3Identity] = None,
        region: Optional[str] = None,
    ) -> None:
        super().__init__(identity=identity)
        if root_dir is None:
            if region is not None:
                raise ValueError(
                    'region is only supported when root_dir is set')
            target: S3ConnectorTarget = S3AnyURIConnectorTarget()
        else:
            target = _parse_s3_root_dir(root_dir, region)
        self._config = S3ConnectorResourceConfig(target=target)
        self.target = target
        self.any_uri = isinstance(target, S3AnyURIConnectorTarget)
        self.bucket = (target.bucket
                       if isinstance(target, S3RootedConnectorTarget) else None)
        self.prefix = (target.prefix
                       if isinstance(target, S3RootedConnectorTarget) else None)
        self.region = (target.region
                       if isinstance(target, S3RootedConnectorTarget) else None)

    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.S3

    @property
    def config(self) -> S3ConnectorResourceConfig:
        return self._config

    @cached_property
    def source_backend(self):
        S3SourceBackend = import_provider("kumo_s3.backend",
                                          "s3").S3SourceBackend
        return S3SourceBackend(identity=self.identity)

    @cached_property
    def sink_backend(self):
        S3SinkBackend = import_provider("kumo_s3.sink", "s3").S3SinkBackend
        return S3SinkBackend(identity=self.identity)

    def source_table_for(self, table_name: str) -> S3SourceTable:
        """Resolve source format and addressing through the provider reader."""
        return self.source_backend.get_table_config(self,
                                                    table_name).source_table

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(bucket={self.bucket!r}, "
                f"prefix={self.prefix!r})")
