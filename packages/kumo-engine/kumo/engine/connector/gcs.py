# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""`GCSConnector`, mirroring the argument list of the cloud SDK's
`kumoai.connector.gcs_connector.GCSConnector`, minus everything that exists
only to round-trip a server-side connector registry. See
`kumo.engine.connector.base`.

TODO(local-sdk): same caveat as `kumo.engine.connector.s3` -- the cloud SDK's
`GCSURI`-based `root_dir` validation isn't ported yet.
"""

from functools import cached_property
from typing import Optional
from urllib.parse import urlsplit

from kumoapi.data_source import (
    DataSourceType,
    GCSAnyURIConnectorTarget,
    GCSConnectorResourceConfig,
    GCSConnectorTarget,
    GCSRootedConnectorTarget,
)
from kumoapi.source_table import GCSSourceTable

from kumo.engine.connector.base import Connector
from kumo.engine.connector.identity import GCSIdentity


def _parse_gcs_root_dir(root_dir: str) -> GCSConnectorTarget:
    parsed = urlsplit(root_dir)
    if parsed.scheme != 'gs':
        raise ValueError(f"root_dir must be a gs:// URI, got {root_dir!r}.")
    prefix = parsed.path.strip('/')
    return GCSRootedConnectorTarget(bucket=parsed.netloc, prefix=prefix)


class GCSConnector(Connector):
    r"""Connect directly to Google Cloud Storage.

    Reads and writes through `kumo_gcs.GCSSourceBackend`/`GCSSinkBackend` --
    install the `gcs` extra of this package to use it.

    Two target modes are available, exactly as in the cloud SDK:

    * Set `root_dir` to constrain the connector to one bucket and optional
      prefix. Table names are relative paths below the root.
    * Leave `root_dir=None` for rootless mode: every read/write uses a full
      `gs://` URI; discovery remains unavailable.

    Args:
        root_dir: Optional GCS bucket/prefix root, e.g.
            `"gs://company-data/events/"`. `None` selects any-URI mode.
        identity: A `GCSIdentity`, passed directly. `None` selects
            `kumo_gcs`'s ambient Application Default Credentials -- there is
            no server-side workload identity in this package.
        requester_pays_project: Optional GCP project to bill for
            requester-pays buckets.
    """
    def __init__(
        self,
        *,
        root_dir: Optional[str] = None,
        identity: Optional[GCSIdentity] = None,
        requester_pays_project: Optional[str] = None,
    ) -> None:
        super().__init__(identity=identity)
        target: GCSConnectorTarget
        if root_dir is None:
            target = GCSAnyURIConnectorTarget()
        else:
            target = _parse_gcs_root_dir(root_dir)
        self._config = GCSConnectorResourceConfig(
            target=target,
            requester_pays_project=requester_pays_project,
        )
        self.target = target
        self.any_uri = isinstance(target, GCSAnyURIConnectorTarget)
        self.bucket = (target.bucket if isinstance(
            target, GCSRootedConnectorTarget) else None)
        self.prefix = (target.prefix if isinstance(
            target, GCSRootedConnectorTarget) else None)
        self.requester_pays_project = self._config.requester_pays_project

    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.GCS

    @property
    def config(self) -> GCSConnectorResourceConfig:
        return self._config

    @cached_property
    def source_backend(self):
        from kumo_gcs.backend import GCSSourceBackend
        return GCSSourceBackend(identity=self.identity)

    @cached_property
    def sink_backend(self):
        from kumo_gcs.sink import GCSSinkBackend
        return GCSSinkBackend(identity=self.identity)

    def source_table_for(self, table_name: str) -> GCSSourceTable:
        """Resolve source format and addressing through the provider reader."""
        return self.source_backend.get_table_config(self,
                                                    table_name).source_table

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(bucket={self.bucket!r}, "
                f"prefix={self.prefix!r})")
