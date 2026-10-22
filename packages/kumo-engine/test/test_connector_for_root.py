# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kumo.engine.connector import (
    FileConnector,
    GCSConnector,
    S3Connector,
    connector_for_root,
)


def test_connector_for_root(tmp_path):
    s3 = connector_for_root('s3://my-bucket/runs/1')
    assert isinstance(s3, S3Connector)
    assert (s3.bucket, s3.prefix) == ('my-bucket', 'runs/1')
    assert isinstance(connector_for_root('gs://my-bucket/runs'), GCSConnector)
    for uri in (str(tmp_path), tmp_path.as_uri()):
        local = connector_for_root(uri)
        assert isinstance(local, FileConnector)
        assert local.config.root_dir == str(tmp_path)
    with pytest.raises(ValueError, match='not s3://'):
        connector_for_root('abfss://c@a.dfs.core.windows.net/x')
