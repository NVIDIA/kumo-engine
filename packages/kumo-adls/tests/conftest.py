# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fakes import FakeService
from kumoapi.data_source import (
    ADLSAnyURIConnectorTarget,
    ADLSConnectorResourceConfig,
    ADLSRootedConnectorTarget,
    DataSourceType,
)


@pytest.fixture
def service(monkeypatch):
    service = FakeService()
    monkeypatch.setattr('kumo_adls.source_access.session_for',
                        lambda *args: service)
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({'id': [1, 2], 'value': [10.0, 20.0]}), sink)
    service.files['curated/orders/events.parquet'] = sink.getvalue().to_pybytes(
    )
    return service


@pytest.fixture
def connector():
    def build(*, account='companylake', prefix='orders', any_uri=False,
              suffix='dfs.core.windows.net'):
        target = ADLSAnyURIConnectorTarget(
        ) if any_uri else ADLSRootedConnectorTarget(
            storage_account=account, filesystem='curated', prefix=prefix,
            endpoint_suffix=suffix)
        return SimpleNamespace(
            id='adls-test', config=ADLSConnectorResourceConfig(target=target),
            identity=None, identity_id=None, source_type=DataSourceType.ADLS)

    return build
