# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
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
    monkeypatch.setattr('kumo_adls.sink.filesystem_for',
                        lambda *args: FakeNativeFilesystem(service))
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


class FakeNativeFilesystem:
    """Native-output-shaped fake; commit only when Arrow closes the stream."""
    def __init__(self, service):
        self.service = service

    def open_output_stream(self, path, metadata=None):
        service = self.service

        class Output(io.BytesIO):
            def close(self):
                if not self.closed:
                    service.files[path] = self.getvalue()
                super().close()

        service.created.append(path)
        return pa.PythonFile(Output(), mode='w')
