# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for warehouse-type derivation from a resolved connector.

A resolved connector spec's ``connector_config`` is the connector's NATIVE
config (``ConfigMixin.config()``, reflected from ``__init__``) — it has fields
like ``name``/``host``/``warehouse`` and a ``_target_``, but NO ``type`` field.
The prior ``_connector_type_from_spec`` read ``config.type`` and therefore
raised ``AttributeError`` for a real resolved warehouse connector (masked in the
authoring tests because they mock ``resolve_connector_and_type_async``). These
tests build specs the exact way production does — ``connector_config =
<conn>.config()`` — and assert the type is derived from the connector class with
no ``AttributeError``.
"""
import asyncio

import pytest
from kumo_core.types import ConnectorType

from kumo.connector import (
    BigQueryConnector,
    DatabricksConnector,
    ParquetConnector,
    SnowflakeConnector,
)
from rest.backends.data_backend.utils import (
    _connector_type_from_spec,
    _connector_type_from_spec_async,
    _connector_type_of,
    resolve_connector_and_type,
    resolve_connector_and_type_async,
)
from rest.config.data_backend import ConnectorSpec, ResolvedConnectorSpec

_CID = 'connector_00000000000000000000000000000000'


def _databricks() -> DatabricksConnector:
    return DatabricksConnector(name='wh',
                               host='https://workspace.cloud.databricks.com',
                               cluster_id='', warehouse_id='sql-warehouse',
                               catalog='main', schema_name='sales')


def _snowflake() -> SnowflakeConnector:
    return SnowflakeConnector(name='wh', account='acc', warehouse='COMPUTE',
                              database='DB', schema_name='SALES')


def _bigquery() -> BigQueryConnector:
    return BigQueryConnector(name='wh', project_id='data-project',
                             dataset_id='sales')


def _resolved_spec(connector) -> ResolvedConnectorSpec:
    # Mirror production: connector_config is the connector's NATIVE config.
    return ResolvedConnectorSpec(connector_id=_CID,
                                 connector_config=connector.config(),
                                 credentials=None)


_CASES = [
    (_databricks, ConnectorType.DATABRICKS),
    (_snowflake, ConnectorType.SNOWFLAKE),
    (_bigquery, ConnectorType.BIGQUERY),
]


def test_native_config_has_no_type_field():
    """Guards the root cause: the native config exposes no ``type`` attribute,
    so the old ``config.type`` read is an ``AttributeError`` — the exact failure
    this fix removes."""
    for factory, _ in _CASES:
        config = factory().config()
        assert not hasattr(config, 'type')
        with pytest.raises(AttributeError):
            _ = config.type  # noqa: B018  (intentional attribute access)


@pytest.mark.parametrize('factory,expected', _CASES)
def test_connector_type_from_spec_resolves_warehouse(factory, expected):
    spec = _resolved_spec(factory())
    # Must not raise AttributeError; must derive the right warehouse type.
    assert _connector_type_from_spec(spec) == expected


@pytest.mark.parametrize('factory,expected', _CASES)
def test_connector_type_from_spec_async_resolves_warehouse(factory, expected):
    spec = _resolved_spec(factory())
    assert asyncio.run(_connector_type_from_spec_async(spec)) == expected


@pytest.mark.parametrize('factory,expected', _CASES)
def test_connector_type_from_unresolved_connector_spec(factory, expected):
    """A (non-resolved) ``ConnectorSpec`` carries the same native config and
    resolves identically."""
    spec = ConnectorSpec(connector_id=_CID, connector_config=factory().config())
    assert _connector_type_from_spec(spec) == expected


@pytest.mark.parametrize('factory,expected', _CASES)
def test_resolve_connector_and_type_returns_instance_and_type(
        factory, expected):
    spec = _resolved_spec(factory())
    connector, connector_type = resolve_connector_and_type(spec)
    assert isinstance(connector, type(factory()))
    assert connector_type == expected


@pytest.mark.parametrize('factory,expected', _CASES)
def test_resolve_connector_and_type_async(factory, expected):
    spec = _resolved_spec(factory())
    connector, connector_type = asyncio.run(
        resolve_connector_and_type_async(spec))
    assert isinstance(connector, type(factory()))
    assert connector_type == expected


def test_connector_type_of_direct_dispatch():
    assert _connector_type_of(_databricks()) == ConnectorType.DATABRICKS
    assert _connector_type_of(_snowflake()) == ConnectorType.SNOWFLAKE
    assert _connector_type_of(_bigquery()) == ConnectorType.BIGQUERY


def test_non_warehouse_connector_fails_loud():
    """A non-warehouse (file/blob) connector has no Custom-SQL dialect, so type
    derivation fails loud with a clear message rather than silently."""
    connector = ParquetConnector('parquet', root_dir='my_test_dir')
    with pytest.raises(ValueError, match='not supported for'):
        _connector_type_of(connector)


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
