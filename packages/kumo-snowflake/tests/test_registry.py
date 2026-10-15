# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metrics returned by model downloads come from the exported bundle."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from kumo_snowflake import registry as registry_module
from kumo_snowflake.registry import SnowflakeModelRegistryMixin
from kumoapi.plugin import ModelRegistryLocation


@pytest.fixture
def fetch(tmp_path, monkeypatch):
    metrics_payload = None

    def export(directory):
        bundle = Path(directory) / 'artifacts' / 'bundle'
        bundle.mkdir(parents=True)
        (bundle / 'training-spec.json').write_text('{}')
        (bundle / 'publication-manifest.json').write_text('{}')
        (bundle / 'weights.bin').write_bytes(b'model weights')
        if metrics_payload is not None:
            (bundle / 'metrics.json').write_text(metrics_payload)

    version = SimpleNamespace(
        model_name='CHURN', version_name='V1',
        fully_qualified_model_name='DB.SCHEMA.CHURN', export=export,
        show_metrics=lambda: pytest.fail('read registry metrics'))
    model = SimpleNamespace(default=version, version=lambda _: version)

    @contextmanager
    def registry(_):
        yield SimpleNamespace(get_model=lambda _: model)

    sink = SnowflakeModelRegistryMixin()
    monkeypatch.setattr(sink, '_registry', registry)
    destination = tmp_path / 'restored'
    destination.mkdir()
    (destination / 'previous').write_bytes(b'previous bundle')

    def run(payload, requested_version):
        nonlocal metrics_payload
        metrics_payload = payload
        return sink.fetch_model(
            SimpleNamespace(), str(destination),
            location=ModelRegistryLocation('CHURN', version=requested_version))

    return run, destination


@pytest.mark.parametrize('requested_version', [None, 'V1'])
@pytest.mark.parametrize('metrics', [{'auc': 0.92, 'loss': 0.1}, None])
def test_fetch_returns_bundle_metrics(fetch, requested_version, metrics):
    run, destination = fetch
    payload = json.dumps(metrics) if metrics is not None else None
    published = run(payload, requested_version)
    assert published.metrics == (metrics or {})
    assert published.version == 'V1'
    assert (destination / 'weights.bin').read_bytes() == b'model weights'
    assert not (destination / 'previous').exists()
    if payload is not None:
        assert (destination / 'metrics.json').read_text() == payload


@pytest.mark.parametrize('payload', ['not JSON', '[]', '{"auc": "invalid"}'])
def test_invalid_metrics_preserve_previous_destination(fetch, payload):
    run, destination = fetch
    with pytest.raises(ValueError):
        run(payload, 'V1')
    assert (destination / 'previous').read_bytes() == b'previous bundle'


@pytest.mark.parametrize('metadata', [{}, {'predictive_query': 'x' * 300}])
def test_publish_records_only_engine_marker_in_catalog(tmp_path, monkeypatch,
                                                       metadata):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'training-spec.json').write_text('{}')
    calls = []
    version = SimpleNamespace(model_name='CHURN', version_name='V1',
                              fully_qualified_model_name='DB.SCHEMA.CHURN')

    def log_model(*args, **kwargs):
        calls.append(kwargs)
        return version

    @contextmanager
    def registry(_):
        yield SimpleNamespace(log_model=log_model)

    sink = SnowflakeModelRegistryMixin()
    monkeypatch.setattr(sink, '_registry', registry)
    monkeypatch.setattr(registry_module, '_bundle_model', lambda _: object())
    monkeypatch.setattr(registry_module, '_carrier_signature', lambda: {})
    published = sink.publish_model(SimpleNamespace(), str(bundle),
                                   location=ModelRegistryLocation('CHURN'),
                                   metadata=metadata, metrics={'auc': 0.92})
    assert calls[0]['metrics'] == {'kumo_engine': '1'}
    assert published.metadata == metadata
    assert published.metrics == {'auc': 0.92}


def test_listing_filters_engine_marker_and_returns_empty_metrics(monkeypatch):
    versions = []
    for name, recorded in [('V1', {
            'kumo_engine': '1',
            'auc': 0.92
    }), ('V2', {
            'other': '1',
            'auc': 0.7
    }), ('V3', {
            'kumo_engine': '0',
            'auc': 0.8
    }), ('V4', {
            'kumo_engine': '1'
    })]:
        versions.append(
            SimpleNamespace(model_name='CHURN', version_name=name,
                            fully_qualified_model_name='DB.SCHEMA.CHURN',
                            show_metrics=lambda data=recorded: data))
    model = SimpleNamespace(versions=lambda: versions)

    @contextmanager
    def registry(_):
        yield SimpleNamespace(models=lambda: [model])

    sink = SnowflakeModelRegistryMixin()
    monkeypatch.setattr(sink, '_registry', registry)
    found = sink.list_models(SimpleNamespace())
    assert [item.version for item in found] == ['V1', 'V4']
    assert [item.metrics for item in found] == [{}, {}]
    assert all(item.metadata == {} for item in found)
