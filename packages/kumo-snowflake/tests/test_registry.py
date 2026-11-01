# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metrics returned by model downloads come from the exported bundle."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
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
        show_metrics=lambda: {
            'kumo_engine': '1',
            'kumo_bundle_id': 'b1',
            'auc': 0.5
        })
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
    assert published.metadata == {'kumo_bundle_id': 'b1'}
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


@pytest.mark.parametrize('metadata', [{}, {
    'predictive_query': 'x' * 300,
    'kumo_bundle_id': 'b1',
    'kumo_engine': '0'
}])
def test_publish_records_pointers_and_holdout_metrics_in_catalog(
        tmp_path, monkeypatch, metadata):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'training-spec.json').write_text('{}')
    calls = []
    version = SimpleNamespace(model_name='CHURN', version_name='V1',
                              fully_qualified_model_name='DB.SCHEMA.CHURN')

    def log_model(*args, **kwargs):
        calls.append(kwargs)
        return version

    def get_model(_):
        raise ValueError('Unable to find model CHURN')

    @contextmanager
    def registry(_):
        yield SimpleNamespace(log_model=log_model, get_model=get_model)

    sink = SnowflakeModelRegistryMixin()
    monkeypatch.setattr(sink, '_registry', registry)
    monkeypatch.setattr(registry_module, '_bundle_model', lambda _: object())
    monkeypatch.setattr(registry_module, '_carrier_signature', lambda: {})
    metrics = {'auroc': 0.9213, 'holdout_loss': 0.5, 'brier': float('nan')}
    published = sink.publish_model(
        SimpleNamespace(), str(bundle),
        location=ModelRegistryLocation('CHURN', comment='Weekly retrain.'),
        metadata=metadata, metrics=metrics)
    assert calls[0]['metrics'] == {
        'auroc': 0.9213, 'holdout_loss': 0.5,
        **({'kumo_bundle_id': 'b1'} if metadata else {}), 'kumo_engine': '1'
    }
    assert calls[0]['version_name'] == 'V1'
    assert calls[0]['comment'] == ('Weekly retrain.\n'
                                   'Holdout: auroc 0.921 · holdout_loss 0.5')
    assert published.metadata == metadata
    assert published.metrics == metrics


def test_publish_names_the_next_version_and_retries_a_taken_one(
        tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'training-spec.json').write_text('{}')
    names = ['V1', 'SLIMY_DONKEY_1', 'V3']
    tried = []

    def log_model(*args, version_name, **kwargs):
        tried.append(version_name)
        if version_name == 'V4':
            names.append('V4')
            raise ValueError('Model CHURN version V4 already existed.')
        return SimpleNamespace(model_name='CHURN', version_name=version_name,
                               fully_qualified_model_name='DB.SCHEMA.CHURN')

    model = SimpleNamespace(versions=lambda: [
        SimpleNamespace(version_name=name) for name in names])

    @contextmanager
    def registry(_):
        yield SimpleNamespace(log_model=log_model, get_model=lambda _: model)

    sink = SnowflakeModelRegistryMixin()
    monkeypatch.setattr(sink, '_registry', registry)
    monkeypatch.setattr(registry_module, '_bundle_model', lambda _: object())
    monkeypatch.setattr(registry_module, '_carrier_signature', lambda: {})
    published = sink.publish_model(SimpleNamespace(), str(bundle),
                                   location=ModelRegistryLocation('CHURN'))
    assert tried == ['V4', 'V5']
    assert published.version == 'V5'
    with pytest.raises(ValueError, match='already existed'):
        sink.publish_model(SimpleNamespace(), str(bundle),
                           location=ModelRegistryLocation('CHURN',
                                                          version='V4'))

    def fail_after_creating(*args, version_name, **kwargs):
        tried.append(version_name)
        names.append(version_name)
        raise RuntimeError('upload failed')

    @contextmanager
    def failing(_):
        yield SimpleNamespace(log_model=fail_after_creating,
                              get_model=lambda _: model)

    tried.clear()
    monkeypatch.setattr(sink, '_registry', failing)
    with pytest.raises(RuntimeError, match='upload failed'):
        sink.publish_model(SimpleNamespace(), str(bundle),
                           location=ModelRegistryLocation('CHURN'))
    assert len(tried) == 3


def test_listing_filters_engine_marker_and_returns_lineage(monkeypatch):
    versions = []
    for name, recorded in [('V1', {
            'kumo_engine': '1',
            'kumo_bundle_id': 'b1',
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
    assert [item.metadata for item in found] == [{'kumo_bundle_id': 'b1'}, {}]


class _Model:
    def __init__(self, aliases):
        self.aliases = aliases

    def show_versions(self):
        return pd.DataFrame([
            dict(name=name, aliases=json.dumps(held))
            for name, held in self.aliases.items()
        ])

    def version(self, version_or_alias):
        name = next((name for name, held in self.aliases.items()
                     if version_or_alias.upper() in held), version_or_alias)
        held = self.aliases[name]

        def export(directory):
            bundle = Path(directory) / 'bundle'
            bundle.mkdir()
            (bundle / 'training-spec.json').write_text(name)
            (bundle / 'publication-manifest.json').write_text('{}')

        return SimpleNamespace(
            version_name=name, model_name='CHURN',
            fully_qualified_model_name='DB.SCHEMA.CHURN', export=export,
            set_alias=lambda alias: held.append(alias.upper()),
            unset_alias=lambda alias: held.remove(alias.upper()),
            show_metrics=lambda: {
                'kumo_engine': '1',
                'kumo_bundle_id': f'id-{name}'
            })


def _aliasing_sink(monkeypatch, model):
    @contextmanager
    def registry(_):
        yield SimpleNamespace(get_model=lambda _: model)

    sink = SnowflakeModelRegistryMixin()
    monkeypatch.setattr(sink, '_registry', registry)
    return sink


def test_set_alias_moves_it_and_refuses_what_snowflake_cannot(monkeypatch):
    from kumo_snowflake.sink import SnowflakeSinkBackend
    from kumoapi.plugin import ModelAliasingSink

    assert isinstance(SnowflakeSinkBackend(), ModelAliasingSink)
    model = _Model({'V1': ['FIRST'], 'V2': ['LAST', 'DEFAULT'], 'V3': []})
    sink = _aliasing_sink(monkeypatch, model)

    def set_alias(version, alias):
        sink.set_model_alias(SimpleNamespace(),
                             ModelRegistryLocation('CHURN', version=version),
                             alias)

    set_alias('V2', 'prod')
    set_alias('V3', 'prod')
    set_alias('V3', 'prod')
    assert model.aliases == {
        'V1': ['FIRST'],
        'V2': ['LAST', 'DEFAULT'],
        'V3': ['PROD']
    }
    for version, alias, match in [('V3', 'staging', 'already holds'),
                                  ('V1', 'default', 'reserved'),
                                  (None, 'prod', 'exact version')]:
        with pytest.raises(ValueError, match=match):
            set_alias(version, alias)
    assert model.aliases['V3'] == ['PROD']


def test_fetch_by_alias_reads_the_aliased_version(monkeypatch, tmp_path):
    sink = _aliasing_sink(monkeypatch, _Model({
        'V1': ['DEFAULT'],
        'V2': ['PROD']
    }))
    published = sink.fetch_model(
        SimpleNamespace(), str(tmp_path / 'out'),
        location=ModelRegistryLocation('CHURN', alias='prod'))
    assert published.version == 'V2'
    assert published.metadata == {'kumo_bundle_id': 'id-V2'}
    assert (tmp_path / 'out' / 'training-spec.json').read_text() == 'V2'
    with pytest.raises(ValueError, match='holds alias'):
        sink.fetch_model(SimpleNamespace(), str(tmp_path / 'other'),
                         location=ModelRegistryLocation('CHURN', alias='V1'))


def test_failed_alias_move_puts_it_back(monkeypatch):
    model = _Model({'V1': ['PROD'], 'V2': []})

    def fail(alias):
        raise RuntimeError('lost connection')

    def failing_version(name):
        version = _Model.version(model, name)
        if name == 'V2':
            version.set_alias = fail
        return version

    model.version = failing_version
    sink = _aliasing_sink(monkeypatch, model)
    with pytest.raises(RuntimeError, match='lost connection'):
        sink.set_model_alias(SimpleNamespace(),
                             ModelRegistryLocation('CHURN', version='V2'),
                             'prod')
    assert model.aliases == {'V1': ['PROD'], 'V2': []}


@pytest.mark.parametrize('task, expected', [
    ('binary_classification', 'TABULAR_BINARY_CLASSIFICATION'),
    ('link_prediction', None),
])
def test_publish_records_the_task_type_monitors_need(tmp_path, monkeypatch,
                                                     task, expected):
    import sys

    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'training-spec.json').write_text('{}')
    (bundle / 'prediction-schema.json').write_text(
        json.dumps({'task_type': task}))
    tasks = SimpleNamespace(
        TABULAR_BINARY_CLASSIFICATION='TABULAR_BINARY_CLASSIFICATION')
    monkeypatch.setitem(sys.modules, 'snowflake.ml.model',
                        SimpleNamespace(type_hints=SimpleNamespace(Task=tasks)))
    calls = []

    @contextmanager
    def registry(_):
        yield SimpleNamespace(log_model=lambda *a, **kw: calls.append(kw) or
                              SimpleNamespace(
                                  model_name='CHURN', version_name='V1',
                                  fully_qualified_model_name='DB.S.CHURN'))

    sink = SnowflakeModelRegistryMixin()
    monkeypatch.setattr(sink, '_registry', registry)
    monkeypatch.setattr(registry_module, '_bundle_model', lambda _: object())
    monkeypatch.setattr(registry_module, '_carrier_signature', lambda: {})
    sink.publish_model(SimpleNamespace(), str(bundle),
                       location=ModelRegistryLocation('CHURN', version='V1'))
    assert calls[0].get('task') == expected


@pytest.mark.parametrize('name', ['USERS_CHURN', 'users_churn$2', '_m',
                                  'ML.USERS_CHURN', 'DB.ML.USERS_CHURN',
                                  '"users-churn"', '"a ""b"""'])
def test_snowflake_identifiers_are_accepted_as_model_names(name):
    SnowflakeModelRegistryMixin().check_model_location(
        SimpleNamespace(), ModelRegistryLocation(name))


@pytest.mark.parametrize('name', ['users-churn', '1churn', 'users churn',
                                  'A.B.C.D', 'ML.', '""'])
def test_other_model_names_are_refused_with_the_rule(name):
    with pytest.raises(ValueError, match="letter or underscore.*USERS_CHURN"):
        SnowflakeModelRegistryMixin().check_model_location(
            SimpleNamespace(), ModelRegistryLocation(name))
