# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metrics returned by model downloads come from the exported bundle."""

import json
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from kumo_snowflake import registry as registry_module
from kumo_snowflake.registry import SnowflakeModelRegistryMixin
from kumoapi.data_source import SnowflakeConnectorResourceConfig
from kumoapi.plugin import ModelRegistryLocation


class _VersionFiles:
    r"""A session over model version files, as LIST and GET see them."""
    def __init__(self, files):
        self.files, self.got = files, []

    def sql(self, query):
        version = re.search(r'/versions/([^/]+)/model/', query).group(1)
        return SimpleNamespace(collect=lambda: [{
            'name': f'versions/{version}/{name}'
        } for name in self.files[version]])

    @property
    def connection(self):
        return SimpleNamespace(cursor=self._cursor)

    @contextmanager
    def _cursor(self):
        yield SimpleNamespace(execute=self._get)

    def _get(self, query):
        version, name, local = re.fullmatch(
            r"GET 'snow://model/DB\.SCHEMA\.CHURN/versions/([^/]+)/(\S+)' "
            r"'file://(\S+)'", query).groups()
        self.got.append(name)
        (Path(local) / Path(name).name).write_bytes(self.files[version][name])


def _version_files(**files):
    return {
        f'model/models/CHURN/artifacts/bundle/{name}': data
        for name, data in {
            'training-spec.json': b'{}',
            'publication-manifest.json': b'{}',
            **files
        }.items()
    } | {'model/code/kumo/__init__.py': b''}


def _sink(monkeypatch, model, files):
    @contextmanager
    def registry(_):
        yield SimpleNamespace(get_model=lambda _: model)

    @contextmanager
    def session(_):
        yield files

    sink = SnowflakeModelRegistryMixin()
    monkeypatch.setattr(sink, '_registry', registry)
    monkeypatch.setattr(sink, '_session', session, raising=False)
    return sink


@pytest.fixture
def fetch(tmp_path, monkeypatch):
    files = _VersionFiles({})
    version = SimpleNamespace(
        model_name='CHURN', version_name='V1',
        fully_qualified_model_name='DB.SCHEMA.CHURN',
        show_metrics=lambda: {
            'kumo_engine': '1',
            'kumo_bundle_id': 'b1',
            'auc': 0.5
        })
    model = SimpleNamespace(default=version, version=lambda _: version)
    sink = _sink(monkeypatch, model, files)
    destination = tmp_path / 'restored'
    destination.mkdir()
    (destination / 'previous').write_bytes(b'previous bundle')

    def run(payload, requested_version):
        extra = {} if payload is None else {'metrics.json': payload.encode()}
        files.files['V1'] = _version_files(**{'weights.bin': b'model weights'},
                                           **extra)
        return sink.fetch_model(
            SimpleNamespace(), str(destination),
            location=ModelRegistryLocation('CHURN', version=requested_version))

    run.files = files
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
    assert not any(name.startswith('model/code/') for name in run.files.got)
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

    @contextmanager
    def registry(_):
        yield SimpleNamespace(log_model=log_model)

    sink = SnowflakeModelRegistryMixin()
    monkeypatch.setattr(sink, '_registry', registry)
    monkeypatch.setattr(registry_module, '_model_args', lambda *_: {})
    metrics = {'auroc': 0.9213, 'holdout_loss': 0.5, 'brier': float('nan')}
    published = sink.publish_model(
        SimpleNamespace(), str(bundle),
        location=ModelRegistryLocation('CHURN', comment='Weekly retrain.'),
        metadata=metadata, metrics=metrics)
    assert calls[0]['metrics'] == {
        'auroc': 0.9213, 'holdout_loss': 0.5,
        **({'kumo_bundle_id': 'b1'} if metadata else {}), 'kumo_engine': '1'
    }
    assert calls[0]['comment'] == ('Weekly retrain.\n'
                                   'Holdout: auroc 0.921 · holdout_loss 0.5')
    assert published.metadata == metadata
    assert published.metrics == metrics


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

        return SimpleNamespace(
            version_name=name, model_name='CHURN',
            fully_qualified_model_name='DB.SCHEMA.CHURN',
            set_alias=lambda alias: held.append(alias.upper()),
            unset_alias=lambda alias: held.remove(alias.upper()),
            show_metrics=lambda: {
                'kumo_engine': '1',
                'kumo_bundle_id': f'id-{name}'
            })


def _aliasing_sink(monkeypatch, model):
    files = _VersionFiles({
        name: _version_files(**{'training-spec.json': name.encode()})
        for name in model.aliases
    })
    return _sink(monkeypatch, model, files)


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
    monkeypatch.setattr(registry_module, '_model_args', lambda *_: {})
    sink.publish_model(SimpleNamespace(), str(bundle),
                       location=ModelRegistryLocation('CHURN'))
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


class _Version:
    model_name, version_name = 'CHURN', 'V1'

    def __init__(self, recorded, comment, refuse=False):
        self.recorded, self._comment, self.refuse = recorded, comment, refuse
        self.calls = []

    def show_metrics(self):
        return dict(self.recorded)

    def set_metric(self, name, value):
        if self.refuse:
            raise RuntimeError('Insufficient privileges to operate on model')
        self.calls.append(('set_metric', name, value))

    @property
    def comment(self):
        return self._comment

    @comment.setter
    def comment(self, comment):
        self.calls.append(('comment', comment))


METRICS = {'auroc': 0.9, 'holdout_loss': 0.5}
LINE = 'Holdout: auroc 0.9 · holdout_loss 0.5'


@pytest.mark.parametrize('recorded, comment, given, calls', [
    ({'kumo_engine': '1', 'auroc': 0.9}, 'Weekly.', None,
     [('set_metric', 'holdout_loss', 0.5), ('comment', f'Weekly.\n{LINE}')]),
    ({'auroc': 0.8}, 'Weekly.\nHoldout: auroc 0.8', 'Daily.',
     [('set_metric', 'auroc', 0.9), ('set_metric', 'holdout_loss', 0.5),
      ('comment', f'Daily.\n{LINE}')]),
    (METRICS, f'Weekly.\n{LINE}', None, []),
    (METRICS, None, None, [('comment', LINE)]),
])
def test_a_reused_version_gets_the_metrics_it_lacks(recorded, comment, given,
                                                    calls):
    version = _Version(recorded, comment)

    registry_module._backfill(version, METRICS, given)

    assert version.calls == calls


def test_a_refused_backfill_warns_naming_ownership(caplog):
    version = _Version({}, None, refuse=True)

    registry_module._backfill(version, METRICS, None)

    assert version.calls == []
    assert 'needs OWNERSHIP of the model' in caplog.text
    assert 'Insufficient privileges' in caplog.text


def _publishing_sink(monkeypatch, versions, calls):
    @contextmanager
    def registry(_):
        def log_model(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(model_name='CHURN', version_name='V9',
                                   fully_qualified_model_name='DB.S.CHURN')

        def get_model(_):
            if versions is None:
                raise ValueError('no such model')
            return SimpleNamespace(versions=lambda: versions)

        yield SimpleNamespace(log_model=log_model, get_model=get_model)

    sink = SnowflakeModelRegistryMixin()
    monkeypatch.setattr(sink, '_registry', registry)
    return sink


def _bundle(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'training-spec.json').write_text('{}')
    return bundle


@pytest.mark.parametrize('versions, reused', [
    (None, None),
    ([SimpleNamespace(version_name='V1', show_metrics=lambda: {
        'kumo_engine': '1', 'kumo_bundle_id': 'other'})], None),
    ([SimpleNamespace(version_name='V2', show_metrics=lambda: {
        'kumo_bundle_id': 'b1'})], None),
    ([SimpleNamespace(version_name='V3', model_name='CHURN',
                      fully_qualified_model_name='DB.S.CHURN',
                      show_metrics=lambda: {
                          'kumo_engine': '1', 'kumo_bundle_id': 'b1'})], 'V3'),
])
def test_a_registered_bundle_is_returned_rather_than_logged_again(
        tmp_path, monkeypatch, caplog, versions, reused):
    calls = []
    sink = _publishing_sink(monkeypatch, versions, calls)
    monkeypatch.setattr(registry_module, '_model_args', lambda *_: {})
    with caplog.at_level('INFO', logger=registry_module.__name__):
        published = sink.publish_model(
            SimpleNamespace(), str(_bundle(tmp_path)),
            location=ModelRegistryLocation('CHURN'),
            metadata={'kumo_bundle_id': 'b1'})
    assert published.version == (reused or 'V9')
    assert published.metadata == {'kumo_bundle_id': 'b1'}
    assert len(calls) == (0 if reused else 1)
    assert ('already holds bundle b1; reused it' in caplog.text) == bool(
        reused)


def test_engine_model_arguments_reach_log_model(tmp_path, monkeypatch):
    seen = []

    def log_model_args(bundle, metadata, target, workdir):
        seen.append((metadata, target))
        return {'model': 'runnable', 'pip_requirements': ['kumo-engine']}

    monkeypatch.setattr(registry_module, '_model_builder',
                        lambda: log_model_args)
    calls = []
    sink = _publishing_sink(monkeypatch, [], calls)
    connector = SimpleNamespace(config=SnowflakeConnectorResourceConfig(
        warehouse='WH', database='DB', schema_name='ML'))
    sink.publish_model(connector, str(_bundle(tmp_path)),
                       location=ModelRegistryLocation('CHURN'),
                       metadata={'kumo.serving_snapshot': '@DB.ML.ST/s'})
    assert calls[0]['model'] == 'runnable'
    assert calls[0]['pip_requirements'] == ['kumo-engine']
    assert seen == [({'kumo.serving_snapshot': '@DB.ML.ST/s'}, {
        'warehouse': 'WH', 'database': 'DB', 'schema': 'ML'})]


def test_a_serving_environment_that_cannot_work_fails_registration(
        tmp_path, monkeypatch):
    def refuse(*_):
        raise registry_module.ServingEnvironmentError('torch has no cu126')

    monkeypatch.setattr(registry_module, '_model_builder', lambda: refuse)
    calls = []
    sink = _publishing_sink(monkeypatch, [], calls)
    connector = SimpleNamespace(config=SnowflakeConnectorResourceConfig(
        warehouse='WH', database='DB', schema_name='ML'))
    with pytest.raises(ValueError, match='torch has no cu126'):
        sink.publish_model(connector, str(_bundle(tmp_path)),
                           location=ModelRegistryLocation('CHURN'))
    assert calls == []


def test_a_bundle_registered_for_another_environment_gets_a_new_version(
        tmp_path, monkeypatch, caplog):
    environments = iter([['torch>=2.15,<2.16'], ['torch>=2.12,<2.13'],
                         ['torch>=2.15,<2.16']])
    monkeypatch.setattr(
        registry_module, '_model_args', lambda *_: {
            'model': 'runnable', 'pip_requirements': next(environments)})
    versions, calls = [], []
    sink = _publishing_sink(monkeypatch, versions, calls)
    bundle = str(_bundle(tmp_path))

    def publish():
        logged = len(calls)
        sink.publish_model(SimpleNamespace(), bundle,
                           location=ModelRegistryLocation('CHURN'),
                           metadata={'kumo_bundle_id': 'b1'})
        if len(calls) > logged:
            versions.append(SimpleNamespace(
                version_name=f'V{len(versions) + 1}', model_name='CHURN',
                fully_qualified_model_name='DB.S.CHURN',
                show_metrics=lambda metrics=calls[-1]['metrics']: metrics))
        return len(calls) > logged

    with caplog.at_level('WARNING', logger=registry_module.__name__):
        assert [publish(), publish(), publish()] == [True, True, False]
    assert 'V1 holds bundle b1 for another serving environment' in caplog.text


@pytest.mark.parametrize('engine', [None, 'raises'])
def test_without_a_usable_engine_the_bundle_travels_in_the_carrier(
        tmp_path, monkeypatch, engine):
    def broken(*_):
        raise ValueError('no prediction schema')

    monkeypatch.setattr(registry_module, '_model_builder',
                        lambda: None if engine is None else broken)
    monkeypatch.setattr(registry_module, '_bundle_model', lambda _: 'carrier')
    monkeypatch.setattr(registry_module, '_carrier_signature',
                        lambda: {'predict': 'declared'})
    calls = []
    sink = _publishing_sink(monkeypatch, [], calls)
    sink.publish_model(SimpleNamespace(), str(_bundle(tmp_path)),
                       location=ModelRegistryLocation('CHURN'))
    assert calls[0]['model'] == 'carrier'
    assert calls[0]['signatures'] == {'predict': 'declared'}
    assert 'pip_requirements' not in calls[0]
