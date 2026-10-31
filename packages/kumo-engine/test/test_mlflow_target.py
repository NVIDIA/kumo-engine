# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""How `MlflowTracking` reaches its server: explicit credentials set exactly
the state MLflow reads, only around each call; ambient ones set nothing. The
last tests pin the MLflow and plugin internals this relies on."""

import os
import threading
from types import SimpleNamespace

import pytest

import kumo.engine as kumo
from kumo.engine.connector import (
    DatabricksConnector,
    DatabricksIdentity,
    S3Identity,
)
from kumo.engine.integrations import _mlflow_target as target_module
from kumo.engine.integrations._mlflow_target import SageMakerMLflow, resolve

mlflow = pytest.importorskip('mlflow')

ARN = 'arn:aws:sagemaker:us-west-2:account:mlflow-tracking-server/demo'


@pytest.fixture
def tracking_uri(tmp_path):
    previous = mlflow.get_tracking_uri()
    uri = f'sqlite:///{tmp_path / "mlflow.db"}'
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri(previous)


def _connector(identity):
    return DatabricksConnector(catalog='main', schema_name='ml',
                               identity=identity, warehouse_id='w')


def test_no_target_reads_mlflows_own_uris_once(tracking_uri):
    target = resolve(None)
    assert target.tracking_uri == tracking_uri
    assert not target.explicit
    assert target.platform == 'mlflow' and target.run_url('1', 'r') is None
    assert resolve('databricks').registry_uri == 'databricks-uc'
    assert resolve('databricks').platform == 'databricks'
    assert resolve('databricks://prod').registry_uri == (
        'databricks-uc://prod')
    server = resolve('http://mlflow:5000')
    assert server.registry_uri == 'http://mlflow:5000'
    assert server.run_url('1', 'r') == (
        'http://mlflow:5000/#/experiments/1/runs/r')
    assert resolve(ARN).platform == 'sagemaker'
    assert resolve(server) is server


def test_ambient_identities_set_no_credentials(monkeypatch, tracking_uri):
    monkeypatch.setenv('DATABRICKS_TOKEN', 'mine')
    monkeypatch.setenv('MLFLOW_TRACKING_URI', tracking_uri)
    target = resolve(_connector(DatabricksIdentity()))
    before = dict(os.environ)
    with target.scope():
        # A profile or notebook identity: Unity Catalog registration
        # downloads through the process-wide URI, so it points at Databricks.
        assert mlflow.get_tracking_uri() == 'databricks'
        assert {k: v for k, v in os.environ.items()
                if k.startswith('DATABRICKS_')} == {'DATABRICKS_TOKEN': 'mine'}
    assert mlflow.get_tracking_uri() == tracking_uri
    assert dict(os.environ) == before

    for target in (resolve(SageMakerMLflow(ARN, S3Identity())),
                   resolve(SageMakerMLflow(ARN))):
        with target.scope():
            assert mlflow.get_tracking_uri() == ARN
            assert target.artifact_repository('s3://b/run') is None
        assert mlflow.get_tracking_uri() == tracking_uri
        assert dict(os.environ) == before


def test_a_databricks_service_principal_lends_a_token_for_each_call(
        monkeypatch):
    client = SimpleNamespace(config=SimpleNamespace(
        host='https://dbc.example', authenticate=lambda: {
            'Authorization': 'Bearer short-lived'
        }))
    monkeypatch.setattr('kumo_databricks.session._client_for',
                        lambda principal: client)
    monkeypatch.setenv('DATABRICKS_CLIENT_SECRET', 'other')
    monkeypatch.setenv('DATABRICKS_CONFIG_PROFILE', 'laptop')
    monkeypatch.delenv('DATABRICKS_TOKEN', raising=False)
    target = resolve(
        _connector(
            DatabricksIdentity(host='https://dbc.example', client_id='sp',
                               client_secret='secret')))

    previous = mlflow.get_tracking_uri()
    with target.scope():
        inside = {k: v for k, v in os.environ.items()
                  if k.startswith('DATABRICKS_')}
        # Unity Catalog registration downloads through the global URI.
        assert mlflow.get_tracking_uri() == 'databricks'
    assert mlflow.get_tracking_uri() == previous
    assert inside == {'DATABRICKS_HOST': 'https://dbc.example',
                      'DATABRICKS_TOKEN': 'short-lived'}
    assert 'secret' not in inside.values()
    assert os.environ['DATABRICKS_CLIENT_SECRET'] == 'other'
    assert os.environ['DATABRICKS_CONFIG_PROFILE'] == 'laptop'
    assert 'DATABRICKS_TOKEN' not in os.environ
    assert (target.tracking_uri, target.registry_uri) == ('databricks',
                                                          'databricks-uc')
    assert target.run_url('7', 'r') == (
        'https://dbc.example/ml/experiments/7/runs/r')


def test_a_cluster_run_links_to_the_workspace_url_not_the_regional_host(
        monkeypatch):
    from mlflow.utils import databricks_utils

    from kumo.engine.integrations._mlflow_target import resolve

    target = resolve('databricks')
    monkeypatch.setattr(databricks_utils, 'get_workspace_info_from_dbutils',
                        lambda: ('https://dbc-1.example', '42'))
    assert target.run_url('7', 'r') == (
        'https://dbc-1.example/ml/experiments/7/runs/r')


def test_an_aws_access_key_signs_sagemaker_calls_and_uploads(
        monkeypatch, tracking_uri):
    boto3 = pytest.importorskip('boto3')
    plugin = pytest.importorskip('sagemaker_mlflow.session')
    monkeypatch.setenv('MLFLOW_TRACKING_URI', tracking_uri)
    monkeypatch.setenv('SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN', 'arn:aws:iam::x')
    target = resolve(
        SageMakerMLflow(ARN, S3Identity(access_key_id='AKIA',
                                        secret_access_key='s3cret')))

    with target.scope():
        assert mlflow.get_tracking_uri() == ARN
        assert 'SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN' not in os.environ
        session = plugin._get_current_session()
        assert session.get_credentials().access_key == 'AKIA'
        repository = target.artifact_repository('s3://bucket/run/artifacts')
    assert mlflow.get_tracking_uri() == tracking_uri
    assert os.environ['MLFLOW_TRACKING_URI'] == tracking_uri
    assert os.environ['SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN'] == 'arn:aws:iam::x'
    assert plugin._get_current_session() is None
    assert repository._access_key_id == 'AKIA'
    assert target.artifact_repository('dbfs:/x') is None
    assert isinstance(session, boto3.Session)


def test_your_own_mlflow_calls_reach_the_trackers_server_as_its_identity(
        monkeypatch, tracking_uri, tmp_path):
    plugin = pytest.importorskip('sagemaker_mlflow.session')
    from kumo.engine.integrations.mlflow import MlflowTracking

    monkeypatch.setattr(MlflowTracking, '_require_experiment', lambda _: 0)
    tracking = MlflowTracking('e', target=SageMakerMLflow(
        ARN, S3Identity(access_key_id='AKIA', secret_access_key='s')))
    with tracking.client() as client:
        assert client._tracking_client.tracking_uri == ARN
        assert mlflow.get_tracking_uri() == ARN
        assert plugin._get_current_session().get_credentials(
        ).access_key == 'AKIA'
    assert mlflow.get_tracking_uri() == tracking_uri

    other = f'sqlite:///{tmp_path / "other.db"}'
    with MlflowTracking('e', target=other).client() as client:
        assert mlflow.get_tracking_uri() == other
        assert client._tracking_client.tracking_uri == other
    assert mlflow.get_tracking_uri() == tracking_uri


def test_explicit_calls_take_turns(monkeypatch):
    target = target_module.Target('databricks', 'databricks-uc',
                                  swap=lambda: target_module.environ(
                                      {'KUMO_PROBE': 'x'}))
    seen = []

    def call():
        with target.scope():
            seen.append(os.environ.get('KUMO_PROBE'))

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert seen == ['x'] * 8 and 'KUMO_PROBE' not in os.environ


def test_ambient_sagemaker_calls_see_its_uri_and_restore_the_global_one(
        tracking_uri, monkeypatch):
    pytest.importorskip('sagemaker_mlflow')  # registers the arn store
    from kumo.engine.integrations.mlflow import MlflowTracking

    seen = []

    def lookup(self, name):
        seen.append((self.tracking_uri, mlflow.get_tracking_uri()))

    monkeypatch.setattr(mlflow.MlflowClient, 'get_experiment_by_name',
                        lookup)
    tracking = MlflowTracking(target=SageMakerMLflow(ARN),
                              experiment_name='e')
    kumo.MLOps('/tmp/x', tracking=tracking)

    assert seen == [(ARN, ARN)]
    assert mlflow.get_tracking_uri() == tracking_uri


def test_an_unreachable_server_fails_when_mlops_is_built(
        tmp_path, monkeypatch):
    from kumo.engine.integrations.mlflow import MlflowTracking

    monkeypatch.setenv('MLFLOW_HTTP_REQUEST_MAX_RETRIES', '0')
    tracking = MlflowTracking(target='http://127.0.0.1:9',
                              experiment_name='e')
    with pytest.raises(RuntimeError, match='MLflow at http://127.0.0.1:9 '
                       'is not usable with these credentials'):
        kumo.MLOps(str(tmp_path), tracking=tracking)
    with pytest.raises(ValueError, match='pass experiment_name'):
        kumo.MLOps(str(tmp_path),
                   tracking=MlflowTracking(target='http://127.0.0.1:9'))


def test_a_missing_unity_catalog_schema_fails_when_mlops_is_built(
        monkeypatch):
    from kumo.engine.integrations import mlflow as integration

    calls = []

    def http_request(creds, endpoint, method, **kwargs):
        calls.append(endpoint)
        return SimpleNamespace(
            status_code=404, reason='Not Found', json=lambda: {
                'error_code': 'CATALOG_DOES_NOT_EXIST',
                'message': "Catalog 'nope' does not exist."
            })

    monkeypatch.setattr('mlflow.utils.rest_utils.http_request', http_request)
    monkeypatch.setattr('mlflow.utils.databricks_utils.'
                        'get_databricks_host_creds', lambda uri: None)
    with pytest.raises(ValueError, match="cannot register 'nope.ml.churn': "
                       'CATALOG_DOES_NOT_EXIST'):
        integration.check_registry_schema('nope.ml.churn', 'databricks-uc')
    assert calls == ['/api/2.1/unity-catalog/schemas/nope.ml']
    integration.check_registry_schema('churn', 'databricks-uc')
    integration.check_registry_schema('a.b.c', 'sqlite:///x.db')
    assert len(calls) == 1


# Pins: MLflow and plugin internals the code above relies on.


def test_pin_set_tracking_uri_keeps_the_module_value(tracking_uri):
    utils = pytest.importorskip('mlflow.tracking._tracking_service.utils')
    assert utils._tracking_uri == tracking_uri


def test_pin_current_experiment_lookup():
    from mlflow.tracking.fluent import _get_experiment_id

    assert callable(_get_experiment_id)


def test_pin_sagemaker_signer_takes_the_thread_session(monkeypatch):
    boto3 = pytest.importorskip('boto3')
    pytest.importorskip('sagemaker_mlflow')
    from sagemaker_mlflow.auth import AuthBoto
    from sagemaker_mlflow.session import use_session

    session = boto3.Session(aws_access_key_id='AKIA',
                            aws_secret_access_key='s3cret')
    with use_session(session):
        auth = AuthBoto('us-west-2', 'sagemaker-mlflow')
    assert auth.creds.access_key == 'AKIA'


def test_pin_system_metric_monitors():
    from kumo.engine.integrations.mlflow import _metric_monitors

    monitors = _metric_monitors()
    assert len(monitors) >= 3
    for monitor in monitors:
        monitor.collect_metrics()
        assert isinstance(monitor.aggregate_metrics(), dict)
        monitor.clear_metrics()


def test_pin_databricks_auth_variables():
    from databricks.sdk.config import Config

    names = {attribute.env for attribute in Config.attributes()}
    assert {'DATABRICKS_HOST', 'DATABRICKS_TOKEN', 'DATABRICKS_CLIENT_ID',
            'DATABRICKS_CLIENT_SECRET', 'DATABRICKS_CONFIG_PROFILE'} <= names


def test_registering_a_registered_bundle_reuses_its_version(
        tracking_uri, caplog):
    from kumo.engine.integrations.mlflow import _register

    target = resolve(None)
    client = target.client()
    run = client.create_run('0')
    info = SimpleNamespace(model_uri=f'runs:/{run.info.run_id}/model',
                           run_id=run.info.run_id, model_id=None)
    location = kumo.ModelRegistryLocation('churn', alias='champion')

    first = _register(target, client, info, location,
                      {'kumo_bundle_id': 'a'})
    with caplog.at_level('INFO'):
        again = _register(target, client, info, location,
                          {'kumo_bundle_id': 'a'})
    other = _register(target, client, info, location,
                      {'kumo_bundle_id': 'b'})

    assert (first.version, again.version, other.version) == ('1', '1', '2')
    assert 'churn version 1 already holds bundle a; reused it' in caplog.text
    assert len(client.search_model_versions("name = 'churn'")) == 2
    assert str(client.get_model_version_by_alias('churn',
                                                 'champion').version) == '2'


def test_the_smoke_script_compiles(tmp_path):
    import py_compile
    from pathlib import Path

    for parent in Path(__file__).resolve().parents:
        script = parent / 'examples' / 'mlops' / 'smoke.py'
        if script.is_file():
            py_compile.compile(str(script), cfile=str(tmp_path / 'out'),
                               doraise=True)
            return
    pytest.skip('examples/mlops is not in this checkout')


def test_a_service_principal_client_ignores_a_token_lent_to_mlflow(
        monkeypatch):
    from kumo_databricks import session

    from kumo.engine.integrations._mlflow_target import _principal

    made = []
    monkeypatch.setattr(session, 'WorkspaceClient',
                        lambda **kwargs: made.append(kwargs))
    monkeypatch.setenv('DATABRICKS_TOKEN', 'short-lived')
    session._client_for(_principal(DatabricksIdentity(
        host='https://dbc.example', client_id='sp', client_secret='secret')))
    assert made[0]['auth_type'] == 'oauth-m2m'


def test_unity_catalog_search_results_with_tags_as_a_method():
    from kumo.engine.integrations.mlflow import _registered_version

    class Search:
        def __init__(self, version):
            self.version = version

        def tags(self):
            raise AssertionError('not a mapping')

    class Page(list):
        token = None

    held = {'1': {'kumo_bundle_id': 'other'}, '2': {'kumo_bundle_id': 'b'}}
    client = SimpleNamespace(
        search_model_versions=lambda *a, **k: Page([Search('1'),
                                                    Search('2')]),
        get_model_version=lambda name, version: SimpleNamespace(
            tags=held[version]))
    assert _registered_version(client, 'm', {'kumo_bundle_id': 'b'}) == '2'
    assert _registered_version(client, 'm', {'kumo_bundle_id': 'x'}) is None


def test_kumo_calls_drop_mlflow_request_header_errors_only(caplog):
    import logging

    from kumo.engine.integrations._mlflow_target import resolve

    headers = logging.getLogger('mlflow.tracking.request_header.registry')
    message = ('Encountered unexpected error during resolving request '
               'headers: %s')
    headers.addHandler(caplog.handler)  # MLflow's loggers do not propagate
    try:
        with caplog.at_level('WARNING', logger=headers.name):
            with resolve('sqlite:///x.db').scope():
                headers.warning(message, 'inside')
                headers.warning('another warning')
            headers.warning(message, 'outside')
    finally:
        headers.removeHandler(caplog.handler)
    assert 'inside' not in caplog.text
    assert 'another warning' in caplog.text and 'outside' in caplog.text
