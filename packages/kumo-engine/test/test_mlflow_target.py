# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""How `MlflowTracking` reaches its server."""

import os
import sys
import threading
from types import SimpleNamespace

import pytest
from packaging.version import Version

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


def _open_uc_storage(kind):
    from mlflow.protos.databricks_uc_registry_messages_pb2 import (
        TemporaryCredentials,
    )
    from mlflow.utils._unity_catalog_utils import (
        get_artifact_repo_from_storage_info,
    )

    token = TemporaryCredentials()
    getattr(token, kind).SetInParent()
    get_artifact_repo_from_storage_info('s3://bucket/model', token,
                                        lambda: token)


@pytest.mark.parametrize('module,kind,extra', [
    ('boto3', 'aws_temp_credentials', 's3'),
    ('azure.core.credentials', 'azure_user_delegation_sas', 'adls'),
    ('google.cloud.storage', 'gcp_oauth_token', 'gcs')])
def test_unity_catalog_names_the_extra_its_storage_needs(
        monkeypatch, module, kind, extra):
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match=(
            rf"MLflow needs '{module}[\w.]*' to reach this model's files in "
            rf"Unity Catalog, and it is not installed. Install it with: pip "
            rf"install 'kumo-engine\[{extra}\]'")):
        with target_module.Target('databricks', 'databricks-uc').scope():
            _open_uc_storage(kind)


def test_an_unwrapped_missing_client_names_its_extra_too():
    with pytest.raises(ImportError, match=r"pip install 'kumo-engine\[s3\]'"):
        with target_module.Target('databricks', 'databricks-uc').scope():
            raise ModuleNotFoundError(name='boto3')


def test_other_missing_modules_pass_through(monkeypatch):
    monkeypatch.setitem(sys.modules, 'boto3', None)
    with pytest.raises(mlflow.exceptions.MlflowException):
        with target_module.Target('databricks', 'databricks').scope():
            _open_uc_storage('aws_temp_credentials')
    with pytest.raises(RuntimeError):
        with target_module.Target('databricks', 'databricks-uc').scope():
            raise RuntimeError() from ModuleNotFoundError(name='snowflake')
    for registry, module in (('databricks-uc', 'snowflake'),
                             ('databricks-uc', 'google.protobuf'),
                             ('databricks', 'boto3')):
        with pytest.raises(ModuleNotFoundError) as raised:
            with target_module.Target('databricks', registry).scope():
                raise ModuleNotFoundError(name=module)
        assert type(raised.value) is ModuleNotFoundError
    nested = RuntimeError()
    nested.__cause__ = ModuleNotFoundError(name='boto3')
    for cause in (None, nested):
        with pytest.raises(RuntimeError):
            with target_module.Target('databricks', 'databricks-uc').scope():
                raise RuntimeError() from cause


def test_ambient_identities_set_no_credentials(monkeypatch, tracking_uri):
    monkeypatch.setenv('DATABRICKS_TOKEN', 'mine')
    monkeypatch.setenv('MLFLOW_TRACKING_URI', tracking_uri)
    target = resolve(_connector(DatabricksIdentity()))
    before = dict(os.environ)
    with target.scope():
        assert mlflow.get_tracking_uri() == 'databricks'
        assert {k: v for k, v in os.environ.items()
                if k.startswith('DATABRICKS_')} == {'DATABRICKS_TOKEN': 'mine'}
    assert mlflow.get_tracking_uri() == tracking_uri
    assert dict(os.environ) == before

    assert resolve(_connector(DatabricksIdentity())).assets.catalog == 'main'
    for target in (resolve(SageMakerMLflow(ARN, S3Identity())),
                   resolve(SageMakerMLflow(ARN))):
        assert type(target.assets).__name__ == 'S3Connector'
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
    pytest.importorskip('sagemaker_mlflow')
    from kumo.engine.integrations.mlflow import MlflowTracking

    seen = []

    def lookup(self, name):
        seen.append((self.tracking_uri, mlflow.get_tracking_uri()))
        return SimpleNamespace(experiment_id='1')

    monkeypatch.setattr(mlflow.MlflowClient, 'get_experiment_by_name',
                        lookup)
    tracking = MlflowTracking(target=SageMakerMLflow(ARN),
                              experiment_name='e')
    kumo.MLOps('/tmp/x', tracking=tracking)

    assert seen == [(ARN, ARN)]
    assert mlflow.get_tracking_uri() == tracking_uri


NO_LINKS = 'MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT'


def test_sagemaker_calls_print_no_mlflow_run_links(monkeypatch):
    from contextlib import nullcontext

    monkeypatch.delenv(NO_LINKS, raising=False)
    seen = []
    for uri in (ARN, 'http://mlflow'):
        with target_module.Target(uri, uri, swap=nullcontext).scope():
            seen.append(os.environ.get(NO_LINKS))

    assert (seen, os.environ.get(NO_LINKS)) == (['true', None], None)


def test_pin_mlflow_reads_its_run_link_switch_per_call():
    from mlflow.environment_variables import (
        MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT,
    )

    assert MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT.name == NO_LINKS


@pytest.mark.skipif(Version(mlflow.__version__) < Version('3.17'),
                    reason='this MLflow still accepts folder stores')
@pytest.mark.parametrize('store', ['file', 'path'])
def test_a_refused_folder_store_says_to_use_sqlite(tmp_path, monkeypatch,
                                                   store):
    from kumo.engine.integrations.mlflow import MlflowTracking

    monkeypatch.delenv('MLFLOW_ALLOW_FILE_STORE', raising=False)
    uri = (tmp_path / 'mlruns').as_uri() if store == 'file' else str(
        tmp_path / 'mlruns')
    tracking = MlflowTracking(target=uri, experiment_name='e')
    with pytest.raises(RuntimeError) as raised:
        kumo.MLOps(str(tmp_path / 'root'), tracking=tracking)
    assert str(raised.value) == (
        f'MLflow refused the local folder store at {uri}: use its default '
        f'sqlite:///mlflow.db or another tracking backend.')


def test_an_unreachable_server_fails_when_mlops_is_built(
        tmp_path, monkeypatch):
    from kumo.engine.integrations.mlflow import MlflowTracking

    monkeypatch.setenv('MLFLOW_HTTP_REQUEST_MAX_RETRIES', '0')
    tracking = MlflowTracking(target='http://127.0.0.1:9',
                              experiment_name='e')
    with pytest.raises(RuntimeError,
                       match='MLflow at http://127.0.0.1:9 is not usable: '):
        kumo.MLOps(str(tmp_path), tracking=tracking)
    with pytest.raises(ValueError, match='needs experiment_name'):
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
    from kumo.engine.integrations.mlflow import _register, _registered_version

    target = resolve(None)
    client = target.client()
    run = client.create_run('0')
    info = SimpleNamespace(model_uri=f'runs:/{run.info.run_id}/model',
                           run_id=run.info.run_id, model_id=None)
    location = kumo.ModelRegistryLocation('churn', alias='champion')

    def register(bundle):
        pointers = {'kumo_bundle_id': bundle}
        return _register(target, client, info, location, pointers,
                         _registered_version(client, 'churn', pointers))

    first = register('a')
    with caplog.at_level('INFO'):
        again = register('a')
    other = register('b')

    assert (first.version, again.version, other.version) == ('1', '1', '2')
    assert 'churn version 1 already holds bundle a; reused it' in caplog.text
    assert len(client.search_model_versions("name = 'churn'")) == 2
    assert str(client.get_model_version_by_alias('churn',
                                                 'champion').version) == '2'


def test_a_registry_reference_downloads_from_the_target_registry(
        tracking_uri, tmp_path):
    from kumo.engine.integrations.mlflow import _download

    uri = f'sqlite:///{tmp_path / "target.db"}'
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment = client.create_experiment(
        'e', artifact_location=str(tmp_path / 'artifacts'))
    model = client.create_logged_model(experiment, name='m')
    (tmp_path / 'bundle').mkdir()
    (tmp_path / 'bundle' / 'MLmodel').write_text('kumo')
    client.log_model_artifacts(model.model_id, str(tmp_path / 'bundle'))
    client.create_registered_model('m')
    client.create_model_version('m', model.model_uri, model_id=model.model_id)
    client.set_registered_model_alias('m', 'champion', '1')
    mlflow.set_tracking_uri(tracking_uri)

    for name, reference in (('alias', 'm@champion'), ('version', 'm/1'),
                            ('latest', 'm/latest'), ('id', model.model_id)):
        local = _download(mlflow, resolve(uri), f'models:/{reference}',
                          tmp_path / name)
        assert open(f'{local}/MLmodel').read() == 'kumo'
    local = _download(mlflow, resolve(uri), 'models:/m@champion/MLmodel',
                      tmp_path / 'file')
    assert open(local).read() == 'kumo'


def test_a_nested_path_goes_before_the_location_query():
    from kumo.engine.integrations.mlflow import _registry_download_uri

    client = SimpleNamespace(
        get_model_version_download_uri=lambda name, version: 's3://b/m?x=y')
    assert _registry_download_uri(
        client, 'models:/m/1/MLmodel') == 's3://b/m/MLmodel?x=y'


@pytest.mark.parametrize('registry',
                         ['databricks-uc', 'databricks', 'uc:http://uc:8080'])
def test_a_databricks_or_unity_catalog_reference_stays_on_mlflow_path(
        registry, tmp_path):
    from kumo.engine.integrations.mlflow import _download

    calls = []
    fake = SimpleNamespace(artifacts=SimpleNamespace(
        download_artifacts=lambda **kwargs: calls.append(kwargs) or ''))
    target = target_module.Target('http://tracking:5000', registry)
    _download(fake, target, 'models:/main.s.m@champion', tmp_path)

    assert calls[0]['artifact_uri'] == 'models:/main.s.m@champion'


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
    headers.addHandler(caplog.handler)
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
