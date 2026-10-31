# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Where `MlflowTracking` records, and the credentials that reach it."""

import logging
import os
import re
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Iterator, Mapping, Optional

from kumo.engine.connector._extras import import_provider

SAGEMAKER_ARN = re.compile(r'arn:aws[a-z-]*:sagemaker:([^:]*):[^:]*:'
                           r'(mlflow-tracking-server|mlflow-app)/(.+)')
_LOCK = threading.RLock()
_NO_LINKS = 'MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT'
_HEADERS = logging.getLogger('mlflow.tracking.request_header.registry')
_quiet = 0
_QUIET_LOCK = threading.Lock()


def _no_header_error(record: logging.LogRecord) -> bool:
    return not record.getMessage().startswith(
        'Encountered unexpected error during resolving request headers')


@contextmanager
def _without_header_warnings() -> Iterator[None]:
    global _quiet
    with _QUIET_LOCK:
        _quiet += 1
        if _quiet == 1:
            _HEADERS.addFilter(_no_header_error)
    try:
        yield
    finally:
        with _QUIET_LOCK:
            _quiet -= 1
            if not _quiet:
                _HEADERS.removeFilter(_no_header_error)


@dataclass(frozen=True)
class SageMakerMLflow:
    r"""A SageMaker managed MLflow tracking server or app, for
    ``MlflowTracking(target=...)``.

    Args:
        tracking_server_arn: the server's (or app's) ARN.
        identity: a ``kumo.engine.connector.S3Identity`` with an access key
            or a role to assume; `None` or ``S3Identity()`` uses the ambient
            AWS credential chain (an instance or execution role,
            ``aws configure``).
    """

    tracking_server_arn: str
    identity: Any = None

    def __post_init__(self) -> None:
        if not SAGEMAKER_ARN.fullmatch(self.tracking_server_arn):
            raise ValueError(
                f'{self.tracking_server_arn!r} is not a SageMaker MLflow '
                f'tracking server or app ARN.')


class Target:
    r"""A tracking URI, its registry URI and how to reach them."""

    def __init__(self, tracking_uri: str, registry_uri: Optional[str],
                 swap: Optional[Callable[[], ContextManager[Any]]] = None,
                 aws_session: Any = None, host: Optional[str] = None,
                 assets: Any = None) -> None:
        self.tracking_uri = tracking_uri
        self.assets = assets
        self.registry_uri = registry_uri
        self._swap = swap
        self._aws_session = aws_session
        self._host = host

    @property
    def platform(self) -> str:
        if SAGEMAKER_ARN.fullmatch(self.tracking_uri or ''):
            return 'sagemaker'
        if (self.tracking_uri or '').partition(':')[0] == 'databricks':
            return 'databricks'
        return 'mlflow'

    def run_url(self, experiment_id: str, run_id: str) -> Optional[str]:
        r"""The run's page in the MLflow UI, where there is one."""
        if self.platform == 'databricks':
            host = self._host or _workspace_host()
            return None if host is None else (
                f"{host.rstrip('/')}/ml/experiments/{experiment_id}/runs/"
                f'{run_id}')
        if self.platform == 'mlflow' and self.tracking_uri.startswith(
            ('http://', 'https://')):
            return (f"{self.tracking_uri.rstrip('/')}/#/experiments/"
                    f'{experiment_id}/runs/{run_id}')
        return None

    @property
    def explicit(self) -> bool:
        r"""Whether calls run with process-wide state set around them."""
        return self._swap is not None

    @contextmanager
    def scope(self) -> Iterator[None]:
        r"""Runs the block with this target's credentials."""
        links = {_NO_LINKS: 'true'} if self.platform == 'sagemaker' else {}
        with _without_header_warnings(), environ(links):
            if self._swap is None:
                yield
            else:
                with _LOCK, self._swap():
                    yield

    def client(self) -> Any:
        mlflow = import_provider('mlflow', 'mlflow')
        return mlflow.MlflowClient(self.tracking_uri, self.registry_uri)

    def artifact_repository(self, uri: str) -> Any:
        r"""An S3 repository signed as this target, or `None`."""
        if self._aws_session is None or not uri.startswith('s3://'):
            return None
        s3 = import_provider('mlflow.store.artifact.s3_artifact_repo',
                             'mlflow')
        frozen = self._aws_session.get_credentials().get_frozen_credentials()
        return s3.S3ArtifactRepository(uri, access_key_id=frozen.access_key,
                                       secret_access_key=frozen.secret_key,
                                       session_token=frozen.token)


def resolve(target: Any) -> Target:
    mlflow = import_provider('mlflow', 'mlflow')
    if isinstance(target, Target):
        return target
    if target is None:
        return Target(mlflow.get_tracking_uri(), mlflow.get_registry_uri())
    if isinstance(target, str):
        if SAGEMAKER_ARN.fullmatch(target):
            return _sagemaker(SageMakerMLflow(target))
        return Target(target, registry_for(target))
    if isinstance(target, SageMakerMLflow):
        return _sagemaker(target)
    from kumo.engine.connector.databricks import DatabricksConnector

    if isinstance(target, DatabricksConnector):
        return _databricks(target)
    raise TypeError(f'MlflowTracking(target=...) takes a DatabricksConnector, '
                    f'a SageMakerMLflow or a tracking URI, not '
                    f'{type(target).__name__}.')


def _workspace_host() -> Optional[str]:
    try:
        utils = import_provider('mlflow.utils.databricks_utils', 'mlflow')
        host, _ = utils.get_workspace_info_from_dbutils()
        if host:
            return host
    except Exception:
        pass
    try:
        return import_provider('databricks.sdk.config',
                               'databricks').Config().host
    except Exception:
        return None


def registry_for(tracking_uri: str) -> str:
    r"""Unity Catalog on Databricks, else the tracking server itself."""
    scheme, colon, rest = tracking_uri.partition(':')
    return f'databricks-uc{colon}{rest}' if scheme == 'databricks' else (
        tracking_uri)


def _principal(identity: Any) -> Any:
    from kumo_core.sources.principals import store

    return None if identity is None else store.get(identity.slug)


def _databricks(connector: Any) -> Target:
    from kumoapi.data_source import DatabricksAuthenticationKind

    principal = _principal(connector.identity)
    if (principal is None or principal.config.authentication
            is DatabricksAuthenticationKind.AMBIENT):
        return Target('databricks', 'databricks-uc',
                      swap=lambda: global_tracking_uri('databricks'),
                      assets=connector)
    client = import_provider('kumo_databricks.session',
                             'databricks')._client_for(principal)
    return Target('databricks', 'databricks-uc',
                  swap=lambda: _databricks_token(client),
                  host=client.config.host, assets=connector)


@contextmanager
def _databricks_token(client: Any) -> Iterator[None]:
    r"""A short-lived ``DATABRICKS_TOKEN``; the secret stays out."""
    config_type = import_provider('databricks.sdk.config',
                                  'databricks').Config
    authorization = client.config.authenticate()['Authorization']
    names = {
        attribute.env
        for attribute in config_type.attributes()
        if (attribute.env or '').startswith('DATABRICKS_')
    }
    with environ({
            **{name: None
               for name in names}, 'DATABRICKS_HOST': client.config.host,
            'DATABRICKS_TOKEN': authorization.partition(' ')[2]
    }), global_tracking_uri('databricks'):
        yield


def _sagemaker(target: SageMakerMLflow) -> Target:
    from kumoapi.data_source import S3AmbientAuthentication

    from kumo.engine.connector import S3Connector

    arn = target.tracking_server_arn
    assets = S3Connector(identity=target.identity)
    principal = _principal(target.identity)
    if principal is None or isinstance(principal.config.authentication,
                                       S3AmbientAuthentication):
        return Target(arn, arn, swap=lambda: global_tracking_uri(arn),
                      assets=assets)
    session = import_provider('kumo_s3.backend',
                              's3')._session_for(principal)
    return Target(arn, arn, swap=lambda: _sagemaker_session(arn, session),
                  aws_session=session, assets=assets)


@contextmanager
def _sagemaker_session(arn: str, session: Any) -> Iterator[None]:
    r"""The SageMaker plugin's state, signing with ``session``."""
    plugin = import_provider('sagemaker_mlflow.session', 'sagemaker')
    with ExitStack() as stack:
        stack.enter_context(environ({'SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN': None}))
        stack.enter_context(global_tracking_uri(arn))
        stack.enter_context(plugin.use_session(session))
        yield


@contextmanager
def global_tracking_uri(uri: str) -> Iterator[None]:
    r"""``mlflow.set_tracking_uri(uri)`` for the block."""
    mlflow = import_provider('mlflow', 'mlflow')
    utils = import_provider('mlflow.tracking._tracking_service.utils',
                            'mlflow')
    if mlflow.get_tracking_uri() == uri:
        yield
        return
    saved, saved_env = utils._tracking_uri, os.environ.get(
        'MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(uri)
    try:
        yield
    finally:
        mlflow.set_tracking_uri(saved)
        if saved_env is not None:
            os.environ['MLFLOW_TRACKING_URI'] = saved_env


@contextmanager
def environ(values: Mapping[str, Optional[str]]) -> Iterator[None]:
    r"""Sets (or, for `None`, unsets) environment variables for the block."""
    saved = {name: os.environ.get(name) for name in values}
    try:
        _apply(values)
        yield
    finally:
        _apply(saved)


def _apply(values: Mapping[str, Optional[str]]) -> None:
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
