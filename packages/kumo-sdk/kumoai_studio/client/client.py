# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from kumoai_studio.client.connector import ConnectorAPI
from kumoai_studio.client.endpoints import Endpoint, HTTPMethod
from kumoai_studio.client.graph import GraphAPI
from kumoai_studio.client.groups import GroupsAPI
from kumoai_studio.client.identity import IdentityAPI
from kumoai_studio.client.jobs import (
    ArtifactExportJobAPI,
    BatchPredictionJobAPI,
    GeneratePredictionTableJobAPI,
    GenerateTrainTableJobAPI,
    TrainingJobAPI,
)
from kumoai_studio.client.pquery import PQueryAPI
from kumoai_studio.client.projects import ProjectsAPI
from kumoai_studio.client.source_table import SourceTableAPI
from kumoai_studio.client.table import TableAPI

API_VERSION = 'v1'

logger = logging.getLogger('kumoai_studio')

# Status codes the Kumo backend uses to signal authentication / authorization
# failures. The Kumo REST service raises ``HTTPException`` with ``401`` for
# missing/invalid/expired/blocked API keys and ``403`` for valid but
# insufficient permissions (see ``rest/deps/authentication.py``). A ``404``
# from the backend means the endpoint itself was not found (wrong URL /
# version skew / proxy misroute), NOT an auth failure — so we deliberately
# do not treat it as one.
_AUTH_STATUS_CODES = frozenset({401, 403})


def _raise_init_error(url: str, exc: BaseException) -> None:
    """Translate a ``requests`` exception raised during client init into a
    user-facing ``ValueError`` with an accurate failure-class message.

    Splits errors into three buckets so callers know whether to rotate
    credentials, check connectivity, or escalate to operators:

    * ``401`` / ``403`` -> authentication / authorization failure
    * any other HTTP status (including 404, 5xx) -> server-side error
    * connect / timeout / other transport errors -> unreachable server
    """
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        status_code = (response.status_code if response is not None else None)
        body = response.text if response is not None else ''
        if status_code in _AUTH_STATUS_CODES:
            raise ValueError(
                f"Client authentication failed for {url!r}. Please check "
                "that your API key / credentials are valid. "
                f"HTTP {status_code}: {body}") from exc
        raise ValueError(
            f"Kumo server at {url!r} returned HTTP {status_code} while "
            f"initializing the SDK. Response: {body}") from exc
    if isinstance(exc, requests.exceptions.Timeout):
        raise ValueError(f"Timed out connecting to Kumo server at {url!r}. "
                         f"Original error: {exc}") from exc
    if isinstance(exc, requests.exceptions.ConnectionError):
        # Server unreachable (devserver not running, wrong port, DNS,
        # firewall). Don't mislead users into rotating credentials.
        raise ValueError(
            f"Could not connect to Kumo server at {url!r}. Verify the "
            f"server is running and the URL is correct. Original error: "
            f"{exc}") from exc
    # Some other RequestException (e.g. SSLError, InvalidURL, ChunkedEncoding).
    # Don't claim it's an auth issue.
    raise ValueError(f"Failed to initialize Kumo SDK client against {url!r}. "
                     f"Exception: {exc}") from exc


class KumoClient:
    r"""Low-level authenticated client for the Kumo public API.

    Most users should call :func:`kumoai_studio.init`, which authenticates this
    client and selects an active project. The typed API properties are useful
    for advanced callers that need direct access to project-owned REST
    resources.
    """
    def __init__(
        self,
        url: str,
        api_key: Optional[str],
        verify_ssl: bool = True,
    ) -> None:
        r"""Creates a client against the Kumo public API, provided a URL of
        the endpoint and an authentication token.

        Args:
            url: the public API endpoint URL.
            api_key: the public API authentication token.
            verify_ssl: whether to verify SSL certificates. Set to False to
                skip SSL certificate verification (equivalent to curl -k).
        """
        self._url = url if '?' in url or '#' in url else url.rstrip('/')
        self._api_url = f"{self._url}/{API_VERSION}"
        self._api_key = api_key
        self._verify_ssl = verify_ssl

        retry_strategy = Retry(
            total=10,  # Maximum number of retries
            connect=3,  # How many connection-related errors to retry on
            read=3,  # How many times to retry on read errors
            status=5,  # How many times to retry on bad status codes (below)
            # Status codes to retry on.
            status_forcelist=[408, 429, 500, 502, 503, 504],
            # Exponential backoff factor: 2, 4, 8, 16 seconds delay)
            backoff_factor=2.0,
        )
        http_adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount('http://', http_adapter)
        session.mount('https://', http_adapter)
        self._session = session
        if self._api_key:
            self._session.headers.update({"X-API-Key": self._api_key})

    def authenticate(self) -> None:
        r"""Validate the configured credentials without requiring a project.

        The group-neutral authentication endpoint verifies credentials before
        :func:`kumoai_studio.init` resolves the active project.

        Raises:
            :class:`ValueError`: If credentials are rejected, the endpoint is
                unreachable, or the server returns an unexpected response.
        """
        try:
            response = self._session.get(self._url + '/v1/authentication',
                                         verify=self._verify_ssl)
            response.raise_for_status()
        except requests.RequestException as e:
            _raise_init_error(self._url, e)

    @property
    def artifact_export_api(self) -> ArtifactExportJobAPI:
        r"""Returns the artifact export API."""
        return ArtifactExportJobAPI(self)

    @property
    def connector_api(self) -> ConnectorAPI:
        r"""Returns the typed connector API."""
        return ConnectorAPI(self)

    @property
    def identity_api(self) -> IdentityAPI:
        r"""Returns credential-free project identity discovery."""
        return IdentityAPI(self)

    @property
    def source_table_api(self) -> SourceTableAPI:
        r"""Returns the typed source table API."""
        return SourceTableAPI(self)

    @property
    def table_api(self) -> TableAPI:
        r"""Returns the typed Kumo Table (snapshot) API."""
        return TableAPI(self)

    @property
    def graph_api(self) -> GraphAPI:
        r"""Returns the typed Graph (metadata and snapshot) API."""
        return GraphAPI(self)

    @property
    def pquery_api(self) -> PQueryAPI:
        r"""Returns the typed Predictive Query API."""
        return PQueryAPI(self)

    @property
    def training_job_api(self) -> TrainingJobAPI:
        r"""Returns the typed Training Job API."""
        return TrainingJobAPI(self)

    @property
    def batch_prediction_job_api(self) -> BatchPredictionJobAPI:
        return BatchPredictionJobAPI(self)

    @property
    def generate_train_table_job_api(self) -> GenerateTrainTableJobAPI:
        r"""Returns the typed Generate-Train-Table Job API."""
        return GenerateTrainTableJobAPI(self)

    @property
    def generate_prediction_table_job_api(
            self) -> GeneratePredictionTableJobAPI:
        return GeneratePredictionTableJobAPI(self)

    @property
    def groups_api(self) -> GroupsAPI:
        r"""Returns the typed API for group operations."""
        return GroupsAPI(self)

    @property
    def projects_api(self) -> ProjectsAPI:
        r"""Returns the typed API for project operations."""
        return ProjectsAPI(self)

    def _inject_project_param(self, endpoint: str, kwargs: dict) -> dict:
        r"""Inject the active project only for legacy unscoped routes.

        Routes whose path already begins with ``/projects/{id}`` are fully
        scoped by that immutable path and must never acquire a conflicting
        process-global query parameter. An explicit ``project_id`` supplied by
        a low-level caller is also preserved.
        """
        from kumoai_studio import global_state
        params = dict(kwargs.get('params', None) or {})
        is_project_path = endpoint.lstrip('/').startswith('projects/')
        if not is_project_path and global_state._project_id is not None:
            params.setdefault('project_id', global_state._project_id)
        if params:
            kwargs['params'] = params
        return kwargs

    def _request(self, endpoint: Endpoint, **kwargs: Any) -> requests.Response:
        r"""Send a HTTP request to the specified endpoint."""
        endpoint_str = endpoint.get_path()
        if endpoint.method == HTTPMethod.GET:
            return self._get(endpoint_str, **kwargs)
        elif endpoint.method == HTTPMethod.POST:
            return self._post(endpoint_str, **kwargs)
        elif endpoint.method == HTTPMethod.PATCH:
            return self._patch(endpoint_str, **kwargs)
        elif endpoint.method == HTTPMethod.PUT:
            return self._put(endpoint_str, **kwargs)
        elif endpoint.method == HTTPMethod.DELETE:
            return self._delete(endpoint_str, **kwargs)
        else:
            raise ValueError(f"Unsupported HTTP method: {endpoint.method}")

    def _get(self, endpoint: str, **kwargs: Any) -> requests.Response:
        r"""Send a GET request to the specified endpoint, with keyword
        arguments, returned objects, and exceptions raised corresponding to
        :meth:`requests.Session.get`.
        """
        kwargs = self._inject_project_param(endpoint, kwargs)
        url = self._format_endpoint_url(endpoint)
        return self._session.get(url=url, verify=self._verify_ssl, **kwargs)

    def _post(self, endpoint: str, **kwargs: Any) -> requests.Response:
        r"""Send a POST request to the specified endpoint, with keyword
        arguments, returned objects, and exceptions raised corresponding to
        :meth:`requests.Session.post`.
        """
        kwargs = self._inject_project_param(endpoint, kwargs)
        url = self._format_endpoint_url(endpoint)
        return self._session.post(url=url, verify=self._verify_ssl, **kwargs)

    def _patch(self, endpoint: str, **kwargs: Any) -> requests.Response:
        r"""Send a PATCH request to the specified endpoint, with keyword
        arguments, returned objects, and exceptions raised corresponding to
        :meth:`requests.Session.patch`.
        """
        kwargs = self._inject_project_param(endpoint, kwargs)
        url = self._format_endpoint_url(endpoint)
        return self._session.patch(url=url, verify=self._verify_ssl, **kwargs)

    def _put(self, endpoint: str, **kwargs: Any) -> requests.Response:
        r"""Send a PUT request to the specified endpoint, with keyword
        arguments, returned objects, and exceptions raised corresponding to
        :meth:`requests.Session.put`.
        """
        kwargs = self._inject_project_param(endpoint, kwargs)
        url = self._format_endpoint_url(endpoint)
        return self._session.put(url=url, verify=self._verify_ssl, **kwargs)

    def _delete(self, endpoint: str, **kwargs: Any) -> requests.Response:
        r"""Send a DELETE request to the specified endpoint, with keyword
        arguments, returned objects, and exceptions raised corresponding to
        :meth:`requests.Session.delete`.
        """
        kwargs = self._inject_project_param(endpoint, kwargs)
        url = self._format_endpoint_url(endpoint)
        return self._session.delete(url=url, verify=self._verify_ssl, **kwargs)

    def _format_endpoint_url(self, endpoint: str) -> str:
        if endpoint[0] == "/":
            endpoint = endpoint[1:]
        return f"{self._api_url}/{endpoint}"
