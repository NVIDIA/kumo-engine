# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import os
from collections.abc import Callable
from typing import Optional
from unittest.mock import Mock

import pytest
from azure.core import MatchConditions
from azure.core.exceptions import ResourceExistsError

from io_utils.fsutil.impl_adls import threadsafe_adls

ENV_RUN_ADLS_TESTS = 'KUMO_RUN_ADLS_TESTS'
ENV_ADLS_TEST_STORAGE_ACCOUNT = 'AZURE_STORAGE_ACCOUNT'
ENV_ADLS_TEST_CONTAINER = 'AZURE_CONTAINER_NAME'


def adls_test_base_uri() -> str:
    """Return the configured, isolated ADLS integration-test filesystem."""
    storage_account = os.getenv(ENV_ADLS_TEST_STORAGE_ACCOUNT)
    container = os.getenv(ENV_ADLS_TEST_CONTAINER)
    if not storage_account or not container:
        raise RuntimeError(
            'ADLS integration tests require AZURE_STORAGE_ACCOUNT and '
            'AZURE_CONTAINER_NAME')
    return f'abfss://{container}@{storage_account}.dfs.core.windows.net'


def _adls_credentials_configured(config_path: Optional[str] = None) -> bool:
    """Return whether an explicit, locally detectable Azure auth is present.

    This check deliberately performs no Azure operation. Test decorators run
    while pytest imports modules, where probing ``DefaultAzureCredential`` can
    make collection wait on IMDS/Azure CLI discovery once for every decorated
    test. Invalid or insufficient configured credentials are exercised by the
    integration test itself.
    """
    if os.getenv(ENV_RUN_ADLS_TESTS, '').lower() in ('1', 'true', 'yes', 'on'):
        return True

    creds_path = config_path or os.getenv('AZURE_CREDENTIALS_PATH')
    if creds_path:
        return os.path.isfile(os.path.expanduser(creds_path))

    if os.getenv('AZURE_STORAGE_CONNECTION_STRING'):
        return True

    tenant_id = os.getenv('AZURE_TENANT_ID')
    client_id = os.getenv('AZURE_CLIENT_ID')
    if (tenant_id and client_id
            and (os.getenv('AZURE_CLIENT_SECRET')
                 or os.getenv('AZURE_FEDERATED_TOKEN_FILE'))):
        return True

    # App Service / Functions and legacy managed-identity endpoints are
    # explicit signals that DefaultAzureCredential can use managed identity
    # without probing the public IMDS endpoint during local test collection.
    return any(
        os.getenv(name)
        for name in ('IDENTITY_ENDPOINT', 'MSI_ENDPOINT', 'IMDS_ENDPOINT'))


def withADLSCredentials(
    func=None, config_path: Optional[str] = None, test_path:
    str = 'abfss://kumopublic@kumopublicdatasets.dfs.core.windows.net/Kumo-Experience-Bug-Bash-Datasets/flight-delays/csv'
):
    """
    Decorator to skip tests if ADLS credentials are not available.

    Usage:
        @withADLSCredentials
        def test_something():
            ...

        @withADLSCredentials()
        def test_something_else():
            ...

        @withADLSCredentials(config_path="/path/to/config.json")
        def test_with_config():
            ...

    Args:
        func: The test function (when used without parentheses)
        config_path: Path to JSON config file with credentials. If None, tests with existing credentials.
        test_path: ADLS path to test access (default: public datasets)

    Config JSON format (optional):
    {
        "azure_ad": {
            "tenant_id": "xxx",
            "client_id": "xxx",
            "client_secret": "xxx"
        },
        "storage": {
            "storage_account_name": "xxx",
            "container_name": "xxx"
        }
    }
    """
    def decorator(test_func: Callable):
        credentials_missing = not _adls_credentials_configured(config_path)
        if inspect.isgeneratorfunction(test_func):
            # A decorator below @pytest.fixture receives the raw generator.
            # Marking fixtures is unsupported by pytest; a generator wrapper
            # skips at fixture setup before the yielded resource can be used.
            from kumo.testing._skip import skip_if
            return skip_if(credentials_missing,
                           "No Azure credentials found for ADLS", test_func)
        # Use a pytest mark rather than a runtime wrapper. Runtime wrappers
        # skip only after pytest has resolved fixtures; ADLS integration
        # fixtures perform provider I/O during setup and would therefore hang
        # before the wrapper got a chance to skip.
        return pytest.mark.skipif(
            credentials_missing,
            reason="No Azure credentials found for ADLS",
        )(test_func)

    # Handle both @withADLSCredentials and @withADLSCredentials() syntax
    if func is not None:
        # Called as @withADLSCredentials (without parentheses)
        return decorator(func)
    else:
        # Called as @withADLSCredentials() or @withADLSCredentials(config_path=...)
        return decorator


# Helper function to load ADLS credentials for use in tests
def load_adls_config(config_path: Optional[str] = None) -> dict:
    """
    Load ADLS credentials configuration from JSON file.

    Returns:
        config dict with azure_ad and storage sections
    """
    creds_path = config_path or os.getenv(
        'AZURE_CREDENTIALS_PATH',
        os.path.expanduser('~/azure/azure_test_credentials.json'))

    with open(creds_path, 'r') as f:
        return json.load(f)


@pytest.fixture
def adls_client():
    """Pytest fixture that provides a mocked ADLS client"""
    # Save Azure environment variables
    tenant_id = os.environ.get('AZURE_TENANT_ID')
    client_id = os.environ.get('AZURE_CLIENT_ID')
    client_secret = os.environ.get('AZURE_CLIENT_SECRET')
    storage_account_key = os.environ.get('AZURE_STORAGE_ACCOUNT_KEY')
    connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')

    # Save old ADLS contexts
    from io_utils.fsutil import impl_adls
    old_impl_ctx = impl_adls.adls_ctx

    old_data_lake_client_fn = threadsafe_adls.data_lake_client

    # Setup mocks
    mock_client = MockADLSClient()
    mock_ctx = MockADLSContext(mock_client)

    impl_adls.adls_ctx = mock_ctx
    threadsafe_adls.data_lake_client = (
        lambda storage_account=None, endpoint_suffix=None: mock_client)

    yield mock_client

    # Cleanup (runs after test)
    for key, value in [
        ('AZURE_TENANT_ID', tenant_id),
        ('AZURE_CLIENT_ID', client_id),
        ('AZURE_CLIENT_SECRET', client_secret),
        ('AZURE_STORAGE_ACCOUNT_KEY', storage_account_key),
        ('AZURE_STORAGE_CONNECTION_STRING', connection_string),
    ]:
        if value:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]

    impl_adls.adls_ctx = old_impl_ctx
    threadsafe_adls.data_lake_client = old_data_lake_client_fn


class MockADLSContext:
    """Mock ADLSContext that uses MockADLSClient"""
    def __init__(self, mock_client):
        self._mock_client = mock_client

    @property
    def client(self):
        """Return the mock client"""
        return self._mock_client

    def get_file_client(self, uri):
        """Get file client for a specific path"""
        fs_client = self._mock_client.get_file_system_client(uri.container)
        result = fs_client.get_file_client(uri.path)
        return result

    def get_directory_client(self, uri):
        """Get directory client for a specific path"""
        fs_client = self._mock_client.get_file_system_client(uri.container)
        return fs_client.get_directory_client(uri.path)


class MockADLSClient:
    """Mock Azure Data Lake Storage client for testing"""
    def __init__(self):
        self._containers = {}
        self._files = {}
        self._metadata = {}
        self._copy_status = {}

    def create_container(self, container_name: str):
        """Mock container creation"""
        self._containers[container_name] = {}
        return Mock(name=container_name)

    def get_file_system_client(self, file_system: str):
        """Get a mock file system (container) client"""
        if file_system not in self._containers:
            raise Exception(f"Container {file_system} does not exist")

        mock_fs = Mock()
        mock_fs.file_system_name = file_system
        mock_fs.create_directory = lambda path: self._create_directory(
            file_system, path)
        mock_fs.get_directory_client = lambda path: self._get_directory_client(
            file_system, path)
        mock_fs.get_file_client = lambda path: self._get_file_client(
            file_system, path)
        mock_fs.get_paths = lambda path="", recursive=True, max_results=None: self._get_paths(
            file_system, path, recursive)

        return mock_fs

    def create_file(self, uri: str, content: str):
        """Helper to create a file with content"""
        # Parse the abfss:// URI
        parts = uri.replace('abfss://', '').split('@')
        container = parts[0]
        rest = parts[1].split('.dfs.core.windows.net/')
        path = rest[1] if len(rest) > 1 else ''

        key = f"{container}/{path}"
        self._files[key] = content.encode() if isinstance(content,
                                                          str) else content

        # Auto-create parent directories
        path_parts = path.split('/')
        for i in range(len(path_parts) - 1):
            dir_path = '/'.join(path_parts[:i + 1])
            dir_key = f"{container}/{dir_path}/"
            if dir_key not in self._files:
                self._files[dir_key] = b""

    def _create_directory(self, file_system: str, path: str):
        """Mock directory creation"""
        # Normalize path - ensure trailing slash for directories
        normalized_path = path.rstrip('/') + '/'
        key = f"{file_system}/{normalized_path}"
        if key not in self._files:
            self._files[key] = b""
        return Mock(path=path)

    def _is_directory(self, file_system: str, path: str) -> bool:
        """Check if a path is a directory"""
        # Normalize path
        normalized_path = path.rstrip('/')
        dir_key = f"{file_system}/{normalized_path}/"

        # It's a directory if:
        # 1. The path/ exists as a key
        # 2. OR there are files under this path
        if dir_key in self._files:
            return True

        # Check if any files start with this prefix
        prefix = f"{file_system}/{normalized_path}/"
        return any(k.startswith(prefix) for k in self._files.keys())

    def _get_directory_client(self, file_system: str, path: str):
        """Get a mock directory client"""
        mock_dir = Mock()
        mock_dir.path = path
        mock_dir.create_directory = lambda: self._create_directory(
            file_system, path)
        mock_dir.delete_directory = lambda: self._delete_path(file_system, path)
        mock_dir.exists = lambda: self._exists(file_system, path)

        # Add get_paths method for directory listing
        def get_paths(recursive=True, max_results=None, **kwargs):
            # List paths under this directory
            search_path = path.rstrip('/')
            return self._get_paths(file_system, search_path, recursive)

        mock_dir.get_paths = get_paths

        def get_directory_properties(**kwargs):
            if not self._is_directory(file_system, path):
                from azure.core.exceptions import ResourceNotFoundError
                raise ResourceNotFoundError(f"Directory not found: {path}")

            # Return a Mock that supports both attribute and dict access
            props = Mock()
            props.name = path.rstrip('/').split('/')[-1] if path else ''
            props.size = 0
            props.is_directory = True
            props.metadata = {'hdi_isfolder': 'true'}
            # Also support dict-style access for isdir() check
            props.get = lambda k, default=None: {
                'name': props.name,
                'size': props.size,
                'is_directory': props.is_directory,
                'metadata': props.metadata
            }.get(k, default)

            return props

        mock_dir.get_directory_properties = get_directory_properties
        return mock_dir

    def _get_file_client(self, file_system: str, path: str):
        """Get a mock file client"""
        mock_file = Mock()
        mock_file.path = path
        mock_file.url = (
            f"https://mock.dfs.core.windows.net/{file_system}/{path}")

        def create_file(**kwargs):
            """Create an empty file"""
            key = f"{file_system}/{path}"
            if (kwargs.get('match_condition') == MatchConditions.IfMissing
                    and 'etag' in kwargs):
                raise TypeError('IfMissing must not include etag')
            if (key in self._files and kwargs.get('match_condition')
                    == MatchConditions.IfMissing):
                raise ResourceExistsError(f"File already exists: {path}")
            self._files[key] = b""
            self._metadata[key] = kwargs.get('metadata') or {}

            # Auto-create parent directories
            path_parts = path.split('/')
            for i in range(len(path_parts) - 1):
                dir_path = '/'.join(path_parts[:i + 1])
                dir_key = f"{file_system}/{dir_path}/"
                if dir_key not in self._files:
                    self._files[dir_key] = b""

            return mock_file

        def upload_data(data, overwrite=True, **kwargs):
            key = f"{file_system}/{path}"
            if (kwargs.get('match_condition') == MatchConditions.IfMissing
                    and 'etag' in kwargs):
                raise TypeError('IfMissing must not include etag')
            if not overwrite and kwargs.get('metadata') is not None:
                raise ValueError(
                    'metadata can be set only when overwrite is enabled')
            if (key in self._files and kwargs.get('match_condition')
                    == MatchConditions.IfMissing):
                raise ResourceExistsError(f"File already exists: {path}")

            # Handle different data types
            if data is None:
                data = b""
            elif isinstance(data, str):
                data = data.encode()
            elif hasattr(data, 'read'):
                # It's a file-like object, read from it
                if hasattr(data, 'seek'):
                    data.seek(0)  # Reset to beginning
                data = data.read()
                if isinstance(data, str):
                    data = data.encode()

            # Ensure it's bytes
            if not isinstance(data, bytes):
                data = bytes(data)

            self._files[key] = data
            self._metadata[key] = kwargs.get('metadata') or {}

            # Auto-create parent directories (same as create_file)
            path_parts = path.split('/')
            for i in range(len(path_parts) - 1):
                dir_path = '/'.join(path_parts[:i + 1])
                dir_key = f"{file_system}/{dir_path}/"
                if dir_key not in self._files:
                    self._files[dir_key] = b""

        def download_file():
            key = f"{file_system}/{path}"
            content = self._files.get(key, b"")
            mock_download = Mock()
            mock_download.readall = lambda: content

            def readinto(file_obj):
                file_obj.write(content)
                return len(content)

            mock_download.readinto = readinto

            return mock_download

        def delete_file():
            return self._delete_path(file_system, path)

        def exists():
            key = f"{file_system}/{path}"
            result = key in self._files
            return result

        def get_file_properties(**kwargs):
            key = f"{file_system}/{path}"
            if key not in self._files:
                from azure.core.exceptions import ResourceNotFoundError
                raise ResourceNotFoundError(f"File not found: {path}")
            content = self._files.get(key, b"")

            # Return a Mock that supports both attribute and dict access
            props = Mock()
            props.name = path.split('/')[-1]
            props.size = len(content)
            props.is_directory = False
            props.metadata = {}
            props.metadata = self._metadata.get(key, {})
            props.etag = f'"{len(content)}-{hash(content)}"'
            # Also support dict-style access
            props.get = lambda k, default=None: {
                'name': props.name,
                'size': props.size,
                'is_directory': props.is_directory,
                'metadata': props.metadata
            }.get(k, default)

            copy_status = self._copy_status.get(key)
            if copy_status is not None:
                props.copy = Mock(status=copy_status)

            return props

        def start_copy_from_url(source_url: str, **kwargs):
            if ".dfs.core.windows.net/" not in source_url:
                raise ValueError(f"Invalid source URL: {source_url}")
            _, path_part = source_url.split(".dfs.core.windows.net/", 1)
            if "/" not in path_part:
                raise ValueError(f"Invalid source URL path: {source_url}")
            src_container, src_path = path_part.split("/", 1)
            src_key = f"{src_container}/{src_path}"
            dst_key = f"{file_system}/{path}"
            if src_key not in self._files:
                from azure.core.exceptions import ResourceNotFoundError
                raise ResourceNotFoundError(f"File not found: {src_path}")
            self._files[dst_key] = self._files[src_key]
            self._copy_status[dst_key] = "success"

        def rename_file(new_name: str, **kwargs):
            destination_container, destination_path = new_name.split('/', 1)
            if (kwargs.get('match_condition') == MatchConditions.IfMissing
                    and 'etag' in kwargs):
                raise TypeError('IfMissing must not include etag')
            source_key = f"{file_system}/{path}"
            if source_key not in self._files:
                from azure.core.exceptions import ResourceNotFoundError
                raise ResourceNotFoundError(f"File not found: {path}")
            destination_key = f"{destination_container}/{destination_path}"
            self._files[destination_key] = self._files.pop(source_key)
            self._metadata[destination_key] = self._metadata.pop(source_key, {})
            return self._get_file_client(destination_container,
                                         destination_path)

        mock_file.upload_data = upload_data
        mock_file.download_file = download_file
        mock_file.delete_file = delete_file
        mock_file.exists = exists
        mock_file.get_file_properties = get_file_properties
        mock_file.create_file = create_file
        mock_file.start_copy_from_url = start_copy_from_url
        mock_file.rename_file = rename_file

        return mock_file

    def _get_paths(self, file_system: str, path: str = "",
                   recursive: bool = True):
        """List paths in a directory"""
        prefix = f"{file_system}/{path}" if path else file_system
        if not prefix.endswith('/'):
            prefix += '/'

        paths = []
        for key in self._files.keys():
            if key.startswith(prefix) and key != prefix:
                relative_path = key[len(prefix):]

                if not recursive and '/' in relative_path.rstrip('/'):
                    continue

                mock_path = Mock()
                # name should be the full path from container root, not just filename
                full_path_from_root = key[len(f"{file_system}/"):]
                mock_path.name = full_path_from_root.rstrip('/')
                mock_path.is_directory = key.endswith('/')
                paths.append(mock_path)

        return iter(paths)

    def _exists(self, file_system: str, path: str) -> bool:
        """Check if path exists (file or directory)"""
        # Normalize path
        normalized_path = path.rstrip('/')
        key = f"{file_system}/{normalized_path}"
        dir_key = f"{key}/"

        # Exists if it's a file or a directory
        return key in self._files or dir_key in self._files

    def _delete_path(self, file_system: str, path: str):
        """Delete a path"""
        key = f"{file_system}/{path}"
        if key in self._files:
            del self._files[key]
        self._metadata.pop(key, None)
        key_dir = f"{key}/"
        if key_dir in self._files:
            del self._files[key_dir]
        self._metadata.pop(key_dir, None)
        # Delete all files under this path
        to_delete = [k for k in self._files.keys() if k.startswith(key_dir)]
        for k in to_delete:
            del self._files[k]
            self._metadata.pop(k, None)


def withADLSServicePrincipal(tenant_id: str = None, client_id: str = None,
                             client_secret: str = None):
    """
    Decorator to test with specific Service Principal credentials.
    Useful for multi-tenant B2B scenarios.
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Save current credentials
            old_tenant = os.environ.get('AZURE_TENANT_ID')
            old_client_id = os.environ.get('AZURE_CLIENT_ID')
            old_client_secret = os.environ.get('AZURE_CLIENT_SECRET')

            try:
                # Set test credentials
                if tenant_id:
                    os.environ['AZURE_TENANT_ID'] = tenant_id
                if client_id:
                    os.environ['AZURE_CLIENT_ID'] = client_id
                if client_secret:
                    os.environ['AZURE_CLIENT_SECRET'] = client_secret

                return func(*args, **kwargs)
            finally:
                # Restore old credentials
                if old_tenant:
                    os.environ['AZURE_TENANT_ID'] = old_tenant
                elif 'AZURE_TENANT_ID' in os.environ:
                    del os.environ['AZURE_TENANT_ID']

                if old_client_id:
                    os.environ['AZURE_CLIENT_ID'] = old_client_id
                elif 'AZURE_CLIENT_ID' in os.environ:
                    del os.environ['AZURE_CLIENT_ID']

                if old_client_secret:
                    os.environ['AZURE_CLIENT_SECRET'] = old_client_secret
                elif 'AZURE_CLIENT_SECRET' in os.environ:
                    del os.environ['AZURE_CLIENT_SECRET']

        return wrapper

    return decorator
