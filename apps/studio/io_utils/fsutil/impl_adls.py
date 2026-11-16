# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import datetime
import gzip
import io
import json
import logging
import os
import os.path as osp
import posixpath
import re
import shutil
import tempfile
import threading
import time
import uuid
from copy import deepcopy
from typing import (
    BinaryIO,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
)
from urllib.parse import quote
from zipfile import ZipFile

from azure.core import MatchConditions
from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    ResourceExistsError,
    ResourceModifiedError,
    ResourceNotFoundError,
    ServiceRequestError,
)
from azure.identity import (
    AzureAuthorityHosts,
    ClientSecretCredential,
    DefaultAzureCredential,
)
from azure.storage.blob import (
    BlobBlock,
    BlobSasPermissions,
    BlobServiceClient,
    generate_blob_sas,
)
from azure.storage.filedatalake import (
    DataLakeDirectoryClient,
    DataLakeFileClient,
    DataLakeServiceClient,
    DirectorySasPermissions,
    FileSasPermissions,
    FileSystemClient,
    generate_directory_sas,
    generate_file_sas,
)

from io_utils.errors import restricted_diagnostic_boundary
from io_utils.fsutil import api
from io_utils.fsutil.common import (
    DEFAULT_DATA_VERSION,
    fsutil_ctx,
    should_include,
)
from io_utils.fsutil.credentials import (
    credentials_for_provider,
    scoped_cache_key,
)
from io_utils.fsutil.logging_policy import logging_policy
from io_utils.fsutil.uri import ADLSURI, AnyURI, LocalURI, ParseURI
from io_utils.retry import retry
from monitoring.tracing import TraceOptions as _TraceOptions
from monitoring.tracing import trace

logger = logging.getLogger(__name__)


def TraceOptions(**kwargs) -> _TraceOptions:
    """Build spans that never export raw Azure provider exceptions."""
    return _TraceOptions(record_exceptions=False, **kwargs)


BUFFER_SIZE = 1024 * 1024  # 1mb
COPY_STATUS_POLL_INTERVAL_SEC = 0.2
COPY_STATUS_POLL_MAX_ATTEMPTS = 20
DELETE_BATCH_SIZE = 200
_UPLOAD_METADATA_KEY = 'kumo_upload_id'


def retry_policy():
    """Retry policy for ADLS operations"""
    ex_types = (
        AzureError,
        ServiceRequestError,
        TimeoutError,
        ConnectionError,
    )
    return retry(
        exceptions=ex_types,
        tries=3,
        delay=2.0,
        backoff=2.0,
        jitter=(0.0, 1.0),
        # Azure exception strings can contain request metadata or token
        # exchange details. Callers surface a sanitized terminal error.
        logger=None,
    )


adls_logging_policy = logging_policy(logger, restricted_failures=True)


def _create_datalake_client(
    *,
    account_url: Optional[str] = None,
    credential: Optional[object] = None,
    connection_string: Optional[str] = None,
    audience: Optional[str] = None,
) -> DataLakeServiceClient:
    # Configure retry at the transport/connection level, not as a direct parameter
    if connection_string:
        return DataLakeServiceClient.from_connection_string(
            connection_string,
            retry_total=3,  # Use retry_total instead
            retry_backoff_factor=0.8,
        )
    kwargs = {}
    if audience is not None:
        kwargs['audience'] = audience
    return DataLakeServiceClient(
        account_url=account_url,
        credential=credential,
        retry_total=3,
        retry_backoff_factor=0.8,
        **kwargs,
    )


def _azure_storage_settings(endpoint_suffix: str) -> tuple[str, str]:
    try:
        return {
            'dfs.core.windows.net': (
                'https://storage.azure.com',
                AzureAuthorityHosts.AZURE_PUBLIC_CLOUD,
            ),
            'dfs.core.usgovcloudapi.net': (
                'https://storage.azure.us',
                AzureAuthorityHosts.AZURE_GOVERNMENT,
            ),
            'dfs.core.chinacloudapi.cn': (
                'https://storage.azure.cn',
                AzureAuthorityHosts.AZURE_CHINA,
            ),
        }[endpoint_suffix]
    except KeyError:
        raise ValueError(
            f'Unsupported ADLS endpoint suffix: {endpoint_suffix}') from None


def _ambient_azure_credential(authority: str):
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    client_id = os.environ.get("AZURE_CLIENT_ID")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET")
    if tenant_id and client_id and client_secret:
        return ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            authority=authority,
        )
    return DefaultAzureCredential(authority=authority)


class ADLSContext:
    """Adapt parsed ADLS URIs to the scoped, process-safe client cache."""
    def get_fs_client_helper(self, uri: ADLSURI) -> FileSystemClient:
        client = threadsafe_adls.data_lake_client(uri.storage_account,
                                                  uri.endpoint_suffix)
        return client.get_file_system_client(uri.container)

    def get_file_client(self, uri: ADLSURI) -> DataLakeFileClient:
        """Get file client for a specific path"""
        # Use the storage account from the URI, not self.client
        return self.get_fs_client_helper(uri).get_file_client(uri.path)

    def get_directory_client(self, uri: ADLSURI) -> DataLakeDirectoryClient:
        """Get directory client for a specific path"""
        # Use the storage account from the URI, not self.client
        return self.get_fs_client_helper(uri).get_directory_client(uri.path)


adls_ctx = ADLSContext()


class ThreadsafeADLS:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: dict[str, DataLakeServiceClient] = {}
        self._blob_clients: dict[str, BlobServiceClient] = {}
        self._pid = os.getpid()

    def _build_client(
        self,
        storage_account: str,
        endpoint_suffix: str = 'dfs.core.windows.net',
    ) -> DataLakeServiceClient:

        audience, authority = _azure_storage_settings(endpoint_suffix)
        logger.debug(
            f"Creating ADLS client for storage account {storage_account}")
        scope = credentials_for_provider('adls')
        if scope is not None:
            credentials = scope.credentials or {}
            credential = credentials.get('credential')
            if credential is None:
                raise RuntimeError(
                    'Missing federated Azure credential for ADLS identity')
            account_url = f"https://{storage_account}.{endpoint_suffix}"
            return _create_datalake_client(account_url=account_url,
                                           credential=credential,
                                           audience=audience)
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if connection_string:
            return _create_datalake_client(connection_string=connection_string)

        account_url = f"https://{storage_account}.{endpoint_suffix}"
        return _create_datalake_client(
            account_url=account_url,
            credential=_ambient_azure_credential(authority), audience=audience)

    def _build_blob_client(
        self,
        storage_account: str,
        endpoint_suffix: str,
    ) -> BlobServiceClient:
        audience, authority = _azure_storage_settings(endpoint_suffix)
        scope = credentials_for_provider('adls')
        credential = None
        if scope is not None:
            credential = (scope.credentials or {}).get('credential')
            if credential is None:
                raise RuntimeError(
                    'Missing federated Azure credential for ADLS identity')
        else:
            connection_string = os.environ.get(
                "AZURE_STORAGE_CONNECTION_STRING")
            if connection_string:
                return BlobServiceClient.from_connection_string(
                    connection_string)
        if credential is None:
            credential = _ambient_azure_credential(authority)

        blob_suffix = _blob_endpoint_suffix(endpoint_suffix)
        return BlobServiceClient(
            account_url=f"https://{storage_account}.{blob_suffix}",
            credential=credential,
            audience=audience,
        )

    def data_lake_client(
        self,
        storage_account: Optional[str] = None,
        endpoint_suffix: str = 'dfs.core.windows.net',
    ) -> DataLakeServiceClient:
        if os.getpid() != self._pid:
            with self._lock:
                if os.getpid() != self._pid:
                    self._clients = {}
                    self._blob_clients = {}
                    self._pid = os.getpid()

        if not storage_account:
            storage_account = os.environ.get("AZURE_STORAGE_ACCOUNT")

        if not storage_account:
            raise RuntimeError(
                "No Azure Storage account provided (env: AZURE_STORAGE_ACCOUNT)"
            )

        scope = credentials_for_provider('adls')
        if scope is not None:
            key = scoped_cache_key('adls',
                                   f'{storage_account}.{endpoint_suffix}')
            client = scope.cache.get(key)
            if client is None:
                client = self._build_client(storage_account, endpoint_suffix)
                scope.cache[key] = client
            return client

        with self._lock:
            cache_key = f'{storage_account}.{endpoint_suffix}'
            client = self._clients.get(cache_key)
            if client is None:
                client = self._build_client(storage_account, endpoint_suffix)
                self._clients[cache_key] = client

        return client

    def blob_service_client(
        self,
        storage_account: str,
        endpoint_suffix: str = 'dfs.core.windows.net',
    ) -> BlobServiceClient:
        cache_key = f'blob:{storage_account}.{endpoint_suffix}'
        scope = credentials_for_provider('adls')
        if scope is not None:
            key = scoped_cache_key('adls', cache_key)
            client = scope.cache.get(key)
            if client is None:
                client = self._build_blob_client(storage_account,
                                                 endpoint_suffix)
                scope.cache[key] = client
            return client
        with self._lock:
            client = self._blob_clients.get(cache_key)
            if client is None:
                client = self._build_blob_client(storage_account,
                                                 endpoint_suffix)
                self._blob_clients[cache_key] = client
        return client


threadsafe_adls = ThreadsafeADLS()


@api.filesize.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@adls_logging_policy()
def filesize(uri: ADLSURI) -> Optional[int]:
    """Get file size in bytes, returns None for directories/containers"""
    try:
        # Return None for container root
        if not uri.path or uri.path == '/':
            return None

        if uri.path.endswith('/'):
            return None

        file_client = adls_ctx.get_file_client(uri)

        # Check if it exists
        if not file_client.exists():
            return None

        # Get properties
        props = file_client.get_file_properties()

        # Return None if it's a directory
        if props.metadata and props.metadata.get('hdi_isfolder') == 'true':
            return None

        return props.size

    except (AzureError, ClientAuthenticationError) as e:
        # Handle cases where path is invalid or is a directory
        # Return None instead of raising error
        if 'ResourceNotFound' in str(e) or 'InvalidResourceName' in str(e):
            return None
        # For other errors, log and return None
        logger.error("ADLS filesize failed for %s (error_type=%s)", uri,
                     type(e).__name__)
        return None


@api.object_identity.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@adls_logging_policy(allowed_exceptions=FileNotFoundError)
def object_identity(uri: ADLSURI) -> api.ObjectIdentity:
    """Capture ADLS size and ETag for one exact file."""
    if not uri.path or uri.path == '/' or uri.path.endswith('/'):
        raise FileNotFoundError(uri.as_adls())
    file_client = adls_ctx.get_file_client(uri)
    try:
        props = file_client.get_file_properties()
    except ResourceNotFoundError:
        raise FileNotFoundError(uri.as_adls()) from None
    if props.metadata and props.metadata.get('hdi_isfolder') == 'true':
        raise FileNotFoundError(uri.as_adls())
    etag = props.etag if isinstance(props.etag, str) and props.etag else None
    version_id = getattr(props, 'version_id', None)
    version = str(version_id) if version_id else None
    if etag is None and version is None:
        raise RuntimeError('ADLS object metadata has no immutable identity')
    return api.ObjectIdentity(
        size=int(props.size),
        etag=etag,
        version=version,
    )


@api.read_binary.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@adls_logging_policy(allowed_exceptions=FileNotFoundError)
def read_binary(uri: ADLSURI, compression: bool) -> bytes:
    try:
        file_client = adls_ctx.get_file_client(uri)
        download = file_client.download_file()
        data = download.readall()
        _check_size_match(local=data, remote=uri)
        return gzip.decompress(data) if compression else data
    except ResourceNotFoundError:
        raise FileNotFoundError(f"not found: {uri.as_url()}") from None


@api.read_text.register
@trace(TraceOptions(trace_params=True))
@adls_logging_policy(allowed_exceptions=FileNotFoundError)
def read_text(uri: ADLSURI, compression: bool,
              encoding: Optional[str] = None) -> str:
    r = read_binary(uri, compression=compression)
    if encoding:
        return r.decode(encoding)
    return r.decode()


@api._write_binary.register
@retry_policy()
@trace(TraceOptions(trace_params=["uri", "compression"]))
@adls_logging_policy(skip_args=[1])
def write_binary(uri: ADLSURI, data: BinaryIO, compression: bool,
                 **kwargs) -> None:
    body: BinaryIO = (io.BytesIO(gzip.compress(data.read()))
                      if compression else data)
    file_client = adls_ctx.get_file_client(uri)
    body.seek(0)
    content = body.read()
    # Azure SDK assumption: upload_data() with empty bytes silently does
    # nothing — the file is never created on ADLS. Use create_file() for
    # 0-byte writes to ensure the file actually exists after the call.
    if len(content) == 0:
        file_client.create_file()
    else:
        file_client.upload_data(content, overwrite=True)


@api.write_text.register
@trace(TraceOptions(trace_params=["uri", "compression", "encoding"]))
@adls_logging_policy(skip_args=[1])
def write_text(uri: ADLSURI, data: str, compression: bool,
               encoding: Optional[str] = None) -> None:
    api.write_binary(
        uri.as_url(),
        data.encode(encoding) if encoding else data.encode(),
        compression=compression,
    )


@api.write_text_if_absent.register
@trace(TraceOptions(trace_params=['uri']))
@adls_logging_policy(skip_args=[1])
def write_text_if_absent(uri: ADLSURI, data: str) -> bool:
    operation_id = uuid.uuid4().hex
    temp_uri = _temporary_uri(uri, operation_id)
    file_client = adls_ctx.get_file_client(temp_uri)
    try:
        _stage_atomic_file(file_client, data.encode(), operation_id)
        source_properties = file_client.get_file_properties()
        try:
            file_client.rename_file(
                f'{uri.container}/{uri.path}',
                source_etag=source_properties.etag,
                source_match_condition=MatchConditions.IfNotModified,
                match_condition=MatchConditions.IfMissing,
            )
        except (ResourceExistsError, ResourceModifiedError):
            return False
        return True
    finally:
        try:
            file_client.delete_file()
        except ResourceNotFoundError:
            # Successful promotion already removed the temporary file.
            pass


@api.dirsize.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@adls_logging_policy()
def dirsize(uri: ADLSURI) -> int:
    try:
        dir_client = adls_ctx.get_directory_client(uri)
        total_size = 0

        paths = dir_client.get_paths(recursive=True)
        for path in paths:
            if not path.is_directory:
                total_size += path.content_length or 0

        return total_size
    except ResourceNotFoundError:
        raise FileNotFoundError(str(uri))


def _check_size_match(*, local: Union[LocalURI, bytes], remote: ADLSURI):
    if isinstance(local, bytes):
        local_size = len(local)
    else:
        local_size = api.filesize(local)
    remote_size = filesize(remote)
    if remote_size != local_size:
        if isinstance(local, bytes):
            err_msg = f"local bytes is {local_size} bytes, "
        else:
            err_msg = f"local file {local.path} is {local_size} bytes, "
        raise RuntimeError(
            f"ADLS file size mismatch, {err_msg}"
            f"remote file {remote.as_url()} is {remote_size} bytes")


@retry_policy()
def _upload(*, local: LocalURI, remote: ADLSURI):
    """Upload local file to ADLS with streaming"""
    file_client = adls_ctx.get_file_client(remote)

    file_size = os.path.getsize(local.path)

    if file_size == 0:
        file_client.create_file()
        file_client.flush_data(0)
    else:
        with open(local.path, "rb") as data:
            file_client.upload_data(data, overwrite=True, max_concurrency=4)

    _check_size_match(local=local, remote=remote)


@retry_policy()
def _download(*, remote: ADLSURI, local: LocalURI):
    """Download file from ADLS with streaming"""
    try:
        file_client = adls_ctx.get_file_client(remote)
        with open(local.path, "wb") as local_f:
            download = file_client.download_file()
            download.readinto(local_f)

        _check_size_match(remote=remote, local=local)
    except ResourceNotFoundError:
        raise FileNotFoundError(f"not found: {remote.as_url()}") from None


class _ADLSReadStream(io.RawIOBase):
    """Readable wrapper around StorageStreamDownloader.

    Allows the downloader to be used as an io.RawIOBase, enabling
    io.BufferedReader wrapping and gzip decompression without downloading
    the entire file to a local temp file first.
    """
    def __init__(self, downloader) -> None:
        self._downloader = downloader

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        data = self._downloader.read(len(b))
        if not data:
            return 0
        n = len(data)
        b[:n] = data
        return n


class _ADLSWriteStream(io.RawIOBase):
    """Writable wrapper that streams data to ADLS via append_data + flush_data.

    Chunks are sent to ADLS as they arrive. flush_data() is called on close()
    to commit the file. If an error occurs during writing, flush_data() is
    skipped to avoid committing partial data.
    """
    def __init__(self, file_client: DataLakeFileClient) -> None:
        self._client = file_client
        self._offset = 0
        self._error = False
        self._client.create_file()

    def writable(self) -> bool:
        return True

    def write(self, b) -> int:
        data = bytes(b)
        if not data:
            return 0
        try:
            self._client.append_data(data, offset=self._offset,
                                     length=len(data))
            self._offset += len(data)
        except Exception:
            self._error = True
            raise
        return len(data)

    def close(self) -> None:
        if not self.closed:
            try:
                if not self._error:
                    self._client.flush_data(self._offset)
            finally:
                super().close()


@api._get_binary_handle.register
@trace(TraceOptions(trace_params=True))
def get_binary_handle(
    uri: ADLSURI,
    mode: str,
    compression: bool = False,
) -> Generator[BinaryIO, None, None]:
    with restricted_diagnostic_boundary(
            safe_message='ADLS stream operation failed',
            exception_types=(AzureError, OSError)):
        if "b" not in mode:
            mode += "b"

        if "r" in mode:
            downloader = adls_ctx.get_file_client(uri).download_file()
            stream: BinaryIO = io.BufferedReader(_ADLSReadStream(downloader))
        else:
            stream = _ADLSWriteStream(adls_ctx.get_file_client(uri))

        try:
            if compression:
                with gzip.GzipFile(fileobj=stream, mode=mode) as gzf:
                    yield gzf  # type: ignore
            else:
                yield stream
        finally:
            stream.close()


@api._get_handle.register
@trace(TraceOptions(trace_params=True))
def get_handle(
    uri: ADLSURI,
    mode: str,
    compression: bool = False,
) -> Generator[TextIO, None, None]:
    with api.get_binary_handle(uri.as_url(), mode + "b",
                               compression=compression) as handle:
        yield io.TextIOWrapper(
            buffer=handle,
            encoding="utf-8",
            write_through=True,
        )


@api.exists.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@adls_logging_policy()
def exists(uri: ADLSURI) -> bool:
    """Check if ADLS path exists (file or directory)"""
    try:
        client = threadsafe_adls.data_lake_client(uri.storage_account,
                                                  uri.endpoint_suffix)
        fs_client = client.get_file_system_client(uri.container)

        # Container root
        if not uri.path or uri.path in ('', '/'):
            return fs_client.exists()

        path_clean = uri.path.rstrip('/')

        # Try directory first (HEAD request, very fast)
        dir_client = fs_client.get_directory_client(path_clean)
        if dir_client.exists():
            return True

        # Try as file (HEAD request, very fast)
        file_client = fs_client.get_file_client(path_clean)
        if file_client.exists():
            return True

        return False

    except (AzureError, ClientAuthenticationError):
        return False
    except Exception:
        raise


@api.isdir.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@adls_logging_policy()
def isdir(uri: ADLSURI) -> bool:
    """Check if ADLS path is a directory"""
    try:
        client = threadsafe_adls.data_lake_client(uri.storage_account,
                                                  uri.endpoint_suffix)
        fs_client = client.get_file_system_client(uri.container)

        # Container root
        if not uri.path or uri.path in ('', '/'):
            return fs_client.exists()

        path_clean = uri.path.rstrip('/')

        # Check if it exists as a directory (HEAD request)
        dir_client = fs_client.get_directory_client(path_clean)
        if dir_client.exists():
            # Verify it's not actually a file by checking if file exists
            file_client = fs_client.get_file_client(path_clean)
            if file_client.exists():
                # Both exist - check metadata for folder marker
                props = file_client.get_file_properties()
                metadata = props.metadata or {}
                return metadata.get('hdi_isfolder') == 'true'
            # Only directory exists
            return True

        return False

    except (AzureError, ClientAuthenticationError):
        return False
    except Exception:
        raise


@api._get_if_present.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@adls_logging_policy()
def get_if_present(uri: ADLSURI) -> Optional[bytes]:
    try:
        file_client = adls_ctx.get_file_client(uri)
        download = file_client.download_file()
        return download.readall()
    except ResourceNotFoundError:
        return None


@adls_logging_policy(allowed_exceptions=FileNotFoundError)
def copy_file(src: str, dst: str):
    src_uri = ParseURI(src)
    dst_uri = ParseURI(dst)

    # Optimize for local <-> ADLS transfers
    if isinstance(src_uri, LocalURI) and isinstance(dst_uri, ADLSURI):
        return _upload(local=src_uri, remote=dst_uri)
    if isinstance(src_uri, ADLSURI) and isinstance(dst_uri, LocalURI):
        return _download(remote=src_uri, local=dst_uri)
    if isinstance(src_uri, ADLSURI) and isinstance(dst_uri, ADLSURI):
        if _server_side_copy(src_uri, dst_uri):
            return None

    with api.get_binary_handle(src, "rb") as fsrc:
        with api.get_binary_handle(dst, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
    assert api.filesize(src) == api.filesize(dst)


def _server_side_copy(src_uri: ADLSURI, dst_uri: ADLSURI) -> bool:
    client = threadsafe_adls.data_lake_client(src_uri.storage_account,
                                              src_uri.endpoint_suffix)
    src_fs = client.get_file_system_client(src_uri.container)
    src_file = src_fs.get_file_client(src_uri.path)

    dst_client = threadsafe_adls.data_lake_client(dst_uri.storage_account,
                                                  dst_uri.endpoint_suffix)
    dst_fs = dst_client.get_file_system_client(dst_uri.container)
    dst_file = dst_fs.get_file_client(dst_uri.path)

    if not hasattr(dst_file, "start_copy_from_url"):
        return False

    try:
        dst_file.start_copy_from_url(src_file.url)
    except Exception as e:
        logger.debug("ADLS server-side copy failed to start (error_type=%s)",
                     type(e).__name__)
        return False

    try:
        for _ in range(COPY_STATUS_POLL_MAX_ATTEMPTS):
            props = dst_file.get_file_properties()
            copy = getattr(props, "copy", None)
            status = getattr(copy, "status", None)
            if status is None:
                return True
            status = str(status).lower()
            if status == "success":
                return True
            if status in ("failed", "aborted"):
                raise RuntimeError(
                    f"ADLS server-side copy failed: {src_uri.as_adls()} -> "
                    f"{dst_uri.as_adls()}")
            time.sleep(COPY_STATUS_POLL_INTERVAL_SEC)
    except Exception as e:
        logger.debug(
            "ADLS server-side copy status check failed "
            "(error_type=%s)",
            type(e).__name__)
        return False

    return True


def copydir(src: str, dst: str) -> None:
    """Server-side ADLS-to-ADLS directory copy using a thread pool.

    Copies all files under src into dst, preserving relative paths within the
    source directory. Each individual file is copied server-side via
    start_copy_from_url (no local I/O), with a download+upload fallback if the
    server-side copy is unavailable.

    Called by generic.py when both paths are ADLS, overwrite=True, and
    raise_on_exists=False.
    """
    src_uri = ParseURI(src)
    dst_uri = ParseURI(dst)
    assert isinstance(src_uri,
                      ADLSURI), f"Expected ADLSURI, got {type(src_uri)}"
    assert isinstance(dst_uri,
                      ADLSURI), f"Expected ADLSURI, got {type(dst_uri)}"

    files, _ = _list_paths_recursive(src_uri)

    if not files:
        logger.warning("copydir: source is empty or does not exist: %s", src)
        return

    logger.debug("copydir: copying %d files from %s to %s", len(files), src,
                 dst)

    src_base = src_uri.path.rstrip('/')
    dst_base = dst_uri.path.rstrip('/')

    def _copy_one(src_path: str) -> None:
        rel = src_path[len(src_base):].lstrip(
            '/') if src_base else src_path.lstrip('/')
        dst_path = f"{dst_base}/{rel}" if dst_base else rel

        file_src = ADLSURI(
            storage_account=src_uri.storage_account,
            container=src_uri.container,
            path=src_path,
        )
        file_dst = ADLSURI(
            storage_account=dst_uri.storage_account,
            container=dst_uri.container,
            path=dst_path,
        )
        if not _server_side_copy(file_src, file_dst):
            copy_file(file_src.as_url(), file_dst.as_url())

    tasks = [fsutil_ctx.thread_pool.submit(_copy_one, f) for f in files]
    for task in tasks:
        task.result()

    logger.debug("copydir: done copying %s to %s", src, dst)


@api.touch.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@adls_logging_policy()
def touch(uri: ADLSURI) -> None:

    try:
        file_client = adls_ctx.get_file_client(uri)

        # Create empty file if it doesn't exist
        exists_result = file_client.exists()

        if not exists_result:
            file_client.create_file()
    except ResourceExistsError:
        pass


@api.has_write_permission.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@adls_logging_policy()
def has_write_permission(uri: ADLSURI) -> bool:
    """Check if current identity has write permission
    Note: ADLS permissions are complex (ACLs + RBAC).
    This attempts a write test to verify permissions.
    """
    base = uri.path.rstrip('/')
    parent = base if uri.path.endswith('/') else posixpath.dirname(base)
    test_path = posixpath.join(parent, f'.kumo_write_probe_{uuid.uuid4().hex}')
    test_file_client = None
    created = False
    cleanup_ok = True
    try:
        test_uri = ADLSURI(storage_account=uri.storage_account,
                           endpoint_suffix=uri.endpoint_suffix,
                           container=uri.container, path=test_path,
                           scheme=uri.scheme)

        test_file_client = adls_ctx.get_file_client(test_uri)
        test_file_client.create_file()
        created = True
    except Exception as error:
        logger.warning('ADLS output write probe failed (%s)',
                       type(error).__name__)
    finally:
        if created:
            assert test_file_client is not None
            try:
                test_file_client.delete_file()
            except ResourceNotFoundError:
                pass
            except Exception as error:
                cleanup_ok = False
                logger.warning(
                    'ADLS output write probe cleanup failed (%s)',
                    type(error).__name__,
                )
    return created and cleanup_ok


@api._listdir.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@adls_logging_policy()
def listdir(
    uri: ADLSURI,
    max_items: Optional[int] = None,
) -> list[str]:

    if max_items is not None:
        if max_items < 1:
            raise ValueError("max_items must be at least 1")
        if max_items > 1000:
            raise ValueError("max_items must be at most 1000")

    try:
        if not uri.path or uri.path in ('', '/'):
            client = threadsafe_adls.data_lake_client(uri.storage_account,
                                                      uri.endpoint_suffix)
            fs_client = client.get_file_system_client(uri.container)
            paths = fs_client.get_paths(path='', recursive=False,
                                        max_results=max_items)
        else:
            dir_client = adls_ctx.get_directory_client(uri)
            paths = dir_client.get_paths(recursive=False, max_results=max_items)

        entries = []
        for path in paths:
            # Return full ADLS URI
            full_uri = deepcopy(uri)
            full_uri.path = path.name
            entries.append(full_uri.as_url())
            if max_items is not None and len(entries) >= max_items:
                break

        return entries
    except ResourceNotFoundError:
        logger.error(f"URI resource not found: {uri.path}")
        return []


def _list_paths_recursive(uri: ADLSURI) -> tuple[list[str], list[str]]:
    client = threadsafe_adls.data_lake_client(uri.storage_account,
                                              uri.endpoint_suffix)
    fs_client = client.get_file_system_client(uri.container)

    if not uri.path or uri.path in ('', '/'):
        base_path = ''
    else:
        base_path = uri.path.rstrip('/')

    if base_path and not isdir(uri) and exists(uri):
        return [base_path], []

    try:
        paths = fs_client.get_paths(path=base_path, recursive=True)
    except ResourceNotFoundError:
        return [], []
    files: list[str] = []
    dirs: list[str] = []
    for path in paths:
        if getattr(path, "is_directory", False):
            dirs.append(path.name.rstrip('/'))
        else:
            files.append(path.name)

    return files, dirs


@api._listdir_detail.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@adls_logging_policy()
def listdir_detail(
    uri: ADLSURI,
    max_items: Optional[int] = None,
) -> list[api.PathDetail]:
    try:
        if not uri.path or uri.path in ('', '/'):
            client = threadsafe_adls.data_lake_client(uri.storage_account,
                                                      uri.endpoint_suffix)
            fs_client = client.get_file_system_client(uri.container)
            paths = fs_client.get_paths(path='', recursive=False,
                                        max_results=max_items)
        else:
            dir_client = adls_ctx.get_directory_client(uri)
            paths = dir_client.get_paths(recursive=False, max_results=max_items)

        entries = []
        for path in paths:
            full_uri = deepcopy(uri)
            full_uri.path = path.name
            entries.append(
                api.PathDetail(path=full_uri.as_url(),
                               is_directory=path.is_directory))

        return entries
    except ResourceNotFoundError:
        return []


@api.makedirs.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@adls_logging_policy()
def makedirs(uri: ADLSURI, exist_ok: bool = False) -> None:
    if not uri.path or uri.path in ('', '/'):
        return

    # Azure SDK assumption: create_directory() is idempotent — it succeeds
    # silently even when the directory already exists, never raising
    # ResourceExistsError. We must check existence ourselves to honour
    # exist_ok=False.
    if not exist_ok and isdir(uri):
        raise FileExistsError(str(uri))

    dir_path = uri.path.rstrip('/')
    parts = [p for p in dir_path.split('/') if p]
    current = ""
    for part in parts:
        current = f"{current}/{part}" if current else part
        dir_client = adls_ctx.get_directory_client(
            ADLSURI(
                storage_account=uri.storage_account,
                container=uri.container,
                path=current,
            ))
        try:
            dir_client.create_directory()
        except ResourceExistsError:
            if not exist_ok:
                # Safety net in case a future SDK version does raise here.
                raise FileExistsError(str(uri)) from None


@api.rm.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@adls_logging_policy()
def rm(
    uri: ADLSURI,
    ignore_errors: bool = False,
    include: Optional[List[str]] = None,
) -> None:
    try:
        if isdir(uri):
            _rmdir(uri, include=include)
        elif exists(uri) and should_include(uri.as_adls(), include):
            file_client = adls_ctx.get_file_client(uri)
            file_client.delete_file()
    except (ResourceNotFoundError, AzureError) as e:
        if not ignore_errors:
            raise
        logger.debug("Ignored ADLS rm error (error_type=%s)", type(e).__name__)


def _rmdir(uri: ADLSURI, include: Optional[List[str]] = None) -> None:
    files, dirs = _list_paths_recursive(uri)

    def _delete_file(path: str) -> None:
        file_uri = ADLSURI(
            storage_account=uri.storage_account,
            container=uri.container,
            path=path,
        )
        if should_include(file_uri.as_adls(), include):
            file_client = adls_ctx.get_file_client(file_uri)
            file_client.delete_file()

    tasks = []
    for path in files:
        tasks.append(fsutil_ctx.thread_pool.submit(_delete_file, path))
        if len(tasks) >= DELETE_BATCH_SIZE:
            for task in tasks:
                task.result()
            tasks = []
    for task in tasks:
        task.result()

    # Delete directories from deepest to shallowest
    dirs = sorted(set(dirs), key=lambda p: p.count('/'), reverse=True)
    if uri.path and uri.path not in ('', '/'):
        dirs.append(uri.path.rstrip('/'))
    for dir_path in dirs:
        dir_uri = ADLSURI(
            storage_account=uri.storage_account,
            container=uri.container,
            path=dir_path,
        )
        try:
            dir_client = adls_ctx.get_directory_client(dir_uri)
            # After file deletion above, check if any children (files or subdirs)
            # remain before deleting this directory. If the include filter left
            # anything behind, leave the directory intact.
            if include is not None:
                remaining = next(iter(dir_client.get_paths(recursive=False)),
                                 None)
                if remaining is not None:
                    continue  # directory still has children — leave it alone
            dir_client.delete_directory()
        except ResourceNotFoundError:
            continue


@api._contains_subdir.register
@trace(TraceOptions(trace_params=True))
@adls_logging_policy()
def contains_subdir(uri: ADLSURI) -> bool:
    """Check if a path contains subdirectories."""
    if not isdir(uri):
        return False

    # Azure SDK assumption: get_directory_client('') raises ValueError for
    # empty paths. Use fs_client.get_paths() directly for container root,
    # same as listdir/listdir_detail.
    if not uri.path or uri.path in ('', '/'):
        client = threadsafe_adls.data_lake_client(uri.storage_account,
                                                  uri.endpoint_suffix)
        fs_client = client.get_file_system_client(uri.container)
        paths = fs_client.get_paths(path='', recursive=False)
    else:
        dir_client = adls_ctx.get_directory_client(uri)
        paths = dir_client.get_paths(recursive=False)

    for path in paths:
        if path.is_directory:
            return True

    return False


@api.get_last_modified_time.register
@trace(TraceOptions(trace_params=True))
@retry_policy()
@adls_logging_policy()
def get_last_modified_time(uri: ADLSURI, suffix: Optional[str] = None) -> float:
    try:
        # For directories, check for _SUCCESS file
        if isdir(uri):
            file_uri = deepcopy(uri)
            file_uri.path = osp.join(uri.path, "_SUCCESS")
            if exists(file_uri):
                uri = file_uri
            else:
                return DEFAULT_DATA_VERSION

        file_client = adls_ctx.get_file_client(uri)
        properties = file_client.get_file_properties()

        if properties.last_modified is None:
            return DEFAULT_DATA_VERSION

        return properties.last_modified.timestamp()
    except ResourceNotFoundError:
        return DEFAULT_DATA_VERSION


_SAS_VALIDITY = datetime.timedelta(hours=2)
_SAS_MAX_VALIDITY = datetime.timedelta(days=7)
_SAS_CLOCK_SKEW = datetime.timedelta(minutes=5)


def _sas_validity_window(
    validity: datetime.timedelta | None = None,
) -> tuple[datetime.datetime, datetime.datetime]:
    validity = validity or _SAS_VALIDITY
    if validity <= datetime.timedelta(0) or validity > _SAS_MAX_VALIDITY:
        raise ValueError('ADLS SAS validity must be greater than zero and no '
                         'more than seven days')
    now = datetime.datetime.now(datetime.timezone.utc)
    return now - _SAS_CLOCK_SKEW, now + validity


@api.generate_directory_read_sas.register
def generate_directory_read_sas(
    uri: ADLSURI,
    *,
    validity: datetime.timedelta | None = None,
) -> str:
    """Create a short-lived user-delegation SAS scoped to one directory."""
    start, expiry = _sas_validity_window(validity)
    service_client = threadsafe_adls.data_lake_client(uri.storage_account,
                                                      uri.endpoint_suffix)
    try:
        delegation_key = service_client.get_user_delegation_key(
            key_start_time=start,
            key_expiry_time=expiry,
        )
    except Exception:
        raise ValueError(
            f"Failed to obtain user delegation key for storage account "
            f"'{uri.storage_account}'. Ensure the identity holds the "
            f"'Storage Blob Delegator' role.") from None
    return generate_directory_sas(
        account_name=uri.storage_account,
        file_system_name=uri.container,
        directory_name=uri.path,
        credential=delegation_key,
        permission=DirectorySasPermissions(
            read=True,
            list=True,
            execute=True,
        ),
        start=start,
        expiry=expiry,
    )


def _blob_endpoint_suffix(dfs_suffix: str) -> str:
    if not dfs_suffix.startswith('dfs.'):
        raise ValueError(f'Invalid ADLS endpoint suffix: {dfs_suffix}')
    return 'blob.' + dfs_suffix[len('dfs.'):]


def _upload_state(uri: ADLSURI, num_parts: int) -> tuple[str, str]:
    if num_parts < 1:
        raise ValueError('num_parts must be greater than zero')
    nonce = uuid.uuid4().hex
    temp_path = f'{uri.path}.kumo-upload-{nonce}-{num_parts}'
    state = {
        'v': 1,
        'uri': uri.as_url(),
        'temp_path': temp_path,
        'nonce': nonce,
        'num_parts': num_parts,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(state, separators=(',', ':'),
                   sort_keys=True).encode()).decode().rstrip('=')
    return encoded, temp_path


def _decode_upload_state(upload_id: str, uri: ADLSURI) -> dict:
    try:
        padding = '=' * (-len(upload_id) % 4)
        state = json.loads(
            base64.urlsafe_b64decode(upload_id + padding).decode())
    except Exception:
        raise ValueError('Invalid ADLS upload id') from None
    if (state.get('v') != 1 or state.get('uri') != uri.as_url()
            or not isinstance(state.get('num_parts'), int)
            or state['num_parts'] < 1):
        raise ValueError('ADLS upload id does not match destination')
    nonce = state.get('nonce')
    if not isinstance(nonce, str) or re.fullmatch(r'[0-9a-f]{32}',
                                                  nonce) is None:
        raise ValueError('Invalid ADLS upload id')
    expected_temp = (f"{uri.path}.kumo-upload-{state.get('nonce')}-"
                     f"{state.get('num_parts')}")
    if state.get('temp_path') != expected_temp:
        raise ValueError('Invalid ADLS upload id')
    return state


def _raw_block_id(nonce: str, part_number: int) -> str:
    return f'kumo:{nonce}:{part_number:08d}'


def _encoded_block_id(nonce: str, part_number: int) -> str:
    return base64.b64encode(_raw_block_id(nonce, part_number).encode()).decode()


def _normalize_block_ids(blocks, expected: set[str]) -> Optional[set[str]]:
    normalized = []
    for block in blocks:
        block_id = getattr(block, 'id', None)
        if block_id is None:
            encoded = getattr(block, 'name', None)
            if not isinstance(encoded, str):
                return None
            try:
                block_id = base64.b64decode(encoded, validate=True).decode()
            except (ValueError, UnicodeDecodeError):
                return None
            if base64.b64encode(block_id.encode()).decode() != encoded:
                return None
        if not isinstance(block_id, str) or block_id not in expected:
            return None
        normalized.append(block_id)
    if len(normalized) != len(set(normalized)):
        return None
    return set(normalized)


def _atomic_promote(
    source: ADLSURI,
    destination: ADLSURI,
    operation_id: str,
) -> None:
    """Conditionally rename a complete HNS file into place.

    The destination precondition prevents silently replacing a value changed
    after this operation inspected it. The source ETag proves the promoted
    bytes are exactly the completed temporary file. A retry after a successful
    rename is accepted only when the destination retains this operation id.
    """
    if (source.storage_account != destination.storage_account
            or source.endpoint_suffix != destination.endpoint_suffix
            or source.container != destination.container):
        raise ValueError('ADLS atomic promotion must remain in one filesystem')

    source_client = adls_ctx.get_file_client(source)
    destination_client = adls_ctx.get_file_client(destination)
    try:
        source_properties = source_client.get_file_properties()
    except ResourceNotFoundError:
        try:
            metadata = destination_client.get_file_properties().metadata or {}
        except ResourceNotFoundError:
            metadata = {}
        if metadata.get(_UPLOAD_METADATA_KEY) == operation_id:
            return
        raise RuntimeError('ADLS atomic promotion source is missing') from None

    kwargs = {
        'source_etag': source_properties.etag,
        'source_match_condition': MatchConditions.IfNotModified,
    }
    try:
        destination_properties = destination_client.get_file_properties()
        kwargs.update({
            'etag': destination_properties.etag,
            'match_condition': MatchConditions.IfNotModified,
        })
    except ResourceNotFoundError:
        kwargs.update({
            'match_condition': MatchConditions.IfMissing,
        })

    try:
        source_client.rename_file(f'{destination.container}/{destination.path}',
                                  **kwargs)
    except (ResourceExistsError, ResourceModifiedError):
        try:
            metadata = destination_client.get_file_properties().metadata or {}
        except ResourceNotFoundError:
            metadata = {}
        if metadata.get(_UPLOAD_METADATA_KEY) != operation_id:
            raise RuntimeError(
                'ADLS atomic promotion destination changed') from None


def _temporary_uri(uri: ADLSURI, operation_id: str) -> ADLSURI:
    return ADLSURI(
        storage_account=uri.storage_account,
        endpoint_suffix=uri.endpoint_suffix,
        container=uri.container,
        path=f'{uri.path}.kumo-atomic-{operation_id}',
        scheme=uri.scheme,
    )


def _stage_atomic_file(
    file_client: DataLakeFileClient,
    payload: bytes,
    operation_id: str,
) -> None:
    metadata = {_UPLOAD_METADATA_KEY: operation_id}
    if payload:
        file_client.upload_data(
            payload,
            overwrite=True,
            metadata=metadata,
            match_condition=MatchConditions.IfMissing,
        )
    else:
        file_client.create_file(
            metadata=metadata,
            match_condition=MatchConditions.IfMissing,
        )


@api.atomic_write_text.register
@adls_logging_policy(skip_args=[1])
def atomic_write_text(uri: ADLSURI, data: str,
                      compression: bool = False) -> None:
    operation_id = uuid.uuid4().hex
    temp_uri = _temporary_uri(uri, operation_id)
    file_client = adls_ctx.get_file_client(temp_uri)
    payload = data.encode()
    if compression:
        payload = gzip.compress(payload)
    try:
        _stage_atomic_file(file_client, payload, operation_id)
        _atomic_promote(temp_uri, uri, operation_id)
    finally:
        try:
            file_client.delete_file()
        except ResourceNotFoundError:
            # Successful promotion may already remove the temporary file.
            pass


class ADLSPresignedHelper(api.PresignedHelper[ADLSURI]):
    """ADLS implementation of :class:`~api.PresignedHelper`.

    Generates SAS download URLs using a user delegation key — the only
    signing mechanism that works without a storage account key.  Requires
    the Azure identity (SP or managed identity) to hold the
    ``Storage Blob Delegator`` role on the storage account.
    """
    def __init__(self, uri: ADLSURI):
        super().__init__(uri)

    @property
    def supports_multipart_upload(self) -> bool:
        return True

    @staticmethod
    def _delete_staging_blob(blob_client) -> None:
        try:
            blob_client.delete_blob(delete_snapshots='include')
        except ResourceNotFoundError:
            # Cleanup is idempotent when the staging blob is already absent.
            pass

    def _generate_download_url(self, uri: ADLSURI) -> str:
        start, expiry = _sas_validity_window()

        service_client = threadsafe_adls.data_lake_client(
            uri.storage_account, uri.endpoint_suffix)
        try:
            user_delegation_key = service_client.get_user_delegation_key(
                key_start_time=start,
                key_expiry_time=expiry,
            )
        except Exception:
            raise ValueError(
                f"Failed to obtain user delegation key for storage account "
                f"'{uri.storage_account}'. Ensure the identity holds the "
                f"'Storage Blob Delegator' role.") from None

        dir_name = posixpath.dirname(uri.path)
        file_name = posixpath.basename(uri.path)
        sas_token = generate_file_sas(
            account_name=uri.storage_account,
            file_system_name=uri.container,
            directory_name=dir_name,
            file_name=file_name,
            credential=user_delegation_key,
            permission=FileSasPermissions(read=True),
            start=start,
            expiry=expiry,
        )
        return (f"https://{uri.storage_account}.{uri.endpoint_suffix}"
                f"/{quote(uri.container, safe='')}/"
                f"{quote(uri.path, safe='/')}?{sas_token}")

    def _create_multipart_upload(
        self,
        uri: ADLSURI,
        num_parts: int,
    ) -> Tuple[str, Dict[int, str]]:
        upload_id, temp_path = _upload_state(uri, num_parts)
        state = _decode_upload_state(upload_id, uri)
        start, expiry = _sas_validity_window()
        service_client = threadsafe_adls.data_lake_client(
            uri.storage_account, uri.endpoint_suffix)
        try:
            delegation_key = service_client.get_user_delegation_key(
                key_start_time=start,
                key_expiry_time=expiry,
            )
        except Exception:
            raise ValueError(
                f"Failed to obtain user delegation key for storage account "
                f"'{uri.storage_account}'. Ensure the identity holds the "
                f"'Storage Blob Delegator' role.") from None

        sas_token = generate_blob_sas(
            account_name=uri.storage_account,
            container_name=uri.container,
            blob_name=temp_path,
            user_delegation_key=delegation_key,
            permission=BlobSasPermissions(create=True, write=True),
            start=start,
            expiry=expiry,
        )
        marker = threadsafe_adls.blob_service_client(
            uri.storage_account,
            uri.endpoint_suffix,
        ).get_blob_client(uri.container, temp_path)
        marker.upload_blob(
            b'',
            overwrite=False,
            metadata={_UPLOAD_METADATA_KEY: upload_id},
        )
        blob_suffix = _blob_endpoint_suffix(uri.endpoint_suffix)
        block_urls = {}
        for part_number in range(1, num_parts + 1):
            block_id = _encoded_block_id(state['nonce'], part_number)
            block_urls[part_number] = (
                f'https://{uri.storage_account}.{blob_suffix}/'
                f'{quote(uri.container, safe="")}/'
                f'{quote(temp_path, safe="/")}?{sas_token}&comp=block'
                f'&blockid={quote(block_id, safe="")}')
        return upload_id, block_urls

    @retry_policy()
    def _complete_multipart_upload(
        self,
        uri: ADLSURI,
        upload_id: str,
        parts: List[api.PartMetadata],
    ) -> None:
        state = _decode_upload_state(upload_id, uri)
        numbers = [part.part_number for part in parts]
        expected = list(range(1, state['num_parts'] + 1))
        temp_uri = ADLSURI(
            storage_account=uri.storage_account,
            endpoint_suffix=uri.endpoint_suffix,
            container=uri.container,
            path=state['temp_path'],
            scheme=uri.scheme,
        )
        blob_client = threadsafe_adls.blob_service_client(
            uri.storage_account,
            uri.endpoint_suffix,
        ).get_blob_client(uri.container, state['temp_path'])
        final_client = adls_ctx.get_file_client(uri)
        if (sorted(numbers) != expected or len(numbers) != len(set(numbers))):
            self._delete_staging_blob(blob_client)
            raise ValueError(
                'ADLS multipart metadata must contain every part exactly once')
        try:
            metadata = final_client.get_file_properties().metadata or {}
        except ResourceNotFoundError:
            metadata = {}
        if metadata.get(_UPLOAD_METADATA_KEY) == upload_id:
            return

        staging_metadata = blob_client.get_blob_properties().metadata or {}
        if staging_metadata.get(_UPLOAD_METADATA_KEY) != upload_id:
            self._delete_staging_blob(blob_client)
            raise ValueError('ADLS multipart upload state failed validation')
        blocks = blob_client.get_block_list(block_list_type='all')
        if isinstance(blocks, tuple) and len(blocks) == 2:
            committed, uncommitted = blocks
        else:
            committed = getattr(blocks, 'committed_blocks', None)
            uncommitted = getattr(blocks, 'uncommitted_blocks', None)
        block_ids = [
            _raw_block_id(state['nonce'], number) for number in expected
        ]
        expected_ids = set(block_ids)
        available = (_normalize_block_ids(uncommitted, expected_ids)
                     if uncommitted is not None else None)
        committed_ids = (_normalize_block_ids(committed, expected_ids)
                         if committed is not None else None)
        if available == expected_ids:
            blob_client.commit_block_list(
                [BlobBlock(block_id=block_id) for block_id in block_ids],
                metadata={_UPLOAD_METADATA_KEY: upload_id},
            )
        elif committed_ids != expected_ids:
            self._delete_staging_blob(blob_client)
            raise ValueError(
                'Uploaded ADLS blocks do not match multipart metadata')
        _atomic_promote(temp_uri, uri, upload_id)


@api.get_presigned_helper.register
def get_presigned_helper_adls(uri: ADLSURI) -> ADLSPresignedHelper:
    return ADLSPresignedHelper(uri)


@api._zip_directory.register
def zip_directory(
    uri: ADLSURI,
    recursive: bool = False,
    output_path: Optional[str] = None,
) -> str:
    if output_path is None:
        output_uri = uri.model_copy(deep=True)
        output_uri.path = uri.path.rstrip('/') + '.zip'
    else:
        output_uri = ParseURI(output_path)
        if not isinstance(output_uri, ADLSURI):
            raise ValueError('ADLS ZIP output must be an ADLS path')
        if (output_uri.storage_account != uri.storage_account
                or output_uri.endpoint_suffix != uri.endpoint_suffix
                or output_uri.container != uri.container):
            raise ValueError(
                'ADLS ZIP output must remain in the source filesystem')
    if exists(output_uri):
        return output_uri.as_url()

    files, _ = _list_paths_recursive(uri)
    if not recursive:
        prefix = uri.path.rstrip('/') + '/' if uri.path else ''
        files = [
            path for path in files if '/' not in path[len(prefix):].rstrip('/')
        ]
    if not files:
        raise ValueError(f'No files found under {uri.as_url()}')

    operation_id = uuid.uuid4().hex
    temp_uri = _temporary_uri(output_uri, operation_id)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, 'output.zip')
            with ZipFile(zip_path, 'w') as zip_file:
                for path in files:
                    file_uri = uri.model_copy(deep=True)
                    file_uri.path = path
                    data = adls_ctx.get_file_client(
                        file_uri).download_file().readall()
                    zip_file.writestr(path, data)
            with open(zip_path, 'rb') as archive:
                adls_ctx.get_file_client(temp_uri).upload_data(
                    archive,
                    overwrite=True,
                    metadata={_UPLOAD_METADATA_KEY: operation_id},
                    match_condition=MatchConditions.IfMissing,
                )
        _atomic_promote(temp_uri, output_uri, operation_id)
    finally:
        try:
            adls_ctx.get_file_client(temp_uri).delete_file()
        except ResourceNotFoundError:
            # Promotion may already remove the temporary archive.
            pass
    return output_uri.as_url()


@api._zip_files.register
def zip_files(
    uri: ADLSURI,
    paths: Sequence[AnyURI],
    output_path: str,
) -> str:
    output_uri = ParseURI(output_path)
    if not isinstance(output_uri, ADLSURI):
        raise ValueError('ADLS ZIP output must be an ADLS path')
    source_identity = (uri.storage_account, uri.endpoint_suffix, uri.container)
    selected = []
    for path in paths:
        if (not isinstance(path, ADLSURI)
                or (path.storage_account, path.endpoint_suffix,
                    path.container) != source_identity):
            raise ValueError('ADLS archive inputs must use one filesystem')
        selected.append(path.path)
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate ADLS files were selected for archive')
    if exists(output_uri):
        return output_uri.as_url()

    operation_id = uuid.uuid4().hex
    temp_uri = _temporary_uri(output_uri, operation_id)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, 'output.zip')
            with ZipFile(zip_path, 'w') as zip_file:
                for path in selected:
                    file_uri = uri.model_copy(deep=True)
                    file_uri.path = path
                    data = adls_ctx.get_file_client(
                        file_uri).download_file().readall()
                    zip_file.writestr(path, data)
            with open(zip_path, 'rb') as archive:
                adls_ctx.get_file_client(temp_uri).upload_data(
                    archive,
                    overwrite=True,
                    metadata={_UPLOAD_METADATA_KEY: operation_id},
                    match_condition=MatchConditions.IfMissing,
                )
        _atomic_promote(temp_uri, output_uri, operation_id)
    finally:
        try:
            adls_ctx.get_file_client(temp_uri).delete_file()
        except ResourceNotFoundError:
            # Promotion may already have removed the temporary object.
            pass
    return output_uri.as_url()


@api.customer_facing_path.register
def customer_facing_path(uri: ADLSURI) -> str:
    return uri.as_url()
