# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import io
import logging
import os
import os.path as osp
import shutil
import threading
from copy import deepcopy
from datetime import datetime
from typing import BinaryIO, Generator, List, Optional, TextIO, Union

from botocore.exceptions import ClientError
from databricks.sdk.errors.base import DatabricksError
from requests.exceptions import ConnectionError
from urllib3.exceptions import MaxRetryError

from io_utils.databricks import DatabricksConnectionInfo
from io_utils.errors import restricted_diagnostic_boundary
from io_utils.fsutil import api
from io_utils.fsutil.common import DEFAULT_DATA_VERSION, TemporaryDirectory
from io_utils.fsutil.logging_policy import logging_policy
from io_utils.fsutil.uri import UCURI, LocalURI, ParseURI
from io_utils.retry import retry
from monitoring.tracing import TraceOptions as _TraceOptions
from monitoring.tracing import trace
from service_framework.deploy_env import (
    DATABRICKS_CLIENT_ID,
    DATABRICKS_CLIENT_SECRET,
    DATABRICKS_HOST,
    DATABRICKS_TOKEN,
)

logger = logging.getLogger(__name__)


def TraceOptions(**kwargs) -> _TraceOptions:
    """Build spans that never export raw Databricks provider exceptions."""
    return _TraceOptions(record_exceptions=False, **kwargs)


BUFFER_SIZE = 1024 * 1024  # 1mb


def retry_policy():
    try:
        from databricks.sdk.errors import platform
        ex_types = (
            platform.ResourceConflict,
            platform.TooManyRequests,
            platform.InternalError,
            platform.TemporarilyUnavailable,
            platform.DeadlineExceeded,
            RuntimeError,
            TimeoutError,
            ValueError,  # invalid_client auth failures
        )
    except ModuleNotFoundError:
        ex_types = (Exception, )

    def _on_auth_error(e: Exception) -> None:
        logger.warning(
            "Databricks auth failed, reconnecting with fresh "
            "credentials from Secrets Manager (error_type=%s)",
            type(e).__name__)
        uc_ctx.reconnect()

    return retry(
        exceptions=ex_types,
        callbacks={_is_auth_error: _on_auth_error},
        tries=3,
        delay=2.0,
        backoff=2.0,
        jitter=0.5,
        logger=logger,
    )


ucv_fs_logging_policy = logging_policy(logger, restricted_failures=True)


def _is_auth_error(exc: Exception) -> bool:
    """Check if an exception is a Databricks authentication failure
    that could be resolved by refreshing credentials from SM."""
    return 'invalid_client' in str(exc)


class UCContext:
    """Wrapper for databricks.sdk.service.FilesAPI
    Imports lazily to allow import in non-databricks environments.

    On auth failure (e.g. after credential rotation), automatically
    reconnects with fresh credentials from AWS Secrets Manager.
    """
    def __init__(self):
        self._local = threading.local()
        self._lock = threading.Lock()
        self._connection_info: Optional[DatabricksConnectionInfo] = None

    @retry(exceptions=(ValueError, MaxRetryError, ConnectionError), tries=7,
           delay=5, backoff=2, logger=logger)
    def _connect(self,
                 connection_info: Optional[DatabricksConnectionInfo] = None):
        from databricks.sdk import WorkspaceClient
        config = self.config(connection_info)
        # Configure timeouts and connection pool.
        # http_timeout_seconds: per-request timeout. Reduced from 300s to 60s
        # to fail fast on stuck connections rather than blocking the pool.
        # retry_timeout_seconds: total retry window for transient failures.
        # max_connections_per_pool: SDK default is 20, increase to handle
        # concurrent file operations without pool exhaustion.
        config.http_timeout_seconds = 60
        config.retry_timeout_seconds = 300
        config.max_connections_per_pool = 40
        config.disable_experimental_files_api_client = False  # pyright: ignore
        self._local.pid = os.getpid()
        self._local.client = WorkspaceClient(config=config)

    def reconnect(self) -> None:
        """Clear cached client and connection info so the next access
        re-reads credentials from AWS Secrets Manager.

        Note: _local is threading.local, so this only clears the current
        thread's cached client. Other threads keep their stale clients
        until each hits an auth failure and triggers its own reconnect.
        This converges eventually but is not atomic across threads.
        TODO: consider a cross-thread invalidation mechanism if the
        per-thread convergence becomes a problem in practice.
        """
        self._local.client = None
        with self._lock:
            self._connection_info = None

    @property
    def client(self):
        from databricks.sdk import WorkspaceClient
        if hasattr(self._local, "pid") and self._local.pid != os.getpid():
            self._local.client = None
        if not hasattr(self._local, "client") or self._local.client is None:
            try:
                self._connect()
            except ModuleNotFoundError:
                raise RuntimeError("calling databricks operations "
                                   "from non-databricks environment") \
                    from None
            except ClientError:
                raise RuntimeError("databricks connection failed") from None

        assert self._local.client is not None
        assert isinstance(self._local.client, WorkspaceClient)
        return self._local.client

    def config(self,
               connection_info: Optional[DatabricksConnectionInfo] = None):
        from databricks.sdk.core import Config
        if connection_info:
            with self._lock:
                self._connection_info = connection_info
        else:
            if not self._connection_info:
                with self._lock:
                    if not self._connection_info:
                        self._connection_info = DatabricksConnectionInfo.get()
        host = self._connection_info.host
        token = self._connection_info.credentials.get("pat", None)
        client_id = self._connection_info.credentials.get('client_id', None)
        client_secret = self._connection_info.credentials.get(
            'client_secret', None)
        if not host:
            raise RuntimeError("no databricks hostname provided (env: "
                               f"{DATABRICKS_HOST.name})")
        elif token:
            return Config(host=host, auth_type="pat",
                          token=token.get_secret_value())
        elif client_id and client_secret:
            return Config(host=host, auth_type="oauth-m2m",
                          client_id=client_id.get_secret_value(),
                          client_secret=client_secret.get_secret_value())
        else:
            raise RuntimeError("no databricks credentials provided (env: "
                               f"{DATABRICKS_TOKEN.name} or "
                               f"{DATABRICKS_CLIENT_ID.name} and "
                               f"{DATABRICKS_CLIENT_SECRET.name})")


uc_ctx = UCContext()


@api.object_identity.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy()
def object_identity(uri: UCURI) -> api.ObjectIdentity:
    """Capture the Unity Catalog file's exact closed-object metadata."""
    from databricks.sdk.errors.platform import NotFound
    try:
        metadata = uc_ctx.client.files.get_metadata(uri.as_path())
    except NotFound:
        raise FileNotFoundError(f"not found: {uri.as_url()}") from None
    if metadata.content_length is None or metadata.last_modified is None:
        raise ValueError("Unity Catalog object identity is incomplete")
    return api.ObjectIdentity(
        size=int(metadata.content_length),
        etag=None,
        version=str(metadata.last_modified),
    )


@api.filesize.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy()
def filesize(uri: UCURI) -> Optional[int]:
    from databricks.sdk.errors.platform import NotFound
    try:
        ret = uc_ctx.client.files.get_metadata(uri.as_path())
        return ret.content_length
    except NotFound:
        return None


@api.read_binary.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy(allowed_exceptions=FileNotFoundError)
def read_binary(uri: UCURI, compression: bool) -> bytes:
    from databricks.sdk._base_client import _StreamingResponse
    from databricks.sdk.errors.platform import NotFound
    try:
        res = uc_ctx.client.files.download(uri.as_path())
    except NotFound:
        raise FileNotFoundError(f"not found: {uri.as_url()}") from None
    if res.contents is None:
        raise FileNotFoundError(f"no contents: {uri.as_url()}")
    assert isinstance(res.contents, _StreamingResponse)
    res.contents.set_chunk_size(BUFFER_SIZE)  # pyright: ignore
    try:
        data = res.contents.read()
    finally:
        # Explicitly close to return connection to pool. Without this,
        # connections are only released via GC, which can cause pool
        # exhaustion and hangs (SDK uses blocking pool by default).
        res.contents.close()
    _check_size_match(local=data, remote=uri)
    return gzip.decompress(data) if compression else data


@api.read_text.register
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy(allowed_exceptions=FileNotFoundError)
def read_text(uri: UCURI, compression: bool,
              encoding: Optional[str] = None) -> str:
    r = read_binary(uri, compression=compression)
    if encoding:
        return r.decode(encoding)
    return r.decode()


@api._write_binary.register
@retry_policy()
@trace(TraceOptions(trace_params=['uri', 'compression']))
@ucv_fs_logging_policy(skip_args=[1])
def write_binary(uri: UCURI, data: BinaryIO, compression: bool,
                 **kwargs) -> None:
    body: BinaryIO = io.BytesIO(gzip.compress(
        data.read())) if compression else data
    uc_ctx.client.files.upload(uri.as_path(), body, overwrite=True)


@api.write_text.register
@trace(TraceOptions(trace_params=['uri', 'compression', 'encoding']))
@ucv_fs_logging_policy(skip_args=[1])
def write_text(uri: UCURI, data: str, compression: bool,
               encoding: Optional[str] = None) -> None:
    api.write_binary(uri.as_url(),
                     data.encode(encoding) if encoding else data.encode(),
                     compression=compression)


@api.dirsize.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy()
def dirsize(uri: UCURI) -> int:
    from databricks.sdk.errors.platform import NotFound
    from databricks.sdk.service.files import DirectoryEntry

    def get_size(entry: DirectoryEntry) -> int:
        if entry.is_directory:
            subdir = deepcopy(uri)
            assert entry.name is not None
            subdir.path = osp.join(subdir.path, entry.name) if len(
                subdir.path) > 0 else entry.name
            return dirsize(subdir)
        else:
            assert entry.file_size is not None
            return entry.file_size

    try:
        return sum(
            map(get_size,
                uc_ctx.client.files.list_directory_contents(uri.as_path())))
    except NotFound:
        raise FileNotFoundError(str(uri))


def _check_size_match(*, local: Union[LocalURI, bytes], remote: UCURI):
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
            f"UC Volumes file size mismatch, {err_msg}"
            f"remote file {remote.as_url()} is {remote_size} bytes")


@retry_policy()
def _upload(*, local: LocalURI, remote: UCURI):
    try:
        uc_ctx.client.files.upload_from(  # pyright: ignore
            remote.as_path(), local.path, overwrite=True)
    except Exception as e:
        uc_ctx._local.client = None
        logger.warning(
            "Upload failed, re-initializing client before retry "
            "(error_type=%s)",
            type(e).__name__)
        raise
    _check_size_match(local=local, remote=remote)


@retry_policy()
def _download(*, remote: UCURI, local: LocalURI):
    """Files API download"""
    from databricks.sdk.errors.platform import NotFound
    try:
        # Use parallel download for large files. databricks
        # sdk will automatically decide whether to use parallelism
        # based on file size. Default cutoff is 50MB.
        # We reduce the parallelism to 5 since fsutil could
        # also be using multi-process downloading.
        uc_ctx.client.files.download_to(  # pyright: ignore
            remote.as_path(), local.path, overwrite=True, use_parallel=True,
            parallelism=5)
    except NotFound:
        raise FileNotFoundError(f"not found: {remote.as_url()}") from None

    _check_size_match(remote=remote, local=local)


@api._get_binary_handle.register
@trace(TraceOptions(trace_params=True))
def get_binary_handle(
    uri: UCURI,
    mode: str,
    compression: bool = False,
) -> Generator[BinaryIO, None, None]:
    with restricted_diagnostic_boundary(
            safe_message='Databricks volume stream operation failed',
            exception_types=(DatabricksError, ClientError, ConnectionError,
                             MaxRetryError, OSError)):
        if 'b' not in mode:
            mode += 'b'

        with TemporaryDirectory() as temp_dir:
            temp_file = osp.join(temp_dir, "temp_file")
            local = LocalURI(path=temp_file)
            try:
                if 'r' in mode:
                    # download initial contents
                    _download(remote=uri, local=local)

                with open(temp_file, mode=mode) as f:
                    if compression:
                        with gzip.GzipFile(fileobj=f, mode=mode) as gzf:
                            yield gzf  # type: ignore
                    else:
                        yield f  # type: ignore
            finally:
                if 'w' in mode:
                    _upload(local=local, remote=uri)


@api._get_handle.register
@trace(TraceOptions(trace_params=True))
def get_handle(
    uri: UCURI,
    mode: str,
    compression: bool = False,
) -> Generator[TextIO, None, None]:
    with api.get_binary_handle(uri.as_url(), mode + 'b',
                               compression=compression) as handle:
        yield io.TextIOWrapper(
            buffer=handle,
            encoding='utf-8',
            write_through=True,
        )


@api.exists.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy()
def exists(uri: UCURI) -> bool:
    from databricks.sdk.errors.platform import (
        BadRequest,
        NotFound,
        PermissionDenied,
    )
    try:
        uc_ctx.client.files.get_metadata(uri.as_path())
        return True
    except (BadRequest, NotFound, PermissionDenied):
        pass
    try:
        uc_ctx.client.files.get_directory_metadata(uri.as_path())
        return True
    except (BadRequest, NotFound, PermissionDenied):
        pass
    return False


@api.isdir.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy()
def isdir(uri: UCURI) -> bool:
    from databricks.sdk.errors.platform import NotFound
    try:
        uc_ctx.client.files.get_directory_metadata(uri.as_path())
        return True
    except NotFound:
        return False


@api._get_if_present.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy()
def get_if_present(uri: UCURI) -> Optional[bytes]:
    from databricks.sdk.errors.platform import NotFound
    try:
        res = uc_ctx.client.files.download(uri.as_path())
        if res.contents is None:
            return None
        try:
            return res.contents.read()
        finally:
            # Explicitly close to return connection to pool.
            res.contents.close()
    except NotFound:
        return None


@ucv_fs_logging_policy(allowed_exceptions=FileNotFoundError)
def copy_file(src: str, dst: str):
    src_uri = ParseURI(src)
    dst_uri = ParseURI(dst)
    # Bypass local tmp file for uploading local file to UCV and downloading
    # UCV file to local file
    if isinstance(src_uri, LocalURI) and isinstance(dst_uri, UCURI):
        return _upload(local=src_uri, remote=dst_uri)
    if isinstance(src_uri, UCURI) and isinstance(dst_uri, LocalURI):
        return _download(remote=src_uri, local=dst_uri)
    else:
        with api.get_binary_handle(src, 'rb') as fsrc:
            with api.get_binary_handle(dst, 'wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)
        assert api.filesize(src) == api.filesize(dst)


@api.touch.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy()
def touch(uri: UCURI) -> None:
    from databricks.sdk.errors.platform import AlreadyExists
    try:
        uc_ctx.client.files.upload(uri.as_path(), io.BytesIO(b''),
                                   overwrite=False)
    except AlreadyExists:
        pass


@api.write_text_if_absent.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy(skip_args=[1])
def write_text_if_absent(uri: UCURI, data: str) -> bool:
    from databricks.sdk.errors.platform import AlreadyExists
    try:
        uc_ctx.client.files.upload(
            uri.as_path(),
            io.BytesIO(data.encode()),
            overwrite=False,
        )
    except AlreadyExists:
        return False
    return True


@api.has_write_permission.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy()
def has_write_permission(uri: UCURI) -> bool:
    from databricks.sdk.errors.platform import NotFound, PermissionDenied
    from databricks.sdk.service.catalog import Privilege, SecurableType

    user_name = uc_ctx.client.current_user.me().user_name
    volume_full_name = f"{uri.catalog}.{uri.uc_schema}.{uri.volume}"
    try:
        owner = uc_ctx.client.volumes.read(volume_full_name).owner
    except (NotFound, PermissionDenied):
        return False

    if owner == user_name:
        # volume owner always has write permissions
        return True

    # get explicitly set permissions
    grants = uc_ctx.client.grants.get_effective(
        full_name=volume_full_name,
        securable_type=SecurableType.VOLUME.value,  # pyright: ignore
        principal=user_name)
    for assignment in grants.privilege_assignments or []:
        for privilege in assignment.privileges or []:
            if privilege.privilege == Privilege.ALL_PRIVILEGES:
                return True
            if privilege.privilege == Privilege.WRITE_VOLUME:
                return True

    return False


@api._listdir.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy()
def listdir(
    uri: UCURI,
    max_items: Optional[int] = None,
) -> list[str]:
    from databricks.sdk.errors.platform import NotFound

    def extract_path(entry):
        return f"dbfs:{entry.path.rstrip('/')}"

    entries = []
    try:
        entries = list(
            map(extract_path,
                uc_ctx.client.files.list_directory_contents(uri.as_path())))
    except NotFound:
        entries = []
    return entries


@api._listdir_detail.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy()
def listdir_detail(
    uri: UCURI,
    max_items: Optional[int] = None,
) -> list[api.PathDetail]:
    from databricks.sdk.errors.platform import NotFound

    def extract_file_info(entry):
        return api.PathDetail(path=f"dbfs:{entry.path.rstrip('/')}",
                              is_directory=entry.is_directory)

    entries = []
    try:
        entries = list(
            map(extract_file_info,
                uc_ctx.client.files.list_directory_contents(uri.as_path())))
    except NotFound:
        entries = []
    return entries


@api.makedirs.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy()
def makedirs(uri: UCURI, exist_ok: bool = False) -> None:
    if not exist_ok and exists(uri):
        raise FileExistsError(f"{uri.as_url()} already exists")
    uc_ctx.client.files.create_directory(uri.as_path())


@api.rm.register
@retry_policy()
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy()
def rm(uri: UCURI, ignore_errors: bool = False,
       include: Optional[List[str]] = None) -> None:
    if include is not None:
        raise NotImplementedError("include is not implemented for UC Volume")
    if isdir(uri):
        for entry in listdir(uri):
            rm(ParseURI(entry))
        uc_ctx.client.files.delete_directory(uri.as_path())
    elif exists(uri):
        uc_ctx.client.files.delete(uri.as_path())
    else:
        pass


@api._contains_subdir.register
@trace(TraceOptions(trace_params=True))
@ucv_fs_logging_policy()
def contains_subdir(uri: UCURI) -> bool:
    """Check if a path contains sub directories."""
    if not isdir(uri):
        return False
    for entry in listdir(uri):
        if isdir(ParseURI(entry)):
            return True
    return False


@api.get_last_modified_time.register
@trace(TraceOptions(trace_params=True))
@retry_policy()
@ucv_fs_logging_policy()
def get_last_modified_time(uri: UCURI, suffix: Optional[str] = None) -> float:
    from databricks.sdk.errors.platform import NotFound
    ucv_path = uri.as_path()
    # UC Volume does not track last modified for directories.
    # The only case where a parquet connector points to a UC Volume path
    # is when the table is the train table. In this case, we retrieve the
    # last modified timestamp from the _SUCCESS file under the dir.
    if isdir(uri):
        file_uri = deepcopy(uri)
        file_uri.path = osp.join(uri.path, '_SUCCESS')
        if exists(file_uri):
            ucv_path = file_uri.as_path()
        else:
            return DEFAULT_DATA_VERSION

    try:
        ret = uc_ctx.client.files.get_metadata(ucv_path)
        if ret.last_modified is None:  # pragma: no cover
            return DEFAULT_DATA_VERSION
        timestamp = datetime.strptime(ret.last_modified,
                                      "%a, %d %b %Y %H:%M:%S %Z")
        return timestamp.timestamp()
    except NotFound:
        return DEFAULT_DATA_VERSION


@api.get_presigned_helper.register
def get_presigned_helper(uri: UCURI) -> None:
    return None


@api._zip_directory.register
def zip_directory(
    uri: UCURI,
    recursive: bool = False,
    output_path: Optional[str] = None,
) -> str:
    raise NotImplementedError("zip_directory is not implemented for UCV")


@api.customer_facing_path.register
def customer_facing_path(uri: UCURI) -> str:
    return uri.as_url()
