# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import logging
import multiprocessing as mp
import os
import queue
import random
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import wraps
from io import BytesIO
from typing import IO, Callable, List, Optional, Tuple, Type, TypeVar, Union
from urllib import parse
from zipfile import ZipFile

import boto3
import botocore
import botocore.config
import botocore.exceptions
import botocore.session
import mypy_boto3_s3.client
import mypy_boto3_s3.service_resource
from botocore.credentials import CredentialProvider
from kumo_s3.download import download_file as download_s3_file
from mypy_boto3_s3 import S3Client
from mypy_boto3_s3.type_defs import (
    CopySourceTypeDef,
    ErrorTypeDef,
    HeadObjectOutputTypeDef,
)

from async_utils.sync import blocking_func
from io_utils.fsutil.common import (
    TemporaryDirectory,
    fsutil_ctx,
    should_include,
)
from io_utils.fsutil.credentials import (
    ContextPropagatingThreadPoolExecutor,
    credentials_for_provider,
    scoped_cache_key,
)
from io_utils.fsutil.s3_connection import (
    active_s3_connection,
    bucket_owner_write_args,
)
from io_utils.fsutil.uri import S3URI, ParseURI
from monitoring.tracing import TraceOptions as _TraceOptions
from monitoring.tracing import trace

logger = logging.getLogger(__name__)


def TraceOptions(**kwargs) -> _TraceOptions:
    """Build spans that never export raw AWS exception payloads."""
    return _TraceOptions(record_exceptions=False, **kwargs)


# No.of times to retry the operation
DEFAULT_MAX_NUM_RETRIES = 0
# No.of seconds between each retry
DEFAULT_DELAY_SECONDS = 1

_LISTINGS_EXPIRY_TIME_SEC = 15

T = TypeVar('T')

_NO_CREDENTIALS_EXCEPTIONS = (
    botocore.exceptions.NoCredentialsError,
    botocore.exceptions.PartialCredentialsError,
)


class ThreadLocalBotoSession(threading.local):
    """Returns a thread-safe boto3 session. The first call to get() will
    initialize the session, and may possibly throw an exception.

    Note, the default boto session `boto3.client('s3')` is not thread-safe. In
    all situations where a boto client may be called from multiple threads, is
    it required to use `threadsafe_boto.s3_client()`.
    """
    def __init__(self) -> None:
        self._region: Optional[str] = None
        self._endpoint_url: Optional[str] = None
        self._credentials: Optional[CredentialProvider] = None
        self._core_session = None
        self._session = None
        self._s3_client = None
        self._s3_resource = None
        self._deployment_identity = None
        self._bucket_regions: dict[str, str] = {}
        # Any retry attempt will include an exponential backoff by a base
        # factor of 2 for a maximum backoff time of 20 seconds.
        # Here we retry for about 7.5 minutes.
        self._config = botocore.config.Config(
            signature_version='s3v4',
            retries={
                'max_attempts': 24,
                'mode': 'standard'
            },
        )

    @staticmethod
    def _has_session_credentials(credentials: Mapping[str, object]) -> bool:
        return bool(credentials.get('ambient_session')) or all(
            map(credentials.get,
                ('aws_access_key_id', 'aws_secret_access_key')))

    def core_session(self) -> botocore.session.Session:
        """botocore Session containing parameters, but no connection"""
        if self._core_session is None:
            self._core_session = botocore.session.Session()
            if self._credentials:
                provider_component = self._core_session.get_component(
                    'credential_provider')
                provider_component.insert_before('env', self._credentials)
        return self._core_session

    def session(self) -> boto3.session.Session:
        """boto3 Session containing a botocore Session and a connection"""
        scope = credentials_for_provider('s3')
        if scope is not None:
            key = scoped_cache_key('s3', 'session')
            session = scope.cache.get(key)
            if session is None:
                credentials = scope.credentials or {}
                ambient_session = credentials.get('ambient_session')
                if ambient_session is not None:
                    if not isinstance(ambient_session, boto3.session.Session):
                        raise RuntimeError(
                            'Invalid ambient AWS session for S3 identity')
                    scope.cache[key] = ambient_session
                    return ambient_session
                if not self._has_session_credentials(credentials):
                    raise RuntimeError(
                        'Incomplete federated AWS credentials for S3 identity')
                connection = active_s3_connection()
                session = boto3.session.Session(
                    aws_access_key_id=credentials['aws_access_key_id'],
                    aws_secret_access_key=credentials['aws_secret_access_key'],
                    aws_session_token=credentials['aws_session_token'],
                    region_name=(connection.region if connection else None)
                    or credentials.get('region') or self._region,
                )
                scope.cache[key] = session
            return session
        connection = active_s3_connection()
        if connection is not None:
            if (self._session is None
                    or self._deployment_identity != connection.identity):
                self._session = boto3.session.Session(
                    aws_access_key_id=connection.access_key,
                    aws_secret_access_key=connection.secret_key,
                    aws_session_token=connection.session_token,
                    region_name=connection.region,
                )
                self._deployment_identity = connection.identity
            return self._session
        if self._session is None:
            self._session = boto3.session.Session(
                region_name=self._region,
                botocore_session=self.core_session(),
            )
        return self._session

    def s3_client(self) -> mypy_boto3_s3.client.S3Client:
        scope = credentials_for_provider('s3')
        if scope is not None:
            key = scoped_cache_key('s3', 'client')
            client = scope.cache.get(key)
            if client is None:
                credentials = scope.credentials or {}
                connection = active_s3_connection()
                config = self._config
                if connection is not None:
                    config = config.merge(connection.boto_config)
                client = self.session().client(
                    's3',
                    config=config,
                    endpoint_url=(connection.endpoint_url
                                  if connection else None),
                    region_name=(connection.region if connection else None)
                    or credentials.get('region') or self._region,
                    verify=(connection.boto_verify if connection else None),
                )
                scope.cache[key] = client
            return client
        connection = active_s3_connection()
        if connection is not None:
            if (self._s3_client is None
                    or self._deployment_identity != connection.identity):
                config = self._config.merge(connection.boto_config)
                self._s3_client = self.session().client(
                    's3',
                    config=config,
                    endpoint_url=connection.endpoint_url,
                    region_name=connection.region,
                    verify=connection.boto_verify,
                )
                self._deployment_identity = connection.identity
            return self._s3_client
        if self._s3_client is None:
            if self._endpoint_url is not None:
                # Use virtual addressing for custom endpoints
                config = self._config.merge(
                    botocore.config.Config(s3={'addressing_style': 'virtual'}))
            else:
                # Use default addressing style for AWS endpoints
                config = self._config
            self._s3_client = self.session().client(
                's3', config=config, endpoint_url=self._endpoint_url)
        return self._s3_client

    def s3_resource(self) -> mypy_boto3_s3.service_resource.S3ServiceResource:
        scope = credentials_for_provider('s3')
        if scope is not None:
            key = scoped_cache_key('s3', 'resource')
            resource = scope.cache.get(key)
            if resource is None:
                credentials = scope.credentials or {}
                connection = active_s3_connection()
                config = self._config
                if connection is not None:
                    config = config.merge(connection.boto_config)
                resource = self.session().resource(
                    's3',
                    config=config,
                    endpoint_url=(connection.endpoint_url
                                  if connection else None),
                    region_name=(connection.region if connection else None)
                    or credentials.get('region') or self._region,
                    verify=(connection.boto_verify if connection else None),
                )
                scope.cache[key] = resource
            return resource
        connection = active_s3_connection()
        if connection is not None:
            if (self._s3_resource is None
                    or self._deployment_identity != connection.identity):
                config = self._config.merge(connection.boto_config)
                self._s3_resource = self.session().resource(
                    's3',
                    config=config,
                    endpoint_url=connection.endpoint_url,
                    region_name=connection.region,
                    verify=connection.boto_verify,
                )
                self._deployment_identity = connection.identity
            return self._s3_resource
        if self._s3_resource is None:
            if self._endpoint_url is not None:
                # Use virtual addressing for custom endpoints
                config = self._config.merge(
                    botocore.config.Config(s3={'addressing_style': 'virtual'}))
            else:
                # Use default addressing style for AWS endpoints
                config = self._config
            self._s3_resource = self.session().resource(
                's3', config=config, endpoint_url=self._endpoint_url)
        return self._s3_resource

    def _reset(self) -> None:
        self._core_session = None
        self._session = None
        self._s3_client = None
        self._s3_resource = None
        self._endpoint_url = None
        self._deployment_identity = None

    def lazy_auth(self) -> None:
        if credentials_for_provider('s3') is not None:
            return
        if active_s3_connection() is not None:
            return
        if self._credentials is not None:
            self._reset()


threadsafe_boto = ThreadLocalBotoSession()


def select_s3_region(path: str) -> None:
    scope = credentials_for_provider('s3')
    connection = active_s3_connection()
    if connection is not None and connection.endpoint_url is not None:
        return
    if scope is None and threadsafe_boto._endpoint_url is not None:
        return
    if (scope is not None and not threadsafe_boto._has_session_credentials(
            scope.credentials or {})):
        return
    uri = ParseURI(path)
    if not isinstance(uri, S3URI):
        raise ValueError('S3 region selection requires an S3 URI')
    bucket = uri.bucket
    if scope is None:
        region = threadsafe_boto._bucket_regions.get(bucket)
    else:
        region_key = scoped_cache_key('s3', f'region:{bucket}')
        region = scope.cache.get(region_key)
    if region is None:
        failure = None
        try:
            response = threadsafe_boto.s3_client().head_bucket(Bucket=bucket)
        except botocore.exceptions.ClientError as error:
            failure = error
            response = error.response
        region = (response.get('BucketRegion')
                  or response.get('ResponseMetadata', {}).get(
                      'HTTPHeaders', {}).get('x-amz-bucket-region'))
        if region is None:
            if failure is not None:
                raise failure
            region = 'us-east-1'
        if scope is None:
            threadsafe_boto._bucket_regions[bucket] = region
        else:
            scope.cache[scoped_cache_key('s3', f'region:{bucket}')] = region
    if scope is None:
        if threadsafe_boto._region != region:
            threadsafe_boto._region = region
            threadsafe_boto._reset()
        return
    if (scope.credentials or {}).get('region') == region:
        return
    scope.credentials = {**(scope.credentials or {}), 'region': region}
    for discriminator in ('session', 'client', 'resource'):
        scope.cache.pop(scoped_cache_key('s3', discriminator), None)


def _unsigned_s3_client() -> mypy_boto3_s3.client.S3Client:
    config = threadsafe_boto._config.merge(  # pylint: disable=protected-access
        botocore.config.Config(signature_version=botocore.UNSIGNED))
    return threadsafe_boto.session().client(
        's3',
        config=config,
        endpoint_url=threadsafe_boto._endpoint_url,  # pylint: disable=protected-access
    )


@dataclass
class _ListResponse:
    # Keys correspond to "files" in the current dir.
    keys: List[str] = field(default_factory=list)

    # Prefixes correspond to "sub directories" in the current dir.
    prefixes: List[str] = field(default_factory=list)


def parse_s3(path: str) -> Tuple[str, str]:
    r"""Return bucket and path of a s3 path.

    Args:
        path (str): A s3 file path.
    """
    uri = ParseURI(path)
    if not isinstance(uri, S3URI):
        raise ValueError(f"S3 path must use the s3 scheme, got {path}")
    return uri.bucket, uri.path.strip("/")


def parse_s3_file(s3_path: str) -> Tuple[str, str]:
    r"""Return bucket and path of a s3 file. Raise exception if the path is to
    a directory (ends in trailing slash). Raise exception if not an S3 URL.
    """
    if not (s3_path.startswith('s3://') or s3_path.startswith('s3a://')):
        raise ValueError(f"S3 path must start with 's3://' or 's3a://"
                         f", but got {s3_path})")
    uri = ParseURI(s3_path)
    if not isinstance(uri, S3URI):
        raise ValueError(f"S3 path must use the s3 scheme, got {s3_path}")
    bucket = uri.bucket
    path = uri.path
    if len(path) == 0 or path[-1] == '/':
        raise ValueError(f"S3 path must point to a file (no trailing slash)"
                         f", but got {s3_path})")
    return bucket, path


def parse_s3_dir(s3_path: str) -> Tuple[str, str]:
    r"""Return bucket and path of a s3 directory. Append a trailing slash if
    not present (assumes that the given path points to a directory). Raise
    exception if not an S3 URL.

    Args:
        path (str): A s3 file path.
    """
    if not (s3_path.startswith('s3://') or s3_path.startswith('s3a://')):
        raise ValueError(f"S3 path must start with 's3://' or 's3a://"
                         f", but got {s3_path})")
    uri = ParseURI(s3_path)
    if not isinstance(uri, S3URI):
        raise ValueError(f"S3 path must use the s3 scheme, got {s3_path}")
    bucket = uri.bucket
    path = uri.path
    if len(path) > 0 and path[-1] != '/':
        path += '/'
    return bucket, path


def executeWithRetry(
        max_num_retries: int = DEFAULT_MAX_NUM_RETRIES,
        delay_s: float = DEFAULT_DELAY_SECONDS) -> Callable[[T], T]:
    r"""Decorator to retry executing a function. By default only tries once
    i.e. no retries

    Usage:
        # Try maximum 3 times with a delay of 10 seconds.
        @executeWithRetry(max_num_retries=3, delay_s=10)

        # For executing only once i.e. no retries
        @executeWithRetry()
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):  # noqa pylint: disable=inconsistent-return-statements
            total_tries = max_num_retries + 1
            assert total_tries > 0
            for i in range(total_tries):
                try:
                    res = func(*args, **kwargs)
                    if i > 0:
                        logger.info(f"Try {i} (of {total_tries}) succeeded")
                    return res
                except Exception as e:  # pylint: disable=broad-except
                    logger.error(
                        "Try %s of %s for %s failed (error_type=%s)",
                        i,
                        total_tries,
                        func.__name__,
                        type(e).__name__,
                    )
                    # Raise the exception if this is the last try
                    if i == max_num_retries:
                        raise e
                    time.sleep(delay_s)

        return wrapper

    return decorator  # type: ignore


class S3FileSystem:
    """Class containing highly optimized s3 routines based on Boto3.

    This class exists to contain common configuration settings about client
    behavior (eg. retry), which may differ between production and unit test
    environments.

    This class is thread safe
    """
    def __init__(
        self,
        process_pool_cls: Union[
            Type[ProcessPoolExecutor],
            Type[ContextPropagatingThreadPoolExecutor]] = ProcessPoolExecutor,
        retry_delay_sec: int = 5,
    ):
        self._process_pool_cls = process_pool_cls
        self._retry_delay_sec = retry_delay_sec

        # Maximum batch size supported by S3 DeleteObjects is 1000
        self._delete_object_batch_size = 999

        # Minimum number of keys to enable multiprocessing on copy operations
        self._copy_multiprocessing_threshold = 500

    @property
    def _s3_client(self):
        return threadsafe_boto.s3_client()

    _THREAD_POOL_SIZE = 50

    @classmethod
    def for_test(cls) -> 'S3FileSystem':
        return cls(
            # moto does not play nicely with process pools
            process_pool_cls=ContextPropagatingThreadPoolExecutor,
            # eliminate retry latency to speed up unit tests
            retry_delay_sec=0,
        )


s3_ctx = S3FileSystem()

# ----------------------- DELETE -----------------------


def _delete_batch(bucket: str, keys: List[str]) -> None:
    """ Call the s3 batch delete endpoint.
    """
    threadsafe_boto.lazy_auth()  # required in threadpool
    objects = []
    for key in keys:
        objects.append({
            'Key': key,
        })

    result = s3_ctx._s3_client.delete_objects(
        Bucket=bucket,
        Delete={
            'Objects': objects,
        },
    )

    errors: List[ErrorTypeDef] = result.get('Errors', [])
    if len(errors) > 0:
        error = errors[0]
        raise RuntimeError(
            f"Failed to delete {len(errors)} keys in batch. Example error "
            f"for Key={error.get('Key')}: Code={error.get('Code')} "
            f"Message={error.get('Message')}")


@trace(TraceOptions(trace_params=True))
def rmdir(path: str, include: Optional[List[str]] = None) -> None:
    r"""Highly optimized version of recursive S3 delete. Able to delete 10k
    files in < 10 seconds.
    """
    logger.debug("Delete recursive %s", path)
    bucket_name, _ = parse_s3_dir(path)
    keys = listdir_recursive(path)
    keys = [k for k in keys if should_include(k, include)]
    logger.debug("Found %s files to delete within %s", len(keys), path)

    _s3_delete_batch_with_retry = executeWithRetry(
        max_num_retries=2,
        delay_s=s3_ctx._retry_delay_sec,
    )(_delete_batch)

    tasks: List[Future] = []
    for i in range(0, len(keys), s3_ctx._delete_object_batch_size):
        task = fsutil_ctx.thread_pool.submit(
            _s3_delete_batch_with_retry,
            bucket_name,
            keys[i:i + s3_ctx._delete_object_batch_size],
        )
        tasks.append(task)
    for task in tasks:
        task.result()

    keys = listdir_recursive(path)
    if len(keys) > 0:
        logger.warning("Failed to delete the following keys in batch: "
                       f"{keys}")
    logger.debug("Done deleting %s", path)


# ----------------------- COPY -----------------------


def _copy_one(src_bucket: str, src_key: str, dst_bucket: str, dst_key: str,
              s3_client: S3Client) -> None:
    threadsafe_boto.lazy_auth()  # required in threadpool
    copy_source = CopySourceTypeDef(
        Bucket=src_bucket,
        Key=src_key,
    )
    extra_args = bucket_owner_write_args()
    s3_client.copy(copy_source, dst_bucket, dst_key, ExtraArgs=dict(extra_args))


def _copy_batch(src_bucket: str, src_prefix: str, src_keys: List[str],
                dst_bucket: str, dst_prefix: str, retry_delay_sec: int,
                executor: Executor, s3_client: S3Client) -> None:

    _s3_copy_one_with_retry = executeWithRetry(
        max_num_retries=2,
        delay_s=retry_delay_sec,
    )(_copy_one)

    tasks: List[Future] = []
    for src_key in src_keys:
        dst_key = dst_prefix + src_key[len(src_prefix):]
        tasks.append(
            executor.submit(_s3_copy_one_with_retry, src_bucket, src_key,
                            dst_bucket, dst_key, s3_client))
    for task in tasks:
        task.result()


def _copy_batch_new_process(src_bucket: str, src_prefix: str,
                            src_keys: List[str], dst_bucket: str,
                            dst_prefix: str, retry_delay_sec: int) -> None:
    client = threadsafe_boto.s3_client()
    with ContextPropagatingThreadPoolExecutor(
            max_workers=S3FileSystem._THREAD_POOL_SIZE) as executor:
        _copy_batch(src_bucket, src_prefix, src_keys, dst_bucket, dst_prefix,
                    retry_delay_sec, executor, client)


@trace(TraceOptions(trace_params=True))
@blocking_func()
def copydir(
    src: str,
    dst: str,
) -> None:
    """ Optimized function for doing s3->s3 copy. As this operation does
    not perform any local IO, the only bottleneck is the number of
    concurrent requests we can make to AWS. For large directories, this
    function will launch multiple thread and Python processes, in order to
    achieve the best parallelism.

    TODO: Replace fsutil.copy() with this function once it is battle-tested
    """
    logger.debug("Copying %s to %s", src, dst)
    src_bucket, src_prefix = parse_s3_dir(src)
    dst_bucket, dst_prefix = parse_s3_dir(dst)

    keys = listdir_recursive(src)
    logger.debug("Found %s files in %s", len(keys), src)
    logger.debug("Estimated copy time: %.1f minutes", len(keys) / 20000)

    if len(keys) == 0:
        raise ValueError("Cannot copy because source directory "
                         "is empty or does not exist: {src}")

    # For small number of files, use the current thread pool as it will be
    # fast enough
    # Process workers cannot inherit request-local ContextVars. Identity-
    # backed operations therefore stay in context-propagating threads even
    # for large copies; otherwise a child process could silently enter the
    # deployment's ambient AWS provider chain.
    if (len(keys) < s3_ctx._copy_multiprocessing_threshold
            or credentials_for_provider('s3') is not None):
        logger.debug("Using thread-pool for copy")
        _copy_batch(src_bucket, src_prefix, keys, dst_bucket, dst_prefix,
                    s3_ctx._retry_delay_sec, fsutil_ctx.thread_pool,
                    s3_ctx._s3_client)
        logger.debug("Done copying %s to %s", src, dst)
        return

    # Since there are 50 threads per process, a maximum of 5 processes will
    # limit max S3 parallelism to 250. Based on load tests, this gives us
    # the best mix of performance and stability.
    processes = min(mp.cpu_count() - 1, 5)

    # Assign batches so that each process gets exactly one batch.
    batch_size = int(len(keys) / processes) + 1

    # Ensure uniform distribution of work among processes, to prevent a
    # single process from handling the most-expensive files.
    random.shuffle(keys)

    logger.debug("Using mutiprocessing for copy")
    with s3_ctx._process_pool_cls(max_workers=processes) as executor:
        tasks: List[Future] = []
        for i in range(0, len(keys), batch_size):
            logger.info(
                f'Copying batch {i} of {len(keys) // batch_size} for {src}')
            task = executor.submit(_copy_batch_new_process, src_bucket,
                                   src_prefix, keys[i:i + batch_size],
                                   dst_bucket, dst_prefix,
                                   s3_ctx._retry_delay_sec)
            tasks.append(task)
        for task in tasks:
            logger.info(f'Waiting for task {task} to complete')
            task.result()
            logger.info(f'Task {task} completed')

    logger.debug("Done copying %s to %s", src, dst)


@trace(TraceOptions(trace_params=True))
def copy_file(src: str, dst: str):
    r"""Copy file from `src` to `dst`. Both `src` and `dst` are on s3."""
    src_bucket, src_key = parse_s3_file(src)
    src_resource = CopySourceTypeDef(
        Bucket=src_bucket,
        Key=src_key,
    )
    dst_bucket, dst_key = parse_s3_file(dst)

    extra_args = bucket_owner_write_args()
    s3_ctx._s3_client.copy(
        src_resource,
        dst_bucket,
        dst_key,
        ExtraArgs=dict(extra_args),
    )


# ----------------------- LIST -----------------------


def _list_objects_v2(bucket: str, prefix: str) -> _ListResponse:
    threadsafe_boto.lazy_auth()  # required in threadpool
    paginator = s3_ctx._s3_client.get_paginator('list_objects_v2')
    response = _ListResponse()

    for resp in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
        for obj in resp.get('Contents', []):
            response.keys.append(obj['Key'])  # type: ignore

        for obj in resp.get('CommonPrefixes', []):
            response.prefixes.append(obj['Prefix'])  # type: ignore

    return response


@trace(TraceOptions(trace_params=True))
def listdir_recursive(path: str) -> List[str]:
    """List all keys below an S3 path, fetching each directory in parallel."""
    logger.debug("Listing directory %s", path)
    _list_objects_v2_with_retry = executeWithRetry(
        max_num_retries=2,
        delay_s=s3_ctx._retry_delay_sec,
    )(_list_objects_v2)

    bucket, prefix = parse_s3_dir(path)
    keys: List[str] = []
    completed: queue.SimpleQueue[Future] = queue.SimpleQueue()
    pending = 0

    def submit(current_prefix: str) -> None:
        nonlocal pending
        pending += 1
        request = fsutil_ctx.thread_pool.submit(_list_objects_v2_with_retry,
                                                bucket, current_prefix)
        request.add_done_callback(completed.put)

    submit(prefix)
    while pending:
        response: _ListResponse = completed.get().result()
        pending -= 1
        keys.extend(response.keys)
        for child_prefix in response.prefixes:
            submit(child_prefix)

    return keys


# ----------------------- EXIST -----------------------


@trace(TraceOptions(trace_params=True))
def exists_s3_object(path: str) -> bool:
    """Path must be a file (no trailing slash).

    Raises exception if the S3 path is invalid, or if we do not have
    permission to read the file.
    """
    bucket, key = parse_s3_file(path)
    try:
        s3_ctx._s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response.get('Error', {}).get('Code') != "404":
            raise
        return False


@trace(TraceOptions(trace_params=True))
def head_object_with_unsigned_fallback(path: str) -> HeadObjectOutputTypeDef:
    bucket, key = parse_s3_file(path)
    try:
        return s3_ctx._s3_client.head_object(Bucket=bucket, Key=key)
    except _NO_CREDENTIALS_EXCEPTIONS:
        if credentials_for_provider('s3') is not None:
            raise
        logger.info("No AWS credentials found; retrying S3 head anonymously")
        return _unsigned_s3_client().head_object(Bucket=bucket, Key=key)


# ----------------------- GET -----------------------


@trace(TraceOptions(trace_params=True))
def get_handle(path: str, compressed: bool = False) -> IO:
    if compressed and not path.endswith('.gz'):
        raise ValueError(f"We only support gzip encoded file for s3,"
                         f" current path: {path}")

    bucket, key = parse_s3_file(path)
    try:
        response = s3_ctx._s3_client.get_object(Bucket=bucket, Key=key)
        body = response['Body']
        if not compressed:
            return body  # type: ignore
        else:
            return gzip.open(BytesIO(body.read()), 'rt')  # type: ignore
    except s3_ctx._s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(f"No such key {key}")


@trace(TraceOptions(trace_params=True))
def download_file(src: str, dst: str):
    r"""Download file from `src` to `dst`."""
    start_time = time.time()
    bucket_name, key = parse_s3_file(src)

    try:
        download_s3_file(
            bucket_name, key, dst, client=lambda: s3_ctx._s3_client,
            unsigned_client=_unsigned_s3_client
            if credentials_for_provider('s3') is None else None)
    except botocore.exceptions.ClientError as error:
        # Preserve fsutil's missing-file contract before generic.copy_file
        # sanitizes other provider errors into a transfer failure.
        if error.response.get(
                'Error', {}).get('Code') in {'404', 'NoSuchKey', 'NotFound'}:
            raise FileNotFoundError('S3 source object not found') from None
        raise

    time_spend = time.time() - start_time
    logger.debug(
        "Download %s takes %.2f sec, %.2f MB/s",
        src,
        time_spend,
        os.path.getsize(dst) / time_spend / 1024 / 1024,
    )


# ----------------------- PUT -----------------------


@trace(TraceOptions(trace_params=True))
def put_file(src: str, dst: str) -> None:
    r"""Put the `src` file to the `dst` path."""
    bucket, key = parse_s3_file(dst)
    threadsafe_boto.s3_client().upload_file(src, bucket, key)


# ----------------------- SIZE ----------------------


@trace(TraceOptions(trace_params=True))
def filesize(path: str) -> Optional[int]:
    r"""Returns the size of the file at path. Note that it will be None if
    the path is a directory.

    Args:
        path (str): A s3 file path.
    """
    bucket, key = parse_s3_file(path)
    try:
        obj = s3_ctx._s3_client.head_object(Bucket=bucket, Key=key)
        return obj['ContentLength']
    except Exception:
        return None


@trace(TraceOptions(trace_params=True))
def dirsize(path: str) -> int:
    r"""Returns the size of a directory in S3. Note that this function call
    may be arbitrarily expensive given the input path, and should be used
    with caution to avoid making many requests to S3.

    Behind the scenes, this code will call ListBuckets, ListObjects,
    and HeadObject many times. If you have a large number of S3 objects
    that are under this path, then this could incur a significant cost."""
    bucket, key = parse_s3_dir(path)
    return sum([
        obj.size
        for obj in threadsafe_boto.s3_resource().Bucket(bucket).objects.filter(
            Prefix=key)
    ])


# ----------------------- OTHER ---------------------


@trace(TraceOptions(trace_params=True))
def generate_presigned_url_for_download(path: str) -> str:
    r"""
    Creates a presigned url for s3 object.
    """
    try:
        s3_client = threadsafe_boto.s3_client()
        bucket, key = parse_s3_file(path)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket,
                'Key': key
            },
        )
        return presigned_url
    except Exception:
        raise ValueError(
            f"Permission error for generating presigned URL for path: "
            f"{path}.")


@trace(TraceOptions(trace_params=True))
def get_bucket_aws_region(bucket: str) -> str:
    r"""
    Return the AWS region of an AWS S3 bucket.
    """
    try:
        response = s3_ctx._s3_client.head_bucket(Bucket=bucket)
    except _NO_CREDENTIALS_EXCEPTIONS:
        if credentials_for_provider('s3') is not None:
            raise
        try:
            response = _unsigned_s3_client().head_bucket(Bucket=bucket)
        except Exception as e:
            raise ValueError(
                f"Error getting AWS region for bucket: {bucket}.") from e
    except Exception as e:
        raise ValueError(
            f"Error getting AWS region for bucket: {bucket}.") from e

    region = response.get('BucketRegion') or response.get(
        'ResponseMetadata', {}).get('HTTPHeaders',
                                    {}).get('x-amz-bucket-region')
    if not region:
        raise ValueError(f"Error getting AWS region for bucket: {bucket}.")
    return region


class S3FileLock:
    def __init__(self, path: str, timeout: int = 60):
        self.lock_acquired = False
        self._s3_client = threadsafe_boto.s3_client()
        o = parse.urlparse(path)
        self.bucket, self.key = o.netloc, o.path.strip("/")
        self.timeout = timeout

    def __enter__(self):
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            try:
                # Try to create the lock file, which is an empty object in S3
                s3_ctx._s3_client.put_object(
                    Bucket=self.bucket,
                    Key=self.key,
                    Body=b'',
                    **bucket_owner_write_args(),
                )
                self.lock_acquired = True
                return self
            except botocore.exceptions.ClientError as e:
                logger.error("File lock %s acquire error (error_type=%s)",
                             self.key,
                             type(e).__name__)
                time.sleep(1)

        raise TimeoutError("Timeout while acquiring the S3 lock")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.lock_acquired:
            self._s3_client.delete_object(Bucket=self.bucket, Key=self.key)
            self.lock_acquired = False


@blocking_func(tracing_options=TraceOptions(trace_params=True))
def zip_files_with_same_prefix(prefix: str, output_path: Optional[str] = None,
                               object_keys: Optional[List[str]] = None) -> str:
    r"""Zip a list of files on s3 with the same prefix. And return the output
    path.

    Args:
        prefix (str): The prefix of objects, note that it has to be multiple
            objects match the prefix.
        output_path (str): The output path, if file already exist, return it.
            If not specified, the default will be prefix + '.zip'.
    """
    # Setup the output path.
    if output_path is None or output_path == '':
        output_path = prefix.strip('/') + '.zip'

    s3_util = S3FileSystem()

    with S3FileLock(output_path + '.lock', timeout=60):
        if exists_s3_object(output_path):
            return output_path

        # Setup the input objects.
        object_keys = (listdir_recursive(prefix + '/')
                       if object_keys is None else list(object_keys))
        if not object_keys:
            raise ValueError('No S3 objects were selected for the archive')
        if len(set(object_keys)) != len(object_keys):
            raise ValueError('Duplicate S3 objects were selected for archive')
        bucket, _ = parse_s3_file(prefix)

        try:
            # Download the S3 objects and add them to the zip file
            with TemporaryDirectory() as temp_dir:
                temp_output_path = os.path.join(temp_dir, "output.zip")
                with ZipFile(temp_output_path, 'w') as zipf:
                    for object_key in object_keys:
                        logger.info("zipping object %s to %s", object_key,
                                    output_path)
                        response = s3_util._s3_client.get_object(
                            Bucket=bucket, Key=object_key)
                        object_data = response['Body'].read()
                        with BytesIO(object_data) as obj_file:
                            zipf.writestr(object_key, obj_file.getvalue())

                put_file(temp_output_path, output_path)
        except Exception:
            raise ValueError(
                f"Permission error for zip files with prefix: {prefix}.")

        return output_path
