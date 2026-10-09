# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import io
import logging
import os
import posixpath
import uuid
from datetime import datetime, timedelta, timezone
from typing import BinaryIO, Generator, List, Optional, Sequence, TextIO

from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
)
from smart_open import open as smart_open

from io_utils.errors import restricted_diagnostic_boundary
from io_utils.fsutil import api
from io_utils.fsutil.common import (
    DEFAULT_DATA_VERSION,
    fsExceptionHandler,
    should_include,
)
from io_utils.fsutil.credentials import credentials_for_provider
from io_utils.fsutil.logging_policy import logging_policy
from io_utils.fsutil.s3_connection import bucket_owner_write_args
from io_utils.fsutil.s3_filesystem import (
    executeWithRetry,
    rmdir,
    s3_ctx,
    select_s3_region,
    threadsafe_boto,
    zip_files_with_same_prefix,
)
from io_utils.fsutil.uri import S3URI, AnyURI
from monitoring.tracing import TraceOptions as _TraceOptions
from monitoring.tracing import trace

logger = logging.getLogger(__name__)


def TraceOptions(**kwargs) -> _TraceOptions:
    """Build spans that never export raw AWS exception payloads."""
    return _TraceOptions(record_exceptions=False, **kwargs)


s3_fs_logging_policy = logging_policy(logger, restricted_failures=True)

_NO_CREDENTIALS_EXCEPTIONS = (NoCredentialsError, PartialCredentialsError)


def _unsigned_s3_client():
    config = threadsafe_boto._config.merge(  # pylint: disable=protected-access
        Config(signature_version=UNSIGNED))
    return threadsafe_boto.session().client(
        's3',
        config=config,
        # Preserve custom endpoints used by tests or S3-compatible storage.
        endpoint_url=threadsafe_boto._endpoint_url,  # pylint: disable=protected-access
    )


def _read_with_unsigned_fallback(operation):
    try:
        return operation(s3_ctx._s3_client)
    except _NO_CREDENTIALS_EXCEPTIONS:
        if credentials_for_provider('s3') is not None:
            raise
        logger.info("No AWS credentials found; retrying S3 read anonymously")
        return operation(_unsigned_s3_client())


@api.filesize.register
@fsExceptionHandler
@s3_fs_logging_policy()
def filesize(uri: S3URI) -> Optional[int]:
    try:
        bucket, key = uri.file_bucket_key()
    except ValueError:
        # uri does not refer to a file
        return None

    try:
        obj = _read_with_unsigned_fallback(
            lambda client: client.head_object(Bucket=bucket, Key=key))
        return obj['ContentLength']
    except Exception as e:
        if not (isinstance(e, s3_ctx._s3_client.exceptions.NoSuchKey)
                or isinstance(e, s3_ctx._s3_client.exceptions.ClientError)):
            logger.warning(
                "Unexpected exception in S3 filesize "
                "(error_type=%s)",
                type(e).__name__)
        return None


@api.object_identity.register
@trace(TraceOptions(trace_params=True))
@s3_fs_logging_policy(allowed_exceptions=FileNotFoundError)
def object_identity(uri: S3URI) -> api.ObjectIdentity:
    """Capture S3 size + ETag without requiring version permissions."""
    bucket, key = uri.file_bucket_key()
    try:
        result = _read_with_unsigned_fallback(
            lambda selected: selected.head_object(Bucket=bucket, Key=key))
    except ClientError as error:
        if _get_error_code(error) in ('404', 'NoSuchKey', 'NotFound'):
            raise FileNotFoundError(uri.as_s3()) from None
        raise
    etag = result.get('ETag')
    if not isinstance(etag, str) or not etag:
        raise RuntimeError('S3 object metadata did not include an ETag')
    return api.ObjectIdentity(
        size=int(result['ContentLength']),
        etag=etag,
        version=None,
    )


@api.read_binary.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@s3_fs_logging_policy(allowed_exceptions=FileNotFoundError)
def read_binary(uri: S3URI, compression: bool) -> bytes:
    client = threadsafe_boto.s3_client()
    bucket, key = uri.file_bucket_key()
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
    except client.exceptions.NoSuchKey:
        raise FileNotFoundError(uri.as_s3())

    # Check if the file is fully downloaded
    expected_size = obj['ContentLength']  # Size from S3 metadata
    file_bytes = obj['Body'].read()
    actual_size = len(file_bytes)  # Size of downloaded data
    if expected_size != actual_size:
        raise IOError(
            f"File {uri} might be incomplete! Expected {expected_size} bytes, "
            f"but got {actual_size} bytes.")

    if compression:
        return gzip.decompress(file_bytes)
    else:
        return file_bytes


@api.read_text.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@s3_fs_logging_policy(allowed_exceptions=FileNotFoundError)
def read_text(uri: S3URI, compression: bool,
              encoding: Optional[str] = None) -> str:
    r = read_binary(uri, compression=compression)
    if encoding:
        return r.decode(encoding)
    return r.decode()


@api._write_binary.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=['uri', 'compression', 'ttl']))
@s3_fs_logging_policy(skip_args=[1])
def write_binary(uri: S3URI, data: BinaryIO, compression: bool,
                 ttl: Optional[timedelta]) -> None:
    client = threadsafe_boto.s3_client()
    bucket, key = uri.file_bucket_key()
    body: BinaryIO = io.BytesIO(gzip.compress(
        data.read())) if compression else data

    extra_args = {}
    if ttl:
        expires_at = datetime.now(timezone.utc) + ttl
        extra_args["Expires"] = expires_at  # S3 object expiry time

    client.upload_fileobj(Bucket=bucket, Key=key, Fileobj=body,
                          ExtraArgs=extra_args)


@api.write_text.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=['uri', 'compression', 'encoding']))
@s3_fs_logging_policy(skip_args=[1])
def write_text(uri: S3URI, data: str, compression: bool,
               encoding: Optional[str] = None) -> None:
    api.write_binary(uri.as_s3(),
                     data.encode(encoding) if encoding else data.encode(),
                     compression=compression)


@api.write_text_if_absent.register
@trace(TraceOptions(trace_params=['uri']))
@s3_fs_logging_policy(skip_args=[1])
def write_text_if_absent(uri: S3URI, data: str) -> bool:
    client = threadsafe_boto.s3_client()
    bucket, key = uri.file_bucket_key()
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data.encode(),
            IfNoneMatch='*',
            **bucket_owner_write_args(),
        )
    except ClientError as error:
        response = error.response
        code = str(response.get('Error', {}).get('Code', ''))
        status = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
        if code in {'PreconditionFailed', 'ConditionalRequestConflict'
                    } or status in {409, 412}:
            return False
        raise
    return True


@api._get_binary_handle.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
def get_binary_handle(
    uri: S3URI,
    mode: str,
    compression: bool = False,
) -> Generator[BinaryIO, None, None]:
    with restricted_diagnostic_boundary(
            safe_message='S3 stream operation failed',
            exception_types=(BotoCoreError, ClientError, OSError)):
        with smart_open(
                uri.as_s3(), mode, encoding=None,
                compression=".gz" if compression else None,
                transport_params={'client': threadsafe_boto.s3_client()
                                  }) as f:  # type: ignore
            yield f  # type: ignore


@api._get_handle.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
def get_handle(
    uri: S3URI,
    mode: str,
    compression: bool = False,
) -> Generator[TextIO, None, None]:
    with restricted_diagnostic_boundary(
            safe_message='S3 stream operation failed',
            exception_types=(BotoCoreError, ClientError, OSError)):
        with smart_open(
                uri.as_s3(), mode, encoding='utf-8',
                compression=".gz" if compression else None,
                transport_params={'client': threadsafe_boto.s3_client()
                                  }) as f:  # type: ignore
            yield f  # type: ignore


@api.dirsize.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@s3_fs_logging_policy()
def dirsize(uri: S3URI) -> int:
    bucket, key = uri.dir_bucket_key()
    return sum([
        obj.size
        for obj in threadsafe_boto.s3_resource().Bucket(bucket).objects.filter(
            Prefix=key)
    ])


@api.exists.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@s3_fs_logging_policy()
def exists(uri: S3URI) -> bool:
    try:
        if isdir(uri):
            return True
        else:
            # Bucket may or may not exist.
            # - Bucket exists and key is "", meaning the bucket is empty.
            # - Bucket does not exist and key is "". This case occurs when
            # the list_objects_v2 response returns AccessDenied instead of
            # NoSuchBucket in the isdir call.
            if not uri.path:
                return False

            _read_with_unsigned_fallback(lambda client: client.head_object(
                Bucket=uri.bucket, Key=uri.path))
            return True
    except ClientError as e:
        code = _get_error_code(e)
        if code in ['NoSuchKey', 'NoSuchBucket', '404']:
            logger.debug(f"No such key or bucket: {code}")
            return False
        # A permission or transient failure is not proof that the object is
        # absent. Let callers decide whether and how to retry it.
        raise


def _get_error_code(e: ClientError) -> Optional[str]:
    error = e.response.get("Error")
    return error.get("Code") if error else None


@api.isdir.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@s3_fs_logging_policy()
def isdir(uri: S3URI) -> bool:
    prefix = os.path.join(uri.path, '')

    def list_objects(client):
        return client.list_objects_v2(
            Bucket=uri.bucket,
            Prefix=prefix,
            MaxKeys=2,
        )

    try:
        res = _read_with_unsigned_fallback(list_objects)
    except ClientError as e:
        code = _get_error_code(e)
        if code == 'AccessDenied':
            return False
        else:
            raise e
    # Bucket with no path
    if not prefix:
        return True

    if res["KeyCount"] == 0:
        return False

    for obj in res["Contents"]:
        # a directory has a key other than itself
        # could be itself with a trailing slash
        key = obj.get("Key")
        if key and key.startswith(prefix):
            return True
    return False


@api._get_if_present.register
@trace(TraceOptions(trace_params=True))
@s3_fs_logging_policy()
def get_if_present(uri: S3URI) -> Optional[bytes]:
    bucket, key = uri.file_bucket_key()

    _get_if_present_with_retry = executeWithRetry(
        max_num_retries=2,
        delay_s=s3_ctx._retry_delay_sec,
    )(_get_if_present)

    return _get_if_present_with_retry(bucket, key)


def _get_if_present(bucket: str, key: str) -> Optional[bytes]:
    try:
        obj = s3_ctx._s3_client.get_object(Bucket=bucket, Key=key)
    except s3_ctx._s3_client.exceptions.NoSuchKey:
        return None
    return obj['Body'].read()


def _listdir_details(
    uri: S3URI,
    max_items: Optional[int] = None,
) -> list[api.PathDetail]:
    # max_items should be in the range of 1 <= max_items <= 1000.
    # This argument is useful for limiting the number of items returned
    # to avoid excess latency and memory usage if not all items are needed.
    if max_items is not None and not (1 <= max_items <= 1000):
        raise ValueError(
            "max_items should be in the range of 1 <= max_items <= 1000"
            "for s3 fsutil.listdir")

    # 'Contents' contains all the files in the directory.
    # 'CommonPrefixes' contains all the subdirectories in the directory.
    # The '/' at the end of a directory is omitted.
    # The if obj['Key'] and not obj['Key'].endswith('/') check is needed to
    # handle a tricky edge case. If a user creates a directory in the aws
    # console, a zero-byte file is created. This makes the list_objects_v2 api
    # think the directory is a file.
    bucket, key = uri.dir_bucket_key()

    def _listdir_with_client(client) -> list[api.PathDetail]:
        entries: list[api.PathDetail] = []

        if max_items is not None:
            response = client.list_objects_v2(Bucket=bucket, Prefix=key,
                                              Delimiter='/', MaxKeys=max_items)

            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'] != key and not obj['Key'].endswith('/'):
                        entries.append(
                            api.PathDetail(
                                path=S3URI(
                                    bucket=bucket,
                                    path=obj['Key'],
                                ).as_s3(),
                                is_directory=False,
                            ))
                    if len(entries) == max_items:
                        break

            if 'CommonPrefixes' in response and len(entries) < max_items:
                for obj in response['CommonPrefixes']:
                    if obj['Prefix'] != key:
                        entries.append(
                            api.PathDetail(
                                path=S3URI(
                                    bucket=bucket,
                                    path=obj['Prefix'][:-1],
                                ).as_s3(),
                                is_directory=True,
                            ))
                    if len(entries) == max_items:
                        break

            return entries

        paginator = client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=key,
                                       Delimiter='/'):

            if 'Contents' in page:
                entries.extend([
                    api.PathDetail(
                        path=S3URI(bucket=bucket, path=obj['Key']).as_s3(),
                        is_directory=False,
                    ) for obj in page['Contents']
                    if obj['Key'] != key and not obj['Key'].endswith('/')
                ])

            if 'CommonPrefixes' in page:
                entries.extend([
                    api.PathDetail(
                        path=S3URI(
                            bucket=bucket,
                            path=obj['Prefix'][:-1],
                        ).as_s3(),
                        is_directory=True,
                    ) for obj in page['CommonPrefixes'] if obj['Prefix'] != key
                ])

        return entries

    return _read_with_unsigned_fallback(_listdir_with_client)


@api._listdir.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@s3_fs_logging_policy()
def listdir(
    uri: S3URI,
    max_items: Optional[int] = None,
) -> list[str]:
    return [entry.path for entry in _listdir_details(uri, max_items=max_items)]


@api._listdir_detail.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@s3_fs_logging_policy()
def listdir_detail(
    uri: S3URI,
    max_items: Optional[int] = None,
) -> list[api.PathDetail]:
    return _listdir_details(uri, max_items=max_items)


@api.makedirs.register
@fsExceptionHandler
@s3_fs_logging_policy()
def makedirs(uri: S3URI, exist_ok: bool = False) -> None:
    # Making directories on S3 is an explicit no-op
    pass


@api.rm.register
@trace(TraceOptions(trace_params=['path']))
@s3_fs_logging_policy()
def rm(uri: S3URI, ignore_errors: bool = False,
       include: Optional[List[str]] = None) -> None:
    try:
        if isdir(uri):
            rmdir(uri.as_s3(), include=include)
        elif exists(uri) and should_include(uri.as_s3(), include):
            s3_ctx._s3_client.delete_object(Bucket=uri.bucket, Key=uri.path)
    except Exception:
        if not ignore_errors:
            raise


@api.touch.register
@trace(TraceOptions(trace_params=True))
@s3_fs_logging_policy()
def touch(uri: S3URI) -> None:
    """Path must be a file (no trailing slash).
    This API updates the last modified time if the file exists.
    """
    s3_ctx._s3_client.put_object(
        Body=b'',
        Bucket=uri.bucket,
        Key=uri.path,
        **bucket_owner_write_args(),
    )


@api.has_write_permission.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@s3_fs_logging_policy()
def has_write_permission(uri: S3URI) -> bool:
    base = uri.path.rstrip('/')
    parent = base if uri.path.endswith('/') else posixpath.dirname(base)
    probe_path = posixpath.join(parent, f'.kumo_write_probe_{uuid.uuid4().hex}')
    bucket = uri.bucket
    upload_id = None

    try:
        response = s3_ctx._s3_client.create_multipart_upload(
            Bucket=bucket,
            Key=probe_path,
            **bucket_owner_write_args(),
        )
        upload_id = response['UploadId']
    except ClientError as e:
        if e.response.get('Error', {}).get('Code') == "AccessDenied":
            return False
        logger.warning('S3 output write probe failed (%s)', type(e).__name__)
        return False
    except _NO_CREDENTIALS_EXCEPTIONS:
        return False
    except Exception as e:
        logger.warning('S3 output write probe failed (%s)', type(e).__name__)
        return False
    finally:
        if upload_id is not None:
            try:
                s3_ctx._s3_client.abort_multipart_upload(
                    Bucket=bucket,
                    Key=probe_path,
                    UploadId=upload_id,
                )
            except Exception as e:
                logger.warning(
                    'S3 output write probe cleanup failed (%s)',
                    type(e).__name__,
                )
                upload_id = None

    return upload_id is not None


@api._contains_subdir.register
@trace(TraceOptions(trace_params=True))
@s3_fs_logging_policy()
def contains_subdir(uri: S3URI) -> bool:
    if not isdir(uri):
        return False
    bucket, prefix = uri.dir_bucket_key()
    assert prefix.endswith('/')

    def _contains_subdir_with_client(client) -> bool:
        paginator = client.get_paginator('list_objects_v2')
        for resp in paginator.paginate(Bucket=bucket, Prefix=prefix,
                                       Delimiter='/'):
            if len(resp.get('CommonPrefixes', [])) > 0:
                return True
        return False

    return _read_with_unsigned_fallback(_contains_subdir_with_client)


@api.get_last_modified_time.register
@trace(TraceOptions(trace_params=['path']))
@s3_fs_logging_policy()
def get_last_modified_time(uri: S3URI, suffix: Optional[str] = None) -> float:
    r"""Return the last modified time of a s3 path. If the path is a folder
    on s3, return the latest modified file inside the folder.

    Args:
        path (str): The s3 path.
        suffix (Optional[str], optional): The suffix of files to scan.
            Defaults to :obj:`None`.
        default_version (float, optional): The default version to return
            if no data is available.. Defaults to :obj:`-1`.
    """
    if not exists(uri):
        return DEFAULT_DATA_VERSION

    try:
        if isdir(uri):
            bucket, key = uri.dir_bucket_key()
            res = _read_with_unsigned_fallback(
                lambda client: client.list_objects_v2(Bucket=bucket, Prefix=key,
                                                      Delimiter='/'))
            return max(
                file['LastModified'] for file in res.get('Contents', [])
                if not suffix or file['Key'].endswith(suffix)).timestamp()
    except Exception as e:
        logger.warning(
            "Unable to get S3 last modified time for %s; returning %s "
            "(error_type=%s)", uri.as_s3(), DEFAULT_DATA_VERSION,
            type(e).__name__)
        return DEFAULT_DATA_VERSION

    bucket, key = uri.file_bucket_key()
    try:
        obj = _read_with_unsigned_fallback(
            lambda client: client.head_object(Bucket=bucket, Key=key))
        return obj['LastModified'].timestamp()
    except s3_ctx._s3_client.exceptions.ClientError as e:
        logger.error(
            "Unable to get S3 last modified time for %s (error_type=%s)",
            uri.as_s3(),
            type(e).__name__)
        return DEFAULT_DATA_VERSION


# ---------------------------------------------------------------------------
# PresignedHelper / zip_directory
# ---------------------------------------------------------------------------


class S3PresignedHelper(api.PresignedHelper[S3URI]):
    """S3 implementation of :class:`~api.PresignedHelper`."""
    def __init__(self, uri: S3URI):
        super().__init__(uri)

    @property
    def supports_multipart_upload(self) -> bool:
        return True

    def _generate_download_url(self, uri: S3URI) -> str:
        from io_utils.fsutil.s3_filesystem import \
            generate_presigned_url_for_download  # noqa: E501
        return generate_presigned_url_for_download(uri.as_s3())

    def _create_multipart_upload(self, uri: S3URI, num_parts):
        bucket, key = uri.file_bucket_key()
        select_s3_region(uri.as_s3())
        s3_client = threadsafe_boto.s3_client()

        s3_path = uri.as_s3()
        resp = s3_client.create_multipart_upload(Bucket=bucket, Key=key)
        if resp['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise ValueError(
                f"Failed to create multipart upload for {s3_path}. "
                f"Response: {resp}")

        upload_id = resp['UploadId']
        presigned_urls = {}
        for part_number in range(1, num_parts + 1):
            presigned_urls[part_number] = s3_client.generate_presigned_url(
                ClientMethod='upload_part',
                Params={
                    'Bucket': bucket,
                    'Key': key,
                    'UploadId': upload_id,
                    'PartNumber': part_number,
                },
                HttpMethod='PUT',
                ExpiresIn=3600,
            )
        return upload_id, presigned_urls

    def _complete_multipart_upload(self, uri: S3URI, upload_id, parts):
        s3_client = threadsafe_boto.s3_client()
        bucket, key = uri.file_bucket_key()

        s3_path = uri.as_s3()
        listed_part_count = sum(
            len(page.get('Parts', []))
            for page in s3_client.get_paginator('list_parts').paginate(
                Bucket=bucket, Key=key, UploadId=upload_id))
        if listed_part_count != len(parts):
            raise ValueError(f"Number of parts returned by list_parts: "
                             f"{listed_part_count} does not match parts "
                             f"metadata: {len(parts)}.")

        logger.debug(f"No.of parts matched for {s3_path}: {len(parts)}, "
                     f"completing multipart upload with upload_id: {upload_id}")
        complete_resp = s3_client.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={
                'Parts': [{
                    'ETag': p.etag,
                    'PartNumber': p.part_number
                } for p in parts]
            },
        )
        if complete_resp['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise ValueError(
                f"Failed to complete multipart upload for {s3_path}. "
                f"Response: {complete_resp}")
        logger.info(f"Completed multipart upload for {s3_path}, "
                    f"upload_id: {upload_id}")


@api.get_presigned_helper.register
def get_presigned_helper(uri: S3URI) -> api.PresignedHelper:
    return S3PresignedHelper(uri)


@api._zip_directory.register
def zip_directory(
    uri: S3URI,
    recursive: bool = False,
    output_path: Optional[str] = None,
) -> str:
    return zip_files_with_same_prefix(uri.as_s3(), output_path=output_path)


@api._zip_files.register
def zip_files(
    uri: S3URI,
    paths: Sequence[AnyURI],
    output_path: str,
) -> str:
    selected = []
    for path in paths:
        if not isinstance(path, S3URI) or path.bucket != uri.bucket:
            raise ValueError('S3 archive inputs must use one bucket')
        selected.append(path.path)
    return zip_files_with_same_prefix(uri.as_s3(), output_path=output_path,
                                      object_keys=selected)


@api.customer_facing_path.register
def customer_facing_path(uri: S3URI) -> str:
    return uri.as_s3()
