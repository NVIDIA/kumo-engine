# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Google Cloud Storage (GCS) implementation for fsutil.

This module provides GCS backend implementation for fsutil operations
using the google-cloud-storage client library.

Safe/idempotent reads and listings use a bounded retry policy for transient
Google API and transport failures. Authentication, malformed-request, and
not-found failures remain single-attempt.
"""
import datetime
import gzip
import logging
import os
import posixpath
import tempfile
import threading
import uuid
from typing import BinaryIO, Generator, List, Optional, Sequence, TextIO
from zipfile import ZipFile

from google.api_core import exceptions as gcs_exceptions
from google.auth import compute_engine
from google.auth import default as auth_default
from google.auth import impersonated_credentials
from google.auth.exceptions import GoogleAuthError
from google.auth.transport import requests
from google.cloud import storage
from kumo_gcs.download import GCSDownloader
from smart_open import open as smart_open

from io_utils.errors import restricted_diagnostic_boundary
from io_utils.fsutil import api
from io_utils.fsutil.common import (
    DEFAULT_DATA_VERSION,
    fsExceptionHandler,
    fsutil_ctx,
    should_include,
)
from io_utils.fsutil.credentials import (
    credentials_for_provider,
    scoped_cache_key,
)
from io_utils.fsutil.logging_policy import logging_policy
from io_utils.fsutil.uri import GCSURI, AnyURI, LocalURI, ParseURI
from io_utils.retry import retry
from monitoring.tracing import TraceOptions as _TraceOptions
from monitoring.tracing import trace

logger = logging.getLogger(__name__)


def TraceOptions(**kwargs) -> _TraceOptions:
    """Build spans that never export raw Google provider exceptions."""
    return _TraceOptions(record_exceptions=False, **kwargs)


gcs_fs_logging_policy = logging_policy(logger, restricted_failures=True)

_GCS_TRANSIENT_ERRORS = (
    gcs_exceptions.TooManyRequests,
    gcs_exceptions.InternalServerError,
    gcs_exceptions.BadGateway,
    gcs_exceptions.ServiceUnavailable,
    gcs_exceptions.GatewayTimeout,
    gcs_exceptions.DeadlineExceeded,
    ConnectionError,
    TimeoutError,
)


def is_retryable_gcs_error(error: BaseException) -> bool:
    return isinstance(error, _GCS_TRANSIENT_ERRORS)


def gcs_retry_policy():
    """Bounded retry for operations explicitly safe to repeat."""
    return retry(exceptions=_GCS_TRANSIENT_ERRORS, tries=3, delay=0.5,
                 backoff=2, max_delay=2, logger=None)


class ThreadLocalGCSClient(threading.local):
    """Thread-safe GCS client wrapper.

    The google-cloud-storage client is not thread-safe due to underlying
    httplib2 dependencies. This wrapper ensures each thread gets its own
    client instance.

    Reference: https://github.com/googleapis/google-cloud-python/issues/5685
    """
    def __init__(self) -> None:
        super().__init__()
        self._client = None

    def client(self) -> storage.Client:
        """Get or create a GCS client for the current thread.

        The client uses Application Default Credentials (ADC), which
        automatically discovers credentials from:
        1. GOOGLE_APPLICATION_CREDENTIALS env var (service account JSON)
        2. gcloud auth application-default login (user credentials)
        3. Workload Identity (for GKE)
        4. Compute Engine metadata service
        """
        scope = credentials_for_provider('gcs')
        if scope is not None:
            key = scoped_cache_key('gcs', 'client')
            client = scope.cache.get(key)
            if client is None:
                credentials = scope.credentials or {}
                credential = credentials.get('credential')
                if credential is None:
                    raise RuntimeError(
                        'Missing federated GCS credential for GCS identity')
                client = storage.Client(
                    project=credentials.get('project'),
                    credentials=credential,
                )
                scope.cache[key] = client
            return client
        if self._client is None:
            # Let google-cloud-storage handle credential discovery
            self._client = storage.Client()
        return self._client

    def _reset(self) -> None:
        """Reset the client (useful for testing)."""
        self._client = None


# Global thread-safe client instance
threadsafe_gcs = ThreadLocalGCSClient()


def _get_blob(uri: GCSURI) -> storage.Blob:
    """Helper to get a Blob object from a GCSURI."""
    return _get_bucket(uri).blob(uri.path)


def _get_bucket(uri: GCSURI) -> storage.Bucket:
    """Helper to get a Bucket object from a GCSURI."""
    client = threadsafe_gcs.client()
    return _client_bucket(client, uri.bucket)


def _client_bucket(client: storage.Client, bucket_name: str) -> storage.Bucket:
    scope = credentials_for_provider('gcs')
    requester_pays_project = None
    if scope is not None and scope.credentials is not None:
        requester_pays_project = scope.credentials.get('requester_pays_project')
    return client.bucket(bucket_name, user_project=requester_pays_project)


class _ScopedGCSClient:
    """Smart-open client adapter preserving requester-pays bucket shape."""
    def __init__(self, client: storage.Client):
        self._client = client
        self._credentials = client._credentials

    def bucket(self, bucket_name: str):
        return _client_bucket(self._client, bucket_name)

    def __getattr__(self, name):
        return getattr(self._client, name)


def _smart_open_client() -> storage.Client:
    return _ScopedGCSClient(threadsafe_gcs.client())  # type: ignore


@api.read_binary.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@gcs_fs_logging_policy(allowed_exceptions=FileNotFoundError)
@gcs_retry_policy()
def read_binary(uri: GCSURI, compression: bool) -> bytes:
    """Read binary data from a GCS object."""
    try:
        blob = _get_blob(uri)

        # Get size before downloading
        blob.reload()
        expected_size = blob.size

        file_bytes = blob.download_as_bytes()

        # Verify download completeness
        actual_size = len(file_bytes)
        if expected_size != actual_size:
            raise IOError(
                f"File {uri} might be incomplete! Expected {expected_size} bytes, "
                f"but got {actual_size} bytes.")

        if compression:
            return gzip.decompress(file_bytes)
        else:
            return file_bytes
    except gcs_exceptions.NotFound:
        raise FileNotFoundError(uri.as_gs())


@api._write_binary.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=['uri', 'compression']))
@gcs_fs_logging_policy(skip_args=[1])
def write_binary(uri: GCSURI, data: BinaryIO, compression: bool,
                 ttl: Optional[object] = None) -> None:
    """Write binary data to a GCS object.

    Note: ttl parameter is ignored for GCS (not supported by this impl.).
    """
    blob = _get_blob(uri)

    if compression:
        compressed_data = gzip.compress(data.read())
        blob.upload_from_string(compressed_data)
    else:
        blob.upload_from_file(data)


@api.read_text.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@gcs_fs_logging_policy(allowed_exceptions=FileNotFoundError)
def read_text(uri: GCSURI, compression: bool,
              encoding: Optional[str] = None) -> str:
    """Read text data from a GCS object."""
    r = read_binary(uri, compression=compression)
    if encoding:
        return r.decode(encoding)
    return r.decode()


@api.write_text.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=['uri', 'compression']))
@gcs_fs_logging_policy()
def write_text(uri: GCSURI, data: str, compression: bool = False,
               encoding: Optional[str] = None) -> None:
    """Write text data to a GCS object."""
    if encoding:
        encoded_data = data.encode(encoding)
    else:
        encoded_data = data.encode()

    blob = _get_blob(uri)

    if compression:
        compressed_data = gzip.compress(encoded_data)
        blob.upload_from_string(compressed_data)
    else:
        blob.upload_from_string(encoded_data)


@api.write_text_if_absent.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=['uri']))
@gcs_fs_logging_policy(skip_args=[1])
def write_text_if_absent(uri: GCSURI, data: str) -> bool:
    try:
        _get_blob(uri).upload_from_string(data.encode(), if_generation_match=0)
    except (gcs_exceptions.PreconditionFailed, gcs_exceptions.Conflict):
        return False
    return True


@api._get_binary_handle.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
def get_binary_handle(
    uri: GCSURI,
    mode: str,
    compression: bool = False,
) -> Generator[BinaryIO, None, None]:
    """Get a binary file handle for reading/writing GCS objects."""
    with restricted_diagnostic_boundary(
            safe_message='GCS stream operation failed',
            exception_types=(gcs_exceptions.GoogleAPIError, GoogleAuthError,
                             OSError)):
        if 'b' not in mode:
            mode = mode + 'b'
        with smart_open(uri.as_gs(), mode, encoding=None,
                        compression=".gz" if compression else None,
                        transport_params={'client': _smart_open_client()
                                          }) as f:  # type: ignore
            yield f  # type: ignore


@api._get_handle.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
def get_handle(
    uri: GCSURI,
    mode: str,
    compression: bool = False,
) -> Generator[TextIO, None, None]:
    """Get a text file handle for reading/writing GCS objects."""
    with restricted_diagnostic_boundary(
            safe_message='GCS stream operation failed',
            exception_types=(gcs_exceptions.GoogleAPIError, GoogleAuthError,
                             OSError)):
        with smart_open(uri.as_gs(), mode, encoding='utf-8',
                        compression=".gz" if compression else None,
                        transport_params={'client': _smart_open_client()
                                          }) as f:  # type: ignore
            yield f  # type: ignore


@api.dirsize.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@gcs_fs_logging_policy()
@gcs_retry_policy()
def dirsize(uri: GCSURI) -> int:
    """Calculate total size of all objects under a GCS prefix (directory)."""
    bucket = _get_bucket(uri)
    prefix = uri.path
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    total_size = 0
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        total_size += blob.size

    return total_size


@api.exists.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@gcs_fs_logging_policy()
@gcs_retry_policy()
def exists(uri: GCSURI) -> bool:
    """Check if a GCS object or prefix exists."""
    try:
        # First check if it's a directory (prefix with objects)
        if _isdir_once(uri):
            return True

        # If path is empty, it doesn't exist.
        if not uri.path:
            return False

        # Check if the specific blob exists
        blob = _get_blob(uri)
        return blob.exists()
    except gcs_exceptions.Forbidden:
        # Access denied - treat as not existing
        logger.debug(f"Access denied when checking existence: {uri.as_gs()}")
        return False
    except Exception as e:
        if is_retryable_gcs_error(e):
            raise
        logger.warning(
            "Unexpected exception checking GCS existence "
            "(error_type=%s)",
            type(e).__name__)
        return False


def _isdir_once(uri: GCSURI) -> bool:
    try:
        # Empty path means bucket root, which is always a directory
        if not uri.path:
            return True

        # Check if there are any objects with this prefix
        bucket = _get_bucket(uri)
        prefix = uri.path
        if not prefix.endswith('/'):
            prefix += '/'

        # List with max 1 result - if we get any, it's a directory
        blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
        return len(blobs) > 0
    except gcs_exceptions.Forbidden:
        return False
    except Exception as e:
        if is_retryable_gcs_error(e):
            raise
        logger.warning("Unexpected exception in GCS isdir (error_type=%s)",
                       type(e).__name__)
        return False


@api.isdir.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@gcs_fs_logging_policy()
@gcs_retry_policy()
def isdir(uri: GCSURI) -> bool:
    """Check if a GCS path represents a directory (prefix with objects)."""
    return _isdir_once(uri)


def _listdir_details(
    uri: GCSURI,
    max_items: Optional[int] = None,
) -> list[api.PathDetail]:
    entries: list[api.PathDetail] = []

    if max_items is not None and not (1 <= max_items <= 1000):
        raise ValueError(
            "max_items should be in the range of 1 <= max_items <= 1000 "
            "for GCS fsutil.listdir")

    bucket_name = uri.bucket
    bucket = _get_bucket(uri)
    prefix = uri.path
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    # Use delimiter='/' to get only direct children (not recursive)
    # When max_items is None, don't pass max_results to get all results
    iterator = bucket.list_blobs(prefix=prefix, delimiter='/',
                                 max_results=max_items if max_items else None)

    # Iterate through all pages if max_items is None, or until we reach max_items
    for page in iterator.pages:
        # Add files (blobs)
        for blob in page:
            # Skip the prefix itself if it exists as a blob
            if blob.name != prefix and not blob.name.endswith('/'):
                entries.append(
                    api.PathDetail(
                        path=f'gs://{bucket_name}/{blob.name}',
                        is_directory=False,
                    ))
                if max_items and len(entries) >= max_items:
                    return entries

        # Add directories (prefixes) from this page
        if hasattr(page, 'prefixes') and page.prefixes:
            for prefix_path in page.prefixes:
                # Remove trailing slash for consistency
                dir_path = prefix_path.rstrip('/')
                if dir_path:
                    entries.append(
                        api.PathDetail(
                            path=f'gs://{bucket_name}/{dir_path}',
                            is_directory=True,
                        ))
                    if max_items and len(entries) >= max_items:
                        return entries

        # Check if we've reached max_items after processing blobs
        if max_items and len(entries) >= max_items:
            break

    return entries


@api._listdir.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@gcs_fs_logging_policy()
@gcs_retry_policy()
def listdir(
    uri: GCSURI,
    max_items: Optional[int] = None,
) -> list[str]:
    return [entry.path for entry in _listdir_details(uri, max_items=max_items)]


@api._listdir_detail.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@gcs_fs_logging_policy()
@gcs_retry_policy()
def listdir_detail(
    uri: GCSURI,
    max_items: Optional[int] = None,
) -> list[api.PathDetail]:
    return _listdir_details(uri, max_items=max_items)


@api.rm.register
@trace(TraceOptions(trace_params=['path']))
@gcs_fs_logging_policy()
def rm(uri: GCSURI, ignore_errors: bool = False,
       include: Optional[List[str]] = None) -> None:
    """Remove a GCS object or directory (prefix)."""
    try:
        if isdir(uri):
            # Remove all objects with this prefix
            _rmdir(uri, include=include)
        elif exists(uri) and should_include(uri.as_gs(), include):
            # Remove single blob
            blob = _get_blob(uri)
            blob.delete()
    except Exception as e:
        if not ignore_errors:
            raise
        else:
            logger.debug("Ignored GCS rm error (error_type=%s)",
                         type(e).__name__)


def _rmdir(uri: GCSURI, include: Optional[List[str]] = None) -> None:
    """Remove all objects under a GCS prefix (directory)."""
    bucket = _get_bucket(uri)
    prefix = uri.path
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    # List all blobs with this prefix
    blobs = bucket.list_blobs(prefix=prefix)

    # TODO(GCS): Implement batch delete for efficiency.
    # Currently deletes one blob at a time sequentially.
    # The S3 implementation uses delete_objects() with up to 999 keys per batch
    # and parallel execution via thread pools.
    # GCS supports batch requests via google-cloud-storage's batch operations:
    # https://cloud.google.com/storage/docs/batch
    # Consider using bucket.delete_blobs() or implementing batch delete with
    # threading for large directories.
    for blob in blobs:
        blob_path = f'gs://{uri.bucket}/{blob.name}'
        if should_include(blob_path, include):
            blob.delete()


@api.makedirs.register
@fsExceptionHandler
@gcs_fs_logging_policy()
def makedirs(uri: GCSURI, exist_ok: bool = False) -> None:
    """Create a directory (no-op for GCS as directories are virtual)."""
    # GCS doesn't have real directories, they're just prefixes
    # So this is a no-op, similar to S3


@api.filesize.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@gcs_fs_logging_policy()
@gcs_retry_policy()
def filesize(uri: GCSURI) -> Optional[int]:
    """Get the size of a GCS object in bytes."""
    try:
        # Empty path means bucket, not a file
        if not uri.path:
            return None

        blob = _get_blob(uri)
        if not blob.exists():
            return None

        # Reload to fetch metadata including size
        blob.reload()
        return blob.size
    except gcs_exceptions.NotFound:
        return None
    except Exception as e:
        if is_retryable_gcs_error(e):
            raise
        logger.warning(
            "Unexpected exception in GCS filesize "
            "(error_type=%s)",
            type(e).__name__)
        return None


@api.object_identity.register
@trace(TraceOptions(trace_params=True))
@gcs_fs_logging_policy(allowed_exceptions=FileNotFoundError)
@gcs_retry_policy()
def object_identity(uri: GCSURI) -> api.ObjectIdentity:
    """Capture the current GCS generation and ETag for conditional reads."""
    if not uri.path:
        raise FileNotFoundError(uri.as_gs())
    blob = _get_blob(uri)
    try:
        blob.reload()
    except gcs_exceptions.NotFound:
        raise FileNotFoundError(uri.as_gs()) from None
    if blob.size is None or blob.generation is None:
        raise RuntimeError('GCS object metadata is incomplete')
    etag = blob.etag if isinstance(blob.etag, str) and blob.etag else None
    return api.ObjectIdentity(
        size=int(blob.size),
        etag=etag,
        version=str(blob.generation),
    )


@api.touch.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@gcs_fs_logging_policy()
def touch(uri: GCSURI) -> None:
    """Create an empty file or update modification time of existing file."""
    blob = _get_blob(uri)
    # Upload empty content - creates new or overwrites existing
    blob.upload_from_string(b'')


@api._get_if_present.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@gcs_fs_logging_policy()
@gcs_retry_policy()
def get_if_present(uri: GCSURI) -> Optional[bytes]:
    """Read file contents if it exists, return None otherwise."""
    try:
        blob = _get_blob(uri)
        if not blob.exists():
            return None
        return blob.download_as_bytes()
    except gcs_exceptions.NotFound:
        return None
    except Exception as e:
        if is_retryable_gcs_error(e):
            raise
        logger.warning(
            "Unexpected exception in GCS get_if_present "
            "(error_type=%s)",
            type(e).__name__)
        return None


@api.has_write_permission.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@gcs_fs_logging_policy()
def has_write_permission(uri: GCSURI) -> bool:
    """Create and remove a zero-byte probe at the selected GCS prefix."""
    base = uri.path.rstrip('/')
    parent = base if uri.path.endswith('/') else posixpath.dirname(base)
    probe_path = posixpath.join(parent, f'.kumo_write_probe_{uuid.uuid4().hex}')
    blob = None
    uploaded = False
    cleanup_ok = True
    try:
        blob = _get_bucket(uri).blob(probe_path)
        blob.upload_from_string(b'')
        uploaded = True
    except gcs_exceptions.NotFound:
        logger.warning('GCS output write probe target was not found')
    except (gcs_exceptions.Forbidden, gcs_exceptions.Unauthorized):
        logger.warning('GCS output write probe was denied')
    except Exception as error:
        logger.warning('GCS output write probe failed (%s)',
                       type(error).__name__)
    finally:
        if uploaded:
            assert blob is not None
            try:
                blob.delete()
            except gcs_exceptions.NotFound:
                pass
            except Exception as error:
                cleanup_ok = False
                logger.warning(
                    'GCS output write probe cleanup failed (%s)',
                    type(error).__name__,
                )
    return uploaded and cleanup_ok


@api._contains_subdir.register
@trace(TraceOptions(trace_params=True))
@gcs_fs_logging_policy()
@gcs_retry_policy()
def contains_subdir(uri: GCSURI) -> bool:
    """Check if a GCS path contains subdirectories."""
    bucket = _get_bucket(uri)
    prefix = uri.path
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    elif not prefix:
        prefix = ''
    # List blobs with delimiter - this groups by "subdirectories"
    iterator = bucket.list_blobs(prefix=prefix, delimiter='/')

    # iterate pages till you find at least one with prefixes
    # this may not be the first page or even the 100th.
    for page in iterator.pages:
        if page.prefixes:
            return True
    return False


@api.get_last_modified_time.register
@trace(TraceOptions(trace_params=['path']))
@gcs_fs_logging_policy()
def get_last_modified_time(uri: GCSURI, suffix: Optional[str] = None) -> float:
    """Return the last modified time of a GCS path.

    If the path is a directory, return the latest modified file inside it.

    Args:
        uri: The GCS URI
        suffix: Optional suffix filter for files to scan

    Returns:
        Timestamp of last modification, or DEFAULT_DATA_VERSION if unavailable
    """
    if not exists(uri):
        return DEFAULT_DATA_VERSION

    try:
        if isdir(uri):
            bucket = _get_bucket(uri)
            prefix = uri.path
            if prefix and not prefix.endswith('/'):
                prefix += '/'

            blobs = bucket.list_blobs(prefix=prefix, delimiter='/')

            # Filter by suffix if provided and find max updated time
            latest_time = None
            for blob in blobs:
                if suffix and not blob.name.endswith(suffix):
                    continue
                if blob.updated:
                    if latest_time is None or blob.updated > latest_time:
                        latest_time = blob.updated

            if latest_time:
                return latest_time.timestamp()
            else:
                return DEFAULT_DATA_VERSION

    except Exception as e:
        logger.warning(
            "Unable to get GCS last modified time for %s; returning %s "
            "(error_type=%s)", uri.as_gs(), DEFAULT_DATA_VERSION,
            type(e).__name__)
        return DEFAULT_DATA_VERSION

    # For files
    try:
        blob = _get_blob(uri)
        blob.reload()
        if blob.updated:
            return blob.updated.timestamp()
        return DEFAULT_DATA_VERSION
    except gcs_exceptions.NotFound as e:
        logger.error(
            "Unable to get GCS last modified time for %s; file not found "
            "(error_type=%s)", uri.as_gs(),
            type(e).__name__)
        return DEFAULT_DATA_VERSION
    except Exception as e:
        logger.error(
            "Unable to get GCS last modified time for %s (error_type=%s)",
            uri.as_gs(),
            type(e).__name__)
        return DEFAULT_DATA_VERSION


# Default signed URL expiration time (1 hour, matching S3 presigned URL default)
DEFAULT_SIGNED_URL_EXPIRATION = datetime.timedelta(hours=1)

# Give multipart uploads enough time for a 5 GiB transfer over slower links.
GCS_UPLOAD_URL_EXPIRATION = datetime.timedelta(hours=1)


def _get_signing_credentials(scopes=None):
    """Return credentials suitable for signing GCS URLs.

    The signing strategy depends on the credential type from ADC:

    1. **SA key file** (``sign_bytes`` available): Returns the credentials
       directly — they can sign locally.
    2. **Workload Identity / GCE**: Self-impersonation via IAM ``signBlob``.
    3. **User credentials** (``gcloud auth``): Impersonates the project's
       default compute SA via ``signBlob``.
    """
    if scopes is None:
        scopes = ['https://www.googleapis.com/auth/devstorage.read_only']

    credentials, _ = auth_default(scopes=scopes)

    if hasattr(credentials, 'sign_bytes'):
        return credentials

    if isinstance(credentials, compute_engine.Credentials):
        credentials.refresh(requests.Request())
        return impersonated_credentials.Credentials(
            source_credentials=credentials,
            target_principal=credentials.service_account_email,
            target_scopes=scopes,
        )

    # Workload Identity Federation credentials expose the impersonated
    # service-account email but cannot sign locally.  Ask IAM Credentials to
    # sign as that same account, just as the GCE path does above.
    service_account_email = getattr(credentials, 'service_account_email', None)
    if service_account_email:
        credentials.refresh(requests.Request())
        return impersonated_credentials.Credentials(
            source_credentials=credentials,
            target_principal=service_account_email,
            target_scopes=scopes,
        )

    # User credentials — impersonate default compute SA.
    credentials.refresh(requests.Request())
    client = threadsafe_gcs.client()
    service_account_email = f"{client.project}@appspot.gserviceaccount.com"
    return impersonated_credentials.Credentials(
        source_credentials=credentials,
        target_principal=service_account_email,
        target_scopes=scopes,
    )


# ---------------------------------------------------------------------------
# PresignedHelper / zip_directory
# ---------------------------------------------------------------------------


def _get_gcs_part_path(base_path: str, part_number: int) -> str:
    """Generate the path for a temporary part object."""
    return f"{base_path}.part{part_number:05d}"


class GCSPresignedHelper(api.PresignedHelper[GCSURI]):
    """GCS implementation of :class:`~api.PresignedHelper`."""
    def __init__(self, uri: GCSURI):
        super().__init__(uri)

    @property
    def supports_multipart_upload(self) -> bool:
        return True

    def _generate_download_url(self, uri: GCSURI) -> str:
        """Generate a V4 signed URL for downloading a GCS object."""
        if isdir(uri):
            raise ValueError(
                f"Unable to generate signed URL for GCS directory: "
                f"{uri.as_gs()}")

        try:
            blob = _get_blob(uri)
            signing_creds = _get_signing_credentials()

            if hasattr(signing_creds, 'sign_bytes') and not isinstance(
                    signing_creds, impersonated_credentials.Credentials):
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=DEFAULT_SIGNED_URL_EXPIRATION,
                    method="GET",
                )
            else:
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=DEFAULT_SIGNED_URL_EXPIRATION,
                    method="GET",
                    credentials=signing_creds,
                )

            return signed_url

        except Exception:
            raise ValueError(
                f"Permission error for generating signed URL for path: "
                f"{uri.as_gs()}.") from None

    def _create_multipart_upload(self, uri: GCSURI, num_parts):
        client = threadsafe_gcs.client()
        bucket = _client_bucket(client, uri.bucket)
        signing_creds = _get_signing_credentials(
            scopes=['https://www.googleapis.com/auth/devstorage.read_write'])

        presigned_urls = {}
        for part_number in range(1, num_parts + 1):
            part_path = _get_gcs_part_path(uri.path, part_number)
            blob = bucket.blob(part_path)
            presigned_urls[part_number] = blob.generate_signed_url(
                version="v4",
                expiration=GCS_UPLOAD_URL_EXPIRATION,
                method="PUT",
                credentials=signing_creds,
            )

        # Use the gs:// path as the "upload_id" for GCS
        upload_id = uri.as_gs()
        logger.info(f"Created GCS composite upload request for {uri.as_gs()} "
                    f"with {num_parts} parts")
        return upload_id, presigned_urls

    def _complete_multipart_upload(self, uri: GCSURI, upload_id,
                                   parts):  # noqa: ARG002
        """Complete a GCS composite upload by composing part objects.

        Unlike S3 multipart upload which uses a server-generated
        ``upload_id`` to track the upload session, GCS composite upload
        tracks parts by path naming convention
        (``{base_path}.part{part_number:05d}``).  The *upload_id*
        parameter is unused here.
        """
        client = threadsafe_gcs.client()
        bucket = _client_bucket(client, uri.bucket)

        part_blobs = []
        for part in sorted(parts, key=lambda p: p.part_number):
            part_path = _get_gcs_part_path(uri.path, part.part_number)
            blob = bucket.blob(part_path)
            if not blob.exists():
                raise ValueError(f"Part {part.part_number} not found at "
                                 f"gs://{uri.bucket}/{part_path}")
            part_blobs.append(blob)

        gs_path = uri.as_gs()
        logger.debug(f"All {len(part_blobs)} parts verified for {gs_path}, "
                     f"composing into final object")

        MAX_COMPOSE_OBJECTS = 32

        if len(part_blobs) <= MAX_COMPOSE_OBJECTS:
            final_blob = bucket.blob(uri.path)
            final_blob.compose(part_blobs)
        else:
            intermediate_blobs = []
            for i in range(0, len(part_blobs), MAX_COMPOSE_OBJECTS):
                batch = part_blobs[i:i + MAX_COMPOSE_OBJECTS]
                intermediate_path = (
                    f"{uri.path}.intermediate{i // MAX_COMPOSE_OBJECTS}")
                intermediate_blob = bucket.blob(intermediate_path)
                intermediate_blob.compose(batch)
                intermediate_blobs.append(intermediate_blob)

            final_blob = bucket.blob(uri.path)
            final_blob.compose(intermediate_blobs)

            for blob in intermediate_blobs:
                blob.delete()

        logger.info(f"Composed {len(part_blobs)} parts into {gs_path}")

        for blob in part_blobs:
            blob.delete()

        logger.info(
            f"Deleted {len(part_blobs)} temporary part objects for {gs_path}")


@api.get_presigned_helper.register
def get_presigned_helper(uri: GCSURI) -> api.PresignedHelper:
    return GCSPresignedHelper(uri)


@api._zip_directory.register
def zip_directory(
    uri: GCSURI,
    recursive: bool = False,
    output_path: Optional[str] = None,
) -> str:
    """Zip all files under a GCS prefix into a single archive.

    Downloads each file, creates a local zip, and uploads the result.
    Uses ``exists()`` check for idempotency — if the zip already exists,
    returns its path immediately.
    """
    output_path = output_path or uri.as_gs().rstrip('/') + '.zip'
    out_uri = ParseURI(output_path)
    if not isinstance(out_uri, GCSURI):
        raise ValueError('GCS ZIP output must be a GCS path')

    # Idempotency: skip if zip already exists
    if exists(ParseURI(output_path)):
        return output_path

    client = threadsafe_gcs.client()
    bucket = _client_bucket(client, uri.bucket)

    prefix = uri.path
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    # List blobs under prefix
    if recursive:
        blobs = list(bucket.list_blobs(prefix=prefix))
    else:
        blobs = list(bucket.list_blobs(prefix=prefix, delimiter='/'))

    # Filter out directory markers
    blobs = [b for b in blobs if not b.name.endswith('/') and b.name != prefix]

    if not blobs:
        raise ValueError(f"No files found under {uri.as_gs()}")

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "output.zip")
        with ZipFile(zip_path, 'w') as zipf:
            for blob in blobs:
                logger.info("zipping gs://%s/%s to %s", uri.bucket, blob.name,
                            output_path)
                data = blob.download_as_bytes()
                zipf.writestr(blob.name, data)

        # Upload the zip
        out_blob = _get_bucket(out_uri).blob(out_uri.path)
        out_blob.upload_from_filename(zip_path)

    return output_path


@api._zip_files.register
def zip_files(
    uri: GCSURI,
    paths: Sequence[AnyURI],
    output_path: str,
) -> str:
    out_uri = ParseURI(output_path)
    if not isinstance(out_uri, GCSURI):
        raise ValueError('GCS ZIP output must be a GCS path')
    if exists(out_uri):
        return output_path

    names = []
    for path in paths:
        if not isinstance(path, GCSURI) or path.bucket != uri.bucket:
            raise ValueError('GCS archive inputs must use one bucket')
        names.append(path.path)
    if len(set(names)) != len(names):
        raise ValueError('Duplicate GCS objects were selected for archive')

    bucket = _client_bucket(threadsafe_gcs.client(), uri.bucket)
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, 'output.zip')
        with ZipFile(zip_path, 'w') as zipf:
            for name in names:
                blob = bucket.blob(name)
                zipf.writestr(name, blob.download_as_bytes())
        _get_bucket(out_uri).blob(out_uri.path).upload_from_filename(zip_path)
    return output_path


# Cap on rewrite_token iterations to avoid unbounded loops; ~5TB fits.
_GCS_REWRITE_MAX_ITERATIONS = 1024


def _server_side_copy(src_uri: GCSURI, dst_uri: GCSURI) -> bool:
    """Server-side GCS-to-GCS copy via ``Blob.rewrite()`` (handles
    cross-region, cross-storage-class, and >5GB objects). Returns
    ``False`` on any failure so callers can fall back to read+write.
    """
    try:
        client = threadsafe_gcs.client()
        src_bucket = _client_bucket(client, src_uri.bucket)
        src_blob = src_bucket.blob(src_uri.path)
        dst_bucket = _client_bucket(client, dst_uri.bucket)
        dst_blob = dst_bucket.blob(dst_uri.path)

        token = None
        for _ in range(_GCS_REWRITE_MAX_ITERATIONS):
            token, _bytes_rewritten, _total_bytes = dst_blob.rewrite(
                src_blob, token=token)
            if token is None:
                return True
        logger.debug("Server-side GCS copy exceeded %d iterations: %s -> %s",
                     _GCS_REWRITE_MAX_ITERATIONS, src_uri.as_gs(),
                     dst_uri.as_gs())
        return False
    except Exception as e:
        logger.debug("Server-side GCS copy failed (error_type=%s)",
                     type(e).__name__)
        return False


_download_lock = threading.Lock()
_ambient_downloader: GCSDownloader | None = None


def _download_client() -> GCSDownloader:
    global _ambient_downloader
    scope = credentials_for_provider('gcs')
    with _download_lock:
        if scope is None:
            if _ambient_downloader is None:
                _ambient_downloader = GCSDownloader()
            return _ambient_downloader
        key = ('gcs', 0, 'downloader')
        downloader = scope.cache.get(key)
        if downloader is None:
            material = scope.credentials or {}
            credential = material.get('credential')
            if credential is None:
                raise RuntimeError(
                    'Missing federated GCS credential for GCS identity')
            downloader = GCSDownloader(
                credentials=credential, project=material.get('project'),
                requester_pays_project=material.get('requester_pays_project'))
            scope.cache[key] = downloader
        return downloader


@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@gcs_fs_logging_policy(allowed_exceptions=FileNotFoundError)
@gcs_retry_policy()
def _download_to_local(src_uri: GCSURI, local_path: str) -> None:
    _download_client().download(src_uri.bucket, src_uri.path, local_path)


@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@gcs_fs_logging_policy()
@gcs_retry_policy()
def _upload_from_local(local_path: str, dst_uri: GCSURI) -> None:
    # Unlike write_binary (which receives a caller-owned, partially-consumed
    # handle on failure and so cannot retry safely), this opens `local_path`
    # itself: a retry re-enters the function and reopens from byte 0, so
    # re-uploading is a same-bytes, last-writer-wins repeat, not a truncation
    # hazard.
    blob = _get_blob(dst_uri)
    with open(local_path, 'rb') as f:
        blob.upload_from_file(f)


def copy_file(src: str, dst: str) -> None:
    """GCS-aware single-file copy.

    GCS<->GCS uses server-side rewrite and GCS<->local streams; anything
    else falls back to read+write through Python memory.
    """
    src_uri = ParseURI(src)
    dst_uri = ParseURI(dst)

    if isinstance(src_uri, GCSURI) and isinstance(dst_uri, GCSURI):
        if _server_side_copy(src_uri, dst_uri):
            return

    if isinstance(src_uri, GCSURI) and isinstance(dst_uri, LocalURI):
        _download_to_local(src_uri, dst_uri.path)
        return

    if isinstance(src_uri, LocalURI) and isinstance(dst_uri, GCSURI):
        _upload_from_local(src_uri.path, dst_uri)
        return

    data = api.read_binary(src)
    api.write_binary(dst, data)


def copydir(src: str, dst: str) -> None:
    """Server-side GCS-to-GCS directory copy via a thread pool, with a
    read+write fallback per file. Used by ``generic.copy`` when both
    paths are GCS, ``overwrite=True``, and ``raise_on_exists=False``.
    """
    src_uri = ParseURI(src)
    dst_uri = ParseURI(dst)
    assert isinstance(src_uri, GCSURI), f"Expected GCSURI, got {type(src_uri)}"
    assert isinstance(dst_uri, GCSURI), f"Expected GCSURI, got {type(dst_uri)}"

    src_bucket = _get_bucket(src_uri)
    src_prefix = src_uri.path
    if src_prefix and not src_prefix.endswith('/'):
        src_prefix += '/'

    blobs = [
        b for b in src_bucket.list_blobs(prefix=src_prefix)
        if not b.name.endswith('/')
    ]
    if not blobs:
        logger.warning("copydir: source is empty or does not exist: %s", src)
        return

    logger.debug("copydir: copying %d files from %s to %s", len(blobs), src,
                 dst)

    src_base = src_prefix.rstrip('/')
    dst_base = dst_uri.path.rstrip('/')

    def _copy_one(src_name: str) -> None:
        rel = src_name[len(src_base):].lstrip(
            '/') if src_base else src_name.lstrip('/')
        dst_path = f"{dst_base}/{rel}" if dst_base else rel
        file_src = GCSURI(bucket=src_uri.bucket, path=src_name)
        file_dst = GCSURI(bucket=dst_uri.bucket, path=dst_path)
        if not _server_side_copy(file_src, file_dst):
            # Fall back to read+write.
            data = api.read_binary(file_src.as_gs())
            api.write_binary(file_dst.as_gs(), data)

    tasks = [fsutil_ctx.thread_pool.submit(_copy_one, b.name) for b in blobs]
    for task in tasks:
        task.result()

    logger.debug("copydir: done copying %s to %s", src, dst)


@api.customer_facing_path.register
def customer_facing_path(uri: GCSURI) -> str:
    return uri.as_gs()
