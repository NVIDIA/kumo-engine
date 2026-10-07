# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from threading import Lock
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from cachetools import TTLCache, cached
from pyarrow import fs

from io_utils.fsutil.uri import GCSURI, S3URI, ParseURI

logger = logging.getLogger(__name__)


class DecimalOutOfBoundsError(ValueError):
    """A scale-zero decimal cannot be represented by Kumo's int64 dtype."""
    def __init__(self, column_name: str):
        self.column_name = column_name
        super().__init__(
            f"Decimal column '{column_name}' contains a scale-zero value "
            "outside the signed 64-bit integer range.")


def fix_types_for_pandas(table: pa.Table) -> pa.Table:
    r"""
    Util method to cast all types in Pyarrow table to corresponding types
    that are compatible with pandas.

    This method currently only handles decimal types in pyarrow.
    """
    new_schema = []
    decimal_cols = {}
    # Define a new schema where all decimal128 columns are cast to int64
    for field in table.schema:
        if pa.types.is_decimal(field.type):
            if field.type.scale == 0:
                new_schema.append((field.name, pa.int64()))
                decimal_cols[field.name] = pa.int64()
            else:
                new_schema.append((field.name, pa.float64()))
                decimal_cols[field.name] = pa.float64()
        else:
            new_schema.append((field.name, field.type))

    # Create a new schema and cast the table
    try:
        new_table = table.cast(pa.schema(new_schema))
        return new_table
    except pa.ArrowInvalid:
        for col_name in decimal_cols.keys():
            col = table[col_name]
            for i in range(len(col)):
                try:
                    pc.cast(col.slice(i, 1), decimal_cols[col_name])
                except pa.ArrowInvalid:
                    raise DecimalOutOfBoundsError(col_name) from None


def safe_to_pandas(table: pa.Table) -> pd.DataFrame:
    r"""
    Convert a Pyarrow table to a pandas DataFrame, casting all timestamp
    columns to object type to avoid issues with out of bound timestamps.
    Casting the object type columns back to datetime coercing errors,
    effectively converting out of bound timestamps to NaT."""
    schema = table.schema
    # Identify columns with timestamp data type
    timestamp_cols = [
        field.name for field in schema if pa.types.is_timestamp(field.type)
    ]
    # Convert the table to pandas, casting all timestamp columns to object
    df = table.to_pandas(timestamp_as_object=True)
    df[timestamp_cols] = df[timestamp_cols].apply(pd.to_datetime,
                                                  errors='coerce')
    return df


# Must stay well below the IRSA/STS credential lifetime (~60min) so a cached
# S3FileSystem built with temporary session-token credentials is rebuilt with
# fresh credentials before they expire.
PYARROW_CLOUD_FS_TTL = 600


def _resolve_s3_filesystem(
        bucket_uri: str) -> Optional[fs.FileSystem]:  # type: ignore
    r"""Build a credentialed pyarrow ``S3FileSystem`` for the given bucket.

    Credential, region and endpoint handling belong to ``kumo_s3``, which the
    SDK's local backend binds through as well, so a bucket is reached the same
    way in either. Returns ``None`` when no credentials resolve, so the caller
    falls back to ``from_uri`` (anonymous / public reads, unchanged). Only
    ``s3://`` is handled; ``gs://`` keeps using ``from_uri``.
    """
    # Lazy import: io_utils.pyarrow loads early; avoid pulling the boto3 stack
    # into modules that never touch S3.
    import kumo_s3

    from io_utils.fsutil.s3_connection import active_s3_connection
    from io_utils.fsutil.s3_filesystem import threadsafe_boto
    from service_framework.aws_region import AWS_REGION

    connection = active_s3_connection()
    if connection is not None and connection.access_key and connection.secret_key:
        return fs.S3FileSystem(  # type: ignore[call-arg]
            **connection.pyarrow_options())

    # lazy_auth wires up any custom credential provider before we read
    # frozen credentials, mirroring the boto3 path.
    threadsafe_boto.lazy_auth()

    bucket = ParseURI(bucket_uri)
    assert isinstance(bucket, S3URI)

    # A vended stage can point at a non-AWS endpoint, so we target that
    # host; the region lazy_auth set on the session signs with SigV4.
    endpoint_url = threadsafe_boto._endpoint_url  # pylint: disable=protected-access
    options = None
    if connection is not None and connection.endpoint_url:
        options = connection.pyarrow_options()
        for key in ('access_key', 'secret_key', 'session_token', 'anonymous',
                    'region'):
            options.pop(key, None)

    return kumo_s3.filesystem(bucket.bucket, session=threadsafe_boto.session(),
                              endpoint_url=endpoint_url,
                              default_region=AWS_REGION, options=options)


@cached(cache=TTLCache(maxsize=10, ttl=PYARROW_CLOUD_FS_TTL), lock=Lock())
def _arrow_cloud_filesystem(path: str, connection_identity=None):
    r"""
    Get a pyarrow filesystem instance for the given cloud bucket.
    Input path is expected to be s3://<bucket> or gs://<bucket>
    Wrapped with a TTL cache to handle credential rotation
    Wrapped with a lock for thread safety
    """
    # cachetools includes this value in its cache key; the function body only
    # needs the path after the caller's connection state has been identified.
    del connection_identity
    if path.startswith("s3://"):
        # Prefer explicit boto3/IRSA credentials so cross-account buckets and
        # fork children do not 403; fall back to from_uri when none resolve
        # (public / anonymous reads).
        filesystem: Optional[fs.FileSystem] = None  # type: ignore
        try:
            filesystem = _resolve_s3_filesystem(path)
        except Exception as e:  # noqa: BLE001 - fall back to native resolution
            from io_utils.fsutil.s3_connection import active_s3_connection
            if active_s3_connection() is not None:
                raise
            logger.warning(
                "Falling back to native pyarrow S3 credentials after "
                "explicit-credential resolution failed (error_type=%s)",
                type(e).__name__)
            logger.debug(
                "Restricted raw pyarrow credential-resolution failure "
                "(path=%s, error_type=%s)",
                path,
                type(e).__name__,
                exc_info=(type(e), e, e.__traceback__),
            )
        if filesystem is not None:
            return filesystem

    # pyarrow's stubs don't re-export FileSystem from pyarrow.fs, though it is
    # available at runtime.
    filesystem, _ = fs.FileSystem.from_uri(path)  # type: ignore
    return filesystem


def arrow_cloud_filesystem(path: str):
    r"""
    Get a pyarrow filesystem instance for the given S3 or GCS path.
    """
    uri = ParseURI(path)
    if isinstance(uri, S3URI):
        uri.path = ""  # only use bucket for cache keying
        from io_utils.fsutil.s3_connection import active_s3_connection
        connection = active_s3_connection()
        identity = connection.identity if connection is not None else None
        return _arrow_cloud_filesystem(uri.as_s3(), identity)
    elif isinstance(uri, GCSURI):
        uri.path = ""  # only use bucket for cache keying
        return _arrow_cloud_filesystem(uri.as_gs())
    else:
        raise ValueError(f"Not a cloud path (S3 or GCS): {path}")
