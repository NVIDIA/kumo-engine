# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import decimal
import json
import os
from typing import Any, Callable, Dict, List, Optional, TypeVar

import numpy as np
import pandas as pd
from cachetools import Cache
from kumo_core.types.conversions import (
    dtype_to_numpy,
    dtype_to_pandas,
    to_dtype,
)
from kumo_core.types.schema import TS_FORMAT_METADATA
from kumoapi.typing import Dtype, Stype
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_object_dtype,
    is_timedelta64_dtype,
)

from io_utils import fsutil, kumo_dataclass, logging
from kumo.config import BaseColumnConfig
from kumo.connector.timestamp_convert import (
    PANDAS_UNIT_OF_INDEX_CLOCK,
    to_epoch_days,
    to_epoch_seconds,
    to_epoch_ticks,
)
from kumo.connector.timestamp_infer import (
    HintKind,
    TemporalHint,
    _is_native_temporal_series,
    infer_temporal_hint,
    map_timestamp_unit,
)

# Default data version for all tables.
DEFAULT_DATA_VERSION = '-1'

PLACEHOLDER = '__PLACEHOLDER__'
SAMPLING_SEED = 42
DEFAULT_CAST_ERROR_THRESHOLD = 0.10

T = TypeVar('T')

logger = logging.getLogger(__name__)

# Env Var set via IAC 2.0 for IAM role associated with the k8s
# service account.
_ENV_AWS_ROLE_ARN = 'AWS_ROLE_ARN'

# Path to local file for OAuth token. Used to get AWS STS token for
# temporary credentials passed to warehouse for accessing S3 buckets.
_LOCAL_WEB_IDENTITY_TOKEN_PATH = \
    '/var/run/secrets/eks.amazonaws.com/serviceaccount/token'

# Role ARN used in test environments (dev VM, CI) to get temporary
# credentials via sts:AssumeRole instead of OIDC web identity.
_ENV_AWS_TEST_ROLE_ARN = 'AWS_TEST_ROLE_ARN'


def get_temporary_aws_credential(
    path: str,
    *,
    validity: datetime.timedelta | None = None,
) -> Dict[str, str]:
    r"""Fetch temporary AWS credentials for warehouse access to S3.

    In production (EKS with IRSA), uses the pod's OIDC token to call
    sts:AssumeRoleWithWebIdentity.

    In test environments (dev VM, CI), uses the machine's existing AWS
    session to call sts:AssumeRole with AWS_TEST_ROLE_ARN.

    Returns:
        Dict containing AWS credentials with keys: 'AccessKeyId',
        'SecretAccessKey', 'SessionToken'. Empty dict if no credentials.
    """
    if fsutil.is_ucv(path):
        return {}
    duration_seconds = int((validity
                            or datetime.timedelta(hours=1)).total_seconds())
    duration_seconds = max(900, min(duration_seconds, 43_200))

    # Production path: IRSA (OIDC token + role ARN)
    if fsutil.exists(_LOCAL_WEB_IDENTITY_TOKEN_PATH):
        with open(_LOCAL_WEB_IDENTITY_TOKEN_PATH, 'r') as f:
            token = f.read()
        role = os.environ.get(_ENV_AWS_ROLE_ARN, '')
        if token and role:
            try:
                import boto3
                sts = boto3.client(
                    'sts',
                    region_name=os.environ.get('AWS_REGION', 'us-west-2'),
                )
                response = sts.assume_role_with_web_identity(
                    RoleArn=role,
                    RoleSessionName='session',
                    WebIdentityToken=token,
                    DurationSeconds=duration_seconds,
                )
                output = response['Credentials']
                return {
                    'AccessKeyId': output['AccessKeyId'],
                    'SecretAccessKey': output['SecretAccessKey'],
                    'SessionToken': output['SessionToken']
                }
            except Exception as e:
                # Provider exceptions may echo the web-identity token, role,
                # or response body.  This path is used to vend credentials to
                # warehouses, so log only locally controlled metadata.
                logger.warning(
                    'AWS web-identity role assumption failed '
                    '(error_type=%s)',
                    type(e).__name__,
                )

    # Test path: assume the test role to get STS credentials. Two
    # strategies are tried in order:
    #
    # 1. IMDS (EC2 instance profile) — used on dev VMs. We query IMDS
    #    directly because boto3.Session() on dev VMs often picks up an
    #    expired SSO profile from ~/.aws/config and fails instead of
    #    falling through to IMDS.
    # 2. Env-var credentials (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY
    #    and optional AWS_SESSION_TOKEN) — used on GitHub Actions CI
    #    runners which are not EC2 instances (no IMDS).
    test_role = os.environ.get(_ENV_AWS_TEST_ROLE_ARN, '')
    if test_role:
        # Strategy 1: IMDS (dev VMs / EC2 instances)
        try:
            import urllib.request

            import boto3
            imds_base = ('http://169.254.169.254/latest/meta-data'
                         '/iam/security-credentials/')
            imds_role = urllib.request.urlopen(imds_base,
                                               timeout=2).read().decode()
            imds_creds = json.loads(
                urllib.request.urlopen(imds_base + imds_role,
                                       timeout=2).read().decode())
            sts = boto3.client(
                'sts',
                aws_access_key_id=imds_creds['AccessKeyId'],
                aws_secret_access_key=imds_creds['SecretAccessKey'],
                aws_session_token=imds_creds['Token'],
            )
            response = sts.assume_role(
                RoleArn=test_role,
                RoleSessionName='kumo-test-session',
                DurationSeconds=duration_seconds,
            )
            creds = response['Credentials']
            return {
                'AccessKeyId': creds['AccessKeyId'],
                'SecretAccessKey': creds['SecretAccessKey'],
                'SessionToken': creds['SessionToken']
            }
        except Exception as e:
            logger.debug('IMDS path failed (error_type=%s)', type(e).__name__)

        # Strategy 2: env-var credentials (CI runners)
        try:
            import boto3
            access_key = os.environ.get('AWS_ACCESS_KEY_ID')
            secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
            if access_key and secret_key:
                sts = boto3.client(
                    'sts',
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    aws_session_token=os.environ.get('AWS_SESSION_TOKEN'),
                )
                response = sts.assume_role(
                    RoleArn=test_role,
                    RoleSessionName='kumo-test-session',
                    DurationSeconds=duration_seconds,
                )
                creds = response['Credentials']
                return {
                    'AccessKeyId': creds['AccessKeyId'],
                    'SecretAccessKey': creds['SecretAccessKey'],
                    'SessionToken': creds['SessionToken']
                }
        except Exception as e:
            logger.warning('Failed to assume test role %s (error_type=%s)',
                           test_role,
                           type(e).__name__)

    # Fall back to the worker's ambient AWS creds (the same ones the S3
    # client uses) so the credentialed COPY can run server-side instead of
    # routing bytes through the worker.
    try:
        import boto3
        creds = boto3.Session().get_credentials()
        frozen = creds.get_frozen_credentials() if creds else None
        if (frozen and frozen.access_key and frozen.secret_key
                and frozen.token):
            return {
                'AccessKeyId': frozen.access_key,
                'SecretAccessKey': frozen.secret_key,
                'SessionToken': frozen.token,
            }
    except Exception as e:
        logger.warning("Ambient AWS credential lookup failed (error_type=%s)",
                       type(e).__name__)

    logger.warning('Could not retrieve temporary credential. Proceed '
                   'without credentials.')

    return {}


def cast_df(
    df: pd.DataFrame,
    pkey: Optional[BaseColumnConfig] = None,
    cols: Optional[List[BaseColumnConfig]] = None,
) -> pd.DataFrame:
    r"""Cast the :class:`pandas.DataFrame` to the :obj:`dtype` configured in
    :obj:`pkey` and :obj:`cols`."""
    dtypes: Dict[str, Dtype] = {}
    if pkey is not None and pkey.dtype is not None:
        dtypes[pkey.name] = pkey.dtype
    for col in (cols or []):
        if col.dtype is not None:
            dtypes[col.name] = col.dtype
    return cast_df_to_dtypes(df, dtypes)


@kumo_dataclass
class ColumnDtypeMismatch:
    col_name: str
    orig_type: str
    # TODO(zack): We should show kumo.dtype instead of pandas DType
    dtype: str
    original_msg: str

    def format_msg(self):
        return (f"- Column Name: {self.col_name}, Expected Data Type: "
                f"{self.dtype}, Initial Type: {self.orig_type}, "
                f"Error Message: {self.original_msg}")


class ColumnDtypeMismatchError(ValueError):
    r"""Raised when the column data type does not match the underlying data.
    """
    def __init__(self, mismatches: List[ColumnDtypeMismatch]):
        self.mismatches = mismatches
        error_parts = [m.format_msg() for m in mismatches]
        errors = '\n'.join(error_parts)
        super().__init__(
            f"Some rows in the table could not be converted into the "
            f"requested data types. Please make sure that all rows in "
            f"the table contain valid data, and that the specified column "
            f"data types match the data type of the underlying data. To "
            f"solve this error, you can consider excluding the offending "
            f"columns when importing the data, change the column data type "
            f"if it is improperly set, or delete the rows containing "
            f"invalid data from the underlying data table. The full list of "
            f"errors is listed here:\n\n{errors}")


def cast_error_frac(col: pd.Series, cast_col: pd.Series) -> float:
    r"""Returns the fraction of casting errors (computed as the difference in
    the number of NA values) resulting from applying casting to col.
    """
    if col.size != cast_col.size:
        raise ValueError("Arguments to cast_error_frac are of different sizes")
    if col.size == 0:
        return 0.0
    orig_na = col.isna().sum()
    cast_na = cast_col.isna().sum() - orig_na
    error_frac = cast_na / col.size
    return error_frac


def cast_col_to_datetime(col: pd.Series) -> pd.Series:
    r"""Cast a column to second-precision ``datetime64``, via the same mechanism
    ingest uses. Best-effort: a value that cannot be parsed becomes ``NaT``.
    """
    # Arrow-backed pandas columns can expose a ChunkedArray, while DiskGraph's
    # resolver accepts one contiguous Arrow Array.
    col = pd.Series(col.to_numpy(), index=col.index, name=col.name)
    hint = infer_temporal_hint(col)
    epoch_seconds = to_epoch_seconds(col, hint)
    return pd.to_datetime(
        pd.Series(epoch_seconds, index=col.index, name=col.name),
        unit='s',
        errors='coerce',
    )


def _resolve_stats_temporal_hint(
    ser: pd.Series,
    col_config: BaseColumnConfig,
) -> TemporalHint:
    r"""Resolve the ``TemporalHint`` the col-stat converter should apply.

    The hint is read from the frozen column config first, so the authoritative
    inference (persisted by :func:`~kumo.graph.table.Table.infer_dtypes_from_stypes`
    and ``update_table_timestamp_units``) is honored rather than re-guessed:
      * a normalized ``timestamp_unit`` yields a ``UNIT`` hint;
      * a non-empty ``metadata['TS_FORMAT']`` yields a ``FORMAT`` hint;
      * an already-datetime-like series (including an object series of typed
        temporal scalars) yields a ``NATIVE`` hint;
      * otherwise the detector runs on the series as a last-resort fallback so a
        missing hint still degrades sensibly.

    The declared unit is normalized through
    :func:`~kumo.connector.timestamp_infer.map_timestamp_unit` so aliases like
    "seconds" are accepted, and an empty-string ``TS_FORMAT`` is treated as unset.
    """
    unit = map_timestamp_unit(col_config.timestamp_unit)
    if unit is not None:
        return TemporalHint.of_unit(unit)
    ts_format = col_config.metadata.get(TS_FORMAT_METADATA)
    if ts_format:  # a non-empty strftime format; '' is treated as unset
        return TemporalHint.of_format(ts_format)
    if _is_native_temporal_series(ser):
        return TemporalHint.native()
    return infer_temporal_hint(ser)


# Day-less temporal formats DiskGraph detects from bare integer columns: a
# 4-digit year (``%Y``) or 6-digit year-month (``%Y%m``). They carry no day, so
# name-based inference must not promote them to a temporal dtype.
_DAY_LESS_TEMPORAL_FORMATS = frozenset({'%Y', '%Y%m'})


def name_inference_temporal_supported(
    col: pd.Series,
    desired_dtype: Dtype,
    error_threshold: Optional[float] = None,
) -> bool:
    r"""Whether a name-based temporal override is supported by sampled values.

    DiskGraph (``resolve_temporal``, via :func:`infer_temporal_hint` and the
    ``to_epoch_*`` appliers) is the sole authority on whether each value parses;
    this function only adds the authorization policy that an automatic
    name-based dtype override requires at least ``1 - error_threshold`` of the
    non-null sampled values to resolve. Returns ``False`` for an all-null sample,
    a column DiskGraph cannot resolve at all, or a bare year / year-month column
    whose only temporal reading would have to invent a day.
    """
    if error_threshold is None or error_threshold > 1.0:
        error_threshold = DEFAULT_CAST_ERROR_THRESHOLD
    # Arrow-backed pandas columns can expose a ChunkedArray, while DiskGraph's
    # resolver accepts one contiguous Arrow Array.
    col = pd.Series(col.to_numpy(), index=col.index, name=col.name)
    non_null = col.notna().to_numpy()
    n = int(non_null.sum())
    if n == 0:
        return False
    hint = infer_temporal_hint(col)
    if hint.kind is HintKind.UNRESOLVED:
        return False
    if (hint.kind is HintKind.FORMAT
            and hint.strftime in _DAY_LESS_TEMPORAL_FORMATS):
        # A day-less (``%Y`` / ``%Y%m``) promotion would produce unexpected
        # times, so leave the column alone; the user can declare it explicitly
        # if a temporal reading is intended.
        return False
    epochs = (to_epoch_days(col, hint) if desired_dtype == Dtype.date else
              to_epoch_seconds(col, hint))
    failures = sum(1 for value, keep in zip(epochs, non_null)
                   if keep and value is None)
    return failures / n <= error_threshold


def coerce_timestamp_stype_series_for_stats(
    ser: pd.Series,
    col_config: BaseColumnConfig,
    timestamp_precision: str = 'seconds',
) -> pd.Series:
    r"""Coerce a ``timestamp``-stype series to a datetime series before the Python
    column-stat kernel reduces it.

    ``compute_stats_python._min``/``_max`` only return a ``pd.Timestamp`` when the
    series is already datetime-like; an int64/float epoch series would fall to the
    numeric branch and produce a bare integer MIN/MAX, which is then serialized
    without the ``kumo_timestamp_`` prefix and later overflows when reinterpreted
    as a datetime. Coercing here at the col-stat production site ensures every
    downstream consumer receives a typed timestamp stat.

    The coercion resolves the column's :class:`TemporalHint`
    (:func:`_resolve_stats_temporal_hint`) and applies
    :func:`~kumo.connector.timestamp_convert.to_epoch_ticks` in the
    ``timestamp_precision`` index clock, then wraps the ticks back into a
    ``datetime64`` series of that precision. The resulting MIN/MAX (and the
    other timestamp stats) are computed in the same domain the on-disk index
    uses and match it rather than drifting.

    Best-effort and narrowly scoped:
      * non-``timestamp``-stype columns pass through untouched;
      * ``timedelta`` duration columns are outside the epoch/date converter's
        domain and pass through untouched;
      * an unconvertible or out-of-range value becomes ``NaT`` and is dropped by
        the kernel, never raising.
    """
    if col_config.stype != Stype.timestamp:
        return ser
    if is_timedelta64_dtype(ser):
        # Duration columns are not epoch/date columns; the converter cannot
        # interpret them, so pass them through untouched.
        return ser

    hint = _resolve_stats_temporal_hint(ser, col_config)
    ticks = to_epoch_ticks(ser, hint, timestamp_precision)
    return pd.to_datetime(pd.Series(ticks, index=ser.index),
                          unit=PANDAS_UNIT_OF_INDEX_CLOCK[timestamp_precision],
                          errors='coerce')


def coerce_decimal_series_for_stats(
    ser: pd.Series,
    col_config: BaseColumnConfig,
) -> pd.Series:
    r"""Coerce a decimal-backed series to its declared numeric dtype before the
    Python column-stat kernel reduces it.

    A warehouse ``DECIMAL``/``NUMERIC`` arrives from pandas as an ``object``
    column of :class:`decimal.Decimal`, which :func:`to_dtype` reports as
    ``string`` -- it has no decimal branch, and a ``Decimal`` is neither
    list-like nor a datetime, so it lands on the ``object`` fallback. The
    caller then keeps only the stats ``string`` supports, silently dropping
    MIN/MAX/MEAN/QUANTILES/STD/HISTOGRAM, and a ``numerical`` column reaches
    ``ColumnStats.from_stats_dict`` without the numerical stats it requires
    (kumo-ai/kumo#30917).

    Only Databricks avoids this, by casting decimals to int64/float64 in
    ``fix_types_for_pandas`` before pandas sees them; the BigQuery, Snowflake
    and Parquet readers all hand the objects straight through.

    Coercing to the column's *declared* dtype rather than to a general float
    keeps the stats in the domain the values are ingested in, so MIN/MAX
    describe the column DiskGraph will hold rather than the source's exact
    decimal. Best-effort: a non-numeric declared dtype, or a series that is
    not object-backed decimals, passes through untouched, and an unconvertible
    value becomes NaN and is dropped by the kernel rather than raising.
    """
    if col_config.dtype is None or not col_config.dtype.is_numerical():
        return ser
    if not is_object_dtype(ser):
        return ser
    sample = ser.dropna()
    if sample.empty or not isinstance(sample.iloc[0], decimal.Decimal):
        return ser
    coerced = pd.to_numeric(ser, errors='coerce')
    if not col_config.dtype.is_int():
        return coerced
    try:
        return coerced.astype('Int64')
    except (TypeError, ValueError):
        # A fractional value in a column declared integral. Reduce the floats
        # rather than truncating them here: the statistics then describe what
        # the column holds, and the column's own type is the thing to fix.
        return coerced


def cast_col_to_numeric(
    col: pd.Series,
    error_threshold: Optional[float] = None,
) -> pd.Series:
    r"""Cast the input column :obj:`col` to a pandas numeric type.
    Raises an exception if casting results in more errors than
    :obj:`error_threshold` (default :obj:`DEFAULT_CAST_ERROR_THRESHOLD`).
    """
    if error_threshold is None or error_threshold > 1.0:
        error_threshold = DEFAULT_CAST_ERROR_THRESHOLD

    cast_col = pd.to_numeric(col, errors='coerce')
    error_frac = cast_error_frac(col, cast_col)
    if error_frac > error_threshold:
        raise ValueError(f"Column '{col.name}' with Dtype({col.dtype}) could "
                         f"not be cast to numeric format. Casting resulted in "
                         f"{error_frac * 100:.1f}% errors.")
    return cast_col


def cast_col_to_str(col: pd.Series) -> pd.Series:
    r"""Cast the input column :obj:`col` to a pandas str type."""
    return col.astype("string")


def _known_dtype(dtype: Any) -> Optional[Dtype]:
    try:
        return to_dtype(dtype)
    except KeyError:
        return None


def cast_df_to_dtypes(
    df: pd.DataFrame,
    dtypes: Dict[str, Dtype],
    error_threshold: Optional[float] = None,
) -> pd.DataFrame:
    r"""Cast the columns in a :class:`pandas.DataFrame` to :class:`Dtype`
    specified in :obj:`dtypes`."""
    # Apply 'coerce' casting only in known cases. And apply
    # error_threshold when 'coerce' casting.
    mismatches: List[ColumnDtypeMismatch] = []

    for col_name, dtype in dtypes.items():
        col = df.index if df.index.name == col_name else df[col_name]
        try:
            # For binary columns, always convert to hex str.
            if dtype == Dtype.binary:
                df[col_name] = col.apply(lambda x: x.hex()
                                         if isinstance(x, bytes) else x)
                continue

            # For boolean columns, always convert to Pandas dtypes
            # This is because Pandas does not carry over dtype information
            # for other operations when using np.dtype('bool').
            # For example, `ser.value_counts()` will return a Series of type
            # 'object' instead of 'bool'. This doesn't happen for
            # pd.BooleanDtype or other dtypes that are not np.dtype('bool').
            elif dtype == Dtype.bool:
                if col.dtype == pd.BooleanDtype():
                    continue

            elif dtype != Dtype.timedelta and dtype == _known_dtype(col.dtype):
                continue

            if dtype.is_temporal() and not is_datetime64_any_dtype(col.dtype):
                df[col_name] = cast_col_to_datetime(col)
            elif not dtype.is_temporal():
                if df.index.name == col_name:
                    df.index = df.index.astype(dtype_to_numpy(dtype))
                elif len(col) > 0 and pd.api.types.is_list_like(col.iloc[0]):
                    # in case of list-like columns, we don't need to cast them
                    # to numerical as list requires them to have object dtype.
                    pass
                else:
                    if dtype.is_numerical():
                        col = cast_col_to_numeric(col, error_threshold)
                        if dtype == dtype.timedelta:
                            dtype = dtype.int64
                    df[col_name] = col.astype(dtype_to_pandas(dtype))

        except (ValueError, TypeError, OverflowError) as e:
            mismatches.append(
                ColumnDtypeMismatch(
                    col_name=col_name,
                    orig_type=str(col.dtype),
                    dtype=str(dtype_to_pandas(dtype)),
                    original_msg=str(e),
                ))

    if len(mismatches) > 0:
        raise ColumnDtypeMismatchError(mismatches)

    # cast generic `object` values to `str`, if requested `dtype`
    # is `Dtype.string`. Preserve NA while casting.
    for col in df.select_dtypes(include='object').columns:
        if col in dtypes and dtypes[col].is_string():
            df[col] = np.where(pd.isnull(df[col]), df[col], df[col].astype(str))
            df[col] = df[col].astype(dtype_to_pandas(dtypes[col]))

    return df


# TODO(Dulloor): This function is not currently used. Consider removing it.
def cached_batch_fetch(entity_names: List[str], cache: Cache,
                       cache_key: Callable[[str], str],
                       fetch_fn: Callable[[List[str]], List[T]]) -> List[T]:
    """
    This function performs a fetch operation using a cache. cache_key is used
    to retrieve and store data for each entity based on a name. The fetch
    function is expected to take in a list of entity names and return a list
    of entity data _in the same order_.

    Args:
        entity_names (List[str]): A list containing the base names of the
            entities to be fetched. In a non-cache scenario, the fetch_fn
            would be passed in this list of entity names.
        cache (cachetools.Cache): A cache that stores data associated with each
            entity using the key `cache_key(entity_name)`
        cache_key (Callable[[str], str]): A function that takes in an entity
            name and returns the key where the entity's data is to be stored
        fetch_fn (Callable[[List[str]], List[Any]]): A function that returns
            the entity data if the data is not already in the cache. This
            function takes in a list of entity_names (without the cache_key
            function applied to them) and returns a list of entity data where
            each data point corresponds to the entity name at the same index
            of the input list.

    Returns:
        all_data (List[Any]): A list of data points corresponding to
            entity_names in the same order they were passed in
    """
    # List to return with placeholders
    all_data = [PLACEHOLDER] * len(entity_names)

    # Get data from cache for cached entities
    cached_entities = [(i, entity_name)
                       for i, entity_name in enumerate(entity_names)
                       if cache_key(entity_name) in cache]
    # Add it to the all_data list in the correct order
    for i, entity_name in cached_entities:
        data = cache.get(cache_key(entity_name))
        if data is not None:
            all_data[i] = data

    # Get the data for any entities not in cache
    not_cached_entities = [(i, entity_name)
                           for i, entity_name in enumerate(entity_names)
                           if cache_key(entity_name) not in cache]
    not_cached_entity_names = [e_name for i, e_name in not_cached_entities]
    not_cached_data = fetch_fn(not_cached_entity_names)

    # Add the data to the all_data list in the correct order
    for j in range(len(not_cached_data)):
        i, entity_name = not_cached_entities[j]
        entity_data = not_cached_data[j]
        all_data[i] = entity_data

    # Store them in the cache
    for name, data in zip(not_cached_entity_names, not_cached_data):
        cache[cache_key(name)] = data

    return all_data
