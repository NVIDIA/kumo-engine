# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snowflake artifact upload and COPY policy shared by hosted and local BP."""
import logging
import re
import uuid
from contextlib import nullcontext
from typing import Any, List, Optional

from kumo_core.artifact_staging import (
    is_local_artifact_path,
    local_parquet_files,
    selected_local_parquet_files,
)

logger = logging.getLogger(__name__)
_ALLOWED_BLOB_PATH_RE = re.compile(r"^(s3://|gcs://|azure://|@)",
                                   flags=re.IGNORECASE)
MAX_COPY_FILES = 1000


class ArtifactWriteError(ValueError):
    """Provider artifact validation or COPY failed."""
    def __init__(self, message: str, *, cause: BaseException | None = None):
        super().__init__(message)
        # Preserve only the classification consumed by the caller's recovery
        # policy.  The public error remains stable and never exposes provider
        # credentials or SQL, while an expired OAuth token can still trigger
        # one idempotent overwrite retry.
        self.errno = getattr(cause, "errno", None)
        self.sqlstate = getattr(cause, "sqlstate", None)


def _to_snowflake_stage_sql_ref(path: str) -> Optional[str]:
    """Convert a path to a Snowflake stage reference usable in SQL, or
    None if the path is not a stage (e.g., cloud storage URL).

    '@<stage>' or '@<db>.<schema>.<stage>/path' is an already-resolved
    stage reference and is returned unchanged.

    Cloud storage URLs (s3://, gcs://, azure://) are not stage paths —
    returns None so the caller knows to treat them as quoted URLs with
    an explicit CREDENTIALS clause.

    Returns:
        The '@'-prefixed stage path to embed directly (unquoted) in SQL,
        or None if `path` is not a stage reference.
    """
    if path.startswith('@'):
        return path
    return None


def _validate_blob_path(path: str) -> None:
    """Reject paths that could allow SQL injection or unexpected schemes.

    Raises ConnectorError if the path is not an s3://, gcs://, azure://
    URL or a Snowflake stage reference (leading '@'), or if it contains
    characters that could break out of the SQL string literal.
    """
    if not _ALLOWED_BLOB_PATH_RE.match(path):
        raise ArtifactWriteError(
            f"Invalid blob storage path: {path!r}. Must start with "
            "'s3://', 'gcs://', 'azure://', or '@<stage>'.")
    # Quotes, backslashes, newlines, and semicolons have no legitimate
    # place in a URL or stage reference, and they could terminate the
    # surrounding SQL string literal.
    forbidden = ("'", '"', "\\", "\n", "\r", ";")
    if any(c in path for c in forbidden):
        raise ArtifactWriteError(
            f"Path contains disallowed characters: {path!r}")


def _validate_selected_artifact_files(artifact_path: str,
                                      artifact_files: List[str]) -> List[str]:
    """Reject selections that are not concrete objects in one container.

    A batch-prediction selection names one winning attempt directory per
    output shard, so its files legitimately sit beside ``artifact_path``
    rather than under it, and containment cannot be required. They must still
    share its scheme and container: the prefix they produce becomes both the
    temporary stage URL and the scope of the minted storage credential.
    """
    files = list(artifact_files)
    if not files or len(set(files)) != len(files):
        raise ArtifactWriteError(
            'Selected Snowflake artifacts require non-empty, unique object '
            'paths.')
    scheme, separator, _ = artifact_path.partition('://')
    for path in files:
        if path.endswith('/') or '*' in path or '?' in path:
            raise ArtifactWriteError(
                'Selected Snowflake artifacts require concrete object paths '
                'without wildcards.')
        # A local artifact path has no scheme to agree with; its files are
        # validated by `selected_local_parquet_files` before staging instead.
        if not separator:
            continue
        if not path.casefold().startswith(f'{scheme.casefold()}://'):
            raise ArtifactWriteError(
                'Selected Snowflake artifacts must share the artifact '
                'storage scheme.')
        if _container(path) != _container(artifact_path):
            raise ArtifactWriteError(
                'Selected Snowflake artifacts must share the artifact '
                'storage container.')
        # Every selected path is interpolated into FROM, FILES and
        # INFER_SCHEMA, so it is held to the same rules as the root path.
        _validate_blob_path(path)
    return files


def _container(path: str) -> str:
    """Scheme and authority, e.g. ``s3://bucket``: the credential boundary.

    Compared byte-exactly, so a selection mixing 's3://' and 'S3://' is
    refused even though the scheme check above casefolds. Making that case
    load would mean normalizing every path before both this and
    ``_common_prefix``, which compares segments exactly too; the canonical
    producer emits one casing, and the current behaviour is a refusal rather
    than a mis-scoped credential.
    """
    return '/'.join(path.split('/')[:3])


def _common_prefix(files: List[str]) -> str:
    """Longest shared directory of ``files``, always ending in ``/``.

    Split on separators rather than characters so two sibling roots such as
    ``runtime-00001`` and ``runtime-00010`` cannot share a partial segment.
    """
    segments = [path.split('/')[:-1] for path in files]
    shared: List[str] = []
    for parts in zip(*segments):
        if len(set(parts)) != 1:
            break
        shared.append(parts[0])
    # Below the container, never at it: this prefix becomes the temporary
    # stage URL and the storage credential's scope.
    if len(shared) <= 3:
        raise ArtifactWriteError(
            'Selected Snowflake artifacts share only their storage container; '
            'refusing to widen the load prefix to its root.')
    return '/'.join(shared) + '/'


def _relative_to(prefix: str, files: List[str]) -> List[str]:
    return [path[len(prefix):] for path in files]


def _quoted_files_clause(relative_files: List[str]) -> str:
    quoted = ', '.join(f"'{name}'" for name in relative_files)
    return f'FILES = ({quoted})'


def _upload_local_parquet_to_user_stage(
        context: Any, artifact_path: str, stage_ref: str, *,
        artifact_files: Optional[List[str]] = None, prepare_file=None,
        resolve_files=local_parquet_files) -> None:
    """Upload a worker-local Parquet dataset to one isolated user-stage path.

    ``SnowflakeCursor._upload`` is the same transfer primitive used by the
    connector's supported ``write_pandas`` helper.  It sends the stage
    location as a bound parameter, so a local filesystem path is never
    interpolated into warehouse SQL.  Each shard gets its own stage sub-prefix
    to avoid basename collisions in nested Parquet datasets.
    """
    # Already resolved and validated by the caller when a selection is given;
    # `_upload` stringifies either form.
    files = (resolve_files(artifact_path)
             if artifact_files is None else list(artifact_files))
    cursor = context.cursor()
    try:
        for index, file_path in enumerate(files):
            with (prepare_file(str(file_path))
                  if prepare_file else nullcontext(file_path)) as prepared:
                escaped_path = str(prepared).replace('\\', '\\\\').replace(
                    "'", "\\'")
                cursor._upload(
                    local_file_name=f"'file://{escaped_path}'",
                    stage_location=f'{stage_ref}part-{index:08d}/',
                    options={
                        'parallel': 4,
                        'source_compression': 'auto_detect',
                        'auto_compress': 'false',
                        'overwrite': 'true',
                    },
                )
    finally:
        cursor.close()


def write_artifacts_to_table(
        connector, table_name: str, artifact_path: str, replace: bool = False,
        *, artifact_files: Optional[List[str]] = None,
        get_temporary_aws_credential=lambda path: {}, upload_local=None,
        resolve_local=selected_local_parquet_files) -> None:
    r"""Write prediction artifacts (parquet) from S3 or a Snowflake
    stage into a customer's Snowflake table via COPY INTO.

    Supports four input path formats:
    - Worker-local Parquet files/directories — uploaded without loading
      them into memory to an isolated Snowflake user-stage prefix, then
      removed after COPY.
    - S3 URLs ('s3://bucket/prefix/') — uses inline STS credentials
      from `get_temporary_aws_credential`.
    - Explicit stage references ('@DB.SCHEMA.STAGE/path/') — passed
      through to SQL unchanged.

    GCS and ADLS are not yet implemented and will raise
    NotImplementedError.

    Args:
        table_name (str): The name of the table to write to.
        artifact_path (str): Path to local parquet artifacts, an s3://
            URL, or a Snowflake stage reference (leading '@').
        replace (bool): Whether to replace the table if it already exists.
        artifact_files: The exact object paths to load, as resolved by the
            caller. A batch-prediction selection spans one attempt
            directory per output shard, so no single prefix addresses it
            without also sweeping abandoned attempts. Omit to retain the
            legacy whole-prefix load.

    Raises:
        ConnectorError: If the schema does not exist or the write fails.
        NotImplementedError: If artifact_path is a cloud storage URL
            other than s3://.
    """
    if artifact_files is not None:
        artifact_files = _validate_selected_artifact_files(
            artifact_path, artifact_files)
    stage_sql_ref = _to_snowflake_stage_sql_ref(artifact_path)
    is_local_path = (stage_sql_ref is None
                     and is_local_artifact_path(artifact_path))
    if artifact_files is not None and stage_sql_ref is not None:
        # A stage FROM clause is emitted unquoted, so a selection would
        # have to be resolved back to '@<stage>/path'. Stage-backed
        # artifacts are not routed through a selection, so refuse rather
        # than build SQL that cannot be parsed.
        raise ArtifactWriteError(
            'Selected Snowflake artifacts are not supported for '
            'stage-backed artifact paths.')
    if artifact_files is not None and is_local_path:
        # Resolve before any warehouse work so a symlinked or missing
        # shard reports itself rather than a generic write failure.
        artifact_files = [str(path) for path in resolve_local(artifact_files)]
    if not is_local_path:
        _validate_blob_path(artifact_path)
    _lower = artifact_path.lower()
    if _lower.startswith('gcs://') or _lower.startswith('azure://'):
        raise NotImplementedError(
            "write_artifacts_to_table does not yet support "
            f"{_lower.split('://')[0]}:// sources. Use s3:// or a "
            "Snowflake stage path.")

    # Resolve the selected remote prefix here, before any provider work
    # or DDL. `_common_prefix` refuses a selection that shares only its
    # storage container, and with `replace=True` the DROP TABLE below
    # would already have destroyed the table by the time it did.
    selected_prefix: Optional[str] = None
    selected_relative_files: List[str] = []
    if artifact_files is not None and not is_local_path:
        selected_prefix = _common_prefix(artifact_files)
        selected_relative_files = _relative_to(selected_prefix, artifact_files)

    logger.info("Beginning import of artifacts to table {} from {} "
                "with replace={}".format(table_name, artifact_path, replace))

    ctx = connector.context
    local_stage_ref: str | None = None
    try:
        if is_local_path:
            local_stage_ref = f'@~/__kumo_artifact_{uuid.uuid4().hex}/'
            (upload_local or _upload_local_parquet_to_user_stage)(
                ctx, artifact_path, local_stage_ref,
                artifact_files=artifact_files)
            # Each selected shard now lives under the isolated stage
            # prefix, so the stage root addresses exactly the selection.
            artifact_files = None
            stage_sql_ref = local_stage_ref

        # Explicit stage URIs resolve directly. Local artifacts have
        # now been uploaded to an isolated user-stage prefix.
        is_stage_path = stage_sql_ref is not None

        qualified_name = connector._qualified_table_name(table_name)

        if replace:
            connector.execute_meta_query(
                ctx, f'DROP TABLE IF EXISTS {qualified_name}')

        # Build source path reference. Snowflake stage paths are
        # referenced unquoted in SQL; cloud storage URLs must be
        # wrapped in string literals.
        src_path = stage_sql_ref if is_stage_path else artifact_path
        copy_files: List[str] = []
        if artifact_files is not None:
            # Anchor on the shared prefix so one statement can name
            # objects that live under different attempt directories.
            # Computed above, before the DROP, so a refusal costs nothing.
            assert selected_prefix is not None
            src_path = selected_prefix
            copy_files = selected_relative_files
        if not src_path.endswith('/'):
            src_path += '/'
        from_clause = src_path if is_stage_path else f"'{src_path}'"

        # Build credentials clause separately so it is never included
        # in the SQL that `execute_meta_query` logs. Empty for stage
        # paths (Snowflake stages carry their own auth).
        creds = ({}
                 if is_stage_path else get_temporary_aws_credential(src_path))
        creds_clause = None
        if creds:
            creds_clause = (f"CREDENTIALS = ("
                            f"AWS_KEY_ID = '{creds['AccessKeyId']}' "
                            f"AWS_SECRET_KEY = '{creds['SecretAccessKey']}' "
                            f"AWS_TOKEN = '{creds['SessionToken']}')")

        # Create table from parquet schema if it doesn't exist.
        # COPY INTO requires the table to exist. Use INFER_SCHEMA
        # with a temporary stage + file format to auto-detect
        # the parquet schema.
        if not connector.table_exists(table_name):
            suffix = uuid.uuid4().hex[:8]
            tmp_fmt = f'__kumo_tmp_fmt_{suffix}'
            tmp_stage = f'__kumo_tmp_stage_{suffix}'
            try:
                # Create temp file format (required by INFER_SCHEMA)
                connector.execute_meta_query(
                    ctx,
                    f"CREATE TEMPORARY FILE FORMAT {tmp_fmt} "
                    # Keep Parquet timestamp annotations intact during both
                    # inference and the later COPY.  Without this Snowflake
                    # infers an Arrow timestamp as an integer epoch value.
                    f"TYPE = PARQUET USE_LOGICAL_TYPE = TRUE")

                if is_stage_path:
                    # Stage paths: wrap in quotes for INFER_SCHEMA
                    infer_location = f"'{src_path}'"
                else:
                    # Cloud storage: create temp stage with credentials
                    stage_sql = (f"CREATE TEMPORARY STAGE {tmp_stage}\n"
                                 f"URL = '{src_path}'")
                    connector.execute_meta_query(ctx, stage_sql,
                                                 creds_clause=creds_clause)
                    infer_location = f"'@{tmp_stage}'"

                # Every selected shard shares one schema fingerprint,
                # so naming a single object both suffices and keeps
                # inference off any abandoned sibling attempt.
                infer_files = (f",\n    FILES => ('{copy_files[0]}')"
                               if copy_files else '')
                infer_sql = (f"CREATE TABLE {qualified_name}\n"
                             f"USING TEMPLATE (\n"
                             f"  SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))\n"
                             f"  FROM TABLE(INFER_SCHEMA(\n"
                             f"    LOCATION => {infer_location},\n"
                             f"    FILE_FORMAT => '{tmp_fmt}'"
                             f"{infer_files})))")
                connector.execute_meta_query(ctx, infer_sql)
            finally:
                try:
                    connector.execute_meta_query(
                        ctx, f"DROP STAGE IF EXISTS {tmp_stage}")
                except Exception as e:
                    logger.warning(
                        'Snowflake output temp-stage cleanup failed (%s)',
                        type(e).__name__,
                    )
                try:
                    connector.execute_meta_query(
                        ctx, f"DROP FILE FORMAT IF EXISTS {tmp_fmt}")
                except Exception as e:
                    logger.warning(
                        'Snowflake output temp-format cleanup failed (%s)',
                        type(e).__name__,
                    )

        # Build COPY INTO to load parquet into the table
        # Snowflake caps one FILES clause at MAX_COPY_FILES, so a wide
        # selection needs several statements. Snowflake's own load
        # history skips files it already ingested, so an interrupted
        # multi-statement load converges on retry rather than
        # duplicating rows.
        batches = [
            copy_files[index:index + MAX_COPY_FILES]
            for index in range(0, len(copy_files), MAX_COPY_FILES)
        ] or [[]]
        for batch in batches:
            files_clause = (f"{_quoted_files_clause(batch)}\n" if batch else '')
            copy_sql = (f"COPY INTO {qualified_name}\n"
                        f"FROM {from_clause}\n"
                        f"{files_clause}"
                        f"FILE_FORMAT = (TYPE = PARQUET "
                        f"USE_LOGICAL_TYPE = TRUE)\n"
                        f"MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE\n"
                        f"PURGE = {'TRUE' if local_stage_ref else 'FALSE'}")

            logger.info(f"Loading parquet from {src_path} into "
                        f"{qualified_name}")
            connector.execute_meta_query(ctx, copy_sql,
                                         creds_clause=creds_clause)

        logger.info(f"Successfully wrote artifact data to "
                    f"'{table_name}'.")

    except Exception as e:
        logger.warning('Snowflake output write failed (%s)', type(e).__name__)
        raise ArtifactWriteError(
            'Failed to write artifacts to the configured Snowflake '
            'output table.', cause=e) from None
    finally:
        if local_stage_ref is not None:
            try:
                # COPY uses PURGE=TRUE. REMOVE also covers inference or
                # COPY failures and partially completed uploads.
                connector.execute_meta_query(ctx, f'REMOVE {local_stage_ref}')
            except Exception as cleanup_error:
                logger.warning(
                    'Snowflake local artifact cleanup failed (%s)',
                    type(cleanup_error).__name__,
                )


def overwrite_table(connector, src_table_name, dst_table_name, *, quote=None):
    """The hosted grant-preserving INSERT OVERWRITE / rename decision."""
    if quote is None:
        from kumo_snowflake.sql import quote_identifier as quote
    if not connector.table_exists(src_table_name):
        raise ArtifactWriteError(
            f"Source table '{src_table_name}' does not exist")
    if connector.table_exists(dst_table_name):
        from kumo_snowflake.sql import quote_identifier
        source_columns = connector.table_columns(src_table_name)
        target_columns = connector.table_columns(dst_table_name)
        if (len(source_columns) != len(target_columns)
                or set(source_columns) != set(target_columns)):
            raise ArtifactWriteError(
                'Source and destination artifact tables have different columns')
        projection = ', '.join(
            quote_identifier(column) for column in target_columns)
        connector.execute_meta_query(
            connector.context,
            f'INSERT OVERWRITE INTO {quote(dst_table_name)} ({projection}) '
            f'SELECT {projection} FROM {quote(src_table_name)}')
    else:
        connector.execute_meta_query(
            connector.context, f'ALTER TABLE {quote(src_table_name)} '
            f'RENAME TO {quote(dst_table_name)}')


def export_table_files(connector, table_name, artifact_path, *, replace,
                       artifact_files=None, staging_table_name=None):
    """Hosted staging-table publication and cleanup for one whole artifact."""
    if staging_table_name is not None and not replace:
        raise ValueError(
            'A Snowflake staging table may be used only with OVERWRITE mode.')
    staging = staging_table_name
    if replace and staging is None:
        staging = f'{table_name}_STAGING'
    if staging is not None:
        staging = f'{staging}_{uuid.uuid4().hex[:12]}'
    try:
        kwargs = {} if artifact_files is None else {
            'artifact_files': artifact_files
        }
        connector.write_artifacts_to_table(staging or table_name, artifact_path,
                                           replace=replace, **kwargs)
        if staging is not None:
            connector.overwrite_table(staging, table_name)
    finally:
        if staging is not None:
            try:
                connector.drop_table_if_exists(staging)
            except Exception as error:
                logger.warning(
                    'Output staging-table cleanup failed (error_type=%s)',
                    type(error).__name__)
