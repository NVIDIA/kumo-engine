# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, TypeVar


class ExplicitFilesystemCredentialsRequired(RuntimeError):
    pass


@dataclass(repr=False)
class FilesystemCredentialScope:
    credentials: Mapping[str, Any] | None
    require_explicit: bool
    cache: dict[tuple[str, int, str], Any] = field(default_factory=dict)

    @property
    def provider(self) -> str | None:
        if self.credentials is None:
            return None
        value = self.credentials.get('provider')
        return str(value) if value else None


_current_scope: contextvars.ContextVar[FilesystemCredentialScope
                                       | None] = (contextvars.ContextVar(
                                           'filesystem_credential_scope',
                                           default=None))


@contextmanager
def credential_scope(
    credentials: Mapping[str, Any] | None,
    *,
    require_explicit: bool,
) -> Iterator[None]:
    scope = FilesystemCredentialScope(credentials, require_explicit)
    token = _current_scope.set(scope)
    try:
        yield
    finally:
        _current_scope.reset(token)


def credentials_for_provider(provider: str) -> FilesystemCredentialScope | None:
    scope = _current_scope.get()
    if scope is None:
        return None
    if scope.provider == provider:
        return scope
    if scope.require_explicit:
        raise ExplicitFilesystemCredentialsRequired(
            f'Explicit {provider.upper()} credentials are required for this '
            'identity-backed connector')
    return None


def has_filesystem_credential_scope() -> bool:
    """Return whether the current operation has an explicit scope boundary."""
    return _current_scope.get() is not None


T = TypeVar('T')


class ContextPropagatingThreadPoolExecutor(ThreadPoolExecutor):
    """Thread pool that preserves request-local filesystem credentials.

    ``ContextVar`` state is not inherited by worker threads. Capturing it for
    each submission makes both ``submit`` and the base executor's ``map``
    fail closed under the exact credential scope active at submission time.
    A fresh copy per task also prevents a prior task's scope from leaking
    into later work on the same long-lived worker thread.
    """
    def submit(self, fn, /, *args, **kwargs):
        context = contextvars.copy_context()
        return super().submit(context.run, fn, *args, **kwargs)


def with_filesystem_credentials(fn: Callable[..., T]) -> Callable[..., T]:
    """Run a connector entrypoint under its request-local credential scope."""
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        scope_factory = getattr(self, 'filesystem_credential_scope', None)
        scope = scope_factory() if scope_factory is not None else nullcontext()
        with scope:
            return fn(self, *args, **kwargs)

    return wrapped


def scoped_cache_key(provider: str, discriminator: str) -> tuple[str, int, str]:
    return provider, threading.get_ident(), discriminator


def pyarrow_filesystem(path: str):
    """Build an explicit PyArrow filesystem for an identity-backed URI."""
    from pyarrow import fs
    filesystem_type: Any = fs.FileSystem  # pyright: ignore[reportPrivateImportUsage]

    if path.startswith(('s3://', 's3a://')):
        from kumo_s3.session import retry_strategy
        scope = credentials_for_provider('s3')
        if scope is None:
            # Local import avoids the credentials <-> S3 provider cycle.
            from io_utils.fsutil.s3_connection import active_s3_connection
            connection = active_s3_connection()
            if connection is not None:
                return fs.S3FileSystem(  # pyright: ignore[reportPrivateImportUsage]
                    **connection.pyarrow_options())
            return filesystem_type.from_uri(path)[0]
        if not (scope.credentials or {}).get('region'):
            # Local import avoids the credentials <-> S3 implementation cycle.
            from io_utils.fsutil.s3_filesystem import select_s3_region
            select_s3_region(path)
        credentials = scope.credentials or {}
        from io_utils.fsutil.s3_connection import active_s3_connection
        connection = active_s3_connection()
        ambient_session = credentials.get('ambient_session')
        if ambient_session is not None:
            # PyArrow cannot consume a boto Session. Freeze the already-
            # attested refreshable provider only for this filesystem object;
            # never re-run PyArrow's independent default provider chain.
            source = ambient_session.get_credentials()
            frozen = source.get_frozen_credentials() if source else None
            if not (frozen and frozen.access_key and frozen.secret_key
                    and frozen.token):
                raise RuntimeError(
                    'Attested ambient AWS session has no temporary credentials')
            options = {
                'access_key': frozen.access_key,
                'secret_key': frozen.secret_key,
                'session_token': frozen.token,
                'anonymous': False,
                'region': credentials.get('region'),
                'retry_strategy': retry_strategy(),
            }
            if connection is not None:
                options.update(connection.pyarrow_options())
                options.update({
                    'access_key': frozen.access_key,
                    'secret_key': frozen.secret_key,
                    'session_token': frozen.token,
                })
            return fs.S3FileSystem(  # pyright: ignore[reportPrivateImportUsage]
                **options)
        if connection is not None:
            return fs.S3FileSystem(  # pyright: ignore[reportPrivateImportUsage]
                **connection.pyarrow_options())
        return fs.S3FileSystem(  # pyright: ignore[reportPrivateImportUsage]
            access_key=credentials.get('aws_access_key_id'),
            secret_key=credentials.get('aws_secret_access_key'),
            session_token=credentials.get('aws_session_token'),
            region=credentials.get('region'),
            retry_strategy=retry_strategy(),
        )
    if path.startswith('gs://'):
        scope = credentials_for_provider('gcs')
        if scope is None:
            return filesystem_type.from_uri(path)[0]
        credentials = scope.credentials or {}
        credential = credentials.get('credential')
        token = getattr(credential, 'token', None)
        expiration = getattr(credential, 'expiry', None)
        if not token or expiration is None:
            raise RuntimeError('Federated GCS credential has no active token')
        return fs.GcsFileSystem(  # pyright: ignore[reportPrivateImportUsage]
            access_token=token,
            credential_token_expiration=expiration,
            project_id=(credentials.get('requester_pays_project')
                        or credentials.get('project')),
        )
    if path.startswith(('abfs://', 'abfss://')):
        # PyArrow's native AzureFileSystem cannot consume the federated
        # Python credential objects identity-backed connectors carry, so ADLS
        # goes through an fsspec adapter over the credentialed SDK clients.
        # Imported lazily: impl_adls imports this module for scope helpers.
        from io_utils.fsutil.adls_arrow import ADLSArrowFileSystem
        return fs.PyFileSystem(  # pyright: ignore[reportPrivateImportUsage]
            fs.FSSpecHandler(ADLSArrowFileSystem.from_uri(path)))
    return filesystem_type.from_uri(path)[0]
