# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Identity objects: pass credentials directly, not by reference.

`kumo_core.sources.principals.SourceBackend`/`Sessions` classes resolve
identity through `kumo_core`'s own process-wide `PrincipalStore`, addressed by
name (`kumo_core.sources.principals.IdentityRef = Union[str, Named]`). That's
a registry, and a caller who wants to pass credentials directly would
otherwise have to register them under a name first and then pass the name --
exactly the indirection this package avoids elsewhere.

`Identity` (and its five per-provider subclasses below) hides that: a caller
builds e.g. `SnowflakeIdentity(account=..., username=..., password=...)` and
passes it straight to `SnowflakeConnector(identity=...)`. Construction
registers the identity into `kumo_core`'s store under a random slug and the
object exposes that slug (`Named.slug`), so from `kumo_core`'s point of view
it's still going through `PrincipalStore` -- credentials still live only in
that process-local store, never copied elsewhere -- but the caller never
sees the "register, then reference by name" two-step.

The slug is a random per-registration key, not derived from `config`/
`credentials` in any way: a slug computed from credential content, even
hashed, would be a deterministic, externally-visible function of a
(frequently low-entropy) secret -- logged or otherwise leaked, it becomes
an offline verifier an attacker can check a guessed credential against,
for a connection configuration that's often already known. A random slug
has none of that risk, at the cost of two `Identity`s built from
identical config/credentials no longer coalescing into one store entry --
each `Identity()` call is its own registration, which is fine here:
nothing in this package looks two `Identity` objects up by whether they
happen to share credentials, and `PrincipalStore` is a process-local,
per-run registry, not a durable one two separate `Identity()` calls need
to agree on.

Config/credential dataclasses (`SnowflakeIdentityConfig`,
`SnowflakePasswordCredentials`, ...) are reused unchanged from
`kumoapi.data_source` -- the same pure, wire-format dataclasses the cloud SDK
uses, including their validation (`validate_identity_credentials_match`,
per-field normalization).
"""

import uuid
from typing import Optional

from kumo_core.sources.principals import store
from kumoapi.data_source import (
    BigQueryIdentityConfig,
    CreateIdentityArgs,
    DatabricksAuthenticationKind,
    DatabricksIdentityConfig,
    DatabricksOAuthCredentials,
    GCSIdentityConfig,
    GoogleAmbientAuthentication,
    GoogleIdentityAuthentication,
    GoogleServiceAccountKeyAuthentication,
    GoogleServiceAccountKeyCredentials,
    GoogleWorkloadIdentityAuthentication,
    IdentityConfig,
    IdentityCredentials,
    S3AccessKeyAuthentication,
    S3AccessKeyCredentials,
    S3AmbientAuthentication,
    S3AssumeRoleAuthentication,
    S3ConnectionConfig,
    S3IdentityConfig,
    SnowflakeAuthenticationKind,
    SnowflakeIdentityConfig,
    SnowflakeKeyPairCredentials,
    SnowflakePasswordCredentials,
    validate_identity_credentials_match,
)


class Identity:
    r"""Base class for per-provider identity objects.

    Args:
        config: The provider's `kumoapi.data_source.*IdentityConfig`.
        credentials: The matching `kumoapi.data_source.*Credentials`, or
            `None` where the authentication kind carries none (ambient AWS/
            GCP, or a Snowflake OAuth token file).
    """

    def __init__(
        self,
        config: IdentityConfig,
        credentials: Optional[IdentityCredentials] = None,
    ) -> None:
        validate_identity_credentials_match(config, credentials)
        self._config = config
        self._credentials = credentials
        # A random key, not derived from config/credentials at all -- see
        # the module docstring for why. `uuid4().hex` is 32 lowercase hex
        # characters, comfortably inside the slug format (kumoapi.
        # data_source's validate_resource_slug: 3-48 lowercase ASCII
        # letters/digits/hyphens, starting with a letter, ending with a
        # letter or digit).
        self._slug = "identity-" + uuid.uuid4().hex
        # Always a fresh name in practice (birthday collision on a random
        # 128-bit value) -- but unlike the earlier credential-derived slug,
        # a collision here is never expected to be the *same* registration,
        # so it must not be treated as one. Call store.create unconditionally
        # rather than guarding on `self._slug not in store.names()` first:
        # store.create already raises on a name that already exists, and a
        # guard that silently skipped registration on that outcome would
        # silently alias this Identity to a stored Principal it doesn't
        # actually share credentials with -- the exact hazard this random
        # scheme exists to avoid, reintroduced by trying to avoid it.
        store.create(
            CreateIdentityArgs(
                display_name=self._slug,
                slug=self._slug,
                config=self._config,
                credentials=self._credentials,
            ))

    @property
    def slug(self) -> str:
        r"""Satisfies `kumo_core.sources.principals.Named`, so this object is
        directly usable wherever an `IdentityRef` is expected.
        """
        return self._slug

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(slug={self._slug!r})"


class SnowflakeIdentity(Identity):
    r"""A Snowflake principal, authenticated by password, key pair, or an
    ambient OAuth token file (Snowpark Container Services).

    Specify at most one of `password`/`private_key`. Neither means OAuth
    token-file authentication, matching `SnowflakeAuthenticationKind.
    OAUTH_TOKEN_FILE`.
    """

    def __init__(
        self,
        *,
        account: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        private_key: Optional[str] = None,
        private_key_passphrase: Optional[str] = None,
        host: Optional[str] = None,
        token_path: Optional[str] = None,
    ) -> None:
        if password is not None and private_key is not None:
            raise ValueError(
                "Specify at most one of password or private_key.")
        credentials: Optional[IdentityCredentials]
        if password is not None:
            authentication = SnowflakeAuthenticationKind.PASSWORD
            credentials = SnowflakePasswordCredentials(password=password)
        elif private_key is not None:
            authentication = SnowflakeAuthenticationKind.KEY_PAIR
            credentials = SnowflakeKeyPairCredentials(
                private_key=private_key,
                private_key_passphrase=private_key_passphrase,
            )
        else:
            authentication = SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE
            credentials = None
        config = SnowflakeIdentityConfig(
            account=account,
            authentication=authentication,
            username=username,
            host=host,
            token_path=token_path,
        )
        super().__init__(config, credentials)


class S3Identity(Identity):
    r"""An AWS principal, authenticated by access key, assume-role, or the
    ambient AWS credential chain (instance profile, `aws configure`,
    environment).

    Specify `access_key_id`/`secret_access_key` together for access-key auth,
    `role_arn` alone for assume-role auth, or neither for the ambient chain.
    """

    def __init__(
        self,
        *,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        role_arn: Optional[str] = None,
        partition: str = 'aws',
        connection: Optional[S3ConnectionConfig] = None,
    ) -> None:
        credentials: Optional[IdentityCredentials]
        if access_key_id is not None or secret_access_key is not None:
            if access_key_id is None or secret_access_key is None:
                raise ValueError(
                    "access_key_id and secret_access_key must be given "
                    "together.")
            authentication = S3AccessKeyAuthentication()
            credentials = S3AccessKeyCredentials(
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                session_token=session_token,
            )
        elif role_arn is not None:
            authentication = S3AssumeRoleAuthentication(
                partition=partition, role_arn=role_arn)
            credentials = None
        else:
            authentication = S3AmbientAuthentication()
            credentials = None
        config = S3IdentityConfig(authentication=authentication,
                                  connection=connection)
        super().__init__(config, credentials)


def _google_authentication(
    *,
    service_account_key: Optional[dict] = None,
    workload_service_account_email: Optional[str] = None,
) -> tuple[GoogleIdentityAuthentication, Optional[IdentityCredentials]]:
    if service_account_key is not None:
        authentication = GoogleServiceAccountKeyAuthentication(
            project_id=service_account_key['project_id'],
            client_email=service_account_key['client_email'],
            client_id=service_account_key['client_id'],
            auth_uri=service_account_key['auth_uri'],
            token_uri=service_account_key['token_uri'],
        )
        credentials: Optional[IdentityCredentials] = (
            GoogleServiceAccountKeyCredentials(
                private_key_id=service_account_key['private_key_id'],
                private_key=service_account_key['private_key'],
            ))
        return authentication, credentials
    if workload_service_account_email is not None:
        return GoogleWorkloadIdentityAuthentication(
            service_account_email=workload_service_account_email), None
    return GoogleAmbientAuthentication(), None


class GCSIdentity(Identity):
    r"""A Google Cloud principal for GCS, authenticated by a service-account
    key, workload identity impersonation, or Kumo's ambient GCP chain.

    Args:
        service_account_key: A parsed service-account key JSON (the dict a
            downloaded key file deserializes to).
        workload_service_account_email: A service account to impersonate via
            workload identity. Mutually exclusive with
            `service_account_key`; omit both for the ambient chain.
    """

    def __init__(
        self,
        *,
        service_account_key: Optional[dict] = None,
        workload_service_account_email: Optional[str] = None,
    ) -> None:
        authentication, credentials = _google_authentication(
            service_account_key=service_account_key,
            workload_service_account_email=workload_service_account_email,
        )
        config = GCSIdentityConfig(authentication=authentication)
        super().__init__(config, credentials)


class BigQueryIdentity(Identity):
    r"""A Google Cloud principal for BigQuery. See `GCSIdentity`."""

    def __init__(
        self,
        *,
        service_account_key: Optional[dict] = None,
        workload_service_account_email: Optional[str] = None,
    ) -> None:
        authentication, credentials = _google_authentication(
            service_account_key=service_account_key,
            workload_service_account_email=workload_service_account_email,
        )
        config = BigQueryIdentityConfig(authentication=authentication)
        super().__init__(config, credentials)


class DatabricksIdentity(Identity):
    r"""A Databricks principal, authenticated by a service principal's OAuth
    client ID/secret, or by Databricks' own ambient auth-resolution chain.

    Specify `host`/`client_id`/`client_secret` together for OAuth
    service-principal auth. Specify none of them for the ambient chain --
    the same chain a bare `databricks.sdk.WorkspaceClient()` resolves,
    which a notebook's attached execution context satisfies on its own.
    """

    def __init__(
        self,
        *,
        host: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> None:
        explicit = (host, client_id, client_secret)
        if any(value is not None for value in explicit) and not all(
                value is not None for value in explicit):
            raise ValueError(
                "host, client_id, and client_secret must be given together, "
                "or all omitted to defer to the ambient auth chain.")
        credentials: Optional[IdentityCredentials]
        if host is not None:
            authentication = DatabricksAuthenticationKind.OAUTH
            credentials = DatabricksOAuthCredentials(
                client_secret=client_secret)
        else:
            authentication = DatabricksAuthenticationKind.AMBIENT
            credentials = None
        config = DatabricksIdentityConfig(
            host=host,
            client_id=client_id,
            authentication=authentication,
        )
        super().__init__(config, credentials)
