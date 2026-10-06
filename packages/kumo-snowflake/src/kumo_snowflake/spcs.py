# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What Snowpark Container Services puts in a running container.

A container gets an OAuth token on its filesystem and the account and endpoint
it belongs to in its environment. An identity that authenticates this way
therefore stores no secret at all -- only where to read the token, which is why
it pairs with no credentials.

The arguments are *built* here rather than the identity itself: constructing
one means calling ``kumoai_studio.create_identity``, and this package does not import
the SDK. The caller passes what comes back:

    import kumoai_studio
    from kumo_snowflake import spcs_identity_create_args

    identity = kumoai_studio.create_identity(spcs_identity_create_args('prod-sf'))
"""

import os

from kumoapi.data_source import (
    CreateIdentityArgs,
    SnowflakeAuthenticationKind,
    SnowflakeIdentityConfig,
)

#: Where the container runtime mounts the OAuth token.
SPCS_TOKEN_PATH = "/snowflake/session/token"

#: What the runtime exports. Read rather than accepted from a caller: this
#: helper describes the container it is running in, and a value passed in
#: would let it describe something else while still claiming to be SPCS.
ACCOUNT_ENV = "SNOWFLAKE_ACCOUNT"
HOST_ENV = "SNOWFLAKE_HOST"


def spcs_identity_create_args(
    display_name: str,
    *,
    description: str = "",
) -> CreateIdentityArgs:
    """Describes the identity this container already has.

    Everything but the name comes from the runtime, and none of it is
    overridable: this function answers "create the identity I am already
    running as", and a caller who wants to describe a *different* principal
    wants :class:`~kumoapi.data_source.CreateIdentityArgs` directly rather than
    a container helper that has stopped describing the container.

    ``token_path`` is deliberately left unset, so the reader resolves the
    mounted path when it connects. Storing it would freeze today's mount point
    into every identity created.

    Args:
        display_name: The name connectors and saved graphs will refer to.
        description: Free text stored alongside the identity.

    Raises:
        ValueError: If the mounted token is absent, or the environment names
            no account -- either of which means this is not a container.
    """
    if not os.path.exists(SPCS_TOKEN_PATH):
        raise ValueError(
            f"no snowflake token at {SPCS_TOKEN_PATH}, so this is not a "
            f"snowpark container services environment. This identity reads a "
            f"token the container runtime mounts, and there is none to read; "
            f"create a key-pair or password identity instead")
    account = os.environ.get(ACCOUNT_ENV)
    if not account:
        raise ValueError(
            f"the snowflake token at {SPCS_TOKEN_PATH} is mounted but "
            f"{ACCOUNT_ENV} is unset, so there is no account to authenticate "
            f"against. A container runtime sets both; one without the other "
            f"is an environment this cannot describe")
    return CreateIdentityArgs(
        display_name=display_name,
        description=description,
        config=SnowflakeIdentityConfig(
            account=account,
            authentication=SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE,
            host=os.environ.get(HOST_ENV) or None,
        ),
    )
