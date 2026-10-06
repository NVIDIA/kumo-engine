# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-request authorization context variable.

The FastAPI middleware sets ``_current_user_ctx`` after JWT verification and
group resolution. StateDB-backed services read it where authorization needs
the caller's identity without threading a ``ctx`` parameter through every
method signature.

Usage (middleware side)::

    from rest.deps.context import _current_user_ctx
    _current_user_ctx.set(user_authorization)

Usage (service side)::

    from rest.deps.context import _current_user_ctx
    self._user_ctx = _current_user_ctx.get(None)
"""
import contextvars
from typing import Optional

from statedb.db.auth import AuthPrincipal

# Per-request context variable, set by FastAPI middleware before call_next.
# Defaults to None so background tasks / tests without middleware see no user.
# Typed against the AuthPrincipal protocol (which UserAuthorization satisfies)
# so this stays a leaf module and does not depend on rest.deps.authentication.
_current_user_ctx: contextvars.ContextVar[
    Optional[AuthPrincipal]] = contextvars.ContextVar(
        "_current_user_ctx", default=None)
