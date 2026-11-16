# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Protocol, Tuple

from statedb.config.common import UserID


class AuthPrincipal(Protocol):
    """Minimal authorization context consumed by statedb services.

    This is the statedb-native abstraction for "who is making this request".
    ``rest.deps.authentication.UserAuthorization`` satisfies it structurally
    (it already exposes all four members), so the rest layer can keep passing
    its richer object down without statedb importing from rest -- breaking the
    ``statedb -> rest`` layering inversion.

    Members are declared as read-only properties so that frozen dataclasses
    (``UserAuthorization`` is a frozen ``kumo_dataclass`` whose ``is_admin`` is
    itself a property) are accepted: a read-only protocol member matches both a
    frozen field and a property, whereas a plain attribute would require it to
    be writable and fail the assignment.
    """
    @property
    def user_id(self) -> UserID:
        ...

    @property
    def user_name(self) -> str:
        ...

    @property
    def kumo_group_ids(self) -> Tuple[str, ...]:
        ...

    @property
    def external_membership_unavailable(self) -> bool:
        ...

    @property
    def is_admin(self) -> bool:
        ...
