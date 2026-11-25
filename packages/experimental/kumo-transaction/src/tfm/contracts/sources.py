# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a data adapter supplies so a token cache can be built from its book."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Protocol

from tfm.contracts.tokenizers import EventBlock


class PlannedBlock(Protocol):
    """A block of whole users, as planned; sent to worker processes, so it must pickle."""

    @property
    def index(self) -> int:
        """Its position in `BlockPlan.blocks`."""
        ...

    @property
    def n_rows(self) -> int:
        """Book rows it spans, its users' and any others'."""
        ...


class BlockPlan[BlockT: PlannedBlock](Protocol):
    """A book's blocks, in entity order."""

    @property
    def blocks(self) -> Sequence[BlockT]: ...

    @property
    def n_users(self) -> int: ...

    @property
    def n_rows(self) -> int: ...

    @property
    def longest_user_id(self) -> int:
        """The longest planned user id, in UTF-8 bytes."""
        ...


class ColumnarSource[BlockT: PlannedBlock](Protocol):
    """A book read in blocks of whole users, ascending by user id.

    Sent to worker processes, so it must pickle. `str(source)` names the book in errors.
    """

    def fingerprint(self) -> dict[str, Any]:
        """Enough to tell a cache it was built from another book; joins the cache's identity."""
        ...

    def plan(
        self, *, rows_per_block: int, users: Collection[str] | None = None
    ) -> BlockPlan[BlockT]:
        """Blocks of at most `rows_per_block` events that never split a user;
        with `users`, only theirs."""
        ...

    def read_block(self, block: BlockT, *, users: Collection[str] | None = None) -> EventBlock:
        """`block`'s events, of `users` only when given."""
        ...

    def write_profiles(self, users: Collection[str] | None, out: Path) -> bool:
        """Write the profiles of `users` (all when None) to `out` as a pragmatiq
        `profiles.parquet` sorted by `user_id`; False, writing nothing, when the book
        has none."""
        ...


__all__ = ["BlockPlan", "ColumnarSource", "PlannedBlock"]
