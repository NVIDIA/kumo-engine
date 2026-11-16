# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge value types used by graph materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

PathDict: TypeAlias = dict[str, str]


@dataclass
class EdgeKey:
    src_table: str
    link_key: str
    dst_table: str
    is_reverse: bool = False

    _REV_EDGE_KEY_PREFIX = "rev_"

    def __post_init__(self) -> None:
        if self.is_reverse:
            if self.link_key.startswith(self._REV_EDGE_KEY_PREFIX):
                raise ValueError(
                    "Reverse edge link key must not include the rev_ prefix")
            self.link_key = self._REV_EDGE_KEY_PREFIX + self.link_key

    def reverse(self) -> "EdgeKey":
        if self.is_reverse:
            return EdgeKey(
                self.dst_table,
                self.link_key.removeprefix(self._REV_EDGE_KEY_PREFIX),
                self.src_table,
            )
        return EdgeKey(self.dst_table, self.link_key, self.src_table, True)

    def to_str(self) -> str:
        return "__".join((self.src_table, self.link_key, self.dst_table))

    def to_tuple(self) -> tuple[str, str, str]:
        return self.src_table, self.link_key, self.dst_table


__all__ = ["EdgeKey", "PathDict"]
