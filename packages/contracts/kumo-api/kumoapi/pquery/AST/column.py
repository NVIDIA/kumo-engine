# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

from pydantic.dataclasses import dataclass

from kumoapi.pquery.AST.ast_node import ASTNode
from kumoapi.pquery.AST.location_interval import ASTQueryLocationInterval
from kumoapi.typing import _FORBID_EXTRA_CONFIG


@dataclass(repr=False, config=_FORBID_EXTRA_CONFIG)
class Column(ASTNode):
    r"""Creates an atomic description of a column :obj:`fqn`.
    Args:
        fqn: A fully-qualified name in the "table.column" format.
    """

    fqn: str = ''

    def __post_init__(self) -> None:
        if len(self.fqn.split('.')) != 2:
            raise ValueError(
                f'The column name {self.fqn} was not given in its '
                f'fully-qualified name form. Format it as "table.column".')

        super().__post_init__()

    @property
    def all_query_columns_with_locations(
            self) -> List[Tuple[str, ASTQueryLocationInterval]]:
        assert self.location is not None
        return [(self.fqn, self.location)]

    @property
    def all_query_columns(self) -> List[str]:
        return [self.fqn]

    def to_string(self, rich: bool = False) -> str:
        return self.fqn
