# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kumoapi.task import TaskType
from kumoapi.typing import Stype

# Configuration for representing nested subgraphs, divided into three
# components:
# |- ColumnData: Representation of a column value.
# |- LinkData: Representation of a link to another table.
#    |- NodeData: Representation of a linked node.
#    |- ResidualData: Representation of residual nodes
#                     (in case the graph is very dense)
#
# Consider the classic use-case of a USER->TRANSACTION->STORE graph.
# Then, a specific instance of a subgraph would look as follows:

# Subgraph:
#   id: 0
#   id_name: USER_ID
#   table_name: USER
#   cols:                              <- User columns:
#   - name: AGE
#     value: 33
#   - name: GENDER
#     value: FEMALE
#   links:
#   - table_name: TRANSACTION          <- User link to transaction table:
#     fkey_name: USER_ID
#     nodes:
#     - id: 1                          <- First transaction:
#       cols:
#       - name: AMOUNT
#         value: 10.0
#       links:
#       - table_name: STORE            <- Transaction link to store table:
#         fkey_name: STORE_ID
#         nodes:
#         - id: 3
#           cols:
#           - name: PRIMARY_TAG
#             value: American
#     - id: 2                          <- Second transaction:
#       cols:
#       - name: AMOUNT
#         value: 40.0
#       links:
#       - table_name: STORE            <- Transaction link to store table:
#         fkey_name: STORE_ID
#         nodes:
#         - id: 4
#           cols:
#           - name: PRIMARY_TAG
#             value: American
#     residual:                        <- Remaining transactions:
#       count: 15
#       start_time: 2022-05-01
#       end_time: 2023-04-22


@dataclass
class ColumnData:
    name: str
    value: Any
    stype: Stype
    grad_score: float | None = None


@dataclass
class NodeData:
    cols: list[ColumnData]
    links: list['LinkData']
    # TODO (zecheng): Deprecate id and id_name
    id: Any | None = None
    id_name: str | None = None

    pkey_name: str | None = None
    pkey: Any | None = None
    fkey_name: str | None = None
    fkey: Any | None = None


@dataclass
class ResidualData:
    count: int
    start_time: datetime | None = None
    end_time: datetime | None = None


@dataclass
class LinkData:
    table_name: str
    fkey_name: str
    nodes: list[NodeData]
    residual: ResidualData | None = None


@dataclass
class Subgraph:
    id: Any
    table_name: str | None
    task_type: TaskType | None
    id_name: str | None = None
    seed_time: datetime | None = None
    cols: list[ColumnData] = field(default_factory=list)
    links: list[LinkData] = field(default_factory=list)


@dataclass
class NodeOutputData:
    id: Any
    cols: list[ColumnData]
    id_name: str | None = None


@dataclass
class EntityOutput:
    task_type: TaskType | None
    # * Binary classification/regression => `float`
    # * Multi-{class|label} classification => `dict[str, float]`
    # * Link prediction (top-k predictions) => list[Any]
    prediction: float | dict[str, float] | list[NodeOutputData] | None = None
    target: float | dict[str, float] | list[NodeOutputData] | None = None


@dataclass
class HistoricalItems:
    nodes: list[NodeOutputData] = field(default_factory=list)
