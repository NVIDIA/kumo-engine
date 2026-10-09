# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Dict, Literal, Optional, Tuple, Union

from pydantic.dataclasses import dataclass

from kumoapi.typing import _FORBID_EXTRA_CONFIG, TimeUnit

Numeric = Union[float, int]
TimeMetric = Dict[str, Tuple[Numeric, TimeUnit]]


@dataclass(config=_FORBID_EXTRA_CONFIG)
class AggregationConfig:
    r"""Configuration for aggregation request, which is used to
    aggregate the values of specified column for a given table.

    Args:
        table (str): The name of the table.
        column (str): The name of the column.
        op (str): The operation to apply to aggregate the columns.
    """
    table: str
    column: str
    op: Literal[
        "sum",
        "mean",
        "max",
        "min",
        "most_frequent",
        "least_frequent",
    ]


@dataclass
class BaseRFM:
    r"""Base class for RFM.

    Args:
        entity_table_name (str, optional): The name of the entity table.
        count (Dict[str, int]): The number of items for each entity.
        avg_rel_time (Dict[str, Tuple[float, TimeUnit]]): The average
            relative time in days for each one-hop item.
        most_recent_rel_time (Dict[str, Tuple[int, TimeUnit]]): The most
            recent (smallest) relative time in days for each one-hop item.
    """
    entity_table_name: Optional[str] = None
    count: Dict[str, Numeric] = field(default_factory=dict)
    avg_rel_time: TimeMetric = field(default_factory=dict)
    most_recent_rel_time: TimeMetric = field(default_factory=dict)


@dataclass
class NodePredRFM(BaseRFM):
    r"""RFM for node prediction.

    Args:
        unique_counts (Dict[str, Dict[str, int]]): The number of unique
            second-hop items for each second hop table.
    """
    unique_counts: Dict[str, Dict[str, Numeric]] = field(default_factory=dict)


@dataclass
class LinkPredRFM(BaseRFM):
    r"""RFM for link prediction.

    Args:
        num_hist_items (int, optional): The number of historical items for
            each entity.
        num_pred_hist_items (int, optional): The number of predicted and
            historical overlapped items for each entity.
        num_true_items (int, optional): The number of truth items for each
            entity.
        num_true_hist_items (int, optional): The number of true and historical
            overlapped items for each entity.
    """
    num_hist_items: Optional[Numeric] = None
    num_pred_hist_items: Optional[Numeric] = None
    num_true_items: Optional[Numeric] = None
    num_true_hist_items: Optional[Numeric] = None
