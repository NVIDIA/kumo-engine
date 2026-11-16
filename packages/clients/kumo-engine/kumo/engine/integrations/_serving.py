# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""What MLflow and Snowflake serving share: the query and snapshot."""

from typing import Any, Tuple

from kumo.engine.connector.base import Connector
from kumo.engine.graph import GraphSnapshot


def parsed_query(trainer: Any, snapshot: GraphSnapshot) -> Any:
    from kumo.engine.pquery import PredictiveQuery

    return PredictiveQuery._from_snapshot(snapshot,
                                          trainer._training_query)._parsed


def snapshot_location(uri: str) -> Tuple[Connector, str]:
    from kumo.engine.connector import connector_for_root

    parent, slash, name = uri.rstrip('/').rpartition('/')
    return connector_for_root(parent or slash or '.'), name
