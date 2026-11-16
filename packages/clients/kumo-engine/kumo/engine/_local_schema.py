# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local graph schemas, without exposing filesystem sources to hosted APIs."""

from dataclasses import dataclass as plain_dataclass
from typing import Dict, Union

from kumoapi.graph import GraphDefinition as HostedGraphDefinition
from kumoapi.source_table import LocalFileSourceTable, SourceTableType
from kumoapi.table import TableDefinition as HostedTableDefinition
from kumoapi.typing import _FORBID_EXTRA_CONFIG
from pydantic import Field
from pydantic.dataclasses import dataclass

LocalSourceTableType = Union[SourceTableType, LocalFileSourceTable]


@dataclass(config=_FORBID_EXTRA_CONFIG)
class TableDefinition(HostedTableDefinition):
    source_table: LocalSourceTableType = Field(discriminator='data_source_type')


@dataclass(config=_FORBID_EXTRA_CONFIG)
class GraphDefinition(HostedGraphDefinition):
    tables: Dict[str, Union[TableDefinition, HostedTableDefinition]]


@plain_dataclass(frozen=True)
class FileConnectorConfig:
    root_dir: str
