# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional

import kumoapi.graph as graph_api
from kumo_core.types.schema import FPID

from io_utils import kumo_dataclass
from statedb.config.common import DBEntry, ProjectID, TableName
from statedb.rest.resource_alias import normalize_resource_alias


@kumo_dataclass
class GraphDBEntryV2(DBEntry):
    id: FPID
    project_id: ProjectID

    table_name_to_dbid: Dict[TableName, FPID]
    col_groups: List[graph_api.ColumnKeyGroup]
    description: str = ''
    alias: Optional[str] = None
    normalized_alias: str = ''

    # Ref counting is used for V2 graph to track how many PQueries (v2) are
    # using the same graph.
    ref_count: int = 1
    created_by: str = ""
    created_by_name: str = ""

    def __post_init__(self) -> None:
        self.normalized_alias = (normalize_resource_alias(self.alias)
                                 if self.alias else '')
