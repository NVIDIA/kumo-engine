# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional

import kumoapi.table as table_api
from kumo_core.types.schema import FPID
from kumoapi.typing import Dtype

from io_utils import kumo_dataclass
from statedb.config.common import DBEntry, ProjectID
from statedb.rest.resource_alias import normalize_resource_alias


@kumo_dataclass
class TableDBEntryV2(DBEntry):
    id: FPID
    project_id: ProjectID

    config: table_api.TableDefinition

    # Historical: column dtype information inferred at table creation time.
    # ``config.cols[].dtype`` is the source of truth for current entries.
    col_dtypes: Dict[str, Dtype]

    # Human-friendly name for the table template.
    alias: Optional[str] = None
    normalized_alias: str = ""

    # Ref counting is used for V2 table to track how many PQueries (v2) are
    # using the same table.
    ref_count: int = 1

    # Stable principal ID that created this project-owned table.
    created_by: str = ""
    created_by_name: str = ""

    def __post_init__(self) -> None:
        self.normalized_alias = (normalize_resource_alias(self.alias)
                                 if self.alias else '')
