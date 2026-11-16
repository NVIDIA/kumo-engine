# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io_utils.dataclasses import kumo_dataclass
from statedb.config.common import (
    ArtifactExportJobID,
    DBEntry,
    ProjectID,
    UserID,
)


@kumo_dataclass
class ArtifactExportJobDBEntry(DBEntry):
    """Immutable request identity for one project-owned Temporal workflow."""

    id: ArtifactExportJobID
    project_id: ProjectID
    created_by: UserID
    request_id: str
    request_digest: str
    dispatch_sealed: bool = False
    cancel_requested: bool = False
