# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

from kumoapi.data_source import ResourceSlug

from io_utils.dataclasses import kumo_dataclass
from statedb.config.common import DBEntry, GroupID, ProjectID


class ProjectStatus(str, Enum):
    """Lifecycle states for a Project."""

    ACTIVE = "active"
    ARCHIVED = "archived"


@kumo_dataclass
class ProjectDBEntry(DBEntry):
    """A scoped workspace referencing tables and graphs.

    A project always belongs to exactly one immutable Kumo group. Visibility within
    that group is controlled by is_private:
    - Private (is_private=True): owner-only.
    - Shared (is_private=False): visible to all members of the group.

    Every connector, table, graph, job, and upload belongs exclusively to this
    project and cannot be linked into another project.
    """

    id: ProjectID  # Auto-generated UUID if not provided
    group_id: GroupID
    slug: ResourceSlug
    display_name: str = ""
    normalized_display_name: str = ""
    created_by: str = ""  # user_id of creator/owner
    created_by_name: str = ""  # display name of creator/owner
    description: str = ""
    is_private: bool = False  # True = owner-only; False = shared with group
    status: ProjectStatus = ProjectStatus.ACTIVE

    def __post_init__(self):
        if not self.id:
            self.id = ProjectID.generate()
