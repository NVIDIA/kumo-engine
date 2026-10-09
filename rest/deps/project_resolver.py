# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from statedb.db.auth import AuthPrincipal
from statedb.db.services import StateServices


class ProjectResolver:
    """Resolve visible projects from current principal and StateDB state.

    External identity-provider results are cached by the group-membership
    adapter. Project ownership, project status, native membership, and live
    external-to-Kumo mappings are deliberately evaluated on every request.
    """
    def __init__(self, state_services: Optional[StateServices] = None):
        self._state_services = state_services

    def configure_state_services(self, state_services: StateServices) -> None:
        self._state_services = state_services

    async def resolve_projects(
            self, user_authorization: AuthPrincipal) -> List[str]:
        if self._state_services is None:
            raise RuntimeError(
                "ProjectResolver StateServices is not configured")
        return await self._state_services.projects.list_visible_project_ids(
            user_authorization)
