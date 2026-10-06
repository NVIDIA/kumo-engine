# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Secret-free project and group value objects."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Group:
    r"""A credential-free native Kumo authorization group.

    Groups are shown as Teams in the Kumo UI. A group owns projects and is
    bound by administrators to the external identities its projects may use.
    SDK connector operations do not accept a group; the active project
    determines it.

    Args:
        id: Immutable group ID.
        slug: Immutable short ID used in SDK selectors and URLs.
        name: Group display name.
    """
    id: str
    slug: str
    name: str

    def __str__(self) -> str:
        return (f"Group(id={self.id!r}, slug={self.slug!r}, "
                f"name={self.name!r})")


@dataclass(frozen=True)
class Project:
    r"""A Kumo project, belonging to a group.

    Projects are created by users within their group. All objects (graphs,
    tables, predictive queries) and jobs belong to a project. Objects cannot
    be shared across projects.

    Args:
        id: Immutable project ID.
        slug: Immutable short ID used in SDK selectors and URLs.
        name: Globally unique project display name.
        group_id: Immutable ID of the group that owns the project.
    """
    id: str
    slug: str
    name: str
    group_id: str

    def __str__(self) -> str:
        return (f"Project(id={self.id!r}, slug={self.slug!r}, "
                f"name={self.name!r})")
