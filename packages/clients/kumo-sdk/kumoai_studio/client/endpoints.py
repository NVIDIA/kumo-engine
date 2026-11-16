# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Optional


class HTTPMethod(Enum):
    r"""HTTP methods supported by the API."""
    GET = "GET"
    POST = "POST"
    DELETE = "DELETE"
    PATCH = "PATCH"
    PUT = "PUT"


@dataclass(frozen=True)
class Endpoint:
    r"""Represents an API endpoint with its path and HTTP method."""
    path: Optional[str] = field(default=None)
    method: HTTPMethod = HTTPMethod.GET

    def validate(self) -> None:
        pass

    def get_path(self) -> str:
        if self.path is None:
            # This should be in validate but is here for type checker
            raise ValueError("Endpoint requires a path")
        return self.path


@dataclass(frozen=True)
class IDEndpoint(Endpoint):
    r"""Represents an API endpoint with an additional id/name in its path."""
    template_path: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        if self.template_path is None:
            raise ValueError("template_path must be set explicitly")
        if "{id}" not in self.template_path:
            raise ValueError("IDEndpoint path must contain '{{id}}': "
                             f"got '{self.template_path}'")

    def with_id(self, resource_id: str) -> 'IDEndpoint':
        assert self.template_path is not None
        return IDEndpoint(template_path=self.template_path,
                          path=self.template_path.format(id=resource_id),
                          method=self.method)

    def validate(self) -> None:
        if self.path is None:
            raise ValueError(
                "IDEndpoint requires with_id() to be called with a resource id"
            )

    @classmethod
    def from_base(cls, base: str, method: HTTPMethod) -> 'IDEndpoint':
        r"""Alternate constructor to build a template_path by appending /{id}
        to base.
        """
        template = f"{base}/{{id}}"
        return cls(template_path=template, method=method)


class ConnectorEndpoints:
    @staticmethod
    def collection(project_id: str, method: HTTPMethod) -> Endpoint:
        return Endpoint(f'/projects/{project_id}/connectors', method)

    @staticmethod
    def item(project_id: str, connector_id: str,
             method: HTTPMethod) -> Endpoint:
        return Endpoint(f'/projects/{project_id}/connectors/{connector_id}',
                        method)

    @staticmethod
    def upload(project_id: str, operation: str,
               method: HTTPMethod) -> Endpoint:
        return Endpoint(
            f'/projects/{project_id}/connectors/uploads/{operation}', method)


class IdentityEndpoints:
    # Project-scoped discovery (any project member).
    @staticmethod
    def list(project_id: str) -> Endpoint:
        return Endpoint(f'/projects/{project_id}/identities', HTTPMethod.GET)

    @staticmethod
    def by_slug(project_id: str, slug: str) -> Endpoint:
        return Endpoint(
            f'/projects/{project_id}/identities/by-slug/{slug}',
            HTTPMethod.GET,
        )

    # Administrator lifecycle (customer-admin API key only).
    ADMIN_BASE: Final[str] = '/identities'

    @staticmethod
    def collection(method: HTTPMethod) -> Endpoint:
        return Endpoint(IdentityEndpoints.ADMIN_BASE, method)

    @staticmethod
    def item(identity_id: str, method: HTTPMethod) -> Endpoint:
        return Endpoint(f'{IdentityEndpoints.ADMIN_BASE}/{identity_id}',
                        method)

    @staticmethod
    def credentials(identity_id: str) -> Endpoint:
        return Endpoint(
            f'{IdentityEndpoints.ADMIN_BASE}/{identity_id}/credentials',
            HTTPMethod.PUT,
        )

    @staticmethod
    def test(identity_id: str) -> Endpoint:
        return Endpoint(f'{IdentityEndpoints.ADMIN_BASE}/{identity_id}/test',
                        HTTPMethod.POST)

    @staticmethod
    def groups(identity_id: str, method: HTTPMethod) -> Endpoint:
        return Endpoint(f'{IdentityEndpoints.ADMIN_BASE}/{identity_id}/groups',
                        method)


class PQueryEndpoints:
    BASE: Final[str] = '/predictive_queries'

    create = Endpoint(BASE, HTTPMethod.POST)
    list = Endpoint(BASE, HTTPMethod.GET)
    get = IDEndpoint.from_base(BASE, HTTPMethod.GET)
    delete = IDEndpoint.from_base(BASE, HTTPMethod.DELETE)

    infer_task_type = Endpoint(f"{BASE}/infer_task_type", HTTPMethod.POST)
    validate = Endpoint(f"{BASE}/validate", HTTPMethod.POST)
    suggest_training_table_plan = Endpoint(
        f"{BASE}/train_table_generation_plan", HTTPMethod.POST)
    suggest_model_plan = Endpoint(f"{BASE}/model_plan", HTTPMethod.POST)


class SourceTableEndpoints:
    @staticmethod
    def base(project_id: str, connector_id: str) -> str:
        return f'/projects/{project_id}/connectors/{connector_id}'

    @classmethod
    def list_tables(cls, project_id: str, connector_id: str) -> Endpoint:
        return Endpoint(
            f'{cls.base(project_id, connector_id)}/source-tables',
            HTTPMethod.GET,
        )

    @classmethod
    def get_table_config(cls, project_id: str, connector_id: str) -> Endpoint:
        return Endpoint(
            f'{cls.base(project_id, connector_id)}/source-table-config',
            HTTPMethod.GET,
        )

    @classmethod
    def get_columns(cls, project_id: str, connector_id: str) -> Endpoint:
        return Endpoint(f'{cls.base(project_id, connector_id)}/columns',
                        HTTPMethod.GET)

    @classmethod
    def get_sample(cls, project_id: str, connector_id: str) -> Endpoint:
        return Endpoint(f'{cls.base(project_id, connector_id)}/sample',
                        HTTPMethod.GET)

    @classmethod
    def _get_download_urls(cls, project_id: str,
                           connector_id: str) -> Endpoint:
        return Endpoint(f'{cls.base(project_id, connector_id)}/download-urls',
                        HTTPMethod.GET)


class GraphEndpoints:
    BASE: Final[str] = "/graphs"
    SNAPSHOTS: Final[str] = "/graphsnapshots"

    create = Endpoint(BASE, HTTPMethod.POST)
    get = IDEndpoint.from_base(BASE, HTTPMethod.GET)

    validate = Endpoint(f"{BASE}/validate", HTTPMethod.POST)
    infer_links = Endpoint(f"{BASE}/infer_links", HTTPMethod.POST)
    create_snapshot = Endpoint(SNAPSHOTS, HTTPMethod.POST)
    get_snapshot = IDEndpoint.from_base(SNAPSHOTS, HTTPMethod.GET)
    get_edge_stats = IDEndpoint.from_base(f"{SNAPSHOTS}/edge_health",
                                          method=HTTPMethod.GET)


class TableEndpoints:
    BASE: Final[str] = "/tables"
    SNAPSHOTS: Final[str] = "/tablesnapshots"

    create = Endpoint(BASE, HTTPMethod.POST)
    get = IDEndpoint.from_base(BASE, HTTPMethod.GET)

    create_snapshot = Endpoint(SNAPSHOTS, HTTPMethod.POST)
    get_snapshot = IDEndpoint.from_base(SNAPSHOTS, HTTPMethod.GET)
    validate = Endpoint(f"{BASE}/validate", HTTPMethod.POST)
    resolve_sql = Endpoint(f"{BASE}/resolve_sql", HTTPMethod.POST)
    infer_metadata = Endpoint(f"{BASE}/infer_metadata", HTTPMethod.POST)


class GroupEndpoints:
    BASE: Final[str] = "/groups"

    list = Endpoint(BASE, HTTPMethod.GET)

    @staticmethod
    def by_slug(slug: str) -> Endpoint:
        return Endpoint(f'{GroupEndpoints.BASE}/by-slug/{slug}',
                        HTTPMethod.GET)


class ProjectEndpoints:
    BASE: Final[str] = "/projects"

    list = Endpoint(BASE, HTTPMethod.GET)
    create = Endpoint(BASE, HTTPMethod.POST)

    @staticmethod
    def get(project_id: str) -> Endpoint:
        return Endpoint(f'{ProjectEndpoints.BASE}/{project_id}',
                        HTTPMethod.GET)

    @staticmethod
    def by_slug(slug: str) -> Endpoint:
        return Endpoint(f'{ProjectEndpoints.BASE}/by-slug/{slug}',
                        HTTPMethod.GET)
