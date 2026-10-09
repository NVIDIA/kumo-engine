# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import kumoapi.data_source as api
import pytest

from rest.deps.authentication import UserAuthorization
from statedb.config.common import IdentityID, UserID
from statedb.db import StateDB
from statedb.db.connector_authorizations import ConnectorAuthorizationService
from statedb.db.groups import GroupService
from statedb.db.identities import IdentityService
from statedb.db.services import ProjectService
from statedb.rest.group import KumoGroupCreateSpec
from statedb.utils.errors import (
    GroupNotFoundError,
    IdentityDoesNotExistError,
    ProjectNotFoundError,
    ResourceSlugConflictError,
)

ADMIN = UserAuthorization.disabled()


def _new_identity_service(db: StateDB, **kwargs) -> IdentityService:
    return IdentityService(db, ConnectorAuthorizationService(db), **kwargs)


def _member(user_id: str, *group_ids) -> UserAuthorization:
    return UserAuthorization(
        user_id=UserID(user_id),
        access_token='token',
        permissions=(),
        kumo_group_ids=tuple(group_ids),
    )


def _identity_args(
    display_name: str,
    role_name: str,
    *,
    slug: str | None = None,
    group_ids: list[str] | None = None,
) -> api.CreateIdentityArgs:
    return api.CreateIdentityArgs(
        display_name=display_name,
        slug=slug,
        group_ids=group_ids or [],
        config=api.S3IdentityConfig(
            authentication=api.S3AssumeRoleAuthentication(
                partition='aws',
                role_arn=f'arn:aws:iam::123456789012:role/{role_name}',
            ), ),
    )


@pytest.mark.asyncio
async def test_explicit_derived_and_bootstrap_slugs(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    groups = GroupService(db)
    projects = ProjectService(db)
    identities = _new_identity_service(db, identity_validator=lambda *_: None)
    try:
        default_group, default_project = await groups.ensure_default_resources()
        assert default_group.slug == api.DEFAULT_GROUP_SLUG == 'default'
        assert default_project.slug == api.DEFAULT_PROJECT_SLUG == 'default'

        explicit_group, derived_group = await groups.create_kumo_groups([
            KumoGroupCreateSpec(display_name='Risk Team', slug='risk-team'),
            KumoGroupCreateSpec(display_name='Ｆｒａｕｄ & Data_Platform'),
        ], ADMIN)
        assert explicit_group.slug == 'risk-team'
        assert derived_group.slug == 'fraud-data-platform'

        explicit_project = await projects.create_project(
            name='Fraud Detection',
            slug='fraud-detection',
            group_id=str(explicit_group.id),
            user_authorization=ADMIN,
        )
        derived_project = await projects.create_project(
            name='Loss / Prevention',
            group_id=str(explicit_group.id),
            user_authorization=ADMIN,
        )
        assert explicit_project.slug == 'fraud-detection'
        assert derived_project.slug == 'loss-prevention'

        explicit_identity = await identities.create(
            _identity_args('Production AWS', 'production', slug='prod-aws'),
            ADMIN,
        )
        derived_identity = await identities.create(
            _identity_args('Analytics & Reporting', 'analytics'),
            ADMIN,
        )
        assert explicit_identity.slug == 'prod-aws'
        assert derived_identity.slug == 'analytics-reporting'

        with pytest.raises(ValueError, match='Reserved resource slug'):
            await projects.create_project(
                name='Reserved',
                slug='default',
                group_id=str(explicit_group.id),
                user_authorization=ADMIN,
            )
        with pytest.raises(ValueError, match='Slug must be'):
            await identities.create(
                _identity_args('Invalid slug', 'invalid', slug='INVALID'),
                ADMIN,
            )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_slugs_are_immutable_and_deleted_identity_slug_is_reusable(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    groups = GroupService(db)
    projects = ProjectService(db)
    identities = _new_identity_service(db, identity_validator=lambda *_: None)
    try:
        group = (await groups.create_kumo_groups([
            KumoGroupCreateSpec(display_name='Original Group',
                                slug='stable-group')
        ], ADMIN))[0]
        updated_group = await groups.update_kumo_group(
            group.id, display_name='Renamed Group', principal=ADMIN)
        assert updated_group.slug == 'stable-group'
        await groups.delete_kumo_group(str(group.id), ADMIN)
        with pytest.raises(ResourceSlugConflictError,
                           match="Group slug 'stable-group'"):
            await groups.create_kumo_groups([
                KumoGroupCreateSpec(display_name='Replacement Group',
                                    slug='stable-group')
            ], ADMIN)

        owner_group = (await groups.create_kumo_groups([
            KumoGroupCreateSpec(display_name='Owner Group', slug='owner-group')
        ], ADMIN))[0]
        project = await projects.create_project(
            name='Original Project',
            slug='stable-project',
            group_id=str(owner_group.id),
            user_authorization=ADMIN,
        )
        renamed_project = await projects.update_project(
            project.id, name='Renamed Project', user_authorization=ADMIN)
        assert renamed_project.slug == 'stable-project'
        await projects.archive_project(project.id, ADMIN)
        with pytest.raises(ResourceSlugConflictError,
                           match="Project slug 'stable-project'"):
            await projects.create_project(
                name='Replacement Project',
                slug='stable-project',
                group_id=str(owner_group.id),
                user_authorization=ADMIN,
            )

        identity = await identities.create(
            _identity_args('Original Identity', 'original',
                           slug='stable-identity'),
            ADMIN,
        )
        renamed_identity = await identities.update_metadata(
            identity.id,
            api.UpdateIdentityArgs(display_name='Renamed Identity'),
            ADMIN,
        )
        assert renamed_identity.slug == 'stable-identity'
        replacement_args = _identity_args('Replacement Identity', 'replacement',
                                          slug='stable-identity')
        with pytest.raises(ResourceSlugConflictError,
                           match="Identity slug 'stable-identity'"):
            await identities.create(replacement_args, ADMIN)
        await identities.delete(identity.id, ADMIN)
        with pytest.raises(IdentityDoesNotExistError):
            await identities.get_admin(identity.id, ADMIN)
        replacement = await identities.create(replacement_args, ADMIN)
        assert replacement.id != identity.id
        assert replacement.slug == 'stable-identity'
        active_identities = await identities.list_admin(ADMIN)
        assert [row.id for row in active_identities] == [replacement.id]
        with pytest.raises(ResourceSlugConflictError,
                           match="Identity slug 'stable-identity'"):
            await identities.create(
                _identity_args('Duplicate Identity', 'duplicate',
                               slug='stable-identity'),
                ADMIN,
            )
        with pytest.raises(IdentityDoesNotExistError):
            await identities.get_admin(identity.id, ADMIN)
        async with db.read_tx() as tx:
            tombstone = await identities._identities.get(tx, identity.id)
        assert tombstone is not None
        assert tombstone.payload.slug == 'stable-identity'
        assert tombstone.payload.encrypted_credentials is None
        assert tombstone.payload.tombstoned_at is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_slug_lookup_is_authorized_and_identity_is_project_filtered(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    groups = GroupService(db)
    projects = ProjectService(db)
    identities = _new_identity_service(db, identity_validator=lambda *_: None)
    try:
        group_a, group_b = await groups.create_kumo_groups([
            KumoGroupCreateSpec(display_name='Team Alpha', slug='team-alpha'),
            KumoGroupCreateSpec(display_name='Team Beta', slug='team-beta'),
        ], ADMIN)
        alpha = _member('alpha-user', group_a.id)
        beta = _member('beta-user', group_b.id)
        assert (await groups.get_kumo_group_by_slug('team-alpha',
                                                    alpha)).id == group_a.id
        assert (await groups.get_kumo_group(str(group_a.id),
                                            alpha)).id == group_a.id
        with pytest.raises(GroupNotFoundError):
            await groups.get_kumo_group_by_slug('team-alpha', beta)
        with pytest.raises(GroupNotFoundError):
            await groups.get_kumo_group(str(group_a.id), beta)

        project_a = await projects.create_project(
            name='Alpha Project',
            slug='alpha-project',
            group_id=str(group_a.id),
            user_authorization=ADMIN,
        )
        project_b = await projects.create_project(
            name='Beta Project',
            slug='beta-project',
            group_id=str(group_b.id),
            user_authorization=ADMIN,
        )
        assert (await projects.get_project_by_slug('alpha-project',
                                                   alpha)).id == project_a.id
        assert (await projects.get_project(project_a.id,
                                           alpha)).id == project_a.id
        with pytest.raises(ProjectNotFoundError):
            await projects.get_project_by_slug('alpha-project', beta)
        with pytest.raises(ProjectNotFoundError):
            await projects.get_project(project_a.id, beta)

        identity = await identities.create(
            _identity_args('Alpha AWS', 'alpha', slug='alpha-aws',
                           group_ids=[str(group_a.id)]),
            ADMIN,
        )
        resolved = await identities.available_for_project_by_slug(
            project_a.id, 'alpha-aws', alpha)
        assert resolved.id == identity.id
        assert (await
                identities.available_for_project_by_id(project_a.id,
                                                       identity.id,
                                                       alpha)).id == identity.id
        with pytest.raises(IdentityDoesNotExistError):
            await identities.available_for_project_by_slug(
                project_b.id, 'alpha-aws', beta)
        with pytest.raises(IdentityDoesNotExistError):
            await identities.available_for_project_by_id(
                project_b.id, identity.id, beta)
        with pytest.raises(IdentityDoesNotExistError):
            await identities.available_for_project_by_slug(
                project_a.id, 'missing-identity', alpha)
        with pytest.raises(IdentityDoesNotExistError):
            await identities.available_for_project_by_id(
                project_a.id, IdentityID.generate(), alpha)
        with pytest.raises(ProjectNotFoundError):
            await identities.available_for_project_by_slug(
                project_a.id, 'alpha-aws', beta)
        with pytest.raises(ProjectNotFoundError):
            await identities.available_for_project_by_id(
                project_a.id, identity.id, beta)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sqlite_concurrent_same_slug_has_exactly_one_winner(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'), pool_size=4)
    groups = GroupService(db)
    projects = ProjectService(db)
    identities = _new_identity_service(db, identity_validator=lambda *_: None)
    try:
        group_results = await asyncio.gather(
            groups.create_kumo_groups([
                KumoGroupCreateSpec(display_name='Concurrent Group One',
                                    slug='shared-group')
            ], ADMIN),
            groups.create_kumo_groups([
                KumoGroupCreateSpec(display_name='Concurrent Group Two',
                                    slug='shared-group')
            ], ADMIN),
            return_exceptions=True,
        )
        assert sum(isinstance(result, list) for result in group_results) == 1
        assert sum(
            isinstance(result, ResourceSlugConflictError)
            for result in group_results) == 1
        winner_group = next(result[0] for result in group_results
                            if isinstance(result, list))

        project_results = await asyncio.gather(
            projects.create_project(
                name='Concurrent Project One',
                slug='shared-project',
                group_id=str(winner_group.id),
                user_authorization=ADMIN,
            ),
            projects.create_project(
                name='Concurrent Project Two',
                slug='shared-project',
                group_id=str(winner_group.id),
                user_authorization=ADMIN,
            ),
            return_exceptions=True,
        )
        assert sum(hasattr(result, 'id') for result in project_results) == 1
        assert sum(
            isinstance(result, ResourceSlugConflictError)
            for result in project_results) == 1

        identity_results = await asyncio.gather(
            identities.create(
                _identity_args('Concurrent Identity One', 'concurrent-one',
                               slug='shared-identity'),
                ADMIN,
            ),
            identities.create(
                _identity_args('Concurrent Identity Two', 'concurrent-two',
                               slug='shared-identity'),
                ADMIN,
            ),
            return_exceptions=True,
        )
        assert sum(
            isinstance(result, api.IdentityResponse)
            for result in identity_results) == 1
        assert sum(
            isinstance(result, ResourceSlugConflictError)
            for result in identity_results) == 1
    finally:
        await db.close()
