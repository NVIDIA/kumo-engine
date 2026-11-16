# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
import pytest_asyncio
from kumoapi.data_source import DEFAULT_GROUP_ID, DEFAULT_PROJECT_ID

from rest.config.encryption import decrypt_idp_id
from rest.deps.authentication import UserAuthorization
from statedb.config.common import (
    GroupID,
    IdentityGroupID,
    IdentityID,
    ProjectID,
    UserID,
)
from statedb.config.timestamp import get_current_timestamps
from statedb.db import StateDB
from statedb.db.groups import GroupService
from statedb.db.services import ProjectService
from statedb.rest.group import (
    GroupMembershipRuleType,
    GroupStatus,
    KumoGroupCreateSpec,
    KumoGroupDBEntry,
    PrincipalType,
)
from statedb.rest.identity import IdentityGroupEntry
from statedb.rest.project import ProjectStatus
from statedb.utils.errors import (
    GroupAlreadyExists,
    GroupInputError,
    GroupNotFoundError,
    GroupRelationConflictError,
    ProjectForbiddenError,
)

ADMIN = UserAuthorization.disabled()
MEMBER = UserAuthorization(
    user_id=UserID('member'),
    access_token='token',
    permissions=(),
)


@pytest_asyncio.fixture
async def service(tmp_path):
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    try:
        yield GroupService(db)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_default_bootstrap_is_persisted_and_idempotent(service) -> None:
    first_group, first_project = await service.ensure_default_resources()
    second_group, second_project = await service.ensure_default_resources()

    assert str(first_group.id) == DEFAULT_GROUP_ID
    assert str(first_project.id) == DEFAULT_PROJECT_ID
    assert first_project.id == ProjectID('default')
    assert first_group.slug == 'default'
    assert first_project.slug == 'default'
    assert first_group == second_group
    assert first_project == second_project
    assert first_project.group_id == first_group.id
    assert first_group.display_name == 'Default'
    assert first_group.status == GroupStatus.ACTIVE
    assert first_project.display_name == 'Default'
    assert first_project.status == ProjectStatus.ACTIVE
    assert first_project.is_private is False
    assert await service.has_all_authenticated_users(first_group.id, ADMIN)


def test_default_project_id_is_canonical_but_generated_ids_remain_opaque(
) -> None:
    assert ProjectID(DEFAULT_PROJECT_ID) == 'default'
    generated = ProjectID.generate()
    assert generated != DEFAULT_PROJECT_ID
    assert generated.startswith('project_')


@pytest.mark.asyncio
async def test_group_creation_rejects_reserved_default_slug(service) -> None:
    with pytest.raises(ValueError, match='Reserved resource slug'):
        await service.create_kumo_groups(
            [KumoGroupCreateSpec(display_name='  Ｄｅｆａｕｌｔ  ')], ADMIN)


@pytest.mark.asyncio
async def test_project_creation_rejects_reserved_default_slug(service) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Other')], ADMIN))[0]
    with pytest.raises(ValueError, match='Reserved resource slug'):
        await ProjectService(service._state_db).create_project(
            name='Default',
            group_id=str(group.id),
            user_authorization=ADMIN,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('corruption', [
    'group_status',
    'group_name',
    'project_status',
    'project_owner',
    'project_private',
    'project_name',
    'membership_rule',
])
async def test_default_bootstrap_rejects_corrupted_canonical_resources(
        service, corruption) -> None:
    group, project = await service.ensure_default_resources()
    async with service._state_db.write_tx() as tx:
        if corruption == 'group_status':
            group.status = GroupStatus.DELETED
            await service._groups.upsert(tx, str(group.id), group)
        elif corruption == 'group_name':
            group.display_name = 'Renamed'
            group.normalized_display_name = 'renamed'
            await service._groups.upsert(tx, str(group.id), group)
        elif corruption == 'project_status':
            project.status = ProjectStatus.ARCHIVED
            await service._projects.upsert(tx, str(project.id), project)
        elif corruption == 'project_owner':
            project.group_id = GroupID('group_' + 'a' * 32)
            await service._projects.upsert(tx, str(project.id), project)
        elif corruption == 'project_private':
            project.is_private = True
            await service._projects.upsert(tx, str(project.id), project)
        elif corruption == 'project_name':
            project.display_name = 'Renamed'
            project.normalized_display_name = 'renamed'
            await service._projects.upsert(tx, str(project.id), project)
        else:
            rule = await service._rules.find_one_by_fields(
                tx, {
                    'group_id': str(group.id),
                    'type': GroupMembershipRuleType.ALL_AUTHENTICATED_USERS,
                })
            assert rule is not None
            await service._rules.delete(tx, rule.id)

    with pytest.raises(RuntimeError, match='Canonical Default'):
        await service.ensure_default_resources()


@pytest.mark.asyncio
async def test_default_resources_reject_all_mutations(service) -> None:
    group, project = await service.ensure_default_resources()
    projects = ProjectService(service._state_db)

    with pytest.raises(ValueError, match='cannot be renamed'):
        await service.update_kumo_group(group.id, display_name='Renamed',
                                        principal=ADMIN)
    with pytest.raises(ValueError, match='cannot be deleted'):
        await service.delete_kumo_group(str(group.id), ADMIN)
    with pytest.raises(ValueError, match='must include all authenticated'):
        await service.set_all_authenticated_users(group.id, False,
                                                  principal=ADMIN)
    with pytest.raises(ValueError, match='cannot be renamed'):
        await projects.update_project(project.id, name='Renamed',
                                      user_authorization=ADMIN)
    with pytest.raises(ValueError, match='must remain shared'):
        await projects.update_project(project.id, is_private=True,
                                      user_authorization=ADMIN)
    with pytest.raises(ValueError, match='cannot be archived'):
        await projects.archive_project(project.id, ADMIN)

    verified_group, verified_project = await service.ensure_default_resources()
    assert verified_group == group
    assert verified_project == project


@pytest.mark.asyncio
async def test_group_administration_requires_explicit_admin(service) -> None:
    with pytest.raises(ProjectForbiddenError,
                       match='Authenticated principal is required'):
        await service.list_kumo_groups(None)
    with pytest.raises(ProjectForbiddenError,
                       match='Administrator access required'):
        await service.create_kumo_groups(
            [KumoGroupCreateSpec(display_name='Denied')], MEMBER)


@pytest.mark.asyncio
async def test_admin_lists_all_active_groups_without_membership(
        service) -> None:
    created = await service.create_kumo_groups([
        KumoGroupCreateSpec(display_name='Engineering'),
        KumoGroupCreateSpec(display_name='Finance'),
    ], ADMIN)

    assert not ADMIN.kumo_group_ids
    assert [group.id for group in await service.list_user_groups(ADMIN)] == [
        created[0].id,
        created[1].id,
    ]


@pytest.mark.asyncio
async def test_ordinary_user_lists_only_current_active_memberships(
        service) -> None:
    first, second, deleted = await service.create_kumo_groups([
        KumoGroupCreateSpec(display_name='First'),
        KumoGroupCreateSpec(display_name='Second'),
        KumoGroupCreateSpec(display_name='Deleted'),
    ], ADMIN)
    await service.delete_kumo_group(str(deleted.id), ADMIN)
    member = UserAuthorization(
        user_id=UserID('ordinary-user'),
        access_token='token',
        permissions=(),
        kumo_group_ids=(str(second.id), str(deleted.id)),
    )

    assert [group.id
            for group in await service.list_user_groups(member)] == [second.id]
    assert first.id not in {
        group.id
        for group in await service.list_user_groups(member)
    }


@pytest.mark.asyncio
async def test_native_membership_unions_manual_and_authenticated_rule(
        service) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Engineering')], ADMIN))[0]
    await service.add_member(group.id, 'user-1', PrincipalType.USER,
                             principal=ADMIN)
    await service.add_member(group.id, 'service-1',
                             PrincipalType.SERVICE_ACCOUNT, principal=ADMIN)
    await service.set_all_authenticated_users(group.id, True, principal=ADMIN)

    assert await service.resolve_native_memberships(
        'user-1', principal_type=PrincipalType.USER,
        authenticated=True) == [str(group.id)]
    assert await service.resolve_native_memberships(
        'user-2', principal_type=PrincipalType.USER,
        authenticated=True) == [str(group.id)]
    assert await service.resolve_native_memberships(
        'service-1', principal_type=PrincipalType.SERVICE_ACCOUNT,
        authenticated=True) == [str(group.id)]
    assert await service.resolve_native_memberships(
        'service-2', principal_type=PrincipalType.SERVICE_ACCOUNT,
        authenticated=True) == []


@pytest.mark.asyncio
async def test_member_input_is_trimmed_and_concurrent_add_is_idempotent(
        service) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Members')], ADMIN))[0]

    results = await asyncio.gather(*(service.add_member(
        group.id,
        value,
        PrincipalType.USER,
        principal=ADMIN,
    ) for value in (' user-1 ', 'user-1', ' user-1 ', 'user-1')))

    assert len({result.id for result in results}) == 1
    assert results[0].principal_id == 'user-1'
    assert await service.list_members(group.id, ADMIN) == [results[0]]
    with pytest.raises(GroupInputError, match='principal_id cannot be empty'):
        await service.add_member(group.id, '   ', PrincipalType.USER,
                                 principal=ADMIN)


@pytest.mark.asyncio
async def test_all_authenticated_concurrent_enable_is_idempotent(
        service) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Everyone')], ADMIN))[0]

    await asyncio.gather(
        *(service.set_all_authenticated_users(group.id, True, principal=ADMIN)
          for _ in range(6)))

    async with service._state_db.read_tx() as tx:
        rows = await service._rules.list_by_field(tx, 'group_id', str(group.id))
    assert len(rows) == 1
    assert await service.has_all_authenticated_users(group.id, ADMIN)


@pytest.mark.asyncio
async def test_external_mapping_is_encrypted_and_applied_live(service) -> None:
    first, second = await service.create_kumo_groups([
        KumoGroupCreateSpec(display_name='First'),
        KumoGroupCreateSpec(display_name='Second'),
    ], ADMIN)
    mapping = await service.create_external_mapping(first.id, 'direct_connect',
                                                    'ldap-engineering',
                                                    principal=ADMIN)

    assert mapping.encrypted_external_group_id != 'ldap-engineering'
    assert decrypt_idp_id(
        mapping.encrypted_external_group_id) == 'ldap-engineering'
    assert await service.map_idp_group_ids(['ldap-engineering']
                                           ) == [str(first.id)]

    await service.delete_external_mapping(first.id, mapping.id, principal=ADMIN)
    replacement = await service.create_external_mapping(second.id,
                                                        'direct_connect',
                                                        'ldap-engineering',
                                                        principal=ADMIN)
    assert replacement.id != mapping.id
    assert await service.map_idp_group_ids(['ldap-engineering']
                                           ) == [str(second.id)]


@pytest.mark.asyncio
async def test_unsupported_external_mappings_are_rejected(service) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Unsupported Provider')], ADMIN))[0]

    with pytest.raises(GroupInputError, match='must be one of: direct_connect'):
        await service.create_external_mapping(group.id, 'unsupported',
                                              'external-team', principal=ADMIN)


@pytest.mark.asyncio
async def test_mapping_inputs_are_normalized_idempotent_and_provider_scoped(
        service) -> None:
    first, second = await service.create_kumo_groups([
        KumoGroupCreateSpec(display_name='First'),
        KumoGroupCreateSpec(display_name='Second'),
    ], ADMIN)

    results = await asyncio.gather(*(service.create_external_mapping(
        first.id,
        provider,
        external_id,
        principal=ADMIN,
    ) for provider, external_id in (
        (' DIRECT_CONNECT ', ' ldap-team '),
        ('direct_connect', 'ldap-team'),
        ('Direct_Connect', ' ldap-team'),
    )))
    assert len({result.id for result in results}) == 1
    assert results[0].identity_provider_id == 'direct_connect'
    assert decrypt_idp_id(results[0].encrypted_external_group_id) == 'ldap-team'
    assert await service.map_idp_group_ids(
        [' ldap-team '],
        identity_provider_id=' DIRECT_CONNECT ') == [str(first.id)]
    with pytest.raises(GroupRelationConflictError, match='already mapped'):
        await service.create_external_mapping(second.id, 'direct_connect',
                                              'ldap-team', principal=ADMIN)
    for provider_id, external_id, field_name in (
        (' ', 'group', 'identity_provider_id'),
        ('okta', 'group', 'identity_provider_id'),
        ('direct_connect', ' ', 'external_group_id'),
        ('direct_connect', 'group\0privileged', 'external_group_id'),
    ):
        with pytest.raises(GroupInputError, match=field_name):
            await service.create_external_mapping(first.id, provider_id,
                                                  external_id, principal=ADMIN)


@pytest.mark.asyncio
async def test_group_owning_project_cannot_be_deleted(service) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Owner')], ADMIN))[0]
    project = await ProjectService(service._state_db).create_project(
        name='Owned project',
        group_id=str(group.id),
        user_authorization=ADMIN,
    )
    assert project.group_id == group.id

    with pytest.raises(ValueError, match='owns a project'):
        await service.delete_kumo_group(str(group.id), ADMIN)

    before_archive = project.timestamps.updated_at
    archived = await ProjectService(service._state_db
                                    ).archive_project(project.id,
                                                      user_authorization=ADMIN)
    assert archived.timestamps.updated_at > before_archive
    with pytest.raises(ValueError, match='owns a project'):
        await service.delete_kumo_group(str(group.id), ADMIN)


@pytest.mark.asyncio
async def test_group_delete_cleans_relations_and_allows_mapping_reassignment(
        service) -> None:
    deleted, replacement = await service.create_kumo_groups([
        KumoGroupCreateSpec(display_name='Deleted'),
        KumoGroupCreateSpec(display_name='Replacement'),
    ], ADMIN)
    await service.add_member(deleted.id, 'user-1', PrincipalType.USER,
                             principal=ADMIN)
    mapping = await service.create_external_mapping(deleted.id,
                                                    'direct_connect',
                                                    'external-team',
                                                    principal=ADMIN)
    await service.set_all_authenticated_users(deleted.id, True, principal=ADMIN)
    binding = IdentityGroupEntry(
        id=IdentityGroupID.generate(),
        identity_id=IdentityID.generate(),
        group_id=deleted.id,
        timestamps=get_current_timestamps(),
    )
    async with service._state_db.write_tx() as tx:
        await service._identity_bindings.upsert(tx, str(binding.id), binding)

    await service.delete_kumo_group(str(deleted.id), ADMIN)

    async with service._state_db.read_tx() as tx:
        assert await service._members.list_by_field(tx, 'group_id',
                                                    str(deleted.id)) == []
        assert await service._external_mappings.list_by_field(
            tx, 'group_id', str(deleted.id)) == []
        assert await service._rules.list_by_field(tx, 'group_id',
                                                  str(deleted.id)) == []
        assert await service._identity_bindings.list_by_field(
            tx, 'group_id', str(deleted.id)) == []

    reassigned = await service.create_external_mapping(replacement.id,
                                                       'direct_connect',
                                                       'external-team',
                                                       principal=ADMIN)
    assert reassigned.id != mapping.id
    with pytest.raises(GroupNotFoundError):
        await service.add_member(deleted.id, 'user-2', PrincipalType.USER,
                                 principal=ADMIN)
    with pytest.raises(GroupNotFoundError):
        await service.create_external_mapping(deleted.id, 'direct_connect',
                                              'another-team', principal=ADMIN)
    with pytest.raises(GroupNotFoundError):
        await service.set_all_authenticated_users(deleted.id, True,
                                                  principal=ADMIN)
    with pytest.raises(GroupNotFoundError):
        await service.remove_member(deleted.id, 'user-1', principal=ADMIN)
    with pytest.raises(GroupNotFoundError):
        await service.delete_external_mapping(deleted.id, mapping.id,
                                              principal=ADMIN)


@pytest.mark.asyncio
async def test_group_names_are_globally_normalized_and_reserved_after_delete(
        service) -> None:
    first = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='  Engineering  ')], ADMIN))[0]

    with pytest.raises(GroupAlreadyExists, match='not available'):
        await service.create_kumo_groups(
            [KumoGroupCreateSpec(display_name='ＥＮＧＩＮＥＥＲＩＮＧ')], ADMIN)

    await service.delete_kumo_group(str(first.id), ADMIN)
    with pytest.raises(GroupAlreadyExists, match='not available'):
        await service.create_kumo_groups(
            [KumoGroupCreateSpec(display_name='engineering')], ADMIN)


@pytest.mark.asyncio
async def test_concurrent_group_create_has_one_winner(service) -> None:
    results = await asyncio.gather(
        service.create_kumo_groups(
            [KumoGroupCreateSpec(display_name='Platform')], ADMIN),
        service.create_kumo_groups(
            [KumoGroupCreateSpec(display_name=' ＰＬＡＴＦＯＲＭ ')], ADMIN),
        return_exceptions=True,
    )

    assert sum(isinstance(result, list) for result in results) == 1
    assert sum(isinstance(result, GroupAlreadyExists)
               for result in results) == 1


@pytest.mark.asyncio
async def test_concurrent_group_update_has_one_winner(service) -> None:
    first, second = await service.create_kumo_groups([
        KumoGroupCreateSpec(display_name='First'),
        KumoGroupCreateSpec(display_name='Second'),
    ], ADMIN)

    results = await asyncio.gather(
        service.update_kumo_group(first.id, display_name='Winner',
                                  principal=ADMIN),
        service.update_kumo_group(second.id, display_name=' ＷＩＮＮＥＲ ',
                                  principal=ADMIN),
        return_exceptions=True,
    )

    assert sum(isinstance(result, KumoGroupDBEntry) for result in results) == 1
    assert sum(isinstance(result, GroupAlreadyExists)
               for result in results) == 1


@pytest.mark.asyncio
async def test_removing_principal_removes_all_manual_principal_types(
        service) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Mixed')], ADMIN))[0]
    await service.add_member(group.id, 'principal-1', PrincipalType.USER,
                             principal=ADMIN)
    await service.add_member(group.id, 'principal-1',
                             PrincipalType.SERVICE_ACCOUNT, principal=ADMIN)

    await service.remove_member(GroupID(group.id), 'principal-1',
                                principal=ADMIN)

    assert await service.list_members(group.id, ADMIN) == []
