# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3

import asyncpg
from kumoapi.data_source import (
    DEFAULT_GROUP_ID,
    DEFAULT_GROUP_SLUG,
    DEFAULT_PROJECT_ID,
    DEFAULT_PROJECT_SLUG,
    derive_resource_slug,
    validate_resource_slug,
)

from rest.config.encryption import (
    decrypt_idp_id,
    encrypt_idp_id,
    hash_idp_group_id,
)
from statedb.config.common import (
    ExternalGroupMappingID,
    GroupID,
    GroupMemberID,
    GroupMembershipRuleID,
    ProjectID,
)
from statedb.db.auth import AuthPrincipal
from statedb.db.group_relations import lock_group_relations
from statedb.db.identities import normalize_display_name
from statedb.db.repositories import JsonRepository
from statedb.db.runtime import StateDB
from statedb.rest.group import (
    ExternalGroupMappingEntry,
    ExternalGroupMappingSpec,
    ExternalIdentityProviderID,
    GroupMemberEntry,
    GroupMembershipRuleEntry,
    GroupMembershipRuleType,
    GroupMemberSpec,
    GroupStatus,
    KumoGroupCreateSpec,
    KumoGroupDBEntry,
    PrincipalType,
)
from statedb.rest.identity import IdentityGroupEntry
from statedb.rest.project import ProjectDBEntry, ProjectStatus

_DEFAULT_ALL_AUTHENTICATED_USERS_RULE_ID = GroupMembershipRuleID(
    'group_membership_rule_00000000000000000000000000000000')
_DEFAULT_BOOTSTRAP_LOCK_KEY = int.from_bytes(b'kumo-def', byteorder='big')


class GroupService:
    def __init__(self, state_db: StateDB) -> None:
        self._state_db = state_db
        self._groups = JsonRepository('groups', KumoGroupDBEntry)
        self._members = JsonRepository('group_members', GroupMemberEntry)
        self._external_mappings = JsonRepository('external_group_mappings',
                                                 ExternalGroupMappingEntry)
        self._rules = JsonRepository('group_membership_rules',
                                     GroupMembershipRuleEntry)
        self._projects = JsonRepository('projects', ProjectDBEntry)
        self._identity_bindings = JsonRepository('identity_groups',
                                                 IdentityGroupEntry)

    async def _get_active_group_in_tx(
        self,
        tx,
        group_id: GroupID | str,
    ) -> KumoGroupDBEntry:
        from statedb.utils.errors import GroupNotFoundError

        row = await self._groups.get(tx, str(group_id))
        if row is None or row.payload.status != GroupStatus.ACTIVE:
            raise GroupNotFoundError(f"Group '{group_id}' not found")
        return row.payload

    async def get_kumo_group_by_slug(
        self,
        slug: str,
        principal: AuthPrincipal | None,
    ) -> KumoGroupDBEntry:
        """Return one visible active group without leaking hidden slugs."""
        from statedb.utils.errors import GroupNotFoundError

        slug = validate_resource_slug(slug, allow_reserved=True)
        principal = self._require_principal(principal)
        async with self._state_db.read_tx() as tx:
            row = await self._groups.find_one_by_fields(
                tx, {
                    'slug': slug,
                    'status': GroupStatus.ACTIVE,
                })
        if row is None or (not principal.is_admin and row.payload.id
                           not in principal.kumo_group_ids):
            raise GroupNotFoundError('Group not found')
        return row.payload

    @staticmethod
    def _clean_required(value: str, field_name: str) -> str:
        from statedb.utils.errors import GroupInputError

        if value is None:
            raise GroupInputError(f'{field_name} cannot be empty')
        cleaned = str(value).strip()
        if not cleaned:
            raise GroupInputError(f'{field_name} cannot be empty')
        return cleaned

    @classmethod
    def _clean_provider_id(
        cls,
        value: str | ExternalIdentityProviderID,
    ) -> ExternalIdentityProviderID:
        if isinstance(value, ExternalIdentityProviderID):
            return value
        provider_id = cls._clean_required(
            value, 'identity_provider_id').casefold()
        try:
            return ExternalIdentityProviderID.normalize(provider_id)
        except ValueError as error:
            from statedb.utils.errors import GroupInputError
            supported = ', '.join(provider.value
                                  for provider in ExternalIdentityProviderID)
            raise GroupInputError(
                'identity_provider_id must be one of: '
                f'{supported}') from error

    @classmethod
    def _clean_external_group_id(cls, value: str) -> str:
        from statedb.utils.errors import GroupInputError

        external_group_id = cls._clean_required(value, 'external_group_id')
        if '\0' in external_group_id:
            raise GroupInputError(
                'external_group_id cannot contain NUL characters')
        return external_group_id

    async def ensure_default_resources(
            self) -> tuple[KumoGroupDBEntry, ProjectDBEntry]:
        from rest.utils.driver_common import get_current_timestamps

        default_name = normalize_display_name('Default')
        default_group_id = GroupID(DEFAULT_GROUP_ID)
        default_project_id = ProjectID(DEFAULT_PROJECT_ID)
        async with self._state_db.write_tx() as tx:
            # Multiple application processes may bootstrap the same new
            # Postgres database concurrently. Serialize the invariant check
            # and creation so no process observes half of the canonical pair.
            if tx.backend == 'postgres':
                await tx.fetchrow(
                    'SELECT pg_advisory_xact_lock($1)',
                    _DEFAULT_BOOTSTRAP_LOCK_KEY,
                )
            group_row = await self._groups.get(tx, str(default_group_id))
            project_row = await self._projects.get(tx, str(default_project_id))
            named_group = await self._groups.find_one_by_fields(
                tx, {'normalized_display_name': default_name})
            named_project = await self._projects.find_one_by_fields(
                tx, {'normalized_display_name': default_name})

            if ((named_group is not None
                 and named_group.id != str(default_group_id))
                    or (named_project is not None
                        and named_project.id != str(default_project_id))):
                raise RuntimeError(
                    "A non-canonical resource is named 'Default'; reset the "
                    'development StateDB before startup')

            if (group_row is None) != (project_row is None):
                raise RuntimeError(
                    'Canonical Default group and project must either both be '
                    'absent or both already satisfy the bootstrap invariants')

            if group_row is None:
                group = KumoGroupDBEntry(
                    id=default_group_id,
                    slug=validate_resource_slug(DEFAULT_GROUP_SLUG,
                                                allow_reserved=True),
                    display_name='Default',
                    normalized_display_name=default_name,
                    description='Default group for all authenticated users',
                    status=GroupStatus.ACTIVE,
                    timestamps=get_current_timestamps(),
                )
                await self._groups.upsert(tx, str(group.id), group)
                rule = GroupMembershipRuleEntry(
                    id=_DEFAULT_ALL_AUTHENTICATED_USERS_RULE_ID,
                    group_id=group.id,
                    type=GroupMembershipRuleType.ALL_AUTHENTICATED_USERS,
                    timestamps=get_current_timestamps(),
                )
                await self._rules.upsert(tx, str(rule.id), rule)
                project = ProjectDBEntry(
                    id=default_project_id,
                    group_id=group.id,
                    slug=validate_resource_slug(DEFAULT_PROJECT_SLUG,
                                                allow_reserved=True),
                    display_name='Default',
                    normalized_display_name=default_name,
                    created_by='system',
                    created_by_name='Kumo',
                    description='Default shared project',
                    is_private=False,
                    status=ProjectStatus.ACTIVE,
                    timestamps=get_current_timestamps(),
                )
                await self._projects.upsert(tx, str(project.id), project)
            else:
                group = group_row.payload
                project = project_row.payload
                if (group.slug != DEFAULT_GROUP_SLUG
                        or group.display_name != 'Default'
                        or group.normalized_display_name != default_name
                        or group.status != GroupStatus.ACTIVE):
                    raise RuntimeError(
                        'Canonical Default group violates bootstrap invariants')
                if (project.slug != DEFAULT_PROJECT_SLUG
                        or project.display_name != 'Default'
                        or project.normalized_display_name != default_name
                        or project.status != ProjectStatus.ACTIVE
                        or project.group_id != default_group_id
                        or project.is_private):
                    raise RuntimeError(
                        'Canonical Default project violates bootstrap invariants'
                    )
                rule_row = await self._rules.find_one_by_fields(
                    tx, {
                        'group_id': str(group.id),
                        'type': GroupMembershipRuleType.ALL_AUTHENTICATED_USERS,
                    })
                if rule_row is None:
                    raise RuntimeError('Canonical Default group is missing its '
                                       'all-authenticated-users rule')
        return group, project

    async def list_kumo_groups(
        self,
        principal: AuthPrincipal | None,
    ) -> list[KumoGroupDBEntry]:
        self._require_admin(principal)
        async with self._state_db.read_tx() as tx:
            return [
                row.payload for row in await self._groups.list_by_fields(
                    tx, {'status': GroupStatus.ACTIVE})
            ]

    async def _get_active_group(self, group_id: str) -> KumoGroupDBEntry:
        from statedb.utils.errors import GroupNotFoundError

        async with self._state_db.read_tx() as tx:
            row = await self._groups.get(tx, str(group_id))
        if row is None or row.payload.status != GroupStatus.ACTIVE:
            raise GroupNotFoundError(f"Group '{group_id}' not found")
        return row.payload

    async def get_kumo_group(
        self,
        group_id: str,
        principal: AuthPrincipal | None,
    ) -> KumoGroupDBEntry:
        from statedb.utils.errors import GroupNotFoundError

        principal = self._require_principal(principal)
        group = await self._get_active_group(group_id)
        if (not principal.is_admin
                and group.id not in principal.kumo_group_ids):
            raise GroupNotFoundError('Group not found')
        return group

    async def get_kumo_group_by_alias(
        self,
        display_name: str,
        user_authorization: AuthPrincipal | None,
    ) -> KumoGroupDBEntry:
        from statedb.utils.errors import GroupNotFoundError

        async with self._state_db.read_tx() as tx:
            row = await self._groups.find_one_by_fields(
                tx, {
                    'status': GroupStatus.ACTIVE,
                    'normalized_display_name':
                    normalize_display_name(display_name),
                })
        if row is None:
            raise GroupNotFoundError(
                f"Group with name '{display_name}' was not found.")
        self._require_group_membership(row.payload.id, user_authorization)
        return row.payload

    async def create_kumo_groups(
        self,
        groups: list[KumoGroupCreateSpec],
        principal: AuthPrincipal | None,
    ) -> list[KumoGroupDBEntry]:
        self._require_admin(principal)
        from rest.utils.driver_common import get_current_timestamps
        from statedb.utils.errors import (
            GroupAlreadyExists,
            ResourceSlugConflictError,
        )

        entries: list[KumoGroupDBEntry] = []
        try:
            async with self._state_db.write_tx() as tx:
                seen_names: set[str] = set()
                seen_slugs: set[str] = set()
                for args in groups:
                    display_name = args.display_name.strip()
                    normalized_name = normalize_display_name(display_name)
                    if not normalized_name:
                        raise ValueError('display_name cannot be empty')
                    if len(display_name) > 64:
                        raise ValueError(
                            'display_name exceeds 64 character limit')
                    slug = (validate_resource_slug(str(args.slug))
                            if args.slug is not None else
                            derive_resource_slug(display_name))
                    if slug in seen_slugs:
                        raise ResourceSlugConflictError(
                            f"Group slug '{slug}' is already in use.")
                    if normalized_name in seen_names or await self._groups.exists_by_fields(
                            tx, {'normalized_display_name': normalized_name}):
                        raise GroupAlreadyExists(
                            'Group name is not available.')
                    # Avoid an SDK selector becoming ambiguous where practical.
                    if (await self._groups.exists_by_fields(
                            tx, {'normalized_display_name': slug})
                            or await self._groups.exists_by_fields(
                                tx, {'slug': normalized_name})):
                        raise GroupAlreadyExists(
                            'Group name conflicts with an existing slug.')
                    seen_names.add(normalized_name)
                    seen_slugs.add(slug)
                    entry = KumoGroupDBEntry(
                        id=GroupID.generate(),
                        slug=slug,
                        display_name=display_name,
                        normalized_display_name=normalized_name,
                        description=args.description,
                        status=GroupStatus.ACTIVE,
                        timestamps=get_current_timestamps(),
                    )
                    entries.append(entry)
                    await self._groups.upsert(tx, str(entry.id), entry)
        except (sqlite3.IntegrityError, asyncpg.UniqueViolationError) as error:
            # The database constraint is authoritative under concurrent
            # creation. Inspect only after rollback to choose a safe 409.
            async with self._state_db.read_tx() as tx:
                for entry in entries:
                    if await self._groups.exists_by_fields(
                            tx, {'slug': entry.slug}):
                        raise ResourceSlugConflictError(
                            f"Group slug '{entry.slug}' is already in use."
                        ) from error
            raise GroupAlreadyExists('Group name is not available.') from error
        return entries

    async def update_kumo_group(
        self,
        group_id: GroupID,
        display_name: str | None = None,
        description: str | None = None,
        *,
        principal: AuthPrincipal | None,
    ) -> KumoGroupDBEntry:
        self._require_admin(principal)
        from statedb.config.timestamp import update_timestamps
        from statedb.utils.errors import GroupAlreadyExists, GroupNotFoundError

        if (str(group_id) == DEFAULT_GROUP_ID and display_name is not None
                and display_name.strip() != 'Default'):
            raise ValueError('The canonical Default group cannot be renamed')

        try:
            async with self._state_db.write_tx() as tx:
                row = await self._groups.get(tx, str(group_id))
                if row is None or row.payload.status != GroupStatus.ACTIVE:
                    raise GroupNotFoundError(f"Group '{group_id}' not found")
                entry = row.payload
                if display_name is not None:
                    display_name = display_name.strip()
                    normalized_name = normalize_display_name(display_name)
                    if not normalized_name:
                        raise ValueError('display_name cannot be empty')
                    if len(display_name) > 64:
                        raise ValueError(
                            'display_name exceeds 64 character limit')
                    if await self._groups.exists_by_fields(
                            tx, {
                                'normalized_display_name': normalized_name,
                            }, exclude_id=str(group_id)):
                        raise GroupAlreadyExists('Group name is not available.')
                    if await self._groups.exists_by_fields(
                            tx, {'slug': normalized_name},
                            exclude_id=str(group_id)):
                        raise GroupAlreadyExists(
                            'Group name conflicts with an existing slug.')
                    entry.display_name = display_name
                    entry.normalized_display_name = normalized_name
                if description is not None:
                    entry.description = description
                entry.timestamps = update_timestamps(entry.timestamps)
                await self._groups.upsert(tx, str(entry.id), entry)
        except (sqlite3.IntegrityError, asyncpg.UniqueViolationError) as error:
            raise GroupAlreadyExists('Group name is not available.') from error
        return entry

    async def delete_kumo_group(
        self,
        group_id: str,
        principal: AuthPrincipal | None,
    ) -> None:
        self._require_admin(principal)
        from statedb.config.timestamp import update_timestamps
        from statedb.utils.errors import GroupNotFoundError

        if str(group_id) == DEFAULT_GROUP_ID:
            raise ValueError('The canonical Default group cannot be deleted')

        async with self._state_db.write_tx() as tx:
            await lock_group_relations(tx, [group_id])
            row = await self._groups.get(tx, str(group_id))
            if row is None or row.payload.status != GroupStatus.ACTIVE:
                raise GroupNotFoundError(f"Group '{group_id}' not found")
            projects = await self._projects.list_by_field(
                tx, 'group_id', str(group_id))
            if projects:
                raise ValueError(
                    'A group that owns a project cannot be deleted')
            for repository in (
                    self._members,
                    self._external_mappings,
                    self._rules,
                    self._identity_bindings,
            ):
                relations = await repository.list_by_field(
                    tx, 'group_id', str(group_id))
                for relation in relations:
                    await repository.delete(tx, relation.id)
            row.payload.status = GroupStatus.DELETED
            row.payload.timestamps = update_timestamps(row.payload.timestamps)
            await self._groups.upsert(tx, row.id, row.payload)

    async def add_member(
        self,
        group_id: GroupID,
        principal_id: str,
        principal_type: PrincipalType,
        *,
        principal: AuthPrincipal | None,
    ) -> GroupMemberEntry:
        from rest.utils.driver_common import get_current_timestamps
        from statedb.utils.errors import GroupRelationConflictError

        self._require_admin(principal)
        principal_id = self._clean_required(principal_id, 'principal_id')
        principal_type = PrincipalType(principal_type)
        entry = GroupMemberEntry(
            id=GroupMemberID.generate(),
            group_id=group_id,
            principal_id=principal_id,
            principal_type=principal_type,
            timestamps=get_current_timestamps(),
        )
        fields = {
            'group_id': str(group_id),
            'principal_id': principal_id,
            'principal_type': principal_type,
        }
        try:
            async with self._state_db.write_tx() as tx:
                await lock_group_relations(tx, [group_id])
                await self._get_active_group_in_tx(tx, group_id)
                existing = await self._members.find_one_by_fields(tx, fields)
                if existing is not None:
                    return existing.payload
                await self._members.upsert(tx, str(entry.id), entry)
            return entry
        except (sqlite3.IntegrityError, asyncpg.UniqueViolationError) as error:
            async with self._state_db.read_tx() as tx:
                existing = await self._members.find_one_by_fields(tx, fields)
            if existing is not None:
                return existing.payload
            raise GroupRelationConflictError(
                'Group membership changed concurrently') from error

    async def remove_member(
        self,
        group_id: GroupID,
        principal_id: str,
        principal_type: PrincipalType | None = None,
        *,
        principal: AuthPrincipal | None,
    ) -> None:
        self._require_admin(principal)
        principal_id = self._clean_required(principal_id, 'principal_id')
        async with self._state_db.write_tx() as tx:
            await lock_group_relations(tx, [group_id])
            await self._get_active_group_in_tx(tx, group_id)
            fields = {
                'group_id': str(group_id),
                'principal_id': principal_id,
                'source': 'manual',
            }
            if principal_type is not None:
                fields['principal_type'] = principal_type
            rows = await self._members.list_by_fields(tx, fields)
            for row in rows:
                await self._members.delete(tx, row.id)

    async def list_members(
        self,
        group_id: GroupID,
        principal: AuthPrincipal | None,
    ) -> list[GroupMemberEntry]:
        self._require_admin(principal)
        await self._get_active_group(str(group_id))
        async with self._state_db.read_tx() as tx:
            entries = [
                row.payload for row in await self._members.list_by_field(
                    tx, 'group_id', str(group_id))
                if row.payload.source == 'manual'
            ]
        return sorted(
            entries,
            key=lambda entry: (
                entry.principal_type.value,
                entry.principal_id,
            ),
        )

    async def replace_members(
        self,
        group_id: GroupID,
        members: list[GroupMemberSpec],
        *,
        principal: AuthPrincipal | None,
    ) -> list[GroupMemberEntry]:
        """Atomically replace only directly managed member rows."""
        from rest.utils.driver_common import get_current_timestamps
        from statedb.utils.errors import GroupRelationConflictError

        self._require_admin(principal)
        normalized: dict[tuple[PrincipalType, str], None] = {}
        for member in members:
            principal_id = self._clean_required(member.principal_id,
                                                'principal_id')
            principal_type = PrincipalType(member.principal_type)
            normalized[(principal_type, principal_id)] = None

        try:
            async with self._state_db.write_tx() as tx:
                await lock_group_relations(tx, [group_id])
                await self._get_active_group_in_tx(tx, group_id)
                group_rows = await self._members.list_by_field(
                    tx, 'group_id', str(group_id))
                manual_by_key = {
                    (row.payload.principal_type, row.payload.principal_id):
                    row.payload
                    for row in group_rows if row.payload.source == 'manual'
                }
                non_manual_keys = {
                    (row.payload.principal_type, row.payload.principal_id)
                    for row in group_rows if row.payload.source != 'manual'
                }
                requested_keys = set(normalized)
                if requested_keys & non_manual_keys:
                    raise GroupRelationConflictError(
                        'Principal membership is managed by another source')

                final_entries: list[GroupMemberEntry] = []
                for principal_type, principal_id in sorted(
                        requested_keys,
                        key=lambda key: (key[0].value, key[1])):
                    key = (principal_type, principal_id)
                    existing = manual_by_key.get(key)
                    if existing is not None:
                        final_entries.append(existing)
                        continue
                    entry = GroupMemberEntry(
                        id=GroupMemberID.generate(),
                        group_id=group_id,
                        principal_id=principal_id,
                        principal_type=principal_type,
                        source='manual',
                        timestamps=get_current_timestamps(),
                    )
                    await self._members.upsert(tx, str(entry.id), entry)
                    final_entries.append(entry)

                for key, entry in manual_by_key.items():
                    if key not in requested_keys:
                        await self._members.delete(tx, str(entry.id))
        except (sqlite3.IntegrityError,
                asyncpg.UniqueViolationError) as error:
            raise GroupRelationConflictError(
                'Direct group membership changed concurrently') from error

        return final_entries

    async def create_external_mapping(
        self,
        group_id: GroupID,
        identity_provider_id: str,
        external_group_id: str,
        *,
        principal: AuthPrincipal | None,
    ) -> ExternalGroupMappingEntry:
        from rest.utils.driver_common import get_current_timestamps
        from statedb.utils.errors import GroupRelationConflictError

        self._require_admin(principal)
        identity_provider_id = self._clean_provider_id(identity_provider_id)
        external_group_id = self._clean_external_group_id(external_group_id)
        external_group_id_hash = hash_idp_group_id(identity_provider_id,
                                                   external_group_id)
        entry = ExternalGroupMappingEntry(
            id=ExternalGroupMappingID.generate(),
            group_id=group_id,
            identity_provider_id=identity_provider_id,
            encrypted_external_group_id=encrypt_idp_id(external_group_id),
            external_group_id_hash=external_group_id_hash,
            timestamps=get_current_timestamps(),
        )
        fields = {
            'identity_provider_id': identity_provider_id,
            'external_group_id_hash': external_group_id_hash,
        }
        try:
            async with self._state_db.write_tx() as tx:
                await lock_group_relations(tx, [group_id])
                await self._get_active_group_in_tx(tx, group_id)
                existing = await self._external_mappings.find_one_by_fields(
                    tx, fields)
                if existing is not None:
                    if existing.payload.group_id == group_id:
                        return existing.payload
                    raise GroupRelationConflictError(
                        'External group is already mapped to another group')
                await self._external_mappings.upsert(tx, str(entry.id), entry)
            return entry
        except (sqlite3.IntegrityError, asyncpg.UniqueViolationError) as error:
            async with self._state_db.read_tx() as tx:
                existing = await self._external_mappings.find_one_by_fields(
                    tx, fields)
            if existing is not None and existing.payload.group_id == group_id:
                return existing.payload
            raise GroupRelationConflictError(
                'External group is already mapped to another group') from error

    async def delete_external_mapping(
        self,
        group_id: GroupID,
        mapping_id: ExternalGroupMappingID,
        *,
        principal: AuthPrincipal | None,
    ) -> None:
        self._require_admin(principal)
        async with self._state_db.write_tx() as tx:
            await lock_group_relations(tx, [group_id])
            await self._get_active_group_in_tx(tx, group_id)
            row = await self._external_mappings.get(tx, str(mapping_id))
            if row is not None and row.payload.group_id == group_id:
                await self._external_mappings.delete(tx, row.id)

    async def list_external_mappings(
        self,
        group_id: GroupID,
        principal: AuthPrincipal | None,
    ) -> list[ExternalGroupMappingEntry]:
        self._require_admin(principal)
        await self._get_active_group(str(group_id))
        async with self._state_db.read_tx() as tx:
            entries = [
                row.payload
                for row in await self._external_mappings.list_by_field(
                    tx, 'group_id', str(group_id))
            ]
        return sorted(
            entries,
            key=lambda entry: (
                entry.identity_provider_id,
                decrypt_idp_id(entry.encrypted_external_group_id),
            ),
        )

    async def replace_external_mappings(
        self,
        group_id: GroupID,
        mappings: list[ExternalGroupMappingSpec],
        *,
        principal: AuthPrincipal | None,
    ) -> list[ExternalGroupMappingEntry]:
        """Atomically replace every external mapping owned by one group."""
        from rest.utils.driver_common import get_current_timestamps
        from statedb.utils.errors import GroupRelationConflictError

        self._require_admin(principal)
        normalized: dict[tuple[ExternalIdentityProviderID, str], str] = {}
        for mapping in mappings:
            provider_id = self._clean_provider_id(
                mapping.identity_provider_id)
            external_group_id = self._clean_external_group_id(
                mapping.external_group_id)
            normalized[(provider_id,
                        external_group_id)] = hash_idp_group_id(
                            provider_id, external_group_id)

        try:
            async with self._state_db.write_tx() as tx:
                await lock_group_relations(tx, [group_id])
                await self._get_active_group_in_tx(tx, group_id)
                current_rows = await self._external_mappings.list_by_field(
                    tx, 'group_id', str(group_id))
                current_by_key = {
                    (
                        row.payload.identity_provider_id,
                        row.payload.external_group_id_hash,
                    ): row.payload
                    for row in current_rows
                }

                requested: list[tuple[str, str, str]] = []
                for (provider_id,
                     external_group_id), external_hash in normalized.items():
                    fields = {
                        'identity_provider_id': provider_id,
                        'external_group_id_hash': external_hash,
                    }
                    owner = await self._external_mappings.find_one_by_fields(
                        tx, fields)
                    if (owner is not None
                            and owner.payload.group_id != group_id):
                        raise GroupRelationConflictError(
                            'External group is already mapped to another group'
                        )
                    requested.append(
                        (provider_id, external_group_id, external_hash))

                final_entries: list[ExternalGroupMappingEntry] = []
                retained_keys: set[tuple[str, str]] = set()
                for provider_id, external_group_id, external_hash in sorted(
                        requested):
                    key = (provider_id, external_hash)
                    retained_keys.add(key)
                    existing = current_by_key.get(key)
                    if existing is not None:
                        final_entries.append(existing)
                        continue
                    entry = ExternalGroupMappingEntry(
                        id=ExternalGroupMappingID.generate(),
                        group_id=group_id,
                        identity_provider_id=provider_id,
                        encrypted_external_group_id=encrypt_idp_id(
                            external_group_id),
                        external_group_id_hash=external_hash,
                        timestamps=get_current_timestamps(),
                    )
                    await self._external_mappings.upsert(
                        tx, str(entry.id), entry)
                    final_entries.append(entry)

                for key, entry in current_by_key.items():
                    if key not in retained_keys:
                        await self._external_mappings.delete(tx, str(entry.id))
        except (sqlite3.IntegrityError,
                asyncpg.UniqueViolationError) as error:
            # A concurrent replacement on another group may win the unique
            # provider/hash key after our ownership preflight. The transaction
            # has rolled back at this point, so no partial replacement leaks.
            raise GroupRelationConflictError(
                'External group is already mapped to another group') from error

        return final_entries

    async def set_all_authenticated_users(
        self,
        group_id: GroupID,
        enabled: bool,
        *,
        principal: AuthPrincipal | None,
    ) -> None:
        from rest.utils.driver_common import get_current_timestamps

        self._require_admin(principal)
        if str(group_id) == DEFAULT_GROUP_ID and not enabled:
            raise ValueError(
                'The canonical Default group must include all authenticated '
                'users')
        try:
            async with self._state_db.write_tx() as tx:
                await lock_group_relations(tx, [group_id])
                await self._get_active_group_in_tx(tx, group_id)
                row = await self._rules.find_one_by_fields(
                    tx, {
                        'group_id': str(group_id),
                        'type': GroupMembershipRuleType.ALL_AUTHENTICATED_USERS,
                    })
                if enabled and row is None:
                    entry = GroupMembershipRuleEntry(
                        id=GroupMembershipRuleID.generate(),
                        group_id=group_id,
                        type=GroupMembershipRuleType.ALL_AUTHENTICATED_USERS,
                        timestamps=get_current_timestamps(),
                    )
                    await self._rules.upsert(tx, str(entry.id), entry)
                elif not enabled and row is not None:
                    await self._rules.delete(tx, row.id)
        except (sqlite3.IntegrityError, asyncpg.UniqueViolationError):
            if not enabled:
                raise
            # A concurrent enable created the same unique rule. The requested
            # end state already holds, so enabling remains idempotent.
            return

    async def has_all_authenticated_users(
        self,
        group_id: GroupID,
        principal: AuthPrincipal | None,
    ) -> bool:
        self._require_admin(principal)
        await self._get_active_group(str(group_id))
        async with self._state_db.read_tx() as tx:
            return await self._rules.exists_by_fields(
                tx, {
                    'group_id': str(group_id),
                    'type': GroupMembershipRuleType.ALL_AUTHENTICATED_USERS,
                })

    async def resolve_native_memberships(
        self,
        principal_id: str,
        *,
        principal_type: PrincipalType,
        authenticated: bool,
    ) -> list[str]:
        async with self._state_db.read_tx() as tx:
            members = await self._members.list_by_fields(
                tx, {
                    'principal_id': principal_id,
                    'principal_type': principal_type,
                })
            group_ids = {str(row.payload.group_id) for row in members}
            if authenticated and principal_type == PrincipalType.USER:
                rules = await self._rules.list_by_field(
                    tx, 'type', GroupMembershipRuleType.ALL_AUTHENTICATED_USERS)
                group_ids.update(str(row.payload.group_id) for row in rules)
            groups = await self._groups.list_by_ids(
                tx, group_ids, fields={'status': GroupStatus.ACTIVE})
        active_ids = {row.id for row in groups}
        return sorted(group_id for group_id in group_ids
                      if group_id in active_ids)

    async def list_user_groups(
        self,
        user_authorization: AuthPrincipal | None,
    ) -> list[KumoGroupDBEntry]:
        user_auth = self._require_principal(user_authorization)
        async with self._state_db.read_tx() as tx:
            if user_auth.is_admin:
                rows = await self._groups.list_by_field(tx, 'status',
                                                        GroupStatus.ACTIVE)
                return sorted(
                    (row.payload for row in rows),
                    key=lambda group: group.normalized_display_name,
                )
            rows = await self._groups.list_by_ids(
                tx, [str(group_id) for group_id in user_auth.kumo_group_ids],
                fields={'status': GroupStatus.ACTIVE})
        by_id = {row.id: row.payload for row in rows}
        return [
            by_id[str(group_id)] for group_id in user_auth.kumo_group_ids
            if str(group_id) in by_id
        ]

    def _require_group_membership(
        self,
        group_id: GroupID,
        user_authorization: AuthPrincipal | None,
    ) -> None:
        from statedb.utils.errors import ProjectForbiddenError

        user_auth = self._require_principal(user_authorization)
        if (not user_auth.is_admin and str(group_id)
                not in {str(value)
                        for value in user_auth.kumo_group_ids}):
            raise ProjectForbiddenError('Not a member of this group')

    @staticmethod
    def _require_principal(
        user_authorization: AuthPrincipal | None, ) -> AuthPrincipal:
        from statedb.utils.errors import ProjectForbiddenError

        if user_authorization is None:
            raise ProjectForbiddenError('Authenticated principal is required')
        return user_authorization

    @classmethod
    def _require_admin(
        cls,
        user_authorization: AuthPrincipal | None,
    ) -> AuthPrincipal:
        from statedb.utils.errors import ProjectForbiddenError

        principal = cls._require_principal(user_authorization)
        if not principal.is_admin:
            raise ProjectForbiddenError('Administrator access required')
        return principal

    async def map_idp_group_ids(
        self,
        idp_group_ids: list[str],
        identity_provider_id: str | ExternalIdentityProviderID = (
            ExternalIdentityProviderID.DIRECT_CONNECT),
    ) -> list[str]:
        provider_id = self._clean_provider_id(identity_provider_id)
        cleaned_group_ids = [
            self._clean_external_group_id(group_id)
            for group_id in idp_group_ids if str(group_id).strip()
        ]
        return await self.map_idp_group_hashes([
            hash_idp_group_id(provider_id, cleaned)
            for cleaned in cleaned_group_ids
        ], identity_provider_id=provider_id)

    async def map_idp_group_hashes(
        self,
        idp_group_hashes: list[str],
        identity_provider_id: str | ExternalIdentityProviderID = (
            ExternalIdentityProviderID.DIRECT_CONNECT),
    ) -> list[str]:
        provider_id = self._clean_provider_id(identity_provider_id)
        async with self._state_db.read_tx() as tx:
            rows = await self._external_mappings.list_all(tx)
            groups = {
                row.id
                for row in await self._groups.list_by_fields(
                    tx, {'status': GroupStatus.ACTIVE})
            }
        wanted = set(idp_group_hashes)
        return sorted({
            str(row.payload.group_id)
            for row in rows if row.payload.external_group_id_hash in wanted
            and row.payload.identity_provider_id == provider_id
            and str(row.payload.group_id) in groups
        })
