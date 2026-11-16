// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, {
    forwardRef,
    useEffect,
    useImperativeHandle,
    useMemo,
    useRef,
    useState
} from "react";

import {
    Alert,
    Button,
    SearchBar,
    Select,
    Title
} from "@kumo-ai/kumo-ui-catalog";
import {
    useCreateGroupsMembersByGroupId,
    useDeleteGroupsMembersByPrincipalId
} from "gen/v0/api/groups/groups";
import type { DirectConnectUserResponse } from "gen/v0/types";
import { AdminGroupMemberResponse, PrincipalType } from "gen/v0/types";

import { useDirectConnectUsers } from "hooks/auth/directConnectAdmin";

import { HandleErrorFunc } from "types/validation";

import { isAuthDisabled } from "utils/authRuntimeConfig";
import { getDirectConnectLoginMethod } from "utils/directConnectAuth";

import AllAuthenticatedUsersToggle from "./AllAuthenticatedUsersToggle";
import GroupMembersTable, { MemberRow } from "./GroupMembersTable";
import { SectionSaveHandle } from "./LdapSsoGroupsSection";

interface GroupUsersSectionProps {
    groupId: string;
    members: AdminGroupMemberResponse[];
    detailsLoaded: boolean;
    handleError: HandleErrorFunc;
    onDirtyChange: (dirty: boolean) => void;
}

// Membership edits are staged locally and persisted by the page-level Save,
// using the additive add/remove endpoints (never a list replace).
const GroupUsersSection = forwardRef<SectionSaveHandle, GroupUsersSectionProps>(
    function GroupUsersSection(
        { groupId, members, detailsLoaded, handleError, onDirtyChange },
        ref
    ) {
        const [search, setSearch] = useState("");
        const [selectedUserId, setSelectedUserId] = useState<string | null>(
            null
        );
        const [pendingAdds, setPendingAdds] = useState<string[]>([]);
        const [pendingRemoves, setPendingRemoves] = useState<string[]>([]);
        const [allAuthEnabled, setAllAuthEnabled] = useState(false);
        const [allAuthDirty, setAllAuthDirty] = useState(false);
        const allAuthRef = useRef<SectionSaveHandle>(null);

        const authDisabled = isAuthDisabled();
        const isOidcLogin =
            !authDisabled && getDirectConnectLoginMethod() === "oidc";

        const { users } = useDirectConnectUsers(true, handleError);
        const addMember = useCreateGroupsMembersByGroupId();
        const removeMember = useDeleteGroupsMembersByPrincipalId();

        // Reset staged edits when navigating to a different group.
        useEffect(() => {
            setPendingAdds([]);
            setPendingRemoves([]);
            setSelectedUserId(null);
            setSearch("");
        }, [groupId]);

        const dirty =
            pendingAdds.length > 0 || pendingRemoves.length > 0 || allAuthDirty;
        useEffect(() => {
            onDirtyChange(dirty);
        }, [dirty, onDirtyChange]);

        const serverMemberIds = useMemo(
            () =>
                new Set(
                    members
                        .filter(
                            (member) =>
                                member.principal_type === PrincipalType.USER
                        )
                        .map((member) => member.principal_id)
                ),
            [members]
        );

        // Effective membership = server members − staged removes + staged adds.
        const memberRows = useMemo<MemberRow[]>(() => {
            const byId = new Map<string, DirectConnectUserResponse>(
                users.map((user) => [user.user_id, user])
            );
            const removed = new Set(pendingRemoves);
            const ids = [
                ...[...serverMemberIds].filter((id) => !removed.has(id)),
                ...pendingAdds
            ];
            return ids.map((id) => {
                const user = byId.get(id);
                return {
                    principalId: id,
                    name: user?.display_name || id,
                    email: user?.email ?? "",
                    role: user?.role ?? "",
                    disabled: user?.disabled ?? false
                };
            });
        }, [serverMemberIds, pendingAdds, pendingRemoves, users]);

        const filteredRows = useMemo(() => {
            const normalized = search.trim().toLowerCase();
            if (!normalized) return memberRows;
            return memberRows.filter((row) =>
                [row.name, row.email, row.role].some((value) =>
                    value.toLowerCase().includes(normalized)
                )
            );
        }, [memberRows, search]);

        // Active workspace accounts not already (effectively) members.
        const availableOptions = useMemo(() => {
            const effective = new Set(memberRows.map((row) => row.principalId));
            return users
                .filter(
                    (user) => !user.disabled && !effective.has(user.user_id)
                )
                .map((user) => ({
                    label: user.display_name
                        ? `${user.display_name} — ${user.email}`
                        : user.email,
                    value: user.user_id
                }));
        }, [users, memberRows]);

        const onStageAdd = () => {
            if (!selectedUserId) return;
            // Re-adding someone staged for removal just cancels the removal.
            if (pendingRemoves.includes(selectedUserId)) {
                setPendingRemoves((current) =>
                    current.filter((id) => id !== selectedUserId)
                );
            } else if (!serverMemberIds.has(selectedUserId)) {
                setPendingAdds((current) => [...current, selectedUserId]);
            }
            setSelectedUserId(null);
        };

        const onStageRemove = (principalId: string) => {
            // Removing a staged add just un-stages it.
            if (pendingAdds.includes(principalId)) {
                setPendingAdds((current) =>
                    current.filter((id) => id !== principalId)
                );
            } else if (serverMemberIds.has(principalId)) {
                setPendingRemoves((current) => [...current, principalId]);
            }
        };

        useImperativeHandle(ref, () => ({
            save: async () => {
                await allAuthRef.current?.save();
                for (const principalId of pendingAdds) {
                    await addMember.mutateAsync({
                        groupId,
                        data: {
                            principal_id: principalId,
                            principal_type: PrincipalType.USER
                        }
                    });
                }
                for (const principalId of pendingRemoves) {
                    await removeMember.mutateAsync({ groupId, principalId });
                }
                setPendingAdds([]);
                setPendingRemoves([]);
            }
        }));

        return (
            <section className="flex flex-col gap-6">
                <Title
                    variant="section-header"
                    className="font-bold text-title-primary">
                    Users
                </Title>
                {!isOidcLogin && (
                    <AllAuthenticatedUsersToggle
                        ref={allAuthRef}
                        groupId={groupId}
                        detailsLoaded={detailsLoaded}
                        onEnabledChange={setAllAuthEnabled}
                        onDirtyChange={setAllAuthDirty}
                    />
                )}
                {allAuthEnabled && !isOidcLogin && (
                    <Alert variant="info" enableClose={false}>
                        <div className="flex flex-col gap-2">
                            <p className="text-sm font-semibold leading-5">
                                All authenticated users are currently members of
                                this group
                            </p>
                            <p className="text-sm font-medium leading-5">
                                The list below only takes effect if you turn
                                this rule off.
                            </p>
                        </div>
                    </Alert>
                )}
                {isOidcLogin ? (
                    <p className="text-sm text-title-secondary leading-5">
                        Group membership is managed by your identity provider —
                        members aren&apos;t visible in Kumo. Use the LDAP/SSO
                        mappings above to control access.
                    </p>
                ) : (
                    !authDisabled && (
                        <>
                            <div className="flex items-center gap-2">
                                <Select
                                    id="group-add-member-select"
                                    className="w-100"
                                    popoverTriggerClassName="w-100"
                                    mode="single"
                                    search
                                    searchPlaceholder="Search users"
                                    placeholder={
                                        availableOptions.length
                                            ? "Select a workspace user"
                                            : "No eligible workspace users"
                                    }
                                    options={availableOptions}
                                    selectedValues={
                                        selectedUserId ? [selectedUserId] : []
                                    }
                                    onValuesChange={(values) =>
                                        setSelectedUserId(values[0] ?? null)
                                    }
                                    disabled={!availableOptions.length}
                                    size="sm"
                                    variant="secondary"
                                />
                                <Button
                                    size="sm"
                                    variant="secondary"
                                    className="h-9 px-4 py-2 text-base"
                                    disabled={!selectedUserId || !detailsLoaded}
                                    onClick={onStageAdd}>
                                    Add
                                </Button>
                            </div>
                            {!availableOptions.length && (
                                <p className="text-sm text-title-secondary leading-5">
                                    All active workspace users are already
                                    members. Create new users from the workspace
                                    Users page first, then add them here.
                                </p>
                            )}
                            <div className="w-full sm:max-w-74">
                                <SearchBar
                                    aria-label="Search"
                                    placeholder="Search"
                                    className="input-group-merge input-group-reverse h-9 w-full px-4 [&_input]:!text-sm [&_input]:placeholder:font-medium [&_svg]:size-4"
                                    value={search}
                                    variant="secondary"
                                    onChange={(event) =>
                                        setSearch(event.target.value)
                                    }
                                />
                            </div>
                            <GroupMembersTable
                                rows={filteredRows}
                                detailsLoaded={detailsLoaded}
                                onRemove={onStageRemove}
                            />
                        </>
                    )
                )}
            </section>
        );
    }
);

export default GroupUsersSection;
