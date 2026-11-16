// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, {
    forwardRef,
    useEffect,
    useImperativeHandle,
    useState
} from "react";

import { Label, Switch } from "@kumo-ai/kumo-ui-catalog";
import {
    useDeleteGroupsAllAuthenticatedUsersByGroupId,
    useGetGroupsAllAuthenticatedUsersByGroupId,
    useUpdateGroupsAllAuthenticatedUsersByGroupId
} from "gen/v0/api/groups/groups";

import { SectionSaveHandle } from "./LdapSsoGroupsSection";

interface AllAuthenticatedUsersToggleProps {
    groupId: string;
    detailsLoaded: boolean;
    onEnabledChange?: (enabled: boolean) => void;
    onDirtyChange: (dirty: boolean) => void;
}

// Membership rule that adds every authenticated user to the group. Flipping
// the switch stages the change; the page-level Save persists it.
const AllAuthenticatedUsersToggle = forwardRef<
    SectionSaveHandle,
    AllAuthenticatedUsersToggleProps
>(function AllAuthenticatedUsersToggle(
    { groupId, detailsLoaded, onEnabledChange, onDirtyChange },
    ref
) {
    const [pendingValue, setPendingValue] = useState<boolean | null>(null);

    const allUsers = useGetGroupsAllAuthenticatedUsersByGroupId(groupId, {
        query: { enabled: Boolean(groupId) }
    });
    const enableAllUsers = useUpdateGroupsAllAuthenticatedUsersByGroupId();
    const disableAllUsers = useDeleteGroupsAllAuthenticatedUsersByGroupId();

    const serverEnabled = Boolean(allUsers.data?.data);
    const effectiveEnabled = pendingValue ?? serverEnabled;
    const dirty = pendingValue !== null && pendingValue !== serverEnabled;

    // Reset staged state when navigating to a different group.
    useEffect(() => {
        setPendingValue(null);
    }, [groupId]);

    useEffect(() => {
        onEnabledChange?.(effectiveEnabled);
    }, [effectiveEnabled, onEnabledChange]);

    useEffect(() => {
        onDirtyChange(dirty);
    }, [dirty, onDirtyChange]);

    useImperativeHandle(ref, () => ({
        save: async () => {
            if (!dirty || pendingValue === null) return;
            if (pendingValue) {
                await enableAllUsers.mutateAsync({ groupId });
            } else {
                await disableAllUsers.mutateAsync({ groupId });
            }
            setPendingValue(null);
            await allUsers.refetch();
        }
    }));

    const busy =
        !detailsLoaded ||
        allUsers.isLoading ||
        enableAllUsers.isPending ||
        disableAllUsers.isPending;

    return (
        <div className="flex items-center gap-3">
            <Switch
                id="group-all-authenticated-users"
                checked={effectiveEnabled}
                disabled={busy}
                onCheckedChange={(checked) =>
                    setPendingValue(checked === serverEnabled ? null : checked)
                }
            />
            <div>
                <Label
                    htmlFor="group-all-authenticated-users"
                    className="text-sm font-medium text-title-primary leading-5">
                    Include all authenticated users
                </Label>
                <p className="text-sm text-title-secondary leading-5">
                    When enabled, every user who can sign in is automatically a
                    member of this group — no individual assignment needed.
                </p>
            </div>
        </div>
    );
});

export default AllAuthenticatedUsersToggle;
