// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useMemo, useState } from "react";
import Spacer from "react-spacer";

import {
    Button,
    ColumnDef,
    DataTable,
    Dialog,
    DialogContent,
    DialogDescription,
    DialogTitle,
    Row
} from "@kumo-ai/kumo-ui-catalog";

export interface MemberRow {
    principalId: string;
    name: string;
    email: string;
    role: string;
    disabled: boolean;
}

interface GroupMembersTableProps {
    rows: MemberRow[];
    detailsLoaded: boolean;
    // Stages the removal; it is persisted by the page-level Save.
    onRemove: (principalId: string) => void;
}

// Confirmation dialog matching the existing delete pattern (DeleteKeyDialog).
interface RemoveMemberDialogProps {
    member: MemberRow | null;
    onOpenChange: (open: boolean) => void;
    onConfirm: () => void;
}

const RemoveMemberDialog = ({
    member,
    onOpenChange,
    onConfirm
}: RemoveMemberDialogProps) => (
    <Dialog open={member !== null} onOpenChange={onOpenChange}>
        <DialogContent data-testid={"alert-dialog-content"} className="top-1/2">
            <div className="flex flex-col justify-between px-7.5 py-6">
                <DialogTitle className="text-title-primary leading-7">
                    Remove User from Group
                </DialogTitle>
                <Spacer height={16} />
                <DialogDescription className="flex-1 p-0 m-0 text-title-primary text-base font-medium leading-5 outline-none">
                    {`Are you sure you want to remove ${
                        member?.name || member?.email || "this user"
                    } from this group? They will keep their workspace account
                    and any other group memberships.`}
                </DialogDescription>
                <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end mt-6">
                    <Button
                        id="cancel-remove-member"
                        variant="outline"
                        onClick={() => onOpenChange(false)}>
                        Cancel
                    </Button>
                    <Button
                        id="confirm-remove-member"
                        variant="destructive"
                        onClick={onConfirm}>
                        Remove
                    </Button>
                </div>
            </div>
        </DialogContent>
    </Dialog>
);

const leftAlignedHeader = (label: string) => (
    <span className="block text-left">{label}</span>
);

// Role, status and password stay editable only on Workspace Users — account
// properties apply across every group, so this table shows them read-only and
// offers the one group-scoped action: removing the membership.
const GroupMembersTable: React.FC<GroupMembersTableProps> = ({
    rows,
    detailsLoaded,
    onRemove
}) => {
    const [memberToRemove, setMemberToRemove] = useState<MemberRow | null>(
        null
    );

    const columns = useMemo<ColumnDef<MemberRow>[]>(
        () => [
            {
                header: () => leftAlignedHeader("Name"),
                accessorKey: "name",
                size: 240,
                cell: ({ row }: { row: Row<MemberRow> }) => (
                    <span className="block text-left">{row.original.name}</span>
                )
            },
            {
                header: () => leftAlignedHeader("Email"),
                accessorKey: "email",
                size: 320,
                cell: ({ row }: { row: Row<MemberRow> }) => (
                    <span className="block text-left">
                        {row.original.email}
                    </span>
                )
            },
            {
                header: () => leftAlignedHeader("Role"),
                accessorKey: "role",
                size: 120,
                cell: ({ row }: { row: Row<MemberRow> }) => (
                    <span className="block text-left capitalize">
                        {row.original.role}
                    </span>
                )
            },
            {
                header: () => leftAlignedHeader(""),
                accessorKey: "principalId",
                size: 120,
                enableSorting: false,
                cell: ({ row }: { row: Row<MemberRow> }) => (
                    <div className="flex justify-start">
                        <Button
                            size="sm"
                            variant="outline"
                            disabled={!detailsLoaded}
                            onClick={() => setMemberToRemove(row.original)}>
                            Remove
                        </Button>
                    </div>
                )
            }
        ],

        [detailsLoaded]
    );

    const removeDialog = (
        <RemoveMemberDialog
            member={memberToRemove}
            onOpenChange={(open) => {
                if (!open) setMemberToRemove(null);
            }}
            onConfirm={() => {
                if (memberToRemove) {
                    onRemove(memberToRemove.principalId);
                    setMemberToRemove(null);
                }
            }}
        />
    );

    if (rows.length === 0) {
        // Reuse the real DataTable so the empty state matches the workspace
        // table exactly; overlay replaces the default "No data found." text.
        return (
            <div className="relative">
                <DataTable
                    id="group-members-table"
                    columns={columns}
                    data={[]}
                    enableGlobalFilter={false}
                />
                <div className="text-muted-foreground pointer-events-none absolute inset-x-px bottom-px top-10 flex items-center justify-center bg-white text-base font-medium leading-5">
                    Add user to the group
                </div>
                {removeDialog}
            </div>
        );
    }

    return (
        <>
            <DataTable
                id="group-members-table"
                columns={columns}
                data={rows}
                enableGlobalFilter={false}
            />
            {removeDialog}
        </>
    );
};

export default GroupMembersTable;
