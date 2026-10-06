// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type ReactNode, useMemo } from "react";

import {
    Button,
    ColumnDef,
    Row,
    Select,
    Switch
} from "@kumo-ai/kumo-ui-catalog";
import type { DirectConnectUserResponse } from "gen/v0/types";

const roleOptions = [
    { label: "User", value: "user" },
    { label: "Admin", value: "admin" }
];

const leftAlignedHeader = (label: string) => (
    <span className="block text-left">{label}</span>
);

const leftAlignedTextCell = (value: string) => (
    <span className="block text-left">{value}</span>
);

const leftAlignedControlCell = (children: ReactNode) => (
    <div className="flex justify-start">{children}</div>
);

interface UseDirectConnectUserColumnsParams {
    isPasswordLogin: boolean;
    onUserChange: (
        userId: string,
        change: Partial<Pick<DirectConnectUserResponse, "role" | "disabled">>
    ) => void;
    onSetPasswordClick: (user: DirectConnectUserResponse) => void;
}

export const useDirectConnectUserColumns = ({
    isPasswordLogin,
    onUserChange,
    onSetPasswordClick
}: UseDirectConnectUserColumnsParams): ColumnDef<DirectConnectUserResponse>[] =>
    useMemo(
        () => [
            {
                header: () => leftAlignedHeader("Name"),
                accessorKey: "display_name",
                size: 240,
                cell: ({ row }: { row: Row<DirectConnectUserResponse> }) =>
                    leftAlignedTextCell(
                        row.original.display_name || row.original.email
                    )
            },
            {
                header: () => leftAlignedHeader("Email"),
                accessorKey: "email",
                size: 320,
                cell: ({ row }: { row: Row<DirectConnectUserResponse> }) =>
                    leftAlignedTextCell(row.original.email)
            },
            {
                header: () => leftAlignedHeader("Role"),
                accessorKey: "role",
                size: 144,
                cell: ({ row }: { row: Row<DirectConnectUserResponse> }) =>
                    leftAlignedControlCell(
                        <Select
                            id={`direct-connect-user-role-${row.original.user_id}`}
                            className="w-28"
                            mode="single"
                            options={roleOptions}
                            popoverTriggerClassName="w-28"
                            search={false}
                            selectedValues={[row.original.role]}
                            onValuesChange={(values) => {
                                const role = values[0] as
                                    | DirectConnectUserResponse["role"]
                                    | undefined;
                                if (!role) return;
                                onUserChange(row.original.user_id, {
                                    role
                                });
                            }}
                            size="sm"
                            variant="secondary"
                        />
                    )
            },
            {
                header: () => leftAlignedHeader("Enabled"),
                accessorKey: "disabled",
                size: 120,
                cell: ({ row }: { row: Row<DirectConnectUserResponse> }) =>
                    leftAlignedControlCell(
                        <Switch
                            id={`user-enabled-${row.index}`}
                            checked={!row.original.disabled}
                            onCheckedChange={(checked) =>
                                onUserChange(row.original.user_id, {
                                    disabled: !checked
                                })
                            }
                        />
                    )
            },
            ...(isPasswordLogin
                ? [
                      {
                          header: () => leftAlignedHeader("Password"),
                          accessorKey: "user_id",
                          size: 160,
                          cell: ({
                              row
                          }: {
                              row: Row<DirectConnectUserResponse>;
                          }) =>
                              leftAlignedControlCell(
                                  <Button
                                      size="sm"
                                      onClick={() =>
                                          onSetPasswordClick(row.original)
                                      }>
                                      Set password
                                  </Button>
                              )
                      }
                  ]
                : [])
        ],
        [isPasswordLogin, onSetPasswordClick, onUserChange]
    );
