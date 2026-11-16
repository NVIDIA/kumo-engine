// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ColumnDef, DeleteIcon, Row } from "@kumo-ai/kumo-ui-catalog";
import type { APIKeySummary } from "gen/v0/types";

import { formatToShortLocalDateString } from "utils/timeConverter";

export const ApiKeysTableColumns = (
    onDeleteClick: () => void
): ColumnDef<APIKeySummary>[] => [
    {
        id: "name",
        header: "Name",
        accessorKey: "name",
        meta: { className: "text-title-secondary font-medium leading-6" }
    },
    {
        id: "created",
        header: "Created",
        accessorKey: "created",
        meta: { className: "text-title-secondary font-medium leading-6" }
    },
    {
        id: "value",
        header: "Value",
        accessorKey: "value",
        enableSorting: false,
        meta: { className: "text-title-secondary font-medium leading-6" }
    },
    {
        id: "delete",
        header: "",
        enableSorting: false,
        accessorKey: "delete",
        meta: { className: "m-0 p-0" },
        cell: () => (
            <div
                className="content-center h-10 cursor-default"
                onClick={(e) => e.stopPropagation()}>
                <div
                    role="button"
                    aria-label="Delete"
                    className="cursor-pointer"
                    onClick={(e) => {
                        e.stopPropagation();
                        onDeleteClick();
                    }}>
                    <DeleteIcon className="size-5 fill-icon-secondary" />
                </div>
            </div>
        )
    }
];

export const AdminApiKeysTableColumns = (
    onDeleteClick: (id: string) => void
): ColumnDef<APIKeySummary>[] => [
    {
        id: "name",
        header: "Name",
        accessorKey: "name",
        meta: { className: "text-title-secondary font-medium leading-6" }
    },
    {
        id: "owner_display_name",
        header: "User",
        accessorKey: "owner_display_name",
        meta: { className: "text-title-secondary font-medium leading-6" }
    },
    {
        id: "created_time",
        header: "Created",
        accessorKey: "created_time",
        meta: { className: "text-title-secondary font-medium leading-6" },
        cell: ({ getValue }) => {
            const value = getValue() as string;
            return (
                <div className="whitespace-nowrap">
                    {formatToShortLocalDateString(value)}
                </div>
            );
        }
    },
    {
        id: "value_preview",
        header: "Value",
        accessorKey: "value_preview",
        enableSorting: false,
        meta: { className: "text-title-secondary font-medium leading-6" }
    },
    {
        id: "delete",
        header: "",
        enableSorting: false,
        accessorKey: "delete",
        meta: { className: "m-0 p-0" },
        cell: ({ row }: { row: Row<APIKeySummary> }) => (
            <div
                className="content-center h-10 cursor-default"
                onClick={(e) => e.stopPropagation()}>
                <div
                    role="button"
                    aria-label="Delete"
                    className="cursor-pointer"
                    onClick={(e) => {
                        e.stopPropagation();
                        onDeleteClick(row.original.id);
                    }}>
                    <DeleteIcon className="size-5 fill-icon-secondary" />
                </div>
            </div>
        )
    }
];
