// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Badge,
    Checkbox,
    ColumnDef,
    Preview,
    Row,
    Table
} from "@kumo-ai/kumo-ui-catalog";

import { TableKind, TableListResponse } from "types/tables";

import { ENABLE_CUSTOM_SQL_TABLES } from "utils/Constants";
import { formatToShortLocalDateString } from "utils/timeConverter";

// Custom SQL columns (Kind badge + Definition-updated). Behind the feature flag
// so the picker keeps its prior columns until Custom SQL Tables ships.
const kindColumn: ColumnDef<TableListResponse> = {
    id: "table_kind",
    header: "Kind",
    accessorKey: "table_kind",
    cell: ({ row }: { row: Row<TableListResponse> }) =>
        row?.original?.table_kind === TableKind.Sql ? (
            <Badge size="sm" variant="pink">
                SQL
            </Badge>
        ) : (
            <Badge size="sm" variant="default">
                Raw
            </Badge>
        )
};

// "Definition updated" reuses the table entry's standard `updated_at` (Q3):
// editing a SQL definition updates the entry, so there is no distinct backend
// field to track.
const definitionUpdatedColumn: ColumnDef<TableListResponse> = {
    id: "definition_updated_at",
    header: "Definition updated",
    accessorFn: (row) => row?.timestamps?.updated_at,
    cell: ({ row }: { row: Row<TableListResponse> }) => {
        const updated = row?.original?.timestamps?.updated_at;
        return updated ? formatToShortLocalDateString(updated) : "-";
    }
};

export const GraphSelectTablesColumns = (
    showSlider: boolean,
    setShowSlider: React.Dispatch<React.SetStateAction<boolean>>,
    setPreviewName: React.Dispatch<React.SetStateAction<string>>
): ColumnDef<TableListResponse>[] => {
    return [
        {
            id: "select",
            header: ({ table }: { table: Table<TableListResponse> }) => (
                <Checkbox
                    checked={
                        table.getIsAllRowsSelected() ||
                        (table.getIsSomeRowsSelected() && "indeterminate")
                    }
                    onCheckedChange={(value: boolean) =>
                        table.toggleAllRowsSelected(!!value)
                    }
                    aria-label="Select all"
                    className="cursor-pointer"
                />
            ),
            cell: ({ row }: { row: Row<TableListResponse> }) => (
                <Checkbox
                    checked={row.getIsSelected()}
                    onCheckedChange={(value: boolean) =>
                        row.toggleSelected(!!value)
                    }
                    aria-label="Select row"
                    className="cursor-pointer"
                />
            ),
            size: 30
        },
        {
            id: "name",
            header: "Table Name",
            accessorKey: "name"
        },
        ...(ENABLE_CUSTOM_SQL_TABLES ? [kindColumn] : []),
        {
            id: "dateCreated",
            header: "Created date",
            accessorKey: "dateCreated"
        },
        {
            id: "connector_id",
            header: "Connector",
            accessorKey: "connector_id"
        },
        {
            id: "source_table_name",
            header: "Source Table",
            accessorKey: "source_table_name"
        },
        ...(ENABLE_CUSTOM_SQL_TABLES ? [definitionUpdatedColumn] : []),
        {
            id: "preview",
            header: "Preview",
            accessorKey: "",
            cell: ({ row }: { row: Row<TableListResponse> }) => (
                <span
                    data-testid={"preview-" + row?.index}
                    onClick={() => {
                        setShowSlider(!showSlider);
                        setPreviewName(row?.original?.name as string);
                    }}
                    className="text-title-primary cursor-pointer">
                    <Preview className="size-4.5" />
                </span>
            )
        }
    ];
};
