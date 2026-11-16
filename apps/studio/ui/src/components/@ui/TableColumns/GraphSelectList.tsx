// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ColumnDef } from "@kumo-ai/kumo-ui-catalog";

import { GraphListResponse } from "types/graphs";

import { formatToShortLocalDateString } from "utils/timeConverter";

export const GraphSelectListColumns = (): ColumnDef<GraphListResponse>[] => {
    return [
        {
            id: "name",
            header: "Name",
            accessorKey: "name",
            size: 150,

            cell: ({ row }: { row: { original: GraphListResponse } }) => (
                <span className="font-medium text-title-secondary">
                    {row.original.name}
                </span>
            )
        },
        {
            id: "user",
            header: "User",
            accessorKey: "created_by_name",
            size: 150,
            cell: ({ row }: { row: { original: GraphListResponse } }) => (
                <span className="text-title-secondary">
                    {row.original.created_by_name}
                </span>
            )
        },
        {
            id: "dateCreated",
            header: "Created",
            accessorKey: "dateCreated",
            size: 150,
            cell: ({ row }: { row: { original: GraphListResponse } }) => {
                const ts = row.original.timestamps?.created_at;
                const dateCreated = row.original.dateCreated;
                if (ts) {
                    return (
                        <span className="text-title-secondary">
                            {formatToShortLocalDateString(ts)}
                        </span>
                    );
                }
                // dateCreated is already display-formatted; render it as-is
                // instead of running it through the formatter again.
                if (dateCreated) {
                    return (
                        <span className="text-title-secondary">
                            {dateCreated}
                        </span>
                    );
                }
                return <></>;
            }
        },
        {
            id: "num_tables",
            header: "Config",
            accessorKey: "num_tables",
            size: 150,
            cell: ({ row }: { row: { original: GraphListResponse } }) => {
                const noOfTables = Number(row.original.num_tables) || 0;
                const noOfLinks = Number(row.original.num_links) || 0;
                return (
                    <span className="text-title-secondary">
                        {`${noOfTables} Table${noOfTables > 1 ? "s" : ""} x ${noOfLinks} Link${noOfLinks > 1 ? "s" : ""}`}
                    </span>
                );
            }
        }
    ];
};
