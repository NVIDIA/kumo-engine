// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    ColumnDef,
    DeleteIcon,
    Row,
    Symbol,
    TextLink,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import { JSTable } from "types/tables";

import { getProjectPath } from "utils/Constants";
import { checkDash } from "utils/Helpers/stringHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { useCustomNavigation } from "utils/navigator";
import { formatToLongLocalDateString } from "utils/timeConverter";

export const ConnectedTablesColumns = (
    onTableDeleteClick: (rowData: JSTable) => void,
    projectId: string
): ColumnDef<JSTable>[] => {
    const navigate = useCustomNavigation();
    const projectPaths = getProjectPath(projectId);
    return [
        {
            id: "name_alias",
            header: () => <span>Table Name</span>,
            accessorKey: "name_alias",
            cell: ({ row }: { row: Row<JSTable> }) => {
                const name = row?.original?.name_alias as string;
                return (
                    <TextLink
                        id={`connected-table-cols-${name}`}
                        pageLink={projectPaths.Tables + name}
                        onClick={(e) => {
                            e?.preventDefault(); // To avoid entire page load
                            navigate(projectPaths.Tables + name, undefined, e);
                        }}
                        title={name}
                    />
                );
            }
        },
        {
            id: "num_cols",
            header: "Dimensions",
            accessorKey: "num_cols",
            cell: ({ row }: { row: Row<JSTable> }) => {
                const col = row?.original?.num_cols as string;
                const num_rows = row?.original?.num_rows;
                const size = row?.original?.size as string;
                return (
                    <div>
                        {col ? <>{`${col} Col, `}</> : ""}
                        {num_rows ? <>{`${num_rows} Rows, `}</> : ""}
                        {size}
                    </div>
                );
            }
        },
        {
            header: "Fact",
            accessorKey: "timeRange",
            cell: ({ row }: { row: Row<JSTable> }) => (
                <>
                    {isValid(row?.original?.timeRange) &&
                    row?.original?.timeRange != "-" ? (
                        <>{checkDash(row?.original?.timeRange as string)}</>
                    ) : (
                        <Symbol type={"emdash"} />
                    )}
                </>
            )
        },
        {
            id: "Updated",
            header: "Updated",
            accessorKey: "updated",
            cell: ({ row }: { row: Row<JSTable> }) => {
                const updated = row?.original?.updated;
                const updatedOrigin = row?.original?.updatedOrigin;
                if (!updated) {
                    return "-";
                }
                return (
                    <Tooltip id={`updated-${row.id}`}>
                        <TooltipTrigger>
                            <span>{updated}</span>
                        </TooltipTrigger>
                        <TooltipContent
                            arrow={false}
                            side={"bottom"}
                            sideOffset={5}>
                            {formatToLongLocalDateString(
                                updatedOrigin as string
                            )}
                        </TooltipContent>
                    </Tooltip>
                );
            }
        },
        {
            id: "action",
            header: "Action",
            size: 50,
            accessorKey: "",
            cell: ({ row }: { row: Row<JSTable> }) => {
                return (
                    <span
                        className="cursor-pointer"
                        onClick={() => onTableDeleteClick(row?.original)}>
                        {<DeleteIcon className="size-5.5" />}
                    </span>
                );
            }
        }
    ];
};
