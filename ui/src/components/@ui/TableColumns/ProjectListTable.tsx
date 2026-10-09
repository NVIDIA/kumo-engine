// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Check,
    ColumnDef,
    Row,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";
import type { ProjectDBEntry } from "gen/v0/types";

export const ProjectListTableColumns = (
    activeTab: "all" | "my-projects",
    selectedProjectId?: string | null
): ColumnDef<ProjectDBEntry>[] => {
    const columns: ColumnDef<ProjectDBEntry>[] = [
        {
            id: "selected",
            header: "",
            enableSorting: false,
            accessorKey: "id",
            size: 20,
            minSize: 20,
            meta: { className: "items-center" },
            cell: ({ row }: { row: Row<ProjectDBEntry> }) => {
                const isSelected = row.original.id === selectedProjectId;
                return isSelected ? (
                    <Check
                        aria-label="Selected"
                        className="!size-5 fill-text-title-primary"
                    />
                ) : null;
            }
        },
        {
            id: "display_name",
            header: "Name",
            accessorKey: "display_name",
            cell: ({ getValue }) => {
                const value = getValue() as string;
                if (value?.length < 20) return <span>{value}</span>;
                return (
                    <Tooltip id={value}>
                        <TooltipTrigger>
                            <span>{value?.substring(0, 20) + "..."}</span>
                        </TooltipTrigger>
                        <TooltipContent
                            arrow={false}
                            sideOffset={5}
                            className="z-[9999999999]">
                            {value}
                        </TooltipContent>
                    </Tooltip>
                );
            }
        },

        ...(activeTab !== "all"
            ? [
                  {
                      id: "access",
                      header: "Access",
                      accessorKey: "is_private",
                      enableSorting: false,
                      cell: ({ row }: { row: Row<ProjectDBEntry> }) =>
                          row.original.is_private ? "Private" : "Shared"
                  } as ColumnDef<ProjectDBEntry>
              ]
            : [
                  {
                      id: "created_by",
                      header: "Owner",
                      accessorKey: "created_by_name",
                      enableSorting: false,
                      cell: ({ row }) => {
                          const value =
                              row.original.created_by_name ||
                              row.original.created_by ||
                              "";
                          if (value?.length < 20) return <span>{value}</span>;
                          return (
                              <Tooltip id={value}>
                                  <TooltipTrigger>
                                      <span>
                                          {value?.substring(0, 20) + "..."}
                                      </span>
                                  </TooltipTrigger>
                                  <TooltipContent
                                      arrow={false}
                                      sideOffset={5}
                                      className="z-[9999999999]">
                                      {value}
                                  </TooltipContent>
                              </Tooltip>
                          );
                      }
                  } as ColumnDef<ProjectDBEntry>
              ]),
        {
            id: "updated_at",
            header: "Last Updated",
            accessorKey: "updated_at"
        }
    ];
    return columns;
};
