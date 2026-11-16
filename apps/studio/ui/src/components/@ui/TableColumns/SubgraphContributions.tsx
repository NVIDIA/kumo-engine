// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    CalendarAdd,
    type ColumnDef,
    type Row,
    Symbol,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import type { XaiColumnArgs } from "types/explorer";
import type { ColumnArgs } from "types/tables";

import { customSort } from "utils/Helpers/sortHelpers";
import { isValid } from "utils/Helpers/typeGuards";

export const SubgraphContributionColumns = (
    scoreIncluded: boolean | undefined
): ColumnDef<ColumnArgs & XaiColumnArgs, unknown>[] => [
    {
        id: "name",
        header: "Column",
        accessorKey: "name",
        cell: ({ row }: { row: Row<ColumnArgs & XaiColumnArgs> }) => {
            const { name, stype } = row.original;
            const isTimestamp = stype?.toLowerCase() === "timestamp";
            return (
                <div className="flex min-w-0 items-center gap-2">
                    {isTimestamp && (
                        <CalendarAdd className="size-3.5 shrink-0 fill-pink-600" />
                    )}
                    <span className="truncate">{name}</span>
                </div>
            );
        }
    },
    {
        id: "value",
        header: "Value",
        accessorKey: "value",
        cell: ({ getValue }: { getValue: () => unknown }) => {
            const val = getValue() as string | number;
            return String(val ?? "").length > 15 ? (
                <Tooltip id={String(val)}>
                    <TooltipTrigger>
                        <span>{String(val).slice(0, 15)}…</span>
                    </TooltipTrigger>
                    <TooltipContent arrow={false} side="bottom" sideOffset={5}>
                        {String(val)}
                    </TooltipContent>
                </Tooltip>
            ) : (
                <span>{val == null ? "" : String(val)}</span>
            );
        }
    },
    ...(scoreIncluded
        ? [
              {
                  id: "grad_score",
                  header: "Score",
                  accessorKey: "grad_score",
                  cell: ({ getValue }: { getValue: () => unknown }) => {
                      const v = getValue() as number;
                      return isValid(v) ? (
                          <>{v.toFixed(2)}</>
                      ) : (
                          <Symbol type="emdash" />
                      );
                  },
                  sortingFn: (
                      a: Row<ColumnArgs & XaiColumnArgs>,
                      b: Row<ColumnArgs & XaiColumnArgs>
                  ) => customSort(b, a, "grad_score")
              } as ColumnDef<ColumnArgs & XaiColumnArgs, unknown>
          ]
        : [])
];
