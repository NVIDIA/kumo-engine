// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow
} from "@/components/ui/table";
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@/components/ui/tooltip";
import { Metric } from "@/types/baseline-metrics-table";

interface BaseLineMetricsTableProps {
    data: Metric[];
    columns: {
        Header: string;
        accessor: keyof Metric;
        tipText?: string;
        colSpan?: number;
    }[];
}

// Utility to check if data is grouped by type
function isGroupedData(data: Metric[]) {
    return data.some((row) => "type" in row);
}

// Group by type utility
function groupByType(data: Metric[]) {
    const groups: Record<string, Metric[]> = {};
    for (const row of data) {
        if (!("type" in row) || !row.type) continue;
        if (!groups[row.type]) groups[row.type] = [];
        groups[row.type].push(row);
    }
    return Object.entries(groups).map(([type, rows]) => ({ type, rows }));
}

export const BaseLineMetricsTable = ({
    data,
    columns
}: BaseLineMetricsTableProps) => {
    const groupedRows = isGroupedData(data) ? groupByType(data) : null;

    return (
        <Table>
            <TableHeader>
                <TableRow>
                    {Object.entries(
                        columns.reduce(
                            (acc, col) => {
                                const key = col.Header;
                                if (!acc[key]) {
                                    acc[key] = {
                                        ...col,
                                        colSpan: col.colSpan ?? 1
                                    };
                                } else {
                                    acc[key].colSpan! += col.colSpan ?? 1;
                                }
                                return acc;
                            },
                            {} as Record<string, (typeof columns)[0]>
                        )
                    ).map(([, col], idx) => (
                        <TableHead
                            className="!group-hover:bg-transparent !hover:bg-transparent !border-1 !border-slate-400 !bg-white text-sm font-semibold text-slate-800"
                            key={idx}
                            colSpan={col.colSpan}>
                            {col.tipText ? (
                                <Tooltip>
                                    <TooltipTrigger>
                                        <span>{col.Header}</span>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                        {col.tipText}
                                    </TooltipContent>
                                </Tooltip>
                            ) : (
                                col.Header
                            )}
                        </TableHead>
                    ))}
                </TableRow>
            </TableHeader>
            <TableBody>
                {groupedRows
                    ? // Render grouped rows with rowSpan
                      groupedRows.map(({ type, rows }, groupIdx) =>
                          rows.map((row, rowIdx) => (
                              <TableRow
                                  className="!group-hover:bg-transparent !hover:bg-transparent !bg-white"
                                  key={`${groupIdx}-${rowIdx}`}>
                                  {rowIdx === 0 && (
                                      <TableCell
                                          rowSpan={rows.length}
                                          className="!group-hover:bg-transparent !hover:bg-transparent !border-1 !border-slate-400 !bg-white text-sm font-semibold text-slate-800">
                                          {type}
                                      </TableCell>
                                  )}
                                  <TableCell className="!group-hover:bg-transparent !hover:bg-transparent !border-1 !border-slate-400 !bg-white text-sm font-semibold text-slate-800">
                                      {row.name}
                                  </TableCell>
                                  {/* Render columns dynamically */}
                                  {columns
                                      .filter(
                                          (col) =>
                                              col.accessor !== "type" &&
                                              col.accessor !== "name"
                                      )
                                      .map((col, idx) => {
                                          // coloring based on presence of color keys
                                          const colorKey = `color_${String(col.accessor)}`;
                                          const bgColor = String(
                                              row[colorKey] ?? undefined
                                          );

                                          return (
                                              <TableCell
                                                  key={idx}
                                                  className="!hover:bg-transparent !group-hover:bg-transparent !border-1 !border-slate-400 text-sm font-normal text-gray-800"
                                                  style={{
                                                      backgroundColor: bgColor
                                                  }}>
                                                  {row[col.accessor]}
                                              </TableCell>
                                          );
                                      })}
                              </TableRow>
                          ))
                      )
                    : // Render flat rows without grouping
                      data.map((row, idx) => (
                          <TableRow
                              key={idx}
                              className="!group-hover:bg-transparent !hover:bg-transparent !bg-white">
                              {columns.map((col, colIdx) => {
                                  const colorKey = `color_${String(col.accessor)}`;
                                  const bgColor = row[colorKey] ?? undefined;

                                  const isNameCol = col.accessor === "name";
                                  const fontWeightClass = isNameCol
                                      ? "font-semibold !bg-white"
                                      : "font-normal";

                                  return (
                                      <TableCell
                                          key={colIdx}
                                          className={`!hover:bg-transparent !group-hover:bg-transparent !border-1 !border-slate-400 text-sm ${fontWeightClass} text-gray-800`}
                                          style={{
                                              backgroundColor: String(bgColor)
                                          }}>
                                          {isNameCol && row.tipText ? (
                                              <Tooltip>
                                                  <TooltipTrigger>
                                                      <span>
                                                          {row[col.accessor]}
                                                      </span>
                                                  </TooltipTrigger>
                                                  <TooltipContent side="top">
                                                      {row.tipText}
                                                  </TooltipContent>
                                              </Tooltip>
                                          ) : (
                                              row[col.accessor]
                                          )}
                                      </TableCell>
                                  );
                              })}
                          </TableRow>
                      ))}
            </TableBody>
        </Table>
    );
};
