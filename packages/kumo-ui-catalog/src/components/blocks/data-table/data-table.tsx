"use client";
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0


import {
    ColumnDef,
    ExpandedState,
    flexRender,
    getCoreRowModel,
    getFilteredRowModel,
    getPaginationRowModel,
    getSortedRowModel,
    PaginationState,
    RowSelectionState,
    SortingState,
    useReactTable,
    VisibilityState
} from "@tanstack/react-table";

import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow
} from "@/components/ui/table";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { DataTablePagination } from "./data-table-pagination";
import { SearchBar } from "@/components/ui/search";
import { StickyColumnOptions } from "@/types/data-table";
import { cn } from "@/lib/utils";
import { SortUpDown } from "@/icons/MaterialIcons/SortUpDown";
import { Skeleton } from "@/components/ui/skeleton";

interface DataTableProps<TData, TValue> {
    id?: string;
    columns: ColumnDef<TData, TValue>[];
    data: TData[];
    pageCount?: number;
    defaultPageSize?: number;
    isLoading?: boolean;
    enableMultiSort?: boolean;
    enableGlobalFilter?: boolean;
    globalFilterPlaceholder?: string;
    columnsVisibility?: VisibilityState;
    onRowClick?: (rowData: TData) => void;
    onRowDoubleClick?: (rowData: TData) => void;
    rowCanExpand?: boolean;
    manualPagination?: boolean;
    manualSorting?: boolean;
    manualFiltering?: boolean;
    manualPageIndex?: number;
    handleManualPageChange?: (page: number) => void;
    manualSortingState?: SortingState;
    setManualSortingState?: React.Dispatch<React.SetStateAction<SortingState>>;
    manualRowSelectionState?: RowSelectionState;
    setManualRowSelectionState?: React.Dispatch<
        React.SetStateAction<RowSelectionState>
    >;
    showTopRoundedBorder?: boolean;
    showBodyCellRightBorder?: boolean;
    numSkeletonRow?: number;
    disableRowSelectionHover?: boolean;
    hideBodyWhenEmpty?: boolean;
    /** When set, disables pagination and makes the table body vertically scrollable at given height (px).*/
    scrollHeight?: number;
}

export function DataTable<TData, TValue>({
    id = "table-id",
    columns,
    data,
    pageCount = 1,
    defaultPageSize = 10,
    isLoading = false,
    enableMultiSort = false,
    enableGlobalFilter = true,
    globalFilterPlaceholder = "Search",
    columnsVisibility = {},
    onRowClick,
    onRowDoubleClick,
    rowCanExpand = false,
    manualPagination = false,
    manualSorting = false,
    manualFiltering = false,
    manualPageIndex,
    handleManualPageChange,
    manualSortingState,
    setManualSortingState,
    manualRowSelectionState,
    setManualRowSelectionState,
    showTopRoundedBorder = true,
    showBodyCellRightBorder = false,
    numSkeletonRow = 5,
    disableRowSelectionHover = false,
    hideBodyWhenEmpty = false,
    scrollHeight
}: DataTableProps<TData, TValue>) {
    const [sortingState, setSortingState] = useState<SortingState>([]);
    const [globalFilter, setGlobalFilter] = useState<string>("");
    const [rowSelectionState, setRowSelectionState] =
        useState<RowSelectionState>({});
    const [expandedState, setExpandedState] = useState<ExpandedState>({});
    const [columnVisibilityState, setColumnVisibilityState] =
        useState<VisibilityState>({});
    const [paginationState, setPaginationState] = useState<PaginationState>({
        pageIndex: 0,
        pageSize: scrollHeight ? Number.MAX_SAFE_INTEGER : defaultPageSize
    });

    const [scrollLeft, setScrollLeft] = useState(0);
    const tableContainerRef = useRef<HTMLDivElement>(null);

    const table = useReactTable({
        data,
        columns,
        pageCount,
        state: {
            pagination: paginationState,
            sorting: manualSortingState ?? sortingState,
            rowSelection: manualRowSelectionState ?? rowSelectionState,
            expanded: expandedState,
            columnVisibility: columnsVisibility ?? columnVisibilityState,
            globalFilter
        },
        enableRowSelection: true,
        enableMultiRowSelection: true,
        enableGlobalFilter,
        enableMultiSort,
        manualPagination: manualPagination,
        manualSorting: manualSorting,
        manualFiltering: manualFiltering,
        onExpandedChange: setExpandedState,
        onGlobalFilterChange: setGlobalFilter,
        onPaginationChange: setPaginationState,
        onRowSelectionChange:
            setManualRowSelectionState ?? setRowSelectionState,
        onSortingChange: setManualSortingState ?? setSortingState,
        onColumnVisibilityChange: setColumnVisibilityState,
        getRowCanExpand: () => rowCanExpand,
        getCoreRowModel: getCoreRowModel(),
        getPaginationRowModel: getPaginationRowModel(),
        getSortedRowModel: getSortedRowModel(),
        getFilteredRowModel: getFilteredRowModel()
    });

    const stickyOffsets = useMemo(() => {
        const leftOffsets: Record<string, number> = {};
        const rightOffsets: Record<string, number> = {};

        let currentLeftOffset = 0;
        const rightStickyColumns: { id: string; width: number }[] = [];

        columns.forEach((column) => {
            const id = column.id as string;
            const width = column.size || 0;
            const meta = column.meta as
                | { sticky?: StickyColumnOptions }
                | undefined;

            if (meta?.sticky?.left !== undefined) {
                leftOffsets[id as string] = currentLeftOffset;
                currentLeftOffset += width;
            }

            if (meta?.sticky?.right !== undefined) {
                rightStickyColumns.push({ id, width });
            }
        });

        let currentRightOffset = 0;
        rightStickyColumns.reverse().forEach(({ id, width }) => {
            rightOffsets[id] = currentRightOffset;
            currentRightOffset += width;
        });

        return { leftOffsets, rightOffsets };
    }, [columns]);

    useEffect(() => {
        const handleScroll = () => {
            if (tableContainerRef?.current) {
                setScrollLeft(tableContainerRef.current?.scrollLeft);
            }
        };

        const container = tableContainerRef?.current;
        if (container) {
            container.addEventListener("scroll", handleScroll);

            return () => container.removeEventListener("scroll", handleScroll);
        }
    }, []);

    const isEmptyHidden =
        hideBodyWhenEmpty && !isLoading && !table.getRowModel().rows?.length;

    // We use table state for client side pagination.
    // For server side, we need to override as done below.
    let { pageIndex } = table.getState().pagination;
    let canPreviousPage = table.getCanPreviousPage();
    let previousPage = () => table.previousPage();
    let canNextPage = table.getCanNextPage();
    let nextPage = () => table.nextPage();
    let showPagination =
        !scrollHeight &&
        Math.ceil(table.getFilteredRowModel().rows.length / defaultPageSize) > 1
            ? true
            : false;

    if (manualPagination === true) {
        pageIndex = manualPageIndex ?? 0;
        canPreviousPage = pageIndex > 0 ? true : false;
        previousPage = () => handlePageChange(pageIndex);
        canNextPage = pageIndex + 1 < pageCount ? true : false;
        nextPage = () => handlePageChange(pageIndex + 2);
        showPagination = pageCount > 1 ? true : false;
    }

    const handlePageChange = (page: number) => {
        if (manualPagination === true && handleManualPageChange) {
            handleManualPageChange(page - 1);
        } else {
            table.setPageIndex(page - 1);
        }
    };

    return (
        <div>
            <div
                className={cn(
                    "border-border-secondary overflow-hidden border",
                    showTopRoundedBorder ? "rounded-t-sm" : "rounded-t-0",
                    isEmptyHidden
                        ? "border-b-0! rounded-b-none"
                        : showPagination
                          ? "rounded-b-0"
                          : "rounded-b-sm"
                )}>
                {enableGlobalFilter && (
                    <SearchBar
                        placeholder={globalFilterPlaceholder}
                        position="top"
                        className="border-none bg-white"
                        variant="secondary"
                        onChange={(event) =>
                            setGlobalFilter(event.target.value)
                        }></SearchBar>
                )}
                <div>
                    <Table
                        dref={tableContainerRef}
                        id={id}
                        maxHeight={scrollHeight}>
                        <TableHeader className="sticky top-0 z-10">
                            {table.getHeaderGroups().map((headerGroup) => (
                                <TableRow key={headerGroup.id}>
                                    {headerGroup.headers.map((header) => {
                                        const columnId = header.column.id;
                                        const meta = header.column.columnDef
                                            .meta as
                                            | { sticky?: StickyColumnOptions }
                                            | undefined;
                                        const width =
                                            header.column.columnDef.size || 0;

                                        const stickyConfig = meta?.sticky
                                            ? { ...meta.sticky }
                                            : null;

                                        if (stickyConfig?.left !== undefined) {
                                            stickyConfig.left =
                                                stickyOffsets.leftOffsets[
                                                    columnId
                                                ];
                                        }

                                        if (stickyConfig?.right !== undefined) {
                                            stickyConfig.right =
                                                stickyOffsets.rightOffsets[
                                                    columnId
                                                ];
                                        }

                                        return (
                                            <TableHead
                                                key={header.id}
                                                sticky={stickyConfig}
                                                scrollLeft={scrollLeft}
                                                width={width}
                                                className={
                                                    !enableGlobalFilter &&
                                                    showTopRoundedBorder
                                                        ? "first:rounded-tl-sm last:rounded-tr-sm"
                                                        : "first:rounded-tl-0 last:rounded-tr-0"
                                                }>
                                                {header.isPlaceholder ? null : (
                                                    <div
                                                        className={
                                                            header.column.getCanSort()
                                                                ? "flex cursor-pointer select-none items-center"
                                                                : ""
                                                        }
                                                        onClick={header.column.getToggleSortingHandler()}>
                                                        {flexRender(
                                                            header.column
                                                                .columnDef
                                                                .header,
                                                            header.getContext()
                                                        )}
                                                        {header.column.getCanSort() && (
                                                            <span className="-mt-0.5 ml-0">
                                                                {{
                                                                    asc: (
                                                                        <SortUpDown
                                                                            className="h-5 w-5"
                                                                            upClassName="fill-icon-primary"
                                                                            downClassName="fill-title-secondary"
                                                                        />
                                                                    ),
                                                                    desc: (
                                                                        <SortUpDown
                                                                            className="!top-2 h-5 w-5"
                                                                            upClassName="fill-title-secondary"
                                                                            downClassName="fill-icon-primary"
                                                                        />
                                                                    )
                                                                }[
                                                                    header.column.getIsSorted() as string
                                                                ] ?? (
                                                                    <SortUpDown className="fill-title-secondary h-5 w-5" />
                                                                )}
                                                            </span>
                                                        )}
                                                    </div>
                                                )}
                                            </TableHead>
                                        );
                                    })}
                                </TableRow>
                            ))}
                        </TableHeader>
                        {!isEmptyHidden && (
                            <TableBody className="**:data-[slot=table-cell]:first:w-8">
                                {isLoading ? (
                                    Array(numSkeletonRow ? numSkeletonRow : 5)
                                        .fill(null)
                                        .map((_, rowIndex) => (
                                            <TableRow
                                                key={`skeleton-row-${rowIndex}`}
                                                className="last:!border-b-0">
                                                {table
                                                    .getVisibleLeafColumns()
                                                    .map(
                                                        (
                                                            _column,
                                                            cellIndex
                                                        ) => {
                                                            const width =
                                                                _column
                                                                    .columnDef
                                                                    .size || 0;
                                                            return (
                                                                <TableCell
                                                                    width={
                                                                        width
                                                                    }
                                                                    key={`skeleton-cell-${rowIndex}-${cellIndex}`}>
                                                                    <div className="flex items-center justify-start">
                                                                        <Skeleton
                                                                            className={
                                                                                "h-4 rounded-sm " +
                                                                                (_column
                                                                                    .columnDef
                                                                                    .id !==
                                                                                "expand"
                                                                                    ? "w-full"
                                                                                    : "w-40")
                                                                            }
                                                                        />
                                                                    </div>
                                                                </TableCell>
                                                            );
                                                        }
                                                    )}
                                            </TableRow>
                                        ))
                                ) : table.getRowModel().rows?.length ? (
                                    table.getRowModel().rows.map((row) => (
                                        <React.Fragment key={row.id}>
                                            <TableRow
                                                key={row.id}
                                                data-state={
                                                    row.getIsSelected() &&
                                                    "selected"
                                                }
                                                disableRowSelectionHover={
                                                    disableRowSelectionHover
                                                }
                                                className={cn(
                                                    onRowClick ||
                                                        onRowDoubleClick ||
                                                        rowCanExpand
                                                        ? "cursor-pointer bg-white"
                                                        : "bg-white",
                                                    showBodyCellRightBorder
                                                        ? "[&>td:not(:first-child):not(:last-child)]:!border-r [&>td:not(:first-child):not(:last-child)]:!border-slate-200"
                                                        : "",
                                                    "last:!border-b-0",
                                                    (
                                                        row.original as {
                                                            rowClassName?: string;
                                                        }
                                                    ).rowClassName ?? ""
                                                )}
                                                onClick={() => {
                                                    if (onRowClick) {
                                                        onRowClick(
                                                            row.original
                                                        );
                                                    }
                                                    if (rowCanExpand) {
                                                        row.toggleExpanded();
                                                    }
                                                }}
                                                onDoubleClick={() => {
                                                    if (onRowDoubleClick) {
                                                        onRowDoubleClick(
                                                            row.original
                                                        );
                                                    }
                                                }}>
                                                {row
                                                    .getVisibleCells()
                                                    .map((cell) => {
                                                        const columnId =
                                                            cell.column.id;
                                                        const meta = cell.column
                                                            .columnDef.meta as
                                                            | {
                                                                  sticky?: StickyColumnOptions;
                                                                  className?: string;
                                                              }
                                                            | undefined;
                                                        const width =
                                                            cell.column
                                                                .columnDef
                                                                .size || 0;

                                                        const stickyConfig =
                                                            meta?.sticky
                                                                ? {
                                                                      ...meta.sticky
                                                                  }
                                                                : null;

                                                        if (
                                                            stickyConfig?.left !==
                                                            undefined
                                                        ) {
                                                            stickyConfig.left =
                                                                stickyOffsets.leftOffsets[
                                                                    columnId
                                                                ];
                                                        }

                                                        if (
                                                            stickyConfig?.right !==
                                                            undefined
                                                        ) {
                                                            stickyConfig.right =
                                                                stickyOffsets.rightOffsets[
                                                                    columnId
                                                                ];
                                                        }

                                                        const cellBgColor =
                                                            (
                                                                row.original as {
                                                                    cellBgColor: string;
                                                                }
                                                            )?.cellBgColor ??
                                                            "";

                                                        const cellClassName =
                                                            meta?.className ??
                                                            "";

                                                        return (
                                                            <TableCell
                                                                key={cell.id}
                                                                className={`${cellBgColor} ${cellClassName}`}
                                                                sticky={
                                                                    stickyConfig
                                                                }
                                                                scrollLeft={
                                                                    scrollLeft
                                                                }
                                                                width={width}
                                                                disableRowSelectionHover={
                                                                    disableRowSelectionHover
                                                                }>
                                                                {flexRender(
                                                                    cell.column
                                                                        .columnDef
                                                                        .cell,
                                                                    cell.getContext()
                                                                )}
                                                            </TableCell>
                                                        );
                                                    })}
                                            </TableRow>
                                            {row.getIsExpanded() && (
                                                <TableRow>
                                                    <TableCell
                                                        colSpan={
                                                            row.getVisibleCells()
                                                                .length
                                                        }
                                                        className="p-0">
                                                        <div className="border-t p-4">
                                                            {
                                                                (
                                                                    row?.original as {
                                                                        component:
                                                                            | string
                                                                            | React.ReactNode;
                                                                    }
                                                                )?.component
                                                            }
                                                        </div>
                                                    </TableCell>
                                                </TableRow>
                                            )}
                                        </React.Fragment>
                                    ))
                                ) : (
                                    <TableRow className="border-border-secondary">
                                        <TableCell
                                            colSpan={columns.length}
                                            className="text-muted-foreground h-24 px-3 py-1.5 text-center text-base">
                                            No data found.
                                        </TableCell>
                                    </TableRow>
                                )}
                            </TableBody>
                        )}
                    </Table>
                </div>
            </div>
            {showPagination && (
                <div className="flex">
                    <DataTablePagination
                        canPreviousPage={canPreviousPage}
                        previousPage={previousPage}
                        pageIndex={pageIndex}
                        canNextPage={canNextPage}
                        nextPage={nextPage}
                        pageCount={pageCount as number}
                        handlePageChange={handlePageChange}
                    />
                </div>
            )}
        </div>
    );
}
