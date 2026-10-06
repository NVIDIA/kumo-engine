// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from "react";

import {
    DataTable,
    EmptyTable,
    SearchBar,
    SortingState,
    Stack,
    useWindowDimensions
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { GraphSelectListColumns } from "components/@ui/TableColumns/GraphSelectList";
import EmptyEntities from "components/Empty/EmptyEntities";

import { useGraphs } from "hooks/graph/graphs";

import { GraphListResponse } from "types/graphs";

import { PageNames } from "utils/Constants";
import { getDataTableFillClasses } from "utils/Helpers/dataTableFill";
import { isValid } from "utils/Helpers/typeGuards";

const ROW_HEIGHT = 44;

// Non-row chrome around the table on this page: page padding + title/
// description header + search row + spacers + table header + pagination
// (≈272px measured). Slightly over to avoid overflow; the page-full fill
// wrapper absorbs the small remainder. Update if the surrounding layout
// changes.
const TABLE_CHROME = 276;

interface GraphSelectListProps {
    setNext: (graph: GraphListResponse) => void;
    projectId?: string;
    onEmptyChange?: (isEmpty: boolean) => void;
}

const GraphSelectList: React.FC<GraphSelectListProps> = ({
    setNext,
    projectId,
    onEmptyChange
}) => {
    const [searchValue, setSearchValue] = useState("");
    const [currentPage, setCurrentPage] = useState(0);
    const [sorting, setSorting] = useState<SortingState>([]);

    const { isDataLoading: loading, graphList } = useGraphs({ projectId });
    const { height } = useWindowDimensions();
    const pageSize = Math.max(
        5,
        Math.floor((height - TABLE_CHROME) / ROW_HEIGHT)
    );

    const columns = useMemo(() => GraphSelectListColumns(), []);

    const filteredGraphs = useMemo(() => {
        const filtered = searchValue.trim()
            ? graphList.filter((g) => {
                  const lower = searchValue.toLowerCase();
                  return (
                      g.name?.toLowerCase().includes(lower) ||
                      g.id?.toLowerCase().includes(lower)
                  );
              })
            : graphList;

        if (!sorting.length) return filtered;

        const field = sorting[0].id as keyof GraphListResponse;
        return [...filtered].sort((a, b) => {
            const aVal = String(a[field] ?? "");
            const bVal = String(b[field] ?? "");
            return sorting[0].desc
                ? bVal.localeCompare(aVal)
                : aVal.localeCompare(bVal);
        });
    }, [graphList, searchValue, sorting]);

    // Slice to current page — DataTable with manualPagination expects only
    // the current page's rows, not the full dataset.
    const pagedGraphs = useMemo(
        () =>
            filteredGraphs.slice(
                currentPage * pageSize,
                (currentPage + 1) * pageSize
            ),
        [filteredGraphs, currentPage, pageSize]
    );

    useEffect(() => {
        setCurrentPage(0);
    }, [searchValue]);

    const handleGraphSelect = useCallback(
        (rowData: GraphListResponse) => {
            setNext(rowData);
        },
        [setNext]
    );

    const isEmpty =
        !loading && graphList?.length === 0 && !isValid(searchValue);

    useEffect(() => {
        onEmptyChange?.(isEmpty);
    }, [isEmpty, onEmptyChange]);

    if (isEmpty) {
        return <EmptyEntities currentPage={PageNames.Graph} />;
    }

    return (
        <>
            <div className="flex flex-row items-center gap-2">
                <SearchBar
                    className="input-group-merge input-group-reverse flex w-74 h-9 py-2 px-4 items-center"
                    placeholder="Search"
                    value={searchValue}
                    variant="secondary"
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                        setSearchValue(e?.target?.value || "");
                    }}
                />
            </div>
            <div className="h-4" />
            {!loading && filteredGraphs?.length === 0 ? (
                <EmptyTable
                    id="empty-graphs-table"
                    showLoading={loading}
                    enableCard={false}>
                    <Stack />
                    <p className="text-center text-title-secondary">
                        {"No graphs found"}
                    </p>
                </EmptyTable>
            ) : (
                // Fill the remaining viewport height with graph rows when the
                // page is full; a short/last page stays content-sized.
                <div
                    className={getDataTableFillClasses(
                        pagedGraphs.length >= pageSize
                    )}>
                    <DataTable
                        id="allGraphs"
                        columns={columns}
                        data={pagedGraphs}
                        isLoading={loading}
                        pageCount={Math.ceil(filteredGraphs.length / pageSize)}
                        defaultPageSize={pageSize}
                        enableGlobalFilter={false}
                        onRowClick={handleGraphSelect}
                        manualPagination={true}
                        manualPageIndex={currentPage}
                        handleManualPageChange={setCurrentPage}
                        manualSorting={true}
                        manualSortingState={sorting}
                        setManualSortingState={setSorting}
                        manualFiltering={true}
                        showTopRoundedBorder={true}
                    />
                </div>
            )}
        </>
    );
};

export default WithErrorBoundary(GraphSelectList);
