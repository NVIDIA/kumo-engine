// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { memo } from "react";

import { ColumnDef, DataTable, SortingState } from "@kumo-ai/kumo-ui-catalog";
import { GenericJob } from "gen/v0/types";

import { getDataTableFillClasses } from "utils/Helpers/dataTableFill";

export interface JobsListDataTableProps {
    data: GenericJob[];
    columns: ColumnDef<GenericJob, unknown>[];
    loading: boolean;
    pageCount: number;
    defaultPageSize: number;
    currentPage: number;
    numSkeletonRow: number;
    columnsVisibility: Record<string, boolean>;
    sorting: SortingState;
    onRowClick: (rowData: object) => void;
    onPageChange: (newPage: number) => void;
    onSortingChange: React.Dispatch<React.SetStateAction<SortingState>>;
    hideBodyWhenEmpty?: boolean;
}

const JobsListDataTable = ({
    data,
    columns,
    loading,
    pageCount,
    defaultPageSize,
    currentPage,
    numSkeletonRow,
    columnsVisibility,
    sorting,
    onRowClick,
    onPageChange,
    onSortingChange,
    hideBodyWhenEmpty
}: JobsListDataTableProps) => {
    // When the page is full (enough jobs to fill it), stretch the table to fill
    // the remaining viewport height so its rows absorb the sub-row remainder
    // left by the page-size math and the table reaches the bottom. A short/last
    // page is left content-sized with plain space below. See the helper for
    // details.
    const fillHeightClasses = getDataTableFillClasses(
        data.length >= defaultPageSize
    );

    return (
        <div className={fillHeightClasses}>
            <DataTable
                id="allJobs"
                columns={columns}
                data={data}
                isLoading={loading}
                pageCount={pageCount}
                defaultPageSize={defaultPageSize}
                enableGlobalFilter={false}
                onRowClick={onRowClick}
                manualPagination={true}
                manualPageIndex={currentPage}
                handleManualPageChange={onPageChange}
                manualSorting={true}
                manualSortingState={sorting}
                setManualSortingState={onSortingChange}
                numSkeletonRow={numSkeletonRow}
                columnsVisibility={columnsVisibility}
                hideBodyWhenEmpty={hideBodyWhenEmpty}
            />
        </div>
    );
};

export default memo(JobsListDataTable);
